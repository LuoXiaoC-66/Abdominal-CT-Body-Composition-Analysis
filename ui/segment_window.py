# -*- coding: utf-8 -*-
"""界面二：自动分割。

- 选择转换好的 NRRD 文件夹与预测结果输出文件夹
- 从“模型管理”中选择要使用的 nnUNet 模型，一键批量分割所有病例
- nnUNet 环境自动处理：
  * 当前 Python 已装 nnunetv2/torch → 直接在当前进程运行；
  * 否则自动扫描本机 conda 环境，找到训练环境后以子进程运行；
  * 找不到时提示手动选择训练环境的 python.exe。
"""
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QVBoxLayout, QWidget,
)

from app import models_config
from app.constants import APP_DIR, MODELS_DIR
from app.models_config import labels_to_dict, validate_model
from app.seg_env import (
    check_python, current_python_has_nnunet, find_python_with_nnunet, setup_nnunet_env,
)
from app.segmentation import predict_folder
from app.style import make_button
from ui.model_manager_dialog import ModelManagerDialog
from ui.progress_dialog import ProgressDialog

_INFER_SCRIPT = APP_DIR / "app" / "nnunet_infer_script.py"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class EnvDetectWorker(QThread):
    """后台检测可用的 nnUNet 环境。"""
    result = Signal(object)     # (python路径, 'GPU'/'CPU') 或 None

    def run(self):
        try:
            res = find_python_with_nnunet()
        except Exception:  # noqa: BLE001
            res = None
        self.result.emit(res)


class EnvSetupWorker(QThread):
    """后台自动配置 nnUNet 环境（建 venv + 装 torch + 装 nnunetv2）。"""
    log = Signal(str)
    result = Signal(bool, str, str)      # (是否成功, python路径, 错误信息)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            ok, py, err = setup_nnunet_env(log_cb=self.log.emit,
                                           should_stop=lambda: self._stop)
        except Exception as e:  # noqa: BLE001
            ok, py, err = False, "", str(e)
        self.result.emit(ok, py, err)


class SegmentWorker(QThread):
    """后台分割线程：支持当前进程直接运行或子进程调用训练环境。"""
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, input_dir, output_dir, model, device,
                 save_probabilities, overwrite, python=None, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.device = device
        self.save_probabilities = save_probabilities
        self.overwrite = overwrite
        self.python = python          # None=当前进程；否则为训练环境 python.exe
        self._stop = False
        self._proc = None
        self._last_done = 0

    def stop(self):
        self._stop = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _reap(proc):
        """尽力回收子进程句柄（带超时，避免挂死）。"""
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def run(self):
        if self.python:
            self._run_subprocess()
        else:
            self._run_inprocess()

    def _run_inprocess(self):
        try:
            count = predict_folder(
                self.input_dir,
                self.output_dir,
                self.model,
                device=self.device,
                save_probabilities=self.save_probabilities,
                overwrite=self.overwrite,
                progress_cb=lambda i, n, msg: (setattr(self, "_last_done", i),
                                               self.progress.emit(i, n, msg)),
                log_cb=self.log.emit,
                should_stop=lambda: self._stop,
            )
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return
        self.done.emit(count)

    def _run_subprocess(self):
        """用训练环境的 python.exe 以子进程方式运行推理。"""
        fd, model_json = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.model, f, ensure_ascii=False)
            cmd = [self.python, str(_INFER_SCRIPT), model_json,
                   self.input_dir, self.output_dir]
            if self.save_probabilities:
                cmd.append("--save-prob")
            if self.overwrite:
                cmd.append("--overwrite")
            cmd += ["--device", self.device]

            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=flags,
            )
            self._proc = proc
            finished = False

            # 用独立读取线程 + 队列，主循环每 0.2s 检查一次“停止”，不会卡在管道上
            q = queue.Queue()

            def _reader():
                try:
                    for line in proc.stdout:
                        q.put(line)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    q.put(None)

            threading.Thread(target=_reader, daemon=True).start()

            while not finished:
                if self._stop:
                    # 用户要求停止：立即终止子进程并结束，不再等待管道/进程
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                    self.log.emit("已停止。")
                    self.done.emit(self._last_done)
                    return
                try:
                    line = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    self.log.emit(_ANSI_RE.sub("", line))
                    continue
                kind = obj.get("type")
                if kind == "progress":
                    self._last_done = int(obj.get("done", 0) or 0)
                    total = int(obj.get("total", 0) or 0)
                    self.progress.emit(self._last_done, total, obj.get("message", ""))
                elif kind == "log":
                    self.log.emit(obj.get("message", ""))
                elif kind == "done":
                    self.done.emit(int(obj.get("count", 0) or 0))
                    finished = True
                    break
                elif kind == "error":
                    self.failed.emit(obj.get("message", ""))
                    finished = True
                    break

            if not finished:
                if self._stop:
                    self.log.emit("已停止。")
                    self.done.emit(self._last_done)
                elif proc.returncode != 0:
                    self.failed.emit(f"nnUNet 进程异常退出（代码 {proc.returncode}）。")
                else:
                    self.failed.emit("nnUNet 进程未正常完成。")
            self._reap(proc)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"启动 nnUNet 环境失败：{e}")
        finally:
            try:
                os.unlink(model_json)
            except OSError:
                pass
            self._proc = None


class SegmentWindow(QMainWindow):
    """自动分割窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("② 自动分割")
        self.resize(900, 720)
        self.models = models_config.load_models()
        self.worker = None
        self.python_path = None       # 手动选择/自动检测到的训练环境 python.exe
        self._env_worker = None
        self._setup_worker = None     # 自动配置环境的线程
        self._env_mode = ""           # 当前 nnUNet 环境的推理设备 'GPU'/'CPU'/''
        self._env_mode_python = None

        self._build_ui()
        self._refresh_model_combo()
        self._init_env_label()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("② 自动分割")
        title.setProperty("class", "pageTitle")
        subtitle = QLabel("选择转换后的 NRRD 文件夹与模型 → 一键批量分割所有病例（结果保存为同名 NRRD）")
        subtitle.setProperty("class", "pageSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # ---- 模型设置 ----
        box_model = QGroupBox("模型设置")
        mb = QVBoxLayout(box_model)
        row = QHBoxLayout()
        row.addWidget(QLabel("使用模型"))
        self.combo_model = QComboBox()
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(self.combo_model, 1)
        btn_manage = make_button("模型管理…")
        btn_manage.clicked.connect(self._open_model_manager)
        row.addWidget(btn_manage)
        mb.addLayout(row)

        self.label_model_info = QLabel("请先在“模型管理”中配置模型文件夹路径。")
        self.label_model_info.setProperty("class", "hint")
        self.label_model_info.setWordWrap(True)
        mb.addWidget(self.label_model_info)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("模型目录"))
        self.line_models_dir = QLineEdit(str(MODELS_DIR))
        self.line_models_dir.setReadOnly(True)
        path_row.addWidget(self.line_models_dir, 1)
        btn_open = make_button("打开")
        btn_open.clicked.connect(self._open_models_dir)
        path_row.addWidget(btn_open)
        mb.addLayout(path_row)

        # nnUNet 环境
        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("nnUNet 环境"))
        self.label_env = QLabel("")
        self.label_env.setWordWrap(True)
        env_row.addWidget(self.label_env, 1)
        btn_detect = make_button("自动检测")
        btn_detect.clicked.connect(self.detect_env)
        btn_setup = make_button("自动配置环境", "primary")
        btn_setup.clicked.connect(self.setup_env)
        btn_choose = make_button("手动选择…")
        btn_choose.clicked.connect(self.choose_env)
        env_row.addWidget(btn_detect)
        env_row.addWidget(btn_setup)
        env_row.addWidget(btn_choose)
        mb.addLayout(env_row)
        root.addWidget(box_model)

        # ---- 路径 ----
        box_path = QGroupBox("路径设置")
        pb = QVBoxLayout(box_path)
        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("输入文件夹"))
        self.line_input = QLineEdit()
        self.line_input.setReadOnly(True)
        self.line_input.setPlaceholderText("界面①转换好的 NRRD 文件所在文件夹")
        row_in.addWidget(self.line_input, 1)
        btn_in = make_button("选择…")
        btn_in.clicked.connect(lambda: self._choose_dir(self.line_input, "选择 NRRD 输入文件夹"))
        row_in.addWidget(btn_in)
        pb.addLayout(row_in)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("输出文件夹"))
        self.line_output = QLineEdit()
        self.line_output.setReadOnly(True)
        self.line_output.setPlaceholderText("分割预测结果保存位置")
        row_out.addWidget(self.line_output, 1)
        btn_out = make_button("选择…")
        btn_out.clicked.connect(lambda: self._choose_dir(self.line_output, "选择预测结果输出文件夹"))
        row_out.addWidget(btn_out)
        pb.addLayout(row_out)
        root.addWidget(box_path)

        # ---- 选项 ----
        box_opt = QGroupBox("选项")
        ob = QHBoxLayout(box_opt)
        self.check_prob = QCheckBox("保存概率图（会占用大量磁盘空间）")
        self.check_overwrite = QCheckBox("覆盖已有结果")
        ob.addWidget(self.check_prob)
        ob.addWidget(self.check_overwrite)
        ob.addStretch(1)
        root.addWidget(box_opt)

        # ---- 操作 ----
        act = QHBoxLayout()
        self.btn_start = make_button("开始分割", "primary")
        self.btn_start.clicked.connect(self.start_segment)
        self.btn_stop = make_button("停止", "danger")
        self.btn_stop.clicked.connect(self.stop_segment)
        self.btn_stop.setEnabled(False)
        act.addWidget(self.btn_start)
        act.addWidget(self.btn_stop)
        act.addStretch(1)
        root.addLayout(act)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.label_state = QLabel("")
        self.label_state.setProperty("class", "hint")
        root.addWidget(self.label_state)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("运行日志…")
        root.addWidget(self.log_view, 1)

    # ------------------------------------------------------------- 逻辑
    def _refresh_model_combo(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        for m in self.models:
            self.combo_model.addItem(m.get("name", "未命名模型"))
        self.combo_model.blockSignals(False)
        self._on_model_changed()

    def _on_model_changed(self):
        idx = self.combo_model.currentIndex()
        if idx < 0 or idx >= len(self.models):
            self.label_model_info.setText("未配置模型。")
            return
        m = self.models[idx]
        folder = m.get("model_folder", "")
        labels = "、".join(labels_to_dict(m).values())
        info = f"模型文件夹：{folder or '（未设置）'}\n分割区域：{labels}"
        self.label_model_info.setText(info)

    def _init_env_label(self):
        self.label_env.setText("开始分割时将自动检测可用环境；也可点“自动检测”或“手动选择”指定训练环境")

    def detect_env(self):
        if self._env_worker is not None and self._env_worker.isRunning():
            return
        dlg = ProgressDialog("正在检测", "正在查找可用的 nnUNet 运行环境…", self)
        self._env_dialog = dlg
        worker = EnvDetectWorker()
        worker.result.connect(lambda res: (dlg.accept(), self._on_env_detected(res)))
        worker.start()
        self._env_worker = worker

    def _on_env_detected(self, res):
        if res:
            py, mode = res
            self.python_path = py
            self._env_mode = mode
            self._env_mode_python = py
            self.label_env.setText(f"已找到：{py}（{mode}）")
        else:
            self.python_path = None
            self._env_mode = ""
            self._env_mode_python = None
            self.label_env.setText("未找到 nnUNet 环境，可手动选择训练环境的 python.exe")

    def choose_env(self):
        py, _ = QFileDialog.getOpenFileName(self, "选择安装了 nnUNet 的 python.exe", "", "python.exe")
        if not py:
            return
        ok, mode = check_python(py)
        if ok:
            self.python_path = py
            self._env_mode = mode
            self._env_mode_python = py
            self.label_env.setText(f"已选择：{py}（{mode}）")
        else:
            self.python_path = None
            self._env_mode = ""
            self._env_mode_python = None
            QMessageBox.warning(self, "提示", "所选 Python 中未检测到 nnunetv2 / torch，请选择你的训练环境 python.exe。")

    def _resolve_device_mode(self, python) -> str:
        """根据模型配置与环境判断实际推理设备：返回 'GPU' / 'CPU' / ''（未知）。"""
        idx = self.combo_model.currentIndex()
        device = self.models[idx].get("device", "auto") if 0 <= idx < len(self.models) else "auto"
        if device == "cuda":
            return "GPU"
        if device == "cpu":
            return "CPU"
        # auto：探测环境实际能力
        if python is None:
            try:
                import torch  # noqa: PLC0415
                return "GPU" if torch.cuda.is_available() else "CPU"
            except Exception:
                return ""
        if getattr(self, "_env_mode_python", None) != python:
            ok, mode = check_python(python)
            self._env_mode = mode if ok else ""
            self._env_mode_python = python
        return self._env_mode

    def setup_env(self):
        """一键自动配置 nnUNet 运行环境（医生无需手动安装）。"""
        if self._setup_worker is not None and self._setup_worker.isRunning():
            return
        dlg = ProgressDialog(
            "正在配置环境",
            "正在自动安装 nnUNet 运行环境（需要联网，首次约需几分钟）…",
            self, cancellable=True)
        self._setup_dialog = dlg
        worker = EnvSetupWorker()
        dlg.cancel_requested.connect(worker.request_stop)
        worker.log.connect(lambda msg: (dlg.set_message(msg),
                                        self.log_view.appendPlainText(msg)))
        worker.result.connect(lambda ok, py, err: (dlg.accept(), self._on_setup_done(ok, py, err)))
        worker.start()
        self._setup_worker = worker

    def _on_setup_done(self, ok: bool, py: str, err: str):
        if ok:
            self.python_path = py
            mode = self._resolve_device_mode(py)
            self.label_env.setText(f"已自动配置：{py}（{mode}）")
            self.log_view.appendPlainText(f"nnUNet 环境配置完成，推理设备：{mode}。")
            msg = f"nnUNet 运行环境已自动配置完成，将使用 {mode} 进行分割。"
            if mode == "CPU":
                msg += "\n\n注意：CPU 推理速度较慢，每个病例可能需要数分钟。"
            QMessageBox.information(self, "配置完成", msg)
        else:
            self.label_env.setText("nnUNet 环境配置未完成")
            self.log_view.appendPlainText(f"配置未完成：{err}")
            if "已取消" in err:
                QMessageBox.information(self, "已取消", "已取消自动配置，可稍后重试。")
            else:
                QMessageBox.critical(
                    self, "配置失败",
                    f"自动配置 nnUNet 环境失败：\n{err}\n\n"
                    "可点击“手动选择…”指定你已有的训练环境 python.exe。")

    def _open_model_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.models_changed.connect(self._on_models_changed)
        dlg.exec()

    def _on_models_changed(self):
        self.models = models_config.load_models()
        self._refresh_model_combo()

    def _open_models_dir(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(MODELS_DIR))

    def _choose_dir(self, line_edit: QLineEdit, caption: str):
        folder = QFileDialog.getExistingDirectory(self, caption)
        if folder:
            line_edit.setText(folder)

    def start_segment(self):
        if self.worker is not None and self.worker.isRunning():
            return
        idx = self.combo_model.currentIndex()
        if idx < 0 or idx >= len(self.models):
            QMessageBox.warning(self, "提示", "请先在“模型管理”中配置模型。")
            return
        model = self.models[idx]
        problem = validate_model(model)
        if problem:
            QMessageBox.warning(self, "提示", problem)
            return
        input_dir = self.line_input.text().strip()
        output_dir = self.line_output.text().strip()
        if not input_dir or not Path(input_dir).exists():
            QMessageBox.warning(self, "提示", "请选择有效的输入文件夹。")
            return
        if not output_dir:
            QMessageBox.warning(self, "提示", "请选择输出文件夹。")
            return
        nrrd_files = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in (".nrrd", ".nii.gz")]
        if not nrrd_files:
            QMessageBox.warning(self, "提示", "输入文件夹中没有 NRRD 文件。")
            return

        # 确定 nnUNet 运行方式
        if current_python_has_nnunet():
            self._begin_seg(None)
            return
        if self.python_path:
            self._begin_seg(self.python_path)
            return
        # 自动检测训练环境（模态）
        dlg = ProgressDialog("正在检测", "正在查找可用的 nnUNet 运行环境…", self)
        self._env_dialog = dlg
        worker = EnvDetectWorker()
        worker.result.connect(lambda res: (dlg.accept(), self._on_detect_for_start(res)))
        worker.start()
        self._env_worker = worker

    def _on_detect_for_start(self, res):
        if res:
            py, mode = res
            self.python_path = py
            self.label_env.setText(f"已找到：{py}（{mode}）")
            self._begin_seg(py)
        else:
            self.python_path = None
            self.label_env.setText("未找到 nnUNet 环境")
            ans = QMessageBox.question(
                self, "未找到 nnUNet 环境",
                "未找到可用的 nnUNet 运行环境。\n\n"
                "是否现在自动安装？\n（需要联网，软件会自动创建环境并安装 "
                "PyTorch + nnUNet V2，首次约需几分钟）")
            if ans == QMessageBox.StandardButton.Yes:
                self.setup_env()
            else:
                QMessageBox.warning(
                    self, "提示",
                    "未找到可用的 nnUNet 运行环境。\n\n"
                    "请点击“自动配置环境”自动安装，或“手动选择…”指定已有的训练环境 python.exe。")

    def _begin_seg(self, python):
        """开始分割。python=None 表示用当前进程，否则用训练环境子进程。"""
        idx = self.combo_model.currentIndex()
        model = self.models[idx]
        input_dir = self.line_input.text().strip()
        output_dir = self.line_output.text().strip()
        device = model.get("device", "auto")
        nrrd_files = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in (".nrrd", ".nii.gz")]

        # 提示当前使用 GPU 还是 CPU
        mode = self._resolve_device_mode(python)
        if mode == "GPU":
            QMessageBox.information(self, "推理设备", "将使用 GPU 进行分割。")
        elif mode == "CPU":
            ans = QMessageBox.question(
                self, "推理设备",
                "当前使用 CPU 进行分割，预测时间可能较长（每个病例可能需要数分钟）。\n是否继续？")
            if ans != QMessageBox.StandardButton.Yes:
                return

        total = len(nrrd_files)
        self.log_view.clear()
        self.log_view.appendPlainText(f"输入文件夹共 {total} 例。")
        self.progress.setVisible(True)
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self._set_running(True)
        self.label_state.setText(f"共 {total} 例，正在加载模型…")

        self.worker = SegmentWorker(input_dir, output_dir, model, device,
                                    self.check_prob.isChecked(), self.check_overwrite.isChecked(),
                                    python=python)
        dlg = ProgressDialog("正在分割", f"共 {total} 例，正在加载模型并分割，请稍候…", self, cancellable=True)
        self._seg_dialog = dlg
        dlg.cancel_requested.connect(self.stop_segment)
        self.worker.log.connect(lambda msg: self.log_view.appendPlainText(msg))
        self.worker.progress.connect(lambda i, n, msg: (self.progress.setValue(i),
                                                        self.label_state.setText(msg),
                                                        dlg.set_progress(i, n),
                                                        dlg.set_message(msg)))
        self.worker.done.connect(lambda count: (dlg.accept(), self._on_done(count)))
        self.worker.failed.connect(lambda msg: (dlg.accept(), self._on_failed(msg)))
        self.worker.start()

    def stop_segment(self):
        if self.worker is not None:
            self.worker.stop()
            self.label_state.setText("正在停止…")

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.combo_model.setEnabled(not running)
        self.line_input.setEnabled(not running)
        self.line_output.setEnabled(not running)

    def _on_done(self, count: int):
        self._set_running(False)
        self.progress.setValue(self.progress.maximum())
        self.label_state.setText(f"分割结束，成功处理 {count} 个病例（详见日志）。")
        self.log_view.appendPlainText(f"完成：成功处理 {count} 个病例。")

    def _on_failed(self, msg: str):
        self._set_running(False)
        self.progress.setVisible(False)
        self.label_state.setText("分割失败。")
        self.log_view.appendPlainText(f"错误：{msg}")
        QMessageBox.critical(self, "分割失败", msg)
