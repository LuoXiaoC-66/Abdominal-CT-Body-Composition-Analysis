# -*- coding: utf-8 -*-
"""界面一：图像转换与定位。

- 添加数据文件夹：在后台线程递归扫描，每扫到一个病例就立即显示在列表中
  （列表只显示病人姓名，详细信息显示在下方），并提示还有多少例未加载
- 轴位 / 冠状位 / 矢状位三视图联动：滚轮切层、鼠标定位；
  轴位选择的层面会同步更新冠状位/矢状位的参考线
- 切换病人时保持所查看的解剖位置（按物理坐标 mm 换算，兼容不同层厚），
  并立即清空旧图避免显示不一致
- 多期相自动只取第一期（通常为平扫）
- 在冠状位/矢状位上通过右键菜单或按钮设置起始/结束层面，导出为 NRRD
- 导出文件名自动命名为 Patient_序号_0000（按导入顺序编号，已有同名格式文件时接着编号）
"""
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QSpinBox, QVBoxLayout, QWidget,
)

from app.constants import EXPORT_BASE_NAME, EXPORT_MODALITY_SUFFIX, WINDOW_LEVEL_PRESETS
from app.dicom import SeriesInfo, collect_import_folders, load_series_image, scan_folder
from app.nrrd_io import export_subvolume, write_image
from app.style import make_button
from ui.image_view import (
    MedicalImageView, numpy_to_qimage,
    RANGE_START_COLOR, RANGE_END_COLOR,
)
from ui.progress_dialog import ProgressDialog

VIEW_AXIAL = "axial"
VIEW_CORONAL = "coronal"
VIEW_SAGITTAL = "sagittal"

_EXPORT_NAME_RE = re.compile(
    rf"^{re.escape(EXPORT_BASE_NAME)}_(\d+){re.escape(EXPORT_MODALITY_SUFFIX)}\.nrrd$",
    re.IGNORECASE)


def find_existing_export_numbers(out_dir) -> set:
    """扫描输出文件夹中已有的 Patient_序号_0000.nrrd 文件，返回序号集合。"""
    folder = Path(out_dir)
    if not folder.exists():
        return set()
    nums = set()
    for f in folder.glob("*.nrrd"):
        m = _EXPORT_NAME_RE.match(f.name)
        if m:
            nums.add(int(m.group(1)))
    return nums


def next_export_start(out_dir) -> int:
    """下一次导出的起始序号：已有最大序号 + 1；没有则从 1 开始。"""
    nums = find_existing_export_numbers(out_dir)
    return (max(nums) + 1) if nums else 1


def _display_name(series: SeriesInfo, folder) -> str:
    """列表/信息里显示的姓名：优先 DICOM 标签，若不可读（含替换符/空白）则用文件夹名。"""
    name = (series.patient_name or "").replace("\ufffd", "").strip(" ?　")
    return name or Path(folder).name


class DICOMLoadWorker(QThread):
    """后台加载 DICOM 序列。"""
    loaded = Signal(object, object)   # (SeriesInfo, sitk.Image)
    failed = Signal(str)

    def __init__(self, series: SeriesInfo, parent=None):
        super().__init__(parent)
        self.series = series

    def run(self):
        try:
            image = load_series_image(self.series)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"读取 DICOM 失败：{e}")
            return
        self.loaded.emit(self.series, image)


class FolderScanWorker(QThread):
    """后台扫描：递归寻找 DICOM，每扫完一个病例立即上报（不阻塞界面）。"""
    entry_loaded = Signal(object, int, int)   # (病例条目, 已处理数, 总数)
    finished = Signal(int)                    # 总数（0 表示未找到）
    failed = Signal(str)

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = Path(root)

    def run(self):
        try:
            folders = collect_import_folders(self.root)
            total = len(folders)
            if total == 0:
                self.finished.emit(0)
                return
            done = 0
            for folder in folders:
                series_list = scan_folder(folder)
                done += 1
                if not series_list:
                    continue
                entry = {
                    "folder": folder,
                    "series_list": series_list,
                    "series_idx": 0,
                    "range": None,          # (start, end) 1 起索引；None = 未设置
                }
                self.entry_loaded.emit(entry, done, total)
            self.finished.emit(total)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportTask:
    """单个病例的导出任务。"""

    def __init__(self, series: SeriesInfo, prefix: str, out_dir: str,
                 use_range: bool, z_start: int, z_end: int):
        self.series = series
        self.prefix = prefix
        self.out_dir = out_dir
        self.use_range = use_range
        self.z_start = z_start      # 0 起索引
        self.z_end = z_end          # 0 起索引


class NRRDExportWorker(QThread):
    """后台导出 NRRD。"""
    progress = Signal(int, int, str)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, tasks, parent=None):
        super().__init__(parent)
        self.tasks = tasks

    def run(self):
        total = len(self.tasks)
        for i, task in enumerate(self.tasks, start=1):
            try:
                image = load_series_image(task.series)
                out_path = Path(task.out_dir) / f"{task.prefix}.nrrd"
                if task.use_range:
                    export_subvolume(image, out_path, task.z_start, task.z_end)
                else:
                    write_image(image, out_path)
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"导出 {task.prefix} 失败：{e}")
                return
            self.progress.emit(i, total, f"已导出 {out_path.name} ({i}/{total})")
        self.done.emit(total)


class ConvertWindow(QMainWindow):
    """图像转换与定位窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("① 图像转换与定位")
        self.resize(1280, 820)

        self.patients = []           # [{folder, series_list, series_idx, range}]
        self.current = -1            # 当前病人索引
        self.image = None            # 当前 SimpleITK 图像
        self.arr = None              # (Z, Y, X)
        self.pos = [0, 0, 0]         # [x, y, z]
        self.wl, self.ww = 40, 350
        self._load_gen = 0           # 加载代次，用于丢弃过期结果
        self._load_workers = []      # 持有正在运行的加载线程，避免销毁
        self._updating = False
        self._scan_worker = None
        self._last_spacing = None    # 上一个已加载图像的 spacing
        self._last_pos_mm = None     # 上一个查看位置的物理坐标 (mm)
        self._view_images = {}       # 各视图当前 QImage 缓存
        self._last_view_slices = None
        self._pending_removals = set()   # 导出成功后待移除的病人文件夹

        self._build_ui()
        self._apply_wl_preset("腹部软组织窗")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        root.addLayout(self._build_left_panel(), 0)
        root.addLayout(self._build_right_panel(), 1)

    def _build_left_panel(self):
        left = QVBoxLayout()
        left.setSpacing(10)

        # ---- 病人与序列 ----
        box = QGroupBox("病人与序列")
        vb = QVBoxLayout(box)
        row = QHBoxLayout()
        self.btn_add_patient = make_button("添加数据文件夹", "default")
        self.btn_add_patient.clicked.connect(self.add_patient_folder)
        self.btn_batch = make_button("批量添加", "default")
        self.btn_batch.clicked.connect(self.add_batch_folder)
        row.addWidget(self.btn_add_patient)
        row.addWidget(self.btn_batch)
        vb.addLayout(row)

        self.list_patients = QListWidget()
        self.list_patients.currentRowChanged.connect(self.on_patient_selected)
        vb.addWidget(self.list_patients, 1)

        self.label_info = QLabel("请添加包含 DICOM 文件的文件夹")
        self.label_info.setWordWrap(True)
        self.label_info.setProperty("class", "hint")
        vb.addWidget(self.label_info)

        nav = QHBoxLayout()
        btn_prev = make_button("◀ 上一例")
        btn_prev.clicked.connect(lambda: self.step_patient(-1))
        btn_next = make_button("下一例 ▶")
        btn_next.clicked.connect(lambda: self.step_patient(1))
        btn_del = make_button("删除当前", "danger")
        btn_del.clicked.connect(self.delete_current_patient)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_del)
        nav.addWidget(btn_next)
        vb.addLayout(nav)
        left.addWidget(box)

        # ---- 窗宽窗位 ----
        wlbox = QGroupBox("窗宽窗位")
        wb = QVBoxLayout(wlbox)
        self.combo_wl = QComboBox()
        self.combo_wl.addItems(list(WINDOW_LEVEL_PRESETS.keys()))
        self.combo_wl.currentTextChanged.connect(self._apply_wl_preset)
        wb.addWidget(self.combo_wl)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("窗位"))
        self.spin_level = QSpinBox()
        self.spin_level.setRange(-1024, 3071)
        self.spin_level.setValue(self.wl)
        self.spin_level.valueChanged.connect(self._on_wl_spin_changed)
        wrow.addWidget(self.spin_level, 1)
        wrow.addWidget(QLabel("窗宽"))
        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 4095)
        self.spin_width.setValue(self.ww)
        self.spin_width.valueChanged.connect(self._on_wl_spin_changed)
        wrow.addWidget(self.spin_width, 1)
        wb.addLayout(wrow)
        left.addWidget(wlbox)

        left.addStretch(1)
        return left

    def _build_right_panel(self):
        right = QVBoxLayout()
        right.setSpacing(10)

        title = QLabel("① 图像转换与定位")
        title.setProperty("class", "pageTitle")
        subtitle = QLabel("导入 DICOM → 浏览轴位/冠状位/矢状位 → 设置导出层面范围 → 转换为 NRRD")
        subtitle.setProperty("class", "pageSubtitle")
        right.addWidget(title)
        right.addWidget(subtitle)

        # ---- 三视图 ----
        views = QHBoxLayout()
        views.setSpacing(10)
        self.views = {}
        self.slice_labels = {}
        for vid, name in ((VIEW_AXIAL, "轴位"), (VIEW_CORONAL, "冠状位"), (VIEW_SAGITTAL, "矢状位")):
            col = QVBoxLayout()
            header = QHBoxLayout()
            cap = QLabel(name)
            cap.setProperty("class", "sectionTitle")
            header.addWidget(cap)
            header.addStretch(1)
            sl = QLabel("-")
            sl.setProperty("class", "hint")
            header.addWidget(sl)
            col.addLayout(header)
            view = MedicalImageView()
            view.clicked.connect(lambda c, r, v=vid: self._on_view_clicked(v, c, r))
            view.hovered.connect(lambda c, r, v=vid: self._on_view_hovered(v, c, r))
            view.scrolled.connect(lambda steps, v=vid: self._on_view_scrolled(v, steps))
            view.context_menu.connect(lambda c, r, v=vid: self._on_view_context_menu(v, c, r))
            col.addWidget(view, 1)
            self.views[vid] = view
            self.slice_labels[vid] = sl
            views.addLayout(col)
        right.addLayout(views, 1)

        # ---- 层面范围 ----
        range_box = QGroupBox("导出层面范围（冠状位/矢状位上右键可直接设置）")
        rb = QHBoxLayout(range_box)
        rb.addWidget(QLabel("起始层面"))
        self.spin_range_start = QSpinBox()
        self.spin_range_start.setRange(1, 1)
        self.spin_range_start.valueChanged.connect(self._on_range_changed)
        rb.addWidget(self.spin_range_start)
        btn_start = make_button("设为当前层")
        btn_start.clicked.connect(lambda: self._set_range_from_current("start"))
        rb.addWidget(btn_start)

        rb.addSpacing(12)
        rb.addWidget(QLabel("结束层面"))
        self.spin_range_end = QSpinBox()
        self.spin_range_end.setRange(1, 1)
        self.spin_range_end.valueChanged.connect(self._on_range_changed)
        rb.addWidget(self.spin_range_end)
        btn_end = make_button("设为当前层")
        btn_end.clicked.connect(lambda: self._set_range_from_current("end"))
        rb.addWidget(btn_end)

        btn_full = make_button("全部范围")
        btn_full.clicked.connect(self._reset_range_full)
        rb.addWidget(btn_full)
        self.label_range_info = QLabel("")
        self.label_range_info.setProperty("class", "hint")
        rb.addWidget(self.label_range_info)
        rb.addStretch(1)
        right.addWidget(range_box)

        # ---- 导出 ----
        export_box = QGroupBox("导出 NRRD")
        eb = QVBoxLayout(export_box)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出文件夹"))
        self.line_out_dir = QLineEdit()
        self.line_out_dir.setReadOnly(True)
        out_row.addWidget(self.line_out_dir, 1)
        btn_out = make_button("选择…")
        btn_out.clicked.connect(self._choose_out_dir)
        out_row.addWidget(btn_out)
        eb.addLayout(out_row)

        self.label_naming = QLabel()
        self.label_naming.setWordWrap(True)
        self.label_naming.setProperty("class", "hint")
        self._update_naming_hint()
        eb.addWidget(self.label_naming)

        opt_row = QHBoxLayout()
        self.check_use_range = QCheckBox("仅导出选定层面范围")
        self.check_use_range.setChecked(True)
        self.check_remove_after_export = QCheckBox("导出成功后自动从列表移除")
        self.check_remove_after_export.setChecked(True)
        opt_row.addWidget(self.check_use_range)
        opt_row.addWidget(self.check_remove_after_export)
        opt_row.addStretch(1)
        self.btn_export_one = make_button("导出当前病例", "primary")
        self.btn_export_one.clicked.connect(self.export_current)
        self.btn_export_all = make_button("导出全部病例", "primary")
        self.btn_export_all.clicked.connect(self.export_all)
        opt_row.addWidget(self.btn_export_one)
        opt_row.addWidget(self.btn_export_all)
        eb.addLayout(opt_row)

        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        eb.addWidget(self.export_progress)
        self.label_export_state = QLabel("")
        self.label_export_state.setProperty("class", "hint")
        eb.addWidget(self.label_export_state)
        right.addWidget(export_box)

        return right

    # ----------------------------------------------------------- 病人管理
    def add_patient_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据文件夹（可包含多层子文件夹）")
        if not folder:
            return
        self._import_root(Path(folder))

    def add_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含多个病人子文件夹的父文件夹")
        if not folder:
            return
        self._import_root(Path(folder))

    def _import_root(self, root: Path):
        """后台扫描 root：弹模态框阻塞其他操作，每扫到一个病例立即加入列表。"""
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        self._set_scanning(True)
        dlg = ProgressDialog("正在扫描", "正在扫描 DICOM 文件，请稍候…", self)
        self._scan_dialog = dlg
        worker = FolderScanWorker(root)
        worker.entry_loaded.connect(
            lambda e, d, t: (self._on_entry_loaded(e, d, t),
                             dlg.set_message(f"正在加载病例… 已处理 {d}/{t}，还有 {t - d} 例未加载")))
        worker.finished.connect(lambda total: (dlg.accept(), self._on_scan_finished(total)))
        worker.failed.connect(lambda msg: (dlg.accept(), self._on_scan_failed(msg)))
        worker.start()
        self._scan_worker = worker

    def _set_scanning(self, scanning: bool):
        self.btn_add_patient.setEnabled(not scanning)
        self.btn_batch.setEnabled(not scanning)
        if scanning:
            self.label_info.setText("正在扫描 DICOM 文件，请稍候…")

    def _on_entry_loaded(self, entry, done, total):
        """每扫完一个病例立即显示。"""
        if any(p["folder"] == entry["folder"] for p in self.patients):
            return
        self.patients.append(entry)
        name = _display_name(entry["series_list"][0], entry["folder"])
        self.list_patients.addItem(name)
        remaining = max(0, total - done)
        self.label_info.setText(
            f"正在加载病例… 已处理 {done}/{total}，还有 {remaining} 例未加载")

    def _on_scan_finished(self, total):
        self._set_scanning(False)
        if total == 0:
            QMessageBox.information(self, "提示", "未在所选文件夹中找到 DICOM 序列。")
            self.label_info.setText("请添加包含 DICOM 文件的文件夹")
            return
        if self.current < 0 and self.patients:
            self.list_patients.setCurrentRow(0)   # 自动选中第一个病例
        elif self.current >= 0 and self.arr is not None:
            self._update_patient_info(self.patients[self.current]["series_list"][0])
        else:
            self.label_info.setText(f"扫描完成，共 {len(self.patients)} 例。")

    def _on_scan_failed(self, msg: str):
        self._set_scanning(False)
        QMessageBox.warning(self, "扫描失败", msg)

    def _add_folders(self, folders):
        """同步添加指定文件夹（供内部/测试复用）。"""
        added = 0
        for folder in folders:
            folder = Path(folder)
            if not folder.exists():
                continue
            if any(p["folder"] == folder for p in self.patients):
                continue
            series_list = scan_folder(folder)
            if not series_list:
                continue
            self.patients.append({
                "folder": folder,
                "series_list": series_list,
                "series_idx": 0,
                "range": None,          # (start, end) 1 起索引；None = 未设置
            })
            added += 1
        if added == 0:
            QMessageBox.information(self, "提示", "未在所选文件夹中找到 DICOM 序列。")
            return
        self._refresh_patient_list()
        self.list_patients.setCurrentRow(len(self.patients) - 1)

    def _refresh_patient_list(self):
        self.list_patients.clear()
        for p in self.patients:
            s = p["series_list"][0]           # 只展示第一期
            name = _display_name(s, p["folder"])
            self.list_patients.addItem(name)

    def step_patient(self, delta: int):
        if not self.patients:
            return
        n = len(self.patients)
        self.list_patients.setCurrentRow((self.current + delta) % n)

    def delete_current_patient(self):
        """从列表删除当前病人（不删除磁盘文件）。"""
        if self.current < 0 or not self.patients:
            QMessageBox.information(self, "提示", "当前没有可删除的病人。")
            return
        entry = self.patients[self.current]
        name = _display_name(entry["series_list"][0], entry["folder"])
        ans = QMessageBox.question(
            self, "删除病人",
            f"确定从列表删除病人“{name}”？\n（仅从列表移除，不会删除磁盘上的 DICOM 文件）")
        if ans != QMessageBox.StandardButton.Yes:
            return
        self.remove_patient(self.current)

    def remove_patient(self, idx: int):
        """按索引删除病人，并自动选中下一个病人。"""
        if idx < 0 or idx >= len(self.patients):
            return
        del self.patients[idx]
        if self.current > idx:
            self.current -= 1
        elif self.current >= len(self.patients):
            self.current = len(self.patients) - 1
        self._refresh_patient_list()
        if not self.patients:
            self.current = -1
            self.image = None
            self.arr = None
            self._clear_views()
            self.label_info.setText("请添加包含 DICOM 文件的文件夹")
            return
        self.list_patients.setCurrentRow(max(0, self.current))

    def remove_patients_by_folders(self, folders) -> int:
        """按文件夹路径批量移除病人（导出成功后调用），返回移除数量。"""
        folders = set(folders)
        if not folders:
            return 0
        removed_current = (self.current >= 0 and self.current < len(self.patients)
                           and self.patients[self.current]["folder"] in folders)
        keep = [p for p in self.patients if p["folder"] not in folders]
        removed = len(self.patients) - len(keep)
        self.patients = keep
        if not self.patients:
            self.current = -1
            self.image = None
            self.arr = None
            self._clear_views()
            self._refresh_patient_list()
            self.label_info.setText("请添加包含 DICOM 文件的文件夹")
            return removed
        if removed_current:
            self.current = min(self.current, len(self.patients) - 1)
        elif self.current >= len(self.patients):
            self.current = len(self.patients) - 1
        if self.current < 0:
            self.current = 0
        self._refresh_patient_list()
        self.list_patients.setCurrentRow(self.current)
        return removed

    def on_patient_selected(self, row: int):
        if row < 0 or row >= len(self.patients):
            return
        # 保存上一个病人的范围设置
        if 0 <= self.current < len(self.patients) and self.arr is not None:
            self.patients[self.current]["range"] = (self.spin_range_start.value(),
                                                    self.spin_range_end.value())
        self.current = row
        self._load_current_series()

    def _load_current_series(self):
        if self.current < 0:
            return
        entry = self.patients[self.current]
        series = entry["series_list"][0]      # 只展示第一期（通常为平扫）
        self._load_gen += 1
        gen = self._load_gen
        # 立即清空旧图，避免切换病人时三视图显示不一致
        self._clear_views()
        self.label_info.setText(f"正在加载：{series.patient_name or series.patient_id} ...")
        self._set_loading(True)
        if getattr(self, "_load_dialog", None) is not None:
            self._load_dialog.accept()      # 关闭上一个加载框
        dlg = ProgressDialog("正在加载", f"正在加载病人图像：{_display_name(series, entry['folder'])} …", self)
        self._load_dialog = dlg
        worker = DICOMLoadWorker(series)
        worker.loaded.connect(lambda s, img, g=gen: (dlg.accept(), self._on_series_loaded(s, img, g)))
        worker.failed.connect(lambda msg, g=gen: (dlg.accept(), self._on_load_failed(msg, g)))
        worker.finished.connect(lambda w=worker: self._discard_load_worker(w))
        self._load_workers.append(worker)   # 持有引用，避免线程未结束即被销毁
        worker.start()
        self._load_worker = worker

    def _discard_load_worker(self, worker):
        """加载线程结束后从注册表中移除。"""
        if worker in self._load_workers:
            self._load_workers.remove(worker)

    def _clear_views(self):
        for vid in (VIEW_AXIAL, VIEW_CORONAL, VIEW_SAGITTAL):
            self.views[vid].clear_image()
            self.slice_labels[vid].setText("-")
        self._view_images = {}
        self._last_view_slices = None

    def _on_load_failed(self, msg: str, gen: int):
        if gen != self._load_gen:
            return
        self._set_loading(False)
        QMessageBox.warning(self, "加载失败", msg)

    def _on_series_loaded(self, series: SeriesInfo, image, gen: int):
        if gen != self._load_gen:
            return
        if self.current < 0 or self.current >= len(self.patients):
            return          # 病人可能已被删除
        self.image = image
        self.arr = sitk.GetArrayFromImage(image)
        Z, Y, X = self.arr.shape
        spacing = image.GetSpacing()          # (x, y, z)
        # 保持上一个病人所看位置的物理坐标（mm），兼容不同层厚
        if self._last_spacing is not None and self._last_pos_mm is not None:
            pos = [max(0, min(int(round(m / sp)), n - 1))
                   for m, sp, n in zip(self._last_pos_mm, spacing, (X, Y, Z))]
        else:
            pos = [X // 2, Y // 2, Z // 2]
        self.pos = pos
        self._last_spacing = spacing
        self._last_pos_mm = [pos[i] * spacing[i] for i in range(3)]
        self._set_view_spacing()

        entry = self.patients[self.current]
        if entry["range"] is None:
            entry["range"] = (1, Z)
        self._set_loading(False)
        self._apply_range_to_spinboxes()
        self._update_patient_info(series)
        self.update_all_views()

    def _set_view_spacing(self):
        """按体素物理间距设置三视图纵横比，厚层重建冠状位/矢状位不“扁”。"""
        if self.image is None:
            return
        sp = self.image.GetSpacing()          # (x, y, z)
        self.views[VIEW_AXIAL].set_pixel_size(sp[0], sp[1])
        self.views[VIEW_CORONAL].set_pixel_size(sp[0], sp[2])
        self.views[VIEW_SAGITTAL].set_pixel_size(sp[1], sp[2])

    def _set_loading(self, loading: bool):
        self.list_patients.setEnabled(not loading)

    def _update_patient_info(self, series: SeriesInfo):
        spacing = self.image.GetSpacing()
        entry = self.patients[self.current]
        lines = [f"姓名：{_display_name(series, entry['folder'])}",
                 f"ID：{series.patient_id or '-'}　日期：{series.study_date or '-'}"]
        lines.append(f"序列：{series.series_description or '-'}（{series.modality}）　{series.instance_count} 层")
        if len(entry["series_list"]) > 1:
            lines.append("（该文件夹含多个序列/期相，已自动选择第一期，通常为平扫）")
        lines.append(f"体数据：{self.arr.shape[2]}×{self.arr.shape[1]}×{self.arr.shape[0]}　"
                     f"层厚/间距：{spacing[2]:.2f} mm")
        lines.append(f"体素间距：{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm")
        self.label_info.setText("\n".join(lines))
        self.label_info.setProperty("class", "")

    # ------------------------------------------------------------ 三视图
    def _window(self, arr2d):
        wmin = self.wl - self.ww / 2.0
        wmax = self.wl + self.ww / 2.0
        out = np.clip(arr2d, wmin, wmax)
        out = (out - wmin) / max(self.ww, 1e-6) * 255.0
        return out.astype(np.uint8)

    def update_all_views(self, force: bool = False):
        """刷新三视图。

        force=True 时强制重建所有视图图像（用于窗宽窗位变化）；
        否则只重建层面发生变化的视图，其余仅更新十字线/范围线（避免滚轮闪烁）。
        """
        if self.arr is None:
            return
        Z, Y, X = self.arr.shape
        x, y, z = self.pos

        prev = self._last_view_slices
        rebuild = {
            VIEW_AXIAL: force or prev is None or z != prev[2],
            VIEW_CORONAL: force or prev is None or y != prev[1],
            VIEW_SAGITTAL: force or prev is None or x != prev[0],
        }
        self._last_view_slices = (x, y, z)

        if rebuild[VIEW_AXIAL]:
            self._view_images[VIEW_AXIAL] = numpy_to_qimage(self._window(self.arr[z]))
        if rebuild[VIEW_CORONAL]:
            self._view_images[VIEW_CORONAL] = numpy_to_qimage(self._window(self.arr[:, y, :]))
        if rebuild[VIEW_SAGITTAL]:
            self._view_images[VIEW_SAGITTAL] = numpy_to_qimage(self._window(self.arr[:, :, x]))

        rows = []
        if self.spin_range_start.maximum() > 0:
            s, e = self.spin_range_start.value() - 1, self.spin_range_end.value() - 1
            rows = [(s, RANGE_START_COLOR, f"起始 {s + 1}"),
                    (e, RANGE_END_COLOR, f"结束 {e + 1}")]

        self.views[VIEW_AXIAL].set_content(self._view_images.get(VIEW_AXIAL), (x, y), [])
        self.views[VIEW_CORONAL].set_content(self._view_images.get(VIEW_CORONAL), (x, z), rows)
        self.views[VIEW_SAGITTAL].set_content(self._view_images.get(VIEW_SAGITTAL), (y, z), rows)

        self.slice_labels[VIEW_AXIAL].setText(f"第 {z + 1}/{Z} 层")
        self.slice_labels[VIEW_CORONAL].setText(f"第 {y + 1}/{Y} 层")
        self.slice_labels[VIEW_SAGITTAL].setText(f"第 {x + 1}/{X} 层")

    def set_pos(self, x: int, y: int, z: int):
        if self.arr is None:
            return
        Z, Y, X = self.arr.shape
        self.pos = [max(0, min(x, X - 1)), max(0, min(y, Y - 1)), max(0, min(z, Z - 1))]
        if self.image is not None:
            sp = self.image.GetSpacing()
            self._last_pos_mm = [self.pos[i] * sp[i] for i in range(3)]
        self.update_all_views()

    def _on_view_clicked(self, vid: str, col: int, row: int):
        if self.arr is None:
            return
        x, y, z = self.pos
        if vid == VIEW_AXIAL:
            x, y = col, row
        elif vid == VIEW_CORONAL:
            x, z = col, row
        else:  # 矢状位：col=y, row=z
            y, z = col, row
        self.set_pos(x, y, z)

    def _on_view_hovered(self, vid: str, col: int, row: int):
        if self.arr is None:
            return
        self.views[vid].set_crosshair(col, row)

    def _on_view_scrolled(self, vid: str, steps: int):
        if self.arr is None:
            return
        x, y, z = self.pos
        if vid == VIEW_AXIAL:
            z += steps
        elif vid == VIEW_CORONAL:
            y += steps
        else:
            x += steps
        self.set_pos(x, y, z)

    def _on_view_context_menu(self, vid: str, col: int, row: int):
        if self.arr is None or vid == VIEW_AXIAL:
            return
        menu = QMenu(self)
        act_jump = QAction(f"跳转到第 {row + 1} 层", self)
        act_start = QAction("将此层设为起始层面", self)
        act_end = QAction("将此层设为结束层面", self)
        menu.addAction(act_jump)
        menu.addSeparator()
        menu.addAction(act_start)
        menu.addAction(act_end)
        chosen = menu.exec(self.cursor().pos())
        if chosen is act_jump:
            x, y, z = self.pos
            self.set_pos(x, y, row)
        elif chosen is act_start:
            self.spin_range_start.setValue(row + 1)
        elif chosen is act_end:
            self.spin_range_end.setValue(row + 1)

    # ------------------------------------------------------------ 范围
    def _apply_range_to_spinboxes(self):
        if self.arr is None:
            return
        Z = self.arr.shape[0]
        entry = self.patients[self.current]
        s, e = entry["range"] or (1, Z)
        self._updating = True
        self.spin_range_start.setRange(1, Z)
        self.spin_range_end.setRange(1, Z)
        self.spin_range_start.setValue(s)
        self.spin_range_end.setValue(e)
        self._updating = False
        self._refresh_range_info()

    def _on_range_changed(self):
        if self._updating or self.arr is None:
            return
        self._updating = True
        s = min(self.spin_range_start.value(), self.spin_range_end.value())
        e = max(self.spin_range_start.value(), self.spin_range_end.value())
        self.spin_range_start.setValue(s)
        self.spin_range_end.setValue(e)
        self._updating = False
        if 0 <= self.current < len(self.patients):
            self.patients[self.current]["range"] = (s, e)
        self._refresh_range_info()
        self.update_all_views()

    def _set_range_from_current(self, which: str):
        if self.arr is None:
            return
        z = self.pos[2] + 1
        if which == "start":
            self.spin_range_start.setValue(z)
        else:
            self.spin_range_end.setValue(z)

    def _reset_range_full(self):
        if self.arr is None:
            return
        self._updating = True
        self.spin_range_start.setValue(1)
        self.spin_range_end.setValue(self.arr.shape[0])
        self._updating = False
        if 0 <= self.current < len(self.patients):
            self.patients[self.current]["range"] = (1, self.arr.shape[0])
        self._refresh_range_info()
        self.update_all_views()

    def _refresh_range_info(self):
        if self.arr is None:
            self.label_range_info.setText("")
            return
        s, e = self.spin_range_start.value(), self.spin_range_end.value()
        self.label_range_info.setText(f"共 {e - s + 1} 层（第 {s} ~ {e} 层）")

    # -------------------------------------------------------- 窗宽窗位
    def _apply_wl_preset(self, name: str):
        preset = WINDOW_LEVEL_PRESETS.get(name)
        if preset is None:
            return
        level, width = preset
        self._updating = True
        self.spin_level.setValue(level)
        self.spin_width.setValue(width)
        self._updating = False
        self.wl, self.ww = level, width
        self.update_all_views(force=True)

    def _on_wl_spin_changed(self):
        if self._updating:
            return
        self.wl, self.ww = self.spin_level.value(), self.spin_width.value()
        if self.combo_wl.currentText() != "自定义":
            self.combo_wl.setCurrentText("自定义")
        self.update_all_views(force=True)

    # ------------------------------------------------------------ 导出
    def _choose_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 NRRD 输出文件夹")
        if folder:
            self.line_out_dir.setText(folder)
            self._update_naming_hint()

    def _update_naming_hint(self):
        """显示自动命名规则，并在选择输出文件夹后提示起始序号。"""
        out_dir = self.line_out_dir.text().strip()
        base = f"{EXPORT_BASE_NAME}_序号{EXPORT_MODALITY_SUFFIX}.nrrd"
        if not out_dir:
            self.label_naming.setText(
                f"导出文件名自动命名：{base}（序号按导入顺序自动编号；"
                "输出文件夹中已有同名格式文件时自动接着编号，不会覆盖）。")
            return
        start = next_export_start(out_dir)
        existing = find_existing_export_numbers(out_dir)
        self.label_naming.setText(
            f"导出文件名自动命名：{base}（序号按导入顺序自动编号）。\n"
            f"当前输出文件夹已有 {len(existing)} 个该格式文件，"
            f"本次将从 {EXPORT_BASE_NAME}_{start:03d}{EXPORT_MODALITY_SUFFIX}.nrrd 开始编号。")

    def _build_task(self, idx: int, number: int) -> ExportTask:
        entry = self.patients[idx]
        series = entry["series_list"][0]      # 只导出第一期
        prefix = f"{EXPORT_BASE_NAME}_{number:03d}{EXPORT_MODALITY_SUFFIX}"
        use_range = self.check_use_range.isChecked()
        if entry["range"]:
            s, e = entry["range"]
            z_start, z_end = s - 1, e - 1
        else:
            z_start, z_end = 0, 0
            use_range = False
        return ExportTask(series, prefix, self.line_out_dir.text(), use_range, z_start, z_end)

    def export_current(self):
        if self.current < 0:
            QMessageBox.information(self, "提示", "请先添加并选择病人。")
            return
        number = next_export_start(self.line_out_dir.text().strip())
        folders = [self.patients[self.current]["folder"]]
        self._start_export([self._build_task(self.current, number)], folders)

    def export_all(self):
        if not self.patients:
            QMessageBox.information(self, "提示", "请先添加病人。")
            return
        start = next_export_start(self.line_out_dir.text().strip())
        tasks = [self._build_task(i, start + i) for i in range(len(self.patients))]
        folders = [p["folder"] for p in self.patients]
        self._start_export(tasks, folders)

    def _start_export(self, tasks, remove_folders=()):
        out_dir = self.line_out_dir.text().strip()
        if not out_dir:
            QMessageBox.information(self, "提示", "请先选择输出文件夹。")
            return
        self._pending_removals = set(remove_folders)
        self.btn_export_one.setEnabled(False)
        self.btn_export_all.setEnabled(False)
        self.export_progress.setVisible(True)
        self.export_progress.setRange(0, len(tasks))
        self.export_progress.setValue(0)
        dlg = ProgressDialog("正在导出", "正在导出 NRRD 文件…", self)
        self._export_dialog = dlg
        worker = NRRDExportWorker(tasks)
        worker.progress.connect(lambda i, n, msg: (self.export_progress.setValue(i),
                                                   self.label_export_state.setText(msg),
                                                   dlg.set_progress(i, n),
                                                   dlg.set_message(f"正在导出：{msg}")))
        worker.failed.connect(lambda msg: (dlg.accept(), self._on_export_failed(msg)))
        worker.done.connect(lambda count: (dlg.accept(), self._on_export_done(count)))
        worker.start()
        self._export_worker = worker

    def _on_export_failed(self, msg: str):
        self.btn_export_one.setEnabled(True)
        self.btn_export_all.setEnabled(True)
        QMessageBox.warning(self, "导出失败", msg)

    def _on_export_done(self, count: int):
        self.btn_export_one.setEnabled(True)
        self.btn_export_all.setEnabled(True)
        self._update_naming_hint()
        removed = 0
        if self.check_remove_after_export.isChecked() and self._pending_removals:
            removed = self.remove_patients_by_folders(self._pending_removals)
        self._pending_removals = set()
        self.label_export_state.setText(f"导出完成，共 {count} 个文件 → {self.line_out_dir.text()}")
        msg = f"已成功导出 {count} 个 NRRD 文件。"
        if removed:
            msg += f"\n已从列表移除 {removed} 例。"
        QMessageBox.information(self, "导出完成", msg)
