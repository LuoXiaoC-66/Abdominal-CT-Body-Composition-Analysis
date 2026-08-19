# -*- coding: utf-8 -*-
"""界面三：数据计算。

- 选择病例图像文件夹（NRRD）与分割结果文件夹（NRRD）
- 自动匹配病例，计算每个分割区域的体积与平均 CT 值等指标
- 表格预览，并按分割区域导出 Excel（每个区域一个工作表 + 汇总）
"""
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from app import models_config
from app.measurement import export_excel, match_files
from app.models_config import labels_to_dict
from app.style import make_button
from ui.model_manager_dialog import ModelManagerDialog
from ui.progress_dialog import ProgressDialog


class CalculateWorker(QThread):
    """后台计算线程。"""
    progress = Signal(int, int, str)
    result = Signal(object)     # pd.DataFrame 预览长表
    done = Signal()
    failed = Signal(str)

    def __init__(self, pairs, label_names, parent=None):
        super().__init__(parent)
        self.pairs = pairs
        self.label_names = label_names

    def run(self):
        try:
            rows = []
            total = len(self.pairs)
            for i, (img, seg, name) in enumerate(self.pairs, start=1):
                self.progress.emit(i, total, f"正在计算 {name} ({i}/{total})")
                from app.measurement import compute_metrics
                metrics = compute_metrics(img, seg, self.label_names)
                row = {"病例": name}
                for m in metrics:
                    row[f"{m.name}_体积(cm³)"] = m.volume_cm3
                    row[f"{m.name}_平均CT(HU)"] = m.mean_hu
                rows.append(row)
            self.result.emit(pd.DataFrame(rows))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportExcelWorker(QThread):
    """后台导出 Excel。"""
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, pairs, label_names, output_path, parent=None):
        super().__init__(parent)
        self.pairs = pairs
        self.label_names = label_names
        self.output_path = output_path

    def run(self):
        try:
            export_excel(self.pairs, self.label_names, self.output_path)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return
        self.done.emit(self.output_path)


class CalculateWindow(QMainWindow):
    """数据计算窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("③ 数据计算")
        self.resize(1000, 720)
        self.models = models_config.load_models()
        self.pairs = []
        self.label_names = {}
        self.calc_worker = None
        self.export_worker = None

        self._build_ui()
        self._refresh_model_combo()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("③ 数据计算")
        title.setProperty("class", "pageTitle")
        subtitle = QLabel("选择病例图像与分割结果文件夹 → 自动计算各分割区域体积与平均CT值 → 导出 Excel")
        subtitle.setProperty("class", "pageSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # ---- 路径 ----
        box_path = QGroupBox("路径设置")
        pb = QVBoxLayout(box_path)
        row_img = QHBoxLayout()
        row_img.addWidget(QLabel("图像文件夹"))
        self.line_image_dir = QLineEdit()
        self.line_image_dir.setReadOnly(True)
        self.line_image_dir.setPlaceholderText("界面①转换好的病例 NRRD 文件夹")
        row_img.addWidget(self.line_image_dir, 1)
        btn_img = make_button("选择…")
        btn_img.clicked.connect(lambda: self._choose_dir(self.line_image_dir, "选择病例图像文件夹"))
        row_img.addWidget(btn_img)
        pb.addLayout(row_img)

        row_seg = QHBoxLayout()
        row_seg.addWidget(QLabel("分割文件夹"))
        self.line_seg_dir = QLineEdit()
        self.line_seg_dir.setReadOnly(True)
        self.line_seg_dir.setPlaceholderText("界面②预测结果 NRRD 文件夹")
        row_seg.addWidget(self.line_seg_dir, 1)
        btn_seg = make_button("选择…")
        btn_seg.clicked.connect(lambda: self._choose_dir(self.line_seg_dir, "选择分割结果文件夹"))
        row_seg.addWidget(btn_seg)
        pb.addLayout(row_seg)

        row_suffix = QHBoxLayout()
        row_suffix.addWidget(QLabel("分割文件后缀"))
        self.line_suffix = QLineEdit()
        self.line_suffix.setPlaceholderText("例如 _seg（图像 case01.nrrd 对应 case01_seg.nrrd）；留空表示同名")
        row_suffix.addWidget(self.line_suffix, 1)
        pb.addLayout(row_suffix)
        root.addWidget(box_path)

        # ---- 区域标签 ----
        box_label = QGroupBox("分割区域标签")
        lb = QVBoxLayout(box_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("按模型载入"))
        self.combo_model = QComboBox()
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(self.combo_model, 1)
        btn_manage = make_button("模型管理…")
        btn_manage.clicked.connect(self._open_model_manager)
        row.addWidget(btn_manage)
        btn_load = make_button("载入所选模型标签")
        btn_load.clicked.connect(self._load_labels_from_model)
        row.addWidget(btn_load)
        lb.addLayout(row)

        self.label_label_info = QLabel("未设置区域名称，将按 区域1/区域2… 命名。")
        self.label_label_info.setProperty("class", "hint")
        self.label_label_info.setWordWrap(True)
        lb.addWidget(self.label_label_info)
        root.addWidget(box_label)

        # ---- 操作 ----
        act = QHBoxLayout()
        self.btn_calc = make_button("开始计算", "primary")
        self.btn_calc.clicked.connect(self.start_calculate)
        self.btn_export = make_button("导出 Excel", "primary")
        self.btn_export.clicked.connect(self.export_result)
        self.btn_export.setEnabled(False)
        act.addWidget(self.btn_calc)
        act.addWidget(self.btn_export)
        act.addStretch(1)
        root.addLayout(act)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.label_state = QLabel("")
        self.label_state.setProperty("class", "hint")
        root.addWidget(self.label_state)

        # ---- 结果预览 ----
        self.table_result = QTableWidget(0, 0)
        self.table_result.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        root.addWidget(self.table_result, 1)

    # ------------------------------------------------------------- 逻辑
    def _refresh_model_combo(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self.combo_model.addItem("（不使用模型标签）")
        for m in self.models:
            self.combo_model.addItem(m.get("name", "未命名模型"))
        self.combo_model.blockSignals(False)

    def _open_model_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.models_changed.connect(self._on_models_changed)
        dlg.exec()

    def _on_models_changed(self):
        self.models = models_config.load_models()
        self._refresh_model_combo()

    def _on_model_changed(self):
        self.label_label_info.setText("选择模型后点击“载入所选模型标签”以使用其区域名称。")

    def _load_labels_from_model(self):
        idx = self.combo_model.currentIndex() - 1
        if idx < 0 or idx >= len(self.models):
            QMessageBox.information(self, "提示", "请先选择模型。")
            return
        self.label_names = labels_to_dict(self.models[idx])
        text = "、".join(f"{k}:{v}" for k, v in sorted(self.label_names.items()))
        self.label_label_info.setText(f"已载入区域标签：{text}")

    def _choose_dir(self, line_edit: QLineEdit, caption: str):
        folder = QFileDialog.getExistingDirectory(self, caption)
        if folder:
            line_edit.setText(folder)

    def start_calculate(self):
        image_dir = self.line_image_dir.text().strip()
        seg_dir = self.line_seg_dir.text().strip()
        if not image_dir or not Path(image_dir).exists():
            QMessageBox.warning(self, "提示", "请选择有效的图像文件夹。")
            return
        if not seg_dir or not Path(seg_dir).exists():
            QMessageBox.warning(self, "提示", "请选择有效的分割结果文件夹。")
            return
        suffix = self.line_suffix.text().strip()
        pairs = match_files(image_dir, seg_dir, suffix)
        if not pairs:
            QMessageBox.warning(self, "提示", "未找到匹配的图像与分割文件，请检查文件夹和后缀设置。")
            return
        self.pairs = pairs
        self.progress.setVisible(True)
        self.progress.setRange(0, len(pairs))
        self.progress.setValue(0)
        self.btn_calc.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.label_state.setText(f"共匹配到 {len(pairs)} 个病例，开始计算…")

        self.calc_worker = CalculateWorker(pairs, self.label_names)
        dlg = ProgressDialog("正在计算", "正在计算各区域体积与平均 CT 值…", self)
        self._calc_dialog = dlg
        self.calc_worker.progress.connect(lambda i, n, msg: (self.progress.setValue(i),
                                                             self.label_state.setText(msg),
                                                             dlg.set_progress(i, n),
                                                             dlg.set_message(msg)))
        self.calc_worker.result.connect(lambda df: (dlg.accept(), self._on_result(df)))
        self.calc_worker.failed.connect(lambda msg: (dlg.accept(), self._on_calc_failed(msg)))
        self.calc_worker.start()

    def _on_result(self, df: pd.DataFrame):
        self.progress.setVisible(False)
        self.btn_calc.setEnabled(True)
        self.label_state.setText(f"计算完成：{len(df)} 个病例。可点击“导出 Excel”。")
        self._fill_table(df)
        self.btn_export.setEnabled(len(df) > 0)

    def _on_calc_failed(self, msg: str):
        self.progress.setVisible(False)
        self.btn_calc.setEnabled(True)
        self.label_state.setText("计算失败。")
        QMessageBox.critical(self, "计算失败", msg)

    def _fill_table(self, df: pd.DataFrame):
        self.table_result.clear()
        self.table_result.setColumnCount(len(df.columns))
        self.table_result.setRowCount(len(df))
        self.table_result.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r, row in df.iterrows():
            for c, val in enumerate(row):
                item = QTableWidgetItem("" if pd.isna(val) else f"{val:g}" if isinstance(val, float) else str(val))
                self.table_result.setItem(int(r), c, item)
        self.table_result.resizeColumnsToContents()

    def export_result(self):
        if not self.pairs:
            QMessageBox.information(self, "提示", "请先执行计算。")
            return
        default_name = "身体成分计算结果.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "保存 Excel 结果", default_name, "Excel 文件 (*.xlsx)")
        if not path:
            return
        self.btn_export.setEnabled(False)
        self.label_state.setText("正在导出 Excel …")
        dlg = ProgressDialog("正在导出", "正在导出 Excel 文件…", self)
        self._export_dialog = dlg
        self.export_worker = ExportExcelWorker(self.pairs, self.label_names, path)
        self.export_worker.done.connect(lambda p: (dlg.accept(), self._on_export_done(p)))
        self.export_worker.failed.connect(lambda msg: (dlg.accept(), self._on_export_failed(msg)))
        self.export_worker.start()

    def _on_export_done(self, path: str):
        self.btn_export.setEnabled(True)
        self.label_state.setText(f"已导出：{path}")
        QMessageBox.information(self, "导出成功", f"Excel 已保存到：\n{path}")

    def _on_export_failed(self, msg: str):
        self.btn_export.setEnabled(True)
        self.label_state.setText("导出失败。")
        QMessageBox.critical(self, "导出失败", msg)
