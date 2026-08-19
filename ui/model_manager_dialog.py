# -*- coding: utf-8 -*-
"""模型管理对话框：增删改 nnUNet 模型配置，并写入 models.json。"""
import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QMessageBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from app import models_config
from app.constants import MODELS_DIR
from app.models_config import dict_to_labels, labels_to_dict, new_model_entry
from app.style import make_button


class ModelManagerDialog(QDialog):
    """管理 models.json 中的模型配置。"""

    models_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.resize(860, 560)
        self.models = models_config.load_models()

        self._build_ui()
        self._refresh_list()
        if self.list_models.count():
            self.list_models.setCurrentRow(0)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # 左：模型列表
        left = QGroupBox("模型列表")
        lv = QVBoxLayout(left)
        self.list_models = QListWidget()
        self.list_models.currentRowChanged.connect(self._on_select)
        lv.addWidget(self.list_models, 1)
        lrow = QHBoxLayout()
        btn_add = make_button("新增")
        btn_add.clicked.connect(self._add_model)
        btn_del = make_button("删除", "danger")
        btn_del.clicked.connect(self._delete_model)
        lrow.addWidget(btn_add)
        lrow.addWidget(btn_del)
        lv.addLayout(lrow)

        hint = QLabel(f"模型文件夹默认放在：{MODELS_DIR}\n"
                      "每个模型文件夹需包含 plans.json 与 fold_*/checkpoint_final.pth")
        hint.setProperty("class", "hint")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        btn_open = make_button("打开模型目录")
        btn_open.clicked.connect(self._open_models_dir)
        lv.addWidget(btn_open)
        root.addWidget(left, 2)

        # 右：编辑区
        right = QGroupBox("模型设置")
        rv = QVBoxLayout(right)

        form = QHBoxLayout()
        form.addWidget(QLabel("名称"))
        self.edit_name = QLineEdit()
        form.addWidget(self.edit_name, 1)
        form.addWidget(QLabel("推理设备"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "cpu"])
        form.addWidget(self.combo_device)
        rv.addLayout(form)

        folder = QHBoxLayout()
        folder.addWidget(QLabel("模型文件夹"))
        self.edit_folder = QLineEdit()
        self.edit_folder.setReadOnly(True)
        folder.addWidget(self.edit_folder, 1)
        btn_browse = make_button("选择…")
        btn_browse.clicked.connect(self._browse_folder)
        folder.addWidget(btn_browse)
        rv.addLayout(folder)

        ck = QHBoxLayout()
        ck.addWidget(QLabel("检查点文件"))
        self.edit_checkpoint = QLineEdit("checkpoint_final.pth")
        ck.addWidget(self.edit_checkpoint, 1)
        ck.addWidget(QLabel("folds"))
        self.edit_folds = QLineEdit("all")
        self.edit_folds.setToolTip("多个 fold 用逗号分隔，例如 0,1,2,3,4；默认 all")
        ck.addWidget(self.edit_folds, 1)
        rv.addLayout(ck)

        rv.addWidget(QLabel("分割区域标签（标签值 → 区域名称）"))
        self.table_labels = QTableWidget(0, 2)
        self.table_labels.setHorizontalHeaderLabels(["标签值", "区域名称"])
        self.table_labels.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_labels.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        rv.addWidget(self.table_labels, 1)

        lrow = QHBoxLayout()
        btn_row_add = make_button("添加标签")
        btn_row_add.clicked.connect(lambda: self._add_label_row())
        btn_row_del = make_button("删除标签")
        btn_row_del.clicked.connect(self._remove_label_row)
        lrow.addWidget(btn_row_add)
        lrow.addWidget(btn_row_del)
        lrow.addStretch(1)
        rv.addLayout(lrow)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = make_button("取消")
        btn_cancel.clicked.connect(self.reject)
        btn_save = make_button("保存", "primary")
        btn_save.clicked.connect(self._save)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        rv.addLayout(btn_row)
        root.addWidget(right, 3)

    # ------------------------------------------------------------- 逻辑
    def _refresh_list(self):
        self.list_models.clear()
        for m in self.models:
            self.list_models.addItem(m.get("name", "未命名模型"))

    def _current_model(self):
        row = self.list_models.currentRow()
        if 0 <= row < len(self.models):
            return self.models[row]
        return None

    def _on_select(self, row: int):
        if row < 0 or row >= len(self.models):
            return
        m = self.models[row]
        self.edit_name.setText(m.get("name", ""))
        self.edit_folder.setText(m.get("model_folder", ""))
        self.edit_checkpoint.setText(m.get("checkpoint", "checkpoint_final.pth"))
        self.edit_folds.setText(",".join(str(f) for f in m.get("folds", ["all"])))
        self.combo_device.setCurrentText(m.get("device", "auto"))
        self.table_labels.setRowCount(0)
        for label, name in sorted(labels_to_dict(m).items()):
            self._add_label_row(str(label), name)

    def _add_label_row(self, label: str = "", name: str = ""):
        row = self.table_labels.rowCount()
        self.table_labels.insertRow(row)
        self.table_labels.setItem(row, 0, QTableWidgetItem(label))
        self.table_labels.setItem(row, 1, QTableWidgetItem(name))

    def _remove_label_row(self):
        row = self.table_labels.currentRow()
        if row >= 0:
            self.table_labels.removeRow(row)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 nnUNet 模型文件夹")
        if folder:
            self.edit_folder.setText(self._resolve_model_folder(folder))

    def _resolve_model_folder(self, folder) -> str:
        """自动定位到包含 plans.json 的模型配置文件夹。

        用户可能选到 数据集文件夹 或更上层，这里自动向下找到含 plans.json 的层。
        """
        folder = Path(folder)
        if (folder / "plans.json").exists():
            return str(folder)
        candidates = sorted(folder.rglob("plans.json"))
        if candidates:
            return str(candidates[0].parent)
        return str(folder)

    def _open_models_dir(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(MODELS_DIR))

    def _add_model(self):
        self.models.append(new_model_entry(f"新模型 {len(self.models) + 1}"))
        self._refresh_list()
        self.list_models.setCurrentRow(len(self.models) - 1)

    def _delete_model(self):
        row = self.list_models.currentRow()
        if row < 0:
            return
        name = self.models[row].get("name", "")
        if QMessageBox.question(self, "删除模型", f"确定删除模型“{name}”？") != QMessageBox.StandardButton.Yes:
            return
        del self.models[row]
        self._refresh_list()
        if self.list_models.count():
            self.list_models.setCurrentRow(0)

    def _collect_form(self) -> bool:
        """把表单内容写回当前模型。返回是否成功。"""
        m = self._current_model()
        if m is None:
            return False
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请填写模型名称。")
            return False
        m["name"] = name
        m["model_folder"] = self.edit_folder.text().strip()
        m["checkpoint"] = self.edit_checkpoint.text().strip() or "checkpoint_final.pth"
        folds = [f.strip() for f in self.edit_folds.text().replace("，", ",").split(",") if f.strip()]
        m["folds"] = folds or ["all"]
        m["device"] = self.combo_device.currentText()

        labels = {}
        for row in range(self.table_labels.rowCount()):
            item0 = self.table_labels.item(row, 0)
            item1 = self.table_labels.item(row, 1)
            if item0 is None or not item0.text().strip():
                continue
            try:
                label = int(item0.text().strip())
            except ValueError:
                QMessageBox.warning(self, "提示", f"第 {row + 1} 行标签值必须是整数。")
                return False
            labels[label] = (item1.text().strip() if item1 else "") or f"区域{label}"
        if not labels:
            QMessageBox.warning(self, "提示", "请至少填写一个分割区域标签。")
            return False
        m["labels"] = dict_to_labels(labels)
        return True

    def _save(self):
        if not self._collect_form():
            return
        models_config.save_models(self.models)
        self.models_changed.emit()
        QMessageBox.information(self, "保存成功", "模型配置已保存到 models.json。")
        self.accept()
