# -*- coding: utf-8 -*-
"""模态进度对话框。

- 应用级模态：显示期间无法操作软件其他窗口
- 显示当前进行的操作说明 + 进度条（支持确定/不确定进度）
- 可选“停止”按钮（用于分割等长耗时任务）
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from app.style import make_button


class ProgressDialog(QDialog):
    """阻塞其他窗口操作的模态进度框。"""

    cancel_requested = Signal()

    def __init__(self, title: str, message: str, parent=None, cancellable: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(400)
        self._cancelled = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(14)

        self.label_message = QLabel(message)
        self.label_message.setWordWrap(True)
        lay.addWidget(self.label_message)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)          # 默认不确定进度
        lay.addWidget(self.progress)

        if cancellable:
            row = QHBoxLayout()
            row.addStretch(1)
            self.btn_cancel = make_button("停止", "danger")
            self.btn_cancel.clicked.connect(self._on_cancel_clicked)
            row.addWidget(self.btn_cancel)
            lay.addLayout(row)

        self.show()
        self.raise_()
        self.activateWindow()

    # ---------------- 对外接口 ----------------
    def set_message(self, text: str):
        """更新操作说明文字。"""
        self.label_message.setText(text)

    def set_progress(self, value: int, maximum: int = None):
        """更新确定进度。"""
        if maximum is not None:
            self.progress.setRange(0, maximum)
        self.progress.setValue(value)

    def set_indeterminate(self):
        """切回不确定进度（滚动条）。"""
        self.progress.setRange(0, 0)

    # ---------------- 内部 ----------------
    def _on_cancel_clicked(self):
        if self._cancelled:
            return
        self._cancelled = True
        self.cancel_requested.emit()
        self.label_message.setText("正在停止…")
        self.btn_cancel.setEnabled(False)
