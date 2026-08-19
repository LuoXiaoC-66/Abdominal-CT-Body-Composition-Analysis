# -*- coding: utf-8 -*-
"""主菜单窗口：三个流程入口卡片。"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget,
)

from app.constants import APP_NAME, APP_VERSION, MODELS_DIR
from app.style import make_button
from ui.calculate_window import CalculateWindow
from ui.convert_window import ConvertWindow
from ui.segment_window import SegmentWindow

FLOWS = [
    ("①", "图像转换与定位",
     "导入病人 DICOM → 轴位/冠状位/矢状位浏览，设置导出层面范围（如隔顶至耻骨联合下缘）→ 转换并保存为 NRRD。"),
    ("②", "自动分割",
     "选择转换后的 NRRD 文件夹与结果输出文件夹，使用 nnUNet V2 模型一键批量分割所有病例。"),
    ("③", "数据计算",
     "选择病例图像与分割结果文件夹，自动计算每个分割区域的体积与平均 CT 值，并按区域导出 Excel。"),
]


class FlowCard(QFrame):
    """可点击的流程卡片。"""
    clicked = Signal(int)

    def __init__(self, index: int, number: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("flowCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(150)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(18)

        badge = QLabel(number)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(52, 52)
        badge.setObjectName("badge")
        lay.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(6)
        t = QLabel(title)
        t.setObjectName("flowTitle")
        text_col.addWidget(t)
        d = QLabel(description)
        d.setObjectName("flowDesc")
        d.setWordWrap(True)
        text_col.addWidget(d)
        text_col.addStretch(1)
        lay.addLayout(text_col, 1)

        arrow = QLabel("›")
        arrow.setObjectName("arrow")
        lay.addWidget(arrow, 0, Qt.AlignmentFlag.AlignVCenter)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit(self.index)
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    """主菜单窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(860, 640)
        self._windows = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(40, 36, 40, 30)
        root.setSpacing(18)

        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("腹部 CT 身体成分自动分割 · 三个流程 · 面向非计算机专业用户")
        subtitle.setObjectName("appSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        root.addSpacing(10)

        for i, (num, name, desc) in enumerate(FLOWS):
            card = FlowCard(i, num, name, desc)
            card.clicked.connect(self._open_flow)
            root.addWidget(card)

        root.addStretch(1)

        footer = QHBoxLayout()
        hint = QLabel(f"模型文件夹：{MODELS_DIR}")
        hint.setObjectName("footer")
        btn_open = make_button("打开模型文件夹")
        btn_open.clicked.connect(self._open_models_dir)
        footer.addWidget(hint)
        footer.addStretch(1)
        footer.addWidget(btn_open)
        root.addLayout(footer)

        self._apply_card_style()

    def _apply_card_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #F2F5FA; }
            QLabel#appTitle { font-size: 30px; font-weight: 800; color: #111827; }
            QLabel#appSubtitle { font-size: 13px; color: #6B7280; }
            QLabel#footer { color: #9CA3AF; font-size: 12px; }
            QFrame#flowCard {
                background: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 14px;
            }
            QFrame#flowCard:hover { border: 2px solid #2D8CF0; background: #F7FAFF; }
            QLabel#badge {
                background: #2D8CF0; color: #FFFFFF; border-radius: 26px;
                font-size: 20px; font-weight: 700;
            }
            QLabel#flowTitle { font-size: 17px; font-weight: 700; color: #111827; }
            QLabel#flowDesc { color: #5B6472; font-size: 13px; }
            QLabel#arrow { color: #2D8CF0; font-size: 30px; font-weight: 700; }
        """)

    def _open_flow(self, index: int):
        if index in self._windows and self._windows[index] is not None:
            win = self._windows[index]
            win.show()
            win.raise_()
            win.activateWindow()
            return
        if index == 0:
            win = ConvertWindow()
        elif index == 1:
            win = SegmentWindow()
        else:
            win = CalculateWindow()
        self._windows[index] = win
        win.show()

    def _open_models_dir(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import os
        os.startfile(str(MODELS_DIR))

    def closeEvent(self, event):
        for win in self._windows.values():
            if win is not None:
                win.close()
        super().closeEvent(event)
