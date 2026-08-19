# -*- coding: utf-8 -*-
"""CT身体成分分析系统 - 启动入口。

运行：python main.py
"""
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.constants import APP_NAME
from app.style import apply_style
from ui.main_window import MainWindow


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("AutoSegment")
    apply_style(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
