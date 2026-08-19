# -*- coding: utf-8 -*-
"""可交互的医学图像视图控件。

- 按物理体素间距做纵横比校正显示 2D 灰度图（轴位/冠状位/矢状位共用）。
  厚层图像重建冠状位/矢状位时，z 方向按层厚拉伸，不再“扁”。
- 叠加十字定位线（三视图联动）
- 支持在冠状位/矢状位上绘制起始/结束范围横线
- 滚轮事件做合并节流：快速滚动时合并为一次跳转，避免逐层重建导致的闪烁
- 左键点击定位、右键弹出上下文菜单
"""
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

import numpy as np

CROSSHAIR_COLOR = QColor(46, 204, 113)          # 绿色十字线
RANGE_START_COLOR = QColor(46, 140, 240)        # 蓝色起始线
RANGE_END_COLOR = QColor(230, 80, 80)           # 红色结束线

WHEEL_DEBOUNCE_MS = 40                           # 滚轮合并窗口（毫秒）


def numpy_to_qimage(arr2d: np.ndarray) -> QImage:
    """uint8 二维数组 -> QImage（灰度）。"""
    arr = np.ascontiguousarray(arr2d)
    h, w = arr.shape
    img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    return img.copy()


class MedicalImageView(QWidget):
    """带十字线和范围线的医学图像视图。"""

    clicked = Signal(int, int)               # 左键点击 -> (图像列 col, 图像行 row)
    hovered = Signal(int, int)               # 鼠标移动 -> (col, row)
    scrolled = Signal(int)                   # 滚轮（已合并）-> 步数
    context_menu = Signal(int, int)          # 右键 -> (col, row)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._crosshair = None               # (col, row)
        self._range_rows = []                # [(row, QColor, label)]
        self._pixel_size = None              # (mm/列, mm/行)；None 时按正方形像素显示
        self._ox = 0.0
        self._oy = 0.0
        self._scale_x = 1.0                  # 每个图像列对应的屏幕像素
        self._scale_y = 1.0                  # 每个图像行对应的屏幕像素
        self._wheel_accum = 0
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.timeout.connect(self._flush_wheel)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ---------- 数据接口 ----------
    def set_pixel_size(self, mm_per_col: float, mm_per_row: float):
        """设置该视图图像每个像素的物理尺寸（mm），用于纵横比校正。"""
        if mm_per_col > 0 and mm_per_row > 0:
            self._pixel_size = (mm_per_col, mm_per_row)
        else:
            self._pixel_size = None
        self.update()

    def set_content(self, image, crosshair, range_rows):
        """一次性设置图像、十字线、范围线，只触发一次重绘（减少闪烁）。"""
        self._image = image
        self._crosshair = crosshair
        self._range_rows = list(range_rows) if range_rows else []
        self.update()

    def set_image(self, qimage: QImage):
        self._image = qimage
        self.update()

    def set_crosshair(self, col: int, row: int, visible: bool = True):
        self._crosshair = (col, row) if visible else None
        self.update()

    def set_range_rows(self, rows):
        """rows: [(row, QColor, label), ...]"""
        self._range_rows = list(rows)
        self.update()

    def clear_image(self):
        self._image = None
        self._crosshair = None
        self._range_rows = []
        self.update()

    # ---------- 绘制 ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#0E1116"))

        if self._image is None:
            painter.setPen(QColor("#3A4150"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "未加载图像")
            return

        img_w, img_h = self._image.width(), self._image.height()
        if self._pixel_size is not None:
            px_w, px_h = self._pixel_size          # mm per 列 / mm per 行
        else:
            px_w = px_h = 1.0
        # 统一缩放系数：同时适配宽度与高度（物理尺寸）
        zoom = min(self.width() / (img_w * px_w), self.height() / (img_h * px_h))
        self._scale_x = px_w * zoom                # 每列像素 -> 屏幕像素
        self._scale_y = px_h * zoom                # 每行像素 -> 屏幕像素
        self._ox = (self.width() - img_w * self._scale_x) / 2.0
        self._oy = (self.height() - img_h * self._scale_y) / 2.0
        target = QRectF(self._ox, self._oy, img_w * self._scale_x, img_h * self._scale_y)
        painter.drawImage(target, self._image)

        # 范围线（先画）
        for row, color, label in self._range_rows:
            y = self._oy + (row + 0.5) * self._scale_y
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(int(self._ox), int(y),
                             int(self._ox + img_w * self._scale_x), int(y))
            if label:
                painter.setPen(color)
                painter.drawText(int(self._ox + 6), int(y - 4), label)

        # 十字线
        if self._crosshair is not None:
            cx, cy = self._crosshair
            px = self._ox + (cx + 0.5) * self._scale_x
            py = self._oy + (cy + 0.5) * self._scale_y
            pen = QPen(CROSSHAIR_COLOR, 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(self._ox), int(py),
                             int(self._ox + img_w * self._scale_x), int(py))
            painter.drawLine(int(px), int(self._oy),
                             int(px), int(self._oy + img_h * self._scale_y))

        painter.setPen(QColor("#5B6472"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    # ---------- 事件 ----------
    def _image_pos(self, pos) -> tuple:
        col = int((pos.x() - self._ox) / self._scale_x)
        row = int((pos.y() - self._oy) / self._scale_y)
        if self._image is not None:
            col = max(0, min(col, self._image.width() - 1))
            row = max(0, min(row, self._image.height() - 1))
        return col, row

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._wheel_accum += (1 if delta > 0 else -1)
        self._wheel_timer.start(WHEEL_DEBOUNCE_MS)

    def _flush_wheel(self):
        steps = self._wheel_accum
        self._wheel_accum = 0
        if steps:
            self.scrolled.emit(steps)

    def mousePressEvent(self, event):
        if self._image is None:
            return
        col, row = self._image_pos(event.position())
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(col, row)
        elif event.button() == Qt.MouseButton.RightButton:
            self.context_menu.emit(col, row)

    def mouseMoveEvent(self, event):
        if self._image is None:
            return
        col, row = self._image_pos(event.position())
        self.hovered.emit(col, row)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._crosshair is not None:
            self._crosshair = None
            self.update()
