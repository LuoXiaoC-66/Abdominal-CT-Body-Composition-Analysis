# -*- coding: utf-8 -*-
"""渲染三个界面的预览截图（用于开发/检查外观）。

运行：python tests/render_preview.py
输出：preview/ 目录下的 PNG
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "preview"


def main():
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    from app.style import apply_style
    apply_style(app)
    for name in ("information", "warning", "critical"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    OUT_DIR.mkdir(exist_ok=True)

    # 生成示例数据
    sys.path.insert(0, str(ROOT / "tools"))
    import make_sample_dicom as gen
    gen.main()
    dicom_folder = gen.DICOM_DIR / "病人001"

    from ui.main_window import MainWindow
    mw = MainWindow()
    mw.show()
    app.processEvents()
    mw.grab().save(str(OUT_DIR / "0_主菜单.png"))
    print("saved 主菜单")

    from ui.convert_window import ConvertWindow
    cw = ConvertWindow()
    cw.show()
    cw.resize(1280, 820)
    cw._add_folders([dicom_folder])
    deadline = time.time() + 20
    while cw.arr is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    cw.set_pos(48, 48, 24)
    cw.spin_range_start.setValue(10)
    cw.spin_range_end.setValue(38)
    app.processEvents()
    cw.grab().save(str(OUT_DIR / "1_图像转换与定位.png"))
    print("saved 图像转换与定位")

    from ui.segment_window import SegmentWindow
    sw = SegmentWindow()
    sw.show()
    sw.resize(900, 680)
    app.processEvents()
    sw.grab().save(str(OUT_DIR / "2_自动分割.png"))
    print("saved 自动分割")

    from ui.calculate_window import CalculateWindow
    cw2 = CalculateWindow()
    cw2.show()
    cw2.resize(1000, 720)
    cw2.line_image_dir.setText(str(gen.IMAGE_DIR))
    cw2.line_seg_dir.setText(str(gen.SEG_DIR))
    cw2.start_calculate()
    deadline = time.time() + 20
    while cw2.table_result.rowCount() < 1 and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()
    cw2.grab().save(str(OUT_DIR / "3_数据计算.png"))
    print("saved 数据计算")

    from ui.model_manager_dialog import ModelManagerDialog
    dlg = ModelManagerDialog()
    dlg.show()
    app.processEvents()
    dlg.grab().save(str(OUT_DIR / "4_模型管理.png"))
    print("saved 模型管理")

    print("done ->", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
