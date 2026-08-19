# -*- coding: utf-8 -*-
"""无界面冒烟测试：验证核心逻辑与三个窗口的构建。

运行：python tests/smoke_test.py
（会临时生成一套模拟 DICOM 数据在系统临时目录，测试后自动清理）
"""
import json as _json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def wait_until(fn, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        time.sleep(0.05)
    return False


def run_export(cw, timeout=30.0):
    """等待窗口的导出线程完成，并让 queued 的 done 信号被处理。"""
    w = cw._export_worker
    if w is None:
        return False
    ok = wait_until(lambda: not w.isRunning(), timeout=timeout)
    from PySide6.QtWidgets import QApplication
    for _ in range(20):                    # 排空 queued 信号（done -> _on_export_done）
        QApplication.processEvents()
        time.sleep(0.01)
    return ok


def write_multi_phase(vol, out_dir, n_phases=3, slice_thickness=5.0):
    """把同一个体数据写成单个序列 UID 下的多个期相（z 每期相从 0 重新开始）。"""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian

    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    rows, cols = vol.shape[1], vol.shape[2]
    stored = np.clip(vol + 1024, 0, 65535).astype(np.uint16)
    n = vol.shape[0]
    idx = 0
    for _phase in range(n_phases):
        for i in range(n):
            idx += 1
            fm = FileMetaDataset()
            fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            fm.MediaStorageSOPInstanceUID = generate_uid()
            fm.TransferSyntaxUID = ExplicitVRLittleEndian
            fm.ImplementationClassUID = generate_uid()
            ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
            ds.SOPClassUID = fm.MediaStorageSOPClassUID
            ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
            ds.SpecificCharacterSet = "ISO_IR 192"
            ds.PatientName = "多期相病人"
            ds.PatientID = "P900"
            ds.StudyDate = "20240815"
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.Modality = "CT"
            ds.SeriesDescription = "腹部增强多期"
            ds.InstanceNumber = idx
            ds.Rows, ds.Columns = rows, cols
            ds.PixelSpacing = [1.5, 1.5]
            ds.SliceThickness = slice_thickness
            ds.ImagePositionPatient = [0.0, 0.0, i * slice_thickness]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.RescaleIntercept = -1024.0
            ds.RescaleSlope = 1.0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = stored[i].tobytes()
            ds.save_as(out_dir / f"slice_{idx:04d}.dcm")
    return series_uid


def main():
    tmp = Path(tempfile.mkdtemp(prefix="autoseg_test_"))
    try:
        # ---------- 0. 生成模拟 DICOM ----------
        print("\n== 0. 生成模拟 DICOM ==")
        sys.path.insert(0, str(ROOT / "tools"))
        import make_sample_dicom as gen
        gen.ROOT = tmp
        gen.DICOM_DIR = tmp / "dicom"
        gen.IMAGE_DIR = tmp / "images"
        gen.SEG_DIR = tmp / "seg"
        gen.main()
        dicom_folder = tmp / "dicom" / "病人001"
        image_nrrd = tmp / "images" / "case001.nrrd"
        seg_nrrd = tmp / "seg" / "case001.nrrd"
        check("模拟 DICOM 已生成", dicom_folder.exists() and len(list(dicom_folder.glob("*.dcm"))) == gen.N_SLICES)

        # ---------- 1. DICOM 扫描与加载 ----------
        print("\n== 1. DICOM 扫描与加载 ==")
        from app.dicom import scan_folder, load_series_image, collect_import_folders
        series_list = scan_folder(dicom_folder)
        check("扫描到 1 个序列", len(series_list) == 1, f"实际 {len(series_list)}")
        s = series_list[0]
        check("序列信息正确", s.modality == "CT" and s.instance_count == gen.N_SLICES,
              f"modality={s.modality}, n={s.instance_count}")
        img = load_series_image(s)
        size = img.GetSize()  # (x, y, z)
        check("体数据尺寸正确", (size[2], size[1], size[0]) == (gen.N_SLICES, gen.SIZE, gen.SIZE),
              f"shape={size}")
        import SimpleITK as sitk
        check("CT 值已转为 HU", sitk.GetArrayFromImage(img).max() > 300, "应为 400（骨骼）")

        # ---------- 1.5 多层嵌套目录导入规则 ----------
        print("\n== 1.5 多层嵌套目录导入规则 ==")
        vol, *_ = gen.build_volume()
        nested_root = tmp / "nested_root"
        p1_series = nested_root / "2024" / "病人A" / "CT" / "series"
        p2_series = nested_root / "2025" / "病人B" / "CT" / "series"
        gen.write_dicom(vol, p1_series)
        gen.write_dicom(vol, p2_series)
        imports = collect_import_folders(nested_root)
        check("多层嵌套导入上一级文件夹",
              set(imports) == {p1_series.parent, p2_series.parent}, f"{imports}")
        check("每个导入文件夹 1 个序列", all(len(scan_folder(f)) == 1 for f in imports))

        flat_root = tmp / "flat_root"
        fa = flat_root / "病人X"
        fb = flat_root / "病人Y"
        gen.write_dicom(vol, fa)
        gen.write_dicom(vol, fb)
        imports2 = collect_import_folders(flat_root)
        check("扁平多病人直接导入子文件夹", set(imports2) == {fa, fb}, f"{imports2}")

        dup_src = list(p1_series.glob("*.dcm"))[0]
        shutil.copy2(dup_src, p1_series / "duplicate.dcm")
        files_after = len(scan_folder(p1_series.parent)[0].files)
        check("重复切片自动去重", files_after == gen.N_SLICES, f"n={files_after}")

        # ---------- 1.6 多期相（单文件夹单序列堆叠）切分 ----------
        print("\n== 1.6 多期相单文件夹切分 ==")
        mp_dir = tmp / "multi_phase"
        write_multi_phase(vol, mp_dir, n_phases=3)
        mp_series = scan_folder(mp_dir)
        check("多期相仅保留第一期", len(mp_series) == 1, f"n={len(mp_series)}")
        check("第一期层面数正确", mp_series[0].instance_count == gen.N_SLICES,
              f"n={mp_series[0].instance_count}")
        mp_img = load_series_image(mp_series[0])
        check("第一期可正常加载", mp_img.GetSize()[2] == gen.N_SLICES)

        # ---------- 2. 导出 NRRD（全部 / 范围） ----------
        print("\n== 2. 导出 NRRD ==")
        from app.nrrd_io import export_subvolume, write_image, read_image
        out_full = tmp / "out" / "case001_full.nrrd"
        write_image(img, out_full)
        out_range = tmp / "out" / "case001_range.nrrd"
        export_subvolume(img, out_range, 10, 19)
        full_img = read_image(out_full)
        range_img = read_image(out_range)
        check("全量导出尺寸正确", full_img.GetSize() == img.GetSize())
        check("范围导出尺寸正确(10层)", range_img.GetSize()[2] == 10, f"z={range_img.GetSize()[2]}")
        check("范围导出 spacing 保留", range_img.GetSpacing() == img.GetSpacing())
        sp = img.GetSpacing()
        origin_expected = list(img.GetOrigin())
        origin_expected[2] += 10 * sp[2]
        check("范围导出 origin 更新", all(abs(a - b) < 1e-3 for a, b in zip(range_img.GetOrigin(), origin_expected)),
              f"origin={range_img.GetOrigin()} expected={origin_expected}")

        # ---------- 3. 数据计算 ----------
        print("\n== 3. 数据计算 ==")
        from app.measurement import compute_metrics, match_files, export_excel
        labels = {1: "内脏脂肪", 2: "皮下脂肪", 3: "骨骼", 4: "骨骼肌"}
        metrics = compute_metrics(image_nrrd, seg_nrrd, labels)
        names = {m.label: m.name for m in metrics}
        check("识别全部 4 个区域", set(names) == {1, 2, 3, 4}, f"labels={names}")
        vol_vis = next(m for m in metrics if m.label == 1)
        vol_sub = next(m for m in metrics if m.label == 2)
        check("内脏脂肪体积 > 0", vol_vis.volume_cm3 > 0, f"{vol_vis.volume_cm3}")
        check("皮下脂肪体积 > 内脏脂肪", vol_sub.volume_cm3 > vol_vis.volume_cm3)
        bone = next(m for m in metrics if m.label == 3)
        check("骨骼平均CT约400HU", abs(bone.mean_hu - 400) < 5, f"{bone.mean_hu}")
        check("皮下脂肪平均CT为负", next(m for m in metrics if m.label == 2).mean_hu < 0)

        pairs = match_files(tmp / "images", tmp / "seg", "")
        check("文件匹配成功", len(pairs) == 1)
        xlsx = tmp / "result.xlsx"
        export_excel(pairs, labels, xlsx)
        import pandas as pd
        sheets = pd.ExcelFile(xlsx).sheet_names
        check("Excel 包含区域工作表", "内脏脂肪" in sheets and "骨骼肌" in sheets, f"sheets={sheets}")
        check("Excel 包含汇总表", "汇总_体积(cm³)" in sheets)

        # ---------- 4. 模型配置 ----------
        print("\n== 4. 模型配置 ==")
        from app.models_config import load_models, save_models, labels_to_dict
        cfg_path = tmp / "models_test.json"
        models = load_models(cfg_path)
        check("默认预置 4 个模型", len(models) == 4, f"n={len(models)}")
        models[0]["model_folder"] = str(tmp / "models")
        save_models(models, cfg_path)
        reloaded = load_models(cfg_path)
        check("配置读写一致", reloaded[0]["model_folder"] == str(tmp / "models"))
        check("标签转换正确", labels_to_dict(reloaded[0])[1] == "内脏脂肪")

        # 相对路径存取（打包可移植）
        from app.models_config import relative_model_folder, resolve_model_folder
        from app.constants import APP_DIR as _APP_DIR
        abs_m = str(_APP_DIR / "models" / "Dataset002_Body" / "nnUNetTrainer__nnUNetPlans__2d")
        rel_m = relative_model_folder(abs_m)
        check("模型路径转相对",
              rel_m.replace("\\", "/") == "models/Dataset002_Body/nnUNetTrainer__nnUNetPlans__2d", rel_m)
        check("相对路径解析回绝对", resolve_model_folder(rel_m) == abs_m, resolve_model_folder(rel_m))

        # ---------- 4.5 nnUNet 环境检测与子进程 ----------
        print("\n== 4.5 nnUNet 环境检测与子进程 ==")
        from app.seg_env import (
            current_python_has_nnunet, find_python_with_nnunet,
            find_base_python, has_nvidia_gpu, driver_cuda_version,
            _python_version, PIP_MIRROR, TORCH_CPU_INDEXES,
        )
        check("pip 镜像源已配置", PIP_MIRROR.startswith("http"), PIP_MIRROR)
        check("torch CPU 源已配置", len(TORCH_CPU_INDEXES) >= 1)
        cuda_ver = driver_cuda_version()
        check("驱动 CUDA 版本可读取", cuda_ver is None or isinstance(cuda_ver, float), f"{cuda_ver}")

        from app.seg_env import (
            _extract_device_mode, _env_has_torch, _env_torch_status,
            _CUDA_VERSION_RE, TORCH_GPU_INDEXES,
        )
        st = _env_torch_status(sys.executable)   # 基础环境无 torch
        check("torch 状态检测（无 torch）", st[0] is False, f"{st}")
        check("GPU torch 使用官方 CUDA 源",
              any("download.pytorch.org/whl/cu" in i for i in TORCH_GPU_INDEXES),
              f"{TORCH_GPU_INDEXES}")
        check("环境 torch 检测函数返回布尔", isinstance(_env_has_torch(sys.executable), bool))
        m_new = _CUDA_VERSION_RE.search("NVIDIA-SMI 610.88 KMD Version: 610.88 CUDA UMD Version: 13.3")
        check("新驱动 CUDA 格式解析",
              m_new is not None and float(f"{m_new.group(1)}.{m_new.group(2)}") == 13.3,
              f"{m_new.groups() if m_new else None}")
        m_old = _CUDA_VERSION_RE.search("Driver Version: 512.74 CUDA Version: 11.6")
        check("旧驱动 CUDA 格式解析",
              m_old is not None and float(f"{m_old.group(1)}.{m_old.group(2)}") == 11.6,
              f"{m_old.groups() if m_old else None}")
        check("ANSI 乱码清洗（GPU）",
              _extract_device_mode("\x1b[0mGPU\x1b[0m\n") == "GPU",
              repr(_extract_device_mode("\x1b[0mGPU\x1b[0m\n")))
        check("ANSI 乱码清洗（CPU+多余输出）",
              _extract_device_mode("some extra\n\x1b[0mCPU\n") == "CPU",
              repr(_extract_device_mode("some extra\n\x1b[0mCPU\n")))
        from app.constants import APP_DIR
        check("当前环境无 nnUNet（本机）", not current_python_has_nnunet())
        env_res = find_python_with_nnunet()
        check("环境扫描正常（有则返回 python+GPU/CPU，无则 None）",
              env_res is None or (isinstance(env_res, tuple) and env_res[1] in ("GPU", "CPU")),
              f"{env_res}")
        base_py = find_base_python()
        check("自动配置可找到基础 Python", Path(base_py).exists(), base_py)
        ver = _python_version(base_py)
        check("基础 Python 版本可读", ver is not None, f"{ver}")
        check("GPU 检测返回布尔值", isinstance(has_nvidia_gpu(), bool))

        model_cfg = tmp / "model_cfg.json"
        model_cfg.write_text(
            _json.dumps({"model_folder": "", "checkpoint": "checkpoint_final.pth",
                         "folds": ["all"], "device": "cpu", "labels": {}},
                        ensure_ascii=False), encoding="utf-8")
        seg_in = tmp / "seg_in"
        seg_in.mkdir()
        shutil.copy2(image_nrrd, seg_in / "case001.nrrd")
        seg_out = tmp / "seg_out"
        script = APP_DIR / "app" / "nnunet_infer_script.py"
        r = subprocess.run(
            [sys.executable, str(script), str(model_cfg), str(seg_in), str(seg_out), "--device", "cpu"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        check("子进程脚本输出 JSON 错误", '"type": "error"' in r.stdout, r.stdout[-200:])
        check("子进程脚本以错误码退出", r.returncode != 0, f"rc={r.returncode}")

        from PySide6.QtWidgets import QApplication as _QA
        _app = _QA.instance() or _QA([])          # 需要 QApplication 才能投递线程信号

        from ui.segment_window import SegmentWorker
        failed_msgs = []
        w = SegmentWorker(str(seg_in), str(seg_out),
                          {"model_folder": "", "checkpoint": "checkpoint_final.pth",
                           "folds": ["all"], "device": "cpu", "labels": {}},
                          "cpu", False, False, python=sys.executable)
        w.failed.connect(failed_msgs.append)
        w.start()
        wait_until(lambda: len(failed_msgs) > 0 or not w.isRunning(), timeout=180)
        from PySide6.QtWidgets import QApplication as _QA
        for _ in range(20):                       # 排空 queued 的 failed 信号
            _QA.processEvents()
            time.sleep(0.01)
        check("SegmentWorker 子进程失败上报",
              len(failed_msgs) > 0 and "nnUNet" in failed_msgs[0],
              f"{failed_msgs[:1]}")

        # 停止子进程分割应能及时结束（不卡死）
        import ui.segment_window as segmod
        fake_script = tmp / "slow_child.py"
        fake_script.write_text(
            "import json, sys, time\n"
            "def emit(o): print(json.dumps(o, ensure_ascii=False), flush=True)\n"
            "emit({'type':'progress','done':0,'total':20,'message':'开始'})\n"
            "for i in range(1, 21):\n"
            "    time.sleep(0.5)\n"
            "    emit({'type':'progress','done':i,'total':20,'message':str(i)})\n"
            "emit({'type':'done','count':20})\n",
            encoding="utf-8")
        orig_script = segmod._INFER_SCRIPT
        segmod._INFER_SCRIPT = fake_script
        try:
            w2 = SegmentWorker(str(seg_in), str(seg_out),
                               {"device": "auto"}, "auto", False, False,
                               python=sys.executable)
            prog = []
            done_msgs = []
            w2.progress.connect(lambda i, n, m: prog.append(i))
            w2.done.connect(done_msgs.append)
            w2.failed.connect(done_msgs.append)
            w2.start()
            ok_start = wait_until(lambda: len(prog) >= 2, timeout=15)
            w2.stop()
            t0 = time.time()
            ok_stop = wait_until(lambda: not w2.isRunning(), timeout=15)
            elapsed = time.time() - t0
            check("停止分割能及时结束",
                  ok_start and ok_stop and elapsed < 12,
                  f"elapsed={elapsed:.1f}s done={done_msgs}")
        finally:
            segmod._INFER_SCRIPT = orig_script

        # ---------- 5. 界面构建（offscreen） ----------
        print("\n== 5. 界面构建 ==")
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication([])
        from app.style import apply_style
        apply_style(app)
        # 测试环境屏蔽弹窗
        for name in ("information", "warning", "critical"):
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

        from ui.main_window import MainWindow
        from ui.convert_window import ConvertWindow
        from ui.segment_window import SegmentWindow
        from ui.calculate_window import CalculateWindow
        from ui.model_manager_dialog import ModelManagerDialog

        mw = MainWindow()
        mw.show()
        check("主菜单窗口构建成功", mw.isVisible())

        cw = ConvertWindow()
        cw.show()
        cw._add_folders([dicom_folder])
        check("病人列表添加成功", len(cw.patients) == 1)
        check("加载中显示模态框", cw._load_dialog is not None and cw._load_dialog.isVisible())
        ok_load = wait_until(lambda: cw.arr is not None)
        check("转换窗口真实加载线程完成", ok_load)
        check("加载完成模态框已关闭", cw._load_dialog is None or not cw._load_dialog.isVisible())
        check("体数据尺寸正确", cw.arr is not None and cw.arr.shape[0] == gen.N_SLICES)
        check("三视图均有图像", all(v._image is not None for v in cw.views.values()))
        check("范围默认全量", cw.spin_range_start.value() == 1 and cw.spin_range_end.value() == gen.N_SLICES)

        # 纵横比校正（层厚 5mm vs 面内 1.5mm）
        check("冠状位纵横比校正生效",
              abs(cw.views["coronal"]._pixel_size[0] - 1.5) < 1e-6 and abs(cw.views["coronal"]._pixel_size[1] - 5.0) < 1e-6,
              f"{cw.views['coronal']._pixel_size}")
        check("矢状位纵横比校正生效",
              abs(cw.views["sagittal"]._pixel_size[0] - 1.5) < 1e-6 and abs(cw.views["sagittal"]._pixel_size[1] - 5.0) < 1e-6,
              f"{cw.views['sagittal']._pixel_size}")

        # 窗宽窗位变化应即时刷新图像（无需切换病人）
        img_before = cw.views["axial"]._image
        cw.spin_level.setValue(200)
        cw.spin_width.setValue(800)
        img_after = cw.views["axial"]._image
        check("窗宽窗位即时刷新图像", img_before is not img_after)
        check("窗宽窗位值生效", cw.wl == 200 and cw.ww == 800, f"wl={cw.wl}, ww={cw.ww}")

        cw._on_view_clicked("coronal", 40, 20)
        check("冠状位点击定位 z=21", cw.pos[2] == 20, f"z={cw.pos[2]}")
        cw.spin_range_start.setValue(5)
        cw.spin_range_end.setValue(15)
        check("范围设置生效", cw.patients[0]["range"] == (5, 15))
        cw.line_out_dir.setText(str(tmp / "ui_out"))
        cw.check_use_range.setChecked(True)
        cw.check_remove_after_export.setChecked(False)   # 本组用例保留病人
        cw.export_current()
        run_export(cw)          # 等待导出线程完成
        check("导出完成模态框已关闭", cw._export_dialog is None or not cw._export_dialog.isVisible())
        out_files = list((tmp / "ui_out").glob("*.nrrd"))
        check("界面导出文件生成", len(out_files) == 1)
        if out_files:
            z = read_image(out_files[0]).GetSize()[2]
            check("界面导出为选定范围(11层)", z == 11, f"z={z}")
            check("导出命名 Patient_序号_0000",
                  out_files[0].name == "Patient_001_0000.nrrd", out_files[0].name)

        # 再次导出当前病例 -> 自动接着编号
        cw.export_current()
        run_export(cw)
        out_files2 = [f.name for f in (tmp / "ui_out").glob("Patient_*_0000.nrrd")]
        check("再次导出自动接着编号",
              sorted(out_files2) == ["Patient_001_0000.nrrd", "Patient_002_0000.nrrd"],
              f"{sorted(out_files2)}")

        # 输出文件夹已有同名格式文件 -> 从最大序号+1 继续
        seed_dir = tmp / "ui_out_seed"
        seed_dir.mkdir()
        (seed_dir / "Patient_005_0000.nrrd").write_bytes(b"dummy")
        cw.line_out_dir.setText(str(seed_dir))
        cw._update_naming_hint()
        check("命名提示显示起始序号", "Patient_006_0000" in cw.label_naming.text(),
              cw.label_naming.text())
        cw.export_current()
        run_export(cw)
        seeded = [f.name for f in seed_dir.glob("Patient_*_0000.nrrd")]
        check("已有文件自动接着编号", "Patient_006_0000.nrrd" in seeded, f"{seeded}")

        # 滚轮合并节流
        from PySide6.QtCore import QPointF, QPoint, Qt
        from PySide6.QtGui import QWheelEvent
        got = []
        cw.views["axial"].scrolled.connect(got.append)
        ev = lambda dy: QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, dy),
                                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                    Qt.ScrollPhase.NoScrollPhase, False)
        for _ in range(3):
            cw.views["axial"].wheelEvent(ev(120))
        ok_deb = wait_until(lambda: len(got) > 0)
        check("滚轮事件合并为一次跳转", ok_deb and got == [3], f"got={got}")

        # 列表只显示姓名 + 不同层厚病人切换时保持物理位置
        phys_dir = tmp / "phys_patient"
        gen.write_dicom(vol, phys_dir, patient_name="测试病人B", patient_id="P002",
                        slice_thickness=2.5, series_description="动脉期")
        cw.set_pos(0, 0, 20)                     # 病人A层厚5mm，物理 z=100mm
        cw._add_folders([phys_dir])              # 自动选中最后一个（病人B）
        ok_phys = wait_until(lambda: cw.image is not None and abs(cw.image.GetSpacing()[2] - 2.5) < 1e-6)
        check("切换不同层厚病人加载完成", ok_phys)
        check("列表只显示病人姓名",
              cw.list_patients.item(0).text() == "测试病人" and cw.list_patients.item(1).text() == "测试病人B",
              f"{cw.list_patients.item(0).text()}, {cw.list_patients.item(1).text()}")
        check("跨病人保持物理位置 z≈40", ok_phys and cw.pos[2] == 40, f"z={cw.pos[2]}")

        # 导出全部：按导入顺序（A=006, B=007）自动编号
        cw.export_all()
        run_export(cw)
        seeded_all = [f.name for f in seed_dir.glob("Patient_*_0000.nrrd")]
        check("导出全部按导入顺序编号", "Patient_007_0000.nrrd" in seeded_all,
              f"{sorted(seeded_all)}")

        # 删除当前病人 + 导出后自动从列表移除
        cw_del = ConvertWindow()
        cw_del.show()
        cw_del._add_folders([dicom_folder, phys_dir])   # A, B；自动选中 B
        check("删除测试：2 个病人", len(cw_del.patients) == 2)
        cw_del.delete_current_patient()                  # 删除当前（B）
        check("删除当前病人", len(cw_del.patients) == 1 and cw_del.current == 0,
              f"n={len(cw_del.patients)}, current={cw_del.current}")
        del_dir = tmp / "ui_out_del"
        cw_del.line_out_dir.setText(str(del_dir))
        cw_del.check_remove_after_export.setChecked(True)
        cw_del.export_current()                          # 导出 A，成功后自动移除
        run_export(cw_del)
        check("导出后自动从列表移除", len(cw_del.patients) == 0, f"n={len(cw_del.patients)}")
        check("导出文件仍正常生成", (del_dir / "Patient_001_0000.nrrd").exists())

        # 异步扫描导入（多层嵌套）
        cw_async = ConvertWindow()
        cw_async.show()
        cw_async._import_root(nested_root)
        check("扫描中显示模态框", cw_async._scan_dialog is not None and cw_async._scan_dialog.isVisible())
        ok_scan = wait_until(lambda: len(cw_async.patients) == 2, timeout=30)
        check("异步扫描导入 2 个病例", ok_scan, f"n={len(cw_async.patients)}")
        check("扫描完成模态框已关闭", cw_async._scan_dialog is None or not cw_async._scan_dialog.isVisible())
        if ok_scan:
            check("异步导入的序列正确", cw_async.patients[0]["series_list"][0].instance_count == gen.N_SLICES)

        # 异步导入多期相病例 -> 自动切分为 3 个序列
        mp_parent = tmp / "mp_parent"
        mp_case = mp_parent / "多期相病例"
        write_multi_phase(vol, mp_case, n_phases=3)
        cw_mp = ConvertWindow()
        cw_mp.show()
        cw_mp._import_root(mp_parent)
        ok_mp = wait_until(lambda: len(cw_mp.patients) == 1 and len(cw_mp.patients[0]["series_list"]) == 1, timeout=30)
        check("界面自动只显示第一期（平扫）", ok_mp,
              f"series={len(cw_mp.patients[0]['series_list']) if cw_mp.patients else 0}")

        sw = SegmentWindow()
        sw.show()
        check("分割窗口构建成功", sw.combo_model.count() >= 1)
        check("环境线程属性已初始化", sw._setup_worker is None and sw._env_worker is None)
        mode = sw._resolve_device_mode(None)   # 当前进程无 torch -> 空；有则 GPU/CPU
        check("推理设备模式解析可用", mode in ("", "GPU", "CPU"), f"mode={mode}")

        cw2 = CalculateWindow()
        cw2.show()
        cw2.line_image_dir.setText(str(tmp / "images"))
        cw2.line_seg_dir.setText(str(tmp / "seg"))
        cw2.start_calculate()
        check("计算窗口匹配到病例", len(cw2.pairs) == 1)
        ok_calc = wait_until(lambda: cw2.table_result.rowCount() >= 1)
        check("计算线程完成并填充表格", ok_calc, f"rows={cw2.table_result.rowCount()}")
        check("计算完成模态框已关闭", cw2._calc_dialog is None or not cw2._calc_dialog.isVisible())

        dlg = ModelManagerDialog()
        check("模型管理对话框构建成功", dlg.list_models.count() >= 1)

        # 模型文件夹自动定位：选到数据集层也能找到 plans.json 层
        cfg_root = tmp / "Dataset001_身体成分"
        cfg_dir = cfg_root / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "plans.json").write_text("{}", encoding="utf-8")
        dlg_resolve = ModelManagerDialog()
        resolved = dlg_resolve._resolve_model_folder(cfg_root)
        check("模型文件夹自动定位到 plans.json 层", Path(resolved) == cfg_dir, resolved)
        check("已有 plans.json 的文件夹不被改动",
              dlg_resolve._resolve_model_folder(cfg_dir) == str(cfg_dir))

        print(f"\n========== 结果：{PASS} 通过，{FAIL} 失败 ==========")
        return 1 if FAIL else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
