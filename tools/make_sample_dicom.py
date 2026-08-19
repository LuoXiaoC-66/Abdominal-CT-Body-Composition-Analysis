# -*- coding: utf-8 -*-
"""生成一套模拟数据，用于体验软件的三个流程（无需真实 DICOM）：

  sample_data/
  ├── dicom/病人001/       # 模拟 DICOM 序列（48 层 CT）
  ├── images/case001.nrrd  # 相当于界面①转换后的图像
  └── seg/case001.nrrd     # 相当于界面②的分割结果

运行：python tools/make_sample_dicom.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent / "sample_data"
DICOM_DIR = ROOT / "dicom"
IMAGE_DIR = ROOT / "images"
SEG_DIR = ROOT / "seg"

N_SLICES = 48
SIZE = 96
PIXEL_SPACING = 1.5      # mm
SLICE_THICKNESS = 5.0    # mm


def m3(mask2d):
    """把 2D 掩码扩展为 3D（每个层面相同）。"""
    return np.repeat(mask2d[None, :, :], N_SLICES, axis=0)


def build_volume():
    """构造模拟腹部 CT（HU）：空气-1000、体表、皮下脂肪、肌肉、内脏脂肪。"""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx = cy = (SIZE - 1) / 2.0
    rx, ry = 42, 34                          # 身体椭圆
    body = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    vol = np.full((N_SLICES, SIZE, SIZE), -1000.0)   # 空气
    vol[m3(body)] = 40.0                             # 软组织基础

    # 皮下脂肪环（身体边缘向内 8 像素）
    inner = ((xx - cx) / (rx - 8)) ** 2 + ((yy - cy) / (ry - 8)) ** 2 <= 1.0
    subcut = body & ~inner
    vol[m3(subcut)] = -95.0

    # 左右腰大肌（两个椭圆）
    muscles = np.zeros_like(body)
    for mx in (cx - 16, cx + 16):
        muscles |= ((xx - mx) / 7) ** 2 + ((yy - cy) / 9) ** 2 <= 1.0
    vol[m3(muscles)] = 48.0

    # 内脏脂肪（中心一团）
    vis = ((xx - cx) / 16) ** 2 + ((yy - cy) / 18) ** 2 <= 1.0
    vis &= ~muscles
    vol[m3(vis)] = -85.0

    # 脊柱（骨骼，中心竖条）
    spine = (np.abs(xx - cx) <= 4) & (np.abs(yy - cy) <= 12)
    vol[m3(spine)] = 400.0
    return vol, body, subcut, vis, spine, muscles


def build_seg(body, subcut, vis, spine, muscles):
    """标签：1=内脏脂肪 2=皮下脂肪 3=骨骼 4=骨骼肌"""
    seg = np.zeros((N_SLICES, SIZE, SIZE), dtype=np.uint8)
    seg[m3(vis)] = 1
    seg[m3(subcut)] = 2
    seg[m3(spine)] = 3
    seg[m3(muscles)] = 4
    return seg


def write_dicom(vol, out_dir, patient_name="测试病人", patient_id="P001",
                study_date="20240815", slice_thickness=None, series_description="腹部平扫"):
    """把 3D 体数据写成一整套 DICOM 序列。slice_thickness 可指定不同层厚。"""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian

    st = slice_thickness if slice_thickness is not None else SLICE_THICKNESS
    out_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    rows, cols = vol.shape[1], vol.shape[2]
    stored = np.clip(vol + 1024, 0, 65535).astype(np.uint16)  # intercept=-1024, slope=1

    for i in range(N_SLICES):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SpecificCharacterSet = "ISO_IR 192"   # UTF-8，保证中文姓名正常
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.StudyDate = study_date
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.SeriesDescription = series_description
        ds.InstanceNumber = i + 1
        ds.Rows, ds.Columns = rows, cols
        ds.PixelSpacing = [PIXEL_SPACING, PIXEL_SPACING]
        ds.SliceThickness = st
        ds.ImagePositionPatient = [0.0, 0.0, i * st]
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
        ds.save_as(out_dir / f"slice_{i + 1:03d}.dcm")


def main():
    # 使用应用内的安全 NRRD 写入（兼容中文路径）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.nrrd_io import write_image

    if ROOT.exists():
        shutil.rmtree(ROOT)
    vol, body, subcut, vis, spine, muscles = build_volume()

    write_dicom(vol, DICOM_DIR / "病人001")
    print(f"[1/3] 模拟 DICOM 序列 -> {DICOM_DIR / '病人001'}")

    # 界面①的产物（图像 nrrd）
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((PIXEL_SPACING, PIXEL_SPACING, SLICE_THICKNESS))
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    write_image(img, str(IMAGE_DIR / "case001.nrrd"))
    print(f"[2/3] 转换后图像 nrrd -> {IMAGE_DIR / 'case001.nrrd'}")

    # 界面②的产物（分割 nrrd）
    seg = build_seg(body, subcut, vis, spine, muscles)
    seg_img = sitk.GetImageFromArray(seg.astype(np.uint8))
    seg_img.SetSpacing((PIXEL_SPACING, PIXEL_SPACING, SLICE_THICKNESS))
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    write_image(seg_img, str(SEG_DIR / "case001.nrrd"))
    print(f"[3/3] 分割结果 nrrd -> {SEG_DIR / 'case001.nrrd'}")

    print("\n完成！现在可以启动软件体验三个流程。")
    print("界面①：添加 " + str(DICOM_DIR / "病人001"))
    print("界面②：输入 " + str(IMAGE_DIR) + "（需配置真实 nnUNet 模型）")
    print("界面③：图像 " + str(IMAGE_DIR) + " / 分割 " + str(SEG_DIR))


if __name__ == "__main__":
    sys.exit(main())
