# -*- coding: utf-8 -*-
"""DICOM 系列扫描与加载。

- 递归查找文件夹内的 DICOM 文件（按扩展名或文件头 DICOM 魔数识别）
- 按 SeriesInstanceUID 分组为多个序列（自动去重重复切片）
- 一个序列内按 InstanceNumber（文件夹排列顺序）排序，并依据 z 坐标回跳/大跳变
  自动切分；多期相堆叠（平扫→动脉→静脉）时**只保留第一期（通常为平扫）**
- 导入规则：把“所有直接包含 DICOM 的文件夹的上一级”作为病例文件夹导入
- 用 SimpleITK 读取为 3D 体数据（自动应用 CT 的 RescaleSlope/Intercept，得到 HU 值）
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima", ".img", ".v2", ".dcm3"}

_MAGIC = b"DICM"


def _looks_like_dicom(path: Path) -> bool:
    """先用扩展名快速判断，再校验文件头 DICM 魔数。"""
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == _MAGIC
    except OSError:
        return False


def find_dicom_files(folder, recursive: bool = True) -> List[Path]:
    """查找文件夹下所有 DICOM 文件（按魔数校验）。

    recursive=True 时递归查找子文件夹；False 时只查找当前文件夹。
    """
    folder = Path(folder)
    results = []
    if not folder.exists():
        return results
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for p in iterator:
        if p.is_file() and _looks_like_dicom(p):
            results.append(p)
    return results


def find_dicom_folders(root) -> List[Path]:
    """递归找出所有“直接包含 DICOM 文件”的文件夹（含 root 自身），返回按路径排序。"""
    root = Path(root)
    result = []
    if not root.exists():
        return result
    for dirpath, _dirnames, filenames in os.walk(root):
        folder = Path(dirpath)
        if any(_looks_like_dicom(folder / f) for f in filenames):
            result.append(folder)
    result.sort(key=str)
    return result


def collect_import_folders(root) -> List[Path]:
    """按用户规则收集需要导入的“病例文件夹”。

    规则：导入所选路径下所有包含 DICOM 文件的文件夹的**上一级**文件夹。
    - 若所选路径本身直接包含 DICOM，则导入该路径本身；
    - 若所选路径不含 DICOM、且其多个直接子文件夹都包含 DICOM（扁平多病人布局），
      则直接导入这些子文件夹，避免把不同病人合并到一个条目。
    """
    root = Path(root)
    dcm_folders = find_dicom_folders(root)
    if not dcm_folders:
        return []
    root_has_dcm = root in dcm_folders

    imports = set()
    for d in dcm_folders:
        if d == root:
            continue
        imports.add(d.parent)

    # 扁平多病人启发式：root 不含 DICOM，但有 ≥2 个直接子文件夹直接含 DICOM
    if not root_has_dcm:
        direct = [d for d in dcm_folders if d.parent == root]
        if len(direct) >= 2:
            imports.discard(root)     # 不要合并这些病人到 root 一个条目
            imports.update(direct)

    if root_has_dcm:
        imports.add(root)
    return sorted(imports, key=str)


def _sort_key(path: Path):
    """DICOM 排序键：优先按 InstanceNumber，其次 ImagePositionPatient 的 Z。

    多期相堆叠在同一文件夹时，文件夹内排列顺序（InstanceNumber）即为正确顺序，
    因此 InstanceNumber 优先级最高；z 仅作为次序相同时的辅助。
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return (0, 0.0, "")
    inst = int(getattr(ds, "InstanceNumber", 0) or 0)
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else 0.0
    return (inst, z, str(path))


@dataclass
class SeriesInfo:
    """一个 DICOM 序列（或期相）的基本信息。"""
    folder: Path
    files: List[str] = field(default_factory=list)      # 已按层面排序的文件路径
    series_uid: str = ""
    modality: str = ""
    series_description: str = ""
    study_description: str = ""
    patient_id: str = ""
    patient_name: str = ""
    study_date: str = ""
    rows: int = 0
    columns: int = 0
    slice_thickness: float = 0.0
    pixel_spacing: Tuple[float, float] = (0.0, 0.0)

    @property
    def instance_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict:
        return {
            "series_uid": self.series_uid,
            "modality": self.modality,
            "series_description": self.series_description,
            "study_description": self.study_description,
            "patient_id": self.patient_id,
            "patient_name": str(self.patient_name),
            "study_date": self.study_date,
            "instance_count": self.instance_count,
            "rows": self.rows,
            "columns": self.columns,
            "slice_thickness": self.slice_thickness,
            "pixel_spacing": list(self.pixel_spacing),
        }


def _read_basic_tags(path: Path) -> Optional[dict]:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else 0.0
    return {
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
        "modality": str(getattr(ds, "Modality", "") or ""),
        "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
        "study_description": str(getattr(ds, "StudyDescription", "") or ""),
        "patient_id": str(getattr(ds, "PatientID", "") or ""),
        "patient_name": str(getattr(ds, "PatientName", "") or ""),
        "study_date": str(getattr(ds, "StudyDate", "") or ""),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0.0) or 0.0),
        "pixel_spacing": tuple(float(x) for x in getattr(ds, "PixelSpacing", [0.0, 0.0])),
        "sort_z": z,
        "instance_number": int(getattr(ds, "InstanceNumber", 0) or 0),
    }


def _split_phases(items):
    """把单个序列按 z 坐标的明显回跳/大跳变切成多个期相。

    items: [(Path, tags)]，已按 InstanceNumber 顺序排列。
    期相堆叠（如 平扫/动脉/静脉 按顺序扫同一个范围）时，期相边界处 z 会
    大幅回退（重新从起始位置开始），据此切分；正常的缺层（z 递增但间隔稍大）
    不会被误切。
    返回：若干个 [(Path, tags)] 子列表。
    """
    if len(items) < 2:
        return [items]
    zs = [it[1]["sort_z"] for it in items]
    pos_dz = [zs[i + 1] - zs[i] for i in range(len(zs) - 1) if zs[i + 1] > zs[i]]
    median = float(np.median(pos_dz)) if pos_dz else 0.0
    if median <= 0:
        return [items]
    chunks = []
    start = 0
    for i in range(len(zs) - 1):
        dz = zs[i + 1] - zs[i]
        # 期相边界：z 明显回退（重新扫描）或正向跳变超过 3 倍常规间隔
        if dz < -0.5 * median or dz > 3.0 * median:
            chunks.append(items[start:i + 1])
            start = i + 1
    chunks.append(items[start:])
    return chunks


def scan_folder(folder) -> List[SeriesInfo]:
    """扫描文件夹，按序列分组返回 SeriesInfo 列表。

    - 按 SOPInstanceUID 去重重复切片
    - 序列内按 InstanceNumber 排序
    - 多期相堆叠时只保留第一期（通常为平扫）
    - 多个序列按最早扫描（最小 InstanceNumber）排序，确保“第一期”在前
    """
    folder = Path(folder)
    files = find_dicom_files(folder)
    groups: dict[str, list] = {}          # uid -> [(Path, tags)]
    seen_sops: dict[str, set] = {}
    for f in files:
        tags = _read_basic_tags(f)
        if tags is None:
            continue
        uid = tags["series_uid"] or str(f.parent)
        sop = tags.get("sop_uid", "")
        if sop and sop in seen_sops.setdefault(uid, set()):
            continue          # 跳过重复切片（如同一序列被重复导出）
        seen_sops[uid].add(sop)
        groups.setdefault(uid, []).append((f, tags))

    # 多个序列按最早扫描顺序排列（第一期=平扫 在前）
    groups_sorted = sorted(groups.items(),
                           key=lambda kv: min(it[1]["instance_number"] for it in kv[1]))

    series_list = []
    for uid, items in groups_sorted:
        items.sort(key=lambda it: (it[1]["instance_number"], it[1]["sort_z"], str(it[0])))
        chunks = _split_phases(items)
        chunk = chunks[0]                 # 只保留第一期（通常为平扫）
        base = items[0][1]
        series_list.append(
            SeriesInfo(
                folder=folder,
                files=[str(p) for p, _ in chunk],
                series_uid=uid,
                modality=base["modality"],
                series_description=base["series_description"],
                study_description=base["study_description"],
                patient_id=base["patient_id"],
                patient_name=base["patient_name"],
                study_date=base["study_date"],
                rows=base["rows"],
                columns=base["columns"],
                slice_thickness=base["slice_thickness"],
                pixel_spacing=base["pixel_spacing"],
            )
        )
    return series_list


def pick_best_series(series_list: List[SeriesInfo]) -> Optional[SeriesInfo]:
    """挑选最佳序列：优先 CT，其次层面数最多。"""
    if not series_list:
        return None
    ct = [s for s in series_list if s.modality.upper() == "CT"]
    candidates = ct if ct else series_list
    return max(candidates, key=lambda s: (s.instance_count, s.rows * s.columns))


def load_series_image(series: SeriesInfo) -> sitk.Image:
    """按层面顺序用 SimpleITK 读取序列，返回 3D 图像（值已转换为 HU）。"""
    if not series.files:
        raise ValueError("序列没有可读取的文件")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series.files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def series_image_to_array(image: sitk.Image) -> np.ndarray:
    """SimpleITK 图像 -> numpy 数组，shape = (Z, Y, X)。"""
    return sitk.GetArrayFromImage(image)
