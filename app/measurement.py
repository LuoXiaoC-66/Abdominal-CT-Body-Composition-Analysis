# -*- coding: utf-8 -*-
"""身体成分数据计算：每个分割区域的体积与平均 CT 值等统计量。

单位说明：
- 体积：mm³（体素数 × 体素体积）；同时给出 cm³（÷1000）。
- CT 值：图像数组已为 HU（SimpleITK 读取时自动应用窗位/截距）。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

from app.nrrd_io import read_image

METRIC_COLUMNS = [
    ("voxel_count", "体素数"),
    ("volume_mm3", "体积(mm³)"),
    ("volume_cm3", "体积(cm³)"),
    ("mean_hu", "平均CT值(HU)"),
    ("std_hu", "标准差(HU)"),
    ("min_hu", "最小CT值(HU)"),
    ("max_hu", "最大CT值(HU)"),
]

REGION_SHEET_COLUMNS = ["病例"] + [label for _, label in METRIC_COLUMNS]


@dataclass
class RegionMetrics:
    """单个病例、单个分割区域的统计结果。"""
    label: int
    name: str
    voxel_count: int = 0
    volume_mm3: float = 0.0
    volume_cm3: float = 0.0
    mean_hu: float = 0.0
    std_hu: float = 0.0
    min_hu: float = 0.0
    max_hu: float = 0.0

    def to_row(self, case_name: str) -> list:
        return [case_name, self.voxel_count, round(self.volume_mm3, 2),
                round(self.volume_cm3, 2), round(self.mean_hu, 2),
                round(self.std_hu, 2), round(self.min_hu, 2), round(self.max_hu, 2)]


def compute_metrics(image_path, seg_path, label_names: Optional[Dict[int, str]] = None) -> List[RegionMetrics]:
    """计算单个病例各分割区域的体积与 CT 统计量。

    label_names: {标签值: 区域名称}，未提供的标签自动命名为 “区域N”。
    """
    label_names = label_names or {}
    image = read_image(image_path)
    seg = read_image(seg_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float64)
    seg_arr = sitk.GetArrayFromImage(seg)
    if arr.shape != seg_arr.shape:
        raise ValueError(f"图像与分割维度不一致: {arr.shape} vs {seg_arr.shape}")

    spacing = image.GetSpacing()  # (x, y, z)
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    labels_found = sorted(int(v) for v in np.unique(seg_arr) if int(v) != 0)
    results = []
    for label in labels_found:
        mask = seg_arr == label
        voxels = int(np.count_nonzero(mask))
        vals = arr[mask]
        name = label_names.get(label, f"区域{label}")
        results.append(
            RegionMetrics(
                label=label,
                name=name,
                voxel_count=voxels,
                volume_mm3=round(voxels * voxel_volume_mm3, 2),
                volume_cm3=round(voxels * voxel_volume_mm3 / 1000.0, 2),
                mean_hu=float(np.mean(vals)),
                std_hu=float(np.std(vals)) if voxels > 1 else 0.0,
                min_hu=float(np.min(vals)) if voxels else 0.0,
                max_hu=float(np.max(vals)) if voxels else 0.0,
            )
        )
    return results


def match_files(image_dir, seg_dir, seg_suffix: str = "") -> List[tuple]:
    """匹配图像文件与分割文件，返回 [(图像路径, 分割路径, 病例名)]。

    seg_suffix: 分割文件名的后缀，例如 "_seg" 表示图像 case01.nrrd 对应
                case01_seg.nrrd；为空表示同名。
    """
    image_dir = Path(image_dir)
    seg_dir = Path(seg_dir)
    images = sorted([p for p in image_dir.glob("*.nrrd")])
    pairs = []
    for img in images:
        candidates = []
        stem = img.stem
        for suffix in (seg_suffix, ""):
            if suffix:
                candidates.append(seg_dir / f"{stem}{suffix}.nrrd")
            else:
                candidates.append(seg_dir / f"{stem}.nrrd")
        for cand in candidates:
            if cand.exists():
                pairs.append((str(img), str(cand), stem))
                break
    return pairs


def build_preview_frame(pairs: List[tuple], label_names: Dict[int, str]) -> pd.DataFrame:
    """构造用于界面预览的长表：行=病例，列=每个区域的体积/平均CT。"""
    rows = []
    for img_path, seg_path, name in pairs:
        metrics = compute_metrics(img_path, seg_path, label_names)
        row = {"病例": name}
        for m in metrics:
            row[f"{m.name}_体积(cm³)"] = m.volume_cm3
            row[f"{m.name}_平均CT(HU)"] = m.mean_hu
        rows.append(row)
    return pd.DataFrame(rows)


def export_excel(pairs: List[tuple], label_names: Dict[int, str], output_path) -> None:
    """按分割区域导出 Excel：
    - 每个区域一个工作表（所有病人的体积/CT 等指标）
    - 说明工作表：标签值 <-> 区域名称、单位说明
    - 汇总工作表：病人 × 区域 的 体积(cm³) 与 平均CT(HU)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 预计算全部结果
    per_case: List[Dict[int, RegionMetrics]] = []
    case_names: List[str] = []
    for img_path, seg_path, name in pairs:
        metrics = compute_metrics(img_path, seg_path, label_names)
        per_case.append({m.label: m for m in metrics})
        case_names.append(name)

    # 收集出现的所有标签（按 label 排序，名称优先用 label_names）
    labels_present: Dict[int, str] = {}
    for case_metrics in per_case:
        for m in case_metrics.values():
            labels_present.setdefault(m.label, label_names.get(m.label, f"区域{m.label}"))
    ordered = sorted(labels_present.items())

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1) 说明
        info_rows = [["列名", "含义", "单位"],
                     ["体素数", "该区域包含的体素个数", "个"],
                     ["体积(mm³)", "区域体积", "mm³"],
                     ["体积(cm³)", "区域体积", "cm³"],
                     ["平均CT值(HU)", "区域内所有体素 CT 值均值", "HU"],
                     ["标准差(HU)", "区域内 CT 值标准差", "HU"],
                     ["最小/最大CT值(HU)", "区域内 CT 值范围", "HU"],
                     ["", "", ""]]
        for label, name in ordered:
            info_rows.append([f"标签 {label}", name, ""])
        pd.DataFrame(info_rows).to_excel(writer, sheet_name="说明", index=False, header=False)

        # 2) 每个区域一个工作表
        for label, name in ordered:
            rows = []
            for i, case_name in enumerate(case_names):
                m = per_case[i].get(label)
                if m is None:
                    rows.append([case_name] + [""] * (len(METRIC_COLUMNS)))
                else:
                    rows.append(m.to_row(case_name))
            pd.DataFrame(rows, columns=REGION_SHEET_COLUMNS).to_excel(
                writer, sheet_name=_safe_sheet_name(name), index=False)

        # 3) 汇总
        sum_vol = pd.DataFrame({"病例": case_names})
        sum_ct = pd.DataFrame({"病例": case_names})
        for label, name in ordered:
            sum_vol[name] = [per_case[i].get(label).volume_cm3 if per_case[i].get(label) else "" for i in range(len(case_names))]
            sum_ct[name] = [per_case[i].get(label).mean_hu if per_case[i].get(label) else "" for i in range(len(case_names))]
        sum_vol.to_excel(writer, sheet_name="汇总_体积(cm³)", index=False)
        sum_ct.to_excel(writer, sheet_name="汇总_平均CT(HU)", index=False)


def _safe_sheet_name(name: str, max_len: int = 30) -> str:
    """Excel 工作表名去非法字符并限长。"""
    for ch in r"\/?*[]:：":
        name = name.replace(ch, "_")
    return (name or "区域")[:max_len]
