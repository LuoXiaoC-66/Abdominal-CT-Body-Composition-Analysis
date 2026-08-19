# -*- coding: utf-8 -*-
"""NRRD 读写封装（基于 SimpleITK，统一处理 spacing/origin/direction）。

注意：SimpleITK/ITK 在 Windows 上无法直接读写含中文等非 ASCII 字符的路径
（例如 C:\\Users\\罗少卿\\...）。本模块自动检测非 ASCII 路径，
通过 ASCII 临时目录中转，保证软件在中文路径下也能正常工作。
"""
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

from app.constants import APP_DIR

_ASCII_TMP: Optional[Path] = None


# --------------------------------------------------------------- 路径处理
def _is_ascii_path(path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_tmp_root() -> Path:
    """返回一个可写且全 ASCII 的临时目录（带缓存）。"""
    global _ASCII_TMP
    if _ASCII_TMP is not None:
        return _ASCII_TMP
    env_tmp = os.environ.get("AUTOSEG_TMP", "").strip()
    candidates = []
    if env_tmp:
        candidates.append(Path(env_tmp))
    candidates += [
        APP_DIR / ".tmp",                                     # 应用内临时目录（随 APP_DIR 可移植）
        Path(os.environ.get("SystemRoot", "C:/Windows")) / "Temp" / "AutoSegment",
        Path(tempfile.gettempdir()),
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            if not _is_ascii_path(c):
                continue
            probe = c / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            _ASCII_TMP = c
            return _ASCII_TMP
        except OSError:
            continue
    _ASCII_TMP = Path(tempfile.mkdtemp(prefix="autoseg_"))
    return _ASCII_TMP


def stage_for_read(path) -> Tuple[str, bool]:
    """把文件转成 SimpleITK 可读的路径。

    返回 (可用路径, 是否临时文件)。若原路径为 ASCII 直接返回；否则复制到 ASCII 临时目录。
    """
    p = Path(path)
    if _is_ascii_path(p):
        return str(p), False
    dest = ascii_tmp_root() / uuid.uuid4().hex[:16] / f"{uuid.uuid4().hex[:8]}{p.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(p), str(dest))
    return str(dest), True


# ------------------------------------------------------------- 图像读写
def read_image(path) -> sitk.Image:
    """读取 nrrd（或任意 SimpleITK 支持的格式）为图像，自动处理非 ASCII 路径。"""
    usable, is_staged = stage_for_read(path)
    img = sitk.ReadImage(usable)
    if is_staged:
        try:
            staged = Path(usable)
            staged.unlink()
            staged.parent.rmdir()          # 删除空目录
        except OSError:
            pass
    return img


def read_array(path) -> np.ndarray:
    """读取为 numpy 数组，shape = (Z, Y, X)。"""
    return sitk.GetArrayFromImage(read_image(path))


def read_spacing(path) -> Tuple[float, float, float]:
    """读取体素间距 (x, y, z)，单位 mm。"""
    return tuple(read_image(path).GetSpacing())


def write_image(image: sitk.Image, path) -> None:
    """保存图像，按扩展名自动判断格式（.nrrd 即 NRRD），自动处理非 ASCII 路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_ascii_path(p):
        sitk.WriteImage(image, str(p))
        return
    tmpf = ascii_tmp_root() / f"{uuid.uuid4().hex[:16]}{p.suffix}"
    sitk.WriteImage(image, str(tmpf))
    shutil.move(str(tmpf), str(p))


def write_array_as_nrrd(
    array: np.ndarray,
    path,
    spacing=(1.0, 1.0, 1.0),
    origin: Optional[Tuple[float, float, float]] = None,
    direction: Optional[Tuple[float, ...]] = None,
    is_label: bool = False,
) -> None:
    """把 numpy 数组（shape=(Z,Y,X)）保存为 NRRD。"""
    arr = np.ascontiguousarray(array)
    if is_label:
        arr = arr.astype(np.uint8)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(tuple(spacing))
    if origin is not None:
        image.SetOrigin(tuple(origin))
    if direction is not None:
        image.SetDirection(tuple(direction))
    write_image(image, path)


def export_subvolume(image: sitk.Image, path, z_start: int, z_end: int) -> None:
    """导出 Z 轴范围 [z_start, z_end]（含端点，0 起索引）的子体积为 NRRD。

    使用 SimpleITK 的切片语法，自动更新 origin，保留 spacing/direction。
    """
    if z_start < 0 or z_end >= image.GetSize()[2] or z_start > z_end:
        raise ValueError(f"层面范围不合法: [{z_start}, {z_end}]，体数据共 {image.GetSize()[2]} 层")
    sub = image[:, :, z_start : z_end + 1]
    write_image(sub, path)
