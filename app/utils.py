# -*- coding: utf-8 -*-
"""通用小工具。"""
import re
from pathlib import Path


def safe_filename(text: str, fallback: str = "unnamed") -> str:
    """把任意字符串转成安全的文件名片段（去除 Windows 非法字符）。"""
    if not text:
        return fallback
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(text)).strip(" .")
    return cleaned or fallback


def unique_path(directory, stem: str, suffix: str = ".nrrd") -> Path:
    """在 directory 下生成不重名路径：stem.nrrd, stem_2.nrrd, stem_3.nrrd ..."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def format_seconds(seconds: float) -> str:
    """把秒格式化为可读的时分秒字符串。"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"
