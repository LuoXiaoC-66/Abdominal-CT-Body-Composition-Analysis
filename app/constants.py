# -*- coding: utf-8 -*-
"""全局常量与默认路径。"""
import sys
from pathlib import Path


def _app_root() -> Path:
    """应用根目录：打包为 exe 时取 exe 所在目录（可整体移动）；开发时取项目根。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


# 应用根目录
APP_DIR = _app_root()

# 模型放置目录（预留路径，用户把 nnUNet 模型文件夹放进来即可）
MODELS_DIR = APP_DIR / "models"

# 模型配置文件
MODELS_CONFIG_PATH = APP_DIR / "models.json"

APP_NAME = "CT身体成分分析系统"
APP_VERSION = "1.0.0"

# 窗宽窗位预设（level, width）
WINDOW_LEVEL_PRESETS = {
    "腹部软组织窗": (40, 350),
    "骨窗": (400, 2000),
    "肺窗": (-600, 1500),
    "自定义": None,
}
DEFAULT_WINDOW_LEVEL = (40, 350)

# 导出文件名（nnUNet 输入要求 统一名称_序号_0000）
EXPORT_BASE_NAME = "Patient"          # 文件名前缀
EXPORT_MODALITY_SUFFIX = "_0000"      # nnUNet 单模态后缀
