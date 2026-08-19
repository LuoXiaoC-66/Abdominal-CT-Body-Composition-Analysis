# -*- coding: utf-8 -*-
"""nnUNet 模型配置读写。

模型配置文件 models.json 存放在项目根目录，每条模型记录：
{
  "id": "唯一标识",
  "name": "显示名称（例如 身体成分-3D）",
  "model_folder": "nnUNet 模型文件夹绝对路径",
  "checkpoint": "checkpoint_final.pth",
  "folds": ["all"] 或 ["0","1","2","3","4"],
  "device": "auto / cuda / cpu",
  "labels": {"1": "内脏脂肪", "2": "皮下脂肪", ...}   # 标签值 -> 区域名称
}
"""
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .constants import APP_DIR, MODELS_CONFIG_PATH

# 默认预置 4 个模型占位（与你的 4 个 nnUNet 模型对应），模型文件夹路径留空，由界面“模型管理”填写。
def resolve_model_folder(folder) -> str:
    """把 models.json 中的模型路径解析为绝对路径（可移植：支持相对 APP_DIR 的路径）。"""
    if not folder:
        return ""
    f = str(folder)
    if os.path.isabs(f) and Path(f).exists():
        return f
    rel = APP_DIR / f
    if rel.exists():
        return str(rel)
    return f


def relative_model_folder(folder) -> str:
    """把模型路径转为相对 APP_DIR 的相对路径（便于整体移动软件文件夹）。"""
    if not folder:
        return ""
    try:
        p = Path(folder)
        if p.is_absolute():
            return str(p.resolve().relative_to(APP_DIR.resolve()))
    except ValueError:
        pass
    return str(folder)


DEFAULT_MODELS: List[dict] = [
    {
        "id": "bc_3d",
        "name": "身体成分-3D",
        "model_folder": "",
        "checkpoint": "checkpoint_final.pth",
        "folds": ["all"],
        "device": "auto",
        "labels": {"1": "内脏脂肪", "2": "皮下脂肪", "3": "骨骼", "4": "骨骼肌", "5": "肌间脂肪"},
    },
    {
        "id": "bc_2d",
        "name": "身体成分-2D",
        "model_folder": "",
        "checkpoint": "checkpoint_final.pth",
        "folds": ["all"],
        "device": "auto",
        "labels": {"1": "内脏脂肪", "2": "皮下脂肪", "3": "骨骼", "4": "骨骼肌", "5": "肌间脂肪"},
    },
    {
        "id": "muscle_3d",
        "name": "肌肉群-3D",
        "model_folder": "",
        "checkpoint": "checkpoint_final.pth",
        "folds": ["all"],
        "device": "auto",
        "labels": {"1": "腰大肌", "2": "腹直肌", "3": "腰方肌", "4": "竖脊肌"},
    },
    {
        "id": "muscle_2d",
        "name": "肌肉群-2D",
        "model_folder": "",
        "checkpoint": "checkpoint_final.pth",
        "folds": ["all"],
        "device": "auto",
        "labels": {"1": "腰大肌", "2": "腹直肌", "3": "腰方肌", "4": "竖脊肌"},
    },
]


def load_models(config_path: Optional[Path] = None) -> List[dict]:
    """读取模型配置；文件不存在时写入默认占位配置。"""
    path = Path(config_path) if config_path else MODELS_CONFIG_PATH
    if not path.exists():
        save_models(DEFAULT_MODELS, path)
        return [dict(m) for m in DEFAULT_MODELS]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data if isinstance(data, list) else data.get("models", [])
        result = []
        for m in models:
            if not isinstance(m, dict):
                continue
            mm = dict(m)
            mm["model_folder"] = resolve_model_folder(mm.get("model_folder", ""))
            result.append(mm)
        return result
    except Exception:
        return [dict(m) for m in DEFAULT_MODELS]


def save_models(models: List[dict], config_path: Optional[Path] = None) -> None:
    """保存模型配置到 models.json（APP_DIR 内的路径存为相对路径，便于整体移动）。"""
    path = Path(config_path) if config_path else MODELS_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for m in models:
        mm = dict(m)
        mm["model_folder"] = relative_model_folder(mm.get("model_folder", ""))
        data.append(mm)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_model_entry(name: str = "新模型") -> dict:
    """创建一个空的模型记录。"""
    return {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "model_folder": "",
        "checkpoint": "checkpoint_final.pth",
        "folds": ["all"],
        "device": "auto",
        "labels": {"1": "区域1"},
    }


def labels_to_dict(model: dict) -> Dict[int, str]:
    """把模型配置里的 labels（键可能是字符串）转成 {int: str}。"""
    out: Dict[int, str] = {}
    for k, v in (model.get("labels") or {}).items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def dict_to_labels(label_map: Dict[int, str]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in sorted(label_map.items())}


def validate_model(model: dict) -> str:
    """检查模型配置是否可用于分割，返回空串表示可用，否则返回原因。"""
    folder = model.get("model_folder", "")
    if not folder:
        return "未设置模型文件夹，请在“模型管理”中选择"
    p = Path(folder)
    if not p.exists():
        return f"模型文件夹不存在：{folder}"
    if not (p / "plans.json").exists():
        return f"模型文件夹中未找到 plans.json：{folder}"
    return ""
