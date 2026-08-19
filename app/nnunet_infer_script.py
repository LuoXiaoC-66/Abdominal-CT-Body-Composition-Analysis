# -*- coding: utf-8 -*-
"""nnUNet 推理子进程脚本（由界面②调用）。

当界面所在的 Python 没有安装 nnunetv2/torch 时，界面会找到训练环境的
python.exe，并以此脚本作为子进程运行分割，避免用户手动安装依赖。

用法：
    python nnunet_infer_script.py <model.json> <输入文件夹> <输出文件夹> \\
        [--save-prob] [--overwrite] [--device auto]

stdout 以 JSON Lines 输出：
    {"type":"progress","done":i,"total":n,"message":"..."}
    {"type":"log","message":"..."}
    {"type":"done","count":n}
    {"type":"error","message":"..."}
"""
import argparse
import json
import sys
from pathlib import Path

# 强制 UTF-8 输出，避免中文在中文 Windows（GBK）下变成乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_json")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--save-prob", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        model = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
        from app.segmentation import predict_folder

        def progress(i, n, msg):
            emit({"type": "progress", "done": i, "total": n, "message": msg})

        def log(msg):
            emit({"type": "log", "message": msg})

        count = predict_folder(
            args.input_dir,
            args.output_dir,
            model,
            device=args.device,
            save_probabilities=args.save_prob,
            overwrite=args.overwrite,
            progress_cb=progress,
            log_cb=log,
        )
        emit({"type": "done", "count": count})
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
