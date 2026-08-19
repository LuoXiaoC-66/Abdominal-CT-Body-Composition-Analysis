# -*- coding: utf-8 -*-
"""nnUNet v2 自动分割封装。

nnunetv2 / torch 只在真正执行分割时才导入（惰性导入），
因此未安装 nnUNet 的机器上，其余两个流程仍可正常使用。

中文路径处理：nnUNet 内部使用 SimpleITK 读写中间文件，在 Windows 上无法
处理中文路径。因此输入/输出路径含非 ASCII 字符时，先复制到 ASCII 临时目录
执行推理，再把结果写回用户指定目录。
"""
import inspect
import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional


from app.nrrd_io import ascii_tmp_root, read_image, write_image


def check_nnunet_available() -> tuple:
    """检查 nnunetv2 是否可用。返回 (是否可用, 提示信息)。"""
    try:
        import torch  # noqa: F401
        import nnunetv2  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, f"未检测到 nnUNet 运行环境：{e}\n\n请在安装了 nnUNet V2 和 PyTorch 的 Python 环境中运行本软件（见 README）。"


def _is_ascii(path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def predict_folder(
    input_folder,
    output_folder,
    model: dict,
    device: str = "auto",
    save_probabilities: bool = False,
    overwrite: bool = False,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """对输入文件夹内所有 NRRD 执行 nnUNet 推理，结果保存为同名 .nrrd。

    返回成功处理的病例数。
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in (".nrrd", ".nii.gz")])
    if not files:
        raise ValueError(f"输入文件夹中没有找到 NRRD 文件：{input_dir}")

    def _log(msg):
        if log_cb:
            log_cb(msg)

    ok, msg = check_nnunet_available()
    if not ok:
        raise RuntimeError(msg)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder = model["model_folder"]
    checkpoint = model.get("checkpoint", "checkpoint_final.pth")
    folds = tuple(model.get("folds") or ["all"])

    _log(f"正在加载模型：{model.get('name', '')}")
    _log(f"模型文件夹：{model_folder}")
    import torch
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    # 按实际版本签名构造预测器，兼容不同 nnunetv2 版本
    init_params = set(inspect.signature(nnUNetPredictor.__init__).parameters)
    init_kwargs = {"verbose": False, "device": dev}
    if "perform_everything_on_device" in init_params:
        init_kwargs["perform_everything_on_device"] = dev.type == "cuda"
    if "install_models" in init_params:
        init_kwargs["install_models"] = True
    predictor = nnUNetPredictor(**init_kwargs)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=folds, checkpoint_name=checkpoint)
    _log(f"模型加载完成，推理设备：{dev}")

    # 按实际版本签名准备 predict_from_files 参数（新旧版本参数名不同）
    predict_params = set(inspect.signature(predictor.predict_from_files).parameters)
    predict_kwargs = {
        "save_probabilities": save_probabilities,
        "overwrite": overwrite,
        "num_processes_preprocessing": 1,
    }
    if "num_processes_segmentation" in predict_params:
        predict_kwargs["num_processes_segmentation"] = 1
    elif "num_processes_segmentation_export" in predict_params:
        predict_kwargs["num_processes_segmentation_export"] = 1
    if "num_processes_postprocessing" in predict_params:
        predict_kwargs["num_processes_postprocessing"] = 1

    _log(f"输入文件夹共 {len(files)} 例，开始分割。")

    # nnUNet 输出统一写入 ASCII 临时目录：避免中文路径问题，也避免把 nnunet 的
    # json 中间文件写进用户输出文件夹
    staging = ascii_tmp_root() / uuid.uuid4().hex[:16]
    stage_in = staging / "in"
    stage_out = staging / "out"
    stage_in.mkdir(parents=True)
    stage_out.mkdir()

    done = 0
    failed = []
    try:
        for i, f in enumerate(files, start=1):
            if should_stop and should_stop():
                _log("已手动停止。")
                break
            out_nrrd = output_dir / f"{f.stem}.nrrd"
            if out_nrrd.exists() and not overwrite:
                _log(f"[{i}/{len(files)}] {f.name} 已存在，跳过。")
                done += 1
                if progress_cb:
                    progress_cb(i, len(files), f"{f.name}（已存在，跳过）")
                continue

            if progress_cb:
                progress_cb(i, len(files), f"正在分割第 {i}/{len(files)} 例：{f.name} ...")
            _log(f"[{i}/{len(files)}] 预测 {f.name}")

            # 输入：非 ASCII 路径时复制到 ASCII 临时目录（nnUNet 内部 SimpleITK 无法处理中文路径）
            src_path = str(f)
            if not _is_ascii(str(f)):
                staged_in = stage_in / f.name
                shutil.copy2(str(f), str(staged_in))
                src_path = str(staged_in)

            # 清空上次的临时输出，避免残留文件干扰
            for p in stage_out.iterdir():
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass

            out_truncated = str(stage_out / f"{f.stem}_nnunet_tmp")
            try:
                predictor.predict_from_files([[src_path]], [out_truncated], **predict_kwargs)
            except Exception as e:  # noqa: BLE001
                failed.append((f.name, str(e)))
                _log(f"第 {i} 例预测异常：{e}")
                continue

            # 输出扩展名可能是 .nii.gz 或 .nrrd（取决于模型 dataset.json 的 file_ending），按实际查找
            candidates = sorted(stage_out.glob(f"{f.stem}_nnunet_tmp.*"))
            candidates = [c for c in candidates if c.suffix.lower() in (".nrrd", ".nii.gz")]
            if not candidates:
                failed.append((f.name, "未生成预测输出文件"))
                _log(f"第 {i} 例未生成预测输出文件。")
                continue

            seg_file = candidates[0]
            try:
                img = read_image(str(seg_file))
                write_image(img, str(out_nrrd))
            except Exception as e:  # noqa: BLE001
                failed.append((f.name, f"读取/保存预测结果失败：{e}"))
                _log(f"第 {i} 例结果保存失败：{e}")
                continue
            try:
                seg_file.unlink()
            except OSError:
                pass
            done += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if failed:
        _log(f"完成：成功 {done} 例，失败 {len(failed)} 例。")
        for name, err in failed:
            _log(f"  - {name}：{err}")
    else:
        _log(f"完成：共 {done} 例全部成功。")

    return done
