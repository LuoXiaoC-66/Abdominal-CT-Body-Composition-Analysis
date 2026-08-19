# -*- coding: utf-8 -*-
"""nnUNet 运行环境检测与自动配置。

软件本身（界面）运行的 Python 里不一定安装了 nnunetv2 / torch（它们体积大、
通常只在训练环境里有）。本模块提供：
- 判断当前 Python 是否可直接运行 nnUNet；
- 扫描本机常见 conda 环境，找出装有 nnunetv2 + torch 的 python.exe；
- 找不到时，**自动创建虚拟环境并安装 nnunetv2 + torch**（一键配置，医生无需动手）。
  pip 默认使用国内镜像源（清华 TUNA），失败自动回退官方源，网络更稳定。
"""
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from .constants import APP_DIR

_CHECK_CODE = (
    "import torch; import nnunetv2; "
    "print('GPU' if torch.cuda.is_available() else 'CPU')"
)

# ANSI 终端颜色/控制码（tqdm、颜色库可能混进 stdout）
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# nvidia-smi 里的 CUDA 版本：旧驱动为 "CUDA Version: 11.6"，新驱动为 "CUDA UMD Version: 13.3"
_CUDA_VERSION_RE = re.compile(r"CUDA\s+(?:UMD\s+)?Version:\s*(\d+)\.(\d+)")


def _extract_device_mode(stdout: str) -> str:
    """从子进程 stdout 中提取 'GPU' / 'CPU'（忽略 ANSI 控制码与其他输出）。"""
    clean = _ANSI_RE.sub("", stdout or "")
    for line in reversed(clean.splitlines()):
        line = line.strip()
        if line in ("GPU", "CPU"):
            return line
    return "CPU"

# 自动创建的 nnUNet 虚拟环境位置（开发模式）
ENV_DIR = APP_DIR / ".nnunet_env"
ENV_PYTHON = ENV_DIR / "Scripts" / "python.exe"

# 打包版自带的迷你嵌入式 Python（首次使用自动向它安装 torch+nnunetv2）
RUNTIME_PYTHON = APP_DIR / "runtime" / "python.exe"

_PREFERRED_MINORS = (12, 11, 10, 9)      # 优先使用这些 Python 版本创建环境

# pip 镜像源（可通过环境变量 AUTOSEG_PIP_INDEX 覆盖；留空则用官方源）
PIP_MIRROR = os.environ.get("AUTOSEG_PIP_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple").strip()
# PyTorch CPU 版索引（官方 + 清华镜像）
TORCH_CPU_INDEXES = [
    "https://download.pytorch.org/whl/cpu",
    "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cpu",
]
# PyTorch GPU 版索引：必须用 PyTorch 官方 CUDA 源（PyPI/普通镜像上的 torch 是 CPU 版）
TORCH_GPU_INDEXES = [
    "https://download.pytorch.org/whl/cu126",
    "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
]


def _flags():
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def current_python_has_nnunet() -> bool:
    """当前 Python 进程是否已安装 nnunetv2 和 torch。"""
    try:
        import torch  # noqa: F401
        import nnunetv2  # noqa: F401
        return True
    except Exception:
        return False


def _candidate_pythons() -> list:
    """返回本机候选 python.exe 路径：
    打包自带运行时优先，其次软件目录附近已配置的 .nnunet_env（venv 形式的 nnUNet 环境），
    再其次当前 Python 与常见 conda 环境。"""
    candidates = []
    if RUNTIME_PYTHON.exists():
        candidates.append(str(RUNTIME_PYTHON))
    # 软件目录及其 1~2 级父目录下的 .nnunet_env（复用已配置好的环境，避免重复安装）
    try:
        bases = [APP_DIR] + [APP_DIR.parents[i] for i in range(min(2, len(APP_DIR.parents)))]
    except Exception:
        bases = [APP_DIR]
    for base in bases:
        py = base / ".nnunet_env" / "Scripts" / "python.exe"
        if py.is_file() and str(py) not in candidates:
            candidates.append(str(py))
    candidates.append(sys.executable)
    roots = []
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        roots.append(Path(conda_exe).resolve().parent.parent)   # conda 根目录
    home_roots = [
        Path.home() / "anaconda3",
        Path.home() / "miniconda3",
        Path.home() / "mambaforge",
        Path("C:/ProgramData/Anaconda3"),
        Path("C:/ProgramData/Miniconda3"),
        Path("D:/Install/Anaconda"),
        Path("D:/Install/Miniconda3"),
    ]
    roots.extend(home_roots)

    seen = set()
    result = []
    for root in roots:
        envs = root / "envs"
        if not envs.is_dir():
            continue
        for env in sorted(envs.iterdir()):
            py = env / "python.exe"
            if py.is_file():
                sp = str(py)
                if sp not in seen:
                    seen.add(sp)
                    result.append(sp)
    return candidates + result


def check_python(python_path, timeout: int = 15):
    """检测某个 python.exe 是否可用 nnUNet。返回 (是否可用, 'GPU'/'CPU')。"""
    py = str(python_path)
    try:
        r = subprocess.run([py, "-c", _CHECK_CODE],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=_flags())
        if r.returncode == 0:
            return True, _extract_device_mode(r.stdout)
    except Exception:
        pass
    return False, ""


def find_python_with_nnunet(timeout: int = 15):
    """扫描本机候选 Python，返回第一个装有 nnUNet 的 (python路径, 'GPU'/'CPU')；找不到返回 None。"""
    for py in _candidate_pythons():
        if os.path.abspath(py) == os.path.abspath(sys.executable):
            continue                    # 当前进程是否可用已单独判断
        ok, mode = check_python(py, timeout=timeout)
        if ok:
            return py, mode
    return None


# ------------------------------------------------------------- 自动配置环境
def _python_version(python_path):
    """返回某个 python 的 (主版本, 次版本)；不可用返回 None。"""
    try:
        r = subprocess.run(
            [str(python_path), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, creationflags=_flags())
        if r.returncode == 0 and r.stdout.strip():
            major, _, minor = r.stdout.strip().splitlines()[-1].partition(".")
            return int(major), int(minor)
    except Exception:
        pass
    return None


def _py_launcher_python(minor: int):
    """通过 Windows py 启动器查找指定次版本的 python.exe。"""
    try:
        r = subprocess.run(
            ["py", f"-3.{minor}", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, creationflags=_flags())
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[-1]
    except Exception:
        pass
    return None


def find_base_python() -> str:
    """找一个适合安装 nnUNet 环境的基础 Python：打包版优先用自带 runtime，否则找 3.9-3.12。"""
    if RUNTIME_PYTHON.exists():
        return str(RUNTIME_PYTHON)
    current = sys.executable
    cur_ver = sys.version_info[:2]
    if (3, 9) <= cur_ver <= (3, 12):
        return current
    for minor in _PREFERRED_MINORS:
        py = _py_launcher_python(minor)
        if py:
            ver = _python_version(py)
            if ver and (3, 9) <= ver <= (3, 12):
                return py
    for py in _candidate_pythons():
        if os.path.abspath(py) == os.path.abspath(current):
            continue
        ver = _python_version(py)
        if ver and (3, 9) <= ver <= (3, 12):
            return py
    return current


def has_nvidia_gpu() -> bool:
    """简单判断本机是否有 NVIDIA 显卡（用于选择 GPU/CPU 版 PyTorch）。"""
    if os.name != "nt":
        return False
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10,
                           creationflags=_flags())
        return r.returncode == 0
    except Exception:
        return False


def driver_cuda_version():
    """返回显卡驱动支持的最高 CUDA 版本（如 11.8 / 12.4）；无法获取返回 None。"""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10,
                           creationflags=_flags())
        if r.returncode != 0:
            return None
        m = _CUDA_VERSION_RE.search(r.stdout)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
    except Exception:
        pass
    return None


def _env_cuda_available(python_path) -> bool:
    """检查某个环境里的 torch 是否真的能使用 GPU。"""
    try:
        r = subprocess.run(
            [str(python_path), "-c", "import torch; print(int(torch.cuda.is_available()))"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, creationflags=_flags())
        return r.returncode == 0 and r.stdout.strip().endswith("1")
    except Exception:
        return False


def _env_has_torch(python_path) -> bool:
    """检查某个环境里是否已安装 torch。"""
    try:
        r = subprocess.run([str(python_path), "-c", "import torch"],
                           capture_output=True, timeout=30, creationflags=_flags())
        return r.returncode == 0
    except Exception:
        return False


def _env_torch_status(python_path):
    """检查环境里 torch 的安装与 GPU 情况。

    返回 (是否安装, GPU是否可用, torch版本字符串, torch.version.cuda)。
    torch.version.cuda 非空表示这是 GPU 版构建；为空则是 CPU 版构建。
    """
    code = ("import torch; print(torch.__version__); "
            "print(int(torch.cuda.is_available())); print(torch.version.cuda)")
    try:
        r = subprocess.run([str(python_path), "-c", code],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, creationflags=_flags())
        if r.returncode != 0:
            return False, False, "", None
        lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
        lines = lines[-3:]                    # 忽略可能混入的警告输出
        if len(lines) < 3:
            return False, False, "", None
        ver, cuda_ok, torch_cuda = lines[0], lines[1] == "1", (lines[2] or None)
        if torch_cuda in ("None", "none"):
            torch_cuda = None
        return True, cuda_ok, ver, torch_cuda
    except Exception:
        return False, False, "", None


def _run_cmd(cmd, log, should_stop, env=None) -> bool:
    """运行命令并实时把输出转发给 log；支持 should_stop 取消。"""
    run_env = None
    if env:
        run_env = dict(os.environ)
        run_env.update(env)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=_flags(), env=run_env)
    except Exception as e:  # noqa: BLE001
        log(f"启动命令失败：{e}")
        return False
    while True:
        if should_stop and should_stop():
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            log("已取消。")
            return False
        line = proc.stdout.readline()
        if line:
            log(line.rstrip())
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.05)
    return proc.returncode == 0


def _pip_install(py, package_args, log, should_stop, indexes, extra_index=None):
    """按索引列表依次尝试 pip 安装，返回 (是否成功, 错误信息)。"""
    for idx in indexes:
        if should_stop and should_stop():
            return False, "已取消"
        cmd = [py, "-m", "pip", "install", "--timeout", "120", "--retries", "5"]
        if idx:
            cmd += ["-i", idx]
        if extra_index:
            cmd += ["--extra-index-url", extra_index]
        cmd += list(package_args)
        log(f"pip 安装 {' '.join(package_args)}（源：{idx or '默认官方源'}）…")
        if _run_cmd(cmd, log, should_stop):
            return True, ""
    return False, "所有安装源均失败，请检查网络后重试"


def setup_nnunet_env(log_cb=None, should_stop=None):
    """一键自动配置 nnUNet 运行环境：建 venv + 装 torch + 装 nnunetv2。

    **安装前先检测本机是否已有可用的 nnUNet 环境，有则直接复用、不安装。**

    返回 (是否成功, python.exe路径, 错误信息)。
    """
    log_lines = deque(maxlen=30)

    def log(msg):
        log_lines.append(msg)
        if log_cb:
            log_cb(msg)

    def fail(msg):
        tail = "\n".join(list(log_lines)[-12:])
        return False, "", f"{msg}\n\n最近日志：\n{tail}"

    # 0) 安装前检测：本机（含软件附近/conda）已有可用的 nnUNet 环境就直接复用
    existing = find_python_with_nnunet()
    if existing:
        py, mode = existing
        log(f"检测到本机已有可用的 nnUNet 环境：{py}（{mode}）")
        log("无需安装，将直接使用该环境。")
        return True, py, ""

    base = find_base_python()
    log(f"使用基础 Python：{base}")
    if should_stop and should_stop():
        return False, "", "已取消"

    # 打包版：软件自带嵌入式 Python（runtime），直接向它安装，无需创建虚拟环境
    if RUNTIME_PYTHON.exists() and os.path.abspath(base) == os.path.abspath(str(RUNTIME_PYTHON)):
        py = str(RUNTIME_PYTHON)
        log("检测到软件自带的 Python 运行环境，将直接安装（无需虚拟环境）。")
    else:
        # 使用独立临时目录，减少下载中断/杀毒软件导致的文件占用问题
        pip_tmp = ENV_DIR / ".pip_tmp"
        try:
            pip_tmp.mkdir(parents=True, exist_ok=True)
        except OSError:
            pip_tmp = None
        run_env = {"TMP": str(pip_tmp), "TEMP": str(pip_tmp)} if pip_tmp else None

        # 1) 创建虚拟环境
        log("正在创建 Python 虚拟环境…")
        if not _run_cmd([base, "-m", "venv", str(ENV_DIR)], log, should_stop):
            return fail("创建虚拟环境失败")
        py = str(ENV_PYTHON)
        if not Path(py).exists():
            return fail(f"虚拟环境创建不完整：{py}")

    # 2) 升级 pip
    log("正在升级 pip…")
    ok, err = _pip_install(py, ["--upgrade", "pip"], log, should_stop,
                           indexes=[PIP_MIRROR] if PIP_MIRROR else [""])
    if not ok:
        return fail(f"pip 升级失败：{err}")

    # 3) 安装 PyTorch：驱动支持 CUDA 12+ 或无法确定时才装 GPU 版（装完验证），否则装 CPU 版
    gpu = has_nvidia_gpu()
    cuda_ver = driver_cuda_version() if gpu else None
    if gpu and cuda_ver is not None and cuda_ver < 12.0:
        log(f"检测到 NVIDIA 显卡，但驱动较旧（CUDA {cuda_ver:.1f}），新版 PyTorch 需要 CUDA 12+。")
        log("已改装 CPU 版 PyTorch（可用但较慢）。更新显卡驱动（GeForce 551+）后重跑“自动配置环境”即可启用 GPU。")
        ok, err = _pip_install(py, ["torch"], log, should_stop,
                               indexes=TORCH_CPU_INDEXES)
    elif gpu:
        installed, cuda_ok, ver, torch_cuda = _env_torch_status(py)
        if cuda_ver is None:
            log("检测到 NVIDIA 显卡，但未能读取驱动 CUDA 版本。")
        else:
            log(f"检测到 NVIDIA 显卡（驱动支持 CUDA {cuda_ver:.1f}）。")
        if installed and cuda_ok:
            log("已存在可用的 GPU 版 PyTorch。")
        elif installed and torch_cuda:
            # torch 是 GPU 版构建，但此刻 GPU 不可用——再确认一次，避免瞬时误判
            if not _env_cuda_available(py):
                log(f"检测到 GPU 版 PyTorch（{ver}），但 GPU 暂不可用（驱动或运行库不兼容）。")
                log("保留现有安装，将使用 CPU 推理（较慢）；如仍想重装，可删除 .nnunet_env 后重试。")
            else:
                log("已存在可用的 GPU 版 PyTorch。")
        elif installed:
            log(f"检测到已有 PyTorch，但为 CPU 版（{ver}）。")
            log("正在卸载旧版并安装 GPU 版 PyTorch（需重新下载，请耐心等待）…")
            _run_cmd([py, "-m", "pip", "uninstall", "-y", "torch"], log, should_stop)
            log("正在安装 GPU 版 PyTorch（下载约 2.4GB，请耐心等待）…")
            ok, err = _pip_install(py, ["torch"], log, should_stop,
                                   indexes=TORCH_GPU_INDEXES,
                                   extra_index=PIP_MIRROR if PIP_MIRROR else None)
            if not ok:
                return fail(f"PyTorch 安装失败：{err}")
            if not _env_cuda_available(py):
                log("警告：GPU 版 PyTorch 已安装，但 GPU 仍不可用（驱动或运行库不兼容），将使用 CPU 推理（较慢）。")
        else:
            log("正在安装 GPU 版 PyTorch（下载约 2.4GB，请耐心等待）…")
            ok, err = _pip_install(py, ["torch"], log, should_stop,
                                   indexes=TORCH_GPU_INDEXES,
                                   extra_index=PIP_MIRROR if PIP_MIRROR else None)
            if not ok:
                return fail(f"PyTorch 安装失败：{err}")
            if not _env_cuda_available(py):
                log("警告：GPU 版 PyTorch 已安装，但 GPU 仍不可用（驱动或运行库不兼容），将使用 CPU 推理（较慢）。")
    else:
        log("未检测到 NVIDIA 显卡，安装 CPU 版 PyTorch（下载较小）…")
        ok, err = _pip_install(py, ["torch"], log, should_stop,
                               indexes=TORCH_CPU_INDEXES)
    if not ok:
        return fail(f"PyTorch 安装失败：{err}")

    # 4) 安装 nnUNet V2
    log("正在安装 nnUNet V2…")
    ok, err = _pip_install(py, ["nnunetv2"], log, should_stop,
                           indexes=[PIP_MIRROR] if PIP_MIRROR else [""])
    if not ok:
        return fail(f"nnUNet V2 安装失败：{err}")

    log("安装完成！")
    return True, py, ""
