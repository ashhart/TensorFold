"""What this computer is: OS, chip, memory, GPU, and which TensorFold backend it can run."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

GIB = 1024**3
CUDA_FLOOR = (8, 9)          # the frozen Python line and the aarch64 native archive: Ada and newer
# the native engine's Linux x86_64 archive adds Ampere (8.6: RTX 30, A6000, A10, A40), tested on an RTX 3090 (1.0.5)
CUDA_FLOOR_X86_NATIVE = (8, 6)


def cuda_floor(arch: str) -> tuple[int, int]:
    return CUDA_FLOOR_X86_NATIVE if arch in ("x86_64", "AMD64", "amd64") else CUDA_FLOOR


@dataclass
class Gpu:
    name: str
    memory_bytes: int | None = None
    compute_capability: str | None = None


@dataclass
class Machine:
    os: str                      # "macOS", "Linux", "Windows"
    os_version: str
    arch: str
    chip: str
    ram_bytes: int
    ram_available_bytes: int
    unified_memory: bool
    gpus: list[Gpu] = field(default_factory=list)
    backend: str = "none"        # "mlx", "cuda" or "none"
    backend_note: str = ""
    tensorfold_version: str | None = None
    hostname: str = ""
    gpu_ceiling_bytes: int | None = None   # Apple Silicon: the most the GPU may use, which caps every budget

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str]) -> str:
    return _query(cmd)[0]


def _query(cmd: list[str]) -> tuple[str, str | None]:
    """A command's output, or the reason there is none: a timeout, an error exit, a binary that cannot run."""

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except subprocess.TimeoutExpired:
        return "", f"{cmd[0]} did not answer within 5 s"
    except OSError as exc:
        return "", f"{cmd[0]}: {exc}"
    if out.returncode != 0:
        said = (out.stderr or out.stdout).strip().splitlines()
        return "", f"{cmd[0]} exit code {out.returncode}" + (f": {said[-1].strip()}" if said else "")
    return out.stdout.strip(), None


def _mac_chip() -> str:
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return brand or "Apple Silicon"


def _nvidia_gpus() -> tuple[list[Gpu], str | None]:
    """The cards nvidia-smi lists, and the reason it listed none when it gave no answer at all."""

    if not shutil.which("nvidia-smi"):
        return [], None
    out, err = _query(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                mem = int(float(parts[1])) * 1024 * 1024
            except ValueError:
                mem = None
            gpus.append(Gpu(parts[0], mem, parts[2]))
    return gpus, err


def _product_name() -> str | None:
    """The machine's own name on Linux (DMI), such as GX10 or DGX Spark, when the firmware says it."""

    try:
        name = Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
    except OSError:
        return None
    return name if name and name.lower() not in ("default string", "to be filled by o.e.m.", "system product name") else None


def _gpu_ceiling() -> int | None:
    """macOS's recommended working set for the GPU (what `iogpu.wired_limit_mb` sets), through MLX when present."""

    try:
        import mlx.core as mx

        return int(mx.device_info()["max_recommended_working_set_size"])
    except Exception:  # noqa: BLE001 - no MLX in this environment: the verdict uses the RAM fractions alone
        return None


def _tensorfold_version() -> str | None:
    """The native engine's version when its binary is installed, else the frozen Python line's (`0.6.6 (Python)`)."""

    from . import native

    binary = native.find_binary()
    if binary:
        return native.version_of(binary)
    try:
        return importlib.metadata.version("tensorfold") + " (Python)"
    except importlib.metadata.PackageNotFoundError:
        return None


def detect() -> Machine:
    """This machine, with what moves read afresh: the GPU ceiling (a sysctl can change it while TensorFold Web runs),
    the engine's version, the memory available now."""

    m = _detect()
    m.tensorfold_version = _tensorfold_version()      # a stat per call: an upgrade shows without a restart
    m.ram_available_bytes = int(psutil.virtual_memory().available)
    if m.backend == "mlx":
        m.gpu_ceiling_bytes = _gpu_ceiling()
    return m


_cached: Machine | None = None


def _detect() -> Machine:
    """The machine, probed once and kept; a probe nvidia-smi left unanswered (a driver reset, a card busy loading) is
    not kept, so the next call asks again instead of saying "no GPU" until TensorFold Web restarts."""

    global _cached
    if _cached is None:
        m, settled = _probe()
        if not settled:
            return m
        _cached = m
    return _cached


def _reset() -> None:
    global _cached
    _cached = None


def _probe() -> tuple[Machine, bool]:
    system = platform.system()
    names = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}
    vm = psutil.virtual_memory()
    m = Machine(os=names.get(system, system), os_version=platform.mac_ver()[0] if system == "Darwin" else
                platform.release(), arch=platform.machine(), chip=platform.processor() or platform.machine(),
                ram_bytes=int(vm.total), ram_available_bytes=int(vm.available), unified_memory=False,
                tensorfold_version=_tensorfold_version(), hostname=platform.node())
    if system == "Darwin" and m.arch == "arm64":
        m.chip = _mac_chip()
        m.unified_memory = True
        m.backend = "mlx"
        m.backend_note = "Apple Silicon: TensorFold runs on the GPU through MLX."
        m.gpu_ceiling_bytes = _gpu_ceiling()
        return m, True
    m.gpus, smi_error = _nvidia_gpus()
    if m.gpus:
        best = m.gpus[0]
        product = _product_name()
        m.chip = best.name + (f" ({product})" if product and product not in best.name else "")
        try:
            cc = tuple(int(x) for x in (best.compute_capability or "0.0").split("."))
        except ValueError:
            cc = (0, 0)
        floor = cuda_floor(m.arch)
        if cc >= floor:
            m.backend = "cuda"
            m.backend_note = f"{best.name}: TensorFold's CUDA engine runs here (compute capability {best.compute_capability})."
            # the GB10 (DGX Spark and its relatives) shares its RAM with the GPU; nvidia-smi gives no memory for it
            if "GB10" in best.name and (best.memory_bytes is None or best.memory_bytes >= 100 * GIB):
                m.unified_memory = True
        else:
            m.backend_note = (f"{best.name} is compute capability {best.compute_capability}; TensorFold needs {floor[0]}.{floor[1]} or newer "
                              + ("(RTX 30 and newer, A6000, A10, A40)." if floor == CUDA_FLOOR_X86_NATIVE else
                                 "(RTX 40, RTX 50, RTX PRO, DGX Spark)."))
    elif smi_error:
        m.backend_note = f"nvidia-smi gave no answer ({smi_error}); it is asked again at the next request."
        return m, False
    else:
        m.backend_note = "No Apple Silicon GPU and no NVIDIA GPU found. You can still drive a TensorFold running on another machine."
    return m, True
