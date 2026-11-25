import re
import shutil
import subprocess

from .pins import InstallError

QUALIFIED_SM = 121
QUALIFIED = "MOSS ComfyUI generation is qualified on GB10 (sm_121) only"


def nvidia_smi(timeout=15):
    """Every GPU's compute capability as the NVIDIA driver reports it, one line a GPU."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        raise InstallError("nvidia-smi was not found")
    try:
        result = subprocess.run([tool, "--query-gpu=compute_cap", "--format=csv,noheader"], stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise InstallError(f"nvidia-smi did not answer within {timeout} s") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip() or result.stdout.decode("utf-8", "replace").strip()
        raise InstallError(f"nvidia-smi exited {result.returncode}: {detail[:200]}")
    return result.stdout.decode("utf-8", "replace")


def require_gb10(query=nvidia_smi):
    """Refuse every GPU but sm_121 by name; a capability that cannot be read is refused, never assumed."""
    try:
        lines = [line.strip() for line in query().splitlines() if line.strip()]
    except (InstallError, OSError, UnicodeError) as error:
        raise InstallError(f"{QUALIFIED}; this GPU's compute capability could not be read ({error}).") from error
    if not lines:
        raise InstallError(f"{QUALIFIED}; this GPU's compute capability could not be read (nvidia-smi listed no GPU).")
    for index, line in enumerate(lines):
        match = re.fullmatch(r"([0-9]{1,2})\.([0-9])", line)
        if match is None:
            raise InstallError(f"{QUALIFIED}; this GPU's compute capability could not be read ({line[:40]!r}).")
        sm = int(match[1]) * 10 + int(match[2])
        if sm != QUALIFIED_SM:
            which = "this GPU" if len(lines) == 1 else f"GPU {index} of {len(lines)}"
            raise InstallError(f"{QUALIFIED}; {which} is sm_{sm}.")
    return f"sm_{QUALIFIED_SM}"
