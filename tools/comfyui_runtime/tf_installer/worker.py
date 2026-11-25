import os

from .device import nvidia_smi, require_gb10
from .install import manifest_file, read_status, runtime_root


def start_worker(comfy_root, arguments, gpu=nvidia_smi, execute=None):
    """Replace this process with the installed MOSS worker; any GPU but GB10's sm_121 is refused before it starts."""
    require_gb10(gpu)
    root = runtime_root(comfy_root)
    manifest = read_status(root)
    worker = manifest_file(root, manifest["artifacts"]["moss"]["path"], executable=True)
    (execute or os.execv)(str(worker), [str(worker), *arguments])
