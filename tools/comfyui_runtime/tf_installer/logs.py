from contextlib import contextmanager
from pathlib import Path
import tempfile

from .pins import InstallError


MAX_LOG_BYTES = 1024 * 1024
MAX_LOG_FILES = 16


def preserve_logs(root, work):
    files = [path for path in sorted(work.glob("*.log")) if not path.is_symlink() and path.is_file()][:MAX_LOG_FILES]
    if not files:
        return None
    logs = root / "logs"
    if logs.is_symlink():
        raise InstallError("The managed logs directory must not be a symlink.")
    logs.mkdir(mode=0o700, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="failed-", dir=logs))
    marker = b"\n[log truncated; first and last portions retained]\n"
    for path in files:
        with path.open("rb") as source:
            data = source.read(MAX_LOG_BYTES + 1)
            if len(data) > MAX_LOG_BYTES:
                half = (MAX_LOG_BYTES - len(marker)) // 2
                source.seek(-half, 2)
                data = data[:half] + marker + source.read(half)
        (destination / path.name).write_bytes(data)
    return destination


@contextmanager
def build_workspace(root, log):
    with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as directory:
        work = Path(directory)
        try:
            yield work
        except BaseException:
            try:
                saved = preserve_logs(root, work)
                if saved is not None:
                    log(f"Installation logs saved to {saved}.")
            except (OSError, InstallError) as error:
                log(f"Could not preserve installation logs: {error}")
            raise
