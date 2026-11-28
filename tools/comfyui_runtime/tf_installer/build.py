import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .archive import extract_archive, verify_hash
from .network import download
from .pins import InstallError, MAX_SOURCE_BYTES, MAX_SOURCE_EXTRACTED_BYTES, SOURCE_REVISION, SOURCE_PATCH_SHA256, SOURCE_ROOT, SOURCE_SHA256, SOURCE_URL, ZIG_VERSION


def host_platform(backend):
    architecture = platform.machine().lower()
    aliases = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
    if platform.system() != "Linux" or architecture not in aliases or backend != "cuda":
        raise InstallError("Media companions currently support Linux x86_64/aarch64 with CUDA only. No files were downloaded or built.")
    return "linux-" + aliases[architecture]


def executable(value, name):
    selected = value or shutil.which(name)
    if not selected:
        raise InstallError(f"{name} was not found; supply its executable path.")
    path = Path(selected).expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise InstallError(f"{name} is not an executable file.")
    return str(path)


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        raise InstallError("Could not stop the installation subprocess group after SIGKILL.") from error


def run_command(arguments, cwd, log, timeout=3600, expected=0, environment=None):
    log = Path(log)
    environment = None if environment is None else {**os.environ, **environment}
    with log.open("wb") as output:
        process = subprocess.Popen(arguments, cwd=cwd, env=environment, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        failed = False
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline or log.stat().st_size > 64 * 1024 * 1024:
                    raise InstallError(f"{Path(arguments[0]).name} exceeded its time or log-size limit.")
                time.sleep(0.1)
            if process.returncode != expected:
                with log.open("rb") as source:
                    source.seek(max(0, log.stat().st_size - 8192))
                    detail = source.read().decode("utf-8", "replace")
                raise InstallError(f"{Path(arguments[0]).name} exited {process.returncode}:\n{detail}")
        except BaseException:
            failed = True
            raise
        finally:
            try:
                stop_process_group(process)
            except (OSError, InstallError) as error:
                if not failed:
                    raise
                print(f"TensorFold subprocess cleanup failed: {error}", file=sys.stderr)
    with log.open("rb") as source:
        text = source.read(65536).decode("utf-8", "replace").strip()
    return text + ("\n[output truncated]" if log.stat().st_size > 65536 else "")


def check_toolchain(zig, nvcc, work, run=run_command):
    zig = executable(zig, "zig")
    nvcc = executable(nvcc, "nvcc")
    actual = run([zig, "version"], work, work / "zig-version.log", timeout=15)
    if actual != ZIG_VERSION:
        raise InstallError(f"Pinned media source requires Zig {ZIG_VERSION}, found {actual!r}.")
    run([nvcc, "--version"], work, work / "nvcc-version.log", timeout=15)
    return zig, nvcc


def prepare_source(work, checkout=None, archive=None, fetch=download, run=run_command):
    if checkout and archive:
        raise InstallError("Select only one of --source-checkout and --source-archive.")
    if checkout:
        path = Path(checkout).expanduser().resolve(strict=True)
        git = executable(None, "git")
        prefix = [git, "--no-replace-objects", "-c", "core.fsmonitor=false", "-C", str(path)]
        revision = run(prefix + ["rev-parse", "HEAD"], work, work / "source-head.log", timeout=15)
        dirty = run(prefix + ["status", "--porcelain=v1", "--untracked-files=all"], work, work / "source-status.log", timeout=30)
        if revision != SOURCE_REVISION or dirty:
            raise InstallError(f"Source checkout must be clean and exactly {SOURCE_REVISION}.")
        source_archive = work / "source.tar"
        run(prefix + ["archive", "--format=tar", f"--prefix={SOURCE_ROOT}/", f"--output={source_archive}", SOURCE_REVISION], work, work / "source-archive.log", timeout=120)
    else:
        source_archive = Path(archive).expanduser().resolve(strict=True) if archive else fetch(SOURCE_URL, work / "source.tar.gz", MAX_SOURCE_BYTES)
        if not source_archive.is_file() or source_archive.stat().st_size > MAX_SOURCE_BYTES:
            raise InstallError("Pinned source archive is missing or too large.")
        verify_hash(source_archive, SOURCE_SHA256)
    source, _ = extract_archive(source_archive, work / "source-unpacked", SOURCE_ROOT, MAX_SOURCE_EXTRACTED_BYTES)
    if (source / ".zig-version").read_text(encoding="ascii").strip() != ZIG_VERSION:
        raise InstallError("Source .zig-version differs from the reviewed toolchain pin.")
    return source


def apply_media_patch(source, work, run=run_command):
    source_patch = Path(__file__).with_name("moss-comfy.patch").resolve(strict=True)
    verify_hash(source_patch, SOURCE_PATCH_SHA256)
    git = executable(None, "git")
    environment = {"GIT_CEILING_DIRECTORIES": str(source.parent.resolve())}
    run([git, "apply", "--check", str(source_patch)], source, work / "media-patch-check.log", timeout=30, environment=environment)
    run([git, "apply", str(source_patch)], source, work / "media-patch.log", timeout=30, environment=environment)


def build_media(source, work, candidate, zig, nvcc, run=run_command):
    apply_media_patch(source, work, run)
    prefix = work / "built"
    options = ["-j2", f"-Dnvcc={nvcc}", "-Dsm=121", "--prefix", str(prefix), "--cache-dir", str(work / "zig-cache")]
    environment = {"ZIG_GLOBAL_CACHE_DIR": str(work / "zig-global-cache")}
    run([zig, "build", "moss-image-cuda", *options], source, work / "moss-build.log", timeout=7200, environment=environment)
    origin = prefix / "bin/tf-moss-image"
    target = candidate / "media/bin/tf-moss-image"
    if origin.is_symlink() or not origin.is_file() or origin.stat().st_size == 0:
        raise InstallError("Build did not produce a regular MOSS companion.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)
    target.chmod(0o755)
    usage = run([str(target)], work, work / "moss-probe.log", timeout=20, expected=2)
    if "tf-moss-image TRANSFORMER_DIR CASE OUT_DIR" not in usage:
        raise InstallError("MOSS companion does not expose the reviewed invocation contract.")
    return {"moss": target}
