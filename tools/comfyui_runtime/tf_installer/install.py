import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

from .archive import extract_archive, file_hash, relative_name, verify_engine, verify_hash
from .build import build_media, check_toolchain, host_platform, prepare_source, run_command
from .device import nvidia_smi, require_gb10
from .logs import build_workspace
from .network import download, fetch_release, get_release, version_tuple
from .pins import InstallError, SOURCE_REVISION, SOURCE_PATCH_SHA256


def runtime_root(comfy_root):
    comfy = Path(comfy_root).expanduser().resolve(strict=True)
    if not (comfy / "main.py").is_file() or not (comfy / "folder_paths.py").is_file():
        raise InstallError("--comfy-root must name a ComfyUI checkout containing main.py and folder_paths.py.")
    root = comfy / "models/tensorfold"
    if root.is_symlink():
        raise InstallError("TensorFold's managed runtime directory must not be a symlink.")
    return root


def manifest_file(root, value, executable=False):
    parts = relative_name(value)
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise InstallError("Runtime artifact paths must not contain symlinks.")
    if not path.is_file() or executable and not path.stat().st_mode & stat.S_IXUSR:
        raise InstallError(f"Runtime artifact is missing or not executable: {value}")
    return path


def read_status(root):
    path = root / "runtime.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise InstallError("No valid runtime.json is installed.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data["schema"]) is not int or data["schema"] != 1 or data["backend"] != "cuda" or data["source_revision"] != SOURCE_REVISION or data["platform"] not in ("linux-x86_64", "linux-aarch64"):
            raise InstallError("Runtime manifest does not match this integration's compatibility contract.")
        version_tuple(data["engine"]["version"])
        for item in (data["engine"], data["artifacts"]["moss"]):
            verify_hash(manifest_file(root, item["path"], executable=True), item["sha256"])
        media = manifest_file(root, data["artifacts"]["moss"]["path"]).parent.parent
        license_files(media / "licenses")
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise InstallError("Malformed runtime manifest.") from error
    return data


@contextmanager
def install_lock(root):
    if fcntl is None:
        raise InstallError("Linux file locking is required for installation.")
    fd = os.open(root / ".install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise InstallError("Unsafe installation lock.")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InstallError("Another TensorFold installation is active.") from error
        yield
    finally:
        os.close(fd)


def activate(root, candidate, manifest, replace=os.replace):
    installs = root / "installs"
    if installs.is_symlink():
        raise InstallError("Managed installs directory must not be a symlink.")
    installs.mkdir(exist_ok=True)
    name = f"{manifest['engine']['version']}-{SOURCE_REVISION[:12]}-{uuid.uuid4().hex}"
    destination = installs / name
    for item in (manifest["engine"], manifest["artifacts"]["moss"]):
        item["path"] = f"installs/{name}/{item['path']}"
    pending = root / f".runtime-{uuid.uuid4().hex}.json"
    moved = False
    try:
        os.rename(candidate, destination)
        moved = True
        with pending.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        replace(pending, root / "runtime.json")
    except BaseException:
        pending.unlink(missing_ok=True)
        if moved:
            shutil.rmtree(destination)
        raise
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return manifest


def license_files(source):
    if source.is_symlink() or not source.is_dir():
        raise InstallError("Media licenses directory is missing or unsafe; use install to rebuild the pinned media.")
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        origin = source / name
        if origin.is_symlink() or not origin.is_file():
            raise InstallError("Media license files are missing or unsafe; use install to rebuild the pinned media.")
    license_tree = source / "LICENSES"
    if license_tree.is_symlink() or not license_tree.is_dir() or any(path.is_symlink() for path in license_tree.rglob("*")):
        raise InstallError("Media license tree is missing or unsafe; use install to rebuild the pinned media.")


def copy_media_licenses(source, candidate):
    license_files(source)
    target = candidate / "media/licenses"
    target.mkdir(parents=True)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(source / name, target / name)
    shutil.copytree(source / "LICENSES", target / "LICENSES")


def reuse_media(root, previous, candidate):
    entry = previous["artifacts"]["moss"]
    origin = manifest_file(root, entry["path"], executable=True)
    target = candidate / "media/bin/tf-moss-image"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)
    verify_hash(target, entry["sha256"])
    target.chmod(0o755)
    copy_media_licenses(origin.parent.parent / "licenses", candidate)
    return {"moss": target}


def install(comfy_root, backend="cuda", zig=None, nvcc=None, source_checkout=None, source_archive=None, action="install", fetch=download, run=run_command, builder=build_media, log=print, gpu=nvidia_smi):
    platform = host_platform(backend)
    try:
        require_gb10(gpu)
    except InstallError as error:
        raise InstallError(f"{error} No files were downloaded or built.") from error
    root = runtime_root(comfy_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with install_lock(root), build_workspace(root, log) as work:
        old = None
        if (root / "runtime.json").exists() or (root / "runtime.json").is_symlink():
            try:
                old = read_status(root)
            except (InstallError, OSError):
                if action == "update":
                    raise InstallError("Installed runtime failed integrity checks; use install to build a verified replacement.")
                log("Existing runtime failed verification; preparing a replacement before changing its manifest.")
        if action == "update" and old is None:
            raise InstallError("No runtime is installed; use install first.")
        release = get_release(work, platform, fetch)
        if old and version_tuple(release["version"]) < version_tuple(old["engine"]["version"]):
            raise InstallError("Refusing to downgrade the active standalone engine.")
        media_current = old and old.get("source_patch_sha256") == SOURCE_PATCH_SHA256
        if media_current and old["engine"]["version"] == release["version"] and old["platform"] == platform:
            log(f"Already current: standalone engine {release['version']}; media source {SOURCE_REVISION}.")
            return old
        log(f"Installing standalone engine {release['version']}; media companions stay pinned to {SOURCE_REVISION}.")
        version, archive, digest = fetch_release(work, release, fetch)
        verify_hash(archive, digest)
        engine, files = extract_archive(archive, work / "engine-unpacked", f"tensorfold-{version}-{platform}")
        verify_engine(engine, files, version)
        actual = run([str(engine / "bin/tensorfold-native"), "--version"], work, work / "engine-version.log", timeout=20)
        if actual != f"tensorfold-native {version}":
            raise InstallError("Downloaded native engine reports the wrong version.")
        candidate = work / "candidate"
        candidate.mkdir()
        shutil.move(str(engine), candidate / "engine")
        if media_current and old["platform"] == platform:
            log("Reusing verified media companions from the unchanged source revision; no compiler is needed.")
            artifacts = reuse_media(root, old, candidate)
        else:
            zig, nvcc = check_toolchain(zig, nvcc, work, run)
            source = prepare_source(work, source_checkout, source_archive, fetch, run)
            log("Building the pinned MOSS CUDA companion; the current runtime remains active until all probes pass.")
            artifacts = builder(source, work, candidate, zig, nvcc, run)
            copy_media_licenses(source, candidate)
        native = candidate / "engine/bin/tensorfold-native"
        manifest = {"schema": 1, "source_revision": SOURCE_REVISION, "source_patch_sha256": SOURCE_PATCH_SHA256, "platform": platform, "backend": backend,
                    "engine": {"version": version, "path": "engine/bin/tensorfold-native", "sha256": file_hash(native)},
                    "artifacts": {name: {"path": str(path.relative_to(candidate)), "sha256": file_hash(path)} for name, path in artifacts.items()}}
        result = activate(root, candidate, manifest)
        log("Runtime activated. GPU generation and performance have not been verified by installation.")
        return result
