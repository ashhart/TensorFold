"""Keep Python oracles and native MLX on the same upstream-resolved dependencies."""
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dependencies():
    return json.loads((ROOT / "native/dependencies.json").read_text())


def require_mlx():
    expected = dependencies()["python"]
    if expected["mlx"] != expected["mlx-metal"]:
        raise RuntimeError("native/dependencies.json must pin mlx and mlx-metal to the same version")
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"Native parity requires {expected}; got {versions}. "
                           "Align .venv with native/dependencies.json before generating oracles.")
    return versions


def check_requirements(project, installed, pins):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    installed = {canonicalize_name(name): version for name, version in installed.items()}
    pins = {canonicalize_name(name): version for name, version in pins.items()}
    failures = []
    for entry in runtime_requirements(project):
        req = Requirement(entry)
        if req.marker and not req.marker.evaluate():
            continue
        name = canonicalize_name(req.name)
        if req.url:
            failures.append(f"{entry}: direct-source dependencies require explicit native compatibility review")
            continue
        version = installed.get(name)
        if version is None or not req.specifier.contains(version):
            failures.append(f"{entry}: installed {version or 'missing'}")
        pinned = pins.get(name)
        if pinned and not req.specifier.contains(pinned):
            failures.append(f"{entry}: native pin {pinned} is incompatible; run sync-upstream to resolve "
                            "upstream requirements and rebuild the native pairing")
    if failures:
        raise RuntimeError("Dependency sync required:\n" + "\n".join(failures))


def runtime_requirements(project):
    return (project["project"]["dependencies"] +
            project["project"].get("optional-dependencies", {}).get("vision", []))


def vision_legacy_pixel_limits():
    from transformers import Qwen2VLImageProcessor
    processor = Qwen2VLImageProcessor(min_pixels=65536, max_pixels=16777216, patch_size=16, merge_size=2)
    return hasattr(processor, "min_pixels") and hasattr(processor, "max_pixels")


def jpeg_version():
    from PIL import features
    version = features.version_feature("libjpeg_turbo")
    if not version:
        raise RuntimeError("Native JPEG parity requires Pillow built with libjpeg-turbo")
    return version


def native_library_version(prefix):
    import ctypes

    class String(ctypes.Structure):
        _fields_ = [("ctx", ctypes.c_void_p)]

    library = ctypes.CDLL(str((prefix / "lib/libmlxc.dylib").resolve()))
    library.mlx_string_new.restype = String
    library.mlx_string_free.argtypes = [String]
    library.mlx_version.argtypes = [ctypes.POINTER(String)]
    library.mlx_version.restype = ctypes.c_int
    library.mlx_string_data.argtypes = [String]
    library.mlx_string_data.restype = ctypes.c_char_p
    value = library.mlx_string_new()
    try:
        if library.mlx_version(ctypes.byref(value)) != 0:
            raise RuntimeError("Installed MLX-C could not query its linked MLX runtime")
        return library.mlx_string_data(value).decode()
    finally:
        library.mlx_string_free(value)


def resolved_dependencies(previous, versions, native_version):
    if versions["mlx"] != versions["mlx-metal"]:
        raise RuntimeError(f"The resolved MLX/Metal versions disagree: {versions}")
    resolved = dict(previous)
    if versions["mlx"] != previous["python"]["mlx"]:
        resolved["mlx_revision"] = "v" + versions["mlx"]
    resolved["python"] = versions
    resolved["rebuild_mlx"] = native_version != versions["mlx"] or versions["mlx"] != previous["python"]["mlx"]
    return resolved


def main():
    import argparse
    import re
    import subprocess
    import tomllib
    parser = argparse.ArgumentParser(description="Check native pins, installed packages and upstream dependency constraints without loading models")
    parser.add_argument("--upstream-ref")
    parser.add_argument("--mlx-prefix", type=Path, default=ROOT / "build/mlx")
    parser.add_argument("--jpeg-prefix", type=Path, default=ROOT / "build/jpeg")
    parser.add_argument("--resolve", action="store_true", help="Install upstream requirements and prepare the matching native dependency record")
    args = parser.parse_args()
    if args.upstream_ref:
        manifest = subprocess.check_output(["git", "show", f"{args.upstream_ref}:pyproject.toml"], cwd=ROOT, text=True)
    else:
        manifest = (ROOT / "pyproject.toml").read_text()
    project = tomllib.loads(manifest)
    from packaging.requirements import Requirement
    import sys
    if args.resolve:
        requirements = runtime_requirements(project) + project.get("project", {}).get("optional-dependencies", {}).get("test", [])
        if any(Requirement(entry).url for entry in requirements):
            raise RuntimeError("Direct-source upstream dependencies require explicit review")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--editable", f"{ROOT}[test,vision]", *requirements], check=True)
        versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm", "transformers", "Pillow")}
    else:
        versions = require_mlx()
    installed = {}
    for entry in runtime_requirements(project):
        name = Requirement(entry).name
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    check_requirements(project, installed, versions if args.resolve else dependencies()["python"])
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    config = args.mlx_prefix / "share/cmake/MLX/MLXConfigVersion.cmake"
    match = re.search(r'set\(PACKAGE_VERSION "([^"]+)"\)', config.read_text()) if config.exists() else None
    jpeg_config = args.jpeg_prefix / "lib/pkgconfig/libturbojpeg.pc"
    jpeg_match = re.search(r"^Version: (.+)$", jpeg_config.read_text(), re.M) if jpeg_config.exists() else None
    jpeg = jpeg_version()
    receipt = subprocess.run([str(ROOT / ".zig-toolchain/zig"), "run", "tools/native_install.zig",
                              "--global-cache-dir", str(ROOT / ".zig-cache/global"), "--",
                              "--mlx-prefix", str(args.mlx_prefix), "--jpeg-prefix", str(args.jpeg_prefix)],
                             cwd=ROOT, text=True, capture_output=True)
    if args.resolve:
        resolved = resolved_dependencies(dependencies(), versions, match[1] if match else None)
        resolved["vision_legacy_pixel_limits"] = vision_legacy_pixel_limits()
        resolved["jpeg_version"] = jpeg
        resolved["rebuild_jpeg"] = not jpeg_match or jpeg_match[1] != jpeg or not (args.jpeg_prefix / "lib/libturbojpeg.a").is_file()
        resolved["rebuild_mlx"] |= not (args.mlx_prefix / "share/cmake/MLXC/MLXCConfigVersion.cmake").is_file()
        if receipt.returncode:
            print("Native installation needs rebuilding: " + receipt.stderr)
            resolved["rebuild_mlx"] = resolved["rebuild_jpeg"] = True
        (ROOT / "build/native-dependencies-resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
        print(f"Resolved upstream requirements: {versions}")
        return
    if receipt.returncode:
        raise RuntimeError("Native install verification failed; rerun setup:\n" + receipt.stderr)
    print(receipt.stderr, end="")
    actual_version = native_library_version(args.mlx_prefix)
    if actual_version != versions["mlx"]:
        raise RuntimeError(f"Loaded native MLX reports {actual_version}, expected {versions['mlx']}")
    if not match or match[1] != versions["mlx"]:
        raise RuntimeError(f"Rebuild native MLX at {dependencies()['mlx_revision']}: {config} must report {versions['mlx']}")
    if vision_legacy_pixel_limits() != dependencies()["vision_legacy_pixel_limits"]:
        raise RuntimeError("Vision processor behavior differs from the native dependency record; sync dependencies")
    if not jpeg_match or jpeg_match[1] != jpeg or dependencies().get("jpeg_version") != jpeg:
        raise RuntimeError(f"Rebuild native libjpeg-turbo at {jpeg}: decoder must match upstream Pillow")
    print(f"PASS: native MLX, Python pins and {'upstream' if args.upstream_ref else 'checkout'} requirements agree: {versions}")


if __name__ == "__main__":
    main()
