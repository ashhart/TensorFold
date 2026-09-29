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
    for entry in project["project"]["dependencies"]:
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
        requirements = project["project"]["dependencies"] + project.get("project", {}).get("optional-dependencies", {}).get("test", [])
        if any(Requirement(entry).url for entry in requirements):
            raise RuntimeError("Direct-source upstream dependencies require explicit review")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--editable", f"{ROOT}[test]", *requirements], check=True)
        versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mlx-lm")}
    else:
        versions = require_mlx()
    installed = {}
    for entry in project["project"]["dependencies"]:
        name = Requirement(entry).name
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    check_requirements(project, installed, versions if args.resolve else dependencies()["python"])
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    config = args.mlx_prefix / "share/cmake/MLX/MLXConfigVersion.cmake"
    match = re.search(r'set\(PACKAGE_VERSION "([^"]+)"\)', config.read_text()) if config.exists() else None
    if args.resolve:
        resolved = resolved_dependencies(dependencies(), versions, match[1] if match else None)
        resolved["rebuild_mlx"] |= not (args.mlx_prefix / "share/cmake/MLXC/MLXCConfigVersion.cmake").is_file()
        (ROOT / "build/native-dependencies-resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
        print(f"Resolved upstream requirements: {versions}")
        return
    if not match or match[1] != versions["mlx"]:
        raise RuntimeError(f"Rebuild native MLX at {dependencies()['mlx_revision']}: {config} must report {versions['mlx']}")
    print(f"PASS: native MLX, Python pins and {'upstream' if args.upstream_ref else 'checkout'} requirements agree: {versions}")


if __name__ == "__main__":
    main()
