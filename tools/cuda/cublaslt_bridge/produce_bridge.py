#!/usr/bin/env python3
"""Build the source-bound bridge on Windows; default plan does not compile, load a DLL or use a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
OFFICIAL = "https://developer.download.nvidia.com/compute/cuda/redist/"
FLAGS = [
    "-std=c++17", "-O3", "--fmad=false", "--ftz=false", "--shared",
    "-gencode=arch=compute_120a,code=sm_120a", "--cudart=static", "-Xcompiler=/EHsc", "-Xcompiler=/MT",
]
PRODUCER_STAGE = "plan"
EXPORTS = {
    "tf_lt_abi_version", "tf_lt_create", "tf_lt_destroy",
    "tf_lt_get_requirements", "tf_lt_last_error", "tf_lt_run", "tf_lt_tune_last",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tree(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink input is not permitted")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def check_source(source, manifest):
    actual = tree(source)
    expected = [{k: row[k] for k in ("path", "bytes", "sha256")} for row in manifest["files"]]
    if actual != sorted(expected, key=lambda row: row["path"]):
        raise ValueError("source bytes or source closure differ from the reviewed manifest")
    names = {row["path"] for row in expected}
    for row in manifest["files"]:
        includes = re.findall(r'#include\s+"([^"]+)"', (source / row["path"]).read_text())
        if includes != row["quoted_includes"] or not set(includes) <= names:
            raise ValueError("source include closure changed")


def extract(archive, destination):
    destination.mkdir()
    with zipfile.ZipFile(archive) as package:
        roots, seen = set(), set()
        entries = package.infolist()
        if len(entries) > 50000 or sum(item.file_size for item in entries) > (2 << 30):
            raise ValueError("ZIP extraction limit reached")
        for item in package.infolist():
            name = item.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            raw_parts = name.rstrip("/").split("/")
            reserved = r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?"
            if any(p in ("", ".", "..") or p.rstrip(" .") != p
                   or re.fullmatch(reserved, p) for p in raw_parts):
                raise ValueError("unsafe Windows ZIP path")
            if not parts or name.startswith("/") or any(p in ("..", ".") or ":" in p for p in parts):
                raise ValueError("unsafe ZIP member")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP symlink is not permitted")
            if not item.is_dir() and name.casefold() in seen:
                raise ValueError("duplicate ZIP member")
            seen.add(name.casefold())
            roots.add(parts[0])
        if len(roots) != 1:
            raise ValueError("expected one official archive root")
        package.extractall(destination)
    result = destination / roots.pop()
    if not result.is_dir():
        raise ValueError("missing official archive root")
    return result


def merge_toolkit(package, toolkit):
    # Keep each complete package separately; only compiler layout directories are merged.
    for directory in ("bin", "include", "lib", "nvvm"):
        source = package / directory
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            target = toolkit / directory / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != digest(item):
                    raise ValueError("non-identical toolkit merge collision")
            else:
                shutil.copyfile(item, target)


def normal(value, roots):
    value = str(value)
    for path, label in sorted(roots.items(), key=lambda row: len(row[0]), reverse=True):
        value = value.replace(path, label)
        value = value.replace(path.replace("\\", "/"), label)
    return value


def run(argv, *, cwd, env, audit, name, roots, accepted=(0,), timeout=1800):
    global PRODUCER_STAGE
    PRODUCER_STAGE = name
    with (audit / (name + ".log")).open("wb") as stream:
        process = subprocess.run([str(arg) for arg in argv], cwd=cwd, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    row = {"argv": [normal(arg, roots) for arg in argv], "cwd": normal(cwd, roots),
           "exit_code": process.returncode, "log_sha256": digest(audit / (name + ".log"))}
    write_json(audit / (name + ".command.json"), row)
    if process.returncode not in accepted:
        raise RuntimeError(name + " failed; inspect its private log")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--expected-host-manifest", type=Path,
                        help="Replay requires the exact previously reviewed host input manifest")
    args = parser.parse_args()
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    lock = json.loads((HERE / "inputs.lock.json").read_text())
    source = HERE / "source"
    check_source(source, manifest)
    if set(manifest["exports"]) != EXPORTS:
        raise ValueError("reviewed ABI export set changed")
    for item in lock["assets"]:
        if not item["url"].startswith(OFFICIAL) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("unreviewed input location or digest")
    if not args.execute:
        print(json.dumps({"mode": "plan", "gpu_calls": 0, "uploads": 0,
                          "source_manifest_sha256": digest(HERE / "source-manifest.json"),
                          "input_lock_sha256": digest(HERE / "inputs.lock.json"),
                          "archives": len(lock["assets"]), "exports": sorted(EXPORTS)}))
        return
    if os.name != "nt":
        raise ValueError("execute requires Windows x64 and an initialized MSVC x64 environment")
    if not args.work or not args.repo_root or not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_revision or ""):
        raise ValueError("execute requires empty work directory, repository root and exact source revision")
    work, repo = args.work.resolve(), args.repo_root.resolve()
    if work.exists():
        raise ValueError("work directory must not exist")
    if work == repo or repo in work.parents or HERE in work.parents:
        raise ValueError("build outputs must stay outside the reviewed source checkout")
    # A separate fixed build directory avoids personal checkout paths in compiler-generated metadata.
    if repo not in HERE.parents:
        raise ValueError("the reviewed producer must be inside the exact source checkout")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision.lower() != args.source_revision.lower():
        raise ValueError("source checkout revision does not match the requested binding")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=all"], text=True).strip():
        raise ValueError("source checkout must be clean")
    producer_before = tree(HERE)
    work.mkdir(parents=True)
    audit, bundle = work / "private-audit", work / "bundle"
    audit.mkdir()
    bundle.mkdir()
    env = dict(os.environ)
    for key in list(env):
        if key.upper() in ("CL", "_CL_", "LINK", "_LINK_", "NVCC_PREPEND_FLAGS",
                           "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_PATH", "CUDA_HOME") or key.upper().startswith("CUDA_PATH_V"):
            del env[key]
    needed = ("VCToolsInstallDir", "WindowsSdkDir", "WindowsSDKVersion", "VCToolsVersion")
    if any(not env.get(key) for key in needed):
        raise ValueError("initialize the exact MSVC x64 toolset before execute")
    msvc = Path(env["VCToolsInstallDir"]).resolve()
    winsdk = Path(env["WindowsSdkDir"]).resolve()
    sdkversion = env["WindowsSDKVersion"].strip("\\/")
    if not re.fullmatch(r"\d+\.\d+\.\d+", env["VCToolsVersion"].strip("\\/")):
        raise ValueError("unexpected MSVC version")
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", sdkversion):
        raise ValueError("unexpected Windows SDK version")
    hostbin = msvc / "bin" / "Hostx64" / "x64"
    if not all((hostbin / name).is_file() for name in ("cl.exe", "link.exe", "dumpbin.exe")):
        raise ValueError("missing selected x64 host tools")
    host_roots = {
        "msvc": msvc,
        "winsdk-include": winsdk / "Include" / sdkversion,
        "winsdk-lib": winsdk / "Lib" / sdkversion,
        "winsdk-bin": winsdk / "bin" / sdkversion / "x64",
    }
    search_paths = {}
    for variable in ("INCLUDE", "LIB", "LIBPATH"):
        values = []
        for raw in env.get(variable, "").split(os.pathsep):
            if not raw:
                continue
            path = Path(raw).resolve()
            if not path.is_dir():
                raise ValueError("missing compiler search directory")
            label = next((name for name, root in host_roots.items()
                          if path == root or root in path.parents), None)
            if label is None:
                label = "search-" + str(len(host_roots))
                host_roots[label] = path
            values.append({"root": label, "path": path.relative_to(host_roots[label]).as_posix()})
        search_paths[variable] = values
    host_manifest = {"schema": "tensorfold.bridge.host-inputs.v1",
                     "msvc_version": env["VCToolsVersion"].strip("\\/"),
                     "windows_sdk_version": sdkversion,
                     "search_paths": search_paths, "roots": {}}
    for name, path in host_roots.items():
        if not path.is_dir():
            raise ValueError("selected host input tree is missing: " + name)
        host_manifest["roots"][name] = tree(path)
    if args.expected_host_manifest:
        if host_manifest != json.loads(args.expected_host_manifest.read_text()):
            raise ValueError("host compiler/SDK bytes differ from the replay pin")
    write_json(audit / "host-inputs.json", host_manifest)
    roots = {str(work): "$BUILD", str(repo): "$REPOSITORY", str(HERE): "$PRODUCER",
             str(msvc): "$MSVC", str(winsdk): "$WINSDK"}
    (work / "downloads").mkdir()
    (work / "packages").mkdir()
    toolkit = work / "cuda"
    toolkit.mkdir()
    input_rows, packages = [], {}
    metadata = work / "downloads" / "redistrib_13.5.0.json"
    urllib.request.urlretrieve(lock["official_metadata_url"], metadata)
    if digest(metadata) != lock["official_metadata_sha256"]:
        raise ValueError("official metadata checksum changed")
    official = json.loads(metadata.read_text())
    for item in lock["assets"]:
        member = official[item["component"]]
        target = member["windows-x86_64"]
        if (member["version"] != item["version"] or target["sha256"] != item["sha256"]
                or int(target["size"]) != item["bytes"]
                or OFFICIAL + target["relative_path"] != item["url"]):
            raise ValueError("input lock differs from the pinned official metadata")
        archive = work / "downloads" / (item["component"] + ".zip")
        urllib.request.urlretrieve(item["url"], archive)
        if archive.stat().st_size != item["bytes"] or digest(archive) != item["sha256"]:
            raise ValueError("official archive size or checksum mismatch: " + item["component"])
        package = extract(archive, work / "packages" / item["component"])
        packages[item["component"]] = package
        files = tree(package)
        input_rows.append({**item, "archive_members": files})
        if item["component"] != "libcublas":
            merge_toolkit(package, toolkit)
    write_json(audit / "sdk-inputs.json", {"schema": "tensorfold.bridge.sdk-inputs.v1", "assets": input_rows})
    toolkit_before = tree(toolkit)
    write_json(audit / "merged-toolkit.json", toolkit_before)
    src = work / "source"
    shutil.copytree(source, src)
    check_source(src, manifest)
    cublas = packages["libcublas"]
    nvcc, cuobjdump = toolkit / "bin" / "nvcc.exe", toolkit / "bin" / "cuobjdump.exe"
    driverlib = toolkit / "lib" / "x64" / "cuda.lib"
    cublaslib = cublas / "lib" / "x64" / "cublasLt.lib"
    for path in (nvcc, cuobjdump, driverlib, cublaslib, cublas / "include" / "cublasLt.h"):
        if not path.is_file():
            raise ValueError("pinned package layout is incompatible with this recipe")
    env["PATH"] = str(toolkit / "bin") + os.pathsep + str(hostbin) + os.pathsep + env["PATH"]
    temporary = work / "tmp"
    temporary.mkdir()
    env["TEMP"] = str(temporary)
    env["TMP"] = str(temporary)
    env["CUDA_PATH"] = str(toolkit)
    env["CUDA_HOME"] = str(toolkit)
    run([nvcc, "--version"], cwd=src, env=env, audit=audit, name="nvcc-version", roots=roots)
    version_text = (audit / "nvcc-version.log").read_text(errors="replace")
    if "V13.5.37" not in version_text:
        raise ValueError("wrong NVCC version")
    run([hostbin / "cl.exe", "/Bv"], cwd=src, env=env, audit=audit,
        name="msvc-version", roots=roots, accepted=(0, 2))
    compiled = work / "compiled"
    compiled.mkdir()
    dll = compiled / "tf_cublaslt_bridge.dll"
    command = [nvcc, *FLAGS, "--compiler-bindir", hostbin, "-I", cublas / "include",
               "bridge.cu", cublaslib, driverlib, "-o", dll]
    run(command, cwd=src, env=env, audit=audit, name="compile-link", roots=roots)
    if not dll.is_file():
        raise ValueError("bridge DLL was not produced")
    for option in ("exports", "imports", "headers", "dependents"):
        run([hostbin / "dumpbin.exe", "/" + option, dll], cwd=src, env=env,
            audit=audit, name="pe-" + option, roots=roots)
    exported = set(re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(tf_lt_\w+)\s*$",
                             (audit / "pe-exports.log").read_text(errors="replace"), re.M))
    if exported != EXPORTS:
        raise ValueError("DLL does not preserve the exact seven-function ABI")
    decoded = audit / "decoded"
    decoded.mkdir()
    run([cuobjdump, "--list-elf", dll], cwd=decoded, env=env, audit=audit,
        name="cuda-list-elf", roots=roots)
    run([cuobjdump, "--extract-elf", "all", dll], cwd=decoded, env=env, audit=audit,
        name="cuda-extract-elf", roots=roots)
    if not list(decoded.glob("*")):
        raise ValueError("CUDA images were not decoded")
    run([cuobjdump, "--list-ptx", dll], cwd=decoded, env=env, audit=audit,
        name="cuda-list-ptx", roots=roots)
    run([cuobjdump, "--dump-elf", dll], cwd=decoded, env=env, audit=audit,
        name="cuda-elf-metadata", roots=roots)
    write_json(audit / "decoded-images.json", tree(decoded))
    vendor = cublas / "bin" / "x64" / "cublasLt64_13.dll"
    if not vendor.is_file() or digest(vendor) != "ca478e1ea932f883ef4fc284f8f33d910079df4e77f9a2ec38d9b4387795f96f":
        raise ValueError("runtime SDK member differs from the reviewed cuBLASLt input")
    shutil.copyfile(dll, bundle / dll.name)
    shutil.copyfile(vendor, bundle / vendor.name)
    for component, package in packages.items():
        for item in package.rglob("*"):
            if item.is_file() and re.search(r"license|notice|copying|eula", item.name, re.I):
                destination = bundle / "licenses" / "NVIDIA" / component / item.relative_to(package)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, destination)
    for filename in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        item = repo / filename
        if not item.is_file():
            raise ValueError("repository legal input is missing")
        destination = bundle / "licenses" / "TensorFold" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)
    for component in packages:
        if not list((bundle / "licenses" / "NVIDIA" / component).rglob("*")):
            raise ValueError("official package legal files were not retained")
    check_source(src, manifest)
    if tree(toolkit) != toolkit_before:
        raise ValueError("compiler toolkit inputs changed during the build")
    for name, path in host_roots.items():
        if tree(path) != host_manifest["roots"][name]:
            raise ValueError("host input tree changed during the build")
    for row in input_rows:
        if tree(packages[row["component"]]) != row["archive_members"]:
            raise ValueError("an extracted SDK package changed during the build")
    if tree(HERE) != producer_before:
        raise ValueError("producer inputs changed during the build")
    write_json(audit / "producer-inputs.json", producer_before)
    write_json(audit / "source-inputs.json", manifest)
    output = {
        "schema": "tensorfold.bridge.output.v1", "source_revision": args.source_revision.lower(),
        "source_manifest_sha256": digest(HERE / "source-manifest.json"),
        "input_lock_sha256": digest(HERE / "inputs.lock.json"),
        "producer_sha256": digest(Path(__file__)),
        "python": {"version": sys.version, "executable_sha256": digest(sys.executable)},
        "runner_image": {key: os.environ.get(key) for key in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")},
        "host_input_manifest_sha256": digest(audit / "host-inputs.json"),
        "sdk_input_manifest_sha256": digest(audit / "sdk-inputs.json"),
        "compile_command_sha256": digest(audit / "compile-link.command.json"),
        "exports": sorted(exported), "outputs": tree(bundle),
        "build_and_static_inspection_only": True, "dll_loads": 0, "gpu_calls": 0,
        "performance_qualification": "unrun", "privacy_clearance": False,
        "historical_active_dll_match_claim": False,
        "cudart_link_mode": "static",
        "msvc_runtime_link_mode": "MT",
        "historical_linkage_difference": "CUDA13.5 Windows defaults hybrid; this fresh port explicitly selects static cudart and CRT",
        "source_tree": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip(),
    }
    write_json(work / "output-manifest.json", output)
    print(json.dumps({"status": "BUILT_PENDING_PACKAGE_AUDIT", "gpu_calls": 0, "uploads": 0,
                      "output_manifest_sha256": digest(work / "output-manifest.json"),
                      "bridge_sha256": digest(dll)}))



if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Keep compiler output and downloaded bytes private; never dump arbitrary exception text.
        print(json.dumps({"status": "FAILED", "stage": PRODUCER_STAGE, "category": type(error).__name__,
                          "outputs_may_be_incomplete": True, "uploads": 0}), file=sys.stderr)
        raise SystemExit(1)
