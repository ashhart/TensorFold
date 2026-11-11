"""Build and audit a downloadable Windows 5090 preview on a CPU-only hosted runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from artifact_gate import qualify, stable_snapshot, secret_scan
from runtime_overlay import create
from runtime_sources import digest, fetch
from privacy_review import Scanner

HERE = Path(__file__).resolve().parent
STAGE = "plan"
OPTIONS = {"optimize": "safe", "strip": True, "qwen_prefill_max_rows": 8192,
           "sm": "120", "target": "x86_64-windows-gnu", "version_fixture": "1.2.3"}


def package_failure(path):
    """Report only source-defined protocol tokens, never unchecked log text."""
    text = path.read_text(errors="replace")
    phases = ("windows-inputs", "tree-check", "archive-create", "archive-check")
    seen = [phase for phase in phases if "TF_PACKAGE_PHASE=" + phase in text.splitlines()]
    refusals = []
    for filename in ("tensorfold.exe", "tf_cublaslt_bridge.dll", "cublasLt64_13.dll", "tfgrammar.dll"):
        for reason in ("PrivateBuildPath", "BadElf", "UnsupportedElf", "UnsupportedExtendedElf",
                       "UnsupportedCompressedSection", "BadFatbin", "UnsupportedImageEncoding",
                       "FileTooBig", "OutOfMemory", "AccessDenied"):
            if f"release privacy: refused {filename} ({reason})" in text.splitlines():
                refusals.append({"file": filename, "reason": reason})
    print(json.dumps({"stage": "package", "completed_protocol_markers": seen,
                      "controlled_refusals": refusals}), flush=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tree(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("toolchain symlink requires review")
        if path.is_file():
            result.append({"path": path.relative_to(root).as_posix(),
                           "bytes": path.stat().st_size, "sha256": digest(path)})
    return result


def verify_publication(repo, directory):
    report = read(directory / "ARTIFACT-REVIEW.json")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    actual_tree = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                     "--untracked-files=all"], text=True).strip()
    if (dirty or report["status"] != "PASSED" or report["source_commit"] != actual
            or report["source_tree"] != actual_tree):
        raise ValueError("publication source binding changed")
    rows = stable_snapshot(directory)
    expected = report["publication_files"]
    actual_files = [row for row in rows if row["path"] != "ARTIFACT-REVIEW.json"]
    by_path = {row["path"]: row for row in actual_files}
    if (report.get("archive_sha256") is not None
            and by_path["tensorfold-1.2.3-windows-x86_64.zip"]["sha256"] != report["archive_sha256"]):
        raise ValueError("publication archive changed after inspection")
    for row in report.get("public_sidecars", []):
        if by_path.get(row["path"]) != row:
            raise ValueError("publication build provenance changed after inspection")
    if actual_files != expected or {row["path"] for row in rows} != {
        "tensorfold-1.2.3-windows-x86_64.zip", "tensorfold-1.2.3-windows-x86_64.zip.sha256",
        "BUILD-PROVENANCE.json", "ARTIFACT-REVIEW.json",
    }:
        raise ValueError("publication output changed after inspection")
    print(json.dumps({"status": "PASSED", "files": len(rows), "source_commit": actual}))



def diagnostic_tail(stage, path, context, work, repo, zig_root, gitleaks):
    audit = work / ("diagnostic-" + stage + "-private")
    audit.mkdir()
    text = "\n".join(path.read_text(errors="replace").splitlines()[-80:])
    roots = [(str(work), "$BUILD"), (str(repo), "$SOURCE"), (str(zig_root), "$ZIG")]
    for key in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "VCToolsInstallDir", "WindowsSdkDir"):
        if os.environ.get(key):
            roots.append((os.environ[key].rstrip("\\/"), "$" + key.upper()))
    for root, label in sorted(roots, key=lambda row: len(row[0]), reverse=True):
        text = text.replace(root, label).replace(root.replace("\\", "/"), label)
    text = "\n".join(line if line.isascii() else "[Non-ASCII diagnostic line withheld]" for line in text.splitlines())
    scanner = Scanner(context, read(HERE / "artifact-policy.json"), audit)
    scanner.scan(text.encode(), "normalized-diagnostic")
    receipt = scanner.close()
    if receipt["findings"]:
        print(json.dumps({"stage": stage, "diagnostic_withheld": "context-finding",
                          "finding_count": len(receipt["findings"])}))
        return
    secret_scan(gitleaks, audit / "readable-spans.txt", audit)
    print(json.dumps({"stage": stage, "reviewed_diagnostic_tail": text}))


def main():
    global STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--verify-publication", type=Path)
    args = parser.parse_args()
    repo = HERE.parents[2]
    if args.verify_publication:
        verify_publication(repo, args.verify_publication)
        return
    if not args.execute:
        print(json.dumps({"mode": "plan", "gpu_calls": 0, "uploads": 0, "options": OPTIONS,
                          "source_builds": ["native-fatbins", "bridge", "grammar", "privacy-checker"]}))
        return
    if os.name != "nt" or not args.work:
        raise ValueError("execute needs Windows x64 and an empty external work directory")
    work = args.work.resolve()
    if work.exists() or repo in work.parents:
        raise ValueError("preview work must be new and outside the source checkout")
    raw_context = os.environ.pop("TF_PREVIEW_CONTEXT_DIGESTS", "")
    context = json.loads(raw_context)
    if (set(context) != {"schema", "normalization", "sha256"} or context["schema"] != 1
            or context["normalization"] != "casefold-alnum-v1" or not context["sha256"]
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in context["sha256"])):
        raise ValueError("hashed private context is missing or malformed")
    source_revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("GITHUB_SHA", source_revision) != source_revision:
        raise ValueError("hosted workflow and checkout revisions differ")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=all"], text=True).strip():
        raise ValueError("preview source must be clean")
    work.mkdir()
    context_path = work / "context-digests.private.json"
    write(context_path, context)
    logs = work / "commands-private"
    logs.mkdir()
    (work / "tmp").mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(work / "tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["ZIG_GLOBAL_CACHE_DIR"] = str(work / "cache-global")
    os.environ["ZIG_LOCAL_CACHE_DIR"] = str(work / "cache-local")
    zig = Path(shutil.which("zig")).resolve()
    zig_root = zig.parent
    commands = []

    def run(name, argv, timeout=5400):
        global STAGE
        STAGE = name
        print(json.dumps({"status": "BUILD_STAGE", "stage": name}), flush=True)
        path = logs / (name + ".log")
        with path.open("wb") as log:
            result = subprocess.run([str(value) for value in argv], cwd=repo,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        normalized = []
        for value in argv:
            value = str(value)
            for root, label in [(str(work), "$BUILD"), (str(repo), "$SOURCE"), (str(zig_root), "$ZIG")]:
                value = value.replace(root, label).replace(root.replace("\\", "/"), label)
            normalized.append(value)
        commands.append({"name": name, "argv": normalized, "exit_code": result.returncode,
                         "log_sha256": digest(path)})
        write(work / "commands.private.json", commands)
        if result.returncode:
            selected_log = path
            if name == "bridge" and (work / "bridge/private-audit/compile-link.log").is_file():
                selected_log = work / "bridge/private-audit/compile-link.log"
            if name == "package":
                package_failure(selected_log)
            try:
                diagnostic_tail(name, selected_log, context_path, work, repo, zig_root, gitleaks)
            except Exception as diagnostic_error:
                result = {"stage": name, "diagnostic_withheld": type(diagnostic_error).__name__}
                receipt = work / ("diagnostic-" + name + "-private") / "gitleaks-status.json"
                if receipt.is_file():
                    result["scanner_status"] = read(receipt)
                print(json.dumps(result))
            raise ValueError("preview command failed: " + name)

    zig_before = tree(zig_root)
    STAGE = "public-source-inputs"
    xgrammar, gitleaks, runtime_manifest = fetch(work)
    native_out, grammar_out, checks = work / "native-out", work / "grammar-out", work / "checks"
    run("privacy-checker", [zig, "build", "compile-release-paths-host", "-j2", "--prefix", checks])
    path_checker = checks / "bin/check-release-paths.exe"
    run("bridge", [sys.executable, HERE / "produce_bridge.py", "--execute", "--work", work / "bridge",
                   "--repo-root", repo, "--source-revision", source_revision])
    bridge = read(work / "bridge/output-manifest.json")
    run("vendor-path-preflight", [path_checker, repo, work / "bridge/bundle/cublasLt64_13.dll"])
    nvcc = work / "bridge/cuda/bin/nvcc.exe"
    run("native", [zig, "build", "fatbins", "native", "-Dtarget=x86_64-windows-gnu",
                   "-Doptimize=safe", "-Dstrip=true", "-Dversion=1.2.3", "-Dqwen-prefill-max-rows=8192",
                   "-Dsm=120", "-Dnvcc=" + str(nvcc), "-j2", "--prefix", native_out])
    native = native_out / "native/bin/tensorfold.exe"
    if not native.is_file():
        raise ValueError("native serving executable was not produced")
    run("grammar", [zig, "build", "grammar", "-Dtarget=x86_64-windows-gnu",
                    "-Dxgrammar=" + str(xgrammar), "-j2", "--prefix", grammar_out])
    grammar = grammar_out / "native/lib/tfgrammar.dll"
    if not grammar.is_file():
        raise ValueError("Windows grammar library was not produced")
    source_path = work / "SOURCE.json"
    run("source-manifest", [sys.executable, repo / "tools/release/windows_manifest.py",
                            "--binary", native, "--fatbins", native_out / "fatbin", "--output", source_path])
    source = read(source_path)
    source["build_options"] = OPTIONS
    source["grammar_binary_sha256"] = digest(grammar)
    source["bridge_output_manifest_sha256"] = digest(work / "bridge/output-manifest.json")
    source["grammar_sources"] = [
        {key: row[key] for key in ("name", "repository", "commit", "archive_sha256")}
        for row in read(runtime_manifest)["sources"]
    ]
    source["qualification"] = (
        "CPU-built Windows 5090 preview; model load, tools, cache equality and performance on this head are unrun"
    )
    write(source_path, source)
    STAGE = "runtime-overlay"
    overlay = create(repo, work, xgrammar, grammar, bridge)
    archives = work / "archives"
    archives.mkdir()
    git_exe = Path(shutil.which("git")).resolve()
    bash = git_exe.parent.parent / "bin/bash.exe"
    if not bash.is_file():
        bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not bash.is_file():
        raise ValueError("Git Bash packaging host is missing")
    arguments = [
        bash, repo / "tools/release/package.sh", "1.2.3", "windows-x86_64", native,
        repo / "LICENSE", repo / "NOTICE", repo / "packaging/RUNTIME.md", repo / "LICENSES",
        repo / "THIRD_PARTY_NOTICES.md", archives, "--path-checker=" + path_checker.as_posix(),
        "--build-root=" + repo.as_posix(), "--overlay=" + overlay.as_posix(),
        "--source-manifest=" + source_path.as_posix(),
    ]
    run("package", [value.as_posix() if isinstance(value, Path) else value for value in arguments])
    archive = archives / "tensorfold-1.2.3-windows-x86_64.zip"
    if not archive.is_file():
        raise ValueError("shared packager did not produce the preview")
    if tree(zig_root) != zig_before:
        raise ValueError("Zig toolchain changed during the build")
    STAGE = "build-provenance"
    provenance = {
        "schema": 1, "source_commit": source_revision, "source_tree": source["source_tree"],
        "build_options": OPTIONS, "commands": commands, "bridge": bridge,
        "bridge_source_inputs": read(work / "bridge/private-audit/source-inputs.json"),
        "bridge_host_inputs": read(work / "bridge/private-audit/host-inputs.json"),
        "bridge_sdk_inputs": read(work / "bridge/private-audit/sdk-inputs.json"),
        "bridge_compiler_command": read(work / "bridge/private-audit/compile-link.command.json"),
        "grammar_source_inputs": read(runtime_manifest), "zig_inputs": zig_before,
        "python": {"version": sys.version, "executable_sha256": digest(Path(sys.executable))},
        "qualification": source["qualification"],
    }
    public = work / "publish"
    public.mkdir()
    provenance_path = public / "BUILD-PROVENANCE.json"
    write(provenance_path, provenance)
    STAGE = "artifact-review"
    report = qualify(repo, work, archive, gitleaks, context_path, [provenance_path])
    # Smoke only the already reviewed CLI, without model or CUDA initialization.
    run("windows-cli-smoke", ["pwsh", "-NoProfile", "-File", repo / "tools/release/smoke-windows.ps1",
                              archive, "1.2.3"], timeout=300)
    shutil.copyfile(archive, public / archive.name)
    companion = archive.with_suffix(archive.suffix + ".sha256")
    shutil.copyfile(companion, public / companion.name)
    report["publication_files"] = stable_snapshot(public)
    report["CPU_CLI_smoke"] = "passed; no model or GPU operations"
    write(public / "ARTIFACT-REVIEW.json", report)
    verify_publication(repo, public)
    context_path.unlink()
    print(json.dumps({"status": "PREVIEW_READY", "gpu_calls": 0, "uploads": 0,
                      "source_commit": source_revision, "archive_sha256": report["archive_sha256"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAILED", "stage": STAGE, "category": type(error).__name__, "uploads": 0}), file=sys.stderr)
        raise SystemExit(1)
