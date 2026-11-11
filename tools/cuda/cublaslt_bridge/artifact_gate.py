"""Review the exact preview ZIP, source bindings, PE/CUDA structures, legal inputs and readable content before upload."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
from cuda_review import MAGIC, iter_containers, zstd_frames
from pe_review import review, cuda_wrappers
from privacy_review import Scanner, RULES
from runtime_sources import digest

HERE = Path(__file__).resolve().parent
MAX_FILE = 1 << 30


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_snapshot(directory):
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("nonregular publication input")
        if path.is_file():
            result.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
                           "sha256": digest(path)})
    return result


def verify_source(repo, source):
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                     "--untracked-files=all"], text=True).strip()
    if (dirty or source.get("source_dirty") is not False or source.get("host_only") is not False
            or source.get("source_commit") != actual or source.get("source_tree") != tree):
        raise ValueError("artifact is not bound to this clean complete source checkout")
    return actual


def unpack(archive, destination):
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if len(entries) > 10000 or sum(entry.file_size for entry in entries) > 2 << 30:
            raise ValueError("preview archive extraction limit")
        names = set()
        for entry in entries:
            name = entry.filename.rstrip("/")
            mode = entry.external_attr >> 16
            expected_mode = (stat.S_IFDIR | 0o755) if entry.is_dir() else (stat.S_IFREG | 0o644)
            if (not name or name.startswith("/") or "\\" in name or ":" in name
                    or any(part in ("", ".", "..") for part in name.split("/"))
                    or name.casefold() in names or entry.create_system != 3
                    or entry.date_time != (1980, 1, 1, 0, 0, 0) or mode != expected_mode
                    or entry.extra or entry.comment):
                raise ValueError("unexpected preview archive member or metadata")
            names.add(name.casefold())
        if package.comment:
            raise ValueError("preview archive comment")
        package.extractall(destination)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("preview archive must contain one package root")
    return roots[0]


def allowed_import(name, bundled, policy):
    folded = name.casefold()
    providers = {item.casefold() for item in policy["nvidia_driver_provider"]["names"]}
    return (folded in bundled or folded in providers or folded in {item.casefold() for item in policy["system_imports"]}
            or re.fullmatch(r"api-ms-win-(?:core|crt|security)-[a-z0-9-]+\.dll", folded) is not None)


def validate_pe(data, path, vendor, policy):
    result = review(data)
    if result["dll"] != path.name.lower().endswith(".dll"):
        raise ValueError("PE image kind does not match the packaged name")
    standard = {".text", ".rdata", ".data", ".bss", ".pdata", ".xdata", ".idata", ".edata",
                ".reloc", ".tls", ".CRT", ".rsrc", ".nv_fatb", ".nvFatBi", ".buildid", ".fptable"}
    for section in result["sections"]:
        if section["name"] not in standard:
            permitted = any(row["section"] == section["name"] and row["bytes"] == section["bytes"]
                            and row["sha256"] == section["sha256"]
                            for row in policy["resolved_opaque_vendor_scopes"])
            if not vendor or not permitted:
                raise ValueError("unreviewed PE section")
    for section in result["sections"]:
        if section["name"] == ".fptable":
            if section["virtual_size"] > section["bytes"] or section["virtual_size"] % 8:
                raise ValueError("MSVC function-pointer table geometry")
            for offset in range(0, section["virtual_size"], 8):
                value = int.from_bytes(data[section["offset"] + offset:section["offset"] + offset + 8], "little")
                if value and not any(row["name"] == ".text" and row["rva"] <= value - result["image_base"]
                                     < row["rva"] + row["virtual_size"] for row in result["sections"]):
                    raise ValueError("MSVC function table points outside reviewed code")
    for item in result["debug"]:
        if item["type"] == 2:
            raise ValueError("CodeView/PDB records are excluded from the downloadable preview")
        if item["type"] not in (12, 13, 16, 20):
            raise ValueError("unreviewed PE debug-record type")
    if any(item["route"][0] not in (16, 24) for item in result["resources"]):
        raise ValueError("unreviewed resource payload type")
    return result



def require_catalog(repo, rows):
    text = (repo / "zig/build/cuda_kernels.zig").read_text()
    start = "pub const kernels = [_]Kernel{"
    if text.count(start) != 1:
        raise ValueError("CUDA source catalog syntax requires review")
    body = text.split(start, 1)[1].split("\n};", 1)[0]
    expected = re.findall(r'\.name\s*=\s*"([A-Za-z0-9_]+)"', body)
    actual = [row["file"] for row in rows]
    if (not expected or len(set(expected)) != len(expected) or len(set(actual)) != len(actual)
            or set(actual) != {name + ".fatbin" for name in expected}):
        raise ValueError("fatbin inputs do not equal the complete source catalog")
    return expected


def require_bridge_cuda(functions, image_count, expected):
    if not image_count or not expected or set(functions) != set(expected):
        raise ValueError("fresh bridge CUDA payload or transform coverage is incomplete")


def secret_scan(executable, corpus, audit):
    config = audit / "gitleaks.toml"
    config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
    ignore = audit / "empty-ignore"
    ignore.mkdir()
    with (audit / "gitleaks.log").open("wb") as log:
        run = subprocess.run([
            str(executable), "dir", str(corpus), "--config", str(config),
            "--gitleaks-ignore-path", str(ignore), "--ignore-gitleaks-allow", "--no-banner", "--redact",
            "--max-target-megabytes", "0", "--max-decode-depth", "4", "--max-archive-depth", "4",
            "--timeout", "900", "--report-format", "json", "--report-path", str(audit / "gitleaks.json"),
        ], cwd=audit, stdout=log, stderr=subprocess.STDOUT, timeout=960,
            env={key: value for key, value in os.environ.items() if not key.startswith("GITLEAKS_")})
    log_text = (audit / "gitleaks.log").read_text(errors="replace")
    log_text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log_text)
    scanned = re.search(r"\bscanned\s+~?(\d+)\s+bytes\b", log_text)
    findings = load(audit / "gitleaks.json")
    status = {"exit_code": run.returncode, "findings": len(findings),
              "scanned_bytes": int(scanned[1]) if scanned else None,
              "expected_bytes": corpus.stat().st_size,
              "execution_error": bool(re.search(r"\b(?:skipped|skipping|error|fatal)\b", log_text, re.I))}
    (audit / "gitleaks-status.json").write_text(json.dumps(status) + "\n")
    if (run.returncode != 0 or findings or scanned is None
            or status["scanned_bytes"] != status["expected_bytes"] or status["execution_error"]):
        raise ValueError("unresolved secret finding, skipped bytes or secret-scanner failure")
    result = {"exit_code": run.returncode, "scanner_sha256": digest(executable),
              "scanned_bytes": int(scanned[1]), "corpus_sha256": digest(corpus),
              "default_rules": True, "findings": 0}
    (audit / "gitleaks-execution.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def resolve_encoded_findings(findings, frames, policy):
    unresolved, resolved = [], []
    for finding in findings:
        expected = next((row for row in policy["encoded_email_coincidences"]
                         if all(row.get(key) == finding.get(key) for key in
                                ("origin", "offset", "encoding", "category", "match_sha256"))), None)
        frame = next((row for row in frames
                      if row["origin"] == finding["origin"] and row["offset"] <= finding["offset"]
                      and finding["offset"] + finding.get("match_bytes", 1) <= row["offset"] + row["bytes"]), None)
        if (expected and frame and expected["vendor_sha256"] == policy["vendor_runtime"]["sha256"]
                and finding["category"] == "email" and finding["encoding"] == "ascii"):
            resolved.append({"finding": finding, "frame": frame,
                             "resolution": expected["required_resolution"]})
        else:
            unresolved.append(finding)
    return unresolved, resolved

def qualify(repo, work, archive, gitleaks, context, public_sidecars=()):
    policy = load(HERE / "artifact-policy.json")
    audit = work / "artifact-review-private"
    audit.mkdir()
    decoded = audit / "unpacked"
    decoded.mkdir()
    package = unpack(archive, decoded)
    source = load(package / "SOURCE.json")
    require_catalog(repo, source["fatbin_inputs"])
    sidecar_snapshot = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in public_sidecars
    ]
    revision = verify_source(repo, source)
    bridge = load(work / "bridge/output-manifest.json")
    if bridge["source_revision"] != revision or bridge["source_tree"] != source["source_tree"]:
        raise ValueError("bridge and native build use different source snapshots")
    if bridge["cudart_link_mode"] != "static" or bridge["msvc_runtime_link_mode"] != "MT":
        raise ValueError("bridge linkage policy changed")
    if source.get("build_options") != {
        "optimize": "safe", "strip": True, "qwen_prefill_max_rows": 8192,
        "sm": "120", "target": "x86_64-windows-gnu", "version_fixture": "1.2.3",
    }:
        raise ValueError("preview native build options changed")
    native = package / "bin/tensorfold.exe"
    if digest(native) != source["binary_sha256"]:
        raise ValueError("native source manifest digest mismatch")
    for line in (package / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = package / name.removeprefix("./")
        if package not in path.resolve().parents or digest(path) != expected:
            raise ValueError("package file checksum mismatch")
    sys.path.insert(0, str(repo / "tools/release"))
    from windows_inputs import runtime_files
    overlay = work / "runtime-overlay"
    runtime_expected = runtime_files(overlay)
    for name in runtime_expected:
        if digest(overlay / name) != digest(package / name):
            raise ValueError("runtime overlay changed during packaging")
    expected_paths = set(runtime_expected) | {
        "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "RUNTIME.md", "VERSION",
        "SOURCE.json", "SHA256SUMS", "bin/tensorfold.exe",
    }
    for name, original in [("LICENSE", repo / "LICENSE"), ("NOTICE", repo / "NOTICE"),
                           ("THIRD_PARTY_NOTICES.md", repo / "THIRD_PARTY_NOTICES.md"),
                           ("RUNTIME.md", repo / "packaging/RUNTIME.md")]:
        if digest(package / name) != digest(original):
            raise ValueError("packaged project legal or runtime documentation differs from source")
    for original in (repo / "LICENSES").rglob("*"):
        if original.is_file():
            name = "LICENSES/" + original.relative_to(repo / "LICENSES").as_posix()
            expected_paths.add(name)
            if digest(package / name) != digest(original):
                raise ValueError("packaged project license differs from source")
    actual_paths = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("package differs from the closed source/runtime inventory")
    libraries = load(package / "WINDOWS-RUNTIMES.json")["libraries"]
    for row in libraries:
        if row["path"] not in {"bin/tf_cublaslt_bridge.dll", "bin/cublasLt64_13.dll", "lib/tfgrammar.dll"}:
            raise ValueError("unexpected bundled runtime")
    expected = {row["path"]: row["sha256"] for row in libraries}
    if expected.get("bin/cublasLt64_13.dll") != policy["vendor_runtime"]["sha256"]:
        raise ValueError("unreviewed NVIDIA runtime member")
    produced = {row["path"]: row["sha256"] for row in bridge["outputs"]}
    if expected.get("bin/tf_cublaslt_bridge.dll") != produced.get("tf_cublaslt_bridge.dll"):
        raise ValueError("packaged bridge differs from the source-produced output")
    runtime_sources = load(work / "runtime-source-manifest.private.json")
    input_lock = load(HERE / "runtime-inputs.lock.json")
    for actual, pinned in zip(runtime_sources["sources"], input_lock["source_archives"], strict=True):
        if any(actual.get(key) != value for key, value in pinned.items()):
            raise ValueError("grammar source provenance changed")
    if source["grammar_binary_sha256"] != expected.get("lib/tfgrammar.dll"):
        raise ValueError("grammar output binding mismatch")
    tool = runtime_sources["gitleaks"]
    if tool["executable_sha256"] != digest(gitleaks) or tool["sha256"] != input_lock["gitleaks"]["sha256"]:
        raise ValueError("secret scanner does not match its pinned archive")
    # Each public attribution is limited to its exact value in a verified shipped legal input.
    intended = []
    for row in load(work / "runtime-legal-inputs.private.json"):
        path = package / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("runtime legal input changed")
        for hit in RULES["email"].finditer(path.read_text(encoding="utf-8", errors="strict")):
            item = {"category": "email", "sha256": hashlib.sha256(hit.group().encode()).hexdigest(),
                    "legal_file": row["path"], "legal_sha256": row["sha256"], "source": row["source"]}
            policy["public_readable_values"].append(item)
            intended.append(item)
    scanner = Scanner(context, policy, audit)
    bundled = {Path(row["path"]).name.casefold() for row in libraries}
    pe_reports, cuda_state = [], {"decoded_bytes": 0, "entries": 0}
    compressed_ranges, dispositions = [], []
    expected_bridge = load(HERE / "source-manifest.json")["device_symbols"]
    bridge_functions, bridge_images = set(), 0
    seen_images = set()
    cuda_report = (audit / "CUDA-structures.jsonl").open("w", encoding="utf-8")

    def emit(data, origin):
        value = hashlib.sha256(data).hexdigest()
        if value not in seen_images:
            seen_images.add(value)
            scanner.scan(data, origin)

    before = stable_snapshot(package)
    try:
        for path in sorted(package.rglob("*")):
            if not path.is_file():
                continue
            if path.stat().st_size > MAX_FILE:
                raise ValueError("packaged input inspection limit")
            data = path.read_bytes()
            relative = path.relative_to(package).as_posix()
            scanner.scan(relative.encode(), "archive-filename")
            emit(data, relative)
            if data[:2] != b"MZ":
                continue
            vendor = digest(path) == policy["vendor_runtime"]["sha256"]
            pe = validate_pe(data, path, vendor, policy)
            pe["CUDA_registration_wrappers"] = cuda_wrappers(data, pe)
            if any(not allowed_import(row["dll"], bundled, policy) for row in pe["imports"]):
                raise ValueError("unbundled non-OS runtime dependency")
            if path.name == "tf_cublaslt_bridge.dll" and set(pe["exports"]) != set(bridge["exports"]):
                raise ValueError("bridge ABI export set changed")
            if path.name == "tfgrammar.dll" and not {"tfg_open", "tfg_close", "tfg_words", "tfg_compile", "tfg_free", "tfg_matcher", "tfg_matcher_free", "tfg_accept", "tfg_rollback", "tfg_budget_ready", "tfg_budget", "tfg_terminated", "tfg_fill", "tfg_drafts"} <= set(pe["exports"]):
                raise ValueError("grammar C ABI exports missing")
            for section in pe["sections"]:
                if section["name"] not in (".nv_fatb", ".cask_re"):
                    continue
                payload = data[section["offset"]:section["offset"] + section["bytes"]]
                for item in iter_containers(payload, lambda value, name: emit(value, relative + ":" + name),
                                            cuda_state, interleaved=vendor and section["name"] == ".cask_re"):
                    cuda_report.write(json.dumps({"image": relative, "section": section["name"], **item}) + "\n")
                    for entry in item["entries"]:
                        if entry["compression"] != "none":
                            compressed_ranges.append({
                                "origin": relative, "offset": section["offset"] + entry["offset"],
                                "bytes": entry["bytes"], "encoded_sha256": entry["sha256"],
                                "decoded_sha256": entry["decoded_sha256"],
                            })
                        if path.name == "tf_cublaslt_bridge.dll":
                            bridge_images += 1
                            bridge_functions.update(
                                row["name"][6:] for row in entry["elf"]["sections"]
                                if row["name"].startswith(".text._ZN9tf_lt_gpu"))
            if vendor:
                for section in pe["sections"]:
                    if section["name"] != ".data":
                        continue
                    payload = data[section["offset"]:section["offset"] + section["bytes"]]
                    for frame in zstd_frames(payload, lambda value, name: emit(value, relative + ":" + name), cuda_state):
                        cuda_report.write(json.dumps({"image": relative, "section": ".data", **frame}) + "\n")
                        compressed_ranges.append({
                            "origin": relative, "offset": section["offset"] + frame["offset"],
                            "bytes": frame["bytes"], "encoded_sha256": frame["sha256"],
                            "decoded_sha256": frame["decoded_sha256"],
                        })
            pe_reports.append({"image": relative, **pe})
        for sidecar in public_sidecars:
            emit(sidecar.read_bytes(), sidecar.name)
        native_data = native.read_bytes()
        for row in source["fatbin_inputs"]:
            path = work / "native-out/fatbin" / row["file"]
            data = path.read_bytes()
            if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("fatbin producer input changed")
            at = native_data.find(data)
            if at < 0 or native_data.find(data, at + 1) >= 0:
                raise ValueError("fatbin is not embedded exactly once in this executable")
            for item in iter_containers(data, lambda value, name: emit(value, row["file"] + ":" + name), cuda_state):
                cuda_report.write(json.dumps({"image": "bin/tensorfold.exe", "fatbin": row["file"], **item}) + "\n")
        require_bridge_cuda(bridge_functions, bridge_images, expected_bridge)
        if cuda_state["entries"] < len(source["fatbin_inputs"]):
            raise ValueError("CUDA coverage is empty")
    finally:
        cuda_report.close()
        context_report = scanner.close()
    (audit / "PE-structures.json").write_text(json.dumps(pe_reports, indent=2) + "\n")
    (audit / "public-attribution-values.json").write_text(json.dumps(intended, indent=2) + "\n")
    unresolved, dispositions = resolve_encoded_findings(context_report["findings"], compressed_ranges, policy)
    (audit / "encoded-match-dispositions.json").write_text(json.dumps(dispositions, indent=2) + "\n")
    if unresolved:
        raise ValueError("unresolved contextual finding")
    secret_receipt = secret_scan(gitleaks, audit / "readable-spans.txt", audit)
    if sidecar_snapshot != [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in public_sidecars
    ]:
        raise ValueError("public build provenance changed during review")
    if before != stable_snapshot(package):
        raise ValueError("archive contents changed during review")
    verify_source(repo, source)
    if digest(archive) != (archive.with_suffix(archive.suffix + ".sha256").read_text().split()[0]):
        raise ValueError("archive companion checksum mismatch")
    report = {"schema": 1, "status": "PASSED", "source_commit": revision, "source_tree": source["source_tree"], "archive_sha256": digest(archive),
              "files": before, "PE_images": len(pe_reports), "CUDA_images": cuda_state["entries"],
              "CUDA_decoded_bytes": cuda_state["decoded_bytes"], "contextual_findings": 0,
              "secret_findings": 0, "full_source_and_toolchain_provenance": True,
              "public_sidecars": sidecar_snapshot, "gitleaks_scanned_bytes": secret_receipt["scanned_bytes"],
              "bridge_CUDA_images": bridge_images, "bridge_device_symbols": sorted(bridge_functions),
              "encoded_coincidences_resolved": len(dispositions),
              "vendor_opaque_metadata": {"input_sha256": policy["vendor_runtime"]["sha256"],
                                         "resolved_by": "Exact official public SDK provenance; readable spans inspected",
                                         "spans": cuda_state.get("vendor_data_spans", [])},
              "limits_reached": False, "GPU_qualification": "unrun",
              "privacy_scope": "This exact preview and bound build inputs; no guarantee of absence of personal data"}
    return report
