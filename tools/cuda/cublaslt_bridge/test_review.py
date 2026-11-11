"""CPU-only planted refusal controls for the preview's preupload boundary."""
import base64
from compression import zstd
import hashlib
import gzip
import json
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from artifact_gate import (unpack, verify_source, stable_snapshot, require_catalog,
                           require_bridge_cuda, resolve_encoded_findings, secret_scan, allowed_import)
from build_preview import verify_publication
from cuda_review import containers
from pe_review import review
from privacy_review import Scanner, word_hash


def minimal_pe():
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 132, 0x8664, 1, 0, 0, 0, 240, 2)
    struct.pack_into("<H", data, 152, 0x20B)
    struct.pack_into("<I", data, 260, 16)
    struct.pack_into("<8sIIIIIIHHI", data, 392, b".rdata\0\0", 16, 4096, 512, 512, 0, 0, 0, 0, 0x40000040)
    return bytes(data)


def cuda_fixture(compressed=False):
    names = b"\0.shstrtab\0"
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + b"\0" * 9,
                         2, 190, 1, 0, 0, 80, 0, 64, 0, 0, 64, 2, 1)
    table = bytes(64) + struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0)
    elf = header + names + bytes(80 - 64 - len(names)) + table
    payload = zstd.compress(elf) if compressed else elf
    entry = struct.pack("<HHIQ", 2, 257, 16, len(payload)) + payload
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(entry)) + entry


def checkout(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    git("init", "--quiet")
    (path / "source.txt").write_text("public synthetic source fixture\n")
    git("add", "source.txt")
    git("-c", "user.name=Public Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--quiet", "-m", "Synthetic source binding")
    record = {"source_commit": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
              "source_dirty": False, "host_only": False}
    return record


class RefusalControls(unittest.TestCase):
    def test_declared_driver_providers_do_not_allow_an_unbundled_toolkit(self):
        policy = {"system_imports": ["KERNEL32.dll"],
                  "nvidia_driver_provider": {"names": ["nvcuda.dll", "nvcudart_hybrid64.dll"]}}
        self.assertTrue(allowed_import("NVCUDA.DLL", set(), policy))
        self.assertTrue(allowed_import("nvcudart_hybrid64.dll", set(), policy))
        self.assertFalse(allowed_import("nvrtc64_130_0.dll", set(), policy))
        self.assertFalse(allowed_import("unreviewed-runtime.dll", set(), policy))

    def test_secret_scanner_uses_actual_log_format_and_refuses_missing_bytes(self):
        for report_full_size, color in ((True, False), (True, True), (False, True)):
            with self.subTest(report_full_size=report_full_size, color=color), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus, executable = root / "corpus.txt", root / "scanner.fixture"
                corpus.write_bytes(b"Public scanner fixture\n")
                executable.write_bytes(b"Synthetic executable identity fixture\n")
                audit = root / "audit"
                audit.mkdir()
                size = corpus.stat().st_size if report_full_size else 0

                def scan(*args, **kwargs):
                    message = f"scanned ~{size} bytes (22 B) in 0.00s"
                    if color:
                        message = "\x1b[1m" + message + "\x1b[0m"
                    kwargs["stdout"].write(("INF " + message + "\n").encode())
                    (audit / "gitleaks.json").write_text("[]")
                    return subprocess.CompletedProcess(args, 0)

                with patch("artifact_gate.subprocess.run", side_effect=scan):
                    if report_full_size:
                        self.assertEqual(secret_scan(executable, corpus, audit)["scanned_bytes"], size)
                    else:
                        with self.assertRaisesRegex(ValueError, "skipped bytes"):
                            secret_scan(executable, corpus, audit)

    def test_pe_overlay_is_refused(self):
        self.assertEqual(review(minimal_pe())["machine"], 0x8664)
        with self.assertRaisesRegex(ValueError, "overlay"):
            review(minimal_pe() + b"unreviewed-payload")

    def test_cuda_decodes_raw_and_zstd_then_refuses_corruption(self):
        decoded = []
        for compressed in (False, True):
            rows = containers(cuda_fixture(compressed), lambda value, _: decoded.append(value))
            self.assertEqual(rows[0]["entries"][0]["elf"]["machine"], 190)
        self.assertEqual(decoded[0], decoded[1])
        damaged = bytearray(cuda_fixture(True))
        damaged[-4:] = b"FAIL"
        with self.assertRaises((ValueError, zstd.ZstdError)):
            containers(bytes(damaged), lambda *_: None)

    def test_cuda_rejects_hidden_bytes_between_containers(self):
        good = cuda_fixture()
        self.assertEqual(len(containers(good + bytes(8) + good, lambda *_: None)), 2)
        with self.assertRaisesRegex(ValueError, "unexplained"):
            containers(good + b"hidden" + good, lambda *_: None)

    def test_digest_context_catches_ascii_wide_and_base64(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text(json.dumps({"schema": 1, "normalization": "casefold-alnum-v1",
                                           "sha256": [word_hash("fixture workbench")]}))
            scanner = Scanner(context, {"public_readable_values": []}, root)
            scanner.scan(b"Fixture Workbench", "plain")
            scanner.scan("FIXTURE-WORKBENCH".encode("utf-16le"), "wide")
            encoded = base64.b64encode(b"public padding " * 8 + b"fixture workbench" + b" public padding" * 8)
            scanner.scan(encoded, "encoded")
            result = scanner.close()
            origins = {row["origin"] for row in result["findings"] if row["category"] == "private-context"}
            self.assertIn("plain", origins)
            self.assertIn("wide", origins)
            self.assertTrue(any(name.startswith("encoded-base64-") for name in origins))

    def test_nested_base64_gzip_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text(json.dumps({"schema": 1, "normalization": "casefold-alnum-v1",
                                           "sha256": [word_hash("fixture workbench")]}))
            scanner = Scanner(context, {"public_readable_values": []}, root)
            payload = gzip.compress(bytes(range(256)) * 2 + b"fixture workbench")
            try:
                with self.assertRaisesRegex(ValueError, "nested encoded binary"):
                    scanner.scan(base64.b64encode(payload), "encoded-gzip-fixture")
            finally:
                scanner.close()

    def test_catalog_requires_every_source_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "zig/build").mkdir(parents=True)
            (root / "zig/build/cuda_kernels.zig").write_text(
                'pub const kernels = [_]Kernel{\\n'
                '    .{ .name = "first", .flags = glue },\\n'
                '    .{ .name = "second", .flags = glue },\\n};\\n')
            complete = [{"file": "first.fatbin"}, {"file": "second.fatbin"}]
            self.assertEqual(len(require_catalog(root, complete)), 2)
            with self.assertRaisesRegex(ValueError, "complete source catalog"):
                require_catalog(root, complete[:1])

    def test_bridge_device_coverage_is_separate_from_other_images(self):
        require_bridge_cuda({"transform-a", "transform-b"}, 1, ["transform-a", "transform-b"])
        with self.assertRaisesRegex(ValueError, "bridge CUDA"):
            require_bridge_cuda(set(), 0, ["transform-a", "transform-b"])
        with self.assertRaisesRegex(ValueError, "bridge CUDA"):
            require_bridge_cuda({"transform-a"}, 1, ["transform-a", "transform-b"])

    def test_encoded_email_disposition_requires_exact_reviewed_offset(self):
        vendor = "a" * 64
        finding = {"origin": "bin/cublasLt64_13.dll", "offset": 100, "encoding": "ascii",
                   "category": "email", "match_sha256": "b" * 64, "match_bytes": 8}
        policy = {"vendor_runtime": {"sha256": vendor}, "encoded_email_coincidences": [
            {**finding, "vendor_sha256": vendor, "required_resolution": "synthetic codec fixture"}]}
        frame = {"origin": finding["origin"], "offset": 80, "bytes": 64}
        self.assertEqual(len(resolve_encoded_findings([finding], [frame], policy)[1]), 1)
        unexpected = {**finding, "offset": 101}
        self.assertEqual(len(resolve_encoded_findings([unexpected], [frame], policy)[0]), 1)
        self.assertEqual(len(resolve_encoded_findings([finding], [], policy)[0]), 1)

    def test_private_context_refuses_plaintext_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text(json.dumps({"schema": 1, "normalization": "casefold-alnum-v1",
                                           "sha256": [word_hash("fictional-control")],
                                           "plaintext": ["fictional-control"]}))
            with self.assertRaisesRegex(ValueError, "plaintext"):
                Scanner(context, {"public_readable_values": []}, root)

    def test_archive_rejects_traversal_with_fixed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, member in [("good", "package/NOTICE"), ("bad", "package/../NOTICE")]:
                archive = root / (label + ".zip")
                with zipfile.ZipFile(archive, "w") as package:
                    info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    package.writestr(info, b"public synthetic notice")
                destination = root / label
                destination.mkdir()
                if label == "good":
                    self.assertEqual((unpack(archive, destination) / "NOTICE").read_bytes(),
                                     b"public synthetic notice")
                else:
                    with self.assertRaisesRegex(ValueError, "archive member"):
                        unpack(archive, destination)

    def test_clean_source_binding_refuses_a_new_untracked_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = checkout(root)
            self.assertEqual(verify_source(root, source), source["source_commit"])
            (root / "untracked-input.txt").write_text("synthetic unreviewed input\n")
            with self.assertRaisesRegex(ValueError, "clean complete"):
                verify_source(root, source)

    def test_final_publication_freeze_refuses_changed_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, public = root / "repo", root / "publish"
            repo.mkdir()
            public.mkdir()
            source = checkout(repo)
            for name in ["tensorfold-1.2.3-windows-x86_64.zip",
                         "tensorfold-1.2.3-windows-x86_64.zip.sha256", "BUILD-PROVENANCE.json"]:
                (public / name).write_bytes(b"synthetic freeze fixture")
            report = {"status": "PASSED", "source_commit": source["source_commit"],
                      "source_tree": source["source_tree"], "publication_files": stable_snapshot(public)}
            (public / "ARTIFACT-REVIEW.json").write_text(json.dumps(report))
            verify_publication(repo, public)
            (public / "tensorfold-1.2.3-windows-x86_64.zip").write_bytes(b"mutated synthetic fixture")
            with self.assertRaisesRegex(ValueError, "changed after inspection"):
                verify_publication(repo, public)


if __name__ == "__main__":
    unittest.main()
