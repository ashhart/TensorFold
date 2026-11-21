from contextlib import contextmanager, redirect_stderr
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tf_installer import archive, build, install, network, pins
import tensorfold_runtime


def tar_fixture(path, entries):
    with tarfile.open(path, "w:gz") as tar:
        for name, payload, kind in entries:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.mode = 0o755
            item.size = len(payload) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                item.linkname = "../../escaped"
            tar.addfile(item, io.BytesIO(payload) if item.size else None)
    return path


def engine_fixture(path, version="1.2.0", extra=None):
    files = {"bin/tensorfold-native": b"native", "VERSION": version.encode(), "LICENSE": b"license", "NOTICE": b"notice", "THIRD_PARTY_NOTICES.md": b"third-party", "RUNTIME.md": b"runtime"}
    files.update(extra or {})
    sums = "".join(f"{hashlib.sha256(data).hexdigest()}  ./{name}\n" for name, data in files.items())
    files["SHA256SUMS"] = sums.encode()
    root = f"tensorfold-{version}-linux-x86_64"
    tar_fixture(path, [(f"{root}/{name}", data, tarfile.REGTYPE) for name, data in files.items()])
    return root


def release_metadata(version="1.2.0", size=100):
    name = f"tensorfold-{version}-linux-x86_64.tar.gz"
    return {"draft": False, "prerelease": False, "tag_name": f"v{version}", "assets": [{"name": part, "size": size if part == name else 100, "browser_download_url": f"{pins.RELEASE_BASE}/v{version}/{part}"} for part in (name, name + ".sha256")]}


def license_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        (root / name).write_text(name)
    (root / "LICENSES").mkdir()
    (root / "LICENSES/Apache.txt").write_text("license")


def gb10():
    return "12.1\n"


@contextmanager
def fake_nvidia_smi(stdout=b"12.1\n", returncode=0, found=True, error=None):
    """nvidia-smi's answer, faked at the subprocess boundary: no process runs and no GPU is needed."""
    answer = subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, b"")
    with patch("shutil.which", side_effect=lambda name, *args, **kwargs: f"/usr/bin/{name}" if found and name == "nvidia-smi" else None), patch("subprocess.run", side_effect=error, return_value=answer) as query:
        yield query


def candidate_fixture(root, name="candidate", version="1.1.0"):
    folder = root / name
    folder.mkdir()
    paths = {"engine": "engine/bin/tensorfold-native", "moss": "media/bin/tf-moss-image"}
    items = {}
    for kind, path in paths.items():
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(kind)
        target.chmod(0o755)
        items[kind] = {"path": path, "sha256": archive.file_hash(target)}
    items["engine"]["version"] = version
    license_fixture(folder / "media/licenses")
    manifest = {"schema": 1, "source_revision": pins.SOURCE_REVISION, "source_patch_sha256": pins.SOURCE_PATCH_SHA256, "platform": "linux-x86_64", "backend": "cuda", "engine": items["engine"], "artifacts": {"moss": items["moss"]}}
    return folder, manifest


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()


class ArchiveTests(TemporaryCase):
    def test_complete_engine_roundtrip(self):
        name = engine_fixture(self.root / "engine.tgz")
        directory, files = archive.extract_archive(self.root / "engine.tgz", self.root / "out", name)
        archive.verify_engine(directory, files, "1.2.0")

    def test_unsafe_archive_members(self):
        cases = [("../escape", tarfile.REGTYPE), ("root/../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("root/back\\slash", tarfile.REGTYPE), ("root/C:drive", tarfile.REGTYPE), ("root//empty", tarfile.REGTYPE), ("elsewhere/a", tarfile.REGTYPE), ("root/link", tarfile.SYMTYPE), ("root/hard", tarfile.LNKTYPE), ("root/device", tarfile.CHRTYPE), ("root/pipe", tarfile.FIFOTYPE)]
        for index, (name, kind) in enumerate(cases):
            with self.subTest(name=name):
                path = tar_fixture(self.root / f"{index}.tgz", [(name, b"bad", kind)])
                with self.assertRaises(pins.InstallError):
                    archive.extract_archive(path, self.root / f"out{index}", "root")
        self.assertFalse((self.root / "escape").exists())

    def test_duplicate_and_size_limits(self):
        duplicate = tar_fixture(self.root / "duplicate.tgz", [("root/a", b"a", tarfile.REGTYPE), ("root/a", b"b", tarfile.REGTYPE)])
        with self.assertRaisesRegex(pins.InstallError, "Duplicate"):
            archive.extract_archive(duplicate, self.root / "dup", "root")
        large = tar_fixture(self.root / "large.tgz", [("root/a", b"abcde", tarfile.REGTYPE)])
        with self.assertRaisesRegex(pins.InstallError, "size limit"):
            archive.extract_archive(large, self.root / "large", "root", max_bytes=4)

    def test_checksums_are_complete(self):
        for mutation in ("extra", "missing", "changed", "version", "host-only"):
            with self.subTest(mutation=mutation):
                name = engine_fixture(self.root / f"{mutation}.tgz", extra={"HOST-ONLY": b"verification"} if mutation == "host-only" else None)
                folder, files = archive.extract_archive(self.root / f"{mutation}.tgz", self.root / mutation, name)
                if mutation == "extra":
                    (folder / "unhashed").write_bytes(b"x")
                    files.add("unhashed")
                elif mutation == "missing":
                    files.remove("NOTICE")
                elif mutation == "changed":
                    (folder / "bin/tensorfold-native").write_bytes(b"changed")
                with self.assertRaises(pins.InstallError):
                    archive.verify_engine(folder, files, "1.2.1" if mutation == "version" else "1.2.0")

    def test_malformed_or_duplicate_checksum(self):
        digest = "a" * 64
        for text in ("garbage", f"{digest}  ../escape", f"{digest}  a\n{digest}  a\n", ""):
            with self.subTest(text=text), self.assertRaises(pins.InstallError):
                archive.parse_sums(text)

    def test_wrong_archive_pin(self):
        file = self.root / "artifact"
        file.write_bytes(b"not the pinned archive")
        with self.assertRaisesRegex(pins.InstallError, "SHA-256 mismatch"):
            archive.verify_hash(file, "0" * 64)

    def test_pax_metadata_bounds_and_path_override(self):
        for index, metadata in enumerate(({"comment": "x" * (1024 * 1024 + 1)}, {"path": "../pax-escape"})):
            path = self.root / f"pax{index}.tgz"
            with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as tar:
                member = tarfile.TarInfo("root/file")
                member.pax_headers = metadata
                member.size = 1
                tar.addfile(member, io.BytesIO(b"x"))
            with self.subTest(index=index), self.assertRaises(pins.InstallError):
                archive.extract_archive(path, self.root / f"paxout{index}", "root")
        self.assertFalse((self.root / "pax-escape").exists())


class ReleaseTests(TemporaryCase):
    def test_exact_asset_selection(self):
        value = network.select_release(release_metadata(), "linux-x86_64")
        self.assertEqual(value["version"], "1.2.0")
        self.assertEqual(len(value["assets"]), 2)

    def test_untrusted_metadata_is_refused(self):
        for change in ("prerelease", "draft", "tag", "url", "duplicate", "missing", "oversized", "size-type"):
            data = release_metadata()
            if change in ("prerelease", "draft"):
                data[change] = True
            elif change == "tag":
                data["tag_name"] = "v1.2.0-beta.1"
            elif change == "url":
                data["assets"][0]["browser_download_url"] = "https://evil.invalid/a"
            elif change == "duplicate":
                data["assets"].append(data["assets"][0])
            elif change == "missing":
                data["assets"].pop()
            else:
                data["assets"][0]["size"] = "100" if change == "size-type" else pins.MAX_ARCHIVE_BYTES + 1
            with self.subTest(change=change), self.assertRaises(pins.InstallError):
                network.select_release(data, "linux-x86_64")

    def test_redirect_policy(self):
        for url in ("http://github.com/a", "https://github.com.evil.invalid/a", "https://user:pass@github.com/a", "https://127.0.0.1/a", "https://github.com:444/a"):
            with self.subTest(url=url), self.assertRaises(pins.InstallError):
                network.checked_url(url)
        network.checked_url("https://release-assets.githubusercontent.com/a?signature=abc")

    def test_asset_digest_must_match_companion(self):
        metadata = release_metadata()
        metadata["assets"][0]["digest"] = "sha256:" + "f" * 64
        release = network.select_release(metadata, "linux-x86_64")

        def fetch(url, destination, limit):
            destination.write_text("0" * 64 + "  " + release["filename"] + "\n")
            return destination

        with self.assertRaisesRegex(pins.InstallError, "disagree"):
            network.fetch_release(self.root, release, fetch)

    def test_streamed_download_bounds_and_redirect_destination(self):
        class Response(io.BytesIO):
            status = 200
            headers = {}
            url = "https://github.com/allowed"

            def geturl(self):
                return self.url

        class Opener:
            def __init__(self, response):
                self.response = response

            def open(self, request, timeout):
                return self.response

        cases = ((b"12345", {}, "https://github.com/a"), (b"12", {"Content-Length": "3"}, "https://github.com/a"), (b"123", {}, "https://evil.invalid/a"), (b"1", {"Content-Length": "-1"}, "https://github.com/a"))
        for index, (payload, headers, url) in enumerate(cases):
            response = Response(payload)
            response.headers = headers
            response.url = url
            with self.subTest(index=index), self.assertRaises(pins.InstallError):
                network.download("https://github.com/a", self.root / f"download{index}", 4, opener=Opener(response))
        path = network.download("https://github.com/a", self.root / "valid", 3, opener=Opener(Response(b"abc")))
        self.assertEqual(path.read_bytes(), b"abc")


class SourceTests(TemporaryCase):
    def test_wrong_source_revision_and_dirty_checkout(self):
        for revision, dirty in (("a" * 40, ""), (pins.SOURCE_REVISION, " M build.zig")):
            def run(arguments, cwd, log, **kwargs):
                return revision if "rev-parse" in arguments else dirty
            with self.subTest(revision=revision, dirty=dirty), patch.object(build, "executable", return_value=sys.executable), self.assertRaisesRegex(pins.InstallError, "clean and exactly"):
                build.prepare_source(self.root, checkout=self.root, run=run)

    def test_source_archive_requires_pinned_digest(self):
        fake = self.root / "source.tgz"
        fake.write_bytes(b"different source")
        with self.assertRaisesRegex(pins.InstallError, "SHA-256 mismatch"):
            build.prepare_source(self.root, archive=fake)

    def test_exact_zig_version(self):
        with patch.object(build, "executable", return_value=sys.executable), self.assertRaisesRegex(pins.InstallError, "requires Zig"):
            build.check_toolchain(None, None, self.root, run=lambda *args, **kwargs: "0.16.0")

    def test_build_targets_and_probes(self):
        work = self.root / "work"
        work.mkdir()
        candidate = self.root / "candidate"
        candidate.mkdir()
        seen = []

        def run(arguments, cwd, log, **kwargs):
            seen.append(arguments)
            if "build" in arguments:
                self.assertIn("-j2", arguments)
                self.assertNotIn("--global-cache-dir", arguments)
                self.assertEqual(kwargs["environment"], {"ZIG_GLOBAL_CACHE_DIR": str(work / "zig-global-cache")})
                self.assertEqual(arguments[2], "moss-image-cuda")
                self.assertEqual([argument for argument in arguments if argument.startswith("-Dsm=")], ["-Dsm=121"])
                target = work / "built/bin/tf-moss-image"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"built")
                return ""
            if str(arguments[0]).endswith("tf-moss-image"):
                self.assertEqual(kwargs["expected"], 2)
                return "tf-moss-image TRANSFORMER_DIR CASE OUT_DIR"
            return "1"

        paths = build.build_media(self.root, work, candidate, "/zig", "/nvcc", run)
        self.assertEqual(set(paths), {"moss"})
        self.assertEqual(seen[0][1:3], ["apply", "--check"])
        self.assertEqual(seen[1][1], "apply")
        self.assertEqual(seen[2][:3], ["/zig", "build", "moss-image-cuda"])
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(not any("optimize" in arg for arg in arguments) for arguments in seen))

    def test_media_patch_inside_comfy_git_checkout(self):
        source = self.root / "ComfyUI/models/source"
        source.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(self.root / "ComfyUI")], check=True)
        (source / "sample.txt").write_text("before\n")
        payload = b"diff --git a/sample.txt b/sample.txt\n--- a/sample.txt\n+++ b/sample.txt\n@@ -1 +1 @@\n-before\n+after\n"
        (self.root / "moss-comfy.patch").write_bytes(payload)
        with patch.object(build, "__file__", str(self.root / "build.py")), patch.object(build, "SOURCE_PATCH_SHA256", hashlib.sha256(payload).hexdigest()):
            build.apply_media_patch(source, self.root)
        self.assertEqual((source / "sample.txt").read_text(), "after\n")
        self.assertFalse((self.root / "ComfyUI/sample.txt").exists())

    def test_tampered_media_source_patch_stops_before_build(self):
        (self.root / "moss-comfy.patch").write_text("tampered patch")
        run = Mock()
        with patch.object(build, "__file__", str(self.root / "build.py")), self.assertRaises(pins.InstallError):
            build.build_media(self.root, self.root, self.root, "/zig", "/nvcc", run)
        run.assert_not_called()

    def test_real_subprocess_failure_and_timeout(self):
        with self.assertRaisesRegex(pins.InstallError, "exited 3"):
            build.run_command([sys.executable, "-c", "raise SystemExit(3)"], self.root, self.root / "failure.log")
        with self.assertRaisesRegex(pins.InstallError, "time or log-size"):
            build.run_command([sys.executable, "-c", "import time; time.sleep(5)"], self.root, self.root / "timeout.log", timeout=0.15)

    def test_long_probe_output_never_looks_empty(self):
        output = build.run_command([sys.executable, "-c", "print(' M a' * 20000)"], self.root, self.root / "long.log")
        self.assertTrue(output)
        self.assertIn("output truncated", output)


class TransactionTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.comfy = self.root / "ComfyUI"
        self.comfy.mkdir()
        (self.comfy / "main.py").touch()
        (self.comfy / "folder_paths.py").touch()
        self.runtime = install.runtime_root(self.comfy)
        self.runtime.mkdir(parents=True)

    def candidate(self, name="candidate", version="1.1.0"):
        return candidate_fixture(self.root, name, version)

    def test_atomic_activation_and_offline_status(self):
        folder, manifest = self.candidate()
        active = install.activate(self.runtime, folder, manifest)
        with patch.object(network, "download", side_effect=AssertionError("Network forbidden")), patch.object(subprocess, "Popen", side_effect=AssertionError("Execution forbidden")):
            self.assertEqual(install.read_status(self.runtime), active)

    def test_failed_activation_preserves_previous(self):
        folder, manifest = self.candidate()
        install.activate(self.runtime, folder, manifest)
        previous = (self.runtime / "runtime.json").read_bytes()
        folder, manifest = self.candidate("next", "1.2.0")
        def rejected(*args):
            raise OSError("activation failure")
        with self.assertRaises(OSError):
            install.activate(self.runtime, folder, manifest, replace=rejected)
        self.assertEqual((self.runtime / "runtime.json").read_bytes(), previous)
        self.assertEqual(len(list((self.runtime / "installs").iterdir())), 1)

    def test_failed_build_preserves_previous(self):
        folder, manifest = self.candidate()
        install.activate(self.runtime, folder, manifest)
        previous = (self.runtime / "runtime.json").read_bytes()
        (self.runtime / manifest["artifacts"]["moss"]["path"]).write_text("corrupt")
        package = self.root / "engine.tgz"
        engine_fixture(package)
        digest = archive.file_hash(package)
        def bad_build(source, work, *args):
            (work / "moss-build.log").write_text("compiler diagnostic retained")
            raise pins.InstallError("build failed")
        messages = []
        with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "check_toolchain", return_value=("zig", "nvcc")), patch.object(install, "get_release", return_value={"version": "1.2.0"}), patch.object(install, "fetch_release", return_value=("1.2.0", package, digest)), patch.object(install, "prepare_source", return_value=self.root), self.assertRaisesRegex(pins.InstallError, "build failed"):
            install.install(self.comfy, gpu=gb10, action="install", run=lambda *args, **kwargs: "tensorfold-native 1.2.0", builder=bad_build, log=messages.append)
        self.assertEqual((self.runtime / "runtime.json").read_bytes(), previous)
        self.assertEqual(json.loads(previous)["engine"]["version"], "1.1.0")
        retained = list((self.runtime / "logs").glob("failed-*/moss-build.log"))
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].read_text(), "compiler diagnostic retained")
        self.assertTrue(any(str(retained[0].parent) in message for message in messages))
        self.assertFalse(list(self.runtime.glob(".install-*")))

    def test_wrong_platform_stops_before_network_and_build(self):
        with patch.object(build.platform, "system", return_value="Windows"), patch.object(install, "get_release", side_effect=AssertionError("Network forbidden")), patch.object(install, "check_toolchain", side_effect=AssertionError("Build forbidden")), self.assertRaisesRegex(pins.InstallError, "Linux"):
            install.install(self.comfy)

    def test_status_rejects_artifact_tampering_and_symlinks(self):
        folder, manifest = self.candidate()
        active = install.activate(self.runtime, folder, manifest)
        path = self.runtime / active["artifacts"]["moss"]["path"]
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(pins.InstallError, "SHA-256 mismatch"):
            install.read_status(self.runtime)
        path.unlink()
        path.symlink_to(self.root / "other")
        with self.assertRaisesRegex(pins.InstallError, "symlinks"):
            install.read_status(self.runtime)

    def test_status_and_update_reject_nonexecutable_binaries(self):
        folder, manifest = self.candidate(version="1.2.0")
        active = install.activate(self.runtime, folder, manifest)
        for name in ("engine", "moss"):
            item = active["engine"] if name == "engine" else active["artifacts"][name]
            path = self.runtime / item["path"]
            path.chmod(0o644)
            with self.subTest(name=name), self.assertRaisesRegex(pins.InstallError, "not executable"):
                install.read_status(self.runtime)
            with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "get_release", side_effect=AssertionError("Network forbidden")), self.subTest(update=name), self.assertRaisesRegex(pins.InstallError, "use install"):
                install.install(self.comfy, gpu=gb10, action="update")
            path.chmod(0o755)

    def test_no_downgrade(self):
        folder, manifest = self.candidate(version="2.0.0")
        install.activate(self.runtime, folder, manifest)
        with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "check_toolchain", return_value=("zig", "nvcc")), patch.object(install, "get_release", return_value={"version": "1.2.0"}), self.assertRaisesRegex(pins.InstallError, "downgrade"):
            install.install(self.comfy, gpu=gb10, action="update")

    def test_no_op_update_needs_no_compiler_or_download(self):
        folder, manifest = self.candidate(version="1.2.0")
        active = install.activate(self.runtime, folder, manifest)
        with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "check_toolchain", side_effect=AssertionError("Compiler forbidden")), patch.object(install, "get_release", return_value={"version": "1.2.0"}), patch.object(install, "fetch_release", side_effect=AssertionError("Artifact download forbidden")):
            self.assertEqual(install.install(self.comfy, gpu=gb10, action="update", log=lambda text: None), active)

    def test_same_cli_does_not_reuse_legacy_moss(self):
        folder, manifest = self.candidate(version="1.2.0")
        manifest.pop("source_patch_sha256")
        previous = install.activate(self.runtime, folder, manifest)
        with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "get_release", return_value={"version": "1.2.0"}), patch.object(install, "fetch_release", side_effect=RuntimeError("rebuild required")) as fetch:
            with self.assertRaisesRegex(RuntimeError, "rebuild required"):
                install.install(self.comfy, gpu=gb10, action="update", log=lambda text: None)
        fetch.assert_called_once()
        self.assertEqual(install.read_status(self.runtime), previous)

    def test_updated_cli_reuses_media_without_compiler(self):
        folder, manifest = self.candidate()
        previous = install.activate(self.runtime, folder, manifest)
        package = self.root / "new-engine.tgz"
        engine_fixture(package)
        with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "check_toolchain", side_effect=AssertionError("Compiler forbidden")), patch.object(install, "get_release", return_value={"version": "1.2.0"}), patch.object(install, "fetch_release", return_value=("1.2.0", package, archive.file_hash(package))):
            result = install.install(self.comfy, gpu=gb10, action="update", run=lambda *args, **kwargs: "tensorfold-native 1.2.0", builder=lambda *args: self.fail("Build forbidden"), log=lambda text: None)
        self.assertEqual(result["engine"]["version"], "1.2.0")
        self.assertEqual(result["artifacts"]["moss"]["sha256"], previous["artifacts"]["moss"]["sha256"])
        self.assertEqual(install.read_status(self.runtime), result)
        self.assertEqual(len(list((self.runtime / "installs").iterdir())), 2)

    def test_reinstall_repairs_corrupt_runtime(self):
        package = self.root / "new-engine.tgz"
        engine_fixture(package)
        source = self.root / "source"
        license_fixture(source)

        def builder(source, work, candidate, zig, nvcc, run):
            paths = {}
            for name, relative in (("moss", "media/bin/tf-moss-image"),):
                path = candidate / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
                path.chmod(0o755)
                paths[name] = path
            return paths

        for damage in ("contents", "moss", "engine"):
            with self.subTest(damage=damage):
                folder, manifest = self.candidate(version="1.2.0")
                install.activate(self.runtime, folder, manifest)
                item = manifest["engine"] if damage == "engine" else manifest["artifacts"]["moss"]
                broken = self.runtime / item["path"]
                if damage == "contents":
                    broken.write_text("corrupt")
                else:
                    broken.chmod(0o644)
                previous = (self.runtime / "runtime.json").read_bytes()
                with patch.object(install, "host_platform", return_value="linux-x86_64"), patch.object(install, "check_toolchain", return_value=("zig", "nvcc")), patch.object(install, "get_release", return_value={"version": "1.2.0"}), patch.object(install, "fetch_release", return_value=("1.2.0", package, archive.file_hash(package))), patch.object(install, "prepare_source", return_value=source):
                    result = install.install(self.comfy, gpu=gb10, action="install", run=lambda *args, **kwargs: "tensorfold-native 1.2.0", builder=builder, log=lambda text: None)
                self.assertNotEqual((self.runtime / "runtime.json").read_bytes(), previous)
                self.assertEqual(install.read_status(self.runtime), result)
                for component in (result["engine"], result["artifacts"]["moss"]):
                    self.assertTrue((self.runtime / component["path"]).stat().st_mode & 0o100)

    def test_boolean_schema_is_refused(self):
        folder, manifest = self.candidate()
        active = install.activate(self.runtime, folder, manifest)
        for field in ("schema",):
            bad = json.loads(json.dumps(active))
            bad["schema"] = True
            (self.runtime / "runtime.json").write_text(json.dumps(bad))
            with self.subTest(field=field), self.assertRaises(pins.InstallError):
                install.read_status(self.runtime)


QUALIFIED = "MOSS ComfyUI generation is qualified on GB10 (sm_121) only"
UNREADABLE = (
    ("nvidia-smi missing", {"found": False}),
    ("driver without compute_cap", {"returncode": 2, "stdout": b'Field "compute_cap" is not a valid field to query.\n'}),
    ("not available", {"stdout": b"[N/A]\n"}),
    ("no GPU listed", {"stdout": b""}),
    ("timeout", {"error": subprocess.TimeoutExpired("nvidia-smi", 15)}),
)


class InstallAdmissionTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.comfy = self.root / "ComfyUI"
        self.comfy.mkdir()
        (self.comfy / "main.py").touch()
        (self.comfy / "folder_paths.py").touch()

    def attempt(self, **answer):
        fetch = Mock(side_effect=pins.InstallError("network reached"))
        forbidden = Mock(side_effect=AssertionError("Build forbidden"))
        with fake_nvidia_smi(**answer) as query, patch.object(install, "host_platform", return_value="linux-aarch64"), patch.object(install, "check_toolchain", forbidden), patch.object(install, "prepare_source", forbidden), self.assertRaises(pins.InstallError) as raised:
            install.install(self.comfy, fetch=fetch, run=forbidden, builder=forbidden, log=lambda text: None)
        forbidden.assert_not_called()
        return str(raised.exception), fetch, query

    def test_sm120_refused_by_name_before_any_download(self):
        for stdout, which in ((b"12.0\n", "this GPU"), (b"12.1\n12.0\n", "GPU 1 of 2")):
            with self.subTest(stdout=stdout):
                message, fetch, query = self.attempt(stdout=stdout)
                self.assertEqual(message, f"{QUALIFIED}; {which} is sm_120. No files were downloaded or built.")
                fetch.assert_not_called()
                self.assertFalse((self.comfy / "models").exists())
                self.assertEqual(query.call_args.args[0], ["/usr/bin/nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])

    def test_sm121_reads_the_capability_then_proceeds(self):
        message, fetch, query = self.attempt(stdout=b"12.1\n")
        query.assert_called_once()
        self.assertEqual(message, "network reached")
        self.assertEqual(fetch.call_args.args[0], pins.RELEASE_API)

    def test_unreadable_capability_refused_not_assumed(self):
        for name, answer in UNREADABLE:
            with self.subTest(name=name):
                message, fetch, query = self.attempt(**answer)
                self.assertTrue(message.startswith(f"{QUALIFIED}; this GPU's compute capability could not be read ("), message)
                self.assertTrue(message.endswith(" No files were downloaded or built."), message)
                fetch.assert_not_called()
                self.assertFalse((self.comfy / "models").exists())


class RunAdmissionTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.comfy = self.root / "ComfyUI"
        self.comfy.mkdir()
        (self.comfy / "main.py").touch()
        (self.comfy / "folder_paths.py").touch()
        self.runtime = install.runtime_root(self.comfy)
        self.runtime.mkdir(parents=True)
        folder, manifest = candidate_fixture(self.root)
        active = install.activate(self.runtime, folder, manifest)
        self.worker = str(self.runtime / active["artifacts"]["moss"]["path"])

    def attempt(self, **answer):
        with fake_nvidia_smi(**answer) as query, patch("os.execv") as execute, redirect_stderr(io.StringIO()) as errors:
            code = tensorfold_runtime.main(["run", "--comfy-root", str(self.comfy), "--", "transformer", "case.safetensors", "out", "--prompt", "prompt.txt"])
        return code, errors.getvalue(), execute, query

    def test_sm120_refused_by_name_before_the_worker_starts(self):
        code, errors, execute, query = self.attempt(stdout=b"12.0\n")
        self.assertEqual((code, errors), (1, f"TensorFold: {QUALIFIED}; this GPU is sm_120.\n"))
        execute.assert_not_called()

    def test_sm121_starts_the_verified_worker(self):
        code, errors, execute, query = self.attempt(stdout=b"12.1\n")
        self.assertEqual((code, errors), (0, ""))
        query.assert_called_once()
        execute.assert_called_once_with(self.worker, [self.worker, "transformer", "case.safetensors", "out", "--prompt", "prompt.txt"])

    def test_unreadable_capability_refused_not_assumed(self):
        for name, answer in UNREADABLE:
            with self.subTest(name=name):
                code, errors, execute, query = self.attempt(**answer)
                self.assertEqual(code, 1)
                self.assertTrue(errors.startswith(f"TensorFold: {QUALIFIED}; this GPU's compute capability could not be read ("), errors)
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
