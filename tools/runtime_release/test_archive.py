import io
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from archive import package, verify


class Archives(unittest.TestCase):
    def archive(self, root, change=None):
        version = "1.2.4"
        name = f"tensorfold-{version}-macos-arm64"
        file = root / f"{name}.tar.gz"
        entries = {"bin/tensorfold-native": b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01fixture",
                   "lib/libtfgrammar.dylib": b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01grammar",
                   "LICENSE": b"license", "NOTICE": b"notice", "THIRD_PARTY_NOTICES.md": b"notices", "VERSION": b"1.2.4\n"}
        if change:
            change(entries)
        with tarfile.open(file, "w:gz") as target:
            for path, body in entries.items():
                member = tarfile.TarInfo(name + "/" + path)
                member.size = len(body)
                target.addfile(member, io.BytesIO(body))
        return file

    def test_repack_is_repeatable_and_independently_verifies_every_original_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.archive(root)
            first = package("1.2.4", root / "tensorfold-1.2.4-macos-arm64.tar.gz", root)
            self.assertEqual(verify("1.2.4", root), first)
            other = root / "other"
            other.mkdir()
            second = package("1.2.4", root / "tensorfold-1.2.4-macos-arm64.tar.gz", other)
            self.assertEqual(first, second)
            metadata = root / "package.json"
            doc = json.loads(metadata.read_text())
            doc["files"][0]["sha256"] = "a" * 64
            metadata.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                verify("1.2.4", root)

    def test_traversal_wrong_platform_missing_grammar_and_host_only_archives_are_refused(self):
        changes = [lambda entries: entries.update({"../escape": b"no"}),
                   lambda entries: entries.update({"bin/tensorfold-native": b"script"}),
                   lambda entries: entries.pop("lib/libtfgrammar.dylib"),
                   lambda entries: entries.update({"HOST-ONLY": b"not qualified"})]
        for change in changes:
            with self.subTest(change=changes.index(change)), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = self.archive(root, change)
                with self.assertRaises(ValueError):
                    package("1.2.4", source, root)
                self.assertFalse((root / "escape").exists())

    def test_signing_host_does_not_require_identical_cross_platform_gzip_compression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.archive(root)
            doc = package("1.2.4", source, root)
            archive = root / f"{doc['id']}.tar.gz"
            original = archive.read_bytes()
            alternate = gzip.compress(gzip.decompress(original), compresslevel=1, mtime=0)
            self.assertNotEqual(original, alternate)
            archive.write_bytes(alternate)
            doc["sha256"] = hashlib.sha256(alternate).hexdigest()
            doc["bytes"] = len(alternate)
            (root / "package.json").write_text(json.dumps(doc))
            self.assertEqual(verify("1.2.4", root), doc)


if __name__ == "__main__":
    unittest.main()
