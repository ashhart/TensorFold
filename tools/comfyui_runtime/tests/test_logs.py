from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tf_installer import logs, pins


class LogTests(unittest.TestCase):
    def test_bounded_logs_keep_error_beginning_and_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            (work / "build.log").write_bytes(b"first error\n" + b"x" * (2 * logs.MAX_LOG_BYTES) + b"\nlast error")
            saved = logs.preserve_logs(root, work)
            data = (saved / "build.log").read_bytes()
            self.assertLessEqual(len(data), logs.MAX_LOG_BYTES)
            self.assertTrue(data.startswith(b"first error\n"))
            self.assertTrue(data.endswith(b"\nlast error"))
            self.assertIn(b"log truncated", data)

    def test_log_links_and_excess_files_are_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            (work / "link.log").symlink_to(root / "private")
            for index in range(logs.MAX_LOG_FILES + 3):
                (work / f"{index:02}.log").write_text("diagnostic")
            saved = logs.preserve_logs(root, work)
            self.assertEqual(len(list(saved.iterdir())), logs.MAX_LOG_FILES)
            self.assertFalse((saved / "link.log").exists())

    def test_managed_log_directory_cannot_escape_by_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            (work / "build.log").write_text("diagnostic")
            (root / "logs").symlink_to(work, target_is_directory=True)
            with self.assertRaises(pins.InstallError):
                logs.preserve_logs(root, work)


if __name__ == "__main__":
    unittest.main()
