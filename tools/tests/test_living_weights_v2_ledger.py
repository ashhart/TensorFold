from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/, as the other tool tests do
from living_weights_v2.ledger import Ledger, LedgerError


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@unittest.skipUnless(os.name == "posix", "file modes are POSIX")
class LedgerPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.umask = os.umask(0o022)

    def tearDown(self):
        os.umask(self.umask)
        self.dir.cleanup()

    def test_new_journal_is_private_under_umask_022(self):
        path = Path(self.dir.name) / "state" / "tensorfold" / "ledger.sqlite3"
        ledger = Ledger(path)
        try:
            columns = "model, fact_digest, subject_key, answer, fact_json, manifest_digest, state"
            ledger.db.execute(f"INSERT INTO entries({columns}) VALUES('m', 'd', 'k', 'Sheffield', '{{}}', 'm', 'committed')")
            ledger.db.commit()
            self.assertEqual(mode(path.parent), 0o700)
            self.assertEqual(mode(path.parent.parent), 0o700)
            for item in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
                with self.subTest(file=item.name):
                    self.assertTrue(item.exists())
                    self.assertEqual(mode(item), 0o600)
        finally:
            ledger.close()

    def test_existing_journal_open_to_others_is_refused(self):
        path = Path(self.dir.name) / "ledger.sqlite3"
        sqlite3.connect(path).close()
        for suffix in ("", "-wal", "-shm"):
            item = Path(str(path) + suffix)
            item.touch()
            os.chmod(path, 0o600)
            for loose in (0o644, 0o640, 0o604):
                with self.subTest(file=item.name, mode=oct(loose)):
                    os.chmod(item, loose)
                    with self.assertRaisesRegex(LedgerError, "open to other users"):
                        Ledger(path).close()
            os.chmod(item, 0o600)
        Ledger(path).close()

    def test_existing_directory_keeps_its_mode(self):
        shared = Path(self.dir.name) / "shared"
        shared.mkdir(mode=0o755)
        Ledger(shared / "ledger.sqlite3").close()
        self.assertEqual(mode(shared), 0o755)
        self.assertEqual(mode(shared / "ledger.sqlite3"), 0o600)


if __name__ == "__main__":
    unittest.main()
