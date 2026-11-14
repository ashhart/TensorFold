from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/, as the other tool tests do
from living_weights_v2 import ledger as ledger_module
from living_weights_v2.ledger import Ledger, LedgerError


def mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def plant(path: Path) -> None:
    """A file anyone may read and write, as another process could leave it."""
    descriptor = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o666)
    os.close(descriptor)
    os.chmod(path, 0o666)


@unittest.skipUnless(os.name == "posix", "file modes are POSIX")
class LedgerPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.umask = os.umask(0o022)
        self.journal = Path(self.dir.name) / "journal"
        self.path = self.journal / "ledger.sqlite3"

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
        self.journal.mkdir(mode=0o700)
        sqlite3.connect(self.path).close()
        for suffix in ("", "-wal", "-shm"):
            item = Path(str(self.path) + suffix)
            item.touch()
            os.chmod(self.path, 0o600)
            for loose in (0o644, 0o640, 0o604):
                with self.subTest(file=item.name, mode=oct(loose)):
                    os.chmod(item, loose)
                    with self.assertRaisesRegex(LedgerError, "mode 600"):
                        Ledger(self.path).close()
            os.chmod(item, 0o600)
        Ledger(self.path).close()

    def test_a_shared_or_linked_directory_is_refused(self):
        shared = Path(self.dir.name) / "shared"
        shared.mkdir(mode=0o755)
        with self.assertRaisesRegex(LedgerError, "mode 700"):
            Ledger(shared / "ledger.sqlite3").close()
        self.assertEqual(mode(shared), 0o755)
        self.assertFalse((shared / "ledger.sqlite3").exists())
        private = Path(self.dir.name) / "private"
        private.mkdir(mode=0o700)
        link = Path(self.dir.name) / "link"
        link.symlink_to(private)
        with self.assertRaisesRegex(LedgerError, "mode 700"):
            Ledger(link / "ledger.sqlite3").close()

    def test_a_linked_database_is_refused(self):
        self.journal.mkdir(mode=0o700)
        target = Path(self.dir.name) / "elsewhere.sqlite3"
        self.path.symlink_to(target)
        with self.assertRaisesRegex(LedgerError, "not a link"):
            Ledger(self.path).close()
        self.assertFalse(target.exists())

    def test_a_database_swapped_in_before_sqlite_opens_is_refused(self):
        connect = sqlite3.connect

        def swap_then_connect(path, *args, **kwargs):
            spare = self.journal / "spare"
            plant(spare)
            os.replace(spare, self.path)
            return connect(path, *args, **kwargs)

        with mock.patch.object(ledger_module.sqlite3, "connect", swap_then_connect):
            with self.assertRaises(LedgerError):
                Ledger(self.path)
        self.assertEqual(os.path.getsize(self.path), 0, "the journal wrote into the planted file")

    def test_a_database_created_in_the_gap_is_refused(self):
        private_dir = ledger_module.private_dir

        def check_then_plant(directory):
            identity = private_dir(directory)
            plant(self.path)
            return identity

        with mock.patch.object(ledger_module, "private_dir", check_then_plant):
            with self.assertRaisesRegex(LedgerError, "mode 600"):
                Ledger(self.path)
        self.assertEqual(os.path.getsize(self.path), 0, "the journal wrote into the planted file")

    def test_a_wal_planted_before_sqlite_opens_is_refused(self):
        connect = sqlite3.connect

        def plant_then_connect(path, *args, **kwargs):
            plant(Path(str(self.path) + "-wal"))
            return connect(path, *args, **kwargs)

        with mock.patch.object(ledger_module.sqlite3, "connect", plant_then_connect):
            with self.assertRaisesRegex(LedgerError, "mode 600"):
                Ledger(self.path)


if __name__ == "__main__":
    unittest.main()
