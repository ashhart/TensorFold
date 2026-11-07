"""Scoped standard-library tests for bounded evidence rendering and safe local file replacement."""

import contextlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import render_evidence as renderer

TABLE = "| Cell | Outcome |\n| :--- | ---: |\n| historical run | interrupted |"


def block(body, newline="\n"):
    return (renderer.START + "\n" + body + "\n" + renderer.END + "\n").replace("\n", newline)


class RenderTests(unittest.TestCase):
    def test_preserves_surrounding_text_and_is_deterministic(self):
        source = "recipe context\n" + block(TABLE) + "recipe tail\n"
        target = "heading\n" + block("old") + "other evidence"
        expected = "heading\n" + block(TABLE) + "other evidence"
        self.assertEqual(renderer.render(source, target), expected)
        self.assertEqual(renderer.render(source, expected), expected)
        self.assertEqual(renderer.render(source, target), expected)

    def test_preserves_crlf_and_mixed_surrounding_text(self):
        target = "heading\nunchanged\r\n" + block("old", "\r\n") + "tail\nno final newline"
        expected = "heading\nunchanged\r\n" + block(TABLE, "\r\n") + "tail\nno final newline"
        self.assertEqual(renderer.render(block(TABLE), target), expected)

    def test_normalizes_crlf_source_to_lf_target(self):
        self.assertEqual(renderer.render(block(TABLE, "\r\n"), block("old")), block(TABLE))

    def test_allows_boundary_blank_lines_only(self):
        self.assertEqual(renderer.render(block("\n" + TABLE + "\n"), block("old")), block(TABLE))

    def test_rejects_missing_duplicate_reversed_and_nonstandalone_markers(self):
        invalid = ["no markers", block(TABLE) * 2, renderer.START + "\n",
                   renderer.END + "\n" + renderer.START + "\n", "prefix " + block(TABLE),
                   block(TABLE).replace(renderer.END, renderer.END + " suffix"),
                   block(TABLE).replace(renderer.START, " " + renderer.START),
                   block(TABLE) + "quoted " + renderer.START]
        for value in invalid:
            with self.subTest(source=value), self.assertRaises(ValueError):
                renderer.render(value, block("old"))
            with self.subTest(target=value), self.assertRaises(ValueError):
                renderer.render(block(TABLE), value)

    def test_rejects_malformed_delimiters(self):
        for delimiter in ("| heading | not delimiter |", "| -- | --- |", "| ---x | --- |",
                          "| :--: | --- |", "| --- | |", "| --- --- | --- |"):
            with self.subTest(delimiter=delimiter), self.assertRaises(ValueError):
                renderer.render(block(TABLE.replace("| :--- | ---: |", delimiter)), block("old"))

    def test_rejects_extra_or_missing_cells(self):
        for row in ("| a | b | c |", "| a |", "| a | b | c | d |"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                renderer.table(block("| a | b |\n| --- | --- |\n" + row))
        with self.assertRaises(ValueError):
            renderer.table(block(TABLE.replace("| :--- | ---: |", "| --- | --- | --- |")))

    def test_rejects_unsupported_table_shapes(self):
        invalid = ["", "not a table", "| header |\n| --- |", TABLE + "\n\n| extra | row |",
                   TABLE.replace("| Cell", "Cell"), TABLE.replace("Outcome |", "Outcome"),
                   TABLE.replace("| historical", " | historical"), TABLE + "\nprose",
                   TABLE.replace("historical run", "historical\rrun")]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                renderer.table(block(value))

    def test_supports_escaped_pipes_without_changing_text(self):
        body = r"| a\|b | code |" + "\n| --- | --- |\n" + r"| `x\|y` | a\\ |"
        self.assertEqual(renderer.table(block(body)), body)

    def test_rejects_unescaped_code_span_pipe_and_escaped_outer_pipe(self):
        for row in ("| `a|b` | value |", r"| a | b\|"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                renderer.table(block("| a | b |\n| --- | --- |\n" + row))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recipe = self.root / "recipe.md"
        self.readme = self.root / "readme.md"
        self.recipe.write_bytes(block(TABLE).encode())
        self.readme.write_bytes(("prefix\r\n" + block("old", "\r\n") + "suffix").encode())
        self.args = ["--recipe", str(self.recipe), "--readme", str(self.readme)]

    def run_cli(self, args=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return renderer.main(self.args if args is None else args)

    def assert_no_temporary_files(self):
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["readme.md", "recipe.md"])

    def test_check_stale_is_nonwriting_including_mode_and_mtime(self):
        before = self.readme.read_bytes()
        metadata = self.readme.stat()
        with mock.patch.object(renderer, "atomic_write") as write:
            self.assertEqual(self.run_cli(self.args + ["--check"]), 1)
        write.assert_not_called()
        self.assertEqual(self.readme.read_bytes(), before)
        self.assertEqual(self.readme.stat().st_mtime_ns, metadata.st_mtime_ns)
        self.assertEqual(self.readme.stat().st_mode, metadata.st_mode)
        self.assert_no_temporary_files()

    def test_atomic_render_preserves_mode_crlf_and_source(self):
        source = self.recipe.read_bytes()
        self.readme.chmod(0o640)
        replace = os.replace
        with mock.patch.object(renderer.os, "replace", wraps=replace) as replacement:
            self.assertEqual(self.run_cli(), 0)
        temporary, destination = replacement.call_args.args
        self.assertEqual(temporary.parent, self.readme.parent)
        self.assertEqual(destination, self.readme)
        self.assertEqual(stat.S_IMODE(self.readme.stat().st_mode), 0o640)
        self.assertEqual(self.readme.read_bytes(), ("prefix\r\n" + block(TABLE, "\r\n") + "suffix").encode())
        self.assertEqual(self.recipe.read_bytes(), source)
        self.assert_no_temporary_files()

    def test_current_check_and_repeated_render_do_not_replace(self):
        self.assertEqual(self.run_cli(), 0)
        before = self.readme.read_bytes()
        metadata = self.readme.stat()
        with mock.patch.object(renderer, "atomic_write") as write:
            self.assertEqual(self.run_cli(self.args + ["--check"]), 0)
            self.assertEqual(self.run_cli(), 0)
        write.assert_not_called()
        self.assertEqual(self.readme.read_bytes(), before)
        self.assertEqual(self.readme.stat().st_ino, metadata.st_ino)
        self.assertEqual(self.readme.stat().st_mtime_ns, metadata.st_mtime_ns)

    def test_malformed_source_or_target_returns_two_without_writing(self):
        for path in (self.recipe, self.readme):
            original = path.read_bytes()
            for malformed in ("missing markers", block(TABLE.replace("---:", "bad")),
                              block(TABLE.replace("historical run |", "extra | historical run |"))):
                if path == self.readme and malformed != "missing markers":
                    continue
                with self.subTest(path=path.name, malformed=malformed):
                    path.write_bytes(malformed.encode())
                    before = self.readme.read_bytes()
                    self.assertEqual(self.run_cli(), 2)
                    self.assertEqual(self.run_cli(self.args + ["--check"]), 2)
                    self.assertEqual(self.readme.read_bytes(), before)
            path.write_bytes(original)
        self.assert_no_temporary_files()

    def test_missing_input_returns_two(self):
        before = self.readme.read_bytes()
        for option in ("--recipe", "--readme"):
            args = self.args.copy()
            args[args.index(option) + 1] = str(self.root / "missing.md")
            self.assertEqual(self.run_cli(args), 2)
        self.assertEqual(self.readme.read_bytes(), before)
        self.assert_no_temporary_files()

    def test_invalid_utf8_returns_two_without_writing(self):
        before = self.readme.read_bytes()
        self.recipe.write_bytes(b"\xff")
        self.assertEqual(self.run_cli(), 2)
        self.assertEqual(self.readme.read_bytes(), before)

    def test_refuses_same_path_and_hardlink(self):
        original = self.recipe.read_bytes()
        for alias in (self.recipe, self.root / "hardlink.md"):
            if alias != self.recipe:
                os.link(self.recipe, alias)
            args = ["--recipe", str(self.recipe), "--readme", str(alias)]
            with self.subTest(alias=alias.name):
                self.assertEqual(self.run_cli(args), 2)
                self.assertEqual(self.run_cli(args + ["--check"]), 2)
                self.assertEqual(alias.read_bytes(), original)
                self.assertTrue(os.path.samefile(self.recipe, alias))
        self.assertEqual(self.recipe.read_bytes(), original)

    def test_refuses_symlink_destination_including_dangling_link(self):
        before = self.readme.read_bytes()
        for name in ("readme.md", "missing.md"):
            alias = self.root / (name + ".link")
            alias.symlink_to(self.root / name)
            args = ["--recipe", str(self.recipe), "--readme", str(alias)]
            with self.subTest(name=name):
                self.assertEqual(self.run_cli(args), 2)
                self.assertEqual(self.run_cli(args + ["--check"]), 2)
                self.assertTrue(alias.is_symlink())
        self.assertEqual(self.readme.read_bytes(), before)
        self.assertFalse((self.root / "missing.md").exists())

    def test_read_failure_does_not_write(self):
        before = self.readme.read_bytes()
        with mock.patch.object(Path, "read_bytes", side_effect=OSError("read failed")):
            self.assertEqual(self.run_cli(), 2)
        self.assertEqual(self.readme.read_bytes(), before)
        self.assert_no_temporary_files()

    def test_write_and_replace_failures_preserve_data_and_remove_temporary(self):
        before = self.readme.read_bytes()
        metadata = self.readme.stat()
        failures = (("fchmod", OSError("mode failed")), ("fsync", OSError("flush failed")),
                    ("replace", OSError("replace failed")))
        for operation, error in failures:
            with self.subTest(operation=operation), mock.patch.object(renderer.os, operation, side_effect=error):
                self.assertEqual(self.run_cli(), 2)
            self.assertEqual(self.readme.read_bytes(), before)
            self.assertEqual(self.readme.stat().st_ino, metadata.st_ino)
            self.assertEqual(self.readme.stat().st_mode, metadata.st_mode)
            self.assert_no_temporary_files()

    def test_temp_creation_failure_preserves_data(self):
        before = self.readme.read_bytes()
        with mock.patch.object(renderer.tempfile, "NamedTemporaryFile", side_effect=OSError("create failed")):
            self.assertEqual(self.run_cli(), 2)
        self.assertEqual(self.readme.read_bytes(), before)
        self.assert_no_temporary_files()

    def test_nonregular_input_returns_two(self):
        before = self.readme.read_bytes()
        for option in ("--recipe", "--readme"):
            args = self.args.copy()
            args[args.index(option) + 1] = str(self.root)
            self.assertEqual(self.run_cli(args), 2)
        self.assertEqual(self.readme.read_bytes(), before)

    def test_default_paths_are_repository_relative_and_ignore_cwd(self):
        expected_root = Path(renderer.__file__).resolve().parents[2]
        self.assertEqual(renderer.DEFAULT_RECIPE, expected_root / "docs/recipes/lfm3-native.md")
        self.assertEqual(renderer.DEFAULT_README, expected_root / "docs/receipts/lfm3/README.md")
        with mock.patch.object(renderer, "DEFAULT_RECIPE", self.recipe):
            with mock.patch.object(renderer, "DEFAULT_README", self.readme):
                self.assertEqual(self.run_cli(["--check"]), 1)

    def test_script_exit_codes_from_another_directory(self):
        command = [sys.executable, "-B", str(Path(renderer.__file__).resolve()), *self.args]
        for extra, expected in ((["--check"], 1), ([], 0), (["--check"], 0), (["--unknown"], 2)):
            result = subprocess.run(command + extra, cwd=self.root, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
