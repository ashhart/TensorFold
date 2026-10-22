"""Every option the build declares is declared for every target, so one build script fits every OS."""
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "build.zig"] + sorted((ROOT / "zig" / "build").glob("*.zig"))
# the option names this target's option list holds, with their kind
LISTED = re.compile(r"^  -D([A-Za-z0-9_-]+)=\[(\w+)\]", re.M)
# the options only one platform reads: the Metal language version, and the Step-5 host regression filter
PLATFORM_ONLY = {"metal-std", "step5-filter"}
# the options the acceptance test does not pass: a release version and the test hooks cannot be combined
NOT_PASSED = {"version", "slide-test-hooks"}
# the options whose value the build reads while it configures, with a value it accepts
VALUES = {"hip-arch": "gfx1151", "sm": "121", "optimize": "fast", "fuzz-optimize": "safe"}


def declared() -> dict[str, list[str]]:
    """Every option name the build declares, with the file and line that declares it."""
    found: dict[str, list[str]] = {}
    for path in SOURCES:
        text = path.read_text()
        for match in re.finditer(r"\bb\.option\(", text):
            for name in names(text, match.end()):
                where = f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, match.start()) + 1}"
                found.setdefault(name, []).append(where)
    return found


def names(text: str, start: int) -> list[str]:
    """The option names a b.option(...) call declares: the quoted names in its second argument."""
    type_end = argument(text, start)
    if type_end < 0:
        return []
    name_end = argument(text, type_end + 1)
    if name_end < 0:
        name_end = len(text)
    return re.findall(r'"([^"]*)"', text[type_end + 1:name_end])


def argument(text: str, start: int) -> int:
    """The index of the comma that ends the argument starting at start, or -1 when the call ends first."""
    depth, index = 1, start
    while index < len(text):
        char = text[index]
        if char == '"':
            end = text.find('"', index + 1)
            if end < 0:
                return -1
            index = end + 1
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return -1
        elif char == "," and depth == 1:
            return index
        index += 1
    return -1


def compiler() -> str:
    """The Zig compiler to run: the ZIG environment variable, or the one on PATH."""
    found = os.environ.get("ZIG") or shutil.which("zig")
    if not found:
        raise unittest.SkipTest("no Zig compiler found (set ZIG, or put zig on PATH)")
    return found


def build_help(*args: str) -> str:
    """The output of `zig build --help` with the given extra arguments."""
    done = subprocess.run([compiler(), "build", "--help", *args], cwd=ROOT, capture_output=True, text=True)
    done.check_returncode()
    return done.stdout + done.stderr


class BuildOptionTests(unittest.TestCase):
    def listed(self) -> dict[str, str]:
        """The options this target declares, with their kind."""
        return {m.group(1): m.group(2) for m in LISTED.finditer(build_help())}

    def test_every_declared_option_is_available_here(self):
        """Every option the build declares anywhere is in this target's option list."""
        listed = self.listed()
        missing = {n: w for n, w in declared().items() if n not in listed and n not in PLATFORM_ONLY}
        self.assertEqual({}, missing, f"options this target does not declare: {missing}")

    def test_every_declared_option_is_accepted(self):
        """Passing every declared option at once is not refused, the way -Dnvcc was on macOS."""
        listed = self.listed()
        args = []
        for name in declared():
            if name in PLATFORM_ONLY or name in NOT_PASSED:
                continue
            kind = listed.get(name, "string")
            if kind == "enum":
                self.assertIn(name, VALUES, f"enum option needs a valid acceptance value: {name}")
            args.append(f"-D{name}={VALUES.get(name, 'true' if kind == 'bool' else 'value')}")
        out = build_help(*args)
        self.assertNotIn("invalid option", out, out[:400])


class BuildOptionHarnessTests(unittest.TestCase):
    def test_failed_acceptance_command_is_rejected(self):
        """A failed command without the words invalid option must still fail the acceptance test."""
        listing = subprocess.CompletedProcess([], 0, "  -Dfuzz-optimize=[enum]\n", "")
        failed = subprocess.CompletedProcess([], 1, "", "configuration failed")
        gate = BuildOptionTests("test_every_declared_option_is_accepted")
        with patch(__name__ + ".compiler", return_value="zig"), \
                patch(__name__ + ".declared", return_value={"fuzz-optimize": []}), \
                patch.object(subprocess, "run", side_effect=[listing, failed]):
            with self.assertRaises(subprocess.CalledProcessError):
                gate.test_every_declared_option_is_accepted()

    def test_acceptance_passes_valid_values_for_both_optimize_enums(self):
        """Both OptimizeMode options receive a valid value rather than the generic string placeholder."""
        gate = BuildOptionTests("test_every_declared_option_is_accepted")
        options = {"optimize": "enum", "fuzz-optimize": "enum"}
        with patch.object(gate, "listed", return_value=options), \
                patch(__name__ + ".declared", return_value={name: [] for name in options}), \
                patch(__name__ + ".build_help", return_value="") as help_command:
            gate.test_every_declared_option_is_accepted()
        self.assertEqual(("-Doptimize=fast", "-Dfuzz-optimize=safe"), help_command.call_args.args)

    def test_enum_without_a_valid_value_is_rejected(self):
        """A newly declared enum needs a valid test value before the acceptance command can run."""
        gate = BuildOptionTests("test_every_declared_option_is_accepted")
        with patch.object(gate, "listed", return_value={"new-mode": "enum"}), \
                patch(__name__ + ".declared", return_value={"new-mode": []}), \
                patch(__name__ + ".build_help", return_value="") as help_command:
            with self.assertRaises(AssertionError):
                gate.test_every_declared_option_is_accepted()
            help_command.assert_not_called()
