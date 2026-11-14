"""A restored build cache stays cached when only file ids changed, and still misses when contents changed."""
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/zig/zig_cache_restat.zig"
BUILD = """const std = @import("std");

pub fn build(b: *std.Build) void {
    const from_c = b.addObject(.{ .name = "from_c", .root_module = b.createModule(.{ .target = b.graph.host }) });
    from_c.root_module.addCSourceFile(.{ .file = b.path("src/hello.c") });
    b.getInstallStep().dependOn(&from_c.step);
    const from_zig = b.addExecutable(.{ .name = "from_zig", .root_module = b.createModule(.{
        .root_source_file = b.path("src/hello.zig"),
        .target = b.graph.host,
    }) });
    const run = b.addRunArtifact(from_zig);
    run.addFileInput(b.path("src/input.txt"));
    run.expectExitCode(0);
    b.getInstallStep().dependOn(&run.step);
}
"""
# The same steps, but the executable links the C object, so a changed C file reaches all three.
LINKED = BUILD.replace("    const run = b.addRunArtifact",
                       "    from_zig.root_module.addObject(from_c);\n    const run = b.addRunArtifact")
SOURCES = {
    "src/hello.c": "int hello(void) { return 7; }\n",
    "src/hello.zig": "pub fn main() void {}\n",
    "src/input.txt": "same contents\n",
}


class RestatTests(unittest.TestCase):
    def setUp(self):
        self.zig = os.environ.get("ZIG") or shutil.which("zig")
        if not self.zig:
            self.skipTest("no Zig compiler found (set ZIG, or put zig on PATH)")
        temporary = tempfile.TemporaryDirectory(prefix="tensorfold-restat-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / ".zig-cache"
        self.env = {**os.environ, "ZIG_LOCAL_CACHE_DIR": str(self.cache), "ZIG_GLOBAL_CACHE_DIR": str(self.cache)}
        (self.root / "src").mkdir()
        (self.root / "build.zig").write_text(BUILD, newline="\n")
        for path, text in SOURCES.items():
            (self.root / path).write_text(text, newline="\n")
        lib = Path(self.zig).resolve().parent / "lib"
        self.restat_command = [self.zig, "run", "-OReleaseFast", "-mcpu=baseline", str(TOOL), "--", str(self.cache),
                               str(lib)]

    def run_zig(self, command):
        result = subprocess.run(command, cwd=self.root, env=self.env, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def restat_then_build(self, *extra):
        self.restat_line = self.run_zig(self.restat_command + list(extra)).strip().splitlines()[-1]
        print(self.restat_line)
        summary = self.run_zig([self.zig, "build", "--summary", "all"]).splitlines()
        steps = {}
        for name, marker in (("c", "compile obj from_c"), ("zig", "compile exe from_zig"), ("run", "run exe from_zig")):
            line = next(line for line in summary if marker in line and "(reused)" not in line)
            steps[name] = " cached" in line
        return steps

    def new_ids(self):
        for path in SOURCES:
            file = self.root / path
            copy = file.with_suffix(".copy")
            shutil.copyfile(file, copy)
            file.unlink()
            copy.rename(file)

    def restore_cache(self):
        saved = self.root / "saved-cache"
        shutil.copytree(self.cache, saved)
        shutil.rmtree(self.cache)
        saved.rename(self.cache)

    def test_only_a_real_change_misses(self):
        self.run_zig([self.zig, "build"])
        all_cached = {"c": True, "zig": True, "run": True}
        self.new_ids()
        self.assertEqual(self.restat_then_build(), all_cached, "a new file id alone must not miss")
        self.new_ids()
        self.restore_cache()
        self.assertEqual(self.restat_then_build(), all_cached, "a restored cache must not miss")
        (self.root / "src/input.txt").write_text("other contents\n", newline="\n")
        self.assertEqual(self.restat_then_build(), {"c": True, "zig": True, "run": False}, "a changed input reruns")
        (self.root / "src/hello.c").write_text("int hello(void) { return 8; }\n", newline="\n")
        self.assertEqual(self.restat_then_build(), {"c": False, "zig": True, "run": True}, "changed C compiles")
        (self.root / "src/hello.zig").write_text("pub fn main() void {\n    _ = 1;\n}\n", newline="\n")
        self.assertEqual(self.restat_then_build(), {"c": True, "zig": False, "run": False}, "changed Zig compiles, runs")


    def test_prune_keeps_one_build(self):
        (self.root / "build.zig").write_text(LINKED, newline="\n")
        calls = "extern fn hello() c_int;\n\npub fn main() void {\n    _ = hello();\n}\n"
        (self.root / "src/hello.zig").write_text(calls, newline="\n")
        self.run_zig([self.zig, "build"])
        self.new_ids()
        self.restore_cache()
        all_cached = {"c": True, "zig": True, "run": True}
        self.assertEqual(self.restat_then_build("--prune"), all_cached, "pruning keeps what can still hit")
        self.assertIn(", 0 pruned", self.restat_line)
        before = {path.name for path in (self.cache / "o").iterdir()}
        (self.root / "src/hello.c").write_text("int hello(void) { return 8; }\n", newline="\n")
        none_cached = {"c": False, "zig": False, "run": False}
        self.assertEqual(self.restat_then_build("--prune"), none_cached, "a changed C file reaches all three")
        found = re.search(r"(\d+) pruned with (\d+) output folders", self.restat_line)
        pruned, folders = int(found[1]), int(found[2])
        self.assertGreaterEqual(pruned, 3, self.restat_line)
        self.assertGreaterEqual(folders, 2, self.restat_line)
        after = {path.name for path in (self.cache / "o").iterdir()}
        self.assertEqual(len(before - after), folders, "the pruned folders are the ones gone")
        self.assertEqual(len(after), len(before), "each pruned folder's build replaced it, so the cache didn't grow")


if __name__ == "__main__":
    unittest.main()
