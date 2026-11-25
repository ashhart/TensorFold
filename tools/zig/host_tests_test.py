"""Real Zig build graphs must include new host checks and keep reviewed external requirements outside CI."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "zig/build/host_tests.zig"
EXCLUDED = (
    "test-grammar-budget",
    "test-grammar-tools",
    "check-release-paths",
    "test-startup-states-timing",
    "test-imprint-ops-timing",
    "test-host-draw-timing",
    "test-qwen35-draft-frame-host",
    "test-qwen35-frame-invalidation-host",
    "test-gradient-serving",
)


@unittest.skipUnless(sys.platform.startswith("linux"), "test-host is a native Linux CI aggregate")
class HostTestGraphTests(unittest.TestCase):
    ASSIGN = '        if (shardFor(name) == 0) shard_a.dependOn(step) else shard_b.dependOn(step);'
    def setUp(self):
        self.zig = os.environ.get("ZIG") or shutil.which("zig")
        if not self.zig:
            self.skipTest("no Zig compiler found (set ZIG, or put zig on PATH)")
        temporary = tempfile.TemporaryDirectory(prefix="tensorfold-host-graph-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copyfile(POLICY, self.root / "host_tests.zig")

    def build(self, extra="", omit=None, args=None):
        excluded = "\n".join(
            f'    b.step("{name}", "external requirement").dependOn(&b.addFail("external gate ran").step);'
            for name in EXCLUDED if name != omit
        )
        source = '''const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    defer @import("host_tests.zig").wire(b, target);
''' + excluded + "\n" + extra + "\n}\n"
        (self.root / "build.zig").write_text(source)
        return subprocess.run([self.zig, "build", *(args or ["test-host", "--summary", "all"])],
                              cwd=self.root, capture_output=True, text=True, timeout=120)

    def assert_failed_with(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def patch_policy(self, old, new):
        policy = self.root / "host_tests.zig"
        source = policy.read_text()
        self.assertIn(old, source)
        policy.write_text(source.replace(old, new, 1))

    def test_new_named_host_test_is_included_after_the_collector_is_declared(self):
        result = self.build('    b.step("test-new-host", "new host test").dependOn(&b.addFail("new host test reached").step);')
        self.assert_failed_with(result, "new host test reached")

    def test_existing_unnamed_unit_tests_are_preserved(self):
        result = self.build('    b.step("test", "existing units").dependOn(&b.addFail("existing units reached").step);')
        self.assert_failed_with(result, "existing units reached")

    def test_new_compile_only_check_is_included(self):
        (self.root / "broken.zig").write_text('comptime { @compileError("new compile check reached"); }\n')
        result = self.build('''    const check = b.addObject(.{ .name = "compile-only", .root_module = b.createModule(.{
        .root_source_file = b.path("broken.zig"), .target = target,
    }) });
    b.step("check-new-host", "compile only").dependOn(&check.step);''')
        self.assert_failed_with(result, "new compile check reached")

    def test_fixture_gates_are_the_reviewed_exclusions(self):
        block = POLICY.read_text().split("const exclusions = [_]Exclusion{", 1)[1].split("};", 1)[0]
        self.assertEqual(sorted(EXCLUDED), sorted(re.findall(r'\.name = "([^"]+)"', block)))

    def test_reviewed_external_requirements_are_not_executed(self):
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("external gate ran", result.stdout + result.stderr)

    def test_alias_to_an_excluded_check_is_rejected(self):
        result = self.build('''    b.step("test-alias", "alias").dependOn(&b.top_level_steps.get("test-grammar-budget").?.step);''')
        self.assert_failed_with(result, "test-host reaches excluded test-grammar-budget")
        self.assertNotIn("external gate ran", result.stdout + result.stderr)

    def test_direct_alias_to_an_excluded_action_is_rejected(self):
        result = self.build('''    const external = b.top_level_steps.get("test-grammar-tools").?;
    b.step("test-action-alias", "alias").dependOn(external.step.dependencies.items[0]);''')
        self.assert_failed_with(result, "test-host reaches excluded test-grammar-tools")
        self.assertNotIn("external gate ran", result.stdout + result.stderr)

    def test_empty_named_check_is_rejected(self):
        result = self.build('    _ = b.step("test-empty", "accidentally disconnected");')
        self.assert_failed_with(result, "named check test-empty has no build steps")

    def test_removed_external_step_requires_review_of_its_exclusion(self):
        result = self.build(omit="test-grammar-tools")
        self.assert_failed_with(result, "stale exclusion test-grammar-tools")

    def test_cross_target_does_not_acquire_a_native_host_aggregate(self):
        result = self.build(args=["--help", "-Dtarget=x86_64-windows-gnu"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotRegex(result.stdout, r"(?m)^\s+test-host\s")

    def test_new_named_host_test_is_in_exactly_one_shard(self):
        extra = '    b.step("test-new-host", "new host test").dependOn(&b.addFail("new host test reached").step);'
        a = self.build(extra=extra, args=["test-host-a", "--summary", "all"])
        b = self.build(extra=extra, args=["test-host-b", "--summary", "all"])
        out = a.stdout + a.stderr + b.stdout + b.stderr
        self.assertEqual(len([r for r in (a, b) if "new host test reached" in r.stdout + r.stderr]), 1, out)
        self.assertEqual(sum(r.returncode != 0 for r in (a, b)), 1, out)

    def test_a_check_in_both_shards_is_rejected(self):
        self.patch_policy(self.ASSIGN, "        shard_a.dependOn(step);\n        shard_b.dependOn(step);")
        extra = '    b.step("test-doubled", "doubled").dependOn(&b.addFail("doubled reached").step);'
        result = self.build(extra=extra)
        self.assert_failed_with(result, "check test-doubled is in both shards")
        self.assertNotIn("doubled reached", result.stdout + result.stderr)

    def test_a_check_in_neither_shard_is_rejected(self):
        self.patch_policy(self.ASSIGN, "        // deliberately unassigned")
        extra = '    b.step("test-orphan", "orphan").dependOn(&b.addFail("orphan reached").step);'
        result = self.build(extra=extra)
        self.assert_failed_with(result, "check test-orphan is in neither shard")
        self.assertNotIn("orphan reached", result.stdout + result.stderr)

    def test_test_host_runs_the_union_of_both_shards(self):
        extra = (
            '    b.step("test-cuda-host", "pinned to shard a").dependOn(&b.addFail("shard a reached").step);\n'
            '    b.step("test-golden", "pinned to shard b").dependOn(&b.addFail("shard b reached").step);'
        )
        result = self.build(extra=extra)
        self.assert_failed_with(result, "shard a reached")
        self.assertIn("shard b reached", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
