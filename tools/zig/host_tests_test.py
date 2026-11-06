"""Real Zig build graphs must include new host checks and keep reviewed external requirements outside CI."""
import os
from pathlib import Path
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
    "test-metal-residency-owners-host",
    "test-exl3-prompt-host",
    "test-qwen35-draft-frame-host",
    "test-qwen35-frame-invalidation-host",
    "test-gemma4-affine-host",
    "test-gradient-serving",
    "test-qwen-serving-host",
)


@unittest.skipUnless(sys.platform.startswith("linux"), "test-host is a native Linux CI aggregate")
class HostTestGraphTests(unittest.TestCase):
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
        self.assertNotIn("test-host", result.stdout)


if __name__ == "__main__":
    unittest.main()
