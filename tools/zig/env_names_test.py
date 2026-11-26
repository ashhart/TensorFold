"""env_names.py: reads found by lookup call, computed lookups failing closed, scopes, gates and the doc's rows."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import env_lookups
import env_names
import env_scope

READS = '''const std = @import("std");
const key_name = "TF_BY_CONSTANT";
pub fn path(env: ?*const std.process.Environ.Map) ?[]const u8 {
    const e = env orelse return null;
    return e.get("HOME") orelse e.get("USERPROFILE");
}
fn proxies(m: *const std.process.Environ.Map) void {
    _ = m.get("https_proxy");
    _ = std.c.getenv("PATH");
    _ = std.c.getenv(key_name);
}
fn knobs(get: *const fn ([]const u8) ?[]const u8) void {
    _ = get("TMPDIR");
}
// _ = std.c.getenv("TF_IN_A_COMMENT");
test "a test block" {
    _ = std.c.getenv("TF_IN_A_TEST");
}
const label = "TF_LOOKS_LIKE_ONE";
'''
COMPUTED = '''const std = @import("std");
fn lookup(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}
pub fn read() void {
    _ = lookup("TF_THROUGH_WRAPPER");
    for ([_][:0]const u8{ "TF_LISTED_A", "TF_LISTED_B" }) |n| _ = std.c.getenv(n);
}
'''
GATED = '''const std = @import("std");
pub fn main() void {
    _ = if (comptime @import("build_options").slide_test_hooks) hook() else 0;
}
fn hook() u32 {
    _ = std.c.getenv("TF_HOOK");
    return 0;
}
'''
GRAPH = "\n".join(["M\t0\tzig/src/server.zig", "M\t1\tzig/src/tool.zig", "M\t2\tzig/src/only_tests.zig",
                   "M\t3\tzig/src/lib/root.zig", "I\t0\tlib\t3", "I\t1\tlib\t3",
                   "C\texe\ttensorfold-native\t0\tdist", "C\texe\ttf-tool\t1\t-", "C\ttest\ttest\t2\t-", ""])


def tree(root: Path, files: dict[str, str]) -> list:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return env_names.sources(root)


class Reads(unittest.TestCase):
    def test_lookups_by_call_find_lowercase_and_underscore_free_names(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(env_lookups, "WRAPPERS", ()), \
                mock.patch.object(env_lookups, "NAMED", {}):
            found, problems = env_names.reads(tree(Path(d), {"zig/src/reads.zig": READS}))
        names = {n for r in found for n in r.names}
        self.assertEqual(names, {"HOME", "USERPROFILE", "https_proxy", "PATH", "TF_BY_CONSTANT", "TMPDIR"})
        self.assertEqual(problems, [])

    def test_a_shaped_literal_no_lookup_reads_needs_an_explanation(self):
        shaped = {"TF_LOOKS_LIKE_ONE": "zig/src/reads.zig:19"}
        found = env_names.problems({}, {}, shaped)
        self.assertIn("no lookup reads it", found[0])
        with mock.patch.object(env_lookups, "NOT_READ", {"TF_LOOKS_LIKE_ONE": "a label"}):
            self.assertEqual(env_names.problems({}, {}, shaped), [])

    def test_a_computed_lookup_fails_closed_until_listed(self):
        files = {"zig/src/computed.zig": COMPUTED}
        with tempfile.TemporaryDirectory() as d:
            srcs = tree(Path(d), files)
            with mock.patch.object(env_lookups, "WRAPPERS", ()), mock.patch.object(env_lookups, "NAMED", {}):
                _, problems = env_names.reads(srcs)
            self.assertEqual(len(problems), 2)
            self.assertTrue(all("computed lookup" in p for p in problems))
            wrapper = (("zig/src/computed.zig", "lookup", 0, "file"),)
            named = {("zig/src/computed.zig", "std.c.getenv(n)"): ("TF_LISTED_A", "TF_LISTED_B")}
            with mock.patch.object(env_lookups, "WRAPPERS", wrapper), mock.patch.object(env_lookups, "NAMED", named):
                found, problems = env_names.reads(srcs)
            self.assertEqual(problems, [])
            self.assertEqual({n for r in found for n in r.names}, {"TF_THROUGH_WRAPPER", "TF_LISTED_A", "TF_LISTED_B"})
            stale = {("zig/src/computed.zig", "std.c.getenv(gone)"): ("TF_GONE",)} | named
            with mock.patch.object(env_lookups, "WRAPPERS", wrapper), mock.patch.object(env_lookups, "NAMED", stale):
                _, problems = env_names.reads(srcs)
            self.assertEqual(len(problems), 1)
            self.assertTrue(problems[0].startswith("warning: ") and "matches no computed lookup" in problems[0])


@mock.patch.object(env_names, "GATES", {("zig/src/server/main.zig", "hook"): "slide_test_hooks"})
class Scopes(unittest.TestCase):
    def test_build_roots_test_blocks_and_gates_set_scopes(self):
        files = {"zig/src/server.zig": 'const lib = @import("lib");\ntest {\n    _ = @import("helper.zig");\n}\n',
                 "zig/src/tool.zig": 'const lib = @import("lib");\nconst t = @import("tool_only.zig");\n',
                 "zig/src/only_tests.zig": 'const h = @import("helper.zig");\n', "zig/src/helper.zig": "",
                 "zig/src/tool_only.zig": "", "zig/src/lib/root.zig": 'const x = @import("x.zig");\n',
                 "zig/src/lib/x.zig": "", "zig/src/server/main.zig": GATED}
        with tempfile.TemporaryDirectory() as d:
            srcs = tree(Path(d), files)
            reached = env_scope.reach(Path(d), GRAPH)
            self.assertEqual(reached["zig/src/lib/x.zig"], {"release", "tool"})
            self.assertEqual(reached["zig/src/tool_only.zig"], {"tool"})
            self.assertEqual(reached["zig/src/helper.zig"], {"test"})
            with mock.patch.object(env_lookups, "WRAPPERS", ()), mock.patch.object(env_lookups, "NAMED", {}):
                found, _ = env_names.reads(srcs)
            self.assertEqual(env_names.scopes(found, reached, srcs), {"TF_HOOK": {"test"}})
            self.assertEqual(env_names.gate_problems(srcs), [])
            loose = GATED.replace("_ = if (comptime", "_ = hook();\n    _ = if (comptime")
            srcs = tree(Path(d), {"zig/src/server/main.zig": loose})
            self.assertIn("called outside slide_test_hooks", env_names.gate_problems(srcs)[0])


@mock.patch.object(env_lookups, "NOT_READ", {})
class Rows(unittest.TestCase):
    def test_rows_need_known_tags_and_matching_scopes(self):
        names = {"HOME": {"release", "test"}, "TF_DEV": {"tool"}}
        ok = {"HOME": ("release, test", "writes-content, loads"), "TF_DEV": ("tool", "logs")}
        self.assertEqual(env_names.problems(names, ok, {}), [])
        bad = {"HOME": ("tool, test", "untagged"), "TF_DEV": ("tool", "logs"), "TF_GONE": ("tool", "other")}
        found = env_names.problems(names, bad, {})
        self.assertTrue(any("TF_GONE is listed but nothing reads it" in f for f in found))
        self.assertTrue(any("HOME scope 'tool, test'" in f for f in found))
        self.assertTrue(any("HOME has tag 'untagged'" in f for f in found))

    def test_written_keeps_tags_and_brings_new_reads_in_untagged(self):
        doc = "# Names\n\nIntro.\n\n## Variables\n\n| Name | Scope | Tags |\n| --- | --- | --- |\n" \
              "| `https_proxy` | release | other |\n| `TF_GONE` | tool | logs |\n"
        out = env_names.written(doc, {"https_proxy": {"release"}, "TF_NEW": {"tool", "test"}})
        self.assertTrue(out.startswith("# Names\n\nIntro.\n\n## Variables\n"))
        self.assertEqual(env_names.listed(out), {"TF_NEW": ("tool, test", "untagged"), "https_proxy": ("release", "other")})
        self.assertEqual(env_names.written(out, {"https_proxy": {"release"}, "TF_NEW": {"tool", "test"}}), out)
        other_host = out.replace("| `https_proxy` | release |", "| `https_proxy` | release, tool |")
        rewritten = env_names.written(other_host, {"https_proxy": {"release"}, "TF_NEW": {"tool", "test"}})
        self.assertEqual(env_names.listed(rewritten)["https_proxy"], ("release, tool", "other"))


if __name__ == "__main__":
    unittest.main()
