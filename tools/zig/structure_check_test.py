"""Import cycles from planted trees: inherited cycles warn, a new cycle edge is an error, a removed one is counted."""
import contextlib
import io
import unittest

import structure_check as sc


def tree(*imports, extra=None):
    """Planted zig/src files, one per 'module:target' pair; the target is a named import."""
    files = {}
    for pair in imports:
        here, there = pair.split(":")
        name = f"zig/src/{here}/{here.replace('/', '_')}_{there.replace('/', '_')}.zig"
        files[name] = f'const x = @import("{there.split("/")[-1]}");\n'
        files.setdefault(f"zig/src/{there}/root.zig", "")
    files.update(extra or {})
    return files


def errors(found):
    return [m for level, m in found if level == "ERROR"]


def warnings(found):
    return [m for level, m in found if level == "WARN"]


class Ratchet(unittest.TestCase):
    def test_no_cycle_is_silent(self):
        found, counts = sc.compare(tree("a:b", "b:c"), tree("a:b", "b:c", "a:c"))
        self.assertEqual(found, [])
        self.assertEqual((counts["cycle_base"], counts["cycle_head"], counts["new_cycle"]), (0, 0, 0))

    def test_an_inherited_cycle_warns_once_and_never_errors(self):
        found, counts = sc.compare(tree("a:b", "b:a"), tree("a:b", "b:a", "c:a"))
        self.assertEqual(errors(found), [])
        self.assertEqual(warnings(found), ["inherited import cycle of 2 modules (2 edges): a, b"])
        self.assertEqual((counts["cycle_head"], counts["new_cycle"], counts["removed_cycle"]), (2, 0, 0))

    def test_a_new_import_closing_a_cycle_errors_on_every_edge_now_in_it(self):
        found, counts = sc.compare(tree("a:b", "b:c"), tree("a:b", "b:c", "c:a"))
        self.assertEqual(len(errors(found)), 3)
        self.assertIn("c -> a: new import closes a cycle (zig/src/c/c_a.zig imports a)", errors(found)[2])
        self.assertIn("a -> b: existing import now in a cycle", errors(found)[0])
        self.assertTrue(all(m.endswith("cycle of 3 modules: a, b, c") for m in errors(found)))
        self.assertEqual(warnings(found), [])
        self.assertEqual(counts["new_cycle"], 3)

    def test_a_new_edge_inside_an_inherited_cycle_errors_and_only_that_edge(self):
        found, _ = sc.compare(tree("a:b", "b:a", "a:c"), tree("a:b", "b:a", "a:c", "b:c", "c:a"))
        self.assertEqual([m.split(":")[0] for m in errors(found)], ["a -> c", "b -> c", "c -> a"])
        found, _ = sc.compare(tree("a:b", "b:a", "b:c", "c:a"), tree("a:b", "b:a", "b:c", "c:a", "c:b"))
        self.assertEqual([m.split(":")[0] for m in errors(found)], ["c -> b"])

    def test_a_second_file_behind_an_inherited_edge_is_not_new(self):
        more = {"zig/src/b/other.zig": '@import("a")'}
        found, _ = sc.compare(tree("a:b", "b:a"), tree("a:b", "b:a", extra=more))
        self.assertEqual(errors(found), [])

    def test_a_removed_cycle_passes_and_is_counted(self):
        found, counts = sc.compare(tree("a:b", "b:c", "c:a"), tree("a:b", "b:c"))
        self.assertEqual(found, [])
        self.assertEqual((counts["cycle_base"], counts["cycle_head"], counts["removed_cycle"]), (3, 0, 3))


class Graph(unittest.TestCase):
    def test_modules_are_dirs_families_and_top_level_files(self):
        self.assertEqual(sc.module_of("zig/src/core/lanes/lanes.zig"), "core")
        self.assertEqual(sc.module_of("zig/src/families/glm/cuda/x.zig"), "families/glm")
        self.assertEqual(sc.module_of("zig/src/main.zig"), "main.zig")

    def test_relative_imports_resolve_and_paths_outside_zig_src_are_not_modules(self):
        files = {"zig/src/families/glm/a.zig": 'const q = @import("../qwen/b.zig");\nconst k = @import("../../../k/x.zig");\n'
                 'const s = @import("sub/c.zig");\nconst m = @import("../qwen/missing.zig");\n',
                 "zig/src/families/glm/sub/c.zig": "", "zig/src/families/qwen/b.zig": '@import("../glm/a.zig")'}
        edges, unresolved = sc.graph(files)
        self.assertEqual(set(edges), {("families/glm", "families/qwen"), ("families/qwen", "families/glm")})
        self.assertEqual(list(unresolved), ["zig/src/families/qwen/missing.zig (no such file)"])

    def test_named_imports_resolve_only_from_literal_evidence(self):
        build = {"build.zig": 'const s = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig") });\n'
                              'm.addImport("sys", s);\nm.addImport("fn_bound", w.thing(target));\n',
                 "zig/build/x.zig": 'm.addImport("glm", b.createModule(.{ .root_source_file = '
                                    'b.path("zig/src/families/glm/root.zig") }));\n'
                                    'm.addImport("two", b.createModule(.{ .root_source_file = b.path("zig/src/a/t.zig") }));\n'
                                    'n.addImport("two", b.createModule(.{ .root_source_file = b.path("zig/src/b/t.zig") }));\n'}
        files = {"zig/src/core/c.zig": '@import("sys"); @import("glm"); @import("std"); @import("builtin");\n'
                                       '@import("fn_bound"); @import("two"); @import("platform"); @import("nope");\n',
                 "zig/src/platform/sys.zig": "", "zig/src/families/glm/root.zig": "", **build}
        edges, unresolved = sc.graph(files)
        self.assertEqual(set(edges), {("core", "platform"), ("core", "families/glm")})
        self.assertEqual(sorted(unresolved), ['@import("fn_bound") (no zig/src root)', '@import("nope") (no zig/src root)',
                                              '@import("two") (two modules)'])

    def test_commented_and_quoted_imports_are_not_edges(self):
        text = ('// const a = @import("a");\nconst s = "// not a comment"; const b = @import("b");\n'
                '    \\\\ @import("c") in a multiline string\nconst q = \'"\'; // @import("d")\n')
        files = {"zig/src/x/x.zig": text, **{f"zig/src/{m}/r.zig": "" for m in "abcd"}}
        self.assertEqual(set(sc.graph(files)[0]), {("x", "b")})

    def test_a_new_unresolved_name_warns_and_an_inherited_one_does_not(self):
        base = {"zig/src/a/a.zig": '@import("gen_old")'}
        head = {"zig/src/a/a.zig": '@import("gen_old"); @import("gen_new")'}
        found, counts = sc.compare(base, head)
        self.assertEqual(len(warnings(found)), 1)
        self.assertIn('@import("gen_new") (no zig/src root): unresolved', warnings(found)[0])
        self.assertEqual(counts["unresolved"], 2)

    def test_components_find_separate_cycles_and_a_long_one_without_recursion(self):
        edges = {("a", "b"): 1, ("b", "a"): 1, ("c", "d"): 1, ("d", "e"): 1, ("e", "c"): 1, ("e", "a"): 1}
        self.assertEqual(sc.components(edges), [["c", "d", "e"], ["a", "b"]])
        ring = {(f"m{i:05}", f"m{(i + 1) % 5000:05}"): 1 for i in range(5000)}
        self.assertEqual(len(sc.components(ring)[0]), 5000)
        self.assertEqual(sc.components({(f"m{i:05}", f"m{i + 1:05}"): 1 for i in range(5000)}), [])

    def test_the_checkout_parses_and_reports(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(sc.main([]), 0)
        self.assertRegex(out.getvalue(), r"^\d+ modules, \d+ module edges, \d+ cycle\(s\)")


if __name__ == "__main__":
    unittest.main()
