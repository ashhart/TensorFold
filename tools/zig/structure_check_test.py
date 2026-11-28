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


def family_files(*edges):
    """One named-family root per module, with the requested relative family imports."""
    files = {}
    for source, target in edges:
        files[f"zig/src/families/{source}/root.zig"] = f'@import("../{target}/root.zig")'
        files.setdefault(f"zig/src/families/{target}/root.zig", "")
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
        self.assertIn("c -> a: new import closes a cycle without the inherited one (zig/src/c/c_a.zig imports a)",
                      errors(found)[2])
        self.assertIn("a -> b: existing import closes a cycle", errors(found)[0])
        self.assertTrue(all(m.endswith("cycle of 3 modules: a, b, c") for m in errors(found)))
        self.assertEqual(warnings(found), [])
        self.assertEqual((counts["new_cycle"], counts["closing_cycle"]), (3, 3))

    def test_a_module_that_only_joins_the_inherited_cycle_warns(self):
        inherited = ("core:native", "native:core", "native:x", "x:core")
        found, counts = sc.compare(tree(*inherited), tree(*inherited, "native:fam", "fam:core", "fam:x"))
        self.assertEqual(errors(found), [])
        self.assertEqual([m.split(":")[0] for m in warnings(found)], ["fam -> core", "fam -> x", "native -> fam"])
        self.assertTrue(all("joins the inherited cycle of 4 modules" in m for m in warnings(found)))
        self.assertEqual((counts["new_cycle"], counts["closing_cycle"]), (3, 0))

    def test_a_cycle_that_closes_beside_the_inherited_one_errors_on_its_own_edges_only(self):
        inherited = ("core:native", "native:core")
        found, _ = sc.compare(tree(*inherited), tree(*inherited, "native:fam", "fam:native", "fam:core"))
        self.assertEqual([m.split(":")[0] for m in errors(found)], ["fam -> native", "native -> fam"])
        self.assertTrue(all(m.endswith("cycle of 2 modules: fam, native") for m in errors(found)))
        self.assertEqual([m.split(":")[0] for m in warnings(found)], ["fam -> core"])

    def test_a_new_edge_inside_an_inherited_cycle_errors_only_where_it_closes_alone(self):
        found, _ = sc.compare(tree("a:b", "b:a", "a:c"), tree("a:b", "b:a", "a:c", "b:c", "c:a"))
        self.assertEqual([m.split(":")[0] for m in errors(found)], ["a -> c", "c -> a"])
        self.assertEqual([m.split(":")[0] for m in warnings(found)], ["b -> c"])
        found, _ = sc.compare(tree("a:b", "b:a", "b:c", "c:a"), tree("a:b", "b:a", "b:c", "c:a", "c:b"))
        self.assertEqual(errors(found), [])

    def test_a_new_module_core_imports_errors_even_inside_the_inherited_cycle(self):
        inherited = ("core:native", "native:core", "native:hip", "hip:core")
        found, counts = sc.compare(tree(*inherited), tree(*inherited, "core:hip"))
        self.assertEqual(errors(found), ["core -> hip: core imports a new module (zig/src/core/core_hip.zig imports hip)"])
        self.assertEqual(counts["closing_cycle"], 0)
        found, _ = sc.compare(tree("core:platform"), tree("core:platform", "core:sys"))
        self.assertEqual(errors(found), ["core -> sys: core imports a new module (zig/src/core/core_sys.zig imports sys)"])
        more = {"zig/src/core/other.zig": '@import("native")'}
        self.assertEqual(errors(sc.compare(tree(*inherited), tree(*inherited, extra=more))[0]), [])

    def test_a_second_file_behind_an_inherited_edge_is_not_new(self):
        more = {"zig/src/b/other.zig": '@import("a")'}
        found, _ = sc.compare(tree("a:b", "b:a"), tree("a:b", "b:a", extra=more))
        self.assertEqual(errors(found), [])

    def test_a_removed_cycle_passes_and_is_counted(self):
        found, counts = sc.compare(tree("a:b", "b:c", "c:a"), tree("a:b", "b:c"))
        self.assertEqual(found, [])
        self.assertEqual((counts["cycle_base"], counts["cycle_head"], counts["removed_cycle"]), (3, 0, 3))


class CrossFamily(unittest.TestCase):
    def test_a_new_family_edge_errors_and_names_the_importing_file(self):
        base = family_files(("qwen4", "flashnext"))
        head = family_files(("qwen4", "flashnext"), ("glm", "qwen27"))
        found, counts = sc.compare(base, head)
        self.assertEqual(errors(found), [
            "families/glm -> families/qwen27: new cross-family import "
            "(zig/src/families/glm/root.zig imports ../qwen27/root.zig)",
        ])
        self.assertEqual((counts["family_base"], counts["family_head"], counts["new_family"]), (1, 2, 1))

    def test_qwen3_to_llama_is_always_allowed(self):
        files = family_files(("qwen3", "llama"))
        found, counts = sc.compare({}, files)
        self.assertEqual(found, [])
        self.assertEqual((counts["new_family"], counts["allowed_family"]), (0, 1))

    def test_a_removed_family_edge_passes_and_shrinks_the_baseline(self):
        base = family_files(("qwen4", "flashnext"), ("glm", "qwen27"))
        head = family_files(("qwen4", "flashnext"))
        found, counts = sc.compare(base, head)
        self.assertEqual(found, [])
        self.assertEqual((counts["family_base"], counts["family_head"], counts["removed_family"]), (2, 1, 1))

    def test_today_main_passes_unchanged_with_its_recorded_baseline(self):
        files = sc.checkout(sc.ROOT)
        found, counts = sc.compare(files, files)
        self.assertEqual(errors(found), [])
        self.assertEqual(
            (counts["family_base"], counts["family_head"], counts["new_family"],
             counts["removed_family"], counts["allowed_family"]),
            (11, 11, 0, 0, 1),
        )


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


class ImportSpelling(unittest.TestCase):
    SINGLE = 'const x = @import("{}");\n'
    MULTI = 'const x: @import(\n    "{}",\n).T = .{{}};\n'
    OLD = sc.re.compile(r'@import\("([^"]+)"\)')

    def scan(self, spelling, target):
        files = {"zig/src/a/user.zig": spelling.format(target), "zig/src/b/root.zig": "", "zig/src/b/x.zig": ""}
        return sc.graph(files)

    def test_single_and_multiline_give_the_same_edge(self):
        single, multi = self.scan(self.SINGLE, "../b/x.zig"), self.scan(self.MULTI, "../b/x.zig")
        self.assertIn(("a", "b"), single[0])
        self.assertEqual(single, multi)

    def test_single_and_multiline_give_the_same_unresolved_diagnostic(self):
        single, multi = self.scan(self.SINGLE, "nowhere"), self.scan(self.MULTI, "nowhere")
        self.assertEqual(len(single[1]), 1)
        self.assertEqual(single, multi)
        missing = self.scan(self.MULTI, "../b/missing.zig")
        self.assertEqual(missing[1], self.scan(self.SINGLE, "../b/missing.zig")[1])
        self.assertEqual(len(missing[1]), 1)

    def test_control_the_old_pattern_misses_the_multiline_spelling(self):
        multi = self.MULTI.format("../b/x.zig")
        self.assertEqual(self.OLD.findall(multi), [])
        self.assertEqual(sc.IMPORT.findall(multi), ["../b/x.zig"])
        self.assertEqual(self.OLD.findall(self.SINGLE.format("../b/x.zig")), ["../b/x.zig"])


class HelperScope(unittest.TestCase):
    """Two build helpers that reuse one module variable name: each addImport binds to its own helper's module."""
    GLM = ('fn addGlm(b: *std.Build, m: *std.Build.Module) void {{\n'
           '    const mod = b.createModule(.{{ .root_source_file = b.path("zig/src/families/glm/api.zig") }});\n'
           '    m.addImport("{}", mod);\n}}\n')
    CORE = ('fn addCore(b: *std.Build, m: *std.Build.Module) void {{\n'
            '    const mod = b.createModule(.{{ .root_source_file = b.path("zig/src/core/api.zig") }});\n'
            '    m.addImport("core_api", mod);\n}}\n')
    SOURCES = {"zig/src/core/user.zig": 'const g = @import("glm_api");\n', "zig/src/core/api.zig": "",
               "zig/src/families/glm/api.zig": ""}

    def build(self, *helpers, name="glm_api"):
        return {"build.zig": "".join(h.format(name) if h is self.GLM else h for h in helpers), **self.SOURCES}

    def test_two_helpers_keep_their_own_module_variable(self):
        files = self.build(self.GLM, self.CORE)
        self.assertEqual(sc.named_modules(files)["glm_api"], "families/glm")
        self.assertEqual(sc.named_modules(files)["core_api"], "core")
        self.assertEqual(set(sc.graph(files)[0]), {("core", "families/glm")})

    def test_the_new_core_dependency_behind_the_first_helper_errors(self):
        found, _ = sc.compare(dict(self.SOURCES, **{"zig/src/core/user.zig": ""}), self.build(self.GLM, self.CORE))
        self.assertEqual(errors(found), ["core -> families/glm: core imports a new module "
                                         "(zig/src/core/user.zig imports glm_api)"])

    def test_control_helper_order_does_not_change_the_result(self):
        first, second = self.build(self.GLM, self.CORE), self.build(self.CORE, self.GLM)
        self.assertEqual(sc.named_modules(first), sc.named_modules(second))
        self.assertEqual(sc.graph(first), sc.graph(second))

    def test_control_one_file_wide_dictionary_would_resolve_to_the_last_helper(self):
        for order, last in (((self.GLM, self.CORE), "core"), ((self.CORE, self.GLM), "families/glm")):
            text = sc.code(self.build(*order)["build.zig"])
            whole_file = {m[2]: sc.ROOT_PATH.findall(m[3]) for m in sc.MODULE_VAR.finditer(text)}
            self.assertEqual(sc.module_of(whole_file["mod"][0]), last)

    def test_a_variable_rebound_to_a_second_module_is_ambiguous_by_name(self):
        text = ('fn add(b: *std.Build, m: *std.Build.Module) void {\n'
                '    var mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/api.zig") });\n'
                '    if (x) mod = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/api.zig") });\n'
                '    m.addImport("either", mod);\n}\n')
        files = {"build.zig": text, "zig/src/native/user.zig": '@import("either");\n', **self.SOURCES}
        self.assertIsNone(sc.named_modules(files)["either"])
        self.assertIn('@import("either") (two modules)', sc.graph(files)[1])

    def test_a_nested_block_sees_its_helpers_variable_and_a_field_init_is_not_a_binding(self):
        text = ('fn add(b: *std.Build, m: *std.Build.Module) void {\n'
                '    const mod = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/api.zig") });\n'
                '    if (x) {\n        m.addImport("glm_api", mod);\n    }\n'
                '    _ = b.addTest(.{ .root_module = b.createModule(.{\n'
                '        .root_source_file = b.path("zig/src/core/t.zig") }) });\n'
                '    const s = "{";\n}\n')
        self.assertEqual(sc.named_modules({"build.zig": text, **self.SOURCES})["glm_api"], "families/glm")

    def test_a_use_with_no_binding_in_scope_takes_the_files_declarations_and_two_modules_are_ambiguous(self):
        param = 'fn wire(m: *std.Build.Module, mod: *std.Build.Module) void {\n    m.addImport("param_api", mod);\n}\n'
        files = {"zig/src/native/user.zig": '@import("param_api");\n', **self.SOURCES}
        one = sc.named_modules({**files, "build.zig": self.GLM.format("glm_api") + param})
        self.assertEqual(one["param_api"], "families/glm")
        two = {**files, "build.zig": self.GLM.format("glm_api") + self.CORE + param}
        self.assertIsNone(sc.named_modules(two)["param_api"])
        self.assertIn('@import("param_api") (two modules)', sc.graph(two)[1])


if __name__ == "__main__":
    unittest.main()
