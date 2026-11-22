#!/usr/bin/env python3
"""Guard coverage and unchanged source math, plus negative fixtures for the SASS comparison gate."""
from pathlib import Path
import hashlib
import re
import unittest
from pdl_sass_diff import compare, TOUCHED

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'flashnext_hc.cu': 17, 'flashnext_batch.cu': 18, 'flashnext_attn.cu': 19, 'route_rows.cu': 4, 'flashnext_hc_dense.cuh': 4, 'flashnext_gdn_lanes.cuh': 4, 'flashnext_gdn_tpr.cuh': 2, 'flashnext_gdn_fused.cuh': 2, 'flashnext_qsa_tile.cuh': 1, 'flashnext_attn_lanes.cuh': 5}
MATH = {'flashnext_hc.cu': 'f3ce7affe407193a36ecb408b27a180a5979c06fdf44150e9a2a372a2c236f3f', 'flashnext_batch.cu': '32c046da3429ab30f0f3b50c4516f0acf836db22f040956f1cd8b4bf098a8275', 'flashnext_attn.cu': '2417f277d5d41cf3d8b5b78c6b5d24e00ab84f53feafe91a0af544b6ad6f6410', 'route_rows.cu': '4a3d8c544c0b0d599fa5915cf86ebbb73c56d420c04f575b6fac3ea4b95adc60', 'flashnext_hc_dense.cuh': '1dccf47826d45b9dd7705692e336541823de14f2ec233f5e131793df04b4c949', 'flashnext_gdn_lanes.cuh': '65cd83cb64cf1b8455fdf6e65c0584741c99a0a647d565f8c642820f44fe46bc', 'flashnext_gdn_tpr.cuh': 'ccf517c1d95e557a8787482cff051b031b1c420d80a6d3445d24a6a95ea4540c', 'flashnext_gdn_fused.cuh': '66dfb9642db22ac08a56e23ac394bbd1bdb9caa3e657b5c77b9d859e04b7bfb6', 'flashnext_qsa_tile.cuh': '021e3cb424a34bef75a756133ef3b4edfb6bbf770ff4b75842a667b763ec91ad', 'flashnext_attn_lanes.cuh': '594346762703dd665d8ec994e479ec71e8150b481c266e32a9e89f33d36b050c'}

def entries(text):
    out = []
    for m in re.finditer(r"\b__global__\b", text):
        end = re.search(r"[;{]", text[m.end():])
        if end and end[0] == "{":
            out.append(text[m.end() + end.end():])
    return out

def source_problems(text, count):
    bodies = entries(text)
    return (len(bodies) != count or any(not body.lstrip().startswith("flashnext_pdl::enter();")
            for body in bodies) or "__restrict__" in text or "__ldg" in text)

class Source(unittest.TestCase):
    def test_every_entry_waits_before_any_body_statement(self):
        for name, count in COUNTS.items():
            text = (ROOT / "zig/kernels/cuda" / name).read_text()
            self.assertFalse(source_problems(text, count), name)
            self.assertTrue(source_problems(text.replace("flashnext_pdl::enter();", "", 1), count), name)

    def test_all_original_math_and_reduction_order_stays_byte_identical(self):
        for name, digest in MATH.items():
            text = (ROOT / "zig/kernels/cuda" / name).read_text()
            text = text.replace(" flashnext_pdl::enter();", "").replace(" __restrict__", "")
            text = text.replace('#include "flashnext_pdl.cuh"\n', "")
            text = text.replace("#define TF_ROUTER_PDL\n", "").replace("#undef TF_ROUTER_PDL\n", "")
            self.assertEqual(digest, hashlib.sha256(text.encode()).hexdigest(), name)

    def test_guard_uses_existing_memory_clobber_dependency_primitives(self):
        text = (ROOT / "zig/kernels/cuda/flashnext_pdl.cuh").read_text()
        self.assertLess(text.index("qmm_frag::grid_launch();"), text.index("qmm_frag::grid_wait();"))
        text = (ROOT / "zig/kernels/cuda/common/qmm_frag.cuh").read_text()
        for instruction in ("griddepcontrol.wait;", "griddepcontrol.launch_dependents;"):
            self.assertIn('asm volatile("' + instruction + '\\n" ::: "memory")', text)

    def test_gate_has_all_modified_images_without_a_new_early_load_exception(self):
        text = (ROOT / "zig/build/cuda.zig").read_text()
        for name in TOUCHED:
            self.assertIn('std.mem.eql(u8, k.name, "' + name + '")', text)
        text = (ROOT / "tools/pdl_early_loads.py").read_text()
        self.assertNotIn("flashnext", text)

def fixture():
    good = "code for sm_121\nFunction : fn\n/*0000*/ ACQBULK;\n/*0010*/ LDG.E R0, [R2];\n"
    return {n + ".fatbin": good for n in TOUCHED} | {"qmmf.fatbin": "unchanged qmmf SASS"}

class Sass(unittest.TestCase):
    def test_unchanged_sass_passes(self):
        self.assertEqual([], compare(fixture(), fixture()))

    def test_untouched_opcode_change_is_refused(self):
        head = fixture(); head["qmmf.fatbin"] += "changed"
        self.assertTrue(compare(fixture(), head))

    def test_missing_image_is_refused(self):
        head = fixture(); del head["route_rows.fatbin"]
        self.assertTrue(compare(fixture(), head))

    def test_early_read_missing_wait_and_predicated_wait_are_refused(self):
        for bad in ("/*0000*/ LDG.E R0, [R2];\n/*0010*/ ACQBULK;", "/*0000*/ EXIT;", "/*0000*/ @P0 ACQBULK;"):
            head = fixture(); head["flashnext_attn.fatbin"] = "code for sm_121\nFunction : fn\n" + bad
            self.assertTrue(compare(fixture(), head))

if __name__ == "__main__":
    unittest.main()
