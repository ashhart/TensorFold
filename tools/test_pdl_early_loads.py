#!/usr/bin/env python3
"""Fixtures for pdl_early_loads.py: early reads fail, reads after the wait and weight-first copies pass."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdl_early_loads as p  # noqa: E402


COPY = "LDGSTS.E.BYPASS.128 [R7], desc[UR10][R4.64]"


def dump(arch, fn, *lines):
    body = "\n".join(f"        /*{i * 16:04x}*/  {x} ;" for i, x in enumerate(lines))
    return f"\tcode for sm_{arch}\n\t\tFunction : {fn}\n{body}\n"


class Scan(unittest.TestCase):
    def check(self, text, problems, functions=1):
        got, n = p.scan(text)
        self.assertEqual(len(got), problems, got)
        self.assertEqual(n, functions)

    def test_read_after_wait_passes(self):
        self.check(dump(121, "tf_glm_a", "ACQBULK", "LDG.E.CONSTANT R4, desc[UR14][R2.64]"), 0)

    def test_constant_load_before_wait_fails(self):
        self.check(dump(121, "tf_glm_a", "LDG.E.CONSTANT R4, desc[UR14][R2.64]", "ACQBULK"), 1)

    def test_generic_load_before_wait_fails(self):
        self.check(dump(121, "tf_glm_a", "LD.E R4, desc[UR4][R2.64]", "ACQBULK"), 1)

    def test_texture_load_before_wait_fails(self):
        self.check(dump(121, "tf_glm_a", "TLD.SCR.LZ R0, R2, R4", "ACQBULK"), 1)

    def test_shared_and_constant_bank_loads_pass(self):
        banks = ("LDS R4, [R2]", "LDC R1, c[0x0][0x37c]", "ULDC UR4, c[0x0][0x208]")
        self.check(dump(121, "tf_glm_a", *banks, "ACQBULK"), 0)

    def test_weight_first_copies_pass(self):
        self.check(dump(121, "_ZN7tf_qmmf11qmmf_kernelILi2E", COPY, "ACQBULK"), 0)
        self.check(dump(121, "glm_mm16_rows", COPY, "ACQBULK"), 0)

    def test_copies_elsewhere_fail(self):
        self.check(dump(121, "glm_mm_reduce", COPY, "ACQBULK"), 1)

    def test_plain_only_kernels_skipped(self):
        self.check(dump(121, "tf_qmmf_shared64", COPY, "LDG.E R4, desc[UR4][R2.64]"), 0, 0)

    def test_a_dependent_kernel_without_its_wait_fails(self):
        self.check(dump(121, "glm_mm16_16", COPY, "LDG.E R4, desc[UR4][R2.64]"), 1)

    def test_pre_pdl_targets_skipped(self):
        self.check(dump(80, "tf_glm_a", "LDG.E R4, desc[UR4][R2.64]"), 0, 0)


class Main(unittest.TestCase):
    def test_no_images_is_usage(self):
        self.assertEqual(p.main(["pdl_early_loads.py", "cuobjdump"]), 2)

    def test_failed_tool_fails(self):
        self.assertEqual(p.main(["pdl_early_loads.py", "/bin/false", "x.fatbin"]), 1)

    def test_empty_dump_fails(self):
        self.assertEqual(p.main(["pdl_early_loads.py", "/bin/true", "x.fatbin"]), 1)


if __name__ == "__main__":
    unittest.main()
