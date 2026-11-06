"""A new instruction, a changed copy and a duplicate definition must each make the CUDA primitive guard fail."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import cuda_common_check as checker

ASM = '__device__ void load(unsigned* x) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0}, [%1];" : "=r"(*x) : "r"(0)); }\n'
DEFINITION = '__device__ float warp_sum(float x) { return x; }\n'


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.kernels = self.root / "zig/kernels/cuda"
        self.kernels.mkdir(parents=True)
        self.unit = self.kernels / "test.cu"
        self.unit.write_text(DEFINITION)
        self.allow = self.root / "allow.json"
        self.admit()

    def admit(self):
        rows = [{"path": p, "kind": k, "fingerprint": f, "count": n, "reason": "Retained order variant."}
                for (p, k, f), n in Counter(x.key() for x in checker.scan(self.root)).items()]
        self.allow.write_text(json.dumps({"schema": 1, "entries": rows}))

    def errors(self):
        return checker.check(self.root, self.allow)[1]

    def test_baseline_and_planted_ldmatrix(self):
        self.assertEqual(self.errors(), [])
        self.unit.write_text(DEFINITION + ASM)
        self.assertIn("ldmatrix", " ".join(self.errors()))

    def test_duplicate_definition(self):
        self.unit.write_text(DEFINITION * 2)
        self.assertIn("count 2, allowed 1", " ".join(self.errors()))

    def test_changed_body(self):
        self.unit.write_text(DEFINITION.replace("return x;", "return x + 1;"))
        self.assertEqual(len(self.errors()), 2)

    def test_comment_and_whitespace_are_not_new_copies(self):
        self.unit.write_text('// ' + ASM + DEFINITION.replace("return", "/* kept */ return"))
        self.assertEqual(self.errors(), [])

    def test_multiline_asm_and_constraint_change(self):
        self.unit.write_text(ASM.replace("asm volatile(", "asm\nvolatile(\n"))
        self.admit()
        self.assertEqual(self.errors(), [])
        self.unit.write_text(self.unit.read_text().replace('"=r"', '"+r"'))
        self.assertEqual(len(self.errors()), 2)

    def test_missing_reason(self):
        data = json.loads(self.allow.read_text())
        data["entries"][0]["reason"] = ""
        self.allow.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "reason"):
            self.errors()

    def test_removed_copy_is_stale(self):
        self.unit.write_text("")
        self.assertIn("count 0, allowed 1", " ".join(self.errors()))

    def test_common_is_the_owner(self):
        common = self.kernels / "common"
        common.mkdir()
        (common / "primitives.cuh").write_text(ASM)
        self.assertEqual(self.errors(), [])

    def test_every_requested_instruction_and_definition(self):
        for op in ("mma.sync", "cp.async", "ld.global.nc", "griddepcontrol"):
            self.unit.write_text(DEFINITION + ASM.replace("ldmatrix.sync.aligned.m8n8.x4.shared.b16", op))
            self.assertIn("asm:" + op, " ".join(self.errors()))
        for name in ("block_sum_serial", "fp4pair", "fp8pair", "e4m3f", "unpack8", "pack8"):
            self.unit.write_text(DEFINITION + DEFINITION.replace("warp_sum", name))
            self.assertIn("definition:" + name, " ".join(self.errors()))


if __name__ == "__main__":
    unittest.main()
