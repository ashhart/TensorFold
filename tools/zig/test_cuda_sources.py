"""Controls for the shader-free CUDA source/include check."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import check_cuda_sources as check


class SourceNames(unittest.TestCase):
    def setUp(self):
        self.compiler = shutil.which("clang++")
        if self.compiler is None:
            self.skipTest("clang++ is required")
        scratch = check.ROOT / "work"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def parse(self, source):
        return subprocess.run(check.command(self.compiler, source, 1210), capture_output=True, text=True)

    def test_include_chain_supplies_unrelated_project_name(self):
        (self.root / "inner.cuh").write_text("__device__ int project_fixture_helper(int x) { return x; }\n")
        (self.root / "outer.cuh").write_text('#include "inner.cuh"\n')
        source = self.root / "fixture.cu"
        source.write_text('#include "outer.cuh"\n__global__ void k(int* x) { *x = project_fixture_helper(*x); }\n')
        self.assertEqual(self.parse(source).returncode, 0)
        (self.root / "outer.cuh").write_text("")
        result = self.parse(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("undeclared identifier 'project_fixture_helper'", result.stderr)

    def test_launch_bound_names_are_resolved(self):
        source = self.root / "bounds.cu"
        source.write_text("__global__ void __launch_bounds__(project_bound) k() {}\n")
        result = self.parse(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("undeclared identifier 'project_bound'", result.stderr)
        source.write_text("constexpr int project_bound = 32;\n" + source.read_text())
        self.assertEqual(self.parse(source).returncode, 0)

    def test_real_exl3_layout(self):
        result = self.parse(check.HEADERS / "exl3_layout.cpp")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_layout_rejects_wrong_coordinate(self):
        header = (check.ROOT / "zig/kernels/cuda/exl3_decode.cuh").read_text()
        bad = header.replace("return (lane >> 2) + 8 * (j >> 2);", "return (lane >> 2);", 1)
        self.assertNotEqual(header, bad)
        (self.root / "bad.cuh").write_text(bad)
        test = (check.HEADERS / "exl3_layout.cpp").read_text().splitlines()
        source = self.root / "layout.cu"
        source.write_text('#include "bad.cuh"\n' + "\n".join(test[1:]) + "\n")
        result = self.parse(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("static assertion failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
