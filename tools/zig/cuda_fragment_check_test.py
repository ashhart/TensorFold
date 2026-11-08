"""A fragment's includer missing a name it needs fails by name and line; every includer in the real tree passes."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also when run from the repository root by path
import cuda_fragment_check as checker

REAL = Path(__file__).resolve().parents[2] / checker.KERNELS
FRAG = """// Body; the caller supplies A, B, WIDE, n, out and pick.
    const int total = A + n * B;
    if constexpr (WIDE) out[0] = pick(total) + LIMIT + helper(total) + TAG;
    else out[0] = total;
    for (int i = 0; i < n; ++i) out[i] += i;
"""
HOST = "namespace fx {\nconstexpr int LIMIT = 4;\n__device__ int helper(int x) { return x; }\n}  // namespace fx\n"
GOOD = """#include "host.cuh"
#define TAG 1
namespace fx {
template <int A, bool WIDE>
__global__ void k(int* out, int n) {
    constexpr int B = 2;
    auto pick = [](int x) { return x; };
#include "frag.cuh"
}
#if WIDE_K2
__global__ void k2(int* out, int n, int A) {
#else
__global__ void k2(int* out, int n, int A, int extra) {
#endif
    constexpr bool WIDE = false, B = 3;
    auto pick = [&](int x) { return x + A; };
#include "frag.cuh"
}
}  // namespace fx
"""
# contract_controls.cu's control before #2463: the rows fragment had since gained P256, IDS and PN.
BEFORE_2463 = """#define PATTN_VLANE 1
#include "prefill_attention.cuh"

namespace tf_prefill_attention {
extern "C" __global__ void __launch_bounds__(256, 1) tf_ctl_pattn_old(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                                                     const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0,
                                                                     int W, int H, int HK, int G, float scale, const int* __restrict__ PT, long long PS) {
    constexpr int D = 64, WARPS = 8, HPC = 4, NS = 8;
    constexpr bool causal = true;
    constexpr int skip = 0;
    constexpr bool HPAGED = false, F8 = false, HM = false;
    constexpr int CAP = 0, PTOK = 1;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}
}  // namespace tf_prefill_attention
"""
OLD = "    constexpr bool HPAGED = false, F8 = false, HM = false;\n    constexpr int CAP = 0, PTOK = 1;\n"
FIX_2463 = "    constexpr bool P256 = false, HPAGED = false, F8 = false, HM = false, IDS = false;\n    constexpr int PN = 0, CAP = 0, PTOK = 1;\n"
# qmmf.cu's shape: one includer sets a switch around the include, the other does not; each branch needs its own name.
SWITCHED = """#ifdef TF_WIDE
    out[0] = ldy;
#else
    out[0] = 0;
#endif

#if __CUDA_ARCH__ >= 800 && !defined(TF_WIDE)
    out[1] = narrow;
#endif
"""
SWITCHERS = """__global__ void wide(int* out, int ldy) {
#define TF_WIDE

#include "switched.cuh"

#undef TF_WIDE
}
__global__ void plain(int* out, int narrow) {
#include "switched.cuh"
}
"""
ROWS_NEEDS = set("""P256 HPAGED F8 HM IDS PN CAP PTOK causal skip ks vs HO first_row q kc vc out p0 W H HK G scale PT PS
D WARPS HPC NS""".split())


def line_of(text, needle):
    return text[:text.index(needle)].count("\n") + 1


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.kernels = self.root / checker.KERNELS

    def write(self, name, text):
        (self.kernels / name).write_text(text)

    def edit(self, name, old, new):
        text = (self.kernels / name).read_text()
        self.assertEqual(text.count(old), 1, old)
        self.write(name, text.replace(old, new))

    def errors(self):
        return checker.check(self.root)[1]


class SyntheticTest(Fixture):
    def setUp(self):
        super().setUp()
        self.kernels.mkdir(parents=True)
        self.write("frag.cuh", FRAG)
        self.write("host.cuh", HOST)
        self.write("good.cu", GOOD)

    def test_green_template_parameters_parameters_constexpr_lambda_and_conditional_heads(self):
        found, errors = checker.check(self.root)
        self.assertEqual(errors, [])
        self.assertEqual([(f, need, len(s)) for f, need, s in found],
                         [("zig/kernels/cuda/frag.cuh", ["A", "B", "WIDE", "n", "out", "pick"], 2)])

    def test_red_one_missing_name(self):
        self.edit("good.cu", "    constexpr int B = 2;\n", "")
        line = line_of((self.kernels / "good.cu").read_text(), '#include "frag.cuh"')
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/good.cu:{line}: frag.cuh needs B"])

    def test_closed_block_dead_code_and_late_declaration_do_not_count(self):
        decl, use = "    constexpr int B = 2;\n", '#include "frag.cuh"\n'
        for new in (decl.replace("constexpr int B = 2;", "{ constexpr int B = 2; }"), "#if 0\n" + decl + "#endif\n"):
            self.write("good.cu", GOOD.replace(decl, new))
            self.assertEqual([e.split(": ", 1)[1] for e in self.errors()], ["frag.cuh needs B"], new)
        self.write("good.cu", GOOD.replace(decl, "").replace(use, use + decl, 1))
        self.assertEqual([e.split(": ", 1)[1] for e in self.errors()], ["frag.cuh needs B"])

    def test_macro_before_the_include_supplies_a_name(self):
        self.edit("good.cu", "    constexpr int B = 2;\n", "#define B 2\n")
        self.assertEqual(self.errors(), [])

    def test_attributed_declaration_is_the_fragments_own(self):
        self.write("frag.cuh", FRAG + "    [[maybe_unused]] int spare = total;\n    out[1] = spare;\n")
        self.write("other.cu", "__device__ void f() { int spare = 0; }\n")
        self.assertEqual(self.errors(), [])

    def test_fragment_branch_on_a_switch_its_includer_sets(self):
        self.write("switched.cuh", SWITCHED)
        self.write("switchers.cu", SWITCHERS)
        self.assertEqual(self.errors(), [])
        swapped = SWITCHERS.replace("#define TF_WIDE\n", "").replace("#undef TF_WIDE\n", "")
        swapped = swapped.replace('narrow) {\n#include "switched.cuh"\n',
                                  'narrow) {\n#define TF_WIDE\n#include "switched.cuh"\n#undef TF_WIDE\n')
        self.write("switchers.cu", swapped)
        wide, plain = [n for n, s in enumerate(swapped.splitlines(), 1) if "switched.cuh" in s]
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/switchers.cu:{wide}: switched.cuh needs narrow",
                                         f"zig/kernels/cuda/switchers.cu:{plain}: switched.cuh needs ldy"])
        # A switch defined under an #if the check can't tell is unknown at the include: every branch is read.
        self.write("switchers.cu", SWITCHERS.replace("#define TF_WIDE\n", "#if TF_WANT_WIDE\n#define TF_WIDE\n#endif\n"))
        wide = line_of(SWITCHERS, '#include "switched.cuh"') + 2
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/switchers.cu:{wide}: switched.cuh needs narrow"])
        # A macro no includer both defines and undefines may come from the command line: every branch is read.
        self.write("switchers.cu", SWITCHERS)
        self.write("switched.cuh", SWITCHED.replace("#ifdef TF_WIDE", "#ifdef TF_OTHER"))
        plain = [n for n, s in enumerate(SWITCHERS.splitlines(), 1) if "switched.cuh" in s][1]
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/switchers.cu:{plain}: switched.cuh needs ldy"])

    def test_lone_includer_missing_a_name_others_declare(self):
        self.write("good.cu", GOOD.split("#if WIDE_K2")[0].replace("auto pick", "auto choose") + "}\n")
        self.write("other.cu", "__device__ void f(int pick) {}\n")
        self.assertEqual([e.split(": ", 1)[1] for e in self.errors()], ["frag.cuh needs pick"])

    def test_nested_fragment_needs_reach_the_outer_includer(self):
        self.write("outer.cuh", "    constexpr int B = 5;\n#include \"frag.cuh\"\n")
        own = GOOD.replace("    constexpr int B = 2;\n", "").replace(", B = 3", "")
        self.write("good.cu", own.replace('#include "frag.cuh"', '#include "outer.cuh"'))
        self.assertEqual(self.errors(), [])
        self.edit("good.cu", "    auto pick = [](int x) { return x; };\n", "")
        self.assertEqual([e.split(": ", 1)[1] for e in self.errors()], ["outer.cuh needs pick"])

    def test_missing_fragment_and_unbalanced_includer_are_invalid(self):
        self.write("bad.cu", "__global__ void k() {\n#include \"nowhere.cuh\"\n}\n")
        with self.assertRaisesRegex(ValueError, "missing nowhere.cuh"):
            self.errors()
        self.write("bad.cu", "__global__ void k() {\n")
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            self.errors()


@unittest.skipUnless(REAL.is_dir(), "no CUDA kernel tree")
class RealTreeTest(Fixture):
    """The real kernels in a scratch copy with #2463's two-line change applied by hand where it is not yet merged."""

    def setUp(self):
        super().setUp()
        shutil.copytree(REAL, self.kernels)
        text = (self.kernels / "contract_controls.cu").read_text()
        self.write("contract_controls.cu", text.replace(OLD, FIX_2463) if text.count(OLD) == 1 else text)

    def test_green(self):
        found, errors = checker.check(self.root)
        self.assertEqual(errors, [])
        rows = {f: (need, sites) for f, need, sites in found}["zig/kernels/cuda/prefill_attention_rows.cuh"]
        self.assertLessEqual(ROWS_NEEDS, set(rows[0]))
        self.assertLessEqual({"zig/kernels/cuda/contract_controls.cu", "zig/kernels/cuda/prefill_attention.cu"},
                             {s.path for s in rows[1]})

    def test_real_includer_missing_one_name(self):
        self.edit("prefill_attention.cu", "F8 = false, HM = true, IDS = true;", "F8 = false, HM = true;")
        text = (self.kernels / "prefill_attention.cu").read_text()
        at = line_of(text, "hpattn_p256_ids_kernel(")
        line = at + text.splitlines()[at:].index('#include "prefill_attention_rows.cuh"') + 1
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/prefill_attention.cu:{line}: prefill_attention_rows.cuh needs IDS"])

    def test_qmmf_switches_pick_each_includers_branch(self):
        text = (self.kernels / "qmmf.cu").read_text()
        k32 = line_of(text, '#define TF_QMMF_K32') + 1
        strided = line_of(text, '#define TF_QMMF_STRIDED') + 1
        self.edit("qmmf.cu", ", STAGES = 3, BK = 32;", ", STAGES = 3;")
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/qmmf.cu:{k32}: qmmf_body.cuh needs BK"])
        self.write("qmmf.cu", text.replace("int group, int ldy) {\n#define", "int group) {\n#define"))
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/qmmf.cu:{strided}: qmmf_body.cuh needs ldy"])

    def test_contract_controls_before_and_after_2463(self):
        self.write("before_2463.cu", BEFORE_2463)
        line = line_of(BEFORE_2463, '#include "prefill_attention_rows.cuh"')
        self.assertEqual(self.errors(), [f"zig/kernels/cuda/before_2463.cu:{line}: prefill_attention_rows.cuh needs {n}"
                                         for n in ("IDS", "P256", "PN")])
        self.write("before_2463.cu", BEFORE_2463.replace(OLD, FIX_2463))
        self.assertEqual(self.errors(), [])


if __name__ == "__main__":
    unittest.main()
