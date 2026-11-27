"""cuda_sass_identity.functions must see every instruction: five-digit addresses and the second encoding word."""
import platform
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cuda_sass_identity as sid

WORD2 = " " * 84


def dump(late_op, late_word2):
    """A canned cuobjdump -sass dump: function `same` and function `late`, whose 0x10000 instruction varies."""
    return f"""
Fatbin elf code:
================
arch = sm_121
code version = [1,8]
host = linux
compile_size = 64bit

	code for sm_121
		Function : same
	.headerflags	@"EF_CUDA_SM121 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM121)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                           /* 0x0000df00ff017b82 */
{WORD2}/* 0x000fe20000000800 */
        /*10000*/                  EXIT ;                                            /* 0x000000000000794d */
{WORD2}/* 0x000fea0003800000 */
		..........


		Function : late
	.headerflags	@"EF_CUDA_SM121 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM121)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                           /* 0x0000df00ff017b82 */
{WORD2}/* 0x000fe20000000800 */
        /*fff0*/                   NOP ;                                             /* 0x0000000000007918 */
{WORD2}/* 0x000fc00000000000 */
        /*10000*/                  {late_op} ;                                       /* 0x0000000000007918 */
{WORD2}/* {late_word2} */
		..........
"""


class FunctionsTest(unittest.TestCase):
    def changed(self, base, head):
        return sorted(k[1] for k in sid.compare(sid.functions(base), sid.functions(head))[0])

    def test_five_digit_address(self):
        """Two functions differing only at 0x10000: the late one changes, the other does not."""
        base = dump("FADD R2, R2, R3", "0x000fc00000000000")
        head = dump("FMUL R2, R2, R3", "0x000fc00000000000")
        self.assertEqual(self.changed(base, head), ["late"])
        self.assertEqual(self.changed(base, base), [])

    def test_second_encoding_word(self):
        """Only the continuation line's word differs (a scheduling control change): still a change."""
        base = dump("NOP", "0x000fc00000000000")
        head = dump("NOP", "0x000fe20000000000")
        self.assertEqual(self.changed(base, head), ["late"])

    def test_lines_kept(self):
        """Each instruction keeps its line and its continuation, in order."""
        fs = sid.functions(dump("NOP", "0x000fc00000000000"))
        self.assertEqual(sorted(fs), [("sm_121", "late"), ("sm_121", "same")])
        self.assertEqual(len(fs[("sm_121", "same")]), 4)
        self.assertEqual(len(fs[("sm_121", "late")]), 6)
        self.assertTrue(fs[("sm_121", "same")][2].startswith("/*10000*/ EXIT ;"))
        self.assertEqual(fs[("sm_121", "same")][3], "/* 0x000fea0003800000 */")


class RecheckTest(unittest.TestCase):
    def run_recheck(self, base, head, base2, head2):
        kept, shaky = sid.recheck((sid.functions(base), sid.functions(head)), (sid.functions(base2), sid.functions(head2)))
        return [k[1] for k in kept], [k[1] for k in shaky]

    def test_real_change_kept(self):
        """A change both builds reproduce counts once, and nothing is unstable."""
        base, head = dump("FADD R2, R2, R3", "0x000fc00000000000"), dump("FMUL R2, R2, R3", "0x000fc00000000000")
        self.assertEqual(self.run_recheck(base, head, base, head), (["late"], []))

    def test_flaky_head(self):
        """The head's first build differed and its rebuild matches the base: not a change, but the head is unstable."""
        base, flaky = dump("NOP", "0x000fc00000000000"), dump("NOP", "0x000fe20000000000")
        self.assertEqual(self.run_recheck(base, flaky, base, base), ([], ["late"]))

    def test_flaky_base(self):
        """A base whose rebuild disagrees with itself is unstable too, even where the head matched it once."""
        base, other = dump("NOP", "0x000fc00000000000"), dump("NOP", "0x000fe20000000000")
        self.assertEqual(self.run_recheck(base, base, other, base), ([], ["late"]))



class FixedAddressTest(unittest.TestCase):
    def test_nvcc_runs_without_address_randomization(self):
        """build() runs nvcc as `setarch <machine> -R nvcc ...`, the setarch part right before nvcc."""
        calls = []

        def fake(cmd, **kw):
            calls.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(sid.subprocess, "run", side_effect=fake):
            sid.build(Path(d), [], ("unit", [], False, 80), Path(d) / "out" / "pass-0.fatbin")
        nvcc = next(c for c in calls if sid.NVCC in c)
        at = nvcc.index(sid.NVCC)
        self.assertEqual(nvcc[at - 3:at], ["setarch", platform.machine(), "-R"])

    def test_refused_setarch_stops(self):
        """Where setarch cannot turn randomization off, the check stops instead of comparing unstable builds."""
        saved = sid.FIXED
        sid.FIXED = ["false"]
        try:
            with self.assertRaises(RuntimeError):
                sid.fixed_addresses()
        finally:
            sid.FIXED = saved


if __name__ == "__main__":
    unittest.main()
