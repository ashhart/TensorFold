"""cuda_sass_identity.functions must see every instruction: five-digit addresses and the second encoding word."""
import unittest

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


if __name__ == "__main__":
    unittest.main()
