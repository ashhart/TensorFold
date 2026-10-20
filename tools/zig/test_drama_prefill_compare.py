"""Exercise the oracle comparison gate with complete, misaligned and regressed prompt metrics."""
import unittest
from drama_prefill_compare import EXPECTED, collect, compare


def receipt(chunk, cosine=0.999, error=1.0):
    lines = []
    for name in sorted(EXPECTED):
        count = 129 - int(name.startswith("tower_prefill."))
        for at in range(0, count, chunk):
            rows = min(chunk, count - at)
            lines.append(f"PREFILL {name} start={at} rows={rows} worst_cos={cosine} max_abs={error}")
    return "\n".join(lines) + "\nRESULT drama3 check PASS:\n"


class PrefillTests(unittest.TestCase):
    def test_chunk_boundaries_do_not_change_coverage(self):
        for chunk in (2, 64, 128):
            self.assertTrue(compare(receipt(1), receipt(chunk))["pass"])

    def test_better_reference_error_passes(self):
        self.assertTrue(compare(receipt(1), receipt(128, 1.0, 0.0))["pass"])

    def test_either_error_regression_fails(self):
        self.assertFalse(compare(receipt(1), receipt(128, 0.99))["pass"])
        self.assertFalse(compare(receipt(1), receipt(128, error=2.0))["pass"])

    def test_missing_or_misaligned_rows_fail(self):
        for text in ("", receipt(128).replace("start=128", "start=127"),
                     receipt(128).replace("rows=1 ", "rows=2 ")):
            with self.assertRaises(ValueError):
                collect(text)

    def test_nonfinite_and_failed_reference_checks_fail(self):
        for text in (receipt(128, float("nan")), receipt(128, error=float("inf")),
                     receipt(128).replace("check PASS", "check FAIL")):
            with self.assertRaises(ValueError):
                collect(text)


if __name__ == "__main__":
    unittest.main()
