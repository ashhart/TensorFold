"""Mutation checks for the actual portable overlap panel."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from typing import Any

import verify

ROOT = Path(__file__).resolve().parent


class PortableTests(unittest.TestCase):
    """Reject changed evidence without depending on a GPU or private file."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the frozen sanitized panel once."""
        cls.data = verify.load(ROOT)

    def check(self, records: list[dict[str, Any]] | None = None,
              provenance: dict[str, Any] | None = None,
              counters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate supplied copies against the actual long fixture."""
        rows, prov, count = self.data
        return verify.verify_data(rows if records is None else records,
                                  prov if provenance is None else provenance,
                                  count if counters is None else counters,
                                  ROOT / "long-manifest.json")

    def test_actual_panel(self) -> None:
        """All 152 actual requests reconstruct the saved summary."""
        self.assertEqual(self.check()["total_requests"], 152)

    def test_missing_request(self) -> None:
        """A missing request cannot pass complete-count gates."""
        with self.assertRaises(ValueError):
            self.check(records=self.data[0][1:])

    def test_changed_output(self) -> None:
        """A valid-looking output hash still fails paired literal parity."""
        records = copy.deepcopy(self.data[0])
        records[-1]["request"]["output_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "parity"):
            self.check(records=records)

    def test_wrong_long_cache(self) -> None:
        """The long host-prefix hit must be the actual 4096-token mark."""
        records = copy.deepcopy(self.data[0])
        row = next(r["request"] for r in records if r["arm"] == "D"
                   and r["case"] == "long" and r["stage"] == "followup")
        row["cached_tokens"] = 0
        with self.assertRaises(ValueError):
            self.check(records=records)

    def test_body_and_generation(self) -> None:
        """Changed body or incomplete generation cannot pass."""
        for field, value in (("request_body_sha256", "0" * 64), ("tokens", 255)):
            records = copy.deepcopy(self.data[0])
            records[0]["request"][field] = value
            with self.assertRaises(ValueError):
                self.check(records=records)

    def test_changed_decode_rate(self) -> None:
        """Client decode rate must reconstruct from literal SSE clocks."""
        records = copy.deepcopy(self.data[0])
        records[0]["request"]["decode_tps"] += 1
        with self.assertRaisesRegex(ValueError, "decode rate"):
            self.check(records=records)

    def test_changed_work(self) -> None:
        """Speculative counters must agree with the original raw fields."""
        records = copy.deepcopy(self.data[0])
        records[0]["request"]["work"]["rounds"] += 1
        with self.assertRaisesRegex(ValueError, "work"):
            self.check(records=records)

    def test_changed_model(self) -> None:
        """A wrong model owner must fail provenance validation."""
        prov = copy.deepcopy(self.data[1])
        prov["models"][0]["repo"] = "TensorFold/Qwen3.8-27B-MLX-4bit"
        with self.assertRaisesRegex(ValueError, "model identity"):
            self.check(provenance=prov)

    def test_changed_host_hook_exposure(self) -> None:
        """The observed hook limitation cannot become a pool-reuse claim."""
        counters = copy.deepcopy(self.data[2])
        counters["arms"]["D"]["seq_pool"]["reuse_hits"] = 1
        with self.assertRaisesRegex(ValueError, "exposure"):
            self.check(counters=counters)


if __name__ == "__main__":
    unittest.main()
