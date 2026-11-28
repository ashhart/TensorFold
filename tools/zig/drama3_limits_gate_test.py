"""drama3_limits_gate.sh: stale records pass; a failed launch, missing image or unaudited router fails."""
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "tools/zig/drama3_limits_gate.sh"
# a GB10 audit of the real registry (nvcc 13.0): 877 stale records, no failed launch or missed control
REAL = (ROOT / "docs/receipts/glm-fp4local-80517fc3/head-797dd2e1/single/limits.log").read_text().splitlines()
SMS = (80, 86, 89, 120, 121)
STALE = "PASS limits sm_{sm} {name}: 30 registers, 0 static shared bytes, bound 128 (record stale)"
ROUTER = "drama3 d3_router_compact"


def pending_tile_launches() -> list[str]:
    """The K32 tile launches the registry holds unmeasured, stale under every nvcc."""
    source = (ROOT / "zig/src/core/contract/nvfp4_tile_pending.zig").read_text()
    return re.findall(r'\.name = "([^"]+)"', source)


def audit(*names: str, extra: tuple[str, ...] = ()) -> list[str]:
    """The real log plus each name's stale record on every SM, its problem count raised to match."""
    log = [line for line in REAL if not line.startswith("FAIL kernel-contract-limits")]
    log += [STALE.format(sm=sm, name=name) for name in names for sm in SMS] + list(extra)
    stale = sum(line.startswith("PASS limits") and line.endswith("(record stale)") for line in log)
    return log + [f"FAIL kernel-contract-limits: {stale} problems"]


def gate(status: int, lines: list[str]) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".log") as log:
        log.write("\n".join(lines) + "\n")
        log.flush()
        return subprocess.run(["sh", str(GATE), str(status), log.name]).returncode


class LimitsGateTests(unittest.TestCase):
    def test_clean_audit_passes(self):
        lines = [STALE.format(sm=sm, name=ROUTER).replace("stale", "current") for sm in SMS]
        self.assertEqual(gate(0, lines + ["PASS kernel-contract-limits: every registered launch fits"]), 0)

    def test_real_registry_with_pending_records_passes(self):
        tiles = pending_tile_launches()
        self.assertEqual(len(tiles), 2)
        self.assertEqual(gate(1, audit(ROUTER, *tiles)), 0)

    def test_router_not_audited_fails(self):
        self.assertNotEqual(gate(1, audit(*pending_tile_launches())), 0)

    def test_failed_launch_fails(self):
        failed = "FAIL limits sm_121 nvfp4_experts_tile tf_nvfp4_tile_swiglu: 300 registers, 0 static shared bytes, bound 128 (record stale): too many registers"
        self.assertNotEqual(gate(1, audit(ROUTER, extra=(failed,))), 0)

    def test_missing_image_fails(self):
        missing = "FAIL limits drama3 d3_rmsnorm: zig-out/fatbin/drama3.fatbin holds no sm_89 image (build with -Dsm=80,86,89,120,121)"
        self.assertNotEqual(gate(1, audit(ROUTER, extra=(missing,))), 0)

    def test_missed_control_fails(self):
        missed = "MISSED limits sm_121 control: past its bounds: 10 registers, 0 static shared bytes, bound 256 (record current)"
        self.assertNotEqual(gate(1, audit(ROUTER, extra=(missed,))), 0)

    def test_problem_count_beyond_stale_records_fails(self):
        lines = audit(ROUTER)
        lines[-1] = lines[-1].replace(" problems", "1 problems")
        self.assertNotEqual(gate(1, lines), 0)

    def test_error_without_summary_fails(self):
        self.assertNotEqual(gate(1, audit(ROUTER)[:-1]), 0)


if __name__ == "__main__":
    unittest.main()
