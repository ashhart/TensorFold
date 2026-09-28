"""GLM's per-request drafter choice: a drafter left after a bad stretch is probed again soon and judged afresh."""

import importlib

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch

COSTS = {"verify": [28.4, 38.8, 43.8, 48.8, 53.8, 58.8, 63.8, 68.8], "mtp": 1.79, "mtp_step": 1.56, "mtp_row": 0.17,
         "block": 3.16, "taps_row": 0.049}


def run(choice, keeps):
    """Drive the choice with each drafter's keep a round (f: 5 drafts, 6 rows; m: 3 drafts, 4 rows)."""
    arms = []
    for i in range(18):
        arm = choice.pick()
        arms.append(arm)
        keep = keeps(arm, i)
        choice.record(arm, 6 if arm == "f" else 4, 0 if arm == "f" else 3, 4, keep)
    return "".join(arms)


def test_a_drafter_left_after_a_bad_stretch_comes_back_when_it_is_better(allocations):  # noqa: F811
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.drafter_choice")
    # the reply's rounds 7-8 are hard for either drafter; afterwards DFlash2 keeps all 6 rows, MTP its 4
    keeps = lambda arm, i: 1 if i in (7, 8) else (6 if arm == "f" else 4)       # noqa: E731
    arms = run(mod.DrafterChoice(COSTS, first="f"), keeps)
    assert arms[:4] == "ffmm" and arms[9] == "m"                                # the hard stretch forced a switch
    assert arms[12:16] == "ffff"                                                 # DFlash2 back after 3 MTP rounds


def test_a_better_drafter_is_kept_after_its_probe(allocations):  # noqa: F811
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.drafter_choice")
    keeps = lambda arm, i: 4 if arm == "m" else 2                                # noqa: E731  MTP better throughout
    arms = run(mod.DrafterChoice(COSTS, first="f"), keeps)
    assert arms[4:].count("f") <= 2
