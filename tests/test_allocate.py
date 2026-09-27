"""A shared round's rows across streams (engine.allocate), the node calibration it reads, and the fitting tool's
labels."""

import importlib.util
import json
from pathlib import Path

import pytest

from tensorfold.drafters import calibration
from tensorfold.engine.allocate import allocate, chain_probabilities

M5 = {w: (36.0 if w <= 16 else 47.0) for w in range(1, 33)}      # the 32-row tensor op's step at 17 rows


def test_rows_go_to_the_likeliest_nodes_across_streams():
    probs = [[0.9, 0.8, 0.4], [0.6, 0.5, 0.1]]
    assert allocate([1, 1], probs, {}, 0.0, 6) == [2, 2]


def test_the_width_with_the_most_tokens_a_millisecond_wins():
    strong = [[0.9 ** (j + 1) for j in range(15)] for _ in range(2)]
    weak = [[0.4 ** (j + 1) for j in range(15)] for _ in range(4)]
    assert sum(allocate([1] * 2, strong, M5, 10.0, 32)) + 2 > 16        # deep chains pay for the step
    assert sum(allocate([1] * 4, weak, M5, 10.0, 32)) + 4 == 16         # shallow ones stop below it


def test_fixed_rows_are_never_cut_and_streams_can_draft_nothing():
    assert allocate([4, 1, 1], [[], [0.9], [0.1]], {}, 0.0, 7) == [0, 1, 0]
    assert allocate([7], [[0.9]], {}, 0.0, 6) == [0]
    assert chain_probabilities([0.5, 0.8], 3) == pytest.approx([0.5, 0.4, 0.32])


def test_a_fitted_table_rises_with_the_score_and_survives_a_file(tmp_path):
    samples = [(0, -0.1, True)] * 9 + [(0, -0.1, False)] + [(0, -3.0, True)] * 2 + [(0, -3.0, False)] * 8
    samples += [(0, -2.0, True)] * 1 + [(0, -2.0, False)] * 1 + [(0, -1.5, True)] * 1 + [(0, -1.5, False)] * 3
    table = calibration.fit(samples)
    row = table.table[0]
    assert all(a <= b for a, b in zip(row, row[1:]))
    assert table.probability(0, -0.1) > 0.8 > 0.3 > table.probability(0, -3.0)
    calibration.save(tmp_path / "c.json", {"sampled": table}, {"prompts": ["x"]})
    again = calibration.load(tmp_path / "c.json")["sampled"]
    assert again.probabilities([-1, 0], [-0.1, -3.0]) == [round(table.probability(0, -0.1), 4),
                                                           round(table.probability(1, -3.0), 4)]


def test_the_tool_labels_a_node_by_its_whole_path(tmp_path):
    spec = importlib.util.spec_from_file_location("fit_tool", Path(__file__).parents[1] / "tools/fit_draft_calibration.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    log = tmp_path / "log.jsonl"
    rounds = [  # the stream commits 5 6 7 8 at positions 10 .. 13
        {"stream": 1, "position": 10, "greedy": False, "kept": [5], "tokens": [6, 9, 7, 4], "parents": [-1, -1, 0, 1],
         "scores": [-0.1, -2.0, -0.3, -2.5]},
        {"stream": 1, "position": 12, "greedy": False, "kept": [6, 7], "tokens": [8, 3], "parents": [-1, 0],
         "scores": [-0.2, -1.0]},
        {"stream": 1, "position": 13, "greedy": False, "kept": [8], "tokens": [1], "parents": [-1], "scores": [-0.5]},
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rounds))
    got = [(d, s, hit) for d, s, hit, _ in tool.samples([str(log)])["sampled"]]
    assert got == [(0, -0.1, True), (0, -2.0, False), (1, -0.3, True), (1, -2.5, False), (0, -0.2, True)]


def test_costs_run_linear_between_measured_totals_and_per_row_past_them():
    from tensorfold.engine.family_depth import extend_costs

    costs = extend_costs({1: 20.0, 16: 29.0, 32: 45.0, 64: 84.0}, 128)
    assert costs[16] == 29.0 and costs[24] == pytest.approx(37.0) and costs[48] == pytest.approx(64.5)
    assert costs[128] == pytest.approx(168.0) and costs[8] == pytest.approx(20.0 + 9.0 * 7 / 15)
