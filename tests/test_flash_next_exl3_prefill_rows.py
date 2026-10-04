"""TENSORFOLD_PREFILL_ROWS on a Flash Next EXL3 pack: unset keeps the 2048-row pieces, their admission and staging."""

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.capacity import Geometry  # noqa: E402
from tensorfold.cuda.geometry import PREFILL_ROWS, exl3_indexed_scratch  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_pack  # noqa: E402

TEXT = {"hidden_size": 256, "hc_count": 4, "num_experts_per_tok": 8, "moe_intermediate_size": 64,
        "num_attention_heads": 4, "head_dim": 64, "linear_num_key_heads": 2, "linear_key_head_dim": 32,
        "linear_num_value_heads": 4, "linear_value_head_dim": 32}
BASE = Geometry(lambda slots: 1000 * slots, reserve=0)


def added(monkeypatch, value: str | None) -> int:
    """Bytes the EXL3 admission adds to the engine's geometry with the variable at ``value`` (None: unset)."""

    if value is None:
        monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", value)
    return exl3_pack.admission(lambda text: BASE)(TEXT).bytes_at(64) - BASE.bytes_at(64)


@pytest.mark.parametrize("value", [None, "", " "])
def test_unset_keeps_the_2048_row_pieces_their_admission_and_staging(value, monkeypatch):
    assert added(monkeypatch, value) == exl3_indexed_scratch(TEXT, exl3_pack.MOE_WINDOW, 2048)
    assert exl3_pack.prompt_rows() == PREFILL_ROWS == 2048


@pytest.mark.parametrize("rows", [4096, 8192])
def test_a_set_value_sizes_the_pieces_and_their_scratch(rows, monkeypatch):
    assert added(monkeypatch, str(rows)) == exl3_indexed_scratch(TEXT, exl3_pack.MOE_WINDOW, rows)
    assert exl3_pack.prompt_rows() == rows
