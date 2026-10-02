"""TENSORFOLD_PREFILL_ROWS: Flash Next prompt-piece rows, or the engine's own plan when unset."""

import pytest

from tensorfold.cuda.geometry import indexed_prefill_rows


def test_unset_or_empty_leaves_the_engine_its_plan(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    assert indexed_prefill_rows() is None
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", " ")
    assert indexed_prefill_rows() is None


@pytest.mark.parametrize("value, rows", [("256", 256), ("2048", 2048), (" 4096 ", 4096), ("16384", 16384)])
def test_a_set_value_is_the_piece_rows(value, rows, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", value)
    assert indexed_prefill_rows() == rows


@pytest.mark.parametrize("value", ["255", "16385", "0", "-2048", "2k", "4096.0"])
def test_anything_else_refuses_at_startup(value, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", value)
    with pytest.raises(ValueError, match="TENSORFOLD_PREFILL_ROWS"):
        indexed_prefill_rows()
