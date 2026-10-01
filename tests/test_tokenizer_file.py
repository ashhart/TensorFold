"""A saved ``truncation`` block in tokenizer.json must not cut the CUDA server's prompts."""

from __future__ import annotations

import pytest

tokenizers = pytest.importorskip("tokenizers")


def test_saved_truncation_and_padding_are_turned_off(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from tensorfold.cuda import tokenizer_file

    tok = Tokenizer(models.WordLevel({"a": 0, "b": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.enable_truncation(max_length=4)
    tok.enable_padding(pad_id=2)
    tok.save(str(tmp_path / "tokenizer.json"))
    text = " ".join(["a", "b"] * 10)
    assert len(Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode(text).ids) == 4
    loaded = tokenizer_file.load(tmp_path)
    assert loaded.truncation is None and loaded.padding is None
    assert len(loaded.encode(text).ids) == 20
