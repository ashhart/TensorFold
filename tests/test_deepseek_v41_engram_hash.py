"""DeepSeek-V4.1-Flash's Engram hashing: primes, offsets, multipliers, n-gram ids against the reference and the map."""

from __future__ import annotations

import dataclasses
import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tensorfold.families.deepseek_v41 import engram_hash
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.engram_hash import Hasher, TokenMap, multipliers, primes, rank_columns

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v41"
CFG = Config.read(FIXTURES)


def _fixture_map() -> tuple[dict, TokenMap]:
    """The fixture's token-map entries as a TokenMap; ids it does not list map to -1."""

    data = json.loads((FIXTURES / "engram_ids.json").read_text())
    entries = {int(k): v for k, v in data["token_map"].items()}
    table = np.full(max(entries) + 1, -1, dtype=np.int32)
    table[list(entries)] = list(entries.values())
    return data, TokenMap(table, data["compressed_vocab_size"])


def _small(heads: int = 3, ngram: int = 4, layers: tuple[int, ...] = (1, 5)) -> tuple[Config, TokenMap]:
    """A reduced config with small primes over a 50-class map of a 200-token vocabulary."""

    found = primes(layers, 97, heads, ngram)
    cfg = dataclasses.replace(CFG, engram_layer_ids=layers, engram_vocab_size=97, engram_n_heads=heads,
                              engram_max_ngram_size=ngram, engram_compressed_vocab_size=50,
                              engram_num_embeddings=tuple(int(s) for s in found.sum(axis=(1, 2))))
    table = np.random.default_rng(0).integers(0, 50, size=200).astype(np.int32)
    table[:50] = np.arange(50)
    return cfg, TokenMap(table, 50)


def _direct(cfg: Config, token_map: TokenMap, raw: list[int]) -> np.ndarray:
    """EN:160-184 one position at a time in Python integers (no image spans)."""

    cache = [int(token_map.table[t]) for t in raw]
    pad = int(token_map.table[cfg.engram_pad_token_id])
    found = primes(cfg.engram_layer_ids, cfg.engram_vocab_size, cfg.engram_n_heads, cfg.engram_max_ngram_size)
    mult = multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size, token_map.size)
    out = []
    for p in range(len(raw)):
        tokens, blocked = [], False
        for shift in range(cfg.engram_max_ngram_size):
            source = cache[max(p - shift, 0)]
            blocked = blocked or p < shift
            tokens.append(pad if blocked else source)
        row = []
        for layer in range(len(cfg.engram_layer_ids)):
            flat = [int(x) for x in found[layer].reshape(-1)]
            products = [t * int(m) for t, m in zip(tokens, mult[layer])]
            rolling, hashes = products[0], []
            for i in range(1, cfg.engram_max_ngram_size):
                rolling ^= products[i]
                hashes += [rolling % int(q) for q in found[layer, i - 1]]
            row.append([h + sum(flat[:c]) for c, h in enumerate(hashes)])
        out.append(row)
    return np.array(out, dtype=np.int64)


def _chunked(hasher: Hasher, raw: list[int], cuts: list[int]) -> np.ndarray:
    bounds = [0, *cuts, len(raw)]
    back = hasher.max_ngram - 1
    return np.concatenate([hasher.ids(raw[max(a - back, 0):a], raw[a:b]) for a, b in pairwise(bounds)])


def test_primes_sum_to_the_table_rows_and_split_at_column_12():
    found = primes(CFG.engram_layer_ids, CFG.engram_vocab_size, CFG.engram_n_heads, CFG.engram_max_ngram_size)
    assert found.shape == (2, 3, 8)
    assert tuple(found.sum(axis=(1, 2))) == CFG.engram_num_embeddings == (384006168, 384016682)
    assert found[0, 0, 0] == 16000057 and found[1, 2, 7] == 16000889
    assert len(set(found.reshape(-1).tolist())) == 48
    hasher = Hasher(CFG, _fixture_map()[1])
    assert hasher.offsets[:, 12].tolist() == [192001740, 192007016]
    assert hasher.offsets[:, 0].tolist() == [0, 0]
    assert hasher.offsets[:, 23].tolist() == [368005705, 368015793]


def test_multipliers_are_the_hard_coded_table():
    got = multipliers(CFG.engram_layer_ids, CFG.engram_max_ngram_size, CFG.engram_compressed_vocab_size)
    assert got.tolist() == [list(engram_hash.MULTIPLIERS[(99092, 4)][layer]) for layer in (1, 14)]
    assert (got % 2 == 1).all() and (got < np.iinfo(np.int64).max // 99092).all()


def test_numpy_rederives_the_real_multipliers():
    bound = np.iinfo(np.int64).max // 99092 // 2
    derived = [np.random.default_rng(10007 * layer).integers(0, bound, size=(4,), dtype=np.int64) * 2 + 1
               for layer in (1, 14)]
    table = engram_hash.MULTIPLIERS[(99092, 4)]
    if [list(d) for d in derived] != [list(table[layer]) for layer in (1, 14)]:
        pytest.skip(f"numpy {np.__version__}'s default_rng stream differs from the one the checkpoint used")


def test_vectorized_ids_equal_the_reference_one_position_at_a_time():
    rng = np.random.default_rng(7)
    for cfg, token_map in (_small(), _small(heads=2, ngram=3, layers=(4,)), (CFG, _fixture_map()[1])):
        known = np.flatnonzero(token_map.table >= 0)
        hasher = Hasher(cfg, token_map)
        for length in (1, 2, 3, 4, 41):
            raw = rng.choice(known, size=length).tolist()
            want = _direct(cfg, token_map, raw)
            columns = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
            assert want.shape == (length, len(cfg.engram_layer_ids), columns)
            assert np.array_equal(hasher.ids([], raw), want)
            for cuts in ([1], [2], [1, 2, 3], sorted(set(rng.integers(1, max(length, 2), size=4).tolist()))):
                cuts = [c for c in cuts if c < length]
                assert np.array_equal(_chunked(hasher, raw, cuts), want), (length, cuts)
            # a longer history reads only its last max_ngram - 1 ids
            if length > 5:
                assert np.array_equal(hasher.ids(raw[:5], raw[5:]), want[5:])


def test_the_fixture_ids_are_reproduced_from_its_token_map():
    data, token_map = _fixture_map()
    hasher = Hasher(CFG, token_map)
    raw, split, want = data["raw_ids"], data["chunk_split"], np.array(data["expected"], dtype=np.int64)
    assert want.shape == (64, 2, 24) and split > 3
    assert hasher.pad == token_map.table[CFG.engram_pad_token_id]
    assert np.array_equal(hasher.ids([], raw), want)
    assert np.array_equal(_chunked(hasher, raw, [split]), want)
    assert np.array_equal(_chunked(hasher, raw, [1, 2, split]), want)
    assert (want >= 0).all() and (want < np.array(CFG.engram_num_embeddings)[None, :, None]).all()


def test_rank_columns_split_each_layer_in_halves():
    data, token_map = _fixture_map()
    ids = Hasher(CFG, token_map).ids([], data["raw_ids"])
    halves = [rank_columns(ids, rank) for rank in (0, 1)]
    assert halves[0].shape == halves[1].shape == (64, 2, 12)
    assert np.array_equal(np.concatenate(halves, axis=-1), ids)
    assert (halves[1] >= Hasher(CFG, token_map).offsets[None, :, 12:13]).all()


def test_hasher_refuses_a_map_or_tables_the_config_disagrees_with():
    _, token_map = _fixture_map()
    with pytest.raises(ValueError, match="classes"):
        Hasher(dataclasses.replace(CFG, engram_compressed_vocab_size=99091), token_map)
    with pytest.raises(ValueError, match="rows"):
        Hasher(dataclasses.replace(CFG, engram_num_embeddings=(384006168, 384016681)), token_map)


def test_token_map_normalizes_and_keeps_first_seen_order(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    vocab = ["<pad>", "The", "the", " THE", "Thé", " ", "  ", "\t", "ﬁ", "fi", "Ab\ufffd", "ab\ufffd"]
    model = tokenizers.models.WordLevel({t: i for i, t in enumerate(vocab)}, unk_token="<pad>")
    path = tmp_path / "tokenizer.json"
    tokenizers.Tokenizer(model).save(str(path))
    token_map = TokenMap.build(path, 6)
    # case, edge spaces and accents fold; whitespace alone stays one class; NFKC folds the ligature; text with
    # U+FFFD is keyed raw, so its case does not fold
    assert token_map.table.tolist() == [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5]
    with pytest.raises(ValueError, match="6 classes"):
        TokenMap.build(path, 7)
    with pytest.raises(ValueError, match="sha256"):
        TokenMap.build(path, 6, "0" * 64)
    assert TokenMap.build(path, 6, token_map.sha256).table.tolist() == token_map.table.tolist()


@pytest.mark.skipif(not os.environ.get("TF_DSV41_MODEL"), reason="set TF_DSV41_MODEL to the checkpoint")
def test_the_real_tokenizer_gives_the_pinned_map():
    pytest.importorskip("tokenizers")
    token_map = TokenMap.build(Path(os.environ["TF_DSV41_MODEL"]) / "tokenizer.json", engram_hash.TOKEN_MAP_SIZE,
                               engram_hash.TOKEN_MAP_SHA256)
    assert token_map.size == 99092 and len(token_map.table) == 129280
    data = json.loads((FIXTURES / "engram_ids.json").read_text())
    for token, compressed in data["token_map"].items():
        assert token_map.table[int(token)] == compressed
