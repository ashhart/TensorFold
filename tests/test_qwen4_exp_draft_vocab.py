"""The packaged MTP draft vocabulary: 80,014 ids, a strict superset of the 79,591-id list it replaces.

The head scores the ids in ``families/qwen4_exp/cuda/draft_vocab.txt``; a token outside that list can never be
drafted, and every draft is verified against the target's own samples, so the list is a pure speed decision --
a missing token costs acceptance and never correctness. That is what makes the one invariant worth checking
here checkable without a GPU: nothing the list this one replaces could draft was dropped.

Credit: the construction is ported from MIA AI Lab's reduced-vocabulary MTP drafting ("mia's recipe",
https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, AGPL-3.0-or-later, Copyright (C) 2026
MiaAI Lab, https://x.com/MiaAI_lab): ``files/build_draft_vocab.py`` and ``files/build_draft_vocab_extend.py``.
What is taken is the technique -- a base list is kept whole as a floor, the byte-fallback range is pinned
whatever its frequency, and only real text adds ids, by frequency -- re-expressed for this engine; no file from
that repository is copied.

The superset is proved by digest rather than by trusting the diff: the 423 additions below are removed from the
packaged list, what is left is written in the file's own format (one id per line, sorted), and it must hash to
the digest ``docs/recipes/qwen3.8-flash-next.md`` publishes for the 79,591-id list. A dropped id changes that
hash, and so does an "addition" the old list already carried -- which would make the delta a lie.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

VOCAB_FILE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt"

# The list this file replaces, as its recipe publishes it (docs/recipes/qwen3.8-flash-next.md).
REPLACED_DIGEST = "88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e"
REPLACED_IDS = 79_591
KEEP_BELOW = 65_536          # the generator retains every id below this whatever its frequency
BYTE_FALLBACK = 256          # ids 0-255: what BPE falls back to for accents, CJK and emoji
VOCAB_SIZE = 248_077         # the tokenizer's 248,044 pieces plus its 33 added tokens; its highest id is 248,076

# The 423 ids this list adds, ranked by frequency over the engine's source, its tests and the CPython stdlib
# beside them. Every one of them is at or above KEEP_BELOW: below it the replaced list is already complete.
ADDED = (
    73_627, 73_709, 73_719, 73_756, 73_829, 73_862, 73_867, 73_908, 73_911, 73_976, 74_048, 74_157,
    74_216, 74_271, 74_355, 74_361, 74_366, 74_397, 74_451, 74_537, 74_622, 74_680, 74_681, 74_713,
    74_808, 74_849, 74_874, 74_970, 75_058, 75_193, 75_242, 75_263, 75_446, 75_629, 75_835, 76_002,
    76_080, 76_130, 76_197, 76_225, 76_366, 76_388, 76_410, 76_512, 76_514, 76_659, 76_660, 76_698,
    76_721, 76_763, 76_793, 76_839, 76_946, 77_047, 77_054, 77_218, 77_237, 77_244, 77_305, 77_311,
    77_332, 77_357, 77_363, 77_379, 77_444, 77_517, 77_518, 77_534, 77_557, 77_709, 77_862, 77_892,
    77_974, 78_040, 78_080, 78_279, 78_285, 78_290, 78_361, 78_464, 78_478, 78_621, 78_827, 78_911,
    78_928, 78_998, 79_015, 79_062, 79_073, 79_103, 79_161, 79_185, 79_327, 79_342, 79_420, 79_460,
    79_461, 79_482, 79_507, 79_526, 79_554, 79_631, 79_636, 79_790, 79_920, 79_925, 79_948, 79_995,
    80_068, 80_113, 80_140, 80_268, 80_358, 80_400, 80_527, 80_531, 80_534, 80_583, 80_655, 80_801,
    80_807, 80_829, 81_025, 81_062, 81_100, 81_109, 81_158, 81_160, 81_189, 81_253, 81_280, 81_347,
    81_385, 81_395, 81_400, 81_406, 81_418, 81_486, 81_487, 81_491, 81_522, 81_660, 81_726, 81_787,
    81_793, 81_796, 81_797, 81_898, 81_921, 81_923, 81_929, 81_930, 81_954, 81_978, 82_005, 82_049,
    82_058, 82_069, 82_092, 82_223, 82_377, 82_391, 82_512, 82_624, 82_637, 82_644, 82_680, 82_712,
    82_721, 82_744, 82_819, 82_923, 82_979, 82_989, 83_008, 83_056, 83_167, 83_210, 83_212, 83_616,
    83_633, 83_642, 83_647, 83_679, 83_749, 83_830, 83_849, 83_866, 84_048, 84_137, 84_209, 84_212,
    84_353, 84_425, 84_477, 84_502, 84_544, 84_930, 85_023, 85_082, 85_123, 85_262, 85_292, 85_365,
    85_377, 85_410, 85_448, 85_449, 85_535, 85_538, 85_603, 85_774, 85_777, 85_817, 85_938, 85_965,
    86_077, 86_120, 86_121, 86_448, 86_511, 86_528, 86_546, 86_695, 86_720, 86_891, 86_908, 86_914,
    86_917, 86_992, 87_119, 87_173, 87_208, 87_245, 87_258, 87_262, 87_279, 87_301, 87_330, 87_423,
    87_574, 87_751, 87_784, 87_930, 87_960, 88_085, 88_150, 88_179, 88_194, 88_274, 88_336, 88_428,
    88_694, 88_738, 88_746, 88_753, 88_798, 88_817, 88_858, 88_872, 88_877, 88_943, 88_963, 88_975,
    88_980, 89_072, 89_095, 89_267, 89_303, 89_328, 89_365, 89_441, 89_460, 89_542, 89_589, 89_621,
    89_642, 89_676, 89_748, 89_795, 89_807, 89_817, 89_843, 90_029, 90_097, 90_101, 90_107, 90_247,
    90_278, 90_288, 90_409, 90_560, 90_696, 90_700, 90_751, 90_933, 90_948, 90_980, 91_021, 91_255,
    91_342, 91_369, 91_385, 91_480, 91_585, 91_603, 91_613, 91_694, 91_779, 91_968, 92_144, 92_164,
    92_264, 92_288, 92_291, 92_307, 92_358, 92_361, 92_386, 92_440, 92_493, 92_506, 92_550, 92_772,
    92_931, 92_937, 92_967, 92_996, 93_053, 93_063, 93_099, 93_182, 93_201, 93_212, 93_437, 93_548,
    93_586, 93_628, 93_631, 93_677, 93_720, 93_852, 93_932, 93_983, 94_057, 94_144, 94_175, 94_183,
    94_337, 94_390, 94_440, 94_441, 94_621, 94_639, 94_697, 94_703, 94_878, 94_896, 94_920, 94_989,
    95_095, 95_114, 95_215, 95_265, 95_365, 95_369, 95_457, 95_527, 95_656, 95_677, 95_680, 103_711,
    110_976, 124_158, 156_532, 157_958, 162_282, 163_981, 167_807, 168_321, 172_190, 174_574, 177_029, 178_091,
    179_324, 181_548, 181_933, 182_349, 182_779, 182_874, 186_060, 188_156, 193_873, 194_066, 196_100, 198_584,
    199_550, 201_916, 207_180, 207_286, 210_194, 210_659, 216_284, 218_881, 220_497, 223_569, 224_535, 225_857,
    228_269, 232_382, 233_323, 234_456, 235_132, 236_370, 238_092, 238_616, 240_553, 241_071, 241_326, 241_460,
    245_875, 246_296, 246_798,
)


def test_the_packaged_draft_vocabulary_is_a_strict_superset_of_the_list_it_replaces():
    ids = [int(line) for line in VOCAB_FILE.read_text().split()]
    assert ids == sorted(set(ids)), "the head's row order is the file's order: keep it sorted and distinct"
    assert 0 <= ids[0] and ids[-1] < VOCAB_SIZE, "an id outside the tokenizer's range can never be scored"
    assert list(ADDED) == sorted(set(ADDED)) and min(ADDED) >= KEEP_BELOW, "the additions are a ranked delta"
    assert set(ADDED) <= set(ids), "every id the ranking chose is in the packaged list"

    # The rules the replaced list already followed, and this one keeps: the byte-fallback range, and every id
    # below KEEP_BELOW -- both retained whatever their frequency, which a narrower corpus drops silently.
    assert set(range(BYTE_FALLBACK)) <= set(ids)
    assert set(range(KEEP_BELOW)) <= set(ids)

    remaining = sorted(set(ids) - set(ADDED))
    assert len(remaining) == REPLACED_IDS and len(ids) == REPLACED_IDS + len(ADDED)
    replaced = "".join(f"{tid}\n" for tid in remaining).encode()
    assert hashlib.sha256(replaced).hexdigest() == REPLACED_DIGEST, (
        "the packaged list minus its documented additions is not the list it replaces: an id was dropped"
    )
