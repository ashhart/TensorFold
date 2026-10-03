"""Flash Next's kept-state keys: a picture's rows match only the same pixels at the same grid, never a token id."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda.image_rows import key


def _vision(spans, hashes, grids, **videos):
    return SimpleNamespace(image_spans=spans, image_hashes=hashes, image_grid_thw=grids,
                           **({"video_hashes": ()} | videos))


PROMPT = [5, 9, 17, 17, 17, 17, 10, 11, 17, 17, 12]


def test_text_prompts_keep_their_ids_and_unhashed_media_gets_no_key():
    assert key(PROMPT, None) == PROMPT
    assert key(PROMPT, SimpleNamespace()) is None


def test_picture_rows_carry_one_negative_key_each_and_text_is_unchanged():
    ids = key(PROMPT, _vision(((2, 6), (8, 10)), ("a", "b"), ((1, 4, 4), (1, 2, 4))))
    assert [ids[i] for i in (0, 1, 6, 7, 10)] == [5, 9, 10, 11, 12]
    assert len(set(ids[2:6])) == 1 and len(set(ids[8:10])) == 1 and ids[2] != ids[8]
    assert ids[2] < 0 and ids[8] < 0


def test_keys_follow_pixels_and_grid():
    def first(digest, grid):
        return key(PROMPT[:7], _vision(((2, 6),), (digest,), (grid,)))[2]

    assert first("a", (1, 4, 4)) != first("b", (1, 4, 4))
    assert first("a", (1, 4, 4)) != first("a", (1, 2, 8))


def test_a_video_s_frame_groups_carry_its_key_beside_a_picture():
    prompt = [5, 17, 17, 9, 30, 18, 18, 31, 32, 18, 18, 31, 12]  # a picture, then a video's two frame groups

    def keyed(video):
        return key(prompt, _vision(((1, 3),), ("a",), ((1, 2, 2),), video_hashes=(video,),
                                   video_grid_thw=((2, 2, 4),), video_spans=((5, 7), (9, 11))))

    ids = keyed("v")
    assert [ids[i] for i in (0, 3, 4, 7, 8, 11, 12)] == [5, 9, 30, 31, 32, 31, 12]
    assert ids[5] < 0 and len({ids[5], ids[6], ids[9], ids[10]}) == 1 and ids[1] != ids[5]
    assert keyed("w")[:5] == ids[:5] and keyed("w")[5] != ids[5]
