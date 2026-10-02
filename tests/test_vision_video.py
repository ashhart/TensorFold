"""Flash Next video input on CUDA: frame groups ride the image path, each its own timestamped vision block."""

import numpy as np
import pytest

from tensorfold.vision.images import ImageInputError, ImageSource, split_images
from tensorfold.vision.qwen_processing import QwenImageProcessor, media_positions
from tensorfold.vision.videos import (DEFAULT_VIDEO_LIMITS, VideoInput, VideoLimits, VideoSource, sample_indices,
                                      video_source)
from tests.test_vision_qwen_mlx import CONFIG, ImageProcessor, Tokenizer, image

DATA = "data:video/mp4;base64,AAAA"


class VideoTokenizer(Tokenizer):
    def convert_ids_to_tokens(self, token):
        return {11: "<video>"}.get(token) or super().convert_ids_to_tokens(token)

    def convert_tokens_to_ids(self, token):
        return {"<video>": 11}.get(token) or super().convert_tokens_to_ids(token)

    def __call__(self, text, **kwargs):
        return {"input_ids": [t for chunk in text.split("<video>") for t in (super().__call__(chunk, **kwargs)
                                                                               ["input_ids"] + [11])][:-1]}


def video(frames=4, side=32):
    return VideoInput(np.zeros((frames, side, side, 3), np.uint8), tuple(range(frames)), 2.0)


def test_each_frame_group_is_its_own_block_and_text_resumes_past_it():
    ids = [100, 8, 11, 9, 101, 8, 11, 9, 102]
    pos, delta, spans, frames = media_positions(ids, np.zeros((0, 3)), [[2, 2, 2]], CONFIG)
    assert spans == () and frames == ((2, 3), (6, 7))
    assert pos[:, 0].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8]] * 3
    assert delta == 0


def test_video_tokens_without_a_video_are_refused():
    with pytest.raises(ValueError, match="Video inputs are not supported"):
        media_positions([8, 11, 9], np.zeros((0, 3)), None, CONFIG)
    with pytest.raises(ValueError, match="without a corresponding video"):
        media_positions([8, 11, 9, 8, 11, 9], np.zeros((0, 3)), [[1, 2, 2]], CONFIG)


def test_video_parts_split_only_where_videos_are_on_and_do_not_count_as_images():
    parts = [{"type": "video_url", "video_url": {"url": DATA}}, {"type": "text", "text": "what moves?"}]
    with pytest.raises(ImageInputError, match="audio and video are unsupported"):
        split_images([{"role": "user", "content": parts}])
    template, sources = split_images([{"role": "user", "content": parts}], allow_videos=True)
    assert template[0]["content"][0] == {"type": "video"} and isinstance(sources[0], VideoSource)
    many = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}] * 4 + parts
    _, sources = split_images([{"role": "user", "content": many}], allow_videos=True)
    assert sum(isinstance(s, ImageSource) for s in sources) == 4                 # the default image count, not 5
    with pytest.raises(ImageInputError, match="at most 2 videos"):
        split_images([{"role": "user", "content": parts * 3}], allow_videos=True)


@pytest.mark.parametrize("url, allow, match", [
    ("http://example.com/a.mp4", True, "data URLs or public HTTPS"),
    ("https://example.com/a.mp4", False, "video URLs are off"),
    ("data:video/mp4;base64," + "A" * (DEFAULT_VIDEO_LIMITS.max_encoded_bytes * 2), True, "encoded byte limit"),
])
def test_video_urls_follow_the_image_rules(url, allow, match):
    with pytest.raises(ImageInputError, match=match):
        video_source({"url": url}, DEFAULT_VIDEO_LIMITS, allow)
    assert video_source({"url": DATA}, DEFAULT_VIDEO_LIMITS, False).url == DATA


def test_a_video_fits_the_request_body_limit():
    assert DEFAULT_VIDEO_LIMITS.max_total_encoded_bytes * 4 // 3 < 32 * 1024**2


def test_frames_are_sampled_at_two_a_second_across_the_whole_video():
    limits = VideoLimits()
    assert sample_indices(300, 30.0, limits).tolist() == np.linspace(0, 299, 20).round().astype(int).tolist()
    assert len(sample_indices(10, 30.0, limits)) == limits.min_frames
    assert len(sample_indices(30 * 3600, 30.0, limits)) == limits.max_frames


def test_each_frame_group_is_timed_at_its_frames_mean():
    clip = VideoInput(np.zeros((3, 32, 32, 3), np.uint8), (0, 2, 4), 2.0)
    assert clip.timestamps(2) == [0.5, 2.0]                   # an odd last frame repeats


def test_prepare_expands_a_video_into_timestamped_frame_groups():
    front = QwenImageProcessor(CONFIG, ImageProcessor(), VideoTokenizer())
    prepared = front.prepare("a<start><video><end>b", [], videos=[video()])
    assert prepared.video_grid_thw.tolist() == [[2, 2, 2]]
    assert prepared.video_pixel_values.shape == (8, 1536)
    assert len(prepared.video_spans) == 2 and prepared.image_spans == ()
    assert prepared.visual_tokens == 2 and prepared.video_hashes == (video().content_hash,)
    text = "".join(chr(t) if t > 20 else f"<{t}>" for t in prepared.token_ids)
    assert "<0.2 seconds><8><11><9>" in text and "<1.2 seconds><8><11><9>" in text


def test_prepare_mixes_images_and_videos_in_prompt_order():
    front = QwenImageProcessor(CONFIG, ImageProcessor(), VideoTokenizer())
    prepared = front.prepare("<start><image><end><start><video><end>", [image()], videos=[video()])
    first_video = prepared.video_spans[0][0]
    assert prepared.image_spans[0][1] <= first_video
    assert prepared.visual_tokens == 4 + 2                    # a 4 x 4 image merged 2 x 2, two frame groups
