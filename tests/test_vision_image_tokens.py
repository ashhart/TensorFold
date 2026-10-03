"""--vision-image-tokens: CUDA Qwen images share a larger visual-token budget, each image capped as before."""

from types import SimpleNamespace as NS

import pytest

from tensorfold import serve_options
from tensorfold.server.prompts import prepare_images
from tensorfold.vision.images import DEFAULT_LIMITS, ImageLimits
from tensorfold.vision.qwen_cuda import MAX_PATCHES, TOKENS_PER_IMAGE, image_runs
from tensorfold.vision.qwen_processing import QwenImageProcessor
from tests.test_vision_qwen_mlx import CONFIG, ImageProcessor, Tokenizer, image
from tests.test_vision_server import Frontend, image_messages


def test_the_default_budget_is_unchanged():
    assert DEFAULT_LIMITS.max_visual_tokens == 4096 == TOKENS_PER_IMAGE


@pytest.mark.parametrize("sizes, runs", [
    ([], []),
    ([MAX_PATCHES], [(0, 1)]),
    ([4096] * 4, [(0, 4)]),                                 # a request that fits one call stays one call
    ([4096] * 5, [(0, 4), (4, 5)]),
    ([10000, 10000, 6000], [(0, 1), (1, 3)]),
    ([1024] * 40, [(0, 16), (16, 32), (32, 40)]),
])
def test_images_encode_in_runs_of_whole_images_within_one_calls_patches(sizes, runs):
    assert image_runs(sizes, MAX_PATCHES) == runs
    assert all(sum(sizes[a:b]) <= MAX_PATCHES or b - a == 1 for a, b in runs)


def test_each_image_keeps_its_cap_however_large_the_shared_budget():
    processor = ImageProcessor()
    front = QwenImageProcessor(CONFIG, processor, Tokenizer())
    front.prepare("<start><image><end>", [image()], max_visual_tokens=16)
    front.prepare("<start><image><end>", [image()], max_visual_tokens=16, max_image_tokens=4)
    assert [call[1]["max_pixels"] for call in processor.calls] == [16 * 32**2, 4 * 32**2]


class BudgetFrontend(Frontend):
    def prepare(self, rendered, images, *, max_prompt_tokens, **budget):
        self.budget = budget
        return super().prepare(rendered, images, max_prompt_tokens=max_prompt_tokens)


def test_the_budget_reaches_the_frontend_only_when_it_is_set():
    front = BudgetFrontend()
    prepare_images(front, image_messages(), str)
    assert front.budget == {}                          # frontends without the keyword keep working
    prepare_images(front, image_messages(), str, limits=ImageLimits(max_visual_tokens=16384))
    assert front.budget == {"max_visual_tokens": 16384}


def _args(**kwargs):
    return NS(vision=True, vision_urls=False, vision_max_images=None, decode_share=None, kv_dtype="bf16", **kwargs)


@pytest.mark.parametrize("tokens", [0, 65537, True])
def test_out_of_range_budgets_refuse(tokens):
    with pytest.raises(ValueError, match="1 to 65,536"):
        serve_options.check(_args(vision_image_tokens=tokens), NS(model_type="qwen3_5", package=NS()), "cuda")


def test_the_budget_needs_vision_and_cuda():
    family = NS(model_type="qwen3_5", package=NS())
    with pytest.raises(ValueError, match="needs --vision"):
        serve_options.check(NS(**{**vars(_args(vision_image_tokens=8192)), "vision": False}), family, "cuda")
    with pytest.raises(ValueError, match="CUDA Qwen image budget"):
        serve_options.check(_args(vision_image_tokens=8192), family, "mlx")


def test_a_cuda_request_carries_the_servers_budget_to_the_frontend():
    from tests.test_vision_server import cuda_app

    front = BudgetFrontend()
    app = cuda_app(front)
    app.image_limits = ImageLimits(max_images=8, max_visual_tokens=16384)
    prepared = app.prepare({"messages": image_messages() * 5, "max_tokens": 2}, True)
    assert len(prepared.vision.image_hashes) == 5 and front.budget == {"max_visual_tokens": 16384}
