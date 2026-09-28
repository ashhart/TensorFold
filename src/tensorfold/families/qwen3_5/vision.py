"""Qwen3.8 image prompt preparation and MLX multimodal prefill metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tensorfold.server.images import PreparedImage


@dataclass
class VisionPrompt:
    """Processor output carried to the scheduler thread for one image request."""

    input_ids: list[int]
    model_inputs: dict[str, Any]
    image_digest: str
    reserved_bytes: int
    release_slot: Callable[[], None] | None = None

    def release(self) -> None:
        callback, self.release_slot = self.release_slot, None
        if callback is not None:
            callback()

    def __del__(self) -> None:
        self.release()


class QwenVisionRuntime:
    """Own the full mlx-vlm model and its matching processor."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def prepare(
        self,
        messages: list[dict[str, Any]],
        image: PreparedImage,
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
    ) -> VisionPrompt:
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if enable_thinking and reasoning_effort:
            template_kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            template_kwargs["tools"] = tools
        formatted = apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            num_images=1,
            **template_kwargs,
        )
        inputs = prepare_inputs(
            self.processor,
            images=[image.image],
            prompts=formatted,
            image_token_index=getattr(self.model.config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(
                self.model.config.model_type, self.processor
            ),
        )
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask", None)
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["mask"] = attention_mask
        model_inputs.update(inputs)
        # MLX streams are thread-local. HTTP validation runs before SSE headers,
        # while model execution belongs to the scheduler thread, so cross that
        # boundary with host arrays rather than thread-owned MLX arrays.
        import numpy as np

        model_inputs = {
            key: np.asarray(value)
            if hasattr(value, "shape") and hasattr(value, "dtype")
            else value
            for key, value in model_inputs.items()
        }
        buffers = image.image.width * image.image.height * 3 + image.encoded_bytes
        buffers += sum(
            int(getattr(value, "nbytes", 0) or 0) for value in model_inputs.values()
        )
        # Vision encoder intermediates substantially exceed the source pixels.
        # Reserve a conservative floor until PromptMemory observes the real peak.
        reserved = buffers + max(1024**3, buffers * 16)
        return VisionPrompt(
            input_ids=[int(token) for token in model_inputs["input_ids"].reshape(-1).tolist()],
            model_inputs=model_inputs,
            image_digest=image.sha256,
            reserved_bytes=reserved,
        )
