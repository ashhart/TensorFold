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

    def discard_buffers(self) -> None:
        """Release processor arrays and the host-image slot after model prefill."""

        self.model_inputs.clear()
        self.release()

    def __del__(self) -> None:
        self.release()


@dataclass(frozen=True)
class VisionBatch:
    """Right-padded text rows and concatenated image patches for one Qwen prefill."""

    input_ids: Any
    model_inputs: dict[str, Any]
    lengths: tuple[int, ...]


def merge_vision_prompts(prompts: list[VisionPrompt]) -> VisionBatch:
    """Merge individually processed Qwen image prompts without retaining source images."""

    import numpy as np

    if len(prompts) < 2:
        raise ValueError("a vision batch requires at least two prompts")
    rows = [np.asarray(prompt.model_inputs["input_ids"]) for prompt in prompts]
    lengths = tuple(int(row.shape[-1]) for row in rows)
    max_length = max(lengths)
    dtype = rows[0].dtype
    input_ids = np.zeros((len(rows), max_length), dtype=dtype)
    mask = np.zeros((len(rows), max_length), dtype=np.int64)
    for index, (row, length, prompt) in enumerate(zip(rows, lengths, prompts)):
        input_ids[index, :length] = row.reshape(-1)
        source_mask = prompt.model_inputs.get("mask")
        mask[index, :length] = (
            np.asarray(source_mask).reshape(-1) if source_mask is not None else np.ones(length, dtype=np.int64)
        )

    merged: dict[str, Any] = {"input_ids": input_ids, "mask": mask}
    keys = set().union(*(prompt.model_inputs for prompt in prompts)) - {"input_ids", "mask"}
    for key in keys:
        values = [prompt.model_inputs.get(key) for prompt in prompts]
        if any(value is None for value in values):
            raise ValueError(f"vision batch input {key!r} is missing from one row")
        arrays = [np.asarray(value) for value in values]
        try:
            merged[key] = np.concatenate(arrays, axis=0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"vision batch input {key!r} has incompatible row shapes") from exc
    return VisionBatch(input_ids=input_ids, model_inputs=merged, lengths=lengths)


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
