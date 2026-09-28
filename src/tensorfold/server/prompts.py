"""Prepare bounded text or image prompts before queueing GPU work."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from tensorfold.server.errors import CapacityError, RequestError
from tensorfold.server.messages import _normalize_tool_call_arguments, normalize_messages


@dataclass
class RenderedPrompt:
    tokens: list[int]
    history_len: int = 0
    vision: Any = None


def has_images(messages):
    return any(isinstance(m, dict) and isinstance(m.get('content'), list)
               and any(isinstance(p, dict) and p.get('type') == 'image_url' for p in m['content'])
               for m in messages or [])


IMAGE_SLOTS = threading.BoundedSemaphore(16)      # requests decoding and processing images at once (host memory)
IMAGE_WAITERS = threading.BoundedSemaphore(128)   # requests waiting for a slot; past that, refused at once
IMAGE_WAIT_S = 60.0


def image_slot():
    """Hold one image-preparation slot, or refuse with 503 when the queue is full or the wait runs out."""
    if not IMAGE_WAITERS.acquire(blocking=False):
        raise CapacityError('image request queue is full; retry shortly')
    try:
        if not IMAGE_SLOTS.acquire(timeout=IMAGE_WAIT_S):
            raise CapacityError('image processing capacity is busy; retry shortly')
    finally:
        IMAGE_WAITERS.release()
    return IMAGE_SLOTS


def prepare_images(frontend, messages, render, *, context_limit=None):
    from tensorfold.vision.images import ImageInputError, load_images, split_images

    if frontend is None:
        raise RequestError('image input requires a supported vision checkpoint served with --vision')
    allow_urls = bool(getattr(frontend, 'allow_urls', False))
    try:
        template, sources = split_images(messages, allow_urls=allow_urls)
    except (ImageInputError, ValueError) as exc:
        raise RequestError(str(exc)) from exc
    slot = image_slot()
    try:
        images = load_images(sources, allow_urls=allow_urls)
        prepared = frontend.prepare(render(template), images, max_prompt_tokens=context_limit)
    except (ImageInputError, ValueError, ImportError) as exc:
        raise RequestError(str(exc)) from exc
    finally:
        slot.release()
    return RenderedPrompt(list(prepared.token_ids), vision=prepared)


def prepare_prompt(app, messages, tools, thinking, prompt, fields):
    if prompt is not None:
        if isinstance(prompt, str):
            with app.tokenizer_lock:
                tokens = [int(t) for t in app.tokenizer.encode(prompt)]
        else:
            tokens = [int(t) for t in prompt]
        return RenderedPrompt(tokens)
    if not has_images(messages):
        tokens, history = app.render(messages, tools, thinking=thinking)
        return RenderedPrompt(tokens, history)
    messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=app.late_system,
                                                                 allow_images=True))
    effort = fields.get('reasoning_effort', app.reasoning_effort)

    def render(template):
        kwargs = dict(add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        if tools:
            kwargs['tools'] = tools
        if thinking and effort:
            kwargs['reasoning_effort'] = effort
        with app.tokenizer_lock:
            return app.tokenizer.apply_chat_template(template, **kwargs)

    return prepare_images(getattr(app, 'vision', None), messages, render, context_limit=app.context_window or None)
