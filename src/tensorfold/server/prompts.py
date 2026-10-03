"""Prepare bounded text or image prompts before queueing GPU work."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from tensorfold.server.errors import CapacityError, RequestError, refusal
from tensorfold.server.messages import _normalize_tool_call_arguments, normalize_messages
from tensorfold.vision.images import DEFAULT_LIMITS, ImageLimits


@dataclass
class RenderedPrompt:
    tokens: list[int]
    history_len: int = 0
    vision: Any = None


def has_images(messages):
    return any(isinstance(m, dict) and isinstance(m.get('content'), list)
               and any(isinstance(p, dict) and p.get('type') in ('image_url', 'video_url') for p in m['content'])
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


def prepare_images(frontend, messages, render, *, context_limit=None, limits: ImageLimits = DEFAULT_LIMITS):
    from tensorfold.vision.images import ImageInputError, ImageSource, load_images, split_images

    if frontend is None:
        raise RequestError('image input requires a supported vision checkpoint served with --vision')
    allow_urls = bool(getattr(frontend, 'allow_urls', False))
    videos = bool(getattr(frontend, 'videos', False))       # a frontend that encodes video frames too
    try:
        template, sources = split_images(messages, limits=limits, allow_urls=allow_urls, allow_videos=videos)
    except (ImageInputError, ValueError) as exc:
        raise RequestError(str(exc)) from exc
    slot = image_slot()
    try:
        images = load_images([s for s in sources if isinstance(s, ImageSource)], limits=limits, allow_urls=allow_urls)
        budget = {} if limits.max_visual_tokens == DEFAULT_LIMITS.max_visual_tokens else \
            {"max_visual_tokens": limits.max_visual_tokens}
        clips = [s for s in sources if not isinstance(s, ImageSource)]
        if clips:
            from tensorfold.vision.videos import load_videos

            budget["videos"] = load_videos(clips, frontend.video_size, allow_urls=allow_urls)
        prepared = frontend.prepare(render(template), images, max_prompt_tokens=context_limit, **budget)
    except (ImageInputError, ValueError, ImportError) as exc:
        raise refusal(str(exc)) from exc                # an image prompt past the window: context_length_exceeded
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
    effort = app.effort_for(fields.get('reasoning_effort'))

    def render(template):
        kwargs = dict(add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        if tools:
            kwargs['tools'] = tools
        if thinking and effort:
            kwargs['reasoning_effort'] = effort
        with app.tokenizer_lock:
            return app.tokenizer.apply_chat_template(template, **kwargs)

    return prepare_images(getattr(app, 'vision', None), messages, render, context_limit=app.context_window or None,
                          limits=getattr(app, 'image_limits', DEFAULT_LIMITS))
