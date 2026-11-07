"""AI depth with Depth-Anything-small (transformers.js's ONNX file): port of Motion Previs Studio's aiDepth.ts."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import math
import os
from pathlib import Path

from motion_media import Encoder, Progress, models_dir, read_frames

DEPTH_MODEL = Path("depth-anything-small") / "model_quantized.onnx"


def ai_depth(video: Path, target: Path, reference: dict, fps: float, progress: Progress) -> None:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("MOTION_DEPTH_THREADS", min(8, os.cpu_count() or 4)))
    session = ort.InferenceSession(str(models_dir() / DEPTH_MODEL), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    # model preprocessing: sides a multiple of 14, ImageNet mean and deviation; long side 518 (the original: short side)
    edge = int(os.environ.get("MOTION_DEPTH_EDGE", 518))
    ratio = reference["width"] / reference["height"]
    width, height = (edge, max(14, round(edge / ratio / 14) * 14)) if ratio >= 1 else (max(14, round(edge * ratio / 14) * 14), edge)
    width = max(14, width // 14 * 14)
    mean, deviation = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    total = max(1, math.ceil((reference["duration"] or 1.0) * fps))
    encoder, low, high = None, None, None
    try:
        for index, frame in enumerate(read_frames(video, fps, width, height)):
            pixels = ((frame.astype(np.float32) / 255.0 - mean) / deviation).transpose(2, 0, 1)[None]
            depth = np.squeeze(session.run(None, {name: pixels})[0]).astype(np.float32)
            # near is bright, as in the original; the range follows the clip slowly so the levels do not flicker
            near, far = float(np.percentile(depth, 99.5)), float(np.percentile(depth, 0.5))
            low, high = (far, near) if low is None else (low + (far - low) * 0.15, high + (near - high) * 0.15)
            grey = (np.clip((depth - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
            if encoder is None:
                encoder = Encoder(target, grey.shape[1], grey.shape[0], fps, "gray", (reference["width"], reference["height"]))
            encoder.write(grey)
            progress.step(index + 1, max(total, index + 1), f"Rendering AI depth {index + 1}/{max(total, index + 1)}")
    finally:
        if encoder:
            encoder.close()
    if encoder is None:
        raise RuntimeError("no frames were read for the AI depth pass")
