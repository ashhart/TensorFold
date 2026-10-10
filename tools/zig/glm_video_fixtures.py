"""GLM-5.3-Flash video input against transformers' processor, on a fixed clip set (#565's video gate).

    python -I tools/zig/glm_video_fixtures.py --model DIR --clips DIR [--native zig-out/bin/tf-glm-video]
        [--max-tokens 16384] [--max-frames 256] [--make]

The reference is transformers 5.19's Glm5NextProcessor (Glm5NextVideoProcessor) on frames PyAV 19.0.1 decodes
(FFmpeg 9.0.2), as video_utils.read_video_pyav reads them, with the processor's own sample_frames and the same
token budget and frame cap the server passes (max_image_tokens, max_frames). A container that keeps no frame count
(Matroska, WebM: read_video_pyav's total_num_frames is 0 and the loader fails) is given its video packets' count, as
libtfvideo counts them. --make draws and encodes the clip set first (H.264 4:2:0, 4:4:4, full range and BT.709 in
MP4 and MOV; HEVC 8 and 10 bit; VP9 WebM; VP8 MKV; MPEG-4; odd sizes; a clip under a second; one past the frame cap).

For each clip it runs tf-glm-video (libtfvideo decoding, families/glm/video.zig preparing) and checks the sampled
indices, the grid, every patch value bit for bit and each frame group's time text against the expanded prompt.

Needs transformers 5.19 with torch and torchvision, and PyAV 19.0.1.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def scene(i: int, w: int, h: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    f = np.empty((h, w, 3), np.uint8)
    f[..., 0] = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    f[..., 1] = ((y + 3 * i) % 256).astype(np.uint8)
    f[..., 2] = ((x + y + 7 * i) % 200 + 30).astype(np.uint8)
    s = max(8, min(w, h) // 4)
    cx, cy = (5 + 4 * i) % max(w - s, 1), (h - s) // 2
    f[cy:cy + s, cx:cx + s] = (250, 240, 20)
    return f


CLIPS = [  # name, container, codec, pixel format, size, rate, frames, options
    ("h264.mp4", "mp4", "libx264", "yuv420p", (320, 180), 25, 80, {}),
    ("h264-odd.mp4", "mp4", "libx264", "yuv420p", (318, 182), 30, 45, {}),
    ("h264-full.mp4", "mp4", "libx264", "yuvj420p", (256, 144), 24, 30, {}),
    ("h264-709.mov", "mov", "libx264", "yuv420p", (640, 360), Fraction(30000, 1001), 60, {"bt709": "1"}),
    ("h264-444.mp4", "mp4", "libx264", "yuv444p", (200, 120), 25, 25, {}),
    ("h264-hd.mp4", "mp4", "libx264", "yuv420p", (1280, 720), 30, 90, {}),
    ("hevc.mp4", "mp4", "libx265", "yuv420p", (320, 240), 25, 50, {"x265-params": "log-level=error"}),
    ("hevc10.mov", "mov", "libx265", "yuv420p10le", (320, 240), 25, 26, {"x265-params": "log-level=error"}),
    ("vp9.webm", "webm", "libvpx-vp9", "yuv420p", (320, 180), 25, 50, {}),
    ("vp8.mkv", "matroska", "libvpx", "yuv420p", (240, 160), 20, 40, {}),
    ("mpeg4.mp4", "mp4", "mpeg4", "yuv420p", (352, 288), 25, 25, {}),
    ("short.mp4", "mp4", "libx264", "yuv420p", (160, 120), 25, 10, {}),
    ("long.mp4", "mp4", "libx264", "yuv420p", (64, 48), 10, 1500, {}),
]


def make(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, fmt, codec, pix, (w, h), rate, n, opts in CLIPS:
        opts = dict(opts)
        bt709 = opts.pop("bt709", None)
        c = av.open(str(folder / name), "w", format=fmt)
        s = c.add_stream(codec, rate=rate, options=opts)
        s.width, s.height, s.pix_fmt = w, h, pix
        if bt709:
            s.codec_context.colorspace = s.codec_context.color_primaries = s.codec_context.color_trc = 1
        for i in range(n):
            for p in s.encode(av.VideoFrame.from_ndarray(scene(i, w, h), format="rgb24").reformat(format=pix)):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()


def load(path: Path, vp, max_frames: int):
    """read_video_pyav, with a counted frame total where the container keeps none."""
    from transformers.video_utils import VideoMetadata

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        total = int(st.frames)
        if total == 0:
            total = sum(1 for p in c.demux(st) if p.size)
            c.seek(0)
        fps = st.average_rate
        meta = VideoMetadata(total_num_frames=total, fps=float(fps), duration=float(total / fps) if fps else 0.0,
                             video_backend="pyav", height=st.height, width=st.width)
        vp.max_frames = max_frames
        indices = vp.sample_frames(meta)
        if len(indices) == 0:
            return None, meta
        frames = []
        c.seek(0)
        for i, frame in enumerate(c.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame)
        meta.frames_indices = indices
        return np.stack([f.to_ndarray(format="rgb24") for f in frames]), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--native", default="zig-out/bin/tf-glm-video")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--make", action="store_true")
    args = ap.parse_args()
    from transformers import AutoProcessor

    clips = Path(args.clips)
    if args.make:
        make(clips)
    proc = AutoProcessor.from_pretrained(args.model)
    vp = proc.video_processor
    failures = 0
    for path in sorted(p for p in clips.iterdir() if p.suffix in {".mp4", ".mov", ".webm", ".mkv"}):
        frames, meta = load(path, vp, args.max_frames)
        out = clips / "out" / path.stem
        out.mkdir(parents=True, exist_ok=True)
        run = subprocess.run([args.native, args.model, str(path), str(out), str(args.max_tokens), str(args.max_frames)],
                             capture_output=True, text=True)
        if frames is None:
            ok = run.returncode != 0 and "VideoTooShort" in run.stderr
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {path.name}: the processor samples no frames; native {'refuses' if run.returncode else 'answers'}")
            continue
        if run.returncode != 0:
            failures += 1
            print(f"FAIL {path.name}: tf-glm-video: {run.stderr.strip()[-300:]}")
            continue
        native = json.loads((out / "native.json").read_text())
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "What moves?"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                   max_image_tokens=args.max_tokens, return_tensors="np")
        ref = enc["pixel_values_videos"].astype(np.float32)
        grid = [int(v) for v in enc["video_grid_thw"][0]]
        mine = np.fromfile(out / "native_patches.f32", dtype=np.float32).reshape(-1, ref.shape[1]) if ref.ndim == 2 else None
        decoded = sorted(set(int(i) for i in meta.frames_indices))
        prompt = proc.tokenizer.decode(enc["input_ids"][0])
        texts = re.findall(r"<\|end_of_image\|>([^<]*)", prompt)
        same = mine is not None and mine.shape == ref.shape and np.array_equal(mine.view(np.uint32), ref.view(np.uint32))
        diff = 0 if mine is None or mine.shape != ref.shape else int((mine.view(np.uint32) != ref.view(np.uint32)).sum())
        checks = {
            "indices": native["indices"] == decoded,
            "grid": native["grid"] == grid,
            "patches": same,
            "texts": native["texts"] == texts,
        }
        ok = all(checks.values())
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {path.name}: frames {meta.total_num_frames}{' (counted)' if native['counted'] else ''} at {float(meta.fps):.3f}, "
              f"{len(decoded)} decoded, grid {grid}, {ref.size} values {'bit-equal' if same else f'({diff} differ)'}; "
              f"times {texts[0]}..{texts[-1]}; {native['ms']} ms"
              + ("" if ok else f" {[k for k, v in checks.items() if not v]} native {native['indices'][:8]} {native['grid']} {native['texts'][:3]}"))
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
