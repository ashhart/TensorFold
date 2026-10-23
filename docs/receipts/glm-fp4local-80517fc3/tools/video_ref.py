"""The clips and the processor's prompt ids for video_http.py, as tools/zig/video_served.py computes them.

usage: video_ref.py TOOLS_ZIG MODEL_DIR CLIPS OUT.json [--make] [--video-tokens 16384]
TOOLS_ZIG: a TensorFold tree's tools/zig (glm_video_fixtures.py: the 13 clips and its loader). MODEL_DIR: the
checkpoint's processor and tokenizer files. Needs transformers 5.19 and PyAV 19.0.1 (docs/vision.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tools")
    ap.add_argument("model")
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--video-tokens", type=int, default=16384)
    args = ap.parse_args()
    sys.path.insert(0, args.tools)
    import av
    import transformers
    from glm_video_fixtures import load, make
    from transformers import AutoProcessor

    clips = Path(args.clips)
    if args.make:
        make(clips)
    proc = AutoProcessor.from_pretrained(args.model)
    vp = proc.video_processor
    q = "What moves?"
    ids, sha = {}, {}
    for p in sorted(p for p in clips.iterdir() if p.suffix in {".mp4", ".mov", ".webm", ".mkv"}):
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        if p.name == "short.mp4":
            continue
        frames, meta = load(p, vp, 256)
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # the default: thinking on
        ref = proc(text=[prompt], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                   max_image_tokens=args.video_tokens, return_tensors="np")["input_ids"][0].tolist()
        ids[p.name] = ref
        print(f"{p.name}: {len(frames)} frames, {len(ref)} ids")
    Path(args.out).write_text(json.dumps({"question": q, "video_tokens": args.video_tokens, "model": args.model,
                                          "transformers": transformers.__version__, "av": av.__version__,
                                          "clips_sha256": sha, "ids": ids}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
