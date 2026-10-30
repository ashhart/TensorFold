"""The reference runtime's spoken replies and their timing: reference_speech.py MODEL DEV SYSTEM_FILE OUT_DIR WAV..."""
import json
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers.generation.streamers import BaseStreamer

import refcommon as rc


class First(BaseStreamer):
    """Notes when the first speech frame is sampled."""

    def __init__(self, dev):
        self.dev, self.t = dev, None

    def put(self, value):
        if self.t is None:
            rc.sync(self.dev)
            self.t = time.perf_counter()

    def end(self):
        pass


def turn(model, proc, codec, system, wav, seed, dev):
    """One reply with the card's audio settings; first audio = the first frame sampled plus its codec decode alone."""
    torch.manual_seed(seed)
    first = First(dev)
    rc.sync(dev)
    t = time.perf_counter()
    inputs = rc.chat(proc, system, wav, dev)
    with torch.no_grad():
        out = model.generate(**inputs, generation_mode="interleaved", max_new_tokens=512, do_sample=False,
                             audio_temperature=0.4, audio_repetition_penalty=1.2, audio_streamer=first)
    rc.sync(dev)
    total = time.perf_counter() - t
    n = int(out.audio_lengths[0])
    codes = out.audio_codes[:1, :, :n]
    rc.sync(dev)
    tc = time.perf_counter()
    with torch.no_grad():
        codec.decode(codes[:, :, :1])
    rc.sync(dev)
    one = time.perf_counter() - tc
    seq = out.sequences[0, inputs["input_ids"].shape[1]:out.sequence_lengths[0]].tolist()
    text = proc.tokenizer.decode([x for x in seq if x != model.config.audio_out_token_id], skip_special_tokens=True)
    with torch.no_grad():
        audio = codec.decode(codes).audio_values[0].squeeze(0).float().cpu().numpy() if n else None
    row = {"wav": Path(wav).name, "reply": text.strip(), "frames": n, "first_frame_s": round((first.t or t) - t, 4),
           "first_audio_s": round((first.t or t) - t + one, 4), "total_s": round(total, 3),
           "speech_s": round(n * 1920 / 24000, 2)}
    return row, audio


def main():
    """A warm-up turn, then every WAV (seed 1000 + its index), replies saved as OUT_DIR/rNN.wav."""
    dev, system, out = sys.argv[2], Path(sys.argv[3]).read_text().strip(), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    model, proc = rc.load(sys.argv[1], dev, torch.bfloat16)
    codec = proc.audio_tokenizer.to(device=dev).eval()
    wavs = sys.argv[5:]
    turn(model, proc, codec, system, wavs[0], 0, dev)
    rows = []
    for i, w in enumerate(wavs):
        rc.wait_idle()
        row, audio = turn(model, proc, codec, system, w, 1000 + i, dev)
        if audio is not None:
            sf.write(out / f"r{i:02d}.wav", audio, 24000, subtype="PCM_16")
        rows.append(row)
        print(json.dumps(row), flush=True)
    keys = ("first_frame_s", "first_audio_s", "total_s", "speech_s")
    median = {k: round(statistics.median(r[k] for r in rows), 4) for k in keys}
    print("MEDIAN", json.dumps(median))
    json.dump({"system": system, "median": median, "rows": rows}, open(out / "replies.json", "w"), indent=1)


if __name__ == "__main__":
    main()
