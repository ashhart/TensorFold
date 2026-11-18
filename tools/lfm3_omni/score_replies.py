"""UTMOS22 and speech-vs-text WER of saved replies: score_replies.py mos|asr MODEL DEV OUT.json NAME:REPLY_DIR ..."""
import glob
import json
import re
import statistics
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def load(path):
    """A WAV as mono float32 and its rate."""
    with wave.open(str(path)) as w:
        sr, ch, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[width]).astype(np.float32) / float(1 << (8 * width - 1))
    return a.reshape(-1, ch).mean(1), sr


def norm(t):
    """Text for WER: lower case, numbers as words, punctuation dropped."""
    from num2words import num2words
    t = t.lower().replace("%", " percent").replace("’", "'")
    t = re.sub(r"(\d+):(\d+)", r"\1 by \2", t)
    t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", t)
    t = re.sub(r"[^a-z' ]", " ", t.replace("-", " ").replace("‑", " "))
    return re.sub(r"\s+", " ", t).strip()


def mos_pass(out, specs):
    """UTMOS22 strong (SpeechMOS, needs torchaudio) of every rNN.wav, into OUT.mos.json."""
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    res = {}
    for spec in specs:
        name, d = spec.split(":", 1)
        scores = []
        for w in sorted(glob.glob(str(Path(d) / "r*.wav"))):
            a, sr = load(w)
            with torch.no_grad():
                scores.append(round(model(torch.from_numpy(np.ascontiguousarray(a))[None], sr).item(), 4))
        res[name] = scores
        print(name, "utmos median", statistics.median(scores), flush=True)
    json.dump(res, open(out.with_suffix(".mos.json"), "w"))


def asr_pass(model_dir, dev, out, specs):
    """The checkpoint's own ASR of every reply against the reply's text (replies.json), merged with the UTMOS pass."""
    import jiwer
    sys.path.insert(0, str(Path(__file__).parent))
    import refcommon as rc
    model, proc = rc.load(model_dir, dev, torch.bfloat16)
    mos, res = json.load(open(out.with_suffix(".mos.json"))), {}
    for spec in specs:
        name, d = spec.split(":", 1)
        rows = json.load(open(Path(d) / "replies.json"))["rows"]
        hyps = []
        for w in sorted(glob.glob(str(Path(d) / "r*.wav"))):
            inputs = rc.chat(proc, "Perform ASR in English.", w, dev)
            with torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=160, generation_mode="text", do_sample=False)
            ids = o.sequences[0, inputs["input_ids"].shape[1]:o.sequence_lengths[0]]
            hyps.append(proc.decode(ids, skip_special_tokens=True).strip())
        refs, scores = [r["reply"] for r in rows], mos[name]
        wer = jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])
        res[name] = {"n": len(hyps), "utmos_median": round(statistics.median(scores), 3),
                     "utmos_min": round(min(scores), 3), "speech_vs_text_wer": round(wer, 4),
                     "per_reply": [{"text": r, "heard": h, "utmos": s} for r, h, s in zip(refs, hyps, scores)]}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "per_reply"}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "mos":
        mos_pass(Path(sys.argv[4]), sys.argv[5:])
    else:
        asr_pass(sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5:])
