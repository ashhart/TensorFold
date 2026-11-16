"""The reference teacher-forced on a dump's reply: reference_forced.py MODEL TARGET NAME DEV DTYPE SYSTEM WAV..."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import refcommon as rc

AUDIO_OUT = 125032
TOP = 4


class Forcer:
    """Scores the model's own greedy picks against a target reply, then feeds the target's (a frame at a time)."""

    def __init__(self, seq, codes, cfg):
        self.seq, self.codes, self.eos_code = seq, codes, cfg.audio_eos_code
        self.end_token = cfg.text_config.eos_token_id
        # the reply opens with the mode marker, which generate() feeds itself: the first pick is slot 1
        self.si = int(len(seq) > 0 and seq[0] == cfg.mixed_start_token_id)
        self.fi = 0
        self.text, self.hits = [], {"text": [0, 0], "codes": [0, 0]}

    def token(self, own, logits):
        """The target's text token for this slot; `own` is what the model chose from `logits`."""
        if self.si >= len(self.seq):
            return self.end_token
        want = int(self.seq[self.si])
        top = torch.topk(logits.float(), TOP)
        self.text.append({"slot": self.si, "ref": want, "pick": int(own), "top_ids": top.indices.tolist(),
                          "top_logits": [float(x) for x in top.values]})
        self.hits["text"][0] += int(int(own) == want)
        self.hits["text"][1] += 1
        self.si += 1
        return want

    def frame(self, own):
        """The target's frame for this slot (the model's own picks fed its later passes); none ends the speech."""
        if not (self.si < len(self.seq) and self.seq[self.si] == AUDIO_OUT and self.fi < self.codes.shape[1]):
            return torch.full_like(own, self.eos_code)
        want = torch.from_numpy(self.codes[:, self.fi]).to(own)
        self.hits["codes"][0] += int((own == want).sum())
        self.hits["codes"][1] += len(want)
        self.si += 1
        self.fi += 1
        return want


def install(model, forcer, size):
    """Routes every text pick and every finished frame of `generate` through `forcer`; returns the undo."""
    cls = type(model)
    sample, frame = cls.__dict__["_sample"], cls.sample_audio_frame

    def pick(c, logits, *a, **k):
        own = sample.__func__(c, logits, *a, **k)
        if logits.shape[-1] == size:
            return own
        suppress = k.get("suppress", a[4] if len(a) > 4 else ())
        scores = logits.float().clone()
        if len(suppress):
            scores[list(suppress)] = -float("inf")
        return own.new_tensor([forcer.token(own, scores)])

    cls._sample = classmethod(pick)
    cls.sample_audio_frame = lambda self, *a, **k: forcer.frame(frame(self, *a, **k))

    def undo():
        cls._sample, cls.sample_audio_frame = sample, frame

    return undo


def main():
    """Every WAV's reply in TARGET/<stem> forced through the reference: its scores as NAME, its states beside."""
    target, name, dev, dtype = Path(sys.argv[2]), sys.argv[3], sys.argv[4], rc.dtype_of(sys.argv[5])
    system = Path(sys.argv[6]).read_text().strip()
    rc.identity_order()
    model, proc = rc.load(sys.argv[1], dev, dtype)
    cfg = model.config
    for wav in sys.argv[7:]:
        rc.wait_idle()
        d = target / Path(wav).stem
        f = Forcer(np.load(d / "gen_seq.npy"), np.load(d / "gen_codes.npy"), cfg)
        undo = install(model, f, cfg.codebook_size)
        lm, steps = model.model.language_model, []
        orig = lm.forward

        def record(*a, **k):
            o = orig(*a, **k)
            steps.append(o.last_hidden_state[0, -1].detach().float().cpu())
            return o

        lm.forward = record
        try:
            with torch.no_grad():
                model.generate(**rc.chat(proc, system, wav, dev), generation_mode="interleaved", max_new_tokens=512,
                               do_sample=False, audio_temperature=0.0, audio_repetition_penalty=1.2)
        finally:
            undo()
            lm.forward = orig
        np.save(d / (Path(name).stem + "_hidden.npy"), torch.stack(steps).numpy())
        out = {"dtype": sys.argv[5], "text": f.hits["text"], "codes": f.hits["codes"], "slots": f.text}
        json.dump(out, open(d / name, "w"))
        print(Path(wav).stem, "text", f.hits["text"], "codes", f.hits["codes"], flush=True)


if __name__ == "__main__":
    main()
