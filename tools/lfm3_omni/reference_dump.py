"""The reference's stage dumps for the parity: reference_dump.py MODEL OUT DEV DTYPE SYSTEM_FILE WAV..."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import refcommon as rc


def f32(t):
    """A tensor as a float32 array on the host."""
    return t.detach().float().cpu().numpy()


def dump(model, proc, codec, system, wav, d, dev, dtype):
    """One spoken turn: front end, encoder, prompt, prefill rows, a greedy reply with its states, depth and codec."""
    d.mkdir(parents=True, exist_ok=True)
    inputs = rc.chat(proc, system, wav, dev)
    ids = inputs["input_ids"][0]
    np.save(d / "input_ids.npy", ids.cpu().numpy().astype(np.int32))
    feats = inputs["input_features"]
    np.save(d / "mel.npy", f32(feats[0]))
    with torch.no_grad():
        enc, _ = model.model.audio_encoder(feats.to(dtype), inputs["feature_attention_mask"].sum(-1))
        np.save(d / "enc.npy", f32(enc[0]))
        np.save(d / "audio_emb.npy", f32(model.model.get_audio_features(feats, inputs["feature_attention_mask"])))
        out = model.model(**dict(inputs.items()), use_cache=False)
        np.save(d / "prefill_hidden.npy", f32(out.last_hidden_state[0]))
        np.save(d / "first_logits.npy", f32(model.lm_head(out.last_hidden_state[0, -1])))
    lm, steps = model.model.language_model, []
    orig = lm.forward

    def record(*a, **k):
        o = orig(*a, **k)
        if len(steps) < 400:
            steps.append(o.last_hidden_state[0, -1].detach().float().cpu())
        return o

    lm.forward = record
    rc.sync(dev)
    t = time.perf_counter()
    with torch.no_grad():
        g = model.generate(**inputs, generation_mode="interleaved", max_new_tokens=512, do_sample=False,
                           audio_temperature=0.0, audio_repetition_penalty=1.2)
    rc.sync(dev)
    total = time.perf_counter() - t
    lm.forward = orig
    seq = g.sequences[0, ids.shape[0]:g.sequence_lengths[0]].cpu().numpy().astype(np.int32)
    n = int(g.audio_lengths[0])
    codes = g.audio_codes[0, :, :n]
    np.save(d / "gen_seq.npy", seq)
    np.save(d / "gen_codes.npy", codes.cpu().numpy().astype(np.int32))
    np.save(d / "step_hidden.npy", torch.stack(steps).numpy())
    seq.tofile(d / "gen_seq.i32")
    codes.cpu().numpy().astype(np.int32).tofile(d / "gen_codes.i32")
    keep = [x for x in seq.tolist() if x != model.config.audio_out_token_id]
    text = proc.tokenizer.decode(keep, skip_special_tokens=True)
    with torch.no_grad():
        z = torch.zeros((1, 12), dtype=torch.long, device=dev)
        masked = torch.ones((1, 12), dtype=torch.bool, device=dev)
        np.save(d / "depth_logits0.npy", f32(model.model.depth_decoder(out.last_hidden_state[0, -1:], z, masked)[0]))
        if n:
            q = codec.quantizer.decode(codes[None]).mT.to(codec.from_quantizer.weight.dtype)
            e = codec.from_quantizer(q)
            dec = codec.decoder
            up = torch.nn.functional.interpolate(e.mT, e.shape[1] * dec.frame_stack, mode="nearest-exact").mT
            np.save(d / "codec_lin.npy", f32(dec.lin(dec.lfm(inputs_embeds=up, use_cache=False).last_hidden_state)[0]))
            wave = codec.decode(codes[None]).audio_values[0].squeeze(0).float().cpu().numpy()
            np.save(d / "reply.npy", wave)
            sf.write(d / "reply.wav", wave, 24000)
    meta = {"wav": Path(wav).name, "text": text, "frames": n, "slots": len(seq), "gen_s": round(total, 3)}
    json.dump(meta, open(d / "meta.json", "w"))
    print(Path(wav).stem, n, len(seq), round(total, 2), text[:100], flush=True)


def main():
    """Every WAV's dumps into OUT/<wav stem>."""
    out, dev, dtype = Path(sys.argv[2]), sys.argv[3], rc.dtype_of(sys.argv[4])
    system = Path(sys.argv[5]).read_text().strip()
    rc.identity_order()
    model, proc = rc.load(sys.argv[1], dev, dtype)
    codec = proc.audio_tokenizer.to(device=dev).eval()
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"system": system, "device": dev, "dtype": sys.argv[4]}, open(out / "system.json", "w"))
    for wav in sys.argv[6:]:
        rc.wait_idle()
        dump(model, proc, codec, system, wav, out / Path(wav).stem, dev, dtype)


if __name__ == "__main__":
    main()
