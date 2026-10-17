#!/usr/bin/env python3
"""Fish Audio Drama 3 TTS: standalone PyTorch reference ("oracle").

Follows SGLang-Omni 416a70c0 sglang_omni/models/fishaudio_drama3 (sglang_model.py,
audio_decoder.py, decoder_layers.py, sampling.py, prompt.py, weights.py) and the
HF transformers 5.15 Qwen3.5-MoE text model, with no sglang / flashinfer / fla.

Precision policy (mirrors the SGLang serving path):
  * weights bf16; routed MoE experts stay FP8 E4M3 on the GPU with 128x128 block
    scales and are dequantized per selected expert on the fly.
  * --moe-act fp8 (default) emulates SGLang's Fp8MoEMethod (block weights,
    activation_scheme=dynamic): every expert GEMM input is quantized per token
    and per 128-wide K group to E4M3 (scale = amax/448), both operands are
    dequantized exactly and multiplied in fp32.  --moe-act bf16 skips the
    activation quantization (dequantized weights, bf16-rounded activations).
  * every RMSNorm, softmax, rotary, GDN gate/decay/recurrent state and attention
    score is computed in fp32; residual streams and layer outputs are bf16.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the codec's PyTorch modules (package fishcodec): see build_codec
if os.environ.get("FISHCODEC_DIR"):
    sys.path.insert(0, os.environ["FISHCODEC_DIR"])
from oracle_prompt import ReferencePart, build_prompt_rows  # noqa: E402

BF16 = torch.bfloat16
FP8_MAX = 448.0

# sampling.py constants (fixed for Drama 3, not request parameters)
RAS_WINDOW_FRAMES = 16
EMPTY_RAS_SLOT = -1
RAS_TEMPERATURE = 1.0
RAS_TOP_P = 0.95
RAS_TOP_K = 30
MINIMUM_TEMPERATURE = 1e-5


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

from oracle_model import Cfg, Drama3

# --------------------------------------------------------------------------- #
class Sampler:
    def __init__(self, temperature: float, top_k: int, top_p: float, greedy: bool, seed: int, device) -> None:
        self.greedy = greedy
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.dev = device
        self.ras_hits = 0

    def top_k_top_p(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        """sample_top_k_top_p: fp32; drop rank >= top_k and inclusive cumprob > top_p; keep rank 0;
        exponential-race (Gumbel-equivalent) draw."""
        scaled = logits.float() / max(temperature, MINIMUM_TEMPERATURE)
        sl, sid = torch.sort(scaled, dim=-1, descending=True)
        cum = torch.cumsum(torch.softmax(sl, -1), -1)
        ranks = torch.arange(sl.shape[-1], device=logits.device).unsqueeze(0)
        drop = (cum > top_p) | (ranks >= top_k)
        drop[:, 0] = False
        probs = torch.softmax(sl.masked_fill(drop, float("-inf")), -1)
        noise = torch.empty_like(probs).exponential_(1, generator=self.gen)
        choice = torch.argmax(probs / noise, -1, keepdim=True)
        return torch.gather(sid, -1, choice).squeeze(-1)

    def plain(self, logits: torch.Tensor) -> torch.Tensor:
        if self.greedy:
            return torch.argmax(logits.float(), -1)
        return self.top_k_top_p(logits, self.temperature, self.top_k, self.top_p)

    def semantic(self, cand_logits, cand_ids, window, sem_begin, codebook_size) -> torch.Tensor:
        """sample_semantic_with_ras.  Greedy: argmax over {eos, semantic band}; the RAS re-draw is
        skipped (an argmax re-draw over the same logits would return the same token)."""
        if self.greedy:
            return cand_ids[torch.argmax(cand_logits, -1)]
        first = cand_ids[self.top_k_top_p(cand_logits, self.temperature, self.top_k, self.top_p)]
        ras = cand_ids[self.top_k_top_p(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)]
        is_rep = (window.unsqueeze(0) == first.unsqueeze(-1)).any(-1)
        is_sem = (first >= sem_begin) & (first < sem_begin + codebook_size)
        hit = is_rep & is_sem
        self.ras_hits += int(hit.sum())
        return torch.where(hit, ras, first)


# --------------------------------------------------------------------------- #
# Codec (Fish S2 modded DAC). FISHCODEC_DIR holds a package `fishcodec` with sglang-omni 416a70c0's
# fishaudio_s2_pro/fish_speech/models/dac/{modded_dac,rvq}.py and descript-audio-codec 1.0.0's
# dac/nn/{layers,quantize}.py
# (as dac_layers.py, dac_quantize.py) plus CodecMixin from dac/model/base.py (dac_base.py), imports made relative.
# --------------------------------------------------------------------------- #
def build_codec(path: str, device) -> torch.nn.Module:
    from functools import partial

    from fishcodec.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
    from fishcodec.rvq import DownsampleResidualVectorQuantize

    # == fishaudio_s2_pro/fish_speech/configs/modded_dac_vq.yaml
    def tcfg():
        return ModelArgs(block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072,
                         n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5,
                         dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)

    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.5,
        downsample_factor=[2, 2], semantic_codebook_size=4096,
        post_module=WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=tcfg()),
        pre_module=WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=tcfg()),
    )
    general = partial(ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000,
                      norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
    codec = DAC(sample_rate=44100, encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=1536,
                decoder_rates=[8, 8, 4, 2], encoder_transformer_layers=[0, 0, 0, 4],
                decoder_transformer_layers=[4, 0, 0, 0], transformer_general_config=general,
                quantizer=quantizer)
    sd = torch.load(os.path.join(path, "codec.pth"), map_location="cpu", mmap=True, weights_only=True)
    res = codec.load_state_dict(sd, strict=False, assign=True)  # == fishaudio_s2_pro.stages.load_codec
    missing = [k for k in res.missing_keys]
    unexpected = [k for k in res.unexpected_keys if not k.endswith(("freqs_cis", "causal_mask"))]
    if missing or unexpected:
        raise RuntimeError(f"codec load: missing={missing[:10]} unexpected={unexpected[:10]}")
    return codec.eval().to(device=device, dtype=torch.float32)


def load_wav_mono(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)    # int16 PCM -> /32768
    return torch.from_numpy(data.T.copy()).mean(0, keepdim=True), sr


@torch.no_grad()
def encode_reference(codec, path: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """FishReferenceEncodeHook.encode_one (path) in fp32: mono -> torchaudio resample to 44.1k ->
    codec.encode -> [10, frames] int64."""
    import torchaudio

    audio, sr = load_wav_mono(path)
    audio = torchaudio.functional.resample(audio, sr, int(codec.sample_rate))
    audios = audio.squeeze(0).unsqueeze(0).to(device)
    lengths = torch.tensor([audios.shape[1]], dtype=torch.long, device=device)
    idx, _ = codec.encode(audios, lengths)
    if idx.ndim == 3:
        idx = idx[0]
    return idx.long().cpu(), audio.squeeze(0)


@torch.no_grad()
def decode_frames(codec, frames: torch.Tensor, device) -> torch.Tensor:
    """Drama3 vocoder: frames [N, 11] -> codes [1, 10, N] -> codec.from_indices -> float wav [N*2048]."""
    codes = frames[:, 1:].T.contiguous().unsqueeze(0).to(device)
    audio = codec.from_indices(codes.clone())          # from_indices clamps codes in place
    return audio[0, 0, : frames.shape[0] * int(codec.frame_length)].float().cpu()


def write_wav(path: str, audio: torch.Tensor, sr: int = 44100) -> None:
    import wave

    pcm = (audio.clamp(-1, 1) * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model: Drama3, rows: torch.Tensor, samp: Sampler, max_frames: int,
             dump: dict | None, dump_frames: int) -> tuple[torch.Tensor, str, dict]:
    c = model.cfg
    dev = model.dev
    L = rows.shape[0]
    assert L >= 2, "a prompt needs at least two rows (tower prefill pairs each row with the next)"
    st = model.new_state(L + max_frames + 1)
    rows_d = rows.to(dev)
    pos = torch.arange(L, device=dev)
    stats = {}
    torch.cuda.synchronize()
    t0 = time.time()
    rec = {} if dump is not None else None
    model.hooks = {} if dump is not None else None
    hn, logits = model.backbone(rows_d, pos, st, rec)
    model.hooks, hooks = None, model.hooks
    model.tower_prefill(hn[:-1], pos[:-1], rows_d[:-1], rows_d[1:], st, rec)
    torch.cuda.synchronize()
    stats["prefill_s"] = time.time() - t0
    if dump is not None:
        dump.update(rec)
        dump.update(hooks)
        dump["positions"] = pos.cpu()
        dump["backbone_final_norm"] = hn.clone()
        dump["lm_head_logits_last"] = logits[-1].float().clone()
    # RAS window = main tokens of the last 16 rows (prompt rows included), oldest first
    hist = rows[-RAS_WINDOW_FRAMES:, 0].tolist()
    window = torch.tensor([EMPTY_RAS_SLOT] * (RAS_WINDOW_FRAMES - len(hist)) + hist, device=dev)
    frames = []
    finish = "length"
    dec_hidden, dec_cand = [], []
    in_frame = rows_d[-1:]
    p = L - 1
    h_row, lg_row = hn[-1:], logits[-1:]
    t1 = time.time()
    for n in range(max_frames):
        rec0 = dump if (dump is not None and n == 0) else None
        if dump is not None and n < dump_frames:
            dec_hidden.append(h_row[0].clone())
            dec_cand.append(lg_row[0].index_select(-1, model.cand_ids).float())
        fr = model.decode_step(lg_row, h_row, torch.tensor([p], device=dev), in_frame, window, samp, st, rec0)
        window = torch.cat([window[1:], fr[:1]])
        if int(fr[0]) == c.eos:
            finish = "stop"
            break
        frames.append(fr)
        p += 1
        in_frame = fr.unsqueeze(0)
        h_row, lg_row = model.backbone(in_frame, torch.tensor([p], device=dev), st)
    torch.cuda.synchronize()
    stats["decode_s"] = time.time() - t1
    stats["frames"] = len(frames)
    stats["frames_per_s"] = len(frames) / max(stats["decode_s"], 1e-9)
    stats["ras_hits"] = samp.ras_hits
    out = torch.stack(frames).cpu() if frames else torch.zeros(0, 1 + c.n_codebooks, dtype=torch.long)
    if dump is not None and dec_hidden:
        dump["decode.backbone_final_norm"] = torch.stack(dec_hidden)
        dump["decode.candidate_logits"] = torch.stack(dec_cand)
    return out, finish, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the fishaudio/drama-3 checkpoint directory (with codec.pth)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref-audio", action="append", default=[])
    ap.add_argument("--ref-text", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--greedy", action="store_true", help="argmax everywhere (deterministic; RAS inert)")
    ap.add_argument("--max-frames", type=int, default=2048, help="SGLang default cap max_new_tokens=2048")
    ap.add_argument("--moe-act", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="directory for safetensors test vectors")
    ap.add_argument("--dump-frames", type=int, default=20)
    args = ap.parse_args()
    if len(args.ref_audio) != len(args.ref_text):
        ap.error("--ref-audio and --ref-text must be given the same number of times")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False   # reference encode: TF32 flips ~3% of codes
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    codec = build_codec(args.model, dev)
    model = Drama3(args.model, "cuda", args.moe_act)
    c = model.cfg
    print(f"[init] load {time.time() - t_start:.1f}s", file=sys.stderr)

    refs, dump = [], ({} if args.dump else None)
    for i, (ra, rt) in enumerate(zip(args.ref_audio, args.ref_text)):
        codes, wav44 = encode_reference(codec, ra, dev)
        print(f"[ref{i}] {ra}: codes {tuple(codes.shape)}", file=sys.stderr)
        refs.append(ReferencePart(transcript=rt, audio_codes=codes))
        if dump is not None:
            dump[f"ref{i}.codes"] = codes
            dump[f"ref{i}.audio_44k"] = wav44
    rows = build_prompt_rows(args.text, refs, tok, num_codebooks=c.n_codebooks, codebook_size=c.codebook_size)
    print(f"[prompt] {rows.shape[0]} rows", file=sys.stderr)
    samp = Sampler(args.temperature, args.top_k, args.top_p, args.greedy, args.seed, dev)
    frames, finish, stats = generate(model, rows, samp, args.max_frames, dump, args.dump_frames)
    print(f"[gen] {stats['frames']} frames finish={finish} prefill {stats['prefill_s']:.2f}s "
          f"decode {stats['decode_s']:.1f}s = {stats['frames_per_s']:.2f} frames/s ras_hits={stats['ras_hits']}",
          file=sys.stderr)
    audio = decode_frames(codec, frames, dev) if frames.shape[0] else torch.zeros(0)
    stats["audio_s"] = audio.numel() / 44100
    stats["peak_gpu_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_wav(args.out, audio)
        print(f"[out] {args.out}: {stats['audio_s']:.2f}s audio", file=sys.stderr)
    if dump is not None:
        os.makedirs(args.dump, exist_ok=True)
        nd = min(args.dump_frames, frames.shape[0])
        dump["prompt_rows"] = rows
        dump["frames"] = frames[:nd].clone()
        dump["frames_all"] = frames
        dump["waveform"] = decode_frames(codec, frames[:nd], dev) if nd else torch.zeros(0)
        tensors = {k: v.detach().contiguous().cpu() for k, v in dump.items()}
        meta = {
            "text": args.text, "ref_text": json.dumps(args.ref_text), "greedy": str(args.greedy),
            "moe_act": args.moe_act, "seed": str(args.seed), "temperature": str(args.temperature),
            "top_k": str(args.top_k), "top_p": str(args.top_p), "finish": finish,
            "max_frames": str(args.max_frames), "stats": json.dumps(stats),
            "source": "tools/drama3/oracle.py (SGLang-Omni 416a70c0 semantics)",
        }
        save_file(tensors, os.path.join(args.dump, "vectors.safetensors"), metadata=meta)
        with open(os.path.join(args.dump, "meta.json"), "w") as f:
            json.dump({**meta, "stats": stats, "keys": {k: [str(v.dtype), list(v.shape)] for k, v in sorted(tensors.items())}}, f, indent=1)
        print(f"[dump] {len(tensors)} tensors -> {args.dump}", file=sys.stderr)
    stats["total_s"] = time.time() - t_start
    print(json.dumps({"finish": finish, **stats}))


if __name__ == "__main__":
    main()
