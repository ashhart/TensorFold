#!/usr/bin/env python3
"""Drama 3's codec (Fish S2 modded DAC, codec.pth) -> drama3-codec-v1 safetensors for the native CUDA codec
(zig/src/families/drama3/codec.zig). One-time and offline; the runtime needs no Python.

Weight norm is folded (DAC.fold_weight_norm) and every convolution is stored in the layout its GEMMs read:
  * causal conv, kernel K, dilation d:   <name>.taps [K, O, C]  (tap j reads input row t - (K-1-j) d), <name>.b [O]
  * strided causal conv, kernel 2s:      <name>.wa / .wb [O, s*C] (previous / current block of s rows), .b
  * downsample conv kernel = stride = 2: <name>.w [O, 2*C], .b
  * transposed conv, kernel 2s, stride s (causal, right unpad s): <name>.wa [s*O, C] (x[t]), .wb [s*O, C] (x[t-1]),
    <name>.b [s*O]; kernel = stride (upsample): .wa only
  * VQ: vq.table [4096 + 9*1024, 1024] = out_proj(codebook) + bias (semantic first), vq.cbn [13312, 8] the
    l2-normalized codebooks, vq.in_w [10, 8, 1024], vq.in_b [10, 8]
Everything is fp32. Also writes the codec's own reference outputs for the parity checks with --ref.
usage: codec_convert.py CODEC_DIR OUT.safetensors
       codec_convert.py --ref CODEC_DIR FRAMES.int32 OUT_REF.safetensors [REF.wav]"""
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import build_codec  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def conv_taps(conv):
    w = conv.weight.detach().float()  # [O, C, K]
    return w.permute(2, 0, 1).contiguous(), conv.bias.detach().float()


def strided(conv, s):
    w = conv.weight.detach().float()  # [O, C, 2s]
    O, C, K = w.shape
    assert K == 2 * s
    wa = w[:, :, :s].permute(0, 2, 1).reshape(O, s * C)  # (j, c), j < s
    wb = w[:, :, s:].permute(0, 2, 1).reshape(O, s * C)
    return wa.contiguous(), wb.contiguous(), conv.bias.detach().float()


def transposed(conv, s):
    w = conv.weight.detach().float()  # [C, O, K]
    C, O, K = w.shape
    wa = w[:, :, :s].permute(2, 1, 0).reshape(s * O, C)  # row (j, o)
    b = conv.bias.detach().float().repeat(s)
    if K == s:
        return wa.contiguous(), None, b
    assert K == 2 * s
    wb = w[:, :, s:].permute(2, 1, 0).reshape(s * O, C)
    return wa.contiguous(), wb.contiguous(), b


def transformer(t, p, out):
    for i, L in enumerate(t.layers):
        q = f"{p}.{i}."
        out[q + "attn_norm"] = L.attention_norm.weight.float()
        out[q + "wqkv"] = L.attention.wqkv.weight.float()
        out[q + "wo"] = L.attention.wo.weight.float()
        out[q + "ffn_norm"] = L.ffn_norm.weight.float()
        out[q + "w13"] = torch.cat([L.feed_forward.w1.weight, L.feed_forward.w3.weight]).float()
        out[q + "w2"] = L.feed_forward.w2.weight.float()
        out[q + "g_attn"] = L.attention_layer_scale.gamma.float()
        out[q + "g_ffn"] = L.ffn_layer_scale.gamma.float()
    out[p + ".norm"] = t.norm.weight.float()


def convnext(b, p, out):
    out[p + ".dw"] = b.dwconv.conv.weight.detach().float()[:, 0, :].contiguous()  # [C, 7]
    out[p + ".dw_b"] = b.dwconv.conv.bias.float()
    out[p + ".ln_w"] = b.norm.weight.float()
    out[p + ".ln_b"] = b.norm.bias.float()
    out[p + ".pw1"] = b.pwconv1.weight.float()
    out[p + ".pw1_b"] = b.pwconv1.bias.float()
    out[p + ".pw2"] = b.pwconv2.weight.float()
    out[p + ".pw2_b"] = b.pwconv2.bias.float()
    out[p + ".gamma"] = b.gamma.float()


def ru(u, p, out):
    blk = u.block
    out[p + ".a1"] = blk[0].alpha.float().reshape(-1)
    out[p + ".c7"], out[p + ".c7_b"] = conv_taps(blk[1].conv)
    out[p + ".a2"] = blk[2].alpha.float().reshape(-1)
    t, b = conv_taps(blk[3].conv)
    out[p + ".c1"], out[p + ".c1_b"] = t[0].contiguous(), b


@torch.no_grad()
def convert(codec_dir, path):
    codec = build_codec(codec_dir, torch.device("cpu"))
    codec.fold_weight_norm()
    q = codec.quantizer
    out = {}
    vqs = [q.semantic_quantizer.quantizers[0], *q.quantizer.quantizers]
    tables, cbn, in_w, in_b = [], [], [], []
    for vq in vqs:
        cb = vq.codebook.weight.float()  # [N, 8]
        z = torch.nn.functional.conv1d(cb.T[None], vq.out_proj.weight.float(), vq.out_proj.bias.float())[0].T
        tables.append(z)
        cbn.append(torch.nn.functional.normalize(cb))
        in_w.append(vq.in_proj.weight.float()[:, :, 0])
        in_b.append(vq.in_proj.bias.float())
    out["vq.table"] = torch.cat(tables).contiguous()
    out["vq.cbn"] = torch.cat(cbn).contiguous()
    out["vq.in_w"] = torch.stack(in_w).contiguous()
    out["vq.in_b"] = torch.stack(in_b).contiguous()
    transformer(q.post_module, "post", out)
    transformer(q.pre_module, "pre", out)
    for i, seq in enumerate(q.upsample):
        out[f"up.{i}.wa"], _, out[f"up.{i}.b"] = transposed(seq[0].conv, seq[0].stride)
        convnext(seq[1], f"up.{i}.cn", out)
    for i, seq in enumerate(q.downsample):
        w = seq[0].conv.weight.detach().float()  # [O, C, 2]
        out[f"down.{i}.w"] = w.permute(0, 2, 1).reshape(w.shape[0], -1).contiguous()
        out[f"down.{i}.b"] = seq[0].conv.bias.float()
        convnext(seq[1], f"down.{i}.cn", out)
    m = codec.decoder.model
    out["dec.conv0"], out["dec.conv0_b"] = conv_taps(m[0].conv)
    for bi in range(4):
        blk = m[1 + bi].block
        p = f"dec.{bi}"
        out[p + ".alpha"] = blk[0].alpha.float().reshape(-1)
        out[p + ".wa"], out[p + ".wb"], out[p + ".b"] = transposed(blk[1].conv, blk[1].stride)
        for r in range(3):
            ru(blk[2 + r], f"{p}.ru{r}", out)
    out["dec.final_alpha"] = m[5].alpha.float().reshape(-1)
    out["dec.final"], out["dec.final_b"] = conv_taps(m[6].conv)
    e = codec.encoder.block
    out["enc.conv0"], out["enc.conv0_b"] = conv_taps(e[0].conv)
    for bi in range(4):
        blk = e[1 + bi].block
        p = f"enc.{bi}"
        for r in range(3):
            ru(blk[r], f"{p}.ru{r}", out)
        out[p + ".alpha"] = blk[3].alpha.float().reshape(-1)
        out[p + ".wa"], out[p + ".wb"], out[p + ".b"] = strided(blk[4].conv, blk[4].stride)
        if not isinstance(blk[5], torch.nn.Identity):
            transformer(blk[5], p + ".tf", out)
    out["enc.final_alpha"] = e[5].alpha.float().reshape(-1)
    out["enc.final"], out["enc.final_b"] = conv_taps(e[6].conv)
    out = {k: v.detach().contiguous().cpu() for k, v in out.items()}
    save_file(out, path, metadata={"format": "drama3-codec-v1"})
    print(f"{len(out)} tensors, {sum(v.numel() for v in out.values()) / 1e6:.1f} M values -> {path}")


@torch.no_grad()
def reference(codec_dir, frames_path, path, wav=None):
    """fp32 (no TF32) intermediates of decoding the frames file, and of encoding WAV."""
    import numpy as np

    dev = torch.device("cuda")
    codec = build_codec(codec_dir, dev)
    q = codec.quantizer
    out = {}
    f = torch.from_numpy(np.fromfile(frames_path, dtype=np.int32).reshape(-1, 11).astype(np.int64))
    codes = f[:, 1:].T.contiguous()[None].to(dev)
    idx = codes.clone()
    idx[:, 0] = idx[:, 0].clamp(max=4095)
    idx[:, 1:] = idx[:, 1:].clamp(max=1023)
    zs = q.semantic_quantizer.from_codes(idx[:, :1])[0]
    zr = q.quantizer.from_codes(idx[:, 1:])[0]
    z = zs + zr
    out["dec.vq"] = z[0].T
    z = q.post_module(z)
    out["dec.post"] = z[0].T
    for i, seq in enumerate(q.upsample):
        z = seq(z)
        out[f"dec.up{i}"] = z[0].T
    x = z
    m = codec.decoder.model
    for li in range(len(m)):
        x = m[li](x)
        out[f"dec.model{li}"] = x[0].T
    out["dec.audio"] = x[0, 0]
    if wav:
        from oracle import encode_reference

        cds, audio = encode_reference(codec, wav, dev)
        out["enc.audio44k"] = audio.float()
        out["enc.codes"] = cds.T.contiguous().int()
        a = audio[None, None].to(dev)
        L = a.shape[-1]
        a = torch.nn.functional.pad(a, (0, (-L) % codec.frame_length))
        y = a
        for li, mod in enumerate(codec.encoder.block):
            y = mod(y)
            out[f"enc.block{li}"] = y[0].T
    out = {k: v.detach().float().contiguous().cpu() if k != "enc.codes" else v.cpu() for k, v in out.items()}
    save_file(out, path)
    print(f"reference: {len(out)} tensors -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "--ref":
        reference(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    else:
        convert(sys.argv[1], sys.argv[2])
