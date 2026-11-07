#!/usr/bin/env python3
"""Public Qwen BF16 reference: forward."""
from __future__ import annotations

import copy
import math
import json
import os
import struct
import torch
from qwen35_reference_inputs import (
    TWO_M150,
    U,
    gamma_series,
    json_safe,
    log,
)


def restore_stored_fp32(model, model_dir):
    """Preserve checkpoint FP32 tensors that Transformers auto-dtype would downcast."""
    from safetensors import safe_open
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as stream:
        weight_map = json.load(stream)["weight_map"]
    state = model.state_dict(keep_vars=True)
    restored = []
    for shard in sorted(set(weight_map.values())):
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        keys = [key for key, meta in header.items()
                if key != "__metadata__" and meta["dtype"] == "F32"]
        if not keys:
            continue
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for key in keys:
                if key not in state:
                    raise ValueError("stored FP32 checkpoint tensor is absent from the reference model: " + key)
                original = tensors.get_tensor(key)
                if state[key].shape != original.shape:
                    raise ValueError("stored FP32 checkpoint tensor shape mismatch: " + key)
                prior = str(state[key].dtype)
                changed = not torch.equal(state[key].detach().float(), original)
                state[key].data = original.clone()
                if state[key].dtype != torch.float32 or not torch.equal(state[key].detach(), original):
                    raise ValueError("stored FP32 checkpoint tensor failed byte preservation: " + key)
                restored.append({"key": key, "shape": list(original.shape), "prior_dtype": prior,
                                 "auto_dtype_changed_values": changed})
    return {"count": len(restored), "tensors": restored,
            "all_stored_fp32_bytes_preserved": True,
            "scope": "both precision arms use unchanged checkpoint values; BF16 projections remain BF16"}


def load_model(model_dir, device_map_cpu=True):
    from transformers import Qwen3_5ForConditionalGeneration
    log("loading %s (CPU, bfloat16, low_cpu_mem_usage=True)" % model_dir)
    model, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        device_map={"": "cpu"} if device_map_cpu else None,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        attn_implementation="eager",
    )
    loading_info["stored_fp32_restore"] = restore_stored_fp32(model, model_dir)
    model.eval()
    return model, loading_info


def model_load_report(model, loading_info):
    lanes = model.model.language_model
    rep = {
        "class": type(model).__name__,
        "language_model_class": type(lanes).__name__,
        "model_dtype": str(model.dtype) if hasattr(model, "dtype") else str(next(model.parameters()).dtype),
        "lm_head_dtype": str(model.lm_head.weight.dtype),
        "embed_dtype": str(lanes.embed_tokens.weight.dtype),
        "device": str(model.lm_head.weight.device),
        "use_cache_default": getattr(model.config, "use_cache", None),
        "attn_implementation": model.config._attn_implementation,
        "text_attn_implementation": lanes.config._attn_implementation,
        "num_hidden_layers": lanes.config.num_hidden_layers,
        "layer_types": list(lanes.config.layer_types),
        "loading_info": json_safe(loading_info),
    }
    # dtype inventory
    dtypes = {}
    for name, p in model.named_parameters():
        dtypes.setdefault(str(p.dtype), [0, []])
        dtypes[str(p.dtype)][0] += 1
        if len(dtypes[str(p.dtype)][1]) < 8:
            dtypes[str(p.dtype)][1].append(name)
    rep["parameter_dtypes"] = {k: {"count": v[0], "examples": v[1]} for k, v in dtypes.items()}
    f32_names = [n for n, p in model.named_parameters() if p.dtype == torch.float32]
    rep["float32_parameter_count"] = len(f32_names)
    rep["float32_parameter_examples"] = f32_names[:6]
    rep["float32_parameter_suffixes"] = sorted(set(n.split(".")[-1] for n in f32_names))
    rep["parameter_count"] = sum(p.numel() for p in model.parameters())
    return rep


def upcast_module(module):
    saved_params = []
    saved_buffers = []
    kept_f32 = 0
    cast = 0
    for sub in module.modules():
        for _name, p in list(sub.named_parameters(recurse=False)):
            cur = p.data
            if cur.dtype == torch.float32:
                kept_f32 += 1
                continue
            saved_params.append((p, cur))
            p.data = cur.to(torch.float32)
            cast += 1
        for name, buf in list(sub.named_buffers(recurse=False)):
            if not isinstance(buf, torch.Tensor):
                continue
            if buf.dtype == torch.float32:
                kept_f32 += 1
                continue
            saved_buffers.append((sub, name, buf))
            setattr(sub, name, buf.to(torch.float32))
            cast += 1
    return {"params": saved_params, "buffers": saved_buffers, "cast": cast, "kept_f32": kept_f32}


def restore_module(saved):
    for p, orig in saved["params"]:
        p.data = orig
    for sub, name, orig in saved["buffers"]:
        setattr(sub, name, orig)


def assert_all_f32(module):
    bad = []
    for sub in module.modules():
        for _name, p in sub.named_parameters(recurse=False):
            if p.is_floating_point() and p.dtype != torch.float32:
                bad.append(str(p.dtype))
        for name, buf in sub.named_buffers(recurse=False):
            if isinstance(buf, torch.Tensor) and buf.is_floating_point() and buf.dtype != torch.float32:
                bad.append(str(buf.dtype))
    return bad


def make_text_masks(lang, hidden_ref, seq3, device):
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    kw = {
        "config": lang.config,
        "inputs_embeds": hidden_ref,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": seq3[0],
    }
    return {"full_attention": create_causal_mask(**kw),
            "linear_attention": create_recurrent_attention_mask(**kw)}


def manual_forward(model, ids, upcast=False, verify_layers=None, verify_report=None,
                   track_params=False, capture=None):
    'Run the exact library module graph over the text model'
    lang = model.model.language_model
    cfg = lang.config
    device = lang.embed_tokens.weight.device
    ids = ids.to(device)
    b, L = ids.shape
    h = lang.embed_tokens(ids)
    if upcast:
        h = h.to(torch.float32)
    if capture is not None:
        capture(0, h)
    seq3 = torch.arange(L, device=device).view(1, 1, -1).expand(3, b, -1)
    rope = lang.rotary_emb(h, seq3)
    masks = make_text_masks(lang, h, seq3, device)
    stats = {"cast": 0, "kept_f32": 0, "layers": 0, "restored": 0}
    for i in range(cfg.num_hidden_layers):
        layer = lang.layers[i]
        h_in = h
        saved = None
        if upcast:
            saved = upcast_module(layer)
            stats["cast"] += saved["cast"]
            stats["kept_f32"] += saved["kept_f32"]
            bad = assert_all_f32(layer)
            if bad:
                raise RuntimeError("layer %d has non-f32 tensors after upcast: %s" % (i, bad[:5]))
        h = layer(h, position_embeddings=rope,
                  attention_mask=masks[cfg.layer_types[i]],
                  position_ids=seq3[0], past_key_values=None)
        if capture is not None:
            capture(i + 1, h)
        if saved is not None:
            restore_module(saved)
            stats["restored"] += 1
            stats["layers"] = i + 1
            if verify_layers and i in verify_layers:
                clone = copy.deepcopy(layer).float()
                h_clone = clone(h_in, position_embeddings=rope,
                                attention_mask=masks[cfg.layer_types[i]],
                                position_ids=seq3[0], past_key_values=None)
                if isinstance(h_clone, tuple):
                    h_clone = h_clone[0]
                diff = float((h_clone.float() - h.float()).abs().max())
                verify_report.append({
                    "layer": i, "layer_type": cfg.layer_types[i],
                    "max_abs_diff": diff,
                    "exact_equal": bool(torch.equal(h_clone, h)),
                    "input_tokens": int(L),
                })
                del clone, h_clone
    if upcast:
        saved = upcast_module(lang.norm)
        stats["cast"] += saved["cast"]
        stats["kept_f32"] += saved["kept_f32"]
        h = lang.norm(h)
        restore_module(saved)
        stats["restored"] += 1
    else:
        h = lang.norm(h)
    if capture is not None:
        capture(cfg.num_hidden_layers + 1, h)
    if track_params:
        stats["parameter_dtypes_after"] = sorted(set(str(p.dtype) for p in model.parameters()))
    return h, stats


def library_forward(model, ids, logits_to_keep=0):
    'Reference forward through the public library entry point'
    lang = model.model.language_model
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output

    handle = lang.norm.register_forward_hook(hook)
    try:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=logits_to_keep)
    finally:
        handle.remove()
    return captured["hidden"], out.logits


def head_logits(x_f32, W, vocab_chunk=4096):
    'Full-vocabulary float32 logits for normalized hidden rows'
    n = x_f32.shape[0]
    vocab = W.shape[0]
    out = torch.empty((n, vocab), dtype=torch.float32)
    x = x_f32.to(torch.float32).contiguous()
    for lo in range(0, vocab, vocab_chunk):
        hi = min(vocab, lo + vocab_chunk)
        out[:, lo:hi] = torch.matmul(x, W[lo:hi].to(torch.float32).T)
    return out


def score_rows(W, hidden_f32, ids, positions, row_chunk=64, vocab_chunk=4096,
               compare_logits=None):
    'Score ids[p] from hidden row p-1 for each absolute position p'
    if not positions:
        return [], {"rows": 0}
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(list(ids), dtype=torch.long)
    cmp_max = 0.0
    cmp_exact = True
    cmp_rows = 0
    out = []
    for lo in range(0, len(positions), row_chunk):
        ps = positions[lo:lo + row_chunk]
        idx = torch.tensor([p - 1 for p in ps], dtype=torch.long)
        logits = head_logits(hidden_f32.index_select(0, idx), W, vocab_chunk)
        lse = torch.logsumexp(logits, dim=-1)
        lp = logits - lse[:, None]
        sel_ids = ids[torch.tensor(ps, dtype=torch.long)]
        sel_lp = lp.gather(1, sel_ids[:, None]).squeeze(1)
        sorted_lp, sorted_ids = torch.sort(lp, dim=-1, descending=True, stable=True)
        top_lp = sorted_lp[:, :20]
        top_ids = sorted_ids[:, :20]
        rank = (lp >= sel_lp[:, None]).sum(dim=-1)
        logit_gather = logits.gather(1, sel_ids[:, None]).squeeze(1)
        if compare_logits is not None:
            ref = compare_logits.index_select(0, idx).to(torch.float32)
            cmp_max = max(cmp_max, float((logits - ref).abs().max()))
            cmp_exact = cmp_exact and bool(torch.equal(logits, ref))
            cmp_rows += len(ps)
        top_ids_l = top_ids.tolist()
        top_lp_l = top_lp.tolist()
        sel_lp_l = sel_lp.tolist()
        lse_l = lse.tolist()
        rank_l = rank.tolist()
        logit_l = logit_gather.tolist()
        sel_ids_l = sel_ids.tolist()
        for j, p in enumerate(ps):
            out.append({
                "position": int(p),
                "token_id": int(sel_ids_l[j]),
                "logprob": float(sel_lp_l[j]),
                "logit": float(logit_l[j]),
                "lse": float(lse_l[j]),
                "rank": int(rank_l[j]),
                "top_ids": [int(t) for t in top_ids_l[j]],
                "top_logprobs": [float(v) for v in top_lp_l[j]],
            })
    info = {"rows": len(out), "compare_rows": cmp_rows,
            "compare_max_abs_diff": cmp_max, "compare_exact": cmp_exact}
    return out, info


def bound_profile(hidden_f32, logits_f32, normalizer_f32, picks, W, chunk=4096):
    'Boxed FP64 reference and bounds for captured fp32 logits'
    rows, H = hidden_f32.shape
    vocab = logits_f32.shape[1]
    x = hidden_f32.to(torch.float64)
    native = logits_f32.to(torch.float64)
    normalizer = normalizer_f32.to(torch.float64)
    ref = torch.empty((rows, vocab), dtype=torch.float64)
    magnitude = torch.empty((rows, vocab), dtype=torch.float64)
    xa = x.abs()
    for lo in range(0, vocab, chunk):
        hi = min(vocab, lo + chunk)
        w = W[lo:hi].to(torch.float64)
        ref[:, lo:hi] = x @ w.T
        magnitude[:, lo:hi] = xa @ w.abs().T
    dot_depth = math.ceil(H / 2048) * 8 + 13
    work = gamma_series(dot_depth) * magnitude
    del magnitude
    dot_bound = work + U * (ref.abs() + work) + TWO_M150
    del work
    dot_error = (native - ref).abs()
    native_ref_lse = torch.logsumexp(native, dim=-1)
    red_work = gamma_series(math.ceil(vocab / 256) + 8) + 12 * U * native_ref_lse.abs()
    red_bound = red_work + U * (native_ref_lse.abs() + red_work) + TWO_M150
    red_error = (normalizer - native_ref_lse).abs()
    ref_lse = torch.logsumexp(ref, dim=-1)
    native_lp = (logits_f32 - normalizer_f32[:, None]).to(torch.float64)
    ref_lp = ref - ref_lse[:, None]
    lp_bound = dot_bound + dot_bound.amax(dim=1)[:, None] + red_bound[:, None]
    lp_bound = lp_bound + U * (native.abs() + normalizer.abs()[:, None]) + TWO_M150
    lp_error = (native_lp - ref_lp).abs()
    finite = bool(torch.isfinite(native).all() and torch.isfinite(ref).all() and
                  torch.isfinite(normalizer).all())
    dot_ok = bool((dot_error <= dot_bound).all())
    red_ok = bool((red_error <= red_bound).all())
    lp_ok = bool((lp_error <= lp_bound).all())
    pick_t = torch.tensor([int(p) for p in picks], dtype=torch.long)
    row_ids = torch.arange(rows)
    sel = native[row_ids, pick_t]
    ranks = (native >= native[row_ids, pick_t, None]).sum(dim=1)
    top_ids = torch.argsort(native, dim=1, descending=True, stable=True)[:, :20]
    top_values = native.gather(1, top_ids)
    ratio = lp_error / lp_bound
    dot_ratio = dot_error / dot_bound
    worst_lp = int(ratio.argmax())
    worst_dot = int(dot_error.argmax())
    worst_dot_ratio = int(dot_ratio.argmax())
    worst_abs = int(lp_error.argmax())
    native_best, ref_best = native.argmax(1), ref.argmax(1)
    top_ref = torch.topk(ref, 2, dim=1).values
    differences = [{"row": rr, "native_token": int(native_best[rr]), "fp64_token": int(ref_best[rr]),
                    "fp64_margin": float(top_ref[rr, 0] - top_ref[rr, 1])}
                   for rr in range(rows) if int(native_best[rr]) != int(ref_best[rr])]
    out = {
        "finite": finite,
        "head_bound_passed": dot_ok, "reduction_bound_passed": red_ok,
        "logprob_bound_passed": lp_ok,
        "worst_head_absolute": float(dot_error.max()),
        "worst_head_bound_ratio": float(dot_ratio.max()),
        "worst_head_absolute_row": worst_dot // vocab, "worst_head_absolute_token": worst_dot % vocab,
        "worst_head_bound_row": worst_dot_ratio // vocab, "worst_head_bound_token": worst_dot_ratio % vocab,
        "worst_reduction_absolute": float(red_error.max()),
        "worst_reduction_row": int(red_error.argmax()),
        "worst_reduction_bound_ratio": float((red_error / red_bound).max()),
        "worst_logprob_absolute": float(lp_error.max()), "worst_logprob_bound_ratio": float(ratio.max()),
        "worst_logprob_bound_row": worst_lp // vocab, "worst_logprob_bound_token": worst_lp % vocab,
        "worst_logprob_absolute_row": worst_abs // vocab, "worst_logprob_absolute_token": worst_abs % vocab,
        "argmax_differences": len(differences), "argmax_difference_rows": differences,
        "fp64_top_margins": (top_ref[:, 0] - top_ref[:, 1]).cpu().tolist(),
        # per-row maxima (for aggregating across capture groups)
        "row_dot_error_max": dot_error.max(dim=1).values.cpu().tolist(),
        "row_dot_ratio_max": dot_ratio.max(dim=1).values.cpu().tolist(),
        "row_red_error": red_error.cpu().tolist(),
        "row_lp_error_max": lp_error.max(dim=1).values.cpu().tolist(),
        "row_lp_ratio_max": ratio.max(dim=1).values.cpu().tolist(),
        "row_dot_bound_max": dot_bound.max(dim=1).values.cpu().tolist(),
        "row_red_bound": red_bound.cpu().tolist(),
        # pick-specific values used by the bounded native comparison
        "pick_ref_lp": ref_lp.gather(1, pick_t[:, None]).squeeze(1).cpu(),
        "pick_lp_bound": lp_bound.gather(1, pick_t[:, None]).squeeze(1).cpu(),
        "pick_dot_bound": dot_bound.gather(1, pick_t[:, None]).squeeze(1).cpu(),
        "pick_ref_lse": ref_lse.cpu(),
        "pick_native_lp": native_lp.gather(1, pick_t[:, None]).squeeze(1).cpu(),
        "pick_native_logit": logits_f32.gather(1, pick_t[:, None]).squeeze(1).cpu(),
        "native_lse": normalizer_f32.cpu(),
        "clone_checks": {
            "selected_exact": bool(torch.equal(
                sel.float().contiguous().view(torch.int32),
                logits_f32.gather(1, pick_t[:, None]).squeeze(1).contiguous().view(torch.int32))),
            "top_ids": top_ids.cpu().tolist(),
            "top_values_bits": top_values.float().contiguous().view(torch.int32).cpu().tolist(),
            "argmax": native_best.cpu().tolist(),
            "rank": ranks.cpu().tolist(),
        },
    }
    return out
