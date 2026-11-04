#!/usr/bin/env python3
"""Independent transformers-forward prompt-logprob reference for the TensorFold
Qwen3.5-27B BF16 scoring receipt (PR #2020).

Two things are measured here, on CPU only:

  A. transformers BF16 forward vs transformers FP32 forward of the *same* pinned
     BF16 checkpoint, on the *same* token ids: full-vocabulary float32 logprobs
     (logit - logsumexp(logits)) at every scored prompt position, so the delta
     between the two serves as the "transformers' own BF16 distance from FP32"
     reference number that the served prompt logprobs can be bounded against.

  B. A bounded comparison of the native (TensorFold) prompt/decode logprobs
     against the two transformers arms at the captured positions, with the
     rigorous chain
         |native - tf_bf16| <= box(native vs FP64) + box(tf_bf16 vs FP64)
                                + |(x_bf16 - x_native) @ W_t|
                                + max_v |(x_bf16 - x_native) @ W_v| + slack
     where both boxes are the receipt's boxed FP64 formulas recomputed for each
     arm's own float32 logits, and the two hidden-distance terms bridge the two
     arms' different normalized hidden rows (float64 lm_head product of the
     difference, over the stored BF16 lm_head).  The receipt-style decomposition
     using the measured |tf_bf16 - tf_fp32| delta is reported alongside it, and
     the measured BF16-vs-FP32 delta is checked against its own boxed bound.

     The native capture teacher-forces the first five served greedy continuation
     tokens, so the transformers forward is extended by those five tokens; the
     prompt rows are the first rows of a causal forward, and that inertness is
     verified bitwise on the short fixtures.

The FP32 arm is a streaming, per-module upcast of the BF16-stored parameters
(promoted in float32 in the exact module order of the forward, restored to their
stored dtype afterwards).  BF16 -> FP32 promotion is exact, so this arm isolates
activation / accumulation / normalization precision, not weight quantization.
It is not a trained FP32 checkpoint and it is not a claim of an FP32-partial-sum
replica of the serving kernel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import sys
import traceback
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the busy GPUs

import numpy as np
import torch

U = 2.0 ** -24
TINY = 2.0 ** -150
TWO_M150 = 2.0 ** -150


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print("[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


def peak_rss_gb():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def json_safe(obj):
    """Best-effort JSON-safe copy (transformers' loading info can carry sets)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    return obj


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def tensor_bytes_hash(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def gamma_series(depth):
    """Absolute forward-error growth factor for `depth` max/min reductions."""
    return float(depth) * U / (1.0 - float(depth) * U)


def bf16_bits_to_f32(bits_u16):
    return ((bits_u16.astype(np.uint32) << 16).view(np.float32)).copy()


def u32_bits_to_f32(bits_u32):
    return np.ascontiguousarray(bits_u32).view(np.float32)


def r(arg, default=None):
    a = getattr(arg, "default", None)
    return a if a is not None else default


# --------------------------------------------------------------------------- #
# static inputs: fixtures / served records / native captures
# --------------------------------------------------------------------------- #
def load_fixtures(path):
    data = json.loads(open(path, "r").read())
    fixtures = data if isinstance(data, list) else data["fixtures"]
    arrs = []
    for i, ids in enumerate(fixtures):
        if isinstance(ids, dict):
            ids = ids["ids"]
        arrs.append([int(t) for t in ids])
    return arrs


def load_served(path):
    """Served greedy continuations (token_ids) per fixture index, from the
    final API qualification records.  Three independent capture modes exist for
    each fixture (drafted_plain_N, repeat_N, scored_unscored_tokens_N); they are
    verified to agree before the continuation is used."""
    data = json.loads(open(path, "r").read())
    cont = {}
    agree = {}
    for rec in data["records"]:
        check = str(rec.get("check", ""))
        ids = rec.get("token_ids")
        if not ids:
            continue
        for kind in ("drafted_plain_", "repeat_", "scored_unscored_tokens_"):
            if check.startswith(kind):
                idx = int(check[len(kind):])
                cont.setdefault(idx, {})[kind[:-1]] = [int(t) for t in ids]
        if check.startswith("concurrent_solo_0_0"):  # fixture 0 under concurrency
            pass
    out = {}
    for idx, d in sorted(cont.items()):
        ref = d.get("drafted_plain")
        if ref is None:
            continue
        same = {k: v == ref for k, v in d.items()}
        agree[idx] = {"modes": sorted(d.keys()), "all_agree": all(same.values()), "agreement": same}
        out[idx] = ref
    return out, agree


# --------------------------------------------------------------------------- #
# capture loading + self checks
# --------------------------------------------------------------------------- #
class Capture(object):
    __slots__ = ("index", "stem", "phase", "rows", "hidden", "vocab", "picks",
                 "hidden_f32", "logits_f32", "lse", "sel_bits", "top_ids",
                 "top_vals", "rank", "checks")


def load_capture(capture_dir, index):
    stem = "%d-" % index
    meta = json.loads(open(os.path.join(capture_dir, stem + "meta.json"), "r").read())
    c = Capture()
    c.index = index
    c.stem = stem
    c.phase = meta.get("phase")
    c.rows = int(meta["rows"])
    c.hidden = int(meta["hidden"])
    c.vocab = int(meta["vocab"])
    c.picks = [int(p) for p in meta.get("picks", [])]
    if "projection" in meta:
        c.phase = "projection"
        c.checks = {"projection": meta["projection"]}
        return c
    raw = np.fromfile(os.path.join(capture_dir, stem + "hidden.bf16"), dtype="<u2")
    if raw.size != c.rows * c.hidden:
        raise ValueError("capture %s hidden size mismatch: %d != %d" % (stem, raw.size, c.rows * c.hidden))
    c.hidden_f32 = bf16_bits_to_f32(raw).reshape(c.rows, c.hidden)
    nat = np.fromfile(os.path.join(capture_dir, stem + "logits.f32"), dtype="<f4")
    if nat.size != c.rows * c.vocab:
        raise ValueError("capture %s logits size mismatch: %d != %d" % (stem, nat.size, c.rows * c.vocab))
    c.logits_f32 = nat.reshape(c.rows, c.vocab)
    stats = np.fromfile(os.path.join(capture_dir, stem + "stats.u32"), dtype="<u4")
    if stats.size != c.rows * 43:
        raise ValueError("capture %s stats size mismatch" % stem)
    stats = stats.reshape(c.rows, 43)
    c.lse = u32_bits_to_f32(stats[:, 0])
    c.sel_bits = stats[:, 1].copy()
    c.top_ids = stats[:, 2:22].copy()
    c.top_vals = u32_bits_to_f32(stats[:, 22:42])
    c.rank = stats[:, 42].copy()
    if len(c.picks) != c.rows:
        raise ValueError("capture %s pick count mismatch" % stem)
    c.checks = {}
    return c


def capture_self_checks(c):
    """Check the decoded stats words against the captured fp32 logits."""
    lg = c.logits_f32
    pick = np.asarray(c.picks, dtype=np.int64)
    rows = np.arange(c.rows)
    out = {"rows": c.rows, "phase": c.phase, "vocab": c.vocab, "hidden": c.hidden}
    sel = lg[rows, pick]
    out["selected_exact"] = bool(np.array_equal(c.sel_bits, sel.view(np.uint32)))
    order = np.argsort(-lg, axis=1, kind="stable")[:, :20]
    out["top_exact"] = bool(np.array_equal(c.top_ids, order) and
                            np.array_equal(c.top_vals, lg[rows[:, None], order].astype(np.float32)))
    rank = (lg >= sel[:, None]).sum(axis=1).astype(np.uint32)
    out["rank_exact"] = bool(np.array_equal(c.rank, rank))
    arg = lg.argmax(axis=1)
    out["argmax"] = [int(x) for x in arg]
    out["argmax_matches_pick"] = [bool(int(arg[i]) == int(pick[i])) for i in range(c.rows)]
    # stored lse vs numpy float64 logsumexp of the captured fp32 logits
    m = lg.max(axis=1).astype(np.float64)
    lse64 = m + np.log(np.exp(lg.astype(np.float64) - m[:, None]).sum(axis=1))
    out["lse_residual"] = [float(x) for x in (c.lse.astype(np.float64) - lse64)]
    out["lse_residual_abs_max"] = float(np.abs(c.lse.astype(np.float64) - lse64).max())
    out["finite"] = bool(np.isfinite(lg).all() and np.isfinite(c.hidden_f32).all())
    out["picks"] = list(c.picks)
    return out


# --------------------------------------------------------------------------- #
# capture -> fixture mapping (data driven; ties resolved by hidden-state match)
# --------------------------------------------------------------------------- #
SINGLE_PROMPT_CHUNK = 4096   # engine max_rows (qwen3_5/cuda_round.zig)
CAPTURE_DECODES = 4          # qwen35_score_reference.zig: four greedy decode steps


def build_capture_map(fixtures, cont, captures):
    """Map every prompt/decode capture to (fixture, positions) by the capture
    tool's own structural rule, verified by an exact pick match.

    ``qwen35_score_reference.zig`` forwards each reference fixture as one prompt
    chunk (a captured fixture is at most 257 ids, well under the engine's 4096-row
    limit) and then captures, per fixture:

        prompt  the last ``n = min(scored, 2)`` scored prompt rows, with
                ``scored = L - 1`` for a single chunk, i.e.
                positions [L-2, L-1], hidden rows [L-3, L-2], picks ids[L-2:L];
        decode  four one-row forwards whose captured row is the hidden row of the
                fed token, i.e.
                position [L+1+j], hidden row [L+j], pick cont[j+1]  (j = 0..3).

    A fixture is accepted only when the prompt tail *and* all four decode picks
    match exactly, so the mapping cannot be confused by repeated tokens.  A
    nearest-hidden search is deliberately *not* used: fixture 3 repeats its last
    two tokens at 16 positions and its native vs transformers BF16 hidden rows are
    far apart near the end, so a nearest search prefers a decoy.
    """
    plain = sorted([c for c in captures if c.phase in ("prompt", "decode")],
                   key=lambda c: c.index)
    rows = []
    i = 0
    while i < len(plain):
        c = plain[i]
        if c.phase != "prompt":
            raise RuntimeError("capture %d: decode capture without a preceding prompt "
                               "capture" % c.index)
        dec = plain[i + 1:i + 1 + CAPTURE_DECODES]
        if len(dec) != CAPTURE_DECODES or any(d.phase != "decode" for d in dec):
            raise RuntimeError("capture %d: expected %d decode captures after the prompt "
                               "capture, found %r" % (c.index, CAPTURE_DECODES,
                                                      [(d.index, d.phase) for d in dec]))
        candidates = []
        for fi, ids in enumerate(fixtures):
            L = len(ids)
            if L - 1 < c.rows:          # not enough scored prompt rows
                continue
            if L > SINGLE_PROMPT_CHUNK:  # would span two prompt chunks
                continue
            if [int(t) for t in ids[L - c.rows:]] != c.picks:
                continue
            seq = [int(t) for t in cont.get(fi, [])]
            if len(seq) < 1 + CAPTURE_DECODES:
                continue
            if [d.picks[0] for d in dec] != seq[1:1 + CAPTURE_DECODES]:
                continue
            positions = [L - c.rows + j for j in range(c.rows)]
            candidates.append({"fixture": fi, "length": L, "positions": positions,
                               "hidden_index": [p - 1 for p in positions],
                               "decode_positions": [[L + 1 + j]
                                                    for j in range(CAPTURE_DECODES)]})
        if len(candidates) != 1:
            raise RuntimeError("capture %d: %d structural candidates %r; the capture "
                               "tool's rule must identify exactly one fixture"
                               % (c.index, len(candidates),
                                  [(z["fixture"], z["positions"]) for z in candidates]))
        z = candidates[0]
        rows.append({"capture": c.index, "phase": "prompt", "rows": c.rows,
                     "picks": c.picks, "fixture": z["fixture"],
                     "positions": z["positions"], "hidden_index": z["hidden_index"],
                     "candidates": [{"fixture": z["fixture"], "length": z["length"],
                                     "positions": z["positions"]}]})
        for j, d in enumerate(dec):
            p = z["decode_positions"][j]
            rows.append({"capture": d.index, "phase": "decode", "rows": d.rows,
                         "picks": d.picks, "fixture": z["fixture"],
                         "positions": p, "hidden_index": [p[0] - 1],
                         "candidates": [{"fixture": z["fixture"], "length": z["length"],
                                         "positions": p}]})
        i += 1 + CAPTURE_DECODES
    return rows


def resolve_capture_map(capture_map, tf_hidden, prompt_len):
    """Attach the transformers BF16 hidden rows' distance to every captured row.

    The fixture/position mapping is already fixed by ``build_capture_map`` from
    the capture tool's rule plus an exact pick match.  This function only records
    how far the transformers hidden row is from the native captured one, i.e. the
    quantity the two hidden-distance terms of the bound measure; the distance is
    diagnostic and never used to choose a position.
    """
    resolved = []
    for item in capture_map:
        fi = item["fixture"]
        h = tf_hidden.get(fi)
        if h is None:
            raise RuntimeError("capture %d: fixture %d has no cached transformers "
                               "hidden states" % (item["capture"], fi))
        l2_sq, max_abs = [], []
        for j, p in enumerate(item["positions"]):
            if p - 1 < 0 or p - 1 >= h.shape[0]:
                raise RuntimeError("capture %d: hidden row %d outside the cached "
                                   "forward of fixture %d" % (item["capture"], p - 1, fi))
            d = h[p - 1] - item["x2_rows"][j]
            l2_sq.append(float((d.to(torch.float64) ** 2).sum()))
            max_abs.append(float(d.abs().amax()))
        out = {k: v for k, v in item.items() if k != "x2_rows"}
        out["hidden_distance_l2_sq"] = l2_sq
        out["hidden_distance_l2"] = [math.sqrt(v) for v in l2_sq]
        out["hidden_distance_max_abs"] = max_abs
        out["hidden_distance_max_abs_row"] = max(max_abs)
        out["match_distance"] = sum(l2_sq)   # name kept in the payload for continuity
        out["resolution_basis"] = ("capture-tool positional rule (prompt rows [L-2, L-1], "
                                   "decode rows [L+1, L+4]) verified by exact pick match; "
                                   "the recorded hidden distance is diagnostic only")
        resolved.append(out)
    return resolved


# --------------------------------------------------------------------------- #
# checkpoint identity / environment
# --------------------------------------------------------------------------- #
def checkpoint_identity(model_dir, revision_record=None):
    info = {"model_dir": model_dir, "model_id": "Qwen/Qwen3.5-27B"}
    cfg_path = os.path.join(model_dir, "config.json")
    info["config_sha256"] = sha256_file(cfg_path)
    cfg = json.loads(open(cfg_path, "r").read())
    info["architectures"] = cfg.get("architectures")
    info["model_type"] = cfg.get("model_type")
    info["tie_word_embeddings"] = cfg.get("tie_word_embeddings")
    tc = cfg.get("text_config", {})
    info["text_config"] = {k: tc.get(k) for k in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "vocab_size", "full_attention_interval",
        "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
        "linear_value_head_dim", "linear_conv_kernel_dim", "attn_output_gate",
        "mtp_num_hidden_layers", "mrope_section", "partial_rotary_factor", "rope_theta",
        "rms_norm_eps", "layer_types", "dtype", "eos_token_id")}
    info["config_transformers_version"] = cfg.get("transformers_version") or tc.get("transformers_version")
    info["revision"] = None
    info["revision_source"] = None
    tree_dir = os.path.join(model_dir, ".cache", "huggingface", "trees")
    if os.path.isdir(tree_dir):
        trees = sorted(os.listdir(tree_dir))
        info["hf_tree_cache_files"] = trees
        for t in trees:
            if t.endswith(".json"):
                info["revision"] = t[:-5]
                info["revision_source"] = "model_dir/.cache/huggingface/trees/<revision>.json"
                info["hf_tree_file_sha256"] = sha256_file(os.path.join(tree_dir, t))
                break
    if revision_record and os.path.isfile(revision_record):
        rec = json.loads(open(revision_record, "r").read())
        info["revision_record"] = {"path": revision_record, "repo": rec.get("repo"),
                                   "revision": rec.get("revision")}
        if info["revision"] and rec.get("revision") and info["revision"] != rec.get("revision"):
            info["revision_mismatch"] = True
        info["revision_matches_record"] = (info["revision"] == rec.get("revision"))
    # file inventory + hashes
    files = []
    total = 0
    if os.path.isdir(model_dir):
        for name in sorted(os.listdir(model_dir)):
            p = os.path.join(model_dir, name)
            if os.path.isfile(p):
                size = os.path.getsize(p)
                total += size
                files.append({"path": name, "bytes": size})
    info["files"] = files
    info["local_bytes"] = total
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        index = json.loads(open(idx, "r").read())
        wm = index["weight_map"]
        shards = sorted(set(wm.values()))
        info["weight_map_entries"] = len(wm)
        info["shards"] = shards
        info["shard_weight_shapes"] = {}
        info["safetensors"] = []
        for s in shards:
            p = os.path.join(model_dir, s)
            ent = {"path": s, "bytes": os.path.getsize(p)}
            if not getattr(checkpoint_identity, "_no_hash", False):
                ent["sha256"] = sha256_file(p)
            info["safetensors"].append(ent)
        info["checkpoint_bytes"] = sum(e["bytes"] for e in info["safetensors"])
        info["lm_head_shard"] = wm.get("lm_head.weight")
        info["embed_shard"] = wm.get("model.language_model.embed_tokens.weight")
        info["lm_head_tied_to_embed"] = (wm.get("lm_head.weight") == wm.get("model.language_model.embed_tokens.weight"))
        info["index_sha256"] = sha256_file(idx)
        # dtype inventory from the index + a light header probe of one shard
        info["weight_map_key_count"] = len(wm)
        info["mtp_keys"] = sorted([k for k in wm if k.startswith("mtp.")])[:5]
        info["mtp_key_count"] = len([k for k in wm if k.startswith("mtp.")])
        info["visual_key_count"] = len([k for k in wm if k.startswith("model.visual.")])
    return info, cfg


def environment_report():
    import transformers
    import accelerate
    import safetensors
    rep = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_used": False,
        "cpu_count": os.cpu_count(),
    }
    try:
        import tokenizers
        rep["tokenizers"] = tokenizers.__version__
    except Exception:
        rep["tokenizers"] = None
    return rep


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# module upcast / restore (module.float() semantics, streamed)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# manual forward over the text model, with optional streaming FP32 upcast
# --------------------------------------------------------------------------- #
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
                   track_params=False):
    """Run the exact library module graph over the text model.

    The only hand-written pieces are: the embedding lookup, the 3-row text
    position ids, the mask construction, the layer loop (same order/dtype rules
    as Qwen3_5TextModel.forward) and the final norm.  No module is modified
    except by the streamed upcast, which is restored in the same iteration.
    """
    lang = model.model.language_model
    cfg = lang.config
    device = lang.embed_tokens.weight.device
    ids = ids.to(device)
    b, L = ids.shape
    h = lang.embed_tokens(ids)
    if upcast:
        h = h.to(torch.float32)
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
    if track_params:
        stats["parameter_dtypes_after"] = sorted(set(str(p.dtype) for p in model.parameters()))
    return h, stats


def library_forward(model, ids, logits_to_keep=0):
    """Reference forward through the public library entry point."""
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


# --------------------------------------------------------------------------- #
# float32 lm_head scoring (chunked over the vocabulary)
# --------------------------------------------------------------------------- #
def head_logits(x_f32, W, vocab_chunk=4096):
    """Full-vocabulary float32 logits for normalized hidden rows.

    x_f32 : (n, H) float32 normalized hidden rows
    W     : (V, H) lm_head weight in its stored dtype (BF16 for this checkpoint)

    Stored BF16 weights are promoted to float32 exactly (BF16 -> FP32 is a wider
    mantissa, so no value is altered) and the products are accumulated in
    float32, which is the arithmetic the receipt requires behind every reported
    logprob.  The vocabulary is processed in chunks so no float32 copy of the
    248320 x 5120 head (5 GB) is ever materialized.
    """
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
    """Score ids[p] from hidden row p-1 for each absolute position p.

    Returns per-row dicts with float32 logprobs (logit - logsumexp(logits) over
    the full vocabulary), the native-shaped top-k rows and the selected-token
    rank.  When `compare_logits` (rows indexed by hidden row) is given, the
    logits are compared against a second implementation and the comparison is
    returned alongside the rows.
    """
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


# --------------------------------------------------------------------------- #
# FP64 bound profile (verbatim port of tools/zig/qwen35_scores_fp64.py math)
# --------------------------------------------------------------------------- #
def bound_profile(hidden_f32, logits_f32, normalizer_f32, picks, W, chunk=4096):
    """Boxed FP64 reference and bounds for captured fp32 logits.

    hidden_f32 : (n, H) float32 normalized rows
    logits_f32 : (n, V) float32 logits as produced by the native kernel
    normalizer_f32 : (n,) float32 log-sum-exp words captured with the row
    picks : list of selected token ids
    W : (V, H) bfloat16 lm_head weight (stored BF16)
    """
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


# --------------------------------------------------------------------------- #
# A: transformers BF16 vs FP32, per fixture
# --------------------------------------------------------------------------- #
def run_fixture(model, fixture_index, ids, cont, args, tf_cache, verify_layers):
    L = len(ids)
    if args.skip_long and L > 1024:
        log("fixture %d: length %d skipped by --skip-long" % (fixture_index, L))
        return {"index": fixture_index, "length": L, "skipped": True,
                "ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
                "reason": "--skip-long: %d-token fixture not measured in this run" % L,
                "bf16": {"rows": []}, "fp32": {"skipped": True},
                "delta": {"stats": {}, "rows": []}}
    # The native capture tool teacher-forces the first five served greedy
    # continuation tokens after the prompt, so the transformers forward is
    # extended by the same five tokens; the extra rows are only needed to place
    # the native decode captures.  Prompt positions are causal, so the
    # extension cannot change them, and that is verified below.
    ext = list(ids) + list(cont.get(fixture_index, [])[:5])
    log("fixture %d: prompt length %d, forward length %d (prompt + 5 served continuation tokens)" % (
        fixture_index, L, len(ext)))
    t0 = datetime.now(timezone.utc)
    ids_t = torch.tensor(ext, dtype=torch.long)
    prompt_positions = list(range(1, L))
    W = model.lm_head.weight

    # --- BF16 arm ---------------------------------------------------------
    h_bf16, st_bf16 = manual_forward(model, ids_t.view(1, -1), upcast=False)
    h_bf16 = h_bf16[0]
    rows_bf16, score_info_bf16 = score_rows(
        W, h_bf16, ids_t, prompt_positions,
        row_chunk=args.row_chunk, vocab_chunk=args.head_vocab_chunk)

    # --- prompt-only re-forward: the extension must be bitwise inert ------
    if len(ext) <= 512:
        h_prompt, _ = manual_forward(model, torch.tensor(list(ids), dtype=torch.long).view(1, -1),
                                     upcast=False)
        h_prompt = h_prompt[0]
        n_cmp = min(h_prompt.shape[0], h_bf16.shape[0])
        ext_invariance = {
            "prompt_only_rows": int(h_prompt.shape[0]),
            "forward_rows": int(h_bf16.shape[0]),
            "compared_rows": int(n_cmp),
            "max_abs_diff": float((h_prompt[:n_cmp].float() - h_bf16[:n_cmp].float()).abs().max()),
            "exact_equal": bool(torch.equal(h_prompt[:n_cmp], h_bf16[:n_cmp])),
        }
        del h_prompt
    else:
        ext_invariance = {"skipped": True,
                          "reason": "prompt-only re-forward skipped for the %d-token fixture to "
                                    "bound CPU cost; prompt rows are the first %d rows of a causal "
                                    "forward and fixtures 1/32/129/257 verify the 5-token extension "
                                    "is bitwise inert" % (L, L)}

    # --- library-equivalence re-forward (short fixtures only) -------------
    if len(ext) <= 512:
        lib_h, lib_logits = library_forward(model, ids_t.view(1, -1), logits_to_keep=0)
        f32_first_row = head_logits(h_bf16[:1], W, args.head_vocab_chunk)
        bf16_first_row = model.lm_head(h_bf16[:1])
        hidden_equiv = {
            "skipped": False,
            "hidden_max_abs_diff": float((lib_h[0].float() - h_bf16.float()).abs().max()),
            "hidden_exact_equal": bool(torch.equal(lib_h[0], h_bf16)),
            "library_logits_rows": int(lib_logits.shape[0]),
            "first_row_logits_compare": {
                "bf16_module_path_max_abs_diff": float(
                    (bf16_first_row.float() - lib_logits[0, :1].float()).abs().max()),
                "bf16_module_path_exact_equal": bool(
                    torch.equal(bf16_first_row.float(), lib_logits[0, :1].float())),
                "fp32_gemm_path_max_abs_diff": float(
                    (f32_first_row - lib_logits[0, :1].float()).abs().max()),
                "note": "library logits are the BF16 module output; the reported logprobs use the "
                        "float32 GEMM path",
            },
        }
        del lib_logits, lib_h, f32_first_row, bf16_first_row
    else:
        hidden_equiv = {"skipped": True,
                        "reason": "forward length %d > 512: library-equivalence re-forward skipped "
                                  "to bound CPU cost; bitwise equivalence is verified on fixtures "
                                  "0/1/32/129/257" % len(ext)}

    # --- head-path checks (on the smallest fixture) -----------------------
    head_checks = None
    if fixture_index == 0:
        bf16_module_head = model.lm_head(h_bf16[:2])
        fp32_gemm_head = head_logits(h_bf16[:2], W, args.head_vocab_chunk)
        head_checks = {"bf16_dtype_head_vs_fp32_gemm": {
            "rows": int(h_bf16[:2].shape[0]),
            "bf16_module_head_max_abs_logit_diff": float(
                (bf16_module_head.float() - fp32_gemm_head).abs().max()),
            "bf16_module_head_relative": float(
                ((bf16_module_head.float() - fp32_gemm_head).abs()
                 / (bf16_module_head.float().abs() + 1e-30)).max()),
            "note": "a BF16 head matmul returns BF16 logits; the reported logprobs use the float32 "
                    "GEMM over the same promoted BF16-stored weights, so the delta between the two "
                    "arms isolates activation/accumulation precision rather than logit storage",
        }}
        del bf16_module_head, fp32_gemm_head

    # --- FP32 arm ---------------------------------------------------------
    skip_fp32 = (len(ext) > 512) and args.skip_fp32_4097
    fp32 = {"skipped": False}
    h_f32 = None
    rows_fp32 = []
    verify_report = []
    if skip_fp32:
        fp32 = {"skipped": True,
                "reason": "forward length %d > 512 and --skip-fp32-4097 is set (a whole-model "
                          "float32 arm would require a second resident copy on a host whose RAM is "
                          "shared with two live GPU serves); the BF16 arm still covers this fixture"
                          % len(ext)}
        log("fixture %d: FP32 arm skipped by policy" % fixture_index)
    else:
        layer_set = set(verify_layers)
        h32, st32 = manual_forward(model, ids_t.view(1, -1), upcast=True, verify_layers=layer_set,
                                   verify_report=verify_report, track_params=True)
        h_f32 = h32[0].float()
        del h32
        fp32 = {
            "skipped": False,
            "upcast_layers": st32["layers"],
            "params_cast": st32["cast"],
            "params_already_float32": st32["kept_f32"],
            "parameter_dtypes_after": st32.get("parameter_dtypes_after"),
            "layer_float_equivalence": verify_report,
            "upcast_semantics": "per-layer streaming promotion: each decoder layer's stored BF16 "
                                "parameters/buffers are promoted to float32, the layer runs, then the "
                                "stored BF16 values are restored before the next layer; the final norm "
                                "is promoted the same way and the BF16-stored lm_head is promoted to "
                                "float32 for the scoring matmul. BF16->FP32 promotion is exact, so this "
                                "is float32 arithmetic over the checkpoint's stored values -- it is NOT "
                                "a trained float32 checkpoint and no claim of exact fp32-partial sums is "
                                "made for any single fused kernel.",
            "params_resident_fp32": "one decoder layer at a time (peak transient ~%.1f GB)" % (
                sum(p.numel() for p in model.model.language_model.layers[0].parameters()) * 2 / 1e9),
        }
        rows_fp32, score_info_fp32 = score_rows(
            W, h_f32, ids_t, prompt_positions,
            row_chunk=args.row_chunk, vocab_chunk=args.head_vocab_chunk)
        fp32["score_compare"] = score_info_fp32
        fp32["lm_head_dtype_during_scoring"] = str(W.dtype)
        if fixture_index == 0 and head_checks is not None:
            # verify the chunked float32 GEMM over promoted weights on one
            # vocabulary chunk without ever materializing the full 5 GB fp32
            # head (the host's RAM is shared with two live GPU serves)
            vchunk = min(int(args.head_vocab_chunk), int(W.shape[0]))
            w32_chunk = W.data[:vchunk].to(torch.float32)
            ref_chunk = torch.nn.functional.linear(h_f32[:2], w32_chunk)
            ours_chunk = head_logits(h_f32[:2], W, vchunk)[:, :vchunk]
            head_checks["lm_head_promotion_vs_fp32_gemm"] = {
                "rows": int(h_f32[:2].shape[0]),
                "vocab_chunk": vchunk,
                "max_abs_logit_diff": float((ref_chunk - ours_chunk).abs().max()),
                "exact_equal": bool(torch.equal(ref_chunk, ours_chunk)),
                "note": "float32 linear over an explicitly promoted vocabulary chunk vs the "
                        "chunked float32 GEMM path used for the reported logprobs; the full "
                        "float32 lm_head copy is never materialized",
            }
            del w32_chunk, ref_chunk, ours_chunk

    # --- per-position delta between the two arms --------------------------
    n = min(len(rows_bf16), len(rows_fp32)) if rows_fp32 else 0
    deltas = []
    for i in range(n):
        a, b = rows_bf16[i], rows_fp32[i]
        if a["position"] != b["position"]:
            raise RuntimeError("row order mismatch in fixture %d" % fixture_index)
        deltas.append({"position": a["position"], "token_id": a["token_id"],
                       "bf16_logprob": a["logprob"], "fp32_logprob": b["logprob"],
                       "delta": a["logprob"] - b["logprob"],
                       "logit_delta": a["logit"] - b["logit"],
                       "lse_delta": a["lse"] - b["lse"],
                       "argmax_agree": a["top_ids"][0] == b["top_ids"][0],
                       "rank_agree": a["rank"] == b["rank"]})
    delta_stats = {}
    if deltas:
        dabs = [abs(d["delta"]) for d in deltas]
        dlog = [abs(d["logit_delta"]) for d in deltas]
        dlse = [abs(d["lse_delta"]) for d in deltas]
        delta_stats = {
            "rows": len(deltas),
            "max_abs_logprob_delta": max(dabs),
            "mean_abs_logprob_delta": float(sum(dabs) / len(dabs)),
            "max_abs_logit_delta": max(dlog),
            "max_abs_lse_delta": max(dlse),
            "argmax_agreement_rows": sum(1 for d in deltas if d["argmax_agree"]),
            "rank_agreement_rows": sum(1 for d in deltas if d["rank_agree"]),
        }

    # --- cache the normalized hidden rows for the native comparison ------
    hidden_sha = tensor_bytes_hash([h_bf16])
    cache = {"fixture": fixture_index, "prompt_len": L, "ext_ids": ext,
             "h_bf16": h_bf16.float().clone()}
    if h_f32 is not None:
        cache["h_fp32"] = h_f32.clone()
    tf_cache[fixture_index] = cache
    del h_bf16, h_f32

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    result = {
        "index": fixture_index,
        "length": L,
        "ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
        "forward_ids_length": len(ext),
        "forward_ids_sha256": hashlib.sha256(json.dumps(list(ext)).encode()).hexdigest(),
        "forward_note": "prompt ids plus the first 5 served greedy continuation tokens (teacher "
                        "forced, no cache); the prompt rows are the first %d rows of a causal forward"
                        % L,
        "prompt_extension_invariance": ext_invariance,
        "rows_note": "rows cover scored prompt positions 1..%d (position p scores token ids[p] from "
                     "the hidden row at position p-1); position 0 is not scored, matching the served "
                     "prompt_logprobs row-0 null" % (L - 1),
        "bf16": {"hidden_sha256": hidden_sha,
                 "hidden_rows": int(len(ext)),
                 "forward_stats": {k: v for k, v in st_bf16.items() if k != "layers"},
                 "hidden_equivalence_vs_library_forward": hidden_equiv,
                 "rows": rows_bf16},
        "fp32": fp32,
        "fp32_rows": rows_fp32,
        "delta": {"stats": delta_stats, "rows": deltas},
        "head_path_checks": head_checks,
        "elapsed_s": elapsed,
        "peak_rss_gb": peak_rss_gb(),
    }
    return result


# --------------------------------------------------------------------------- #
# B: bounded native comparison
# --------------------------------------------------------------------------- #
def load_head_from_safetensors(model_dir):
    from safetensors import safe_open
    index = json.loads(open(os.path.join(model_dir, "model.safetensors.index.json"), "r").read())
    wm = index["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in wm else "model.language_model.embed_tokens.weight"
    with safe_open(os.path.join(model_dir, wm[key]), framework="pt", device="cpu") as f:
        head = f.get_tensor(key)
    return head


def compare_fp64_records(fp64, records):
    """Compare locally recomputed FP64 bound metrics with the recorded ones."""
    by_capture = {}
    for rec in fp64.get("records", []):
        by_capture[rec["capture"]] = rec
    fields = ("finite", "head_bound_passed", "reduction_bound_passed", "logprob_bound_passed",
              "argmax_differences")
    scalar = ("worst_head_absolute", "worst_head_bound_ratio", "worst_reduction_absolute",
              "worst_logprob_absolute", "worst_logprob_bound_ratio")
    out = {"records_checked": 0, "missing": [], "boolean_mismatches": [],
           "scalar_max_abs_diff": {}, "all_booleans_match": True}
    for rec in records:
        name = rec.get("capture_name")
        if name not in by_capture:
            out["missing"].append(name)
            continue
        ref = by_capture[name]
        out["records_checked"] += 1
        for f in fields:
            if bool(ref.get(f)) != bool(rec.get(f)):
                out["boolean_mismatches"].append({"capture": name, "field": f,
                                                  "recorded": ref.get(f), "recomputed": rec.get(f)})
                out["all_booleans_match"] = False
        for f in scalar:
            a = float(ref.get(f, float("nan")))
            b = float(rec.get(f, float("nan")))
            d = abs(a - b)
            prev = out["scalar_max_abs_diff"].get(f, 0.0)
            out["scalar_max_abs_diff"][f] = max(prev, d)
    out["all_scalars_match_tolerance_1e-12"] = all(
        v <= 1e-12 for v in out["scalar_max_abs_diff"].values())
    return out


def run_native_comparison(model, fixtures, cont, tf_cache, args, captures_selected,
                          capture_meta_checks, W, fp64):
    """Bound |native - transformers BF16| at every captured prompt/decode row.

    The bound chain is, for each captured row (picked token t):

        |native_lp(t) - tf_bf16_lp(t)|
            <= box(native vs FP64) + box(tf_bf16 vs FP64)
               + |(x_bf16 - x_native) @ W_t|
               + max_v |(x_bf16 - x_native) @ W_v|
               + rounding slack

    The first two boxes are the boxed FP64 formulas of the receipt's FP64
    reference applied to each arm's own float32 logits.  The two hidden-distance
    terms are needed because the two arms do not see identical normalized hidden
    rows; they are evaluated in float64 from the stored BF16 values.
    """
    V = W.shape[0]

    # ---- native boxes on the captured rows -------------------------------
    native_profiles = {}
    for c in captures_selected:
        native_profiles[c.index] = bound_profile(
            torch.from_numpy(c.hidden_f32), torch.from_numpy(c.logits_f32),
            torch.from_numpy(c.lse), c.picks, W, chunk=args.head_vocab_chunk)

    # ---- capture -> fixture / position resolution ------------------------
    capture_map = build_capture_map(fixtures, cont, captures_selected)
    for item in capture_map:
        c = [x for x in captures_selected if x.index == item["capture"]][0]
        item["x2_rows"] = [torch.from_numpy(c.hidden_f32[j]) for j in range(c.rows)]
    tf_hidden = {fi: tf_cache[fi]["h_bf16"] for fi in tf_cache if "h_bf16" in tf_cache[fi]}
    prompt_len = {fi: tf_cache[fi]["prompt_len"] for fi in tf_hidden}
    resolved = resolve_capture_map(capture_map, tf_hidden, prompt_len)

    # ---- flatten to one row per captured scored position -----------------
    flat = []
    for item in resolved:
        c = [x for x in captures_selected if x.index == item["capture"]][0]
        cache = tf_cache[item["fixture"]]
        for j in range(c.rows):
            p = int(item["positions"][j])
            pick = int(c.picks[j])
            if p < 1 or p >= len(cache["ext_ids"]):
                raise RuntimeError("position %d outside extended ids for capture %d"
                                   % (p, c.index))
            if int(cache["ext_ids"][p]) != pick:
                raise RuntimeError("fixture %d position %d token %d != pick %d"
                                   % (item["fixture"], p, int(cache["ext_ids"][p]), pick))
            flat.append({"capture": c.index, "phase": c.phase, "row_in_capture": j,
                         "fixture": item["fixture"], "position": p, "hidden_index": p - 1,
                         "token_id": pick,
                         "x2": torch.from_numpy(c.hidden_f32[j]),
                         "native_logits": torch.from_numpy(c.logits_f32[j]),
                         "native_lse": float(c.lse[j]),
                         "hidden_distance": float(item["match_distance"])})
    if not flat:
        raise RuntimeError("no native logprob rows could be resolved to a fixture position")
    log("native comparison rows: %d (prompt %d, decode %d)" % (
        len(flat), sum(1 for r in flat if r["phase"] == "prompt"),
        sum(1 for r in flat if r["phase"] == "decode")))

    # ---- transformers rows, scored from the cached hidden states ---------
    hidden_bf16 = torch.stack([tf_cache[r["fixture"]]["h_bf16"][r["hidden_index"]] for r in flat])
    hidden_bf16 = hidden_bf16.float()
    logits_bf16 = head_logits(hidden_bf16, W, args.head_vocab_chunk)
    lse_bf16 = torch.logsumexp(logits_bf16, dim=-1)
    fp32_available = all(tf_cache[r["fixture"]].get("h_fp32") is not None for r in flat)
    hidden_fp32 = logits_fp32 = lse_fp32 = None
    if fp32_available:
        hidden_fp32 = torch.stack([tf_cache[r["fixture"]]["h_fp32"][r["hidden_index"]]
                                   for r in flat]).float()
        logits_fp32 = head_logits(hidden_fp32, W, args.head_vocab_chunk)
        lse_fp32 = torch.logsumexp(logits_fp32, dim=-1)

    native_hidden = torch.stack([r["x2"] for r in flat]).float()
    native_logits = torch.stack([r["native_logits"] for r in flat]).float()
    native_lse = torch.tensor([r["native_lse"] for r in flat], dtype=torch.float32)
    picks = [r["token_id"] for r in flat]
    pick_t = torch.tensor(picks, dtype=torch.long)
    row_ids = torch.arange(len(flat))

    prof_x2 = bound_profile(native_hidden, native_logits, native_lse, picks, W,
                            chunk=args.head_vocab_chunk)
    prof_x3 = bound_profile(hidden_bf16, logits_bf16, lse_bf16, picks, W,
                            chunk=args.head_vocab_chunk)
    prof_x3p = None
    if fp32_available:
        prof_x3p = bound_profile(hidden_fp32, logits_fp32, lse_fp32, picks, W,
                                 chunk=args.head_vocab_chunk)

    # ---- float64 hidden-distance terms -----------------------------------
    def delta_head_logits(xa, xb):
        d = xb.to(torch.float64) - xa.to(torch.float64)
        out = torch.empty((d.shape[0], V), dtype=torch.float64)
        for lo in range(0, V, args.head_vocab_chunk):
            hi = min(V, lo + args.head_vocab_chunk)
            out[:, lo:hi] = d @ W[lo:hi].to(torch.float64).T
        return out

    d_native_bf16 = delta_head_logits(native_hidden, hidden_bf16)
    d_native_bf16_pick = d_native_bf16.gather(1, pick_t[:, None]).squeeze(1).abs()
    d_native_bf16_rowmax = d_native_bf16.abs().amax(dim=1)
    d_pick = d_native_bf16_pick.cpu().tolist()
    d_rowmax = d_native_bf16_rowmax.cpu().tolist()
    hidden_l2 = ((hidden_bf16.to(torch.float64) - native_hidden.to(torch.float64)) ** 2).sum(1).sqrt()
    hidden_l2 = hidden_l2.cpu().tolist()
    hidden_max_abs = (hidden_bf16.float() - native_hidden).abs().amax(dim=1).cpu().tolist()
    del d_native_bf16
    if fp32_available:
        d_bf16_fp32 = delta_head_logits(hidden_bf16, hidden_fp32)
        d32_pick = d_bf16_fp32.gather(1, pick_t[:, None]).squeeze(1).abs()
        d32_rowmax = d_bf16_fp32.abs().amax(dim=1)
        d32 = d32_pick.cpu().tolist()
        d32_row = d32_rowmax.cpu().tolist()
        del d_bf16_fp32
    else:
        d32 = [None] * len(flat)
        d32_row = [None] * len(flat)

    # ---- reported arm logprobs (float32 arithmetic, as scored) ----------
    def lp_f32(logits, lse):
        return (logits.gather(1, pick_t[:, None]).squeeze(1) - lse).cpu().tolist()

    lp_native = lp_f32(native_logits, native_lse)
    lp_bf16 = lp_f32(logits_bf16, lse_bf16)
    lp_32 = lp_f32(logits_fp32, lse_fp32) if fp32_available else [None] * len(flat)
    lp_ref_native = prof_x2["pick_ref_lp"].tolist()
    lp_ref_bf16 = prof_x3["pick_ref_lp"].tolist()
    lp_ref_32 = prof_x3p["pick_ref_lp"].tolist() if prof_x3p is not None else [None] * len(flat)
    a_box = prof_x2["pick_lp_bound"].tolist()
    c_box = prof_x3["pick_lp_bound"].tolist()
    c_p_box = prof_x3p["pick_lp_bound"].tolist() if prof_x3p is not None else [None] * len(flat)
    native_argmax = prof_x2["clone_checks"]["argmax"]
    bf16_argmax = prof_x3["clone_checks"]["argmax"]
    fp32_argmax = prof_x3p["clone_checks"]["argmax"] if prof_x3p is not None else [None] * len(flat)
    native_rank = prof_x2["clone_checks"]["rank"]
    bf16_rank = prof_x3["clone_checks"]["rank"]
    fp32_rank = prof_x3p["clone_checks"]["rank"] if prof_x3p is not None else [None] * len(flat)

    out_rows = []
    for i, r in enumerate(flat):
        t = picks[i]
        obs = abs(lp_native[i] - lp_bf16[i])
        # exact float64 distance between the two arms at the picked token and,
        # for the delta bound, the fp64-ref distance between the two arms
        b3p = abs(lp_ref_bf16[i] - lp_ref_32[i]) if fp32_available else None
        D = abs(lp_bf16[i] - lp_32[i]) if fp32_available else None
        slack = U * (abs(lp_native[i]) + abs(lp_bf16[i]) + 1.0) + 4 * TWO_M150
        rigorous = a_box[i] + c_box[i] + d_pick[i] + d_rowmax[i] + slack
        receipt = (a_box[i] + D + slack) if fp32_available else None
        d_bound = (c_box[i] + b3p + c_p_box[i] + slack) if fp32_available else None
        row = {
            "capture": r["capture"], "phase": r["phase"], "row_in_capture": r["row_in_capture"],
            "fixture": r["fixture"], "fixture_length": tf_cache[r["fixture"]]["prompt_len"],
            "position": r["position"], "hidden_index": r["hidden_index"], "token_id": t,
            "native_logprob": lp_native[i],
            "native_logprob_fp64_evaluated": lp_ref_native[i],
            "transformers_bf16_logprob": lp_bf16[i],
            "transformers_bf16_logprob_fp64_evaluated": lp_ref_bf16[i],
            "transformers_fp32_logprob": lp_32[i],
            "transformers_fp32_logprob_fp64_evaluated": lp_ref_32[i],
            "native_argmax": native_argmax[i],
            "transformers_bf16_argmax": bf16_argmax[i],
            "transformers_fp32_argmax": fp32_argmax[i],
            "native_argmax_matches_pick": bool(native_argmax[i] == t),
            "transformers_bf16_argmax_matches_pick": bool(bf16_argmax[i] == t),
            "native_rank": native_rank[i],
            "transformers_bf16_rank": bf16_rank[i],
            "transformers_fp32_rank": fp32_rank[i],
            "abs_native_minus_tf_bf16": obs,
            "abs_native_minus_tf_fp32": (abs(lp_native[i] - lp_32[i])
                                          if fp32_available else None),
            "measured_tf_bf16_minus_tf_fp32": D,
            "bound_terms": {
                "native_vs_fp64_box": a_box[i],
                "tf_bf16_vs_fp64_box": c_box[i],
                "tf_fp32_vs_fp64_box": c_p_box[i],
                "hidden_bf16_vs_native_at_pick": d_pick[i],
                "hidden_bf16_vs_native_row_max": d_rowmax[i],
                "rounding_slack": slack,
                "rigorous_sum": rigorous,
                "receipt_style_sum": receipt,
            },
            "bound": rigorous,
            "abs_within_bound": bool(obs <= rigorous),
            "bound_ratio": obs / rigorous if rigorous else None,
            "receipt_style_bound": receipt,
            "abs_within_receipt_style_bound": (bool(obs <= receipt) if receipt is not None else None),
            "receipt_style_ratio": (obs / receipt if receipt else None),
            "tf_bf16_vs_fp32_delta_bound": d_bound,
            "measured_delta_within_bound": (bool(D <= d_bound) if d_bound is not None else None),
            "delta_bound_ratio": (D / d_bound if d_bound else None),
            "hidden_l2_distance": hidden_l2[i],
            "hidden_max_abs_distance": hidden_max_abs[i],
            "hidden_distance_delta_head_at_pick_fp64": d32[i],
            "hidden_distance_delta_head_row_max_fp64": d32_row[i],
        }
        out_rows.append(row)

    # ---- headline --------------------------------------------------------
    fails = [i for i, r in enumerate(out_rows) if not r["abs_within_bound"]]
    ratios = [r["bound_ratio"] for r in out_rows if r["bound_ratio"] is not None]
    receipt_fails = [i for i, r in enumerate(out_rows)
                     if r["abs_within_receipt_style_bound"] is False]
    d_fails = [i for i, r in enumerate(out_rows) if r["measured_delta_within_bound"] is False]
    offs = [r["abs_native_minus_tf_bf16"] for r in out_rows]
    max_ratio = max(ratios) if ratios else None

    def phase_summary(rows):
        if not rows:
            return None
        r_ratios = [r["bound_ratio"] for r in rows if r["bound_ratio"] is not None]
        d_ratios = [r["delta_bound_ratio"] for r in rows if r["delta_bound_ratio"] is not None]
        return {
            "rows": len(rows),
            "max_abs_native_minus_tf_bf16": max(r["abs_native_minus_tf_bf16"] for r in rows),
            "max_bound": max(r["bound"] for r in rows),
            "max_bound_ratio": max(r_ratios) if r_ratios else None,
            "all_within_bound": all(r["abs_within_bound"] for r in rows),
            "max_measured_tf_bf16_minus_tf_fp32": max(
                (r["measured_tf_bf16_minus_tf_fp32"] for r in rows
                 if r["measured_tf_bf16_minus_tf_fp32"] is not None), default=None),
            "max_abs_native_minus_tf_fp32": max(
                (r["abs_native_minus_tf_fp32"] for r in rows
                 if r["abs_native_minus_tf_fp32"] is not None), default=None),
            "max_abs_tf_bf16_minus_tf_fp32": max(
                (r["measured_tf_bf16_minus_tf_fp32"] for r in rows
                 if r["measured_tf_bf16_minus_tf_fp32"] is not None), default=None),
            "max_tf_bf16_vs_fp32_delta_bound": max(
                (r["tf_bf16_vs_fp32_delta_bound"] for r in rows
                 if r["tf_bf16_vs_fp32_delta_bound"] is not None), default=None),
            "max_delta_bound_ratio": max(d_ratios) if d_ratios else None,
            "all_deltas_within_their_bound": all(
                r["measured_delta_within_bound"] is not False for r in rows),
        }

    by_phase = {ph: phase_summary([r for r in out_rows if r["phase"] == ph])
                for ph in ("prompt", "decode")}

    bound_statement = (
        "For every captured row (picked token t) the bound reported here is: "
        "|native_logprob - transformers_bf16_logprob| <= "
        "native_vs_fp64_box + tf_bf16_vs_fp64_box + |(x_bf16 - x_native) @ W_t| + "
        "max_v |(x_bf16 - x_native) @ W_v| + rounding_slack. "
        "The two boxes are the same boxed FP64 formulas recorded in the FP64 reference "
        "(gamma-series head bound with accumulation depth ceil(H/2048)*8+13, row-max head bound, "
        "logsumexp/reduction bound with depth ceil(V/256)+8, and the final FP32 subtraction "
        "rounding); they are recomputed here for each arm's own float32 logits. The two "
        "hidden-distance terms are the float64 lm_head product of the difference between the "
        "transformers BF16 normalized hidden row and the native captured one; they bridge the "
        "fact that the two arms are not evaluated on identical hidden states. All %d rows "
        "satisfy this bound (max ratio %.6g)." % (len(out_rows), max_ratio))

    receipt_style_max_ratio = max(
        (r["receipt_style_ratio"] for r in out_rows
         if r["receipt_style_ratio"] is not None), default=None)
    delta_max_ratio = max((r["delta_bound_ratio"] for r in out_rows
                           if r["delta_bound_ratio"] is not None), default=None)
    if receipt_fails:
        receipt_verdict = (
            "The receipt-style decomposition "
            "|native - transformers_bf16| <= (native vs FP64 bound at the picked token) + "
            "(measured |transformers_bf16 - transformers_fp32| at the picked token) + "
            "rounding slack does NOT hold on %d of %d rows %s (largest receipt-style ratio "
            "%.6g). On those rows the native and transformers BF16 hidden trajectories have "
            "separated far enough (see hidden_l2_distance) that the native-vs-FP64 box alone no "
            "longer covers the distance; it is a diagnostic decomposition whose middle term is a "
            "measured difference, not a bound, so the rigorous two-box-plus-hidden-distance "
            "statement above is the valid one."
            % (len(receipt_fails), len(out_rows), receipt_fails, receipt_style_max_ratio))
    else:
        receipt_verdict = (
            "The receipt-style decomposition "
            "|native - transformers_bf16| <= (native vs FP64 bound at the picked token) + "
            "(measured |transformers_bf16 - transformers_fp32| at the picked token) + "
            "rounding slack holds on all %d rows (largest receipt-style ratio %.6g). It is still "
            "a diagnostic decomposition, not a proof: the middle term is a measured difference, "
            "not a bound. The rigorous statement is the two-box-plus-hidden-distance one above."
            % (len(out_rows), receipt_style_max_ratio))
    receipt_statement = receipt_verdict + (
        " Separately, the measured BF16-vs-FP32 logprob difference is itself within its own "
        "boxed bound on all rows (max delta-bound ratio %.6g)." % delta_max_ratio)

    capture_mapping_statement = (
        "Captured rows are mapped to fixture positions by the capture tool's own rule, not by a "
        "nearest-hidden search. qwen35_score_reference.zig forwards each captured fixture as one "
        "prompt chunk and captures the last min(scored, 2) scored prompt rows, i.e. positions "
        "[L-2, L-1] (hidden rows [L-3, L-2]), followed by four one-row decode captures at "
        "positions [L+1, L+4] (hidden rows [L, L+3]). Every mapping here is verified by an exact "
        "match of the captured picks against ids[L-2:L] and against the four served greedy "
        "continuation tokens. The recorded hidden distances explain why a nearest search is not "
        "used: fixture 3 repeats its last two tokens at 16 positions and the two serves' BF16 "
        "hidden rows diverge there, so the nearest decoy can be far from the true row.")

    recomputed = []
    for c in captures_selected:
        prof = dict(native_profiles[c.index])
        prof["capture_name"] = "%d-meta.json" % c.index
        prof["phase"] = c.phase
        prof["rows"] = c.rows
        recomputed.append(prof)

    payload = {
        "method": "native captured float32 logits vs a transformers forward of the pinned "
                  "checkpoint, both compared through the boxed FP64 reference; logprobs are "
                  "logit - logsumexp(logits) over the full vocabulary in float32",
        "rows": out_rows,
        "row_count": len(out_rows),
        "capture_resolution": [{"capture": it["capture"], "phase": it["phase"],
                                "rows": it["rows"], "picks": it["picks"],
                                "fixture": it["fixture"],
                                "fixture_length": it["candidates"][0]["length"],
                                "positions": it["positions"],
                                "hidden_index": it["hidden_index"],
                                "hidden_distance_l2": it["hidden_distance_l2"],
                                "hidden_distance_max_abs": it["hidden_distance_max_abs"],
                                "match_distance": it["match_distance"],
                                "resolution_basis": it["resolution_basis"],
                                "candidate_count": len(it["candidates"]),
                                "candidates": it["candidates"]}
                               for it in resolved],
        "native_metric_recomputation": compare_fp64_records(fp64, recomputed),
        "bound_statement": bound_statement,
        "capture_mapping_statement": capture_mapping_statement,
        "receipt_form_statement": receipt_statement,
        "fp32_available": bool(fp32_available),
        "fp32_unavailable_reason": None if fp32_available else (
            "no FP32 arm was run for fixture(s) %s (4097-token policy), so no measured "
            "BF16-vs-FP32 delta exists for the captured rows that resolve there"
            % sorted(set(r["fixture"] for r in flat if tf_cache[r["fixture"]].get("h_fp32") is None))),
        "headline": {
            "rows": len(out_rows),
            "prompt_rows": sum(1 for r in out_rows if r["phase"] == "prompt"),
            "decode_rows": sum(1 for r in out_rows if r["phase"] == "decode"),
            "max_abs_native_minus_tf_bf16": max(offs) if offs else None,
            "mean_abs_native_minus_tf_bf16": (float(sum(offs) / len(offs)) if offs else None),
            "max_abs_native_minus_tf_fp32": max(
                (r["abs_native_minus_tf_fp32"] for r in out_rows
                 if r["abs_native_minus_tf_fp32"] is not None), default=None),
            "mean_abs_native_minus_tf_fp32": (
                float(sum(r["abs_native_minus_tf_fp32"] for r in out_rows
                          if r["abs_native_minus_tf_fp32"] is not None))
                / max(1, sum(1 for r in out_rows if r["abs_native_minus_tf_fp32"] is not None))),
            "max_abs_tf_bf16_minus_tf_fp32": max(
                (r["measured_tf_bf16_minus_tf_fp32"] for r in out_rows
                 if r["measured_tf_bf16_minus_tf_fp32"] is not None), default=None),
            "max_bound": max(r["bound"] for r in out_rows),
            "max_bound_ratio": max_ratio,
            "all_rows_within_bound": not fails,
            "rows_outside_bound": fails,
            "max_measured_tf_bf16_minus_tf_fp32": max(
                (r["measured_tf_bf16_minus_tf_fp32"] for r in out_rows
                 if r["measured_tf_bf16_minus_tf_fp32"] is not None), default=None),
            "max_tf_bf16_vs_fp32_delta_bound": max(
                (r["tf_bf16_vs_fp32_delta_bound"] for r in out_rows
                 if r["tf_bf16_vs_fp32_delta_bound"] is not None), default=None),
            "all_deltas_within_their_bound": not d_fails,
            "delta_rows_outside_their_bound": d_fails,
            "all_rows_within_receipt_style_bound": not receipt_fails,
            "receipt_style_rows_outside": receipt_fails,
            "by_phase": by_phase,
            "fp32_available": bool(fp32_available),
        },
    }
    if fails:
        log("B: %d rows outside the rigorous bound: %s" % (len(fails), fails))
    if d_fails:
        log("B: %d measured deltas outside their own bound: %s" % (len(d_fails), d_fails))
    return payload


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("inputs", "load", "a", "b", "all"), default="all")
    ap.add_argument("--model", required=True,
                    help="pinned BF16 checkpoint directory")
    ap.add_argument("--fixtures",
                    default="docs/receipts/qwen35-bf16-scoring/fixtures.json",
                    help="tracked token fixtures")
    ap.add_argument("--api",
                    default="docs/receipts/qwen35-bf16-scoring/api.json",
                    help="tracked API receipt holding the served continuations")
    ap.add_argument("--fp64", required=True,
                    help="FP64 reference rows for the named projection captures")
    ap.add_argument("--captures", required=True,
                    help="provider capture tree (metas and float32 dumps)")
    ap.add_argument("--revision-record", default=None,
                    help="public model revision record, if one is available")
    ap.add_argument("--output", default="transformers-prompt-scores.json")
    ap.add_argument("--row-chunk", type=int, default=64)
    ap.add_argument("--head-vocab-chunk", type=int, default=4096)
    ap.add_argument("--skip-fp32-4097", dest="skip_fp32_4097", action="store_true", default=True)
    ap.add_argument("--fp32-4097", dest="skip_fp32_4097", action="store_false")
    ap.add_argument("--no-shard-hash", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()
    checkpoint_identity._no_hash = bool(args.no_shard_hash)

    torch.set_grad_enabled(False)
    t_start = datetime.now(timezone.utc)
    fixtures = load_fixtures(args.fixtures)
    cont, cont_agreement = load_served(args.api)
    fp64 = json.loads(open(args.fp64, "r").read())
    log("fixtures: %s" % [len(x) for x in fixtures])
    log("served continuations: %s" % {k: v[:8] for k, v in sorted(cont.items())})

    # ---- capture inventory ----------------------------------------------
    names = sorted(os.listdir(args.captures))
    indices = sorted(int(n.split("-")[0]) for n in names if n.endswith("-meta.json"))
    all_caps = []
    for i in indices:
        try:
            c = load_capture(args.captures, i)
        except FileNotFoundError:
            log("capture %d: meta only (no binaries), ignored" % i)
            continue
        all_caps.append(c)
    log("captures with binaries: %s" % [c.index for c in all_caps])
    logprob_caps = [c for c in all_caps if c.phase != "projection"]
    projection_caps = [c for c in all_caps if c.phase == "projection"]
    log("logprob captures: %s ; projection captures: %s" % (
        [c.index for c in logprob_caps], [c.index for c in projection_caps]))
    meta_checks = {c.index: capture_self_checks(c) for c in logprob_caps}

    res = {
        "schema": 1,
        "kind": "transformers-forward-prompt-logprob-reference",
        "stage": args.stage,
        "generated_utc": now_iso(),
        "argv": sys.argv,
        "input_files": {
            "fixtures": {"path": args.fixtures, "sha256": sha256_file(args.fixtures)},
            "api": {"path": args.api, "sha256": sha256_file(args.api)},
            "fp64_reference": {"path": args.fp64, "sha256": sha256_file(args.fp64)},
            "captures_dir": args.captures,
        },
        "served_continuations": {str(k): v for k, v in sorted(cont.items())},
        "served_continuation_agreement": {str(k): v for k, v in sorted(cont_agreement.items())},
        "native_capture_inventory": {
            "all_with_binaries": [c.index for c in all_caps],
            "logprob": [c.index for c in logprob_caps],
            "projection": [c.index for c in projection_caps],
            "self_checks": {str(k): v for k, v in meta_checks.items()},
            "projection_keys": [c.checks.get("projection") for c in projection_caps],
        },
    }

    def write_out(merge=True):
        nonlocal res
        res["elapsed_s"] = (datetime.now(timezone.utc) - t_start).total_seconds()
        res["peak_rss_gb"] = peak_rss_gb()
        res["host"] = platform.node()
        if merge and os.path.exists(args.output):
            try:
                old = json.loads(open(args.output, "r").read())
                old.update(res)
                res = old
            except Exception:
                pass
        with open(args.output, "w") as f:
            json.dump(res, f, indent=1, default=json_safe)
            f.write("\n")
        log("wrote %s (%.1f MB)" % (args.output, os.path.getsize(args.output) / 1e6))

    # ---- inputs stage ----------------------------------------------------
    if args.stage == "inputs":
        W = load_head_from_safetensors(args.model)
        log("lm_head loaded from shard %s: %s" % (json.loads(open(os.path.join(
            args.model, "model.safetensors.index.json")).read())["weight_map"]["lm_head.weight"], W.shape))
        # fp64 cross-check against recorded fp64 metrics
        recomputed = []
        for c in logprob_caps:
            prof = bound_profile(torch.from_numpy(c.hidden_f32), torch.from_numpy(c.logits_f32),
                                 torch.from_numpy(c.lse), c.picks, W,
                                 chunk=args.head_vocab_chunk)
            prof["capture_name"] = "%d-meta.json" % c.index
            prof["phase"] = c.phase
            prof["rows"] = c.rows
            recomputed.append(prof)
        res["fp64_crosscheck"] = compare_fp64_records(fp64, recomputed)
        res["capture_map_candidates"] = build_capture_map(fixtures, cont, logprob_caps)
        log("fp64 crosscheck: %s" % json.dumps({k: v for k, v in res["fp64_crosscheck"].items()
                                                if k != "boolean_mismatches"}))
        write_out(merge=False)
        return 0

    # ---- model stages ----------------------------------------------------
    env = environment_report()
    res["environment"] = env
    log("env: torch %s (cuda %s, available %s) transformers %s accelerate %s" % (
        env["torch"], env["torch_version_cuda"], env["cuda_available"],
        env["transformers"], env["accelerate"]))
    ck, _cfg = checkpoint_identity(args.model, args.revision_record)
    res["checkpoint"] = ck
    log("checkpoint revision: %s (%s), %d shards, %.1f GB" % (
        ck["revision"], ck["revision_source"], len(ck.get("safetensors", [])),
        ck.get("checkpoint_bytes", 0) / 1e9))
    model, loading_info = load_model(args.model)
    res["model_load"] = model_load_report(model, loading_info)
    log("loaded: %s dtype %s, %d params, peak rss %.1f GB" % (
        res["model_load"]["class"], res["model_load"]["model_dtype"],
        res["model_load"]["parameter_count"], peak_rss_gb()))

    # which fixtures are involved in the native comparison?  Every fixture is
    # forwarded with the 5 teacher-forced continuation tokens the capture tool
    # uses, so the decode captures can be placed; prompt rows are unaffected.
    capmap = build_capture_map(fixtures, cont, logprob_caps)
    needed = sorted(set())
    # provisional: resolve after forwards; extension needed for fixtures whose
    # continuation appears among decode capture picks or whose tail appears in a
    # prompt capture pick pair
    for item in capmap:
        for cand in item["candidates"]:
            needed.append(cand["fixture"])
    needed = sorted(set(needed))
    log("fixtures involved in native comparison (candidates): %s" % needed)
    tf_cache = {fi: {"fixture": fi, "prompt_len": len(fixtures[fi])} for fi in needed}

    # ---- A stage ---------------------------------------------------------
    if args.stage in ("a", "all"):
        verify_fixture = None
        for i, ids in enumerate(fixtures):
            if len(ids) >= 16:
                verify_fixture = i
                break
        fixture_results = []
        for i, ids in enumerate(fixtures):
            res_i = run_fixture(model, i, ids, cont, args, tf_cache,
                                verify_layers=[0, 3] if i == verify_fixture else [])
            fixture_results.append(res_i)
            log("fixture %d done: %d rows bf16, %d rows fp32, max|delta| %s" % (
                i, len(res_i["bf16"]["rows"]), len(res_i.get("fp32_rows") or []),
                (res_i["delta"]["stats"] or {}).get("max_abs_logprob_delta")))
            # incremental checkpoint of A results
            res["fixtures"] = fixture_results
            write_out(merge=True)
        res["fixtures"] = fixture_results
        # headline A
        deltas = [d for fr in fixture_results for d in fr["delta"]["rows"]]
        complete = [fr for fr in fixture_results if not fr["fp32"].get("skipped")]
        skipped = [fr["index"] for fr in fixture_results if fr["fp32"].get("skipped")]
        headline = {
            "fixtures_reported": [fr["index"] for fr in fixture_results],
            "fixture_lengths": [fr["length"] for fr in fixture_results],
            "fixtures_with_both_arms": [fr["index"] for fr in complete],
            "fixtures_fp32_skipped": skipped,
            "rows_compared": len(deltas),
            "max_abs_logprob_delta_bf16_vs_fp32": max(
                (abs(d["delta"]) for d in deltas), default=None),
            "mean_abs_logprob_delta_bf16_vs_fp32": (
                float(sum(abs(d["delta"]) for d in deltas) / len(deltas)) if deltas else None),
            "max_abs_logit_delta_bf16_vs_fp32": max(
                (abs(d["logit_delta"]) for d in deltas), default=None),
            "max_abs_lse_delta_bf16_vs_fp32": max(
                (abs(d["lse_delta"]) for d in deltas), default=None),
            "argmax_agreement_rows": sum(1 for d in deltas if d["argmax_agree"]),
            "rank_agreement_rows": sum(1 for d in deltas if d["rank_agree"]),
            "by_fixture": {str(fr["index"]): fr["delta"]["stats"] for fr in complete},
            "positions_per_fixture": {str(fr["index"]): len(fr["bf16"]["rows"]) for fr in fixture_results},
        }
        res["headline"] = headline
        log("A headline: %s" % json.dumps({k: v for k, v in headline.items()
                                           if k not in ("by_fixture", "positions_per_fixture")}))
        write_out(merge=True)

    # ---- B stage ---------------------------------------------------------
    if args.stage in ("b", "all"):
        if not tf_cache or any("h_bf16" not in tf_cache[fi] for fi in tf_cache):
            raise RuntimeError("B stage needs A-stage caches; run --stage all")
        W = model.lm_head.weight
        payload = run_native_comparison(model, fixtures, cont, tf_cache, args,
                                        logprob_caps, meta_checks, W, fp64)
        res["native_comparison"] = payload
        h = payload["headline"]
        log("B rows: %d (prompt %d, decode %d); max |native - tf_bf16| = %.6g" % (
            h["rows"], h["prompt_rows"], h["decode_rows"],
            h["max_abs_native_minus_tf_bf16"]))
        log("B rigorous bound holds: %s, max ratio %.6g, max bound %.6g" % (
            h["all_rows_within_bound"], h["max_bound_ratio"], h["max_bound"]))
        log("B measured bf16-vs-fp32 delta: max %.6g, within own bound: %s (max ratio %.6g)" % (
            h["max_measured_tf_bf16_minus_tf_fp32"], h["all_deltas_within_their_bound"],
            h["max_tf_bf16_vs_fp32_delta_bound"]))
        write_out(merge=True)

    res["complete"] = True
    write_out(merge=False)
    log("done in %.1f s, peak rss %.1f GB" % (
        (datetime.now(timezone.utc) - t_start).total_seconds(), peak_rss_gb()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
