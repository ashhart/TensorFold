#!/usr/bin/env python3
"""Public Qwen BF16 reference: inputs."""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import platform
import resource
import sys
import torch

U = 2.0 ** -24
TINY = 2.0 ** -150
TWO_M150 = 2.0 ** -150
SINGLE_PROMPT_CHUNK = 4096
CAPTURE_DECODES = 4

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print("[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


def peak_rss_gb():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def json_safe(obj):
    "Best-effort JSON-safe copy (transformers' loading info can carry sets)"
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
    'Absolute forward-error growth factor for `depth` max/min reductions'
    return float(depth) * U / (1.0 - float(depth) * U)


def bf16_bits_to_f32(bits_u16):
    return ((bits_u16.astype(np.uint32) << 16).view(np.float32)).copy()


def u32_bits_to_f32(bits_u32):
    return np.ascontiguousarray(bits_u32).view(np.float32)


def r(arg, default=None):
    a = getattr(arg, "default", None)
    return a if a is not None else default


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
    'Served greedy continuations (token_ids) per fixture index, from the'
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
    'Check the decoded stats words against the captured fp32 logits'
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


def build_capture_map(fixtures, cont, captures):
    'Map every prompt/decode capture to (fixture, positions) by the capture'
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
    "Attach the transformers BF16 hidden rows' distance to every captured row"
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
