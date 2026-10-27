#!/usr/bin/env python3
"""Independent reference run for the Qwen3.8-Flash-Next EXL3 pack: golden token ids and logits from the
Python TensorFold CUDA engine (tensorfold 0.6.x, ``families/qwen4_exp/cuda``), which no oracle of ours wrote.

The Python engine runs an EXL3 pack on ONE GPU with every routed expert resident in VRAM (43.5 GiB here); a
16 GiB card cannot hold that and the engine has no expert offload on CUDA (``--ssd-experts`` is MLX only,
``--tp 2`` is refused for EXL3 packs). This harness changes WHERE the routed-expert trellis buffer lives and
nothing else: ``exl3.expert_table`` is re-executed from its own source with the one line that allocates
``big`` on the device replaced by page-locked host memory registered with CUDA (UVA: the kernels read the
same bytes over PCIe through the same device pointer). Every kernel, the maths and the sampling are the
reference engine's own. The capacity admission is bypassed (the weights do not "fit" by its rule), and the
engine is built directly (``decode.Engine``) without MTP, graphs or the HTTP server.

Usage (the test machine, one card):
  ~/bin/gpurun 1 env TORCH_EXTENSIONS_DIR=$HOME/tensorfold-c-ref/ext \
      /data/src/kt-venv/bin/python tools/exl3_ref_golden.py --tf ~/tf-final/src \
      --pack /data/models/Qwen3.8-Flash-Next-exl3-3.05bpw --out golden.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PROMPT_IDS = [760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13]
TEXTS = ["The capital of France is", "Once upon a time, in a small village by the sea"]


def host_mapped_factory(torch, keep):
    """A CUDA-visible uint8 tensor of n bytes in page-locked host memory (cudaHostRegister, portable|mapped)."""

    import numpy as np

    cudart = torch.cuda.cudart()

    def make(n: int):
        arr = np.empty((n,), dtype=np.uint8)
        err = cudart.cudaHostRegister(arr.ctypes.data, n, 3)     # cudaHostRegisterPortable | cudaHostRegisterMapped
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({n}) failed: {err}")
        keep.append(arr)

        class _Iface:
            __cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (arr.ctypes.data, False),
                                        "version": 2, "strides": None}

        t = torch.as_tensor(_Iface(), device="cuda")
        if t.data_ptr() != arr.ctypes.data:
            raise RuntimeError("device pointer of the registered host buffer differs from its host pointer (no UVA)")
        return t

    return make


def patch_expert_table(exl3_mod, make_host):
    src = inspect.getsource(exl3_mod.expert_table)
    old = "big = torch.empty((total,), dtype=torch.uint8, device=device)"
    if src.count(old) != 1:
        raise RuntimeError("expert_table source changed; the one-line host-memory patch no longer applies")
    src = src.replace(old, "big = _HOST_MAPPED(total)")
    ns = exl3_mod.__dict__
    ns["_HOST_MAPPED"] = make_host
    exec(compile(src, exl3_mod.__file__ + " [host-mapped experts]", "exec"), ns)
    return src


def top(torch, logits_row, k=5):
    row = logits_row.float()
    lse = float(torch.logsumexp(row, dim=-1))
    v, i = torch.topk(row, k)
    return [int(x) for x in i.cpu()], [float(x) for x in v.cpu()], lse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True, help="tf-final/src (the reference tensorfold package, read only)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--layers-only", action="store_true", help="load and stop (smoke test)")
    ap.add_argument("--build-only", action="store_true",
                    help="JIT-build the reference's CUDA extensions for sm_120 without touching a GPU "
                         "(run with CUDA_VISIBLE_DEVICES= outside the GPU lock)")
    args = ap.parse_args()
    sys.path.insert(0, args.tf)
    t_start = time.time()

    import torch

    if args.build_only:
        torch.cuda.get_device_capability = lambda *a, **k: (12, 0)
        torch.cuda.is_available = lambda: True
        from tensorfold.families.qwen4_exp.cuda.engine import build_kernels as _bk
        _bk(exl3=True)
        print("[ref] extensions built", flush=True)
        return 0
    import tensorfold
    from tokenizers import Tokenizer

    from tensorfold.families.qwen4_exp.cuda import exl3 as x3mod
    from tensorfold.families.qwen4_exp.cuda import decode as dec
    from tensorfold.families.qwen4_exp.cuda.engine import build_kernels
    from tensorfold.families.qwen4_exp.cuda.forward import commit

    torch.cuda.set_device(0)
    keep: list = []
    patched = patch_expert_table(x3mod, host_mapped_factory(torch, keep))
    x3mod.PREFILL_ROWS = 64        # the n-gram staging / expert scratch rows (prompts here are <= 40 tokens)

    tok = Tokenizer.from_file(str(Path(args.pack) / "tokenizer.json"))
    build_kernels(exl3=True)
    t0 = time.time()
    w = x3mod.load(Path(args.pack), "cuda", mtp=False, draft_vocab=None)
    load_s = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"[ref] loaded in {load_s:.0f} s, device peak {mem:.2f} GiB, host-mapped expert buffers "
          f"{sum(a.nbytes for a in keep) / 2**30:.2f} GiB, centred offset {w.meta.get('centred_offset')}", flush=True)
    if args.layers_only:
        return 0
    e = dec.Engine(w, capacity=128, max_rows=1, prefill_rows=64, graphs=False)

    @torch.no_grad()
    def teacher_forced(ids):
        """Serial path: prefill ids[0], then one decode step per token; logits after every position."""
        rows = []
        dec.prefill_begin(e, ids, mtp=False)
        last = dec.prefill_chunk(e, ids[:1], 0, mtp=False)
        rows.append(top(torch, last[0]))
        for t in ids[1:]:
            lg = e.forward([t])
            rows.append(top(torch, lg[0]))
            commit(w, e.st, e.buf, 1, 1)
        return rows

    @torch.no_grad()
    def greedy(ids, n):
        """The engine's own prompt path (one prefill chunk), then n-1 greedy serial decode steps."""
        dec.prefill_begin(e, ids, mtp=False)
        last = dec.prefill_chunk(e, ids, 0, mtp=False)
        steps = [top(torch, last[0])]
        out = [steps[0][0][0]]
        while len(out) < n:
            lg = e.forward([out[-1]])
            steps.append(top(torch, lg[0]))
            commit(w, e.st, e.buf, 1, 1)
            out.append(steps[-1][0][0])
        return out, steps

    def pack_rows(rows, ids=None):
        res = []
        for p, (ti, tv, lse) in enumerate(rows):
            r = {"pos": p, "top5_ids": ti, "top5_logits": tv, "logsumexp": lse, "margin12": tv[0] - tv[1]}
            if ids is not None and p + 1 < len(ids):
                r["next_actual"] = ids[p + 1]
            res.append(r)
        return res

    prompts = [{"name": "ids16", "ids": PROMPT_IDS, "text": tok.decode(PROMPT_IDS)}]
    for text in TEXTS:
        prompts.append({"name": text, "ids": tok.encode(text, add_special_tokens=False).ids, "text": text})
    results = []
    for p in prompts:
        t1 = time.time()
        tf_rows = teacher_forced(p["ids"])
        gen, steps = greedy(p["ids"], args.gen)
        # the prefill path's last-row logits against the serial path's at the same position
        d = abs(tf_rows[-1][1][0] - steps[0][1][0])
        results.append({**p, "teacher_forced": pack_rows(tf_rows, p["ids"]),
                        "teacher_forced_top1": [r[0][0] for r in tf_rows],
                        "greedy_ids": gen, "greedy_text": tok.decode(gen), "greedy_steps": pack_rows(steps),
                        "prefill_vs_serial_last_top1_logit_absdiff": d, "seconds": time.time() - t1})
        print(f"[ref] {p['name']!r}: {tok.decode(gen)!r} ({time.time() - t1:.1f} s)", flush=True)

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    golden = {
        "what": "Independent reference: Python TensorFold CUDA engine (qwen4_exp/cuda) on the EXL3 pack; routed "
                "expert trellises in host-mapped pinned memory (only allocation site changed), serial decode path, "
                "no MTP, no graphs, bf16 KV, greedy argmax",
        "pack": args.pack,
        "command": " ".join(sys.argv),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {"tensorfold": tensorfold.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
                     "capability": list(torch.cuda.get_device_capability()),
                     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
                     "nvcc": sh("nvcc --version | tail -1")},
        "load_seconds": load_s, "device_peak_gib": mem, "centred_offset": w.meta.get("centred_offset"),
        "patched_expert_table_line": "big = _HOST_MAPPED(total)",
        "harness_changes": ["exl3.expert_table: routed-expert trellis buffer in cudaHostRegister'd host memory (UVA)",
                            "exl3.PREFILL_ROWS=64; Engine(capacity=128, max_rows=1, prefill_rows=64, graphs=False); "
                            "load(mtp=False, draft_vocab=None); capacity admission not run"],
        "notes": "teacher_forced[p] are the logits after position p (predicting ids[p+1]); greedy_steps[0] is "
                 "the prefill chunk's last row, later steps serial decode",
        "prompts": results,
    }
    Path(args.out).write_text(json.dumps(golden, indent=1))
    print(f"[ref] wrote {args.out}", flush=True)
    del patched
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
