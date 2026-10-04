"""Dev-time oracle (runs in the PyTorch image, never at runtime): fixtures for the Zig bit checks from the Python engine's own ops."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import torch

from tensorfold.cuda.kernels import gdn as deltanet
from tensorfold.families.qwen3_5.cuda import glue

DEV = torch.device("cuda:0")
HK, HV, DK = 16, 48, 128            # Qwen3.8-27B linear attention heads (config.json)
HIDDEN, INTER, EPS = 5120, 17408, 1e-6


def save(out: Path, params: dict, arrays: dict[str, torch.Tensor], extra: dict | None = None) -> None:
    """manifest.json plus one raw little-endian file per tensor."""

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params, "arrays": {}}
    for name, t in arrays.items():
        t = t.detach().contiguous().cpu()
        raw = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
        (out / f"{name}.bin").write_bytes(raw)
        manifest["arrays"][name] = {"file": f"{name}.bin", "dtype": str(t.dtype).replace("torch.", ""),
                                    "shape": list(t.shape)}
    manifest.update(extra or {})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def gdn_replay(out: Path, g: torch.Generator) -> dict:
    layers, streams, width, stride = 4, 2, 16, 8
    k = (torch.randn(layers, width, HK, DK, generator=g, device=DEV) / DK ** 0.5).to(torch.bfloat16)
    v = torch.randn(layers, width, HV, DK, generator=g, device=DEV).to(torch.bfloat16)
    gate = torch.exp(-torch.nn.functional.softplus(torch.randn(layers, width, HV, generator=g, device=DEV)))
    beta = torch.sigmoid(torch.randn(layers, width, HV, generator=g, device=DEV))
    states = torch.randn(streams, layers, HV, DK, DK, generator=g, device=DEV) * 0.05
    before = states.clone()
    rows = torch.zeros(streams, stride, dtype=torch.int32, device=DEV)
    rows[0, :5] = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32)
    rows[1, :3] = torch.tensor([8, 9, 11], dtype=torch.int32)
    counts = torch.tensor([5, 3], dtype=torch.int32, device=DEV)
    ptrs = deltanet.replay_table([k[i] for i in range(layers)], [v[i] for i in range(layers)],
                                 [gate[i] for i in range(layers)], [beta[i] for i in range(layers)],
                                 [[states[s, i] for i in range(layers)] for s in range(streams)])
    table = deltanet.to_device(ptrs, torch.int64, DEV)
    result = deltanet.replay(table, layers, streams, rows, counts, k[0], v[0])
    deltanet.replay(table, layers, streams, rows, counts, k[0], v[0], in_place=True)
    torch.cuda.synchronize()
    save(out, {"layers": layers, "streams": streams, "width": width, "row_stride": stride, "hk": HK, "hv": HV,
               "dv": DK}, {"k": k, "v": v, "g": gate, "beta": beta, "states": before, "rows": rows,
                           "counts": counts, "out": result, "states_after": states})
    return {"changed_states": bool((states != before).any().item())}


def gdn_window(out: Path, g: torch.Generator, parents: list[int]) -> dict:
    width = len(parents)
    q = (torch.randn(width, HK, DK, generator=g, device=DEV) / DK).to(torch.bfloat16)
    k = (torch.randn(width, HK, DK, generator=g, device=DEV) / DK ** 0.5).to(torch.bfloat16)
    v = torch.randn(width, HV, DK, generator=g, device=DEV).to(torch.bfloat16)
    gate = torch.exp(-torch.nn.functional.softplus(torch.randn(width, HV, generator=g, device=DEV)))
    beta = torch.sigmoid(torch.randn(width, HV, generator=g, device=DEV))
    state = torch.randn(HV, DK, DK, generator=g, device=DEV) * 0.05
    plan = deltanet.plan([parents], DEV)
    y = deltanet.tree(q, k, v, gate, beta, plan, state=state)
    torch.cuda.synchronize()
    save(out, {"width": width, "slots": plan.slots, "max_rows": plan.max_rows, "hk": HK, "hv": HV, "dv": DK},
         {"q": q, "k": k, "v": v, "g": gate, "beta": beta, "state": state, "plan": plan.entries, "y": y},
         {"parents": parents})
    return {"slots": plan.slots, "max_rows": plan.max_rows}


def compiled(fn) -> dict:
    """The JIT cache's one specialization of ``fn``: signature, constants, and launch args specialized as multiples of 16."""

    found = {}
    for entry in getattr(fn, "device_caches", {}).values():
        for part in entry if isinstance(entry, tuple) else (entry,):
            if not isinstance(part, dict):
                continue
            for ck in part.values():
                src = getattr(ck, "src", None)
                if src is None:
                    continue
                for attr in ("signature", "constants", "constexprs", "attrs"):
                    val = getattr(src, attr, None)
                    if val is not None:
                        found[attr] = repr(val)
                found["metadata"] = repr(getattr(ck, "metadata", None))
                kinds = list(src.signature.values())
                position = {i: p for p, i in enumerate(i for i, k in enumerate(kinds) if k != "constexpr")}
                found["divisible16"] = sorted(position[key[0]] for key, specs in src.attrs.items()
                                              if key[0] in position and ["tt.divisibility", 16] in [list(x) for x in specs])
    return found


def triton_files(cache: Path, name: str, out: Path) -> dict:
    """Copies the kernel's cubin, metadata and PTX from a fresh TRITON_CACHE_DIR; the PTX entry is the ABI."""

    metas = [p for p in glob.glob(str(cache / "**" / f"{name}.json"), recursive=True)]
    if len(metas) != 1:
        raise SystemExit(f"expected one compiled {name} in {cache}, found {metas}")
    base = Path(metas[0]).with_suffix("")
    for ext in ("cubin", "json", "ptx"):
        shutil.copy(f"{base}.{ext}", out / f"kernel.{ext}")
    ptx = (out / "kernel.ptx").read_text()
    entry = re.search(r"\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
    params = re.findall(r"\.param\s+\.(\w+)\s+(?:\.ptr\s+\.global\s+\.align\s+\d+\s+)?(\w+)", entry.group(2))
    return {"ptx_entry": entry.group(1), "ptx_params": params, "cache_dir": str(base.parent.name)}


def triton_swiglu(out: Path, g: torch.Generator, cache: Path) -> dict:
    width = 16
    gate = torch.randn(width, INTER, generator=g, device=DEV).to(torch.bfloat16)
    up = torch.randn(width, INTER, generator=g, device=DEV).to(torch.bfloat16)
    act, xs = glue.swiglu(gate, up)
    torch.cuda.synchronize()
    out.mkdir(parents=True, exist_ok=True)
    abi = triton_files(cache, "_swiglu", out)
    args = [{"kind": "ptr", "array": "gate"}, {"kind": "ptr", "array": "up"},
            {"kind": "ptr", "array": "out", "output": True}, {"kind": "ptr", "array": "xs", "output": True}]
    save(out, {"N": INTER, "BLOCK": 1024, "num_warps": 4},
         {"gate": gate, "up": up, "out": torch.zeros_like(act), "xs": torch.zeros_like(xs), "expected_out": act,
          "expected_xs": xs},
         {"args": args, "grid": [width, -(-INTER // 1024), 1], "abi": abi, "jit": (jit := compiled(glue._swiglu)),
          "divisible16": jit["divisible16"]})
    return abi


def triton_add_rmsnorm(out: Path, g: torch.Generator, cache: Path) -> dict:
    width = 16
    x = torch.randn(width, HIDDEN, generator=g, device=DEV).to(torch.bfloat16)
    r = torch.randn(width, HIDDEN, generator=g, device=DEV).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=DEV)).to(torch.bfloat16)
    h, y, xs = glue.add_rmsnorm(x, r, w, EPS)
    torch.cuda.synchronize()
    out.mkdir(parents=True, exist_ok=True)
    abi = triton_files(cache, "_add_rmsnorm", out)
    bits = struct.unpack("<I", struct.pack("<f", EPS))[0]
    args = [{"kind": "ptr", "array": "x"}, {"kind": "ptr", "array": "r"}, {"kind": "ptr", "array": "w"},
            {"kind": "ptr", "array": "h", "output": True}, {"kind": "ptr", "array": "y", "output": True},
            {"kind": "ptr", "array": "xs", "output": True}, {"kind": "f32", "bits": bits}]
    save(out, {"D": HIDDEN, "BLOCK": 8192, "HAS_R": True, "num_warps": 8},
         {"x": x, "r": r, "w": w, "h": torch.zeros_like(h), "y": torch.zeros_like(y), "xs": torch.zeros_like(xs),
          "expected_h": h, "expected_y": y, "expected_xs": xs},
         {"args": args, "grid": [width, 1, 1], "abi": abi, "jit": (jit := compiled(glue._add_rmsnorm)),
          "divisible16": jit["divisible16"]})
    return abi


def main() -> int:
    out = Path(sys.argv[1])
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    g = torch.Generator(device=DEV)
    g.manual_seed(20261003)
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "capability": list(torch.cuda.get_device_capability())}
    import triton
    report["triton"] = triton.__version__
    report["gdn_replay"] = gdn_replay(out / "fixtures/gdn_replay", g)
    report["gdn_chain"] = gdn_window(out / "fixtures/gdn_chain", g, [-1] + list(range(15)))
    report["gdn_tree"] = gdn_window(out / "fixtures/gdn_tree", g, [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
    report["triton_swiglu"] = triton_swiglu(out / "fixtures/triton_swiglu", g, cache)
    report["triton_add_rmsnorm"] = triton_add_rmsnorm(out / "fixtures/triton_add_rmsnorm", g, cache)
    ext = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    ninja = sorted(glob.glob(str(ext / "**" / "tensorfold_gdn_v2" / "build.ninja"), recursive=True))
    sos = sorted(glob.glob(str(ext / "**" / "tensorfold_gdn_v2" / "tensorfold_gdn_v2*.so"), recursive=True))
    report["gdn_build_ninja"] = ninja
    report["gdn_extension"] = sos
    if ninja:
        text = Path(ninja[0]).read_text()
        report["gdn_nvcc"] = {k: v.strip() for k, v in re.findall(r"^(nvcc|cuda_cflags|cuda_post_cflags) = (.*)$",
                                                                    text, re.M)}
    (out / "oracle.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
