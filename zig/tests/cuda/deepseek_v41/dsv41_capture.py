"""pytest plugin: records CUDA extension calls during GPU test suites as `tf-dsv41-test` replay fixtures."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

OUT = Path(os.environ.get("TF_DSV41_CAPTURE", "capture"))
CALL_MAX = int(os.environ.get("TF_DSV41_CAPTURE_CALL_MB", "96")) << 20      # storages a call
TOTAL_MAX = int(os.environ.get("TF_DSV41_CAPTURE_TOTAL_MB", "3072")) << 20   # every call (before + after)
PER_FUNC = int(os.environ.get("TF_DSV41_CAPTURE_PER_FUNC", "48"))            # calls an (extension, function)
FUNC_MAX = int(os.environ.get("TF_DSV41_CAPTURE_FUNC_MB", "320")) << 20      # bytes an (extension, function)
PER_TEST = int(os.environ.get("TF_DSV41_CAPTURE_PER_TEST", "3"))  # calls a (test, extension, function), all params

# extension -> functions recorded (their arguments hold no device addresses: a replay on new buffers is exact)
ALLOW = {
    "tensorfold_exl3_linear_v4": {"rot_in", "linear", "linear_skip", "unpack"},
    "tf_dsv41_x3seg_v3": {"rot_in", "linear"},
    "tf_dsv41_dense3_v1": {"linear", "relayout", "unpack"},
    "tf_dsv41_attn_cuda_v1": {"attn", "topk"},
    "tf_dsv41_mhc_cuda_v1": {"run", "coef"},
    "tf_dsv41_mhc_pf_v1": {"run"},
    "tf_dsv41_router_gemv_v2": {"route"},
    "tf_dsv41_pfdense_v1": {"gemm", "dequant"},
    "tensorfold_exl3_experts_v1": {"down_combine", "combine"},
}

_state = {"bytes": 0, "seq": 0, "per": {}, "per_bytes": {}, "per_test": {}, "test": "setup"}


def _storage(t: torch.Tensor):
    return t.untyped_storage()


class _Rec:
    """The storages of one call, by storage address, and the argument tree referring to them."""

    def __init__(self) -> None:
        self.storages: dict[int, torch.UntypedStorage] = {}
        self.ok = True

    def arg(self, v):
        if isinstance(v, torch.Tensor):
            if not v.is_cuda:
                self.ok = False
                return None
            s = _storage(v)
            base = s.data_ptr()
            self.storages.setdefault(base, s)
            return {"t": hex(base), "off": v.data_ptr() - base, "dtype": str(v.dtype).replace("torch.", ""),
                    "shape": list(v.shape), "stride": list(v.stride())}
        if isinstance(v, bool):
            return {"b": v}
        if isinstance(v, int):
            return {"i": v}
        if isinstance(v, float):
            return {"f": v}
        if v is None:
            return {"n": True}
        if isinstance(v, (list, tuple)):
            return {"l": [self.arg(x) for x in v]}
        self.ok = False
        return None


def _snapshot(storages) -> dict[int, bytes]:
    torch.cuda.synchronize()
    return {k: bytes(torch.empty(0, dtype=torch.uint8, device=s.device).set_(s).cpu().numpy().tobytes())
            for k, s in storages.items()}


def _record(ext: str, func: str, fn, args, kwargs):
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    key = (ext, func)
    tkey = (_state["test"], ext, func)
    if (capturing or func not in ALLOW.get(ext, ()) or _state["per"].get(key, 0) >= PER_FUNC
            or _state["per_test"].get(tkey, 0) >= PER_TEST):
        return fn(*args, **kwargs)
    rec = _Rec()
    a = [rec.arg(v) for v in args]
    kw = {k: rec.arg(v) for k, v in kwargs.items()}
    size = sum(s.nbytes() for s in rec.storages.values())
    if (not rec.ok or size > CALL_MAX or _state["bytes"] + 2 * size > TOTAL_MAX
            or _state["per_bytes"].get(key, 0) + 2 * size > FUNC_MAX):
        return fn(*args, **kwargs)
    before = _snapshot(rec.storages)
    out = fn(*args, **kwargs)
    if isinstance(out, torch.Tensor) or (isinstance(out, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in out)):
        return out                                     # a returned tensor is not an in-place output: not replayed
    after = _snapshot(rec.storages)
    _state["per"][key] = _state["per"].get(key, 0) + 1
    _state["per_bytes"][key] = _state["per_bytes"].get(key, 0) + 2 * size
    _state["per_test"][tkey] = _state["per_test"].get(tkey, 0) + 1
    _state["bytes"] += 2 * size
    _state["seq"] += 1
    d = OUT / f"{_state['seq']:05d}_{ext}_{func}"
    d.mkdir(parents=True, exist_ok=True)
    names = {}
    arrays = {}
    for i, k in enumerate(rec.storages):
        names[hex(k)] = f"s{i}"
        (d / f"s{i}_before.bin").write_bytes(before[k])
        (d / f"s{i}_after.bin").write_bytes(after[k])
        arrays[f"s{i}_before"] = {"file": f"s{i}_before.bin", "dtype": "uint8", "shape": [len(before[k])]}
        arrays[f"s{i}_after"] = {"file": f"s{i}_after.bin", "dtype": "uint8", "shape": [len(after[k])]}

    def rename(node):
        if isinstance(node, dict):
            if "t" in node:
                node["t"] = names[node["t"]]
            for v in node.values():
                if isinstance(v, (list, dict)):
                    rename(v)
        elif isinstance(node, list):
            for v in node:
                rename(v)
        return node

    manifest = {"params": {"kind": "replay", "storages": len(rec.storages)}, "arrays": arrays,
                "call": {"ext": ext, "func": func, "args": rename(a), "kwargs": rename(kw), "test": _state["test"]}}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return out


class _Proxy:
    """A loaded extension whose functions record their calls."""

    def __init__(self, mod, name: str) -> None:
        self._mod, self._name = mod, name

    def __getattr__(self, attr):
        fn = getattr(self._mod, attr)
        if not callable(fn) or attr not in ALLOW.get(self._name, ()):
            return fn
        name = self._name

        def call(*args, **kwargs):
            return _record(name, attr, fn, args, kwargs)

        return call


def pytest_configure(config):
    import tensorfold.cuda.build as B

    real = B.load

    def load(name, *args, **kwargs):
        mod = real(name, *args, **kwargs)
        return _Proxy(mod, name) if name in ALLOW else mod

    B.load = load
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"dsv41_capture: recording {len(ALLOW)} extensions to {OUT}", flush=True)


def pytest_runtest_setup(item):
    _state["test"] = re.sub(r"[^A-Za-z0-9_.\[\]-]", "_", item.nodeid)[-160:]


def pytest_sessionfinish(session, exitstatus):
    per = {f"{e}.{f}": n for (e, f), n in sorted(_state["per"].items())}
    (OUT / "capture.json").write_text(json.dumps({"calls": _state["seq"], "bytes": _state["bytes"], "per": per},
                                                 indent=1))
