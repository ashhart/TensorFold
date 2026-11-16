"""Tiny-shape MLA dump (TFLINGA1). Random values in Ling-3.0-tiny tensor shapes, not checkpoint weights."""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"
TINY_CONFIG = FIXTURES / "tiny_config.json"
LAYER = 3
MAGIC_M1 = b"TFLINGM1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lf = load("ling_forward", HERE / "ling_forward.py")


class FakeCkpt:
    def __init__(self, tensors):
        self.tensors = tensors
        self.keys = set(tensors)

    def t(self, name):
        return self.tensors[name]

    def lin(self, module):
        return self.t(module + ".weight").double()


class CaptureModel(lf.Model):
    def lin(self, x, module):
        y = super().lin(x, module)
        self.captured[module] = y.detach().clone()
        if module.endswith("dense"):
            self.y_pre_dense = x.detach().clone()
        return y


def f32_bytes(t):
    return t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()


def f64_bytes(t):
    return t.detach().cpu().contiguous().to(torch.float64).numpy().tobytes()


def make_weights(cfg, g):
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    qa = cfg["q_lora_rank"]
    lw = cfg["kv_lora_rank"]
    nope, rope, dv = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    p = f"model.layers.{LAYER}.attention."
    scale = cfg.get("initializer_range", 0.02)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * scale

    def rand1(*shape):
        return 0.5 + 0.5 * torch.randn(*shape, generator=g, dtype=torch.float64)

    return {
        p + "q_a_proj.weight": rand(qa, hidden),
        p + "q_a_layernorm.weight": rand1(qa),
        p + "q_b_proj.weight": rand(H * (nope + rope), qa),
        p + "kv_a_proj_with_mqa.weight": rand(lw + rope, hidden),
        p + "kv_a_layernorm.weight": rand1(lw),
        p + "kv_b_proj.weight": rand(H * (nope + dv), lw),
        p + "g_proj.weight": rand(H, hidden),
        p + "dense.weight": rand(hidden, H * dv),
    }


def capture(cfg, tensors_f64, x, pos0):
    p = f"model.layers.{LAYER}.attention."
    model = CaptureModel(FakeCkpt(tensors_f64), cfg, lambda z: z)
    model.captured = {}
    model.y_pre_dense = None
    with torch.no_grad():
        out = model.mla(x, p, pos0)
    H = cfg["num_attention_heads"]
    q = model.captured[p + "q_b_proj"].view(x.shape[0], H, -1)
    kv = model.captured[p + "kv_a_proj_with_mqa"]
    g = model.captured[p + "g_proj"]
    return {
        "q": q,
        "kv": kv,
        "g": g,
        "y_pre": model.y_pre_dense,
        "y": out,
        "w_rms": tensors_f64[p + "kv_a_layernorm.weight"],
        "kvb": tensors_f64[p + "kv_b_proj.weight"],
        "Wo": tensors_f64[p + "dense.weight"],
    }


def run_cli(ckpt_dir, acts_path, out_path, pos0):
    subprocess.check_call(
        [
            sys.executable,
            str(HERE / "ling_forward.py"),
            str(ckpt_dir),
            str(out_path),
            "--mla-layer",
            str(LAYER),
            "--acts",
            str(acts_path),
            "--pos0",
            str(pos0),
        ]
    )


def write_case_files(tmp, cfg, tensors_f32, x, pos0, a1_path):
    ckpt_dir = Path(tmp) / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    save_file(tensors_f32, ckpt_dir / "model.safetensors")
    acts_path = Path(tmp) / f"acts_p{pos0}.bin"
    acts_path.write_bytes(f64_bytes(x))
    run_cli(ckpt_dir, acts_path, a1_path, pos0)
    raw = a1_path.read_bytes()
    assert raw[:8] == b"TFLINGA1", raw[:8]
    T, hidden, layer, p0 = struct.unpack_from("<4I", raw, 8)
    assert T == x.shape[0] and hidden == cfg["hidden_size"] and layer == LAYER and p0 == pos0
    n = T * hidden
    y64 = torch.frombuffer(bytearray(raw[24 : 24 + n * 8]), dtype=torch.float64).clone().view(T, hidden)
    ybf = torch.frombuffer(bytearray(raw[24 + n * 8 : 24 + 2 * n * 8]), dtype=torch.float64).clone().view(T, hidden)
    return y64, ybf


def pack_mixer(cfg, tensors_f64, cases, path):
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    w_rms = tensors_f64[f"model.layers.{LAYER}.attention.kv_a_layernorm.weight"]
    kvb = tensors_f64[f"model.layers.{LAYER}.attention.kv_b_proj.weight"]
    Wo = tensors_f64[f"model.layers.{LAYER}.attention.dense.weight"]
    with open(path, "wb") as f:
        f.write(MAGIC_M1)
        f.write(struct.pack("<4I", hidden, LAYER, H, len(cases)))
        f.write(f32_bytes(w_rms))
        f.write(f32_bytes(kvb))
        f.write(f32_bytes(Wo))
        for case in cases:
            T = case["q"].shape[0]
            f.write(struct.pack("<2I", T, case["pos0"]))
            f.write(f32_bytes(case["q"].reshape(T, -1)))
            f.write(f32_bytes(case["kv"]))
            f.write(f32_bytes(case["g"]))
            f.write(f64_bytes(case["y_pre"]))
            f.write(f64_bytes(case["y64"]))
            f.write(f64_bytes(case["ybf"]))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ling-mla-dumps")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(TINY_CONFIG.read_text())
    g = torch.Generator().manual_seed(0x74696E79)
    tensors_f64 = make_weights(cfg, g)
    tensors_f32 = {k: v.float().contiguous() for k, v in tensors_f64.items()}
    tensors_cli = {k: v.double() for k, v in tensors_f32.items()}
    specs = (
        (4, 0, 0x6D6C61, "mla_tiny_t4_pos0.bin"),
        (2, 30000, 0x6D6C62, "mla_tiny_t2_pos30000.bin"),
    )
    packed = []
    with tempfile.TemporaryDirectory() as tmp:
        for T, pos0, seed, a1_name in specs:
            gx = torch.Generator().manual_seed(seed)
            x = torch.randn(T, cfg["hidden_size"], generator=gx, dtype=torch.float64)
            a1_path = out / a1_name
            y64, ybf = write_case_files(tmp, cfg, tensors_f32, x, pos0, a1_path)
            cap = capture(cfg, tensors_cli, x, pos0)
            worst = (cap["y"] - y64).abs().max().item()
            print(f"CaptureModel vs TFLINGA1 T={T} pos0={pos0} worst abs {worst:.6e}")
            if worst > 1e-12:
                raise SystemExit(f"CLI y64 does not match CaptureModel: {worst}")
            packed.append(
                {
                    "pos0": pos0,
                    "q": cap["q"],
                    "kv": cap["kv"],
                    "g": cap["g"],
                    "y_pre": cap["y_pre"],
                    "y64": y64,
                    "ybf": ybf,
                }
            )
    mixer_path = out / "mla_tiny_mixer.bin"
    pack_mixer(cfg, tensors_cli, packed, mixer_path)
    print("wrote", mixer_path, "bytes", mixer_path.stat().st_size)
    for _, _, _, name in specs:
        p = out / name
        print("wrote", p, "bytes", p.stat().st_size)


if __name__ == "__main__":
    main()
