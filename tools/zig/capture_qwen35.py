"""Capture real-dimension native Qwen operations and compare their arithmetic with stock MLX operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, normalize_qk

from gen_qwen35_kernels import specs

DTYPES = {k: getattr(mx, k) for k in ("bfloat16", "float32", "uint8", "uint32", "int32")}


def raw(x):
    mx.eval(x)
    return np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)


class Capture:
    def __init__(self, model: Path, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.weights = mx.load(str(model / "model.safetensors"))
        self.operations = []
        self.metrics = []
        self.kernels = {s["key"]: s for s in specs()}
        self.counter = 0

    def save(self, x):
        name = f"{self.counter:04d}.npy"
        self.counter += 1
        np.save(self.out / name, raw(x))
        return name

    def weight(self, name):
        return self.weights["language_model.model." + name]

    def compare(self, name, actual, trusted):
        a, b = np.array(actual.astype(mx.float32)), np.array(trusted.astype(mx.float32))
        delta = a - b
        item = dict(op=name, count=a.size, max_abs=float(np.max(np.abs(delta))),
                    rms=float(np.sqrt(np.mean(delta**2))), relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
                    finite=bool(np.all(np.isfinite(a))))
        self.metrics.append(item)
        print(item, flush=True)
        if not item["finite"]:
            raise AssertionError(name + " produced nonfinite output")

    def operation(self, key, inputs, shapes, dtypes, grid, group, weight_names=None):
        spec = self.kernels[key]
        types = [DTYPES[t] for _, t in spec["template"]]
        kernel = mx.fast.metal_kernel(name="qwen35_" + key,
            input_names=[a.name for a in spec["inputs"]], output_names=[a.name for a in spec["outputs"]],
            source=spec["body"], header=spec["header"])
        result = kernel(inputs=inputs, output_shapes=shapes, output_dtypes=dtypes,
            grid=grid, threadgroup=group, template=[(n, t) for (n, _), t in zip(spec["template"], types)])
        mx.eval(*result)
        op = dict(op=key, grid=grid, threadgroup=group, inputs=[], outputs=[])
        slot = 0
        for arg, x, weight in zip(spec["inputs"], inputs, weight_names or [None] * len(inputs)):
            bound = dict(slot=slot)
            if weight:
                bound.update(weight="language_model.model." + weight, sha256=hashlib.sha256(raw(x).tobytes()).hexdigest())
            else:
                bound["file"] = self.save(x)
            if arg.size < 8:
                bound["constant"] = True
            op["inputs"].append(bound)
            slot += 1
            for suffix, array in (("_shape", list(x.shape)), ("_strides", []), ("_ndim", [x.ndim])):
                if arg.name + suffix in spec["body"]:
                    if suffix == "_strides":
                        raise ValueError("stride fixture needs explicit handling")
                    op["inputs"].append(dict(slot=slot, constant=True, file=self.save(mx.array(array, mx.int32))))
                    slot += 1
        for x in result:
            op["outputs"].append(dict(slot=slot, file=self.save(x), bytes=raw(x).nbytes))
            slot += 1
        self.operations.append(op)
        return result

    def projection(self, name, rows):
        w, sc, bi = [self.weight(name + "." + k) for k in ("weight", "scales", "biases")]
        n, k = w.shape[0], w.shape[1] * 8
        x = mx.random.normal((rows, k)).astype(mx.bfloat16)
        key = f"qmm_{n}_{k}"
        spec = self.kernels[key]["launch"]
        grid = (spec["threads"] * ((n + spec["columns"] - 1) // spec["columns"]), (rows + 7) // 8, 1)
        y, = self.operation(key, [x, w, sc, bi, mx.array([1], mx.float32)], [(rows, n)], [mx.bfloat16],
            grid, (spec["threads"], 1, 1), [None, name + ".weight", name + ".scales", name + ".biases", None])
        trusted = mx.quantized_matmul(x, w, sc, bi, transpose=True, group_size=64, bits=4)
        self.compare(f"{key}/rows{rows}", y, trusted)
        solo = []
        for r in range(rows):
            yi, = mx.fast.metal_kernel(name="qwen35_solo_" + key,
                input_names=[a.name for a in self.kernels[key]["inputs"]], output_names=["OUT"],
                source=self.kernels[key]["body"], header=self.kernels[key]["header"])(
                inputs=[x[r:r+1], w, sc, bi, mx.array([1], mx.float32)], output_shapes=[(1, n)],
                output_dtypes=[mx.bfloat16], grid=(grid[0], 1, 1), threadgroup=(spec["threads"], 1, 1))
            solo.append(yi)
        if not np.array_equal(raw(y), raw(mx.concatenate(solo))):
            raise AssertionError(key + " window differs from solo")

    def glue(self, rows):
        rand = lambda shape: mx.random.normal(shape).astype(mx.bfloat16)
        h, r = rand((rows, 2048)), rand((rows, 2048))
        nw = self.weight("layers.0.input_layernorm.weight")
        eps = mx.array([1e-6], mx.float32)
        x, = self.operation("norm_nores", [h, nw, eps], [(rows, 2048)], [mx.bfloat16], (128, rows, 1), (128, 1, 1))
        self.compare("norm_nores", x, mx.fast.rms_norm(h, nw, 1e-6))
        ho, x = self.operation("norm", [h, r, nw, eps], [(rows, 2048)] * 2, [mx.bfloat16] * 2,
                               (128, rows, 1), (128, 1, 1))
        self.compare("norm", x, mx.fast.rms_norm(h + r, nw, 1e-6))
        qkv, cs = rand((rows, 6144)), rand((3, 6144))
        cw = self.weight("layers.0.linear_attn.conv1d.weight")
        a, b = rand((rows, 16)), rand((rows, 16))
        al, dt = self.weight("layers.0.linear_attn.A_log"), self.weight("layers.0.linear_attn.dt_bias")
        windows = mx.arange(rows)[:, None] + mx.arange(4)[None, :]
        q, k, v, g, beta, co = self.operation("gdn_pre", [qkv, cs, cw, windows, a, b, al, dt],
            [(rows, 16, 128)] * 3 + [(rows, 16)] * 2 + [(rows, 3, 6144)],
            [mx.bfloat16] * 3 + [mx.float32, mx.bfloat16, mx.bfloat16], (32, 48, rows), (32, 1, 1))
        conv = nn.silu(mx.conv1d(mx.concatenate([cs, qkv])[None], cw, groups=6144))[0]
        qr, kr, vr = [part.reshape(rows, 16, 128) for part in mx.split(conv, 3, axis=-1)]
        qr, kr = normalize_qk(qr, kr, inv_scale=128**-0.5, eps=1e-6)
        for name, actual, trusted in (("gdn_q", q, qr), ("gdn_k", k, kr), ("gdn_v", v, vr),
                                      ("gdn_g", g, compute_g(al, a, dt)), ("gdn_beta", beta, mx.sigmoid(b))):
            self.compare(name, actual, trusted)
        state = mx.random.normal((1, 16, 128, 128)).astype(mx.float32) * 0.01
        for save in (0, 1):
            y, states = self.operation("gdn_chain", [q, k, v, g, beta, state, mx.array([rows, save], mx.int32)],
                [(rows, 16, 128), (rows if save else 1, 16, 128, 128)], [mx.bfloat16, mx.float32],
                (32, 128, 16), (32, 4, 1))
            yr, sr = gated_delta_kernel(q[None], k[None], v[None], g[None], beta[None], state)
            self.compare("gdn_chain_y", y, yr[0])
            self.compare("gdn_chain_state", states[-1], sr[0])
        z = rand((rows, 16, 128))
        gn = self.weight("layers.0.linear_attn.norm.weight")
        post, = self.operation("gdn_post", [y, z, gn, eps], [(rows, 16, 128)], [mx.bfloat16],
                               (32, 16, rows), (32, 1, 1))
        from mlx_lm.models.activations import precise_swiglu, swiglu
        norm = mx.fast.rms_norm(y, gn, 1e-6)
        self.compare("gdn_post", post, precise_swiglu(y, z, norm))
        gate, up = rand((rows, 6144)), rand((rows, 6144))
        act, = self.operation("mlp_act", [gate, up], [(rows, 6144)], [mx.bfloat16], (6144, rows, 1), (256, 1, 1))
        self.compare("mlp_act", act, swiglu(gate, up))

    def attention(self, rows, prefix):
        cap, nch = prefix + rows + 17, (prefix + rows + 127) // 128
        q = mx.random.normal((8, rows, 256)).astype(mx.bfloat16)
        k, v = [mx.random.normal((2, cap, 256)).astype(mx.bfloat16) for _ in range(2)]
        dims = mx.array([prefix, rows, cap, nch, 0, rows, 0, 0], mx.int32)
        shapes = [(8, rows, nch), (8, rows, nch), (8, rows, nch, 256)]
        partial = self.operation("attn_partial", [q, k, v, mx.array([256**-0.5], mx.float32), dims],
            shapes, [mx.float32] * 3, (512, nch, 2 * rows), (512, 1, 1))
        y, = self.operation("attn_merge", [*partial, dims], [(rows, 8, 256)], [mx.bfloat16],
                            (32, 8, rows), (32, 1, 1))
        mask = mx.arange(prefix + rows)[None, :] <= prefix + mx.arange(rows)[:, None]
        trusted = mx.fast.scaled_dot_product_attention(q[None], k[:, :prefix+rows][None], v[:, :prefix+rows][None],
                                                       scale=256**-0.5, mask=mask)[0].transpose(1, 0, 2)
        self.compare(f"attention/{rows}/{prefix}", y, trusted)

    def layout(self, rows):
        from metal_source import Arg
        text = (Path(__file__).resolve().parents[2] / "zig/kernels/metal/qwen3_5/layout.metal").read_text()
        rotate = text[text.index("inline bfloat qwen35_rotate"):text.index("kernel void qwen35_queries")]

        def call(key, ins, outs, inputs, shapes, grid, group, names=None):
            tail = text.split("kernel void " + key + "(", 1)[1]
            body = tail.split(") {", 1)[1].split("\n}", 1)[0]
            if key == "qwen35_head_norm":
                body = re.sub(r"\brow\b", "threadgroup_position_in_grid.x", body)
                body = re.sub(r"\bt\b", "thread_position_in_threadgroup.x", body)
                body = re.sub(r"\blane\b", "thread_index_in_simdgroup", body)
                body = re.sub(r"\bsg\b", "simdgroup_index_in_threadgroup", body)
            else:
                body = re.sub(r"\b(?:p|pos)\.", "thread_position_in_grid.", body)
            for i, d in enumerate("xyzw"):
                body = body.replace("dims." + d, f"dims[{i}]")
            self.kernels[key] = dict(inputs=ins, outputs=outs, body=body, header=rotate if "rotate(" in body else "", template=())
            result = self.operation(key, inputs, shapes, [DTYPES[a.dtype] for a in outs], grid, group, names)
            slots = {"qwen35_head_norm": [0, 1, 3, 2], "qwen35_queries": [0, 2, 1], "qwen35_keys": [0, 1, 4, 2, 3]}.get(key)
            if slots:
                for bound, slot in zip(self.operations[-1]["inputs"] + self.operations[-1]["outputs"], slots):
                    bound["slot"] = slot
            return result

        bf = lambda n: Arg(n, "bfloat16")
        ids = mx.array([42 + i for i in range(rows)], mx.uint32)
        w, sc, bi = [self.weight("embed_tokens." + k) for k in ("weight", "scales", "biases")]
        emb, = call("qwen35_embed", [Arg("ids", "uint32"), Arg("weight", "uint8"), bf("scales"), bf("biases")],
            [bf("out")], [ids, w.view(mx.uint8), sc, bi], [(rows, 2048)], (1024, rows, 1), (256, 1, 1),
            [None, "embed_tokens.weight", "embed_tokens.scales", "embed_tokens.biases"])
        deq = mx.dequantize(w[ids], sc[ids], bi[ids], group_size=64, bits=4)
        self.compare("embed", emb, deq)
        qraw = mx.random.normal((rows, 4096)).astype(mx.bfloat16)
        wn = self.weight("layers.3.self_attn.q_norm.weight")
        q, = call("qwen35_head_norm", [bf("x"), bf("weight"), Arg("dims", "uint32", 4)], [bf("out")],
            [qraw, wn, mx.array([rows, 8, 4096, 512], mx.uint32)], [(rows, 8, 256)], (64 * rows * 8, 1, 1), (64, 1, 1))
        self.compare("head_norm", q, mx.fast.rms_norm(qraw.reshape(rows, 8, 512)[..., :256], wn, 1e-6))
        for offset in (0, 127, 32768, 65536):
            qr, = call("qwen35_queries", [bf("x"), Arg("dims", "uint32", 4)], [bf("out")],
                [q, mx.array([offset, rows, 0, 0], mx.uint32)], [(8, rows, 256)], (256, 8, rows), (256, 1, 1))
            trusted = mx.fast.rope(q.transpose(1, 0, 2), 64, traditional=False, base=1e7, scale=1, offset=offset)
            self.compare("rope/" + str(offset), qr, trusted)
        kn = mx.random.normal((rows, 2, 256)).astype(mx.bfloat16)
        v = mx.random.normal((rows, 2, 256)).astype(mx.bfloat16)
        keys, values = call("qwen35_keys", [bf("k"), bf("v"), Arg("dims", "uint32", 4)], [bf("keys"), bf("values")],
            [kn, v, mx.array([0, rows, 0, 0], mx.uint32)], [(2, rows, 256)] * 2, (256, 2, rows), (256, 1, 1))
        self.compare("cache_keys", keys, mx.fast.rope(kn.transpose(1, 0, 2), 64, traditional=False, base=1e7, scale=1, offset=0))
        self.compare("cache_values", values, v.transpose(1, 0, 2))
        mixed = mx.random.normal((rows, 2048)).astype(mx.bfloat16)
        gated, = call("qwen35_attention_gate", [bf("x"), bf("q")], [bf("out")], [mixed, qraw], [(rows, 2048)],
                       (2048, rows, 1), (256, 1, 1))
        self.compare("attention_gate", gated, mixed * mx.sigmoid(qraw.reshape(rows, 8, 512)[..., 256:].reshape(rows, 2048)))

    def finish(self):
        (self.out / "manifest.json").write_text(json.dumps(dict(ops=self.operations), indent=2) + "\n")
        (self.out / "fidelity-ops.json").write_text(json.dumps(dict(operations=self.metrics), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    mx.random.seed(42)
    c = Capture(args.model, args.output)
    names = ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z", "layers.0.linear_attn.in_proj_a",
             "layers.3.self_attn.q_proj", "layers.3.self_attn.k_proj", "layers.0.mlp.down_proj", "embed_tokens"]
    for name in names:
        for rows in (1, 7, 16, 32):
            c.projection(name, rows)
    for rows in (1, 7, 16):
        c.glue(rows)
        c.layout(rows)
        for prefix in (0, 127, 128, 129, 255, 256):
            c.attention(rows, prefix)
    c.finish()


if __name__ == "__main__":
    main()
