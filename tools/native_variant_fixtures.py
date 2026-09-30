"""Capture original optional Metal variants for independent native dispatch tests.

The original upstream tests still make their own assertions. This recorder additionally
saves each distinct launch's inputs, parameters, and outputs. Native replay uses only
the checked-in embedded kernel catalog, never executable source from a fixture.
"""
import argparse
import hashlib
import json
import re
import sys
import struct
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.native_runtime import require_mlx

def gemma_quantization_variants(capture):
    from tensorfold.kernels.gemma.v1 import glue, moe
    from tensorfold.kernels.nemotron.lightning.v1 import rows as row_kernels

    def quantized(shape, group, bits, seed):
        w = (mx.random.normal(shape, key=mx.random.key(seed)) * .05).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=group, bits=bits)
        return SimpleNamespace(weight=q, scales=s, biases=b, group_size=group, bits=bits)

    for group in (32, 64, 128):
        router = quantized((32, 512), group, 8, 1)
        gate = quantized((32, 512, 512), group, 4, 2)
        up = quantized((32, 512, 512), group, 4, 3)
        down = quantized((32, 512, 512), group, 4, 4)
        qkv = quantized((1024, 512), group, 4, 5)
        for rows in (1, 7, 16):
            capture.test = f"gemma-groups-{group}-rows-{rows}"
            x = (mx.random.normal((rows, 512), key=mx.random.key(6)) * .1).astype(mx.bfloat16)
            logits = moe.router_logits(x, router)
            ids, probabilities = moe.route(logits, mx.ones((32,), mx.bfloat16), 4)
            act = moe.expert_gateup(x, ids, 4, gate, up)
            y = moe.expert_down(act, ids, probabilities, 4, down)
            for r in range(rows):
                serial = moe.expert_gateup(x[r:r + 1], ids[r * 4:(r + 1) * 4], 4, gate, up)
                expected = moe.expert_down(serial, ids[r * 4:(r + 1) * 4], probabilities[r * 4:(r + 1) * 4], 4, down)
                assert mx.array_equal(y[r:r + 1], expected).item()
            positions = mx.array(list(range(rows)) + [0] * max(0, 8 - rows), mx.int32)
            norm = mx.ones((256,), mx.bfloat16)
            inverse = mx.ones((128,), mx.float32)
            eps = mx.array([1e-6] * 8)
            geometry = dict(heads=2, kv_heads=1, head_dim=256, values_are_keys=False)
            fused = glue.qkv_rows(x, qkv.weight, qkv.scales, qkv.biases, group, norm, norm, inverse, positions, eps, **geometry)
            apart = glue.qkv_prep(row_kernels.qmv(x, qkv.weight, qkv.scales, qkv.biases, group), norm, norm, inverse, positions, eps, **geometry)
            assert all(mx.array_equal(a, b).item() for a, b in zip(fused, apart))


def fingerprint(source, header):
    return hashlib.sha256((header + "\0" + source).encode()).hexdigest()


class Capture:
    def __init__(self, directory):
        self.directory = directory
        self.cases = []
        self.seen = set()
        self.test = "manual"
        self.extra = {}
        self.original = mx.fast.metal_kernel
        self.catalog = {}
        self.fingerprints = {}
        for path in sorted(Path("native/metal").glob("*.metal")):
            if path.stem.startswith(("flash_", "glm_", "ds4_", "gemma_", "affine_rows", "lane_qmm_", "prism_", "row_forward_", "row_attention_", "row_qmv", "lane_fuse_", "lane_gdn_", "lane_attention_", "simd_qmm_", "q4_", "nemotron_")):
                self.catalog[(fingerprint(path.read_text(), path.with_suffix(".h").read_text()),
                              path.stem.endswith("_dep"))] = path.stem
                self.fingerprints[path.stem] = fingerprint(path.read_text(), path.with_suffix(".h").read_text())

    def pytest_runtest_setup(self, item):
        self.test = item.nodeid

    def kernel(self, **spec):
        kernel = self.original(**spec)
        source = spec["source"]
        constants = []
        # Python specializes dimensions as leading constexpr declarations;
        # the embedded native body receives those same values as templates.
        float_declarations = ""
        while match := re.match(r"  constexpr (int|float) (\w+) = ([^;]+);\n", source):
            if match[1] == "float":
                constants.append((match[2] + "_BITS", struct.unpack("<i", struct.pack("<f", float(match[3])))[0]))
                float_declarations += f"  const float {match[2]} = as_type<float>(uint({match[2]}_BITS));\n"
            else:
                constants.append((match[2], int(match[3])))
            source = source[match.end():]
        source = float_declarations + source
        header = re.sub(r"\n\[\[max_total_threads_per_threadgroup\(\d+\)\]\]\n$", "", spec.get("header", ""))
        source_hash = fingerprint(source, header)
        key = self.catalog.get((source_hash, "DEP" in spec["input_names"]))
        if key is None:
            normalized_hash = fingerprint(source.rstrip() + "\n", header)
            key = self.catalog.get((normalized_hash, "DEP" in spec["input_names"]))
            if key is not None:
                source_hash = normalized_hash
        if spec["name"].startswith("tf_glm5_"):
            alias = re.sub(r"^tf_glm5_(?:fused_)?", "", spec["name"]).rsplit("_", 1)[0]
            alias = alias if alias.startswith("ds4_") else "glm_" + alias
            if self.fingerprints.get(alias) == source_hash:
                key = alias
        if key is None:
            return kernel

        def launch(**call):
            # These upstream integration tests trace mx.compile; their primitive
            # kernels are recorded by the individual tests with concrete arrays.
            if any(name in self.test for name in ("test_each_kernel_keeps_one_metal_signature", "test_each_fused_layer_is_mlx_lm_s_layer")):
                return kernel(**call)
            call["inputs"] = [x if isinstance(x, mx.array) else mx.array(x) for x in call["inputs"]]
            signature = repr((self.test, key, constants, call.get("template"),
                              [x.shape for x in call["inputs"]], call["grid"], call["threadgroup"]))
            if signature in self.seen:
                return kernel(**call)
            self.seen.add(signature)
            # These variant fixtures check contiguous inputs; the separate attention
            # fixtures retain strided-capacity and partial-cache coverage.
            call["inputs"] = [mx.contiguous(x) for x in call["inputs"]]
            # Some variants deliberately leave unused outputs unwritten (e.g. the
            # terminal recurrent state for a branched tree). Initialize both sides.
            call["init_value"] = 0
            mutated = [1] if key in ("glm_moe_box", "glm_moe_present") else []
            before = {i: mx.array(np.array(call["inputs"][i])) for i in mutated}
            out = kernel(**call)
            mx.eval(*call["inputs"], *out)
            name = f"case{len(self.cases):05}"
            arrays = {f"input{i}": before.get(i, x) for i, x in enumerate(call["inputs"])}
            arrays.update(self.extra)
            arrays.update({f"mutation{i}": call["inputs"][i] for i in mutated})
            arrays.update({f"output{i}": x for i, x in enumerate(out)})
            mx.save_safetensors(str(self.directory / f"{name}.safetensors"), arrays)
            templates = []
            for label, value in [*constants, *call.get("template", [])]:
                if isinstance(value, bool):
                    templates.append(dict(name=label, boolean=value, kind="boolean"))
                elif isinstance(value, int):
                    templates.append(dict(name=label, integer=value, kind="integer"))
                else:
                    templates.append(dict(name=label, dtype=str(value).split(".")[-1], kind="dtype"))
            self.cases.append(dict(name=name, kernel=key, source_sha256=source_hash,
                                   test=self.test, templates=templates, grid=call["grid"], group=call["threadgroup"],
                                   input_count=len(call["inputs"]), output_count=len(out), mutated_inputs=mutated))
            return out
        return launch


def large_family_variants(capture):
    from tests.test_deepseek_v4_kernels import affine, fp4
    from tests.test_glm5_row_kernels import _moe
    from tensorfold.families.deepseek_v4 import moe as DM
    from tensorfold.kernels.deepseek.v4 import rows as DR, moe as DK
    from tensorfold.kernels.glm.flash.v1 import hc as HC, moe as MK, stream_moe as SM
    from tensorfold.kernels.glm.flash.v1 import kernels as GK
    from tensorfold.families.glm5_next import model as GM
    from tensorfold.kernels import inputs
    capture.test = "large_family_components"
    mx.set_default_device(mx.gpu)
    mx.random.seed(741)
    q = affine(1024, 512)
    rows = mx.random.normal((5, 512)).astype(mx.bfloat16)
    assert mx.array_equal(GK.qmv_rows(rows, q), mx.concatenate([q(row[None]) for row in rows])).item()
    x = mx.random.normal((5, 1024)).astype(mx.bfloat16)
    grouped = DR.qmv_rows_grouped(x, q, 2)
    reference = mx.concatenate([DR.grouped_one_row(x[r:r + 1], q, 2) for r in range(5)])
    assert mx.array_equal(grouped, reference).item()
    for hashed in (False, True):
        gate, up, down = fp4(8, 512, 512), fp4(8, 512, 512), fp4(8, 512, 512)
        shared = DM.Shared(affine(512, 512), affine(512, 512), affine(512, 512), 10.0)
        table = mx.stack([mx.random.permutation(8)[:4] for _ in range(16)]) if hashed else None
        moe = DM.MoE(mx.random.normal((8, 512)).astype(mx.bfloat16), mx.random.normal((8,)), table,
                     gate, up, down, shared, SimpleNamespace(num_experts_per_tok=4, routed_scaling_factor=1.5, swiglu_limit=10.0))
        for rows in (1, 5, 16):
            x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
            ids = mx.arange(rows, dtype=mx.uint32)
            logits = mx.concatenate([x[r:r + 1].astype(mx.float32) @ moe.router for r in range(rows)])
            grouped = DK.route(logits, moe, ids)
            routed = DK.routed(x, moe, *grouped)
            shared_out = shared(x, True)
            joint = DK.combine(routed, shared_out, moe.top)
            serial = mx.concatenate([DK.combine(DK.routed(x[r:r + 1], moe, *DK.route(logits[r:r + 1], moe, ids[r:r + 1])),
                                                shared_out[r:r + 1], moe.top) for r in range(rows)])
            assert mx.array_equal(joint, serial).item()
    cfg = SimpleNamespace(hc_mult=4, rms_norm_eps=1e-6, hc_sinkhorn_iters=20, hc_eps=1e-6)
    hc = GM.HC(mx.random.normal((24, 16384)).astype(mx.bfloat16), mx.random.normal((24,)), mx.array([0.5]*3), cfg)
    x = mx.random.normal((3, 4, 4096)).astype(mx.bfloat16)
    norm = mx.ones((4096,), mx.bfloat16)
    hc.fn_packed = None
    want = HC.hc_step(x, None, hc, norm, 1e-6)
    hc.fn_packed = HC.pack_hc_fn(hc.fn.astype(mx.bfloat16))
    got = HC.hc_step(x, None, hc, norm, 1e-6)
    assert all(mx.array_equal(a, b).item() for a, b in zip(got, want))
    moe = _moe()
    experts = moe.cfg.n_routed_experts
    # Reverse physical slots to exercise indirection instead of identity addresses.
    order = mx.arange(experts - 1, -1, -1, dtype=mx.int32)
    pool = {name: SimpleNamespace(weight=q.weight[order], scales=q.scales[order], biases=q.biases[order])
            for name, q in (("gate", moe.gate), ("up", moe.up), ("down", moe.down))}
    streamer = SimpleNamespace(pool=pool, slot_of=order, layer_ids=[inputs.ints((0,))],
                               box=mx.zeros((256,), mx.uint32), hold=lambda token, *args: token)
    for rows in (1, 5, 16):
        x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
        want = MK.moe_rows(moe, x)
        got = SM.moe_rows(moe, x, streamer, 0)
        assert mx.array_equal(got, want).item()
    box = mx.zeros((256,), mx.uint32)
    mx.eval(SM.present(mx.array([1, 3, 1, 6], dtype=mx.uint32), box, experts))
    assert np.array_equal(np.asarray(box)[:experts], np.isin(np.arange(experts), [1, 3, 6]).astype(np.uint32)), np.asarray(box)[:experts]


def extra_variants(capture):
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm, simd_qmm_bits
    from tools.native_legacy import row_forward, row_qmv
    for group in (32, 64, 128):
        weight = (mx.random.normal((512, 1024), key=mx.random.key(group)) * .1).astype(mx.bfloat16)
        q, scales, biases = mx.quantize(weight, group_size=group, bits=4)
        for rows in (1, 3, 8):
            x = mx.random.normal((rows, 1024), key=mx.random.key(rows)).astype(mx.bfloat16)
            parts = row_forward.row_parts(x)
            mx.eval(row_qmv.qmv(x, q, scales, biases, group))
            mx.eval(row_forward.row_qmv_gate_up_act(x, q, scales, biases, group))
            norm_weight = mx.ones((1024,), dtype=mx.bfloat16)
            residual = mx.zeros((rows, 512), dtype=mx.bfloat16)
            for norm in (False, True):
                for epilogue in ("plain", "act", "residual"):
                    capture.test = f"quantized-groups-{group}-rows-{rows}-norm-{norm}-{epilogue}"
                    mx.eval(row_forward.row_qmv_variant(x, q, scales, biases, group,
                            norm=(parts, norm_weight, 1e-6) if norm else None,
                            epilogue=epilogue, res=residual if epilogue == "residual" else None))
    for kind, rows in (("scalar", 1), ("mma", 3)):
        capture.test = f"simd-dependency-{kind}"
        weight = mx.random.normal((512, 1024), key=mx.random.key(73)).astype(mx.bfloat16)
        q, scales, biases = mx.quantize(weight, group_size=64, bits=4)
        x = mx.ones((rows, 1024), dtype=mx.bfloat16)
        expected = simd_qmm.qmm(x, q, scales, biases, kind=kind)
        actual = simd_qmm.qmm(x, q, scales, biases, kind=kind, dep=mx.sum(x))
        assert bool(mx.array_equal(expected, actual).item())
    for bits in simd_qmm_bits.BITS:
        weights = mx.quantize(mx.random.normal((72, 1024), key=mx.random.key(bits)).astype(mx.bfloat16), group_size=64, bits=bits)
        for kind, rows in (("scalar", 1), ("scalar", 4), ("mma", 3), ("mma", 17)):
            capture.test = f"simd-bits-{bits}-{kind}-{rows}"
            x = mx.random.normal((rows, 1024), key=mx.random.key(rows)).astype(mx.bfloat16)
            mx.eval(simd_qmm_bits.qmm(x, *weights, bits, kind=kind))


def grouped_lane_variants(capture):
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm
    for bits in (2, 3, 5, 6, 8):
        for n, k in ((36, 128), (64, 1024), (5120, 1024)):
            weights = mx.quantize(mx.random.normal((n, k), key=mx.random.key(bits + k)).astype(mx.bfloat16), group_size=32, bits=bits)
            x = mx.random.normal((128, k), key=mx.random.key(33)).astype(mx.bfloat16)
            sbt = lane_qmm.pack_scales(weights[1], weights[2])
            capture.test = f"grouped-lane-{bits}-{n}-{k}"
            full = lane_qmm.lane_matmul(x, weights[0], sbt, group=32)
            for rows in (1, 3, 17, 33, 65, 128):
                for tiled in (False, True) if n % 32 == 0 else (False,):
                    q = lane_qmm.tile_weight(weights[0], group=32, bits=bits) if tiled else weights[0]
                    actual = lane_qmm.lane_matmul(x[:rows], q, sbt, group=32, tiled=tiled)
                    assert mx.array_equal(actual, full[:rows]).item()


def flash_variants(capture):
    """Exercise optional grouped experts, routing ties, embedding and projection layouts."""
    from tools.native_legacy import flash

    def weights(shape, seed):
        value = (mx.random.normal(shape, key=mx.random.key(seed)) * .02).astype(mx.bfloat16)
        q, s, b = mx.quantize(value, group_size=32, bits=4)
        return SimpleNamespace(weight=q, scales=s, biases=b)

    def same(a, b):
        assert a.shape == b.shape and bool(mx.array_equal(a, b).item())

    matrix = weights((128, 512), 100)
    gate_weight = mx.random.normal((513, 2560), key=mx.random.key(88)).astype(mx.bfloat16)
    for rows in (1, 3, 16):
        for threads in (256, 512):
            capture.test = f"flash-router-{rows}-{threads}"
            x = mx.random.normal((rows, 2560), key=mx.random.key(rows)).astype(mx.bfloat16)
            logits = flash.router(x, gate_weight, threads=threads, dtype=mx.float32)
            same(flash.router(x, gate_weight, threads=threads, dtype=mx.bfloat16), logits.astype(mx.bfloat16))
    for rows in (1, 3, 8, 16, 32):
        capture.test = f"flash-projection-{rows}"
        x = mx.random.normal((rows, 512), key=mx.random.key(rows)).astype(mx.bfloat16)
        baseline = mx.concatenate([flash.qmv(x[i:i + 1], matrix) for i in range(rows)])
        for rps in (1, 2, 4, 8):
            same(flash.qmv_rows(x, matrix, rows_per_simdgroup=rps), baseline)
            if rows <= 8:
                for sg in (1, 2, 4):
                    same(flash.qmv(x, matrix, rows_per_simdgroup=rps, simdgroups=sg), baseline)
        for tile in (1, 4):
            ids = mx.array([0, 1, 63, 64, 126, 127], dtype=mx.uint32)
            expected = mx.dequantize(matrix.weight, matrix.scales, matrix.biases, group_size=32, bits=4)[ids]
            same(flash.embed_rows(ids, matrix, tile=tile), mx.tile(expected, (1, tile)))
        projected = mx.concatenate((x, x * mx.array(.25, dtype=mx.bfloat16)), axis=-1)
        same(flash.swiglu(projected, projected, width=512, gate_at=0, up_at=512),
             flash.swiglu(projected[:, :512], projected[:, 512:]))

    for experts in (32, 512):
        gate, up = weights((experts, 128, 512), 201), weights((experts, 128, 512), 202)
        down = weights((experts, 512, 128), 203)
        shared = (weights((128, 512), 204), weights((128, 512), 205))
        shared_down = weights((512, 128), 206)
        for rows in (1, 3, 16):
            for mode in ("ties", "dominant", "random"):
                capture.test = f"flash-experts-{experts}-{rows}-{mode}"
                x = mx.random.normal((rows, 512), key=mx.random.key(rows)).astype(mx.bfloat16)
                if mode == "ties":
                    logits = mx.zeros((rows, experts + 1), dtype=mx.float32)
                elif mode == "dominant":
                    logits = mx.broadcast_to(mx.arange(experts + 1, dtype=mx.float32) * (16 / experts),
                                             (rows, experts + 1))
                else:
                    logits = mx.random.normal((rows, experts + 1), key=mx.random.key(rows + experts))
                ids, route_weights = flash.route(logits, 10, experts)
                reference_ids = [sorted(range(experts), key=lambda i: (-row[i], i))[:10]
                                 for row in logits.tolist()]
                same(ids, mx.array(reference_ids, dtype=mx.uint32))
                assert bool(mx.all(mx.abs(mx.sum(route_weights.astype(mx.float32), axis=-1) - 1) < .01).item())
                act, picks, probability = flash.expert_gateup(x, logits, 10, experts, gate, up, shared)
                same(ids, picks)
                group = flash.expert_group(logits, 10, experts)
                same(group[0], picks)
                same(group[1], probability)
                same(flash.grouped_gateup(x, group, gate, up, shared), act)
                y = flash.expert_down_y(act, picks, down, shared_down)
                same(flash.grouped_down(act, group, down, shared_down), y)
                mx.eval(flash.expert_down(act, picks, probability, logits, 10, experts, down, shared_down))
                # The optional route without a shared slot has a separate launch specialization.
                plain_act, plain_picks, plain_probability = flash.expert_gateup(x, logits, 10, experts, gate, up)
                same(plain_act, act[:, :10])
                mx.eval(flash.expert_down(plain_act, plain_picks, plain_probability, logits, 10, experts, down))


def retired_glue_variants(capture):
    """Keep removed folded/stacked glue covered against the independent lane helpers."""
    from tensorfold.kernels.qwen.dense.v1 import lane_glue, lane_tree
    from tools.native_legacy import row_forward

    def random(shape, seed):
        return (mx.random.normal(shape, key=mx.random.key(seed)) * .1).astype(mx.bfloat16)

    def same(a, b):
        assert a.shape == b.shape and bool(mx.array_equal(a, b).item())

    for count in (1, 3, 8):
        capture.test = f"retired-glue-{count}"
        h, delta, weight = random((1, count, 512), 31), random((1, count, 512), 32), mx.ones((512,), dtype=mx.bfloat16)
        for residual in (None, delta):
            ref = lane_glue.norm_xs(h, residual, weight, 1e-6)
            for a, b in zip(row_forward.add_norm(h, residual, weight, 1e-6), ref):
                same(a, b)
        gu = random((1, count, 1024), 33)
        same(row_forward.mlp_act(gu), lane_glue.mlp_act(mx.contiguous(gu[..., :512]), mx.contiguous(gu[..., 512:])))
        y, cs, cw = random((1, count, 1544), 34), random((1, 3, 1024), 35), random((1024, 4), 36)
        alog, dt = mx.zeros((4,), dtype=mx.float32), mx.zeros((4,), dtype=mx.bfloat16)
        parents = list(range(-1, count - 1))
        windows = lane_tree._conv_windows(parents, 3)
        pre = row_forward.gdn_pre(y, cs, cw, windows, alog, dt, nk=2, nv=4, dk=128, dv=128)
        ref = lane_glue.gdn_pre(mx.contiguous(y[..., :1024]), cs, cw, windows,
                                mx.contiguous(y[..., 1540:]), mx.contiguous(y[..., 1536:1540]),
                                alog, dt, nk=2, nv=4, dk=128, dv=128)
        for a, b in zip(pre[:5], ref):
            same(a, b)
        source = lane_glue._GDN_PRE.replace("float(Ain[w * NV + hv])", "float(Ain[w * ZS + AO + hv])")
        source = source.replace("float(Bin[w * NV + hv])", "float(Bin[w * ZS + BO + hv])")
        run = mx.fast.metal_kernel(name="retired_fused_pre", source=source,
                  input_names=["QKV", "CS", "CW", "windows", "Ain", "Bin", "ALOG", "DT"],
                  output_names=["Q", "Kout", "Vout", "G", "BETA"])
        fused = run(inputs=[mx.contiguous(y[..., :1024]), cs, cw, windows, y, y, alog, dt],
                    template=[("NK", 2), ("NV", 4), ("DK", 128), ("DV", 128), ("TAPS", 4),
                              ("ZS", 1544), ("AO", 1540), ("BO", 1536)],
                    grid=(32, 8, count), threadgroup=(32, 1, 1),
                    output_shapes=[a.shape for a in ref], output_dtypes=[a.dtype for a in ref])
        for a, b in zip(fused, ref):
            same(a, b)
        state = mx.zeros((1, 4, 128, 128), dtype=mx.float32)
        rec, _ = row_forward.gated_delta(*pre[:5], state, parents)
        same(rec, lane_tree.gated_delta_tree(*pre[:5], state, parents))
        same(row_forward.gdn_post(rec, y, weight[:128], 1e-6, zo=1024),
             lane_glue.gdn_post(rec, mx.contiguous(y[..., 1024:1536]), weight[:128], 1e-6))


def nemotron_variants(capture):
    from tools.native_legacy import nemotron
    for rows in (1, 3, 16):
        for level in (-1000, 0, 1000):
            for shared in (0, 1, 2):
                for biased in (False, True):
                    capture.test = f"nemotron-route-{rows}-{level}-{shared}-{biased}"
                    logits = mx.full((rows, 128), level, dtype=mx.bfloat16)
                    bias = mx.arange(128, dtype=mx.float32) if biased else mx.zeros((128,))
                    ids, weights = nemotron.route(logits, bias, 6, mx.array([2.5]), shared_slots=shared)
                    routed = list(range(127, 121, -1)) if biased else list(range(6))
                    expected = routed + list(range(128, 128 + shared))
                    assert ids.tolist() == [expected] * rows
                    expected_weight = 0 if level < 0 else 2.5 / 6
                    assert bool(mx.all(mx.abs(weights[:, :6] - expected_weight) < 1e-6).item())
                    if shared:
                        assert bool(mx.all(weights[:, 6:] == 1).item())
        for dims in (512, 2688):
            capture.test = f"nemotron-add-norm-{rows}-{dims}"
            h = mx.random.normal((rows, dims), key=mx.random.key(dims + rows)).astype(mx.bfloat16)
            scale = mx.ones((dims,), dtype=mx.bfloat16)
            eps = mx.array([1e-5], dtype=mx.float32)
            routed = mx.ones((rows, 6, dims), dtype=mx.bfloat16)
            probability = mx.ones((rows, 6), dtype=mx.float32)
            shared = mx.ones((rows, dims), dtype=mx.bfloat16)
            for delta, actual in (
                (6, nemotron.add_norm_experts(h, routed, probability, scale, eps)),
                (7, nemotron.add_norm_moe(h, routed, probability, shared, scale, eps)),
            ):
                expected = nemotron.add_norm(h, mx.full(h.shape, delta, dtype=mx.bfloat16), scale, eps)
                for a, b in zip(actual, expected):
                    assert bool(mx.array_equal(a, b).item())


def row_attention_variants(capture):
    from tensorfold.kernels.qwen.dense.v1.row_attention import row_sdpa, paths_of
    for dims, group in ((32, 8), (128, 4), (256, 1), (256, 6)):
        for prefix in (0, 127, 128, 129, 4095):
            for parents in ((-1, 0, 1, 2, 3), (-1, 0, 0, 1, 2)):
                capture.test = f"row-attention-{dims}-{group}-{prefix}-{parents}"
                q = mx.random.normal((1, 2 * group, len(parents), dims), key=mx.random.key(21)).astype(mx.bfloat16)
                k = mx.random.normal((1, 2, prefix + len(parents) + 17, dims), key=mx.random.key(22)).astype(mx.bfloat16)
                v = mx.random.normal(k.shape, key=mx.random.key(23)).astype(mx.bfloat16)
                actual = row_sdpa(q, k, v, dims ** -.5, prefix, parents)
                _, paths = paths_of(parents)
                for node, path in enumerate(paths):
                    ids = mx.array(list(range(prefix)) + [prefix + row for row in path], dtype=mx.int32)
                    expected = row_sdpa(q[:, :, node:node + 1], mx.take(k, ids, axis=2), mx.take(v, ids, axis=2),
                                        dims ** -.5, prefix + len(path) - 1, (-1,))
                    assert bool(mx.array_equal(actual[:, :, node:node + 1], expected).item())


def attention_and_ple_variants(capture):
    from tensorfold.kernels.qwen.dense.v1 import lane_attention
    from tools.native_legacy import flash
    import numpy as np
    from tools.native_legacy import tree_attention

    parents = [-1, 0, 0, 1, 2, 2, 4, 3]
    from tensorfold.kernels.qwen.dense.v1.lane_tree import tree_paths
    _, paths = tree_paths(parents)
    for prefix in (0, 31, 513, 10007):
        capture.test = f"retired-tree-attention-{prefix}"
        q = mx.random.normal((1, 4, 8, 256), key=mx.random.key(42)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, prefix + 8, 256), key=mx.random.key(43)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape, key=mx.random.key(44)).astype(mx.bfloat16)
        actual = tree_attention.lane_tree_sdpa(q, k, v, .0625, parents)
        for node, path in enumerate(paths):
            indices = mx.array(list(range(prefix)) + [prefix + row for row in path], dtype=mx.int32)
            expected = lane_attention.lane_sdpa(q[:, :, node:node + 1], mx.take(k, indices, axis=2), mx.take(v, indices, axis=2), .0625)
            assert bool(mx.array_equal(actual[:, :, node:node + 1], expected).item())

    direct = lane_attention.DIRECT_P
    try:
        for dims in (128, 256):
            for length in (513, 10007):
                for rows in (1, 3, 8):
                    capture.test = f"attention-partial-{dims}-{length}-{rows}"
                    q = mx.random.normal((1, 4, rows, dims), key=mx.random.key(rows)).astype(mx.bfloat16)
                    k = mx.random.normal((1, 2, length, dims), key=mx.random.key(length)).astype(mx.bfloat16)
                    v = mx.random.normal(k.shape, key=mx.random.key(length + 1)).astype(mx.bfloat16)
                    lane_attention.DIRECT_P = True
                    reference = lane_attention.lane_sdpa(q, k, v, dims ** -.5)
                    lane_attention.DIRECT_P = False
                    actual = lane_attention.lane_sdpa(q, k, v, dims ** -.5)
                    assert bool(mx.array_equal(actual, reference).item())
    finally:
        lane_attention.DIRECT_P = direct
    for dims in (64, 128):
        tables = SimpleNamespace(weights=[], scales=[], biases=[], starts=None, dims=dims)
        starts = [0]
        dense = []
        ids = []
        for g in range(8):
            count = 5 + 2 * g
            w = mx.random.normal((count, dims), key=mx.random.key(g + dims)).astype(mx.bfloat16)
            q, s, b = mx.quantize(w, group_size=32, bits=4)
            tables.weights.append(q)
            tables.scales.append(s)
            tables.biases.append(b)
            dense.append(mx.dequantize(q, s, b, group_size=32, bits=4))
            ids.extend([starts[-1], starts[-1] + 1, starts[-1] + count // 2, starts[-1] + count - 1])
            starts.append(starts[-1] + count)
        tables.starts = mx.array(starts[:-1], dtype=mx.uint32)
        full = mx.concatenate(dense)
        for rows in (1, 3, 16):
            capture.test = f"ple-eight-groups-{dims}-{rows}"
            indices = np.asarray([np.roll(ids, r) for r in range(rows)], dtype=np.uint32)
            actual = flash.ple_lookup(indices, tables)
            expected = full[mx.array(indices)].reshape(rows, -1)
            assert bool(mx.array_equal(actual, expected).item())


def simd_dense_fixtures(directory):
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as sq
    cases = []
    for group in (32, 64):
        for n, k in ((32, 128), (72, 192), (80, 512), (128, 4096), (6144, 128), (6152, 128), (128, 10304)):
            sq.mma_one_row.clear()
            w = mx.random.normal((n, k), key=mx.random.key(n + k)).astype(mx.bfloat16)
            weights = mx.quantize(w, group_size=group, bits=4)
            scalar_ok = sq.check(*weights, group_size=group)
            if not scalar_ok:
                sq.mma_one_row.add((n, k, group))
            key = f"shape{len(cases):03}"
            x = (mx.random.normal((129, k), key=mx.random.key(99)) * 0.5).astype(mx.bfloat16)
            arrays = dict(weight=weights[0], scales=weights[1], biases=weights[2], x=x)
            rows = (1, 2, 3, 4, 8, 16, 17, 24, 25, 65, 129)
            for count in rows:
                arrays[f"out{count}"] = sq.qmm(x[:count], *weights, group_size=group)
            mx.save_safetensors(str(directory / f"{key}.safetensors"), arrays)
            cases.append(dict(key=key, group=group, scalar_ok=scalar_ok, rows=rows))
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} calibrated SIMD shapes: {sum(c['scalar_ok'] for c in cases)} scalar-compatible", flush=True)


def simd_bits_fixtures(directory):
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm_bits as sq, affine_rows
    cases = []
    for bits in sq.BITS:
        for n, k in ((32, 128), (72, 192), (80, 512), (6152, 128), (17408, 5120), (5120, 17408), (5120, 6144), (1024, 5120), (48, 5120)):
            sq.fallback.clear()
            weight = (mx.random.normal((n, k), key=mx.random.key(7)) * .02).astype(mx.bfloat16)
            weights = mx.quantize(weight, group_size=64, bits=bits)
            scalar_ok = sq.check(*weights, bits)
            x = (mx.random.normal((129, k), key=mx.random.key(99)) * .5).astype(mx.bfloat16)
            arrays = dict(weight=weights[0], scales=weights[1], biases=weights[2], x=x)
            rows = (1, 2, 3, 4, 8, 16, 17, 33, 65, 128, 129)
            for count in rows:
                arrays[f"out{count}"] = sq.qmm(x[:count], *weights, bits, kind="mma")
                arrays[f"fallback{count}"] = affine_rows.qmm(x[:count], *weights, 64, bits)
            key = f"shape{len(cases):03}"
            mx.save_safetensors(str(directory / f"{key}.safetensors"), arrays)
            cases.append(dict(key=key, group=64, bits=bits, scalar_ok=scalar_ok, rows=rows))
        for sizes, k in (((48, 48), 1024), ((5120, 1024, 1024), 128), ((24, 24, 24, 24), 512)):
            n = sum(sizes)
            weights = mx.quantize(mx.random.normal((n, k), key=mx.random.key(bits)).astype(mx.bfloat16), group_size=64, bits=bits)
            scalar_ok = sq.check(*weights, bits)
            x = mx.random.normal((17, k), key=mx.random.key(99)).astype(mx.bfloat16)
            arrays = dict(weight=weights[0], scales=weights[1], biases=weights[2], x=x)
            rows = (1, 3, 8, 17)
            for count in rows:
                arrays[f"out{count}"] = sq.qmm(x[:count], *weights, bits, kind="mma")
                arrays[f"fallback{count}"] = affine_rows.qmm(x[:count], *weights, 64, bits)
            key = f"shape{len(cases):03}"
            mx.save_safetensors(str(directory / f"{key}.safetensors"), arrays)
            cases.append(dict(key=key, group=64, bits=bits, scalar_ok=scalar_ok, rows=rows, members=sizes))
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} calibrated 5/6/8-bit SIMD shapes and affine fallback outputs", flush=True)


def flash_weight_fixtures(directory):
    from types import SimpleNamespace
    from tensorfold.kernels.qwen.flash_next.v1 import base, rows
    from tests.test_flash_next_affine import quantized, bf16
    rng = np.random.default_rng(96173)
    cases = []
    formats = [(b, g) for b in (2, 3, 4, 5, 6, 8) for g in (32, 64, 128)]
    for a_fmt in formats:
        for b_fmt in formats:
            folder = directory / f"case{len(cases):04}"
            folder.mkdir(parents=True, exist_ok=True)
            a = quantized(rng, (8, 512), *a_fmt)
            b = quantized(rng, (4, 512), *b_fmt)
            weights = {f"language_model.model.{name}.{suffix}": getattr(part, suffix)
                       for name, part in (("a", a), ("b", b)) for suffix in ("weight", "scales", "biases")}
            dense = bf16(rng, (4, 512))
            weights["language_model.model.dense.weight"] = dense
            mx.save_safetensors(str(folder / "model.safetensors"), weights)
            config = {"quantization": {"bits": a_fmt[0], "group_size": a_fmt[1],
                                       "language_model.model.b": {"bits": b_fmt[0], "group_size": b_fmt[1]},
                                       "model.dense": False}}
            (folder / "config.json").write_text(json.dumps(config))
            combined = base.QWeights.of(a, b)
            stacked = SimpleNamespace(weight=combined.weight, scales=combined.scales, biases=combined.biases,
                                      bits=combined.bits, group_size=combined.group)
            x = bf16(rng, (3, 512))
            expected = {"input": x, "format": mx.array([combined.bits, combined.group], mx.int32),
                        "a.dequant": mx.dequantize(a.weight, a.scales, a.biases, bits=a_fmt[0], group_size=a_fmt[1]),
                        "a.embed": mx.dequantize(a.weight[[0, 7]], a.scales[[0, 7]], a.biases[[0, 7]], bits=a_fmt[0], group_size=a_fmt[1]),
                        "b.projection": rows.qmv_rows(x, b), "stack.projection": rows.qmv_rows(x, stacked),
                        "b.matmul": mx.quantized_matmul(x, b.weight, b.scales, b.biases, transpose=True, bits=b_fmt[0], group_size=b_fmt[1]),
                        "dense.projection": x @ dense.T}
            expected.update({f"stack.{key}": getattr(combined, key) for key in ("weight", "scales", "biases")})
            mx.save_safetensors(str(folder / "expected.safetensors"), expected)
            cases.append(folder.name)
    (directory / "weights.json").write_text(json.dumps(cases))
    print(f"Saved {len(cases)} mixed-format Flash checkpoint and exact-widening cases", flush=True)


def flash_prefill_mm_fixtures(capture):
    import mlx.nn as nn
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as mm
    mm._kernels.clear()
    capture.test = "flash-prefill-mm-self-check"
    probes = {}
    original_qmm, original_gather = mm.qmm, mm.gather_sorted
    count = 0
    def record_probe(index, x, w, s, b):
        probes.update({f"probe{index}.{key}": value for key, value in
                       dict(input=x, weight=w, scales=s, biases=b).items()})
    def record_qmm(x, w, s, b):
        nonlocal count
        record_probe(count, x, w, s, b)
        count += 1
        return original_qmm(x, w, s, b)
    def record_gather(x, w, s, b, ids, tile):
        record_probe(2, x, w, s, b)
        probes["probe2.ids"] = ids
        return original_gather(x, w, s, b, ids, tile)
    mm.qmm, mm.gather_sorted = record_qmm, record_gather
    try:
        self_check = mm._self_check()
    finally:
        mm.qmm, mm.gather_sorted = original_qmm, original_gather
    mx.save_safetensors(str(capture.directory / "self-check.safetensors"), probes)
    cases = []
    def save(mode, x, w, s, b, output, *, ids=None, tile=None, decision=True, bits=4):
        name = f"mm-{len(cases):03}"
        tensors = dict(input=x, weight=w, scales=s, biases=b, output=output)
        if ids is not None:
            tensors["ids"] = ids
        mx.eval(tensors)
        mx.save_safetensors(str(capture.directory / f"{name}.safetensors"), tensors)
        cases.append(dict(name=name, mode=mode, bits=bits, group=x.shape[-1] // s.shape[-1],
                          tile=tile, decision=decision))
    for m, n, k in ((1, 33, 256), (63, 129, 320), (64, 640, 256), (65, 641, 256),
                    (511, 640, 256), (512, 640, 256), (513, 640, 256),
                    (128, 8191, 256), (128, 8192, 256), (129, 8193, 256), (2048, 324, 320)):
        capture.test = f"flash-prefill-qmm-{m}-{n}-{k}"
        x = mx.random.normal((m, k), key=mx.random.key(m)).astype(mx.bfloat16)
        w, s, b = mx.quantize(mx.random.normal((n, k), key=mx.random.key(n)).astype(mx.bfloat16), group_size=32, bits=4)
        save("qmm", x, w, s, b, mm.qmm(x, w, s, b))
        for decision in (False, True):
            mm._tiles[:] = [decision]
            save("matmul", x, w, s, b, mm.matmul(x, w, s, b), decision=decision)
        layer = nn.QuantizedLinear(k, n, bias=False, group_size=32, bits=4)
        layer.update(dict(weight=w, scales=s, biases=b))
        save("linear", x, w, s, b, mm.linear(layer, x))
    for n in (1, 16):
        x = mx.random.normal((2048, 32), key=mx.random.key(n)).astype(mx.bfloat16)
        layer = nn.QuantizedLinear(32, n, bias=False, group_size=32, bits=4)
        w, s, b = mx.quantize(mx.random.normal((n, 32), key=mx.random.key(n + 100)).astype(mx.bfloat16), group_size=32, bits=4)
        layer.update(dict(weight=w, scales=s, biases=b))
        mm._tiles[:] = [True]
        save("linear", x, layer.weight, layer.scales, layer.biases, mm.linear(layer, x))
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            x = mx.random.normal((2, 256, 256), key=mx.random.key(bits)).astype(mx.bfloat16)
            w, s, b = mx.quantize(mx.random.normal((640, 256), key=mx.random.key(group)).astype(mx.bfloat16), group_size=group, bits=bits)
            mm._tiles[:] = [True]
            y = mm.matmul(x.reshape(-1, 256), w, s, b, group=group, bits=bits).reshape(2, 256, 640)
            save("linear", x, w, s, b, y, bits=bits)
    for group in (32, 64, 128):
        for m, experts, n in ((63, 16, 65), (64, 16, 64), (65, 16, 33),
                              (895, 16, 64), (896, 16, 64), (897, 16, 64), (257, 37, 65)):
            k = 256
            capture.test = f"flash-prefill-gather-{group}-{m}-{experts}-{n}"
            x = mx.random.normal((m, k), key=mx.random.key(m)).astype(mx.bfloat16)
            w, s, b = mx.quantize(mx.random.normal((experts, n, k), key=mx.random.key(experts)).astype(mx.bfloat16), group_size=group, bits=4)
            # Empty experts, a dominant expert, and tail experts exercise the tile scan.
            ids = mx.sort(mx.array([0 if i % 3 else experts - 1 if i % 5 else i % experts for i in range(m)], mx.uint32))
            for tile in (None, *mm.SHAPES, (64, 64, 2, 2)):
                save("gather", x, w, s, b, mm.gather_sorted(x, w, s, b, ids, tile), ids=ids, tile=tile)
    mm._tiles.clear()
    (capture.directory / "mm.json").write_text(json.dumps(dict(self_check=self_check, cases=cases), indent=2) + "\n")


def flash_prefill_hc_fixtures(capture):
    from tensorfold.kernels.qwen.flash_next.v1 import base, hc, prefill_hc
    from tests.test_flash_next_affine import quantized, bf16

    rng = np.random.default_rng(82936)
    cases = []
    shapes = []
    for index, (bits, group) in enumerate((b, g) for b in (2, 3, 4, 5, 6, 8) for g in (32, 64, 128)):
        shapes.append((4, 256, 384, (17, 65, 257)[index % 3], bool(index % 2), bool(index % 3),
                       (bits, group), (8 if bits != 8 else 3, 128 if group != 128 else 32)))
    for count in (17, 63, 64, 65, 257, 2048):
        for pending, inject in ((False, False), (True, True)):
            shapes.append((4, 2560, 320, count, pending, inject, (4, 32), (4, 32)))
    for streams in (1, 8):
        for pending, inject in ((False, True), (True, False)):
            shapes.append((streams, 256, 128, 65, pending, inject, (5, 64), (6, 128)))
    for streams, dims, low, count, pending, inject, df, uf in shapes:
        name = f"hyper{len(cases):03}"
        capture.test = name
        wide = streams * dims
        down = quantized(rng, (low + (streams if inject else 0), wide), *df, scale=.02)
        up = quantized(rng, (wide, low), *uf, scale=.02)
        scale = mx.array(1 + .1 * rng.normal(size=wide), mx.float32)
        eps = mx.array([1e-6], mx.float32)
        entry = SimpleNamespace(down=base.QWeights(down.weight, down.scales, down.biases, *df),
                                up=base.QWeights(up.weight, up.scales, up.biases, *uf), scale=scale, low=low)
        h = bf16(rng, (count, wide * 2))[:, ::2]
        branch = bf16(rng, (count, dims))
        gates = bf16(rng, (count, streams))
        residual, ssp = hc.hc_norm(h, streams=streams, **(
            dict(write_back="plain", branch=(branch,), inject=gates) if pending else {}))
        arrays = dict(h=h, branch=branch, pending_inject=gates, scale=scale, eps=eps,
                      residual=residual, ssp=ssp)
        original_qmm = prefill_hc._qmm
        def record_qmm(x, weight):
            result = original_qmm(x, weight)
            arrays["normed" if weight is entry.down else "act"] = x
            arrays["dn" if weight is entry.down else "projected"] = result
            return result
        prefill_hc._qmm = record_qmm
        try:
            written, mixed, inj = prefill_hc.hyper_connection(entry, h, (branch, gates) if pending else None,
                                                            streams=streams, eps=eps)
        finally:
            prefill_hc._qmm = original_qmm
        assert mx.array_equal(written, residual).item()
        assert (inj is not None) == inject
        arrays["mixed"] = mixed
        if inject:
            arrays["inject"] = inj
        arrays.update({f"{key}.{suffix}": getattr(weight, suffix) for key, weight in (("down", down), ("up", up))
                       for suffix in ("weight", "scales", "biases")})
        mx.eval(*arrays.values())
        mx.save_safetensors(str(capture.directory / f"{name}.safetensors"), arrays)
        cases.append(dict(name=name, streams=streams, low=low, pending=pending, inject=inject,
                          down_bits=df[0], down_group=df[1], up_bits=uf[0], up_group=uf[1]))
    (capture.directory / "hyper.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} complete Flash prefill hyper-connections and their intermediate arrays", flush=True)


def deepseek_prefill_moe_fixtures(directory):
    import copy
    from tests import dsv4_fakes as fake
    from tests.test_flash_next_affine import bf16
    from tensorfold.families.deepseek_v4.weights import load_backbone

    rng = np.random.default_rng(91743)
    original_text, original_dims = fake.TEXT, fake.D
    groups = []
    try:
        for geometry, (dims, experts, inner, top, hashed, limit) in enumerate((
            (128, 8, 64, 2, True, 10.), (256, 16, 128, 3, False, 0.),
            (256, 256, 128, 6, False, 1.), (4096, 16, 2048, 6, True, 10.),
            (128, 16, 128, 16, False, .25),
        )):
            fake.TEXT = copy.deepcopy(original_text)
            fake.D = dims
            fake.TEXT.update(hidden_size=dims, num_hidden_layers=1, compress_ratios=[0, 0],
                             n_routed_experts=experts, moe_intermediate_size=inner, num_experts_per_tok=top,
                             num_hash_layers=int(hashed), swiglu_limit=limit)
            checkpoint = f"checkpoint{geometry}"
            folder = fake.write_checkpoint(directory / checkpoint, seed=213 + geometry)
            model = load_backbone(folder)
            layer = model.layers[0].moe
            cases = []
            lengths = (17, 64, 257, 2048) if dims == 4096 else (1, 3, 4, 7, 8, 16, 17, 21, 22, 31, 32, 63, 64, 65, 257, 2048)
            for count in lengths:
                name = f"moe{geometry}-{count}"
                x = bf16(rng, (count, dims * 2), scale=.3)[:, ::2]
                if count == 7:
                    x = mx.zeros_like(x)
                tokens = mx.array([(j * 7 + 3) % fake.VOCAB for j in range(count)], mx.uint32)
                logits = x.astype(mx.float32) @ layer.router
                scores = layer.scores(x, False)
                ids, weights = layer.route(scores, tokens)
                selected = layer.experts(x, ids)
                routed = layer.combine(weights, selected, x.dtype)
                shared = layer.shared(x, False)
                output = layer(x, tokens, False)
                assert mx.array_equal(routed + shared, output).item()
                arrays = dict(input=x, logits=logits, scores=scores, ids=ids, weights=weights,
                              experts=selected, routed=routed, shared=shared, output=output)
                mx.eval(*arrays.values())
                mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
                cases.append(name)
            groups.append(dict(checkpoint=checkpoint, cases=cases))
            print(f"Saved DeepSeek MoE D={dims}, E={experts}, top={top}, hashed={hashed}: {len(cases)} cases", flush=True)
    finally:
        fake.TEXT, fake.D = original_text, original_dims
    (directory / "moe.json").write_text(json.dumps(groups, indent=2) + "\n")


def deepseek_prefill_hc_fixtures(directory):
    import copy
    from tests import dsv4_fakes as fake
    from tests.test_flash_next_affine import bf16
    from tensorfold.families.deepseek_v4.weights import load_backbone
    from tensorfold.families.deepseek_v4.mtp import load as load_mtp
    from tensorfold.families.glm5_next.model import hc_expand

    rng = np.random.default_rng(48013)
    original_text, original_dims, original_hc = fake.TEXT, fake.D, fake._hc
    groups = []
    try:
        for geometry, (dims, packed, eps, hc_eps, iterations) in enumerate((
            (128, False, 1e-6, 1e-6, 20), (256, True, 1e-5, 1e-5, 1),
            (4096, False, 1e-6, 1e-6, 20), (4096, True, 1e-5, 1e-6, 32),
        )):
            fake.TEXT = copy.deepcopy(original_text)
            fake.D = dims
            fake.TEXT.update(hidden_size=dims, num_hidden_layers=1, compress_ratios=[0, 0],
                             rms_norm_eps=eps, hc_eps=hc_eps, hc_sinkhorn_iters=iterations)
            def write_hc(tensors, name, mixes):
                original_hc(tensors, name, mixes)
                if packed:
                    for key in ("fn", "base", "scale"):
                        tensors[f"{name}.{key}"] = tensors[f"{name}.{key}"].astype(mx.bfloat16)
            fake._hc = write_hc
            checkpoint = f"checkpoint{geometry}"
            folder = fake.write_checkpoint(directory / checkpoint, seed=137 + geometry)
            fake.write_mtp(folder / "drafter", seed=141 + geometry)
            model = load_backbone(folder)
            mtp = load_mtp(model, folder / "drafter/model.safetensors")
            cases = []
            for count in ((1, 17, 63, 64, 65, 257, 2048) if dims < 4096 else (1, 17, 64, 257, 2048)):
                name = f"hc{geometry}-{count}"
                streams = bf16(rng, (count, 4, dims * 2))[:, :, ::2]
                branch = bf16(rng, (count, dims), scale=.3)
                if count == 17:
                    streams = mx.zeros_like(streams)
                arrays = dict(input=streams, branch=branch)
                for prefix, block in (("target", model.layers[0]), ("mtp", mtp.block)):
                    for kind in ("attn", "ffn"):
                        hc = getattr(block, f"{kind}_hc")
                        collapsed, post, comb = hc.split(streams, False)
                        key = f"{prefix}-{kind}-"
                        arrays.update({key + "collapsed": collapsed, key + "post": post, key + "comb": comb,
                                       key + "expanded": hc_expand(branch, streams, post, comb, False)})
                    head = model.head_hc if prefix == "target" else mtp.head_hc
                    norm = model.norm if prefix == "target" else mtp.norm
                    arrays[prefix + "-head"] = mx.fast.rms_norm(head(streams, count == 1), norm, eps)
                mx.eval(*arrays.values())
                mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
                cases.append(name)
            groups.append(dict(checkpoint=checkpoint, cases=cases))
            print(f"Saved DeepSeek HC D={dims}, packed={packed}: {len(cases)} cases", flush=True)
    finally:
        fake.TEXT, fake.D, fake._hc = original_text, original_dims, original_hc
    (directory / "hc.json").write_text(json.dumps(groups, indent=2) + "\n")


def glm_prefill_moe_fixtures(directory):
    import copy
    from tests import glm5_fakes as fakes
    from tensorfold.families.glm5_next.weights import load_backbone
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    from tests.test_flash_next_affine import bf16

    rng = np.random.default_rng(41097)
    original_text, original_dims = fakes.TEXT, fakes.D
    original_tiles = list(prefill_mm._tiles)
    groups = []
    try:
        for geometry, (dims, experts, inner, top, shared, norm, limit, mixed, group) in enumerate((
            (128, 8, 64, 2, True, True, 10., False, 64),
            (256, 16, 128, 3, True, False, 0., True, 64),
            (256, 288, 128, 8, False, True, 1., False, 32),
            (4096, 8, 2048, 8, True, True, 10., False, 64),
            (128, 16, 128, 16, False, False, 0., False, 128),
        )):
            cfg = copy.deepcopy(original_text)
            cfg.update(hidden_size=dims, num_hidden_layers=1, layer_types=["linear_attention"],
                       mlp_layer_types=["sparse"], num_nextn_predict_layers=0, n_routed_experts=experts,
                       num_experts_per_tok=top, moe_intermediate_size=inner, n_shared_experts=int(shared),
                       norm_topk_prob=norm, swiglu_limit=limit)
            fakes.TEXT, fakes.D = cfg, dims
            prefix = "model.language_model.layers.0.mlp."
            overrides = {}
            for expert in range(experts):
                for key, fmt in zip(("gate_proj", "up_proj", "down_proj"),
                                    ((2, 32), (3, 64), (8, 128)) if mixed else ((4, group),) * 3):
                    overrides[f"{prefix}experts.{expert}.{key}"] = dict(bits=fmt[0], group_size=fmt[1])
            if mixed:
                for key, fmt in zip(("gate_proj", "up_proj", "down_proj"), ((5, 128), (6, 64), (4, 32))):
                    overrides[f"{prefix}shared_experts.{key}"] = dict(bits=fmt[0], group_size=fmt[1])
            checkpoint = f"checkpoint{geometry}"
            folder = fakes.write_checkpoint(directory / checkpoint, seed=89 + geometry, mtp=False, overrides=overrides)
            if not shared:
                weight_map = {}
                for shard in sorted(folder.glob("*.safetensors")):
                    tensors = {key: value for key, value in mx.load(str(shard)).items()
                               if not key.startswith(prefix + "shared_experts.")}
                    if geometry == 4 and prefix + "gate.e_score_correction_bias" in tensors:
                        tensors[prefix + "gate.e_score_correction_bias"] = mx.zeros((experts,), dtype=mx.float32)
                    mx.save_safetensors(str(shard), tensors)
                    weight_map.update({key: shard.name for key in tensors})
                (folder / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=weight_map)))
            model = load_backbone(folder)
            layer = model.layers[0].mlp
            cases = []
            lengths = (17, 63, 64, 257, 2048) if dims == 4096 else (1, 3, 4, 7, 8, 16, 17, 21, 22, 31, 32, 63, 64, 65, 257, 2048)
            for count in lengths:
                x = bf16(rng, (count, dims), scale=.2)
                if count == 7:
                    x = mx.zeros_like(x)
                for tiles in (False, True):
                    prefill_mm._tiles[:] = [tiles]
                    name = f"moe{geometry}-{count}-{int(tiles)}"
                    logits = x.astype(mx.float32) @ layer.router
                    ids, weights = layer.route(logits)
                    selected = layer.experts(x, ids)
                    routed = layer.combine(weights, selected, x.dtype)
                    arrays = dict(input=x, logits=logits, ids=ids, weights=weights, experts=selected,
                                  routed=routed, output=layer(x, rows_exact=False))
                    expected = routed
                    if layer.shared is not None:
                        arrays["shared"] = layer.shared(x, rows_exact=False)
                        expected = expected + arrays["shared"]
                    assert mx.array_equal(expected, arrays["output"]).item()
                    mx.eval(*arrays.values())
                    mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
                    cases.append(dict(name=name, tiles=tiles))
            groups.append(dict(checkpoint=checkpoint, cases=cases))
            print(f"Saved GLM MoE geometry D={dims}, E={experts}, inner={inner}, {len(cases)} cases", flush=True)
    finally:
        fakes.TEXT, fakes.D = original_text, original_dims
        prefill_mm._tiles[:] = original_tiles
    (directory / "moe.json").write_text(json.dumps(groups, indent=2) + "\n")


def glm_prefill_mla_fixtures(directory):
    import copy
    from tests import glm5_fakes as fakes
    from tensorfold.families.glm5_next import mla as mla_module
    from tensorfold.families.glm5_next.weights import load_backbone
    from tests.test_flash_next_affine import bf16

    rng = np.random.default_rng(81302)
    original_text, original_dims = fakes.TEXT, fakes.D
    groups = []
    try:
        for geometry, (dims, heads, nope, rank, qrank, ih, idim, top, tail, absorbed, mixed) in enumerate((
            (128, 2, 64, 128, 64, 2, 64, 16, True, False, False),
            (256, 3, 128, 256, 128, 3, 128, 16, False, True, True),
            (256, 2, 128, 128, 128, 2, 64, 1024, True, False, True),
            (4096, 64, 256, 512, 1536, 32, 128, 2048, True, False, False),
        )):
            cfg = copy.deepcopy(original_text)
            cfg.update(hidden_size=dims, num_hidden_layers=1, layer_types=["deepseek_sparse_attention"],
                       mlp_layer_types=["dense"], num_nextn_predict_layers=0, num_attention_heads=heads,
                       qk_nope_head_dim=nope, v_head_dim=nope, kv_lora_rank=rank, q_lora_rank=qrank,
                       index_n_heads=ih, index_head_dim=idim, index_topk=top, index_kpool_always_select_tail=tail)
            fakes.TEXT, fakes.D = cfg, dims
            checkpoint = f"checkpoint{geometry}"
            names = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "indexer.wq_b", "indexer.wk",
                     "indexer.weights_proj", "kv_b_proj", "o_proj")
            formats = ((2, 32), (3, 64), (4, 128), (5, 32), (6, 64), (8, 128), (4, 128), (5, 128))
            overrides = {f"model.language_model.layers.0.self_attn.{key}": dict(bits=b, group_size=g)
                         for key, (b, g) in zip(names, formats)} if mixed else {}
            folder = fakes.write_checkpoint(directory / checkpoint, seed=61 + geometry, mtp=False, overrides=overrides)
            if absorbed:
                tensors = {}
                shards = sorted(folder.glob("*.safetensors"))
                for shard in shards:
                    tensors.update(mx.load(str(shard)))
                prefix = "model.language_model.layers.0.self_attn."
                fmt = overrides.get(prefix + "kv_b_proj", dict(bits=4, group_size=64))
                w = mx.dequantize(*(tensors.pop(prefix + "kv_b_proj." + key) for key in ("weight", "scales", "biases")), **fmt)
                w = w.reshape(heads, 2 * nope, rank)
                for key, value, bits in (("embed_q", mx.contiguous(w[:, :nope].swapaxes(-1, -2)), 5),
                                         ("unembed_out", mx.contiguous(w[:, nope:]), 6)):
                    overrides[prefix + key] = dict(bits=bits, group_size=128)
                    tensors.update(zip((prefix + key + "." + field for field in ("weight", "scales", "biases")),
                                       mx.quantize(value, bits=bits, group_size=128)))
                names = sorted(tensors)
                mid = len(names) // 2
                weight_map = {}
                for shard, keys in zip(shards, (names[:mid], names[mid:])):
                    mx.save_safetensors(str(shard), {key: tensors[key] for key in keys})
                    weight_map.update({key: shard.name for key in keys})
                (folder / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=weight_map)))
                config = json.loads((folder / "config.json").read_text())
                config["quantization"].update(overrides)
                (folder / "config.json").write_text(json.dumps(config))
            model = load_backbone(folder)
            attn = model.layers[0].attn
            cache = model.make_cache()[0]
            cases = []
            lengths = ((2047, 17, 513, 1) if dims == 4096 else (512, 513, 17, 1) if top == 1024 else
                       (17, 15, 1, 17, 63, 64, 65, 511, 512, 513, 2048, 1) if geometry == 0 else
                       (1, 15, 1, 17, 63, 64, 65, 511, 512, 513, 2048, 1))
            for step, count in enumerate(lengths):
                decode = step == len(lengths) - 1
                name = f"mla{geometry}-{step}"
                x = bf16(rng, (count, dims), scale=.2)
                if step == 3 and dims < 4096:
                    x = mx.zeros_like(x)  # Tied index scores must preserve partition ordering.
                arrays = dict(input=x)
                def save_cache(prefix):
                    for key in ("keys", "ik", "ig", "pool"):
                        value = getattr(cache, key)
                        if value is not None:
                            end = cache.offset // 4 if key == "pool" else cache.offset
                            arrays[prefix + key] = value[:end]
                save_cache("previous.")
                chunks = []
                original_prefill, original_absorb = attn._prefill, attn.absorb
                original_scores = attn.index_scores
                original_take, original_softmax, original_matmul = mx.take, mx.softmax, mx.matmul
                original_attention, original_project = mx.fast.scaled_dot_product_attention, mla_module.project
                def record_prefill(q, iq, iw, *args):
                    arrays.update(q=q, iq=iq, iw=iw)
                    return original_prefill(q, iq, iw, *args)
                def record_absorb(q):
                    out = original_absorb(q)
                    arrays["ql"] = out
                    return out
                def record_scores(*args):
                    out = original_scores(*args)
                    chunks.append(dict(scores=out))
                    return out
                def record_take(value, indices, axis=None, **kwargs):
                    if chunks and axis == 0 and value.ndim == 2 and value.shape[1] == rank:
                        chunks[-1]["safe_ids"] = indices.astype(mx.int32)
                    return original_take(value, indices, axis=axis, **kwargs)
                def record_softmax(value, *args, **kwargs):
                    out = original_softmax(value, *args, **kwargs)
                    if chunks and kwargs.get("precise"):
                        chunks[-1].update(attention_scores=value, probabilities=out)
                    return out
                def record_matmul(left, right, *args, **kwargs):
                    out = original_matmul(left, right, *args, **kwargs)
                    if chunks and left is chunks[-1].get("probabilities"):
                        chunks[-1]["output"] = out
                    return out
                def record_attention(*args, **kwargs):
                    out = original_attention(*args, **kwargs)
                    chunks.append(dict(output=out[0].transpose(1, 0, 2)))
                    return out
                def record_project(value, weight, **kwargs):
                    if weight is attn.o_proj:
                        arrays["flat"] = value
                    return original_project(value, weight, **kwargs)
                if not decode:
                    attn._prefill, attn.absorb, attn.index_scores = record_prefill, record_absorb, record_scores
                    mx.take, mx.softmax, mx.matmul = record_take, record_softmax, record_matmul
                    mx.fast.scaled_dot_product_attention, mla_module.project = record_attention, record_project
                try:
                    arrays["output"] = attn(x, [cache], (count,), decode)
                finally:
                    attn._prefill, attn.absorb, attn.index_scores = original_prefill, original_absorb, original_scores
                    mx.take, mx.softmax, mx.matmul = original_take, original_softmax, original_matmul
                    mx.fast.scaled_dot_product_attention, mla_module.project = original_attention, original_project
                save_cache("next.")
                for chunk, values in enumerate(chunks):
                    arrays.update({f"chunk{chunk}.{key}": value for key, value in values.items()})
                mx.eval(*arrays.values())
                mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
                cases.append(dict(name=name, decode=decode, chunks=len(chunks)))
            groups.append(dict(checkpoint=checkpoint, cases=cases))
            print(f"Saved GLM MLA geometry D={dims}, H={heads}, rank={rank}, {len(cases)} cases", flush=True)
    finally:
        fakes.TEXT, fakes.D = original_text, original_dims
    (directory / "mla.json").write_text(json.dumps(groups, indent=2) + "\n")


def glm_prefill_kda_fixtures(directory):
    import copy
    from tests import glm5_fakes as fakes
    from tensorfold.families.glm5_next import kda as kda_module
    from tensorfold.families.glm5_next.model import hc_expand
    from tensorfold.families.glm5_next.weights import load_backbone
    from tests.test_flash_next_affine import bf16

    rng = np.random.default_rng(53147)
    original_text, original_dims = fakes.TEXT, fakes.D
    groups = []
    try:
        for geometry, (dims, heads, dim, mixed) in enumerate(((128, 2, 64, False), (256, 3, 128, True),
                                                            (4096, 64, 128, False))):
            cfg = copy.deepcopy(original_text)
            cfg.update(hidden_size=dims, num_hidden_layers=1, layer_types=["linear_attention"],
                       mlp_layer_types=["dense"], num_nextn_predict_layers=0)
            cfg["linear_attn_config"].update(num_heads=heads, head_dim=dim)
            cfg["hc_sinkhorn_iters"] = 8 if mixed else 20
            cfg["hc_eps"] = 1e-5 if mixed else 1e-6
            cfg["rms_norm_eps"] = 1e-6 if dims == 4096 else 1e-5
            fakes.TEXT, fakes.D = cfg, dims
            checkpoint = f"checkpoint{geometry}"
            names = ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj", "f_b_proj", "g_b_proj", "o_proj")
            formats = ((2, 32), (3, 64), (4, 128), (5, 32), (6, 64), (8, 128), (3, 128), (6, 64), (5, 128))
            overrides = {f"model.language_model.layers.0.self_attn.{key}": dict(bits=b, group_size=g)
                         for key, (b, g) in zip(names, formats)} if mixed else {}
            folder = fakes.write_checkpoint(directory / checkpoint, seed=19 + geometry, mtp=False, overrides=overrides)
            model = load_backbone(folder)
            layer = model.layers[0]
            cache = model.make_cache()[0]
            cases = []
            lengths = (17, 64, 257, 1) if dims == 4096 else (1, 16, 17, 31, 32, 63, 64, 65, 257, 2048, 1)
            for step, count in enumerate(lengths):
                decode = step == len(lengths) - 1
                name = f"kda{geometry}-{step}"
                x = bf16(rng, (count, 4, dims), scale=.2)
                collapsed, post, comb = layer.attn_hc.split(x, rows_exact=decode)
                normed = mx.fast.rms_norm(collapsed, layer.in_norm, layer.eps)
                arrays = dict(input=x, collapsed=collapsed, post=post, comb=comb, normed=normed)
                if cache.ssm is not None:
                    arrays.update({"previous.conv": cache.conv, "previous.state": cache.ssm})
                original_delta = kda_module.K.gated_delta
                original_project, original_silu = kda_module.project, kda_module.silu
                def record_delta(q, k, v, g, beta, entry):
                    y, state = original_delta(q, k, v, g, beta, entry)
                    arrays.update(q=q, k=k, value=v, decay=g, beta=beta, recurrent=y)
                    return y, state
                def record_project(value, weight, **kwargs):
                    out = original_project(value, weight, **kwargs)
                    if weight is layer.attn.in_proj:
                        arrays["projection"] = out
                    if weight is layer.attn.o_proj:
                        arrays["gated"] = value
                    return out
                def record_silu(value):
                    out = original_silu(value)
                    arrays["convolved"] = out
                    return out
                kda_module.K.gated_delta = record_delta
                kda_module.project, kda_module.silu = record_project, record_silu
                try:
                    output = layer.attn(normed, [cache], (count,), decode)
                finally:
                    kda_module.K.gated_delta = original_delta
                    kda_module.project, kda_module.silu = original_project, original_silu
                arrays.update(output=output)
                arrays.update({"next.conv": cache.conv, "next.state": cache.ssm})
                expanded = hc_expand(output, x, post, comb, rows_exact=decode)
                arrays["expanded"] = expanded
                ffn = layer.ffn_hc.split(expanded, rows_exact=decode)
                arrays.update(zip(("ffn.collapsed", "ffn.post", "ffn.comb"), ffn))
                mx.eval(*arrays.values())
                mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
                cases.append(dict(name=name, decode=decode))
            groups.append(dict(checkpoint=checkpoint, cases=cases))
            print(f"Saved GLM KDA/HC geometry D={dims}, H={heads}, d={dim}, {len(cases)} cases", flush=True)
    finally:
        fakes.TEXT, fakes.D = original_text, original_dims
    (directory / "kda.json").write_text(json.dumps(groups, indent=2) + "\n")


def flash_prefill_gdn_fixtures(capture):
    import mlx.nn as nn
    from tensorfold.families.qwen4_exp import model_layers, decode
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    from tests.test_flash_next_affine import quantized, bf16

    rng = np.random.default_rng(10629)
    cases = []
    for hidden, nk, nv, dk, dv, taps, batch, activation, bits, groups in (
        (256, 2, 4, 32, 64, 4, 2, "sigmoid", (4,) * 5, (32, 64, 128, 32, 64)),
        (384, 3, 6, 64, 32, 2, 1, "silu", (2, 3, 5, 6, 8), (128, 64, 32, 128, 64)),
        (2560, 16, 48, 128, 128, 4, 1, "sigmoid", (4,) * 5, (32,) * 5),
    ):
        cfg = SimpleNamespace(linear_num_key_heads=nk, linear_num_value_heads=nv,
                              linear_key_head_dim=dk, linear_value_head_dim=dv, linear_conv_kernel_dim=taps,
                              hidden_size=hidden, rms_norm_eps=1e-6, output_gate_type=activation)
        layer = model_layers.GatedDeltaNet(cfg)
        names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
        for name, b, group in zip(names, bits, groups):
            shape = getattr(layer, name).weight.shape
            weight = quantized(rng, shape, b, group, scale=.02)
            linear = nn.QuantizedLinear(shape[1], shape[0], bias=False, group_size=group, bits=b)
            linear.weight, linear.scales, linear.biases = weight.weight, weight.scales, weight.biases
            setattr(layer, name, linear)
        layer.conv1d.weight = bf16(rng, layer.conv1d.weight.shape, scale=.2)
        layer.A_log = mx.array(rng.uniform(-2, 0, nv), mx.bfloat16)
        layer.dt_bias = mx.array(rng.uniform(-2, 0, nv), mx.bfloat16)
        layer.norm.weight = mx.array(rng.uniform(.8, 1.2, dv), mx.bfloat16)
        stacked, _ = decode._stacked([getattr(layer, name) for name in names[:4]])
        layer.__dict__["stacked"] = stacked if isinstance(stacked, nn.QuantizedLinear) else None
        cache = model_layers.LinearCache()
        for count in (1, 17, 31, 32, 63, 64, 65, 257, 2048, 1):
            name = f"gdn{len(cases):03}"
            capture.test = name
            x = bf16(rng, (batch, count, hidden))
            arrays = {"input": x, "conv": layer.conv1d.weight, "a_log": layer.A_log,
                      "dt_bias": layer.dt_bias, "norm": layer.norm.weight}
            cached = cache.ssm is not None
            if cached:
                arrays.update({"previous.conv": cache.conv, "previous.state": cache.ssm})
            original_update, original_linear = model_layers.gated_delta_update, prefill_mm.linear
            def record_update(q, k, v, *args, **kwargs):
                out, state = original_update(q, k, v, *args, **kwargs)
                arrays.update(q=q, k=k, v=v, recurrent=out)
                return out, state
            def record_linear(projection, value):
                if projection is layer.out_proj:
                    arrays["gated"] = value
                return original_linear(projection, value)
            model_layers.gated_delta_update, prefill_mm.linear = record_update, record_linear
            try:
                arrays["output"] = layer(x, cache)
            finally:
                model_layers.gated_delta_update, prefill_mm.linear = original_update, original_linear
            arrays.update({"next.conv": cache.conv, "next.state": cache.ssm})
            for key, part in zip(("qkv", "z", "b", "a", "out"), names):
                arrays.update({f"{key}.{suffix}": getattr(getattr(layer, part), suffix)
                               for suffix in ("weight", "scales", "biases")})
            mx.eval(*arrays.values())
            mx.save_safetensors(str(capture.directory / f"{name}.safetensors"), arrays)
            cases.append(dict(name=name, config=dict(key_heads=nk, value_heads=nv, key_dims=dk, value_dims=dv,
                                                     activation=activation, epsilon=1e-6),
                              cached=cached, bits=bits, groups=groups, stacked=layer.__dict__["stacked"] is not None))
    (capture.directory / "gdn.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} complete Flash GDN layers, batched projections and cache continuation", flush=True)


def flash_prefill_moe_fixtures(directory):
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from tensorfold.families.qwen4_exp.model_layers import SparseMoE
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    from tests.test_flash_next_affine import quantized, bf16

    rng = np.random.default_rng(74013)
    cases = []
    for geometry, (dims, inner, shared_inner, experts, top, batch, bits, groups) in enumerate((
        (256, 128, 128, 32, 4, 1, (4,) * 7, (32,) * 7),
        (384, 192, 128, 16, 3, 2, (4, 5, 6, 2, 3, 8, 5), (64,) * 7),
        (2560, 640, 640, 32, 10, 1, (4,) * 7, (32,) * 7),
        (256, 128, 128, 512, 10, 1, (4,) * 7, (32,) * 7),
        (128, 128, 128, 64, 64, 1, (4,) * 7, (32,) * 7),
    )):
        cfg = SimpleNamespace(hidden_size=dims, num_experts=experts, num_experts_per_tok=top,
                              moe_intermediate_size=inner, shared_expert_intermediate_size=shared_inner)
        layer = SparseMoE(cfg)
        layer.gate.weight = bf16(rng, (experts, dims), scale=.02)
        parts = ((layer.switch_mlp, "gate_proj"), (layer.switch_mlp, "up_proj"), (layer.switch_mlp, "down_proj"),
                 (layer.shared_expert, "gate_proj"), (layer.shared_expert, "up_proj"),
                 (layer.shared_expert, "down_proj"), (layer, "shared_expert_gate"))
        weights = {"router": layer.gate.weight}
        keys = ("gate", "up", "down", "shared_gate", "shared_up", "shared_down", "shared_route")
        for (module, attr), key, b, group in zip(parts, keys, bits, groups):
            shape = getattr(module, attr).weight.shape
            weight = quantized(rng, shape, b, group, scale=.02)
            if len(shape) == 3:
                linear = QuantizedSwitchLinear(shape[2], shape[1], shape[0], bias=False, group_size=group, bits=b)
            else:
                linear = nn.QuantizedLinear(shape[1], shape[0], bias=False, group_size=group, bits=b)
            linear.weight, linear.scales, linear.biases = weight.weight, weight.scales, weight.biases
            setattr(module, attr, linear)
            weights.update({f"{key}.{suffix}": getattr(linear, suffix) for suffix in ("weight", "scales", "biases")})
        weight_name = f"moe_weights{geometry}"
        mx.eval(*weights.values())
        mx.save_safetensors(str(directory / f"{weight_name}.safetensors"), weights)
        lengths = (511, 512, 513) if top == 64 else (1, 7, 16, 17, 63, 64, 65, 257, 2048)
        for count in lengths:
            x = bf16(rng, (batch, count, dims))
            if count == 7:
                x = mx.zeros_like(x)  # Equal router probabilities exercise partition ties.
            output = layer(x)
            ids, probs = layer.route(x)
            if prefill_mm.moe_applies(layer, x):
                flat = ids.reshape(-1)
                order = mx.argsort(flat)
                pos = mx.argsort(order).astype(mx.int32)
                index = flat[order].astype(mx.uint32)
                rows = x.reshape(count, dims)[order // top]
                sw = layer.switch_mlp
                gate = prefill_mm._experts(rows, sw.gate_proj, index)
                up = prefill_mm._experts(rows, sw.up_proj, index)
                y = prefill_mm._experts(sw.activation(up, gate), sw.down_proj, index)
                routed = (y[pos].reshape(batch, count, top, dims) * probs[..., None]).sum(axis=-2)
                xf = x.reshape(count, dims)
                se = layer.shared_expert
                shared = prefill_mm.linear(se.down_proj, nn.silu(prefill_mm.linear(se.gate_proj, xf)) * prefill_mm.linear(se.up_proj, xf))
                shared = (shared * mx.sigmoid(prefill_mm.linear(layer.shared_expert_gate, xf))).reshape(x.shape)
            else:
                routed = (layer.switch_mlp(x, ids) * probs[..., None]).sum(axis=-2)
                shared = layer.shared_expert(x) * mx.sigmoid(layer.shared_expert_gate(x))
            assert mx.array_equal(output, routed + shared).item()
            name = f"moe{len(cases):03}"
            arrays = dict(input=x, output=output, ids=ids, weights=probs, routed=routed, shared=shared)
            mx.eval(*arrays.values())
            mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
            cases.append(dict(name=name, weights=weight_name, top=top, bits=bits, groups=groups))
    (directory / "moe.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} complete Flash MoE layers, routing, shared experts and sorted-gather boundaries", flush=True)


def flash_prefill_attention_fixtures(capture):
    import mlx.nn as nn
    from tensorfold.families.qwen4_exp.model_layers import SparseAttention, AttentionCache
    from tensorfold.kernels.qwen.flash_next.v1 import prefill, prefill_mm
    from tests.test_flash_next_affine import quantized, bf16

    rng = np.random.default_rng(29516)
    cases = []
    for geometry, (hidden, heads, kvh, dims, ih, idim, top, bits, groups) in enumerate((
        (256, 4, 1, 64, 2, 64, 8, (4, 5, 6, 3, 8), (32,) * 5),
        (2560, 24, 2, 256, 4, 128, 512, (4,) * 5, (32,) * 5),
        (128, 6, 2, 256, 3, 64, 8, (4,) * 5, (32,) * 5),
    )):
        rotary = 32 if dims == 64 else 64
        cfg = SimpleNamespace(hidden_size=hidden, num_attention_heads=heads, num_key_value_heads=kvh,
                              head_dim=dims, rotary_dim=rotary, rope_theta=10000000, rms_norm_eps=1e-6,
                              indexer_n_heads=ih, indexer_head_dim=idim, indexer_compress_ratio=4, indexer_budget=4*top)
        layer = SparseAttention(cfg)
        parts = ((layer, "q_proj"), (layer, "k_proj"), (layer, "v_proj"), (layer, "o_proj"), (layer.indexer, "index_qk_proj"))
        weights = {}
        for (module, attr), key, b, group in zip(parts, ("q", "k", "v", "out", "index"), bits, groups):
            shape = getattr(module, attr).weight.shape
            w = quantized(rng, shape, b, group, scale=.02)
            linear = nn.QuantizedLinear(shape[1], shape[0], bias=False, group_size=group, bits=b)
            linear.weight, linear.scales, linear.biases = w.weight, w.scales, w.biases
            setattr(module, attr, linear)
            weights.update({f"{key}.{suffix}": getattr(linear, suffix) for suffix in ("weight", "scales", "biases")})
        for key, norm in (("q_scale", layer.q_norm), ("k_scale", layer.k_norm),
                          ("iq_scale", layer.indexer.q_layernorm), ("ik_scale", layer.indexer.k_layernorm)):
            norm.weight = bf16(rng, norm.weight.shape, scale=.1)
            weights[key] = 1 + norm.weight.astype(mx.float32)
        weight_name = f"attention_weights{geometry}"
        mx.eval(*weights.values())
        mx.save_safetensors(str(capture.directory / f"{weight_name}.safetensors"), weights)
        for past, count, kernel_select, hs in (
            (0, 17, True, 1), (0, 2048, True, 1), (2032, 17, True, 1), (2048, 1, True, 1),
            (2048, 64, True, 1), (4095, 1, True, 1), (4095, 17, True, 1), (4095, 17, True, 2),
            (4096, 1, True, 1), (4096, 65, True, 2), (8192, 257, False, 1), (8192, 257, True, 1),
        ):
            name = f"attention{len(cases):03}"
            capture.test = name
            layer.__dict__["kernel_select"] = kernel_select
            cache = AttentionCache()
            cache.offset = past
            x = bf16(rng, (1, count, hidden))
            arrays = dict(input=x)
            if past:
                cache.keys = bf16(rng, (1, kvh, past, dims))
                cache.values = bf16(rng, cache.keys.shape)
                cache.index_keys = bf16(rng, (1, past, idim))
                if past // 4 > top:
                    layer.indexer.pool(cache.index_keys, cache, past // 4 - 1)
                arrays.update({"previous.keys": cache.keys, "previous.values": cache.values, "previous.raw": cache.index_keys})
                if cache.pooled is not None:
                    arrays["previous.pooled"] = cache.pooled
            previous_pooled = cache.pooled is not None
            qg = prefill_mm.linear(layer.q_proj, x).reshape(1, count, heads, 2*dims)
            arrays["queries"] = mx.fast.rope(layer.q_norm(qg[..., :dims]).transpose(0, 2, 1, 3), rotary,
                                              traditional=False, base=10000000, scale=1.0, offset=past)
            arrays["index_queries"] = layer.indexer.project(x)[0]
            selected, sdpa, heads_fn = prefill.selected, mx.fast.scaled_dot_product_attention, prefill.heads_a_simdgroup
            chunks = []
            def record_selected(*args, **kwargs):
                out = selected(*args, **kwargs)
                arrays["attended"] = out
                return out
            def record_sdpa(*args, **kwargs):
                out = sdpa(*args, **kwargs)
                chunks.append(out)
                return out
            prefill.selected = record_selected
            mx.fast.scaled_dot_product_attention = record_sdpa
            prefill.heads_a_simdgroup = lambda: (hs, 1)
            try:
                arrays["output"] = layer(x, cache)
            finally:
                prefill.selected, mx.fast.scaled_dot_product_attention, prefill.heads_a_simdgroup = selected, sdpa, heads_fn
            if chunks:
                arrays["attended"] = mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(1, count, heads*dims)
            arrays.update({"next.keys": cache.keys[:, :, :cache.offset], "next.values": cache.values[:, :, :cache.offset],
                           "next.raw": cache.index_keys[:, :cache.offset]})
            if cache.pooled is not None:
                arrays["next.pooled"] = cache.pooled
            mx.eval(*arrays.values())
            mx.save_safetensors(str(capture.directory / f"{name}.safetensors"), arrays)
            cases.append(dict(name=name, weights=weight_name, config=dict(heads=heads, kv_heads=kvh, dims=dims,
                              rotary_dims=rotary, index_heads=ih, index_dims=idim, top=top,
                              kernel_select=kernel_select, heads_per_simdgroup=hs), past=past,
                              previous_pooled=previous_pooled, pooled=cache.pooled is not None, bits=bits, groups=groups))
    (capture.directory / "attention.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} complete Flash prefill attention layers and sparse/cache boundary cases", flush=True)


def flash_prefill_ple_fixtures(directory):
    import math
    import mlx.nn as nn
    from tensorfold.families.qwen4_exp.model import PLELayer, LinearCache
    from tests.test_flash_next_affine import quantized, bf16

    rng = np.random.default_rng(88512)
    cases = []
    for geometry, (streams, dims, taps, dilation, batch, bits, groups) in enumerate((
        (1, 256, 2, 2, 2, (3, 5), (64, 32)),
        (4, 2560, 4, 3, 1, (4, 4), (32, 32)),
        (8, 256, 4, 3, 1, (8, 6), (128, 64)),
    )):
        cfg = SimpleNamespace(hc_count=streams, hidden_size=dims, ple_embed_dim=dims, rms_norm_eps=1e-6,
                              ngram_size=dilation, ple_conv_kernel_size=taps, heads_per_ngram=8, ple_eos=31,
                              ngram_vocab_size_base=128, ngram_vocab_divisor=128, ngram_shards=8,
                              vocab_size=256, seed=1234, group_size=32, bits=4)
        layer = PLELayer(cfg, 0)
        for shard in layer.ple_embedding.shards:
            shard.weight = bf16(rng, shard.weight.shape)
        weights = {}
        for attr, key, b, group in zip(("key_proj", "value_proj"), ("key", "value"), bits, groups):
            shape = getattr(layer, attr).weight.shape
            w = quantized(rng, shape, b, group, scale=.02)
            linear = nn.QuantizedLinear(shape[1], shape[0], bias=False, group_size=group, bits=b)
            linear.weight, linear.scales, linear.biases = w.weight, w.scales, w.biases
            setattr(layer, attr, linear)
            weights.update({f"{key}.{suffix}": getattr(linear, suffix) for suffix in ("weight", "scales", "biases")})
        for key, norm in (("key_scale", layer.norm_key), ("query_scale", layer.norm_query), ("conv_scale", layer.norm_conv)):
            norm.weight = bf16(rng, norm.weight.shape, scale=.1)
            weights[key] = 1 + norm.weight.astype(mx.float32)
        layer.conv1d.weight = bf16(rng, layer.conv1d.weight.shape, scale=.1)
        weights["conv"] = layer.conv1d.weight
        weight_name = f"ple_weights{geometry}"
        mx.eval(*weights.values())
        mx.save_safetensors(str(directory / f"{weight_name}.safetensors"), weights)
        cache = LinearCache()
        for count in (1, 17, 63, 64, 65, 257, 2048, 1):
            name = f"ple{len(cases):03}"
            h = bf16(rng, (batch, count, streams*dims))
            tokens = rng.integers(0, 256, size=(batch, count))
            tokens[:, count // 2] = cfg.ple_eos
            history = cache.history if cache.history is not None else np.full((batch, dilation-1), cfg.ple_eos, np.int64)
            embedding = layer.ple_embedding(layer.ple_embedding.ids(history, tokens))
            arrays = dict(h=h, embedding=embedding)
            cached = cache.ple_conv is not None
            if cached:
                arrays["previous"] = cache.ple_conv
            shape = (batch, count, streams, dims)
            key = layer.norm_key(layer.key_proj(embedding)).reshape(shape)
            query = layer.norm_query(h).reshape(shape)
            gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(dims)
            gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
            gated = (mx.sigmoid(gate) * layer.value_proj(embedding)[..., None, :]).reshape(h.shape)
            arrays.update(gated=gated, normed=layer.norm_conv(gated), branch=layer(h, tokens, cache), tail=cache.ple_conv)
            mx.eval(*arrays.values())
            mx.save_safetensors(str(directory / f"{name}.safetensors"), arrays)
            cases.append(dict(name=name, weights=weight_name, streams=streams, dilation=dilation,
                              cached=cached, bits=bits, groups=groups))
    (directory / "ple.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} Flash PLE projection/gating/convolution cases with long-prompt continuation", flush=True)


def flash_affine_variants(capture):
    from tensorfold.kernels.qwen.flash_next.v1 import base, rows, hc, experts, embed
    from tests.test_flash_next_affine import quantized, bf16, same
    rng = np.random.default_rng(91721)
    from types import SimpleNamespace
    for bits in (2, 3, 5, 6, 8):
        parts = [quantized(rng, (3, 160), bits, 32) for _ in range(8)]
        table = SimpleNamespace(host=None, bits=bits, group=32, dims=160,
                                starts=mx.arange(8, dtype=mx.uint32) * 3,
                                weights=[p.weight for p in parts], scales=[p.scales for p in parts], biases=[p.biases for p in parts])
        capture.test = f"flash-ple-lookup-{bits}"
        mx.eval(embed.ple_lookup(np.arange(48).reshape(3, 16) % 24, table))
    for streams in (1, 4, 8):
        for dims in (256, 2560):
            wide = streams * dims
            for count in (1, 4, 16):
                capture.test = f"flash-ple-{streams}-{dims}-{count}"
                h = bf16(rng, (count, wide))
                kv = bf16(rng, (count, wide + dims))
                scales = [mx.array(1 + .1 * rng.normal(size=wide), mx.float32) for _ in range(3)]
                gated, normed = embed.ple_gate(kv, h, *scales, mx.array([1e-6], mx.float32), streams=streams)
                for taps in (1, 4):
                    for dilation in (1, 3):
                        tail = bf16(rng, (dilation * (taps - 1), wide))
                        weight = mx.array(.1 * rng.normal(size=(wide, taps)), mx.float32)
                        mx.eval(embed.ple_conv(mx.concatenate([tail, normed]), weight, gated, h,
                                              streams=streams, dilation=dilation))
    generation = base._generation
    try:
        for gen in (13, 15, 17):
            base._generation = lambda: gen
            for fmt in ([(4, 32)] if gen != 17 else [(b, g) for b in (2, 3, 4, 5, 6, 8) for g in (32, 64, 128)]):
                for n in ((320,) if fmt == (4, 32) else (320, 322, 321)):
                    weight = quantized(rng, (n, 2560), *fmt)
                    x = bf16(rng, (65, 2560))
                    for count in (1, 2, 3, 16, 17, 32, 33, 65):
                        capture.test = f"flash-affine-project-{gen}-{fmt}-{n}-{count}"
                        result = rows.qmv_rows(x[:count], weight)
                        one = rows.qmv_rows(x[:1], weight)
                        assert same(result[:1], one)
            for fmt, up_fmt in ([( (4, 32), (4, 32) )] if gen != 17 else [
                ((b, g), (b, g)) for b in (2, 3, 4, 5, 6, 8) for g in (32, 64)
            ] + [((5, 128), (6, 64)), ((4, 32), (8, 64))]):
                for inject in (False, True):
                    down = quantized(rng, (320 + (4 if inject else 0), 10240), *fmt, scale=.02)
                    up = quantized(rng, (10240, 320), *up_fmt)
                    scale = mx.array(1 + .1 * rng.normal(size=10240), mx.float32)
                    eps = mx.array([1e-6], mx.float32)
                    down_q = base.QWeights(down.weight, down.scales, down.biases, *fmt)
                    up_q = base.QWeights(up.weight, up.scales, up.biases, *up_fmt)
                    h, ssp = hc.hc_norm(bf16(rng, (16, 10240)), streams=4)
                    capture.extra = {"down_weight": down.weight, "down_scales": down.scales, "down_biases": down.biases,
                                     "down_format": mx.array(fmt, mx.int32)}
                    for count in (1, 2, 3, 8, 16):
                        capture.test = f"flash-affine-hc-{gen}-{fmt}-{up_fmt}-{inject}-{count}"
                        mixed, inj = rows.hc_project(h[:count], ssp[:count], down_q, up_q, scale, eps=eps, streams=4, low=320)
                        mx.eval(mixed)
                        if inject:
                            mx.eval(inj)
                    capture.extra = {}
            formats = [((4, 32), (4, 32))] if gen != 17 else [
                ((b, g), (8 if b != 8 else 3, 128 if g != 128 else 32))
                for b in (2, 3, 4, 5, 6, 8) for g in (32, 64, 128)]
            for fmt, shared_fmt in formats:
                e, k, n, top = 32, 512, 128, 4
                gate, up = (quantized(rng, (e, n, k), *fmt) for _ in range(2))
                down = quantized(rng, (e, k, n), *fmt)
                sg, su = (quantized(rng, (n, k), *shared_fmt) for _ in range(2))
                sd = quantized(rng, (k, n), *shared_fmt)
                x = bf16(rng, (16, k))
                logits = mx.array(rng.normal(size=(16, e + 1)), mx.float32)
                for count in (1, 2, 3, 16):
                    for shared in (None, (sg, su)):
                        capture.test = f"flash-affine-experts-{gen}-{fmt}-{shared_fmt}-{count}-{shared is not None}"
                        act, picks, weights = experts.expert_gateup(x[:count], logits[:count, :e + int(shared is not None)], top, e, gate, up, shared)
                        mx.eval(act, picks, weights)
                        if shared:
                            mx.eval(experts.expert_down_y(act, picks, down, sd))
    finally:
        base._generation = generation
        capture.extra = {}


def main():
    require_mlx()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--affine-only", action="store_true")
    parser.add_argument("--tensor-quantization", action="store_true")
    parser.add_argument("--bonsai-only", action="store_true")
    parser.add_argument("--gemma-only", action="store_true")
    parser.add_argument("--large-families", action="store_true")
    parser.add_argument("--row-attention", action="store_true")
    parser.add_argument("--simd-dense", action="store_true")
    parser.add_argument("--simd-bits", action="store_true")
    parser.add_argument("--flash-affine", action="store_true")
    parser.add_argument("--flash-weights", action="store_true")
    parser.add_argument("--flash-prefill-hc", action="store_true")
    parser.add_argument("--flash-prefill-mm", action="store_true")
    parser.add_argument("--flash-prefill-gdn", action="store_true")
    parser.add_argument("--glm-prefill-kda", action="store_true")
    parser.add_argument("--glm-prefill-mla", action="store_true")
    parser.add_argument("--glm-prefill-moe", action="store_true")
    parser.add_argument("--deepseek-prefill-hc", action="store_true")
    parser.add_argument("--deepseek-prefill-moe", action="store_true")
    parser.add_argument("--flash-prefill-moe", action="store_true")
    parser.add_argument("--flash-prefill-attention", action="store_true")
    parser.add_argument("--flash-prefill-ple", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.deepseek_prefill_moe:
        deepseek_prefill_moe_fixtures(args.directory)
        return
    if args.deepseek_prefill_hc:
        deepseek_prefill_hc_fixtures(args.directory)
        return
    if args.glm_prefill_moe:
        glm_prefill_moe_fixtures(args.directory)
        return
    if args.glm_prefill_mla:
        glm_prefill_mla_fixtures(args.directory)
        return
    if args.glm_prefill_kda:
        glm_prefill_kda_fixtures(args.directory)
        return
    if args.flash_weights:
        flash_weight_fixtures(args.directory)
        return
    if args.flash_prefill_moe:
        flash_prefill_moe_fixtures(args.directory)
        return
    if args.flash_prefill_ple:
        flash_prefill_ple_fixtures(args.directory)
        return
    if args.simd_dense:
        simd_dense_fixtures(args.directory)
        return
    if args.simd_bits:
        simd_bits_fixtures(args.directory)
        return
    capture = Capture(args.directory)
    mx.fast.metal_kernel = capture.kernel
    if args.flash_prefill_mm:
        try:
            flash_prefill_mm_fixtures(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        required = {"flash_prefill_qmm", "flash_prefill_gather", "flash_prefill_offsets"}
        if required - {case["kernel"] for case in capture.cases}:
            raise RuntimeError("Flash prefill matmul kernels were not captured")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} Flash prefill matmul launches", flush=True)
        return
    if args.flash_prefill_attention:
        try:
            flash_prefill_attention_fixtures(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        if "flash_prefill_gqa" not in {case["kernel"] for case in capture.cases}:
            raise RuntimeError("Flash prefill GQA kernel was not captured")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} Flash prefill attention kernel launches", flush=True)
        return
    if args.flash_prefill_gdn:
        try:
            flash_prefill_gdn_fixtures(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        if "flash_prefill_gdn" not in {case["kernel"] for case in capture.cases}:
            raise RuntimeError("Flash prefill scalar gated-delta kernel was not captured")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} Flash prefill recurrence launches", flush=True)
        return
    if args.flash_prefill_hc:
        try:
            flash_prefill_hc_fixtures(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        required = {name for name in capture.fingerprints if name.startswith("flash_prefill_hc_")}
        if len(required) != 3 or required - {case["kernel"] for case in capture.cases}:
            raise RuntimeError("Flash prefill hyper-connection kernels were not captured")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} Flash prefill kernel launches", flush=True)
        return
    if args.flash_affine:
        try:
            flash_affine_variants(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        required = {name for name in capture.fingerprints if name.startswith("flash_") and not name.startswith("flash_prefill_")}
        missing = required - {case["kernel"] for case in capture.cases}
        if missing:
            raise RuntimeError(f"Missing Flash affine kernels: {sorted(missing)}")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} Flash affine and architecture launches", flush=True)
        return
    if args.row_attention:
        try:
            row_attention_variants(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        required = {"row_attention_partial", "row_attention_merge"}
        if required - {case["kernel"] for case in capture.cases}:
            raise RuntimeError("Row attention launches were not captured")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} row attention launches", flush=True)
        return
    if args.large_families:
        try:
            code = pytest.main(["-q", "-rs", "--basetemp", str(args.directory / "pytest"),
                                "tests/test_glm5_row_kernels.py", "tests/test_glm5_fused.py",
                                "tests/test_glm5_ported_kernels.py", "tests/test_deepseek_v4_kernels.py",
                                "-k", "not real_weights"], plugins=[capture])
            if code:
                raise SystemExit(code)
            large_family_variants(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        required = {name for name in capture.fingerprints if name.startswith(("glm_", "ds4_"))}
        missing = required - {case["kernel"] for case in capture.cases}
        print(f"Saved {len(capture.cases)} large-family launches; uncovered: {sorted(missing)}", flush=True)
        if missing:
            raise RuntimeError(f"Missing large-family kernels: {sorted(missing)}")
        return
    if args.affine_only or args.tensor_quantization or args.bonsai_only or args.gemma_only:
        try:
            code = pytest.main(["-q", "-rs", "tests/test_gemma4_kernels.py" if args.gemma_only else "tests/test_bonsai.py" if args.bonsai_only else "tests/test_lane_qmm.py" if args.tensor_quantization else "tests/test_affine_rows_metal.py"], plugins=[capture])
            if code:
                raise SystemExit(code)
            if args.gemma_only:
                gemma_quantization_variants(capture)
            if args.tensor_quantization:
                grouped_lane_variants(capture)
        finally:
            mx.fast.metal_kernel = capture.original
        if not capture.cases:
            raise RuntimeError("No affine launches captured")
        if args.gemma_only:
            required = {name for name in capture.catalog.values() if name.startswith("gemma_")}
            missing = required - {case["kernel"] for case in capture.cases}
            if missing:
                raise RuntimeError(f"Gemma kernel coverage missing: {sorted(missing)}")
        (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
        print(f"Saved {len(capture.cases)} affine launches", flush=True)
        return
    try:
        code = pytest.main(["-q", "-rs", "--disable-warnings", "tests/test_lane_fuse.py",
                            "tests/test_lane_gdn.py", "tests/test_row_forward.py",
                            "tests/test_simd_qmm.py", "tools/native_legacy/test_nemotron_rows.py",
                            "tests/test_flash_expert_down.py"], plugins=[capture])
        if code:
            raise SystemExit(code)
        extra_variants(capture)
        retired_glue_variants(capture)
        flash_variants(capture)
        nemotron_variants(capture)
        attention_and_ple_variants(capture)
        row_attention_variants(capture)
    finally:
        mx.fast.metal_kernel = capture.original
    if not capture.cases:
        raise RuntimeError("No kernel launches captured")
    (args.directory / "cases.json").write_text(json.dumps(capture.cases, indent=2) + "\n")
    counts = {}
    for case in capture.cases:
        counts[case["kernel"]] = counts.get(case["kernel"], 0) + 1
    required = {name for name in capture.catalog.values()
                if name.startswith(("row_forward_", "row_qmv", "lane_fuse_", "lane_gdn_", "simd_qmm_"))}
    required.update("nemotron_" + name for name in (
        "rows_qmv", "rows_expert_up", "rows_expert_down", "route", "add_norm_plain", "add_norm_moe", "add_norm_experts"))
    required.update("q4_" + name for name in (
        "qmv", "qmv_rows", "embed_rows", "swiglu", "route", "expert_gateup", "expert_group",
        "grouped_gateup", "expert_down_y", "grouped_down", "expert_down"))
    required.update(("q4_ple_lookup", "q4_router_float", "q4_router_bfloat", "lane_attention_partial", "lane_attention_partial_128"))
    required.update(("lane_attention_tail", "lane_attention_tree_merge"))
    required.update(("row_attention_partial", "row_attention_merge"))
    if missing := required - counts.keys():
        raise RuntimeError(f"Required native variant coverage missing: {sorted(missing)}")
    print(json.dumps(counts, indent=2), flush=True)
    print(f"Saved {len(capture.cases)} launches across {len(counts)} embedded variants", flush=True)


if __name__ == "__main__":
    main()
