"""Capture original optional Metal variants for independent native dispatch tests.

The original upstream tests still make their own assertions. This recorder additionally
saves each distinct launch's inputs, parameters, and outputs. Native replay uses only
the checked-in embedded kernel catalog, never executable source from a fixture.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


def fingerprint(source, header):
    return hashlib.sha256((header + "\0" + source).encode()).hexdigest()


class Capture:
    def __init__(self, directory):
        self.directory = directory
        self.cases = []
        self.seen = set()
        self.test = "manual"
        self.original = mx.fast.metal_kernel
        self.catalog = {}
        for path in sorted(Path("native/metal").glob("*.metal")):
            if path.stem.startswith(("row_forward_", "row_qmv", "lane_fuse_", "lane_gdn_", "lane_attention_", "simd_qmm_", "q4_", "nemotron_")):
                self.catalog[(fingerprint(path.read_text(), path.with_suffix(".h").read_text()),
                              path.stem.endswith("_dep"))] = path.stem

    def pytest_runtest_setup(self, item):
        self.test = item.nodeid

    def kernel(self, **spec):
        kernel = self.original(**spec)
        source = spec["source"]
        constants = []
        if spec["name"].startswith("simd_qmm_"):
            # Python specializes dimensions as leading constexpr declarations;
            # the embedded native body receives those same values as templates.
            while match := re.match(r"  constexpr int (\w+) = (-?\d+);\n", source):
                constants.append((match[1], int(match[2])))
                source = source[match.end():]
        source_hash = fingerprint(source, spec.get("header", ""))
        key = self.catalog.get((source_hash, "DEP" in spec["input_names"]))
        if key is None:
            return kernel

        def launch(**call):
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
            out = kernel(**call)
            mx.eval(*call["inputs"], *out)
            name = f"case{len(self.cases):05}"
            arrays = {f"input{i}": x for i, x in enumerate(call["inputs"])}
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
                                   input_count=len(call["inputs"]), output_count=len(out)))
            return out
        return launch


def extra_variants(capture):
    from tensorfold.kernels.qwen.dense.v1 import row_forward, simd_qmm
    for group in (32, 64, 128):
        weight = (mx.random.normal((512, 1024), key=mx.random.key(group)) * .1).astype(mx.bfloat16)
        q, scales, biases = mx.quantize(weight, group_size=group, bits=4)
        for rows in (1, 3, 8):
            x = mx.random.normal((rows, 1024), key=mx.random.key(rows)).astype(mx.bfloat16)
            parts = row_forward.row_parts(x)
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


def flash_variants(capture):
    """Exercise optional grouped experts, routing ties, embedding and projection layouts."""
    from tensorfold.kernels.qwen.flash_next.v1 import kernels as flash

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


def nemotron_variants(capture):
    from tensorfold.kernels.nemotron.lightning.v1 import kernels as nemotron
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


def attention_and_ple_variants(capture):
    from tensorfold.kernels.qwen.dense.v1 import lane_attention
    from tensorfold.kernels.qwen.flash_next.v1 import kernels as flash
    import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    capture = Capture(args.directory)
    mx.fast.metal_kernel = capture.kernel
    try:
        code = pytest.main(["-q", "--disable-warnings", "tests/test_lane_fuse.py",
                            "tests/test_lane_gdn.py", "tests/test_row_forward.py",
                            "tests/test_simd_qmm.py", "tests/test_nemotron_rows.py",
                            "tests/test_flash_expert_down.py"], plugins=[capture])
        if code:
            raise SystemExit(code)
        extra_variants(capture)
        flash_variants(capture)
        nemotron_variants(capture)
        attention_and_ple_variants(capture)
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
    if missing := required - counts.keys():
        raise RuntimeError(f"Required native variant coverage missing: {sorted(missing)}")
    print(json.dumps(counts, indent=2), flush=True)
    print(f"Saved {len(capture.cases)} launches across {len(counts)} embedded variants", flush=True)


if __name__ == "__main__":
    main()
