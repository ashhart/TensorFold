#!/usr/bin/env python3
"""Compare two NVFP4 A16 fatbins at the same WARPS=4 GLM TP2 entry points.

This is an isolated operator diagnostic, NOT served throughput or full kernel
qualification. Run only on an idle GPU with authorization. It allocates about
1.90 GiB of synthetic expert weights, plus small buffers and optional eviction
storage. It does not load a checkpoint or change a service.

Example:
  python bench_nvfp4_candidate_v8.py baseline.fatbin candidate.fatbin
  python bench_nvfp4_candidate_v8.py baseline.fatbin candidate.fatbin --flush-mib 64
  python bench_nvfp4_candidate_v8.py baseline.fatbin candidate.fatbin --candidate-low8

--candidate-low8 selects nvfp4_expert_low8_kernel for the candidate ONLY.
Additional untimed checks include shuffled members, hand-made cnt>8 items,
BF16 down outputs, finite extreme inputs and NaN/+Inf/-Inf classifications.

Optional --route-summary JSON has either the direct mapping below or a top-level
"mean_unique_by_rows" containing it:
  {"2": 13.2, "3": 18.0, "4": 23.0, "8": 42.0}
Means count ROUTED experts only (not reserved shared expert 288). Synthetic
plans approximate these means; they are not per-token router replay. Fractional
means are represented by floor/ceil counts across --plans reproducible plans.

--flush-mib is off by default. If enabled, select a size larger than measured L2;
a sequential write is only an eviction approximation, not proof of cold caches.
Flush writes are outside CUDA-event timing. No profiler or synchronization is
inserted inside either tested kernel.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


EXPERTS, TOPK, SLOTS, WARPS = 288, 8, 9, 4
DEFAULT_MEANS = {2: 13.2, 3: 18.0, 4: 23.0, 8: 42.0}
torch = None  # Import only after argument validation; --self-test is CPU-only.


def report(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def checked(code, operation):
    if code:
        raise RuntimeError(f"{operation}: CUDA driver error {code}")


def unique_counts(rows, mean, plans):
    if not math.isfinite(mean) or not TOPK <= mean <= min(EXPERTS, rows * TOPK):
        raise ValueError(f"rows={rows}: mean_unique must be in [8, {min(EXPERTS, rows * TOPK)}]")
    lower = math.floor(mean)
    upper_count = round((mean - lower) * plans)
    return [lower + (i < upper_count) for i in range(plans)]


def synthesize_picks(rows, unique, seed):
    """Exactly eight distinct IDs/row and exactly `unique` IDs in their union."""
    if not TOPK <= unique <= min(EXPERTS, rows * TOPK):
        raise ValueError("infeasible routed union")
    rng = random.Random(seed)
    pool = rng.sample(range(EXPERTS), unique)
    picks = [[pool[(r * TOPK + j) % unique] for j in range(TOPK)] for r in range(rows)]
    # Swap occurrences across rows without introducing duplicates. This retains
    # the union and frequency multiset while avoiding a purely cyclic layout.
    if rows > 1:
        for _ in range(rows * 64):
            r0, r1 = rng.sample(range(rows), 2)
            j0, j1 = rng.randrange(TOPK), rng.randrange(TOPK)
            a, b = picks[r0][j0], picks[r1][j1]
            if a != b and b not in picks[r0] and a not in picks[r1]:
                picks[r0][j0], picks[r1][j1] = b, a
    for row in picks:
        rng.shuffle(row)
    rng.shuffle(picks)
    assert all(len(set(row)) == TOPK for row in picks)
    assert len(set(v for row in picks for v in row)) == unique
    return [row + [EXPERTS] for row in picks]


def host_plan(picks, tile=16, shuffle_seed=None):
    flat = [v for row in picks for v in row]
    members, items = [], []
    for expert in sorted(set(flat)):
        positions = [j for j, value in enumerate(flat) if value == expert]
        if shuffle_seed is not None:
            random.Random(shuffle_seed + expert).shuffle(positions)
        # Production decode planner splits an expert's members into <=16 rows.
        for start in range(0, len(positions), tile):
            part = positions[start:start + tile]
            items.append([expert, len(members), len(part)])
            members.extend(part)
    return items, members


@dataclass
class Plan:
    items: object
    counts: object
    members: object
    actual_items: int
    bound_items: int
    routed_unique: int
    digest: str


def device_plan(rows, unique, seed, tile=16, shuffle_members=False):
    picks = synthesize_picks(rows, unique, seed)
    items, members = host_plan(picks, tile, seed if shuffle_members else None)
    encoded = json.dumps({"picks": picks, "items": items, "members": members}, separators=(",", ":")).encode()
    pairs = rows * SLOTS
    # Match engine maxItems bound (not the CPU-known actual item count).
    bound = min(pairs, EXPERTS + 1) + pairs // 16
    return Plan(torch.tensor(items, dtype=torch.int32, device="cuda"),
                torch.tensor([len(items)], dtype=torch.int32, device="cuda"),
                torch.tensor(members, dtype=torch.int32, device="cuda"),
                len(items), bound, unique, hashlib.sha256(encoded).hexdigest())


class Native:
    def __init__(self, path, label, low8=False):
        self.label = label
        self.driver = C.CDLL("libcuda.so.1")
        self.module = C.c_void_p()
        self.driver.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.driver.cuModuleUnload.argtypes = [C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        self.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
        self.driver.cuLaunchKernel.argtypes = [
            C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
            C.c_uint, C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        checked(self.driver.cuModuleLoad(C.byref(self.module), str(path).encode()), f"{label}: load")
        self.functions, self.resident = {}, {}
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        attrs = {"max_threads": 0, "static_shared_bytes": 1, "constant_bytes": 2,
                 "local_bytes_per_thread": 3, "registers_per_thread": 4,
                 "ptx_version": 5, "binary_version": 6, "max_dynamic_shared_bytes": 8}
        kernel = "nvfp4_expert_low8_kernel" if low8 else "nvfp4_expert_kernel"
        for parts, epilogue, name in ((2, 2, "gate_up"), (1, 0, "down_f32"), (1, 3, "down_bf16")):
            symbol = (f"_ZN16tf_nvfp4_experts{len(kernel)}{kernel}ILi{parts}ELi{epilogue}ELi4"
                      "EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi")
            function = C.c_void_p()
            checked(self.driver.cuModuleGetFunction(C.byref(function), self.module, symbol.encode()), symbol)
            occupancy = C.c_int()
            checked(self.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                C.byref(occupancy), function, WARPS * 32, 0), "occupancy")
            resource = {}
            for key, attr in attrs.items():
                value = C.c_int()
                error = self.driver.cuFuncGetAttribute(C.byref(value), attr, function)
                resource[key] = value.value if not error else {"unavailable_cuda_error": error}
            self.functions[parts, epilogue] = function
            self.resident[parts, epilogue] = max(1, occupancy.value) * sms
            report(kind="kernel", label=label, name=name, symbol=symbol, warps=WARPS,
                   sms=sms, active_blocks_per_sm=occupancy.value,
                   resident_grid_cap=self.resident[parts, epilogue], dynamic_shared_bytes=0,
                   resources=resource, note="Occupancy is theoretical, not measured achieved occupancy")

    def launch(self, parts, x, w, scales, plan, out, k, n, slots, epilogue=None):
        if epilogue is None:
            epilogue = 2 if parts == 2 else 0
        key = (parts, epilogue)
        arguments = [C.c_void_p(x.data_ptr()), C.c_int(x.stride(0)), C.c_int(slots),
                     C.c_void_p(w.data_ptr()), C.c_void_p(scales.data_ptr()), C.c_int(k // 32),
                     C.c_int(n // 32), C.c_void_p(plan.items.data_ptr()), C.c_void_p(plan.counts.data_ptr()),
                     C.c_void_p(plan.members.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int(n),
                     C.c_float(10.0 if parts == 2 else 0.0), C.c_int(EXPERTS)]
        argv = (C.c_void_p * len(arguments))(*(C.cast(C.byref(v), C.c_void_p) for v in arguments))
        grid = min((plan.bound_items * (n // 32) + WARPS - 1) // WARPS, self.resident[key])
        checked(self.driver.cuLaunchKernel(
            self.functions[key], grid, 1, 1, WARPS * 32, 1, 1, 0,
            C.c_void_p(torch.cuda.current_stream().cuda_stream), argv, None), f"{self.label}: launch")

    def close(self):
        if self.module.value:
            checked(self.driver.cuModuleUnload(self.module), f"{self.label}: unload")
            self.module = C.c_void_p()


def weights(parts, n, k):
    packed = torch.empty((EXPERTS, n // 32, k // 32, parts, 144), dtype=torch.int32, device="cuda")
    packed.random_(-2147483648, 2147483648)
    packed[..., 128:].bitwise_and_(0x1F1F1F1F).bitwise_or_(0x20202020)
    scales = torch.rand((EXPERTS, parts), device="cuda") * 0.01 + 0.005
    return packed, scales


def guarded(rows, cols, dtype):
    storage = torch.full((rows * cols + 64,), -123, dtype=dtype, device="cuda")
    return storage, storage[32:-32].view(rows, cols)


def validate_output(storage, out, rows, require_finite=True):
    assert bool((storage[:32] == -123).all() and (storage[-32:] == -123).all()), "Output guard changed"
    if require_finite:
        assert bool(torch.isfinite(out).all()), "Nonfinite result"
    assert bool((out.view(rows, SLOTS, -1)[:, TOPK] == -123).all()), "Reserved shared slot changed"


def digest(tensor):
    """Exact byte digest, copying at most 32 MiB at once rather than 2 GiB."""
    byte = tensor.contiguous().view(torch.uint8).flatten()
    h = hashlib.sha256()
    chunk = 32 << 20
    for start in range(0, byte.numel(), chunk):
        h.update(byte[start:start + chunk].cpu().numpy().tobytes())
    return h.hexdigest()


def load_means(path):
    if path is None:
        return DEFAULT_MEANS.copy()
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError("route summary must be a JSON object")
    obj = obj.get("mean_unique_by_rows", obj)
    if not isinstance(obj, dict):
        raise ValueError("mean_unique_by_rows must be an object")
    return {int(key): float(value) for key, value in obj.items()}


def same_classes_and_finite_bits(actual, expected, label):
    """NaN payloads may differ; signs of infinities and every finite bit may not."""
    for name, classify in (("nan", torch.isnan), ("+inf", torch.isposinf), ("-inf", torch.isneginf)):
        assert torch.equal(classify(actual), classify(expected)), f"{label}: {name} classification differs"
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].contiguous().view(torch.uint8),
                       expected[finite].contiguous().view(torch.uint8)), f"{label}: finite bits differ"


def extra_correctness(variants, weight_tensors, seed, repeats):
    """Direct candidate-kernel checks beyond production's low8 dispatch window."""
    up_w, up_s, down_w, down_s = weight_tensors
    # cnt17/31/32 intentionally bypass the production planner's <=16 item split.
    # The old and low8 kernels both promise a complete r0 loop, not truncation.
    cases = [("awkward_members", rows, rows) for rows in (1, 7, 9, 15, 16, 17, 31, 32)]
    cases += [(kind, rows, 16) for kind in ("finite_extreme", "nonfinite") for rows in (1, 3, 8)]
    for ci, (kind, rows, tile) in enumerate(cases):
        plan = device_plan(rows, TOPK, seed + 700001 + ci, tile=tile, shuffle_members=True)
        generator = torch.Generator(device="cuda").manual_seed(seed + ci)
        x = torch.randn((rows, 4112), dtype=torch.bfloat16, device="cuda", generator=generator)
        if kind != "awkward_members":
            info = torch.finfo(torch.bfloat16)
            values = [0.0, -0.0, info.tiny, -info.tiny, info.max, -info.max, 1e4, -1e4]
            if kind == "nonfinite":
                # A single nonfinite term can yield signed infinity or NaN
                # depending on that output's FP4 code; don't put NaN everywhere.
                kinds = [float("nan"), float("inf"), -float("inf")]
                for row in range(rows):
                    x[row, 0] = kinds[row % len(kinds)]
            else:
                pattern = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
                # Repeat within real input columns; leave padded columns independent.
                x[:, :4096].copy_(pattern.repeat(4096 // len(values)).view(1, 4096).expand(rows, -1))
        inputs = [x, plan.items, plan.counts, plan.members]
        before = [t.clone() for t in inputs]
        us, up = guarded(rows * SLOTS, 1024, torch.bfloat16)
        fs, f32 = guarded(rows * SLOTS, 4096, torch.float32)
        bs, bf16 = guarded(rows * SLOTS, 4096, torch.bfloat16)
        expected = None
        for repeat in range(repeats):
            for label, native in variants.items():
                for out in (up, f32, bf16):
                    out.fill_(-123)
                native.launch(2, x, up_w, up_s, plan, up, 4096, 1024, SLOTS)
                native.launch(1, up, down_w, down_s, plan, f32, 1024, 4096, 0)
                native.launch(1, up, down_w, down_s, plan, bf16, 1024, 4096, 0, epilogue=3)
                torch.cuda.synchronize()
                for storage, out in ((us, up), (fs, f32), (bs, bf16)):
                    validate_output(storage, out, rows, require_finite=kind == "awkward_members")
                if expected is None:
                    expected = [t.clone() for t in (up, f32, bf16)]
                for name, actual, want in zip(("gate_up", "down_f32", "down_bf16"), (up, f32, bf16), expected):
                    same_classes_and_finite_bits(actual, want, f"{kind} rows={rows} {label} {name}")
        for saved, current in zip(before, inputs):
            assert torch.equal(saved.view(torch.uint8), current.view(torch.uint8)), f"{kind}: input/plan bytes changed"
        classifications = {}
        for name, tensor in zip(("gate_up", "down_f32", "down_bf16"), expected):
            active = tensor.view(rows, SLOTS, -1)[:, :TOPK]
            classifications[name] = {"nan": int(torch.isnan(active).sum()),
                                     "positive_inf": int(torch.isposinf(active).sum()),
                                     "negative_inf": int(torch.isneginf(active).sum()),
                                     "finite": int(torch.isfinite(active).sum())}
        report(kind="extra_correctness", status="PASS", case=kind, rows=rows,
               item_tile=tile, shuffled_members=True, plan_sha256=plan.digest,
               max_item_count=int(plan.items[:, 2].max()), repeats=repeats,
               finite_output_bits_equal=True, nonfinite_classification_equal=True,
               classifications=classifications, input_bytes_unchanged=True,
               guards_intact=True, shared_slot_untouched=True,
               note="Untimed kernel check; cnt>16 is deliberately outside normal planner output; finite extreme inputs may overflow")
        if kind in ("finite_extreme", "nonfinite"):
            # SwiGLU/clamp may turn nonfinite projections into finite values.
            # Also feed down directly so its FP32 and BF16 epilogues genuinely
            # see extreme and nonfinite inputs independently of the activation.
            direct = torch.randn((rows * SLOTS, 1024), dtype=torch.bfloat16,
                                 device="cuda", generator=generator)
            if kind == "nonfinite":
                kinds = [float("nan"), float("inf"), -float("inf")]
                for pair in range(rows * SLOTS):
                    direct[pair, 0] = kinds[pair % len(kinds)]
            else:
                direct.copy_(pattern.repeat(1024 // len(values)).view(1, 1024).expand_as(direct))
            direct_before = direct.clone()
            want_down = None
            for repeat in range(repeats):
                for label, native in variants.items():
                    f32.fill_(-123)
                    bf16.fill_(-123)
                    native.launch(1, direct, down_w, down_s, plan, f32, 1024, 4096, 0)
                    native.launch(1, direct, down_w, down_s, plan, bf16, 1024, 4096, 0, epilogue=3)
                    torch.cuda.synchronize()
                    for storage, out in ((fs, f32), (bs, bf16)):
                        validate_output(storage, out, rows, require_finite=False)
                    if want_down is None:
                        want_down = (f32.clone(), bf16.clone())
                    for name, actual, want in zip(("down_f32", "down_bf16"), (f32, bf16), want_down):
                        same_classes_and_finite_bits(actual, want, f"{kind} direct {label} {name}")
            assert torch.equal(direct.view(torch.uint8), direct_before.view(torch.uint8)), "Direct-down input changed"
            for saved, current in zip(before[1:], inputs[1:]):
                assert torch.equal(saved.view(torch.uint8), current.view(torch.uint8)), "Direct-down plan changed"
            classifications = {}
            for name, tensor in zip(("down_f32", "down_bf16"), want_down):
                active = tensor.view(rows, SLOTS, -1)[:, :TOPK]
                classifications[name] = {"nan": int(torch.isnan(active).sum()),
                                         "positive_inf": int(torch.isposinf(active).sum()),
                                         "negative_inf": int(torch.isneginf(active).sum()),
                                         "finite": int(torch.isfinite(active).sum())}
            report(kind="extra_correctness", status="PASS", case=f"{kind}_direct_down", rows=rows,
                   finite_output_bits_equal=True, nonfinite_classification_equal=True,
                   classifications=classifications, input_bytes_unchanged=True,
                   guards_intact=True, shared_slot_untouched=True)


def self_test():
    for rows, mean in DEFAULT_MEANS.items():
        for i, count in enumerate(unique_counts(rows, mean, 10)):
            picks = synthesize_picks(rows, count, 4091 + i)
            assert picks == synthesize_picks(rows, count, 4091 + i)
            assert all(row[-1] == EXPERTS and len(set(row[:-1])) == TOPK for row in picks)
            items, members = host_plan(picks)
            flat = [v for row in picks for v in row]
            assert sorted(members) == list(range(rows * SLOTS))
            assert all(0 < cnt <= 16 and all(flat[j] == expert for j in members[first:first + cnt])
                       for expert, first, cnt in items)
    for rows, count in ((1, 8), (32, 8), (32, 256)):
        picks = synthesize_picks(rows, count, 13)
        items, _ = host_plan(picks)
        assert all(0 < cnt <= 16 for _, _, cnt in items)
    for rows in (9, 15, 16, 17, 31, 32):
        picks = synthesize_picks(rows, TOPK, 13)
        items, members = host_plan(picks, tile=rows, shuffle_seed=109)
        assert all(cnt == rows for _, _, cnt in items)
        assert sorted(members) == list(range(rows * SLOTS))
        flat = [v for row in picks for v in row]
        assert all(all(flat[j] == expert for j in members[first:first + cnt]) for expert, first, cnt in items)
    assert len("nvfp4_expert_kernel") == 19
    assert len("nvfp4_expert_low8_kernel") == 24
    report(kind="self_test", status="PASS", gpu_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--rows", default="2,3,4,8")
    parser.add_argument("--plans", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--correctness-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4091)
    parser.add_argument("--route-summary", type=Path)
    parser.add_argument("--candidate-low8", action="store_true", help="Use candidate low8 symbols; baseline always uses original symbols")
    parser.add_argument("--flush-mib", type=int, default=0, help="Optional 0–256 MiB eviction write; excluded from event timing")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.baseline is None or args.candidate is None:
        parser.error("baseline and candidate fatbins are required")
    if not args.baseline.is_file() or not args.candidate.is_file():
        parser.error("both fatbins must exist")
    if args.plans < 1 or args.rounds < 1 or args.reps < 3 or args.correctness_repeats < 3:
        parser.error("plans/rounds >=1; reps/correctness-repeats >=3 required")
    if not 0 <= args.flush_mib <= 256:
        parser.error("flush-mib must be 0–256")
    rows_list = [int(value) for value in args.rows.split(",")]
    means = load_means(args.route_summary)
    for rows in rows_list:
        if not 1 <= rows <= 32 or rows not in means:
            parser.error(f"rows={rows}: require 1–32 and an explicit mean_unique_by_rows entry")
        unique_counts(rows, means[rows], args.plans)

    global torch
    import torch as torch_module
    torch = torch_module
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.init()
    # Keep primary context current before loading Driver API modules.
    anchor = torch.empty(1, device="cuda")
    variants = {}
    try:
        variants["baseline"] = Native(args.baseline, "baseline")
        variants["candidate"] = Native(args.candidate, "candidate", low8=args.candidate_low8)
        device = torch.cuda.get_device_properties(0)
        report(kind="provenance", seed=args.seed, torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(), device_capability=torch.cuda.get_device_capability(),
               l2_cache_bytes=getattr(device, "L2_cache_size", None),
               fatbins={label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for label, path in (("baseline", args.baseline), ("candidate", args.candidate))},
               precision="NVFP4 weights, BF16 input/SwiGLU, FP32 timed down; extra BF16 down checks; GLM clip=10; WARPS=4",
               candidate_low8=args.candidate_low8,
               up_shape=[1024, 4096], down_shape=[4096, 1024], experts=EXPERTS, topk=TOPK,
               route_mode="synthetic approximate union means, NOT per-token route replay",
               means=means, plans=args.plans, repetitions=args.reps, rounds=args.rounds,
               flush_mib=args.flush_mib, flush_in_timing=False,
               includes_router=False, includes_tp=False, includes_service=False,
               grid_policy="production maxItems upper bound capped by each function's theoretical occupancy")
        up_w, up_s = weights(2, 1024, 4096)
        down_w, down_s = weights(1, 4096, 1024)
        weight_tensors = (up_w, up_s, down_w, down_s)
        weight_hashes = [digest(t) for t in weight_tensors]
        flush = torch.empty(args.flush_mib << 20, dtype=torch.uint8, device="cuda") if args.flush_mib else None
        for rows in rows_list:
            counts = unique_counts(rows, means[rows], args.plans)
            random.Random(args.seed + rows).shuffle(counts)
            plans = [device_plan(rows, count, args.seed + rows * 100003 + i) for i, count in enumerate(counts)]
            x = torch.randn((rows, 4112), dtype=torch.bfloat16, device="cuda")
            up_storage, up = guarded(rows * SLOTS, 1024, torch.bfloat16)
            down_storage, down = guarded(rows * SLOTS, 4096, torch.float32)
            inputs = [x] + [t for plan in plans for t in (plan.items, plan.counts, plan.members)]
            before = [t.clone() for t in inputs]

            def chain(label, plan):
                native = variants[label]
                native.launch(2, x, up_w, up_s, plan, up, 4096, 1024, SLOTS)
                native.launch(1, up, down_w, down_s, plan, down, 1024, 4096, 0)

            for plan in plans:
                expected = None
                for repeat in range(args.correctness_repeats):
                    for label in ("baseline", "candidate"):
                        up.fill_(-123)
                        down.fill_(-123)
                        chain(label, plan)
                        torch.cuda.synchronize()
                        validate_output(up_storage, up, rows)
                        validate_output(down_storage, down, rows)
                        if expected is None:
                            expected = (up.clone(), down.clone())
                        assert torch.equal(up.view(torch.uint8), expected[0].view(torch.uint8)), f"{label}: gate-up bits differ"
                        assert torch.equal(down.view(torch.uint8), expected[1].view(torch.uint8)), f"{label}: down bits differ"
            # Check before intentionally overwriting x: restoring it below must
            # not accidentally conceal a kernel that modified input padding.
            for saved, current in zip(before, inputs):
                assert torch.equal(saved.view(torch.uint8), current.view(torch.uint8)), "Correctness run changed input/plan bytes"
            x.zero_()
            for plan in plans:
                zero_expected = None
                for label in variants:
                    up.fill_(-123)
                    down.fill_(-123)
                    chain(label, plan)
                    torch.cuda.synchronize()
                    validate_output(up_storage, up, rows)
                    validate_output(down_storage, down, rows)
                    assert bool((up.view(rows, SLOTS, -1)[:, :TOPK] == 0).all()), "Zero up reference failed"
                    assert bool((down.view(rows, SLOTS, -1)[:, :TOPK] == 0).all()), "Zero down reference failed"
                    if zero_expected is None:
                        zero_expected = (up.clone(), down.clone())
                    assert torch.equal(up.view(torch.uint8), zero_expected[0].view(torch.uint8)), "Zero up bits differ"
                    assert torch.equal(down.view(torch.uint8), zero_expected[1].view(torch.uint8)), "Zero down bits differ"
            assert bool((x.view(torch.uint8) == 0).all()), "Zero-input bytes changed"
            for saved, current in zip(before[1:], inputs[1:]):
                assert torch.equal(saved.view(torch.uint8), current.view(torch.uint8)), "Zero run changed plan bytes"
            x.copy_(before[0])
            for label in variants:
                for plan in plans:
                    chain(label, plan)
            torch.cuda.synchronize()
            samples = {label: [] for label in variants}
            by_plan = {label: [[] for _ in plans] for label in variants}
            for rep in range(args.reps):
                order = ("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")
                for label in order:
                    events = []
                    for iteration in range(args.rounds):
                        for index, plan in enumerate(plans):
                            if flush is not None:
                                flush.fill_((rep + iteration + index) % 251)
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            start.record()
                            chain(label, plan)
                            end.record()
                            events.append((index, start, end))
                    events[-1][2].synchronize()
                    durations = []
                    for index, start, end in events:
                        duration = start.elapsed_time(end)
                        durations.append(duration)
                        by_plan[label][index].append(duration)
                    samples[label].append(statistics.mean(durations))
            for saved, current in zip(before, inputs):
                assert torch.equal(saved.view(torch.uint8), current.view(torch.uint8)), "Input or plan bytes changed"
            validate_output(up_storage, up, rows)
            validate_output(down_storage, down, rows)
            medians = {label: statistics.median(values) for label, values in samples.items()}
            report(kind="result", status="PASS", rows=rows, target_mean_unique=means[rows],
                   actual_mean_unique=statistics.mean(counts), unique_counts=counts,
                   union_over_rows_topk=statistics.mean(counts) / (rows * TOPK),
                   plans=[{"routed_unique": p.routed_unique, "actual_items": p.actual_items,
                           "bound_items": p.bound_items, "plan_sha256": p.digest} for p in plans],
                   byte_equal=True, correctness_repeats=args.correctness_repeats,
                   zero_reference=True, input_bytes_unchanged=True, guards_intact=True,
                   shared_slot_untouched=True, milliseconds=samples, median_ms=medians,
                   per_plan_median_ms={label: [statistics.median(v) for v in values] for label, values in by_plan.items()},
                   speedup_baseline_over_candidate=medians["baseline"] / medians["candidate"],
                   timing_note="CUDA events around up+down; excludes eviction write, includes any launch gaps between them")
        extra_correctness(variants, weight_tensors, args.seed, args.correctness_repeats)
        assert weight_hashes == [digest(t) for t in weight_tensors], "Weight/scale bytes changed"
        report(kind="summary", status="PASS", weight_bytes_unchanged=True,
               weight_sha256=weight_hashes, weight_bytes=sum(t.numel() * t.element_size() for t in weight_tensors),
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               qualification="Synthetic finite/nonfinite operator comparison only; not full contract, actual routing replay, or served speedup")
    finally:
        torch.cuda.synchronize()
        for native in variants.values():
            native.close()
        del anchor


if __name__ == "__main__":
    main()
