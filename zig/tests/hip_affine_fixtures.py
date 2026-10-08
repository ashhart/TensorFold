"""Independent FP32 emulation and deterministic arithmetic-regression search."""
import numpy as np


def bf(x):
    return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)


def rounded(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def run(x, words, scales, biases, group, variant="base"):
    k = x.shape[1]
    q = ((words[:, np.arange(k) // 8] >> ((np.arange(k) % 8) * 4)) & 15).astype(np.float32)
    x, scales, biases = bf(x), bf(scales), bf(biases)
    acc = np.zeros((32, x.shape[0], words.shape[0]), dtype=np.float32)
    for lane in range(32):
        passes = list(range(lane * 16, k, 512))
        if variant == "pass_reverse":
            passes.reverse()
        for at in passes:
            total = np.zeros((x.shape[0], 1), dtype=np.float32)
            if variant == "sum_pair":
                for j in range(0, 16, 2):
                    total = total + (x[:, at+j, None] + x[:, at+j+1, None])
            dot = np.zeros_like(acc[lane])
            for j in range(0, 16, 4):
                products = []
                for h in range(4):
                    v = x[:, at + j + h, None]
                    if variant != "sum_pair":
                        total = total + v
                    if variant == "bf16":
                        total = bf(rounded(total))
                    products.append(v * q[None, :, at + j + h])
                chain = ((products[0] + products[1]) + (products[2] + products[3])) if variant == "pair" else (((products[0] + products[1]) + products[2]) + products[3])
                dot = dot + chain
            scale = scales[None, :, at // group]
            bias = biases[None, :, at // group]
            if variant == "fma":
                term = (scale.astype(np.float64) * dot.astype(np.float64) +
                        (total * bias).astype(np.float64)).astype(np.float32)
            elif variant == "fma_bias":
                term = (total.astype(np.float64) * bias.astype(np.float64) +
                        (scale * dot).astype(np.float64)).astype(np.float32)
            else:
                term = scale * dot + total * bias
            if variant == "ftz":
                term = np.where(np.abs(term) < np.finfo(np.float32).tiny, np.float32(0), term)
            acc[lane] = (acc[lane] + total * bias) + scale * dot if variant == "split" else acc[lane] + term
    if variant == "lanes":
        return acc
    offsets = variant if isinstance(variant, tuple) else (16, 8, 4, 2, 1)
    for off in offsets:
        before = acc.copy()
        acc[:32-off] = before[:32-off] + before[off:]
    return rounded(acc[0])


def search_orders():
    """Greedy deterministic search covering shuffle permutations and epilogue/grouping changes."""
    import itertools
    orders = [p for p in itertools.permutations((16, 8, 4, 2, 1)) if p != (16, 8, 4, 2, 1)]
    candidates = {}
    def reduce(acc, offsets):
        acc = acc.copy()
        for off in offsets:
            before = acc.copy()
            acc[:32-off] = before[:32-off] + before[off:]
        return rounded(acc[0])
    for seed in range(1, 1001):
        args = generate(seed, 64, 576)
        lanes = run(*args, 64, "lanes")
        base = reduce(lanes, (16, 8, 4, 2, 1))
        changed = [reduce(lanes, order) != base for order in orders]
        changed += [run(*args, 64, v) != base for v in ("pair", "split", "fma", "fma_bias")]
        changed = np.stack(changed)
        for row, col in zip(*np.nonzero(changed.any(0))):
            candidates[(seed, int(row), int(col))] = set(np.nonzero(changed[:, row, col])[0])
    needed = set(range(len(orders) + 4))
    selected = []
    while needed:
        best = max(candidates, key=lambda key: len(candidates[key] & needed))
        if not candidates[best] & needed:
            raise AssertionError("search did not cover all variants")
        selected.append(best)
        needed -= candidates[best]
    print(selected)


def generate(seed, group, k, rows=32, outputs=24):
    rng = np.random.default_rng(seed)
    # Broad exponents create rounding-sensitive cancellation, unlike small LCG data.
    def values(shape, lo, hi):
        signs = rng.integers(0, 2, shape, dtype=np.uint16)
        exponents = rng.integers(lo + 127, hi + 127, shape, dtype=np.uint16)
        mantissas = rng.integers(0, 128, shape, dtype=np.uint16)
        return (signs << 15) | (exponents << 7) | mantissas
    return (values((rows, k), -20, 12),
            rng.integers(0, 2**32, (outputs, k // 8), dtype=np.uint32),
            values((outputs, k // group), -8, 0),
            values((outputs, k // group), -8, 0))



def matrix_inputs(group, k):
    seed = 1701 + group + k
    def nxt():
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xffffffff
        return seed
    def values(count, divisor):
        v = np.array([(nxt() >> 16) - 32768 for _ in range(count)], dtype=np.float32)
        return rounded(v / np.float32(divisor))
    x = values(32 * k, 4096).reshape(32, k)
    w = np.array([nxt() for _ in range(24 * k // 8)], dtype=np.uint32).reshape(24, k // 8)
    scales = values(24 * k // group, 1048576).reshape(24, k // group)
    biases = values(24 * k // group, 262144).reshape(24, k // group)
    return x, w, scales, biases


if __name__ == "__main__":
    import argparse
    import struct
    parser = argparse.ArgumentParser(description="Emit fixture hex to stdout; requires NumPy.")
    parser.add_argument("fixture", choices=("matrix", "sensitive", "search", "accuracy"))
    args = parser.parse_args()
    if args.fixture == "search":
        search_orders()
    elif args.fixture == "accuracy":
        for group in (32, 64):
            for k in (64, 192, 512, 576, 1088):
                x, w, s, b = matrix_inputs(group, k)
                b = rounded(np.float32(-8) * bf(s) * (np.float32(1) + bf(b)))
                q = ((w[:, np.arange(k)//8] >> ((np.arange(k)%8)*4)) & 15).astype(np.float64)
                reference = bf(x).astype(np.float64) @ (q * bf(s)[:, np.arange(k)//group] + bf(b)[:, np.arange(k)//group]).T
                errors = [np.linalg.norm(bf(run(x, w, s, b, group, v)).astype(np.float64) - reference) / np.linalg.norm(reference) for v in ("base", "bf16")]
                assert errors[0] < 0.0021 < errors[1]
                print(group, k, *errors)
    elif args.fixture == "matrix":
        for group in (32, 64):
            for k in (64, 192, 512, 576, 1088):
                result = run(*matrix_inputs(group, k), group)
                print(result.astype("<u2").tobytes().hex())
    else:
        import itertools
        variants = [p for p in itertools.permutations((16, 8, 4, 2, 1)) if p != (16, 8, 4, 2, 1)]
        variants += ["fma", "fma_bias", "bf16", "pair", "split", "pass_reverse", "sum_pair", "ftz"]
        covered = set()
        for seed, row, col, k, group in ((685, 31, 9, 576, 64), (598, 13, 8, 576, 64), (538, 24, 18, 576, 64), (126, 14, 22, 1088, 64), (26, 17, 6, 576, 32)):
            x, w, scales, biases = generate(seed, group, k)
            # Repeat the sensitive row across all batch widths, with two output columns.
            inputs = np.repeat(x[row:row+1], 2, axis=0), np.repeat(w[col:col+1], 2, axis=0), np.repeat(scales[col:col+1], 2, axis=0), np.repeat(biases[col:col+1], 2, axis=0)
            result = run(*inputs, group)
            for index, variant in enumerate(variants):
                if np.any(run(*inputs, group, variant) != result):
                    covered.add(index)
            payload = struct.pack("<5Q", 2, 2, k, 4, group)
            payload += b"".join(v.astype(v.dtype.newbyteorder("<")).tobytes() for v in (*inputs, result))
            print(payload.hex())
        # Constructive sum-order cancellation and FP32 subnormal epilogue cases.
        for subnormal in (False, True):
            x = np.zeros((2, 64), dtype=np.uint16)
            if subnormal:
                x[:, 0] = 0x0080
            else:
                x[:, :5] = (0x3f80, 0x3380, 0x3380, 0x3380, 0xbf80)
            w = np.zeros((1, 8), dtype=np.uint32)
            scales = np.zeros((1, 1), dtype=np.uint16)
            biases = np.full((1, 1), 0x3f00 if subnormal else 0x3f80, dtype=np.uint16)
            inputs = x, w, scales, biases
            result = run(*inputs, 64)
            assert result[:, 0].tolist() == ([0x0040, 0x0040] if subnormal else [0, 0])
            for index, variant in enumerate(variants):
                if np.any(run(*inputs, 64, variant) != result):
                    covered.add(index)
            payload = struct.pack("<5Q", 2, 1, 64, 4, 64)
            payload += b"".join(v.astype(v.dtype.newbyteorder("<")).tobytes() for v in (*inputs, result))
            print(payload.hex())
        assert len(covered) == len(variants), [v for i, v in enumerate(variants) if i not in covered]
        # Exact BF16 halfway values: even low endpoint, then odd low endpoint.
        x = np.zeros((2, 64), dtype=np.uint16)
        x[:, 0] = (0x3f80, 0x3f81)
        x[:, 1] = 0x3b80
        w = np.zeros((1, 8), dtype=np.uint32)
        scales = np.zeros((1, 1), dtype=np.uint16)
        biases = np.full((1, 1), 0x3f80, dtype=np.uint16)
        result = run(x, w, scales, biases, 64)
        assert result[:, 0].tolist() == [0x3f80, 0x3f82]
        payload = struct.pack("<5Q", 2, 1, 64, 4, 64)
        payload += b"".join(v.astype(v.dtype.newbyteorder("<")).tobytes() for v in (x, w, scales, biases, result))
        print(payload.hex())
