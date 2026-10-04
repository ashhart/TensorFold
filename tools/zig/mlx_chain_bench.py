"""MLX's cost per dependent dispatch beside tf-dispatch-bench: N chained tiny custom kernels (GPU lock)."""

from __future__ import annotations

import statistics
import time

import mlx.core as mx

SHAPES = (("tiny: 1 threadgroup x 32", 1, 32), ("wide: 1024 threadgroups x 64", 1024, 64))
COUNTS = (16, 64, 256, 1024)
REPS = 15

kernel = mx.fast.metal_kernel(name="tf_chain_step", input_names=["x"], output_names=["y"],
                              source="  const uint i = thread_position_in_grid.x;\n  y[i] = x[i] * 3u + 1u;\n")


def chain(x: mx.array, count: int, groups: int, threads: int) -> mx.array:
    for _ in range(count):
        x = kernel(inputs=[x], grid=(groups * threads, 1, 1), threadgroup=(threads, 1, 1),
                   output_shapes=[x.shape], output_dtypes=[mx.uint32])[0]
    return x


def slope(xs: list[float], ys: list[float]) -> float:
    mx_, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx_) ** 2 for x in xs)


def main() -> None:
    print(f"MLX {mx.__version__}; wall us of one evaluated chain, median of {REPS}; slope = us per dependent call")
    for name, groups, threads in SHAPES:
        x0 = mx.zeros((groups * threads,), dtype=mx.uint32)
        mx.eval(chain(x0, 64, groups, threads))
        medians = []
        for count in COUNTS:
            times = []
            for _ in range(REPS):
                start = time.perf_counter()
                out = chain(x0, count, groups, threads)
                mx.eval(out)
                times.append((time.perf_counter() - start) * 1e6)
            want = 0
            for _ in range(count):
                want = (want * 3 + 1) & 0xFFFFFFFF
            assert int(out[0].item()) == want and int(out[-1].item()) == want
            medians.append(statistics.median(times))
        cells = "".join(f"{m:>9.1f}" for m in medians)
        print(f"{name:<30}{cells}   {slope([float(c) for c in COUNTS], medians):.3f} us/call")


if __name__ == "__main__":
    main()
