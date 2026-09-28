"""Practical tensor-core peak on this GPU: torch bf16 matmuls and a plain Triton bf16 dot kernel."""
import torch, triton, triton.language as tl

def t(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps

for m, n, k in ((4096, 4096, 4096), (2048, 8192, 4096), (16384, 1024, 4096)):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ms = t(lambda: x @ w.t())
    print(f"torch bf16 {m}x{n}x{k}: {2*m*n*k/ms/1e9:6.1f} TFLOP/s")
    if hasattr(torch, "float8_e4m3fn"):
        try:
            x8 = x.to(torch.float8_e4m3fn); w8 = w.to(torch.float8_e4m3fn); one = torch.ones((), device="cuda")
            ms = t(lambda: torch._scaled_mm(x8, w8.t(), one, one, out_dtype=torch.bfloat16))
            print(f"torch fp8  {m}x{n}x{k}: {2*m*n*k/ms/1e9:6.1f} TFLOP/s")
        except Exception as e:
            print("fp8:", type(e).__name__, str(e)[:100])

@triton.jit
def _mm(X, W, O, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :])
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :])
        if SCALE:
            acc = acc + tl.dot(x, tl.trans(w)) * 1.0001 + 0.5
        else:
            acc = tl.dot(x, tl.trans(w), acc)
    tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

m, n, k = 4096, 4096, 4096
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
for bm, bn, bk, warps, st in ((128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (64, 64, 64, 4, 2), (32, 64, 64, 4, 2), (64, 128, 64, 4, 3)):
    for scale in (False, True):
        g = (m // bm, n // bn)
        try:
            ms = t(lambda: _mm[g](x, w, o, m, N=n, K=k, BM=bm, BN=bn, BK=bk, SCALE=scale, num_warps=warps, num_stages=st))
            print(f"triton bm{bm} bn{bn} bk{bk} w{warps} s{st} {'scaled per 64' if scale else 'plain acc   '}: {2*m*n*k/ms/1e9:6.1f} TFLOP/s")
        except Exception as e:
            print(bm, bn, type(e).__name__)
