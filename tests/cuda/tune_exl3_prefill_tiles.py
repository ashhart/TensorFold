"""Tile search for exl3 prefill._gemm per dense (K, N) at M rows: time + bit-equality with the current tiles()."""
import sys, time
import torch, triton
from tensorfold.cuda.exl3 import prefill as P

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SHAPES = [(6144, 2048), (6144, 640), (6144, 576), (2048, 4096), (4096, 6144), (6144, 3072), (3072, 6144), (6144, 512),
          (512, 6144)]
CFGS = [(bm, bk, w, s, g) for bm in (64, 128) for bk in (32, 64) for w in (4, 8) for s in (2, 3, 4) for g in (8,)]


def run(x, wq, h, svh, out, k, n, cfg):
    bm, bk, w, s, g = cfg
    P._gemm[(triton.cdiv(M, bm) * (n // P.BN),)](x, wq, h, svh, svh, out, M, out.stride(0), K=k, N=n, BM=bm, BK=bk,
                                                 GROUP=g, HAS_BIAS=False, SCALE=P.HAD_SCALE, num_warps=w, num_stages=s)


def main():
    torch.manual_seed(0)
    ws = P.Workspace(); h = ws.hadamard("cuda")
    for k, n in [s for s in SHAPES if s[1] % 128 == 0]:
        x = (torch.randn(M, k, device="cuda") * 0.5).half(); wq = (torch.randn(k, n, device="cuda") * 0.02).half()
        svh = torch.randn(n, device="cuda").half(); out = torch.empty(M, n, device="cuda", dtype=torch.bfloat16)
        base = P.tiles(k, n); run(x, wq, h, svh, out, k, n, base); ref = out.clone()
        res = []
        for cfg in CFGS:
            try:
                run(x, wq, h, svh, out, k, n, cfg); torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(8): run(x, wq, h, svh, out, k, n, cfg)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 8 * 1e3
                res.append((ms, cfg, torch.equal(out, ref)))
            except Exception:  # noqa: BLE001
                pass
        res.sort()
        tb = next(r[0] for r in res if r[1] == base)
        best = res[0]
        best_exact = next((r for r in res if r[2]), (tb, base, True))
        print(f"{k}x{n}: current {base} {tb:.3f} ms | best {best[1]} {best[0]:.3f} ms ({tb/best[0]:.2f}x, exact {best[2]})"
              f" | best exact {best_exact[1]} {best_exact[0]:.3f} ms ({tb/best_exact[0]:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
