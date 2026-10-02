"""_index_scores tiling sweep (keys/program BTT, warps, stages): time + bit-equality with today's config."""
import time, torch, triton
from tensorfold.families.glm_moe_dsa.cuda.fused import _index_scores, BT
NH, D = 32, 128
def main():
    torch.manual_seed(0)
    for T in (32768, 131072):
        n = 128
        q = (torch.randn(n, NH, D, device="cuda") * 0.5).bfloat16(); w = torch.randn(n, NH, device="cuda").float()
        ic = (torch.randn(T, D, device="cuda") * 0.5).bfloat16(); pos = torch.tensor([T - n - 1], dtype=torch.int64, device="cuda")
        for pack in (False, True):
            def go(btt, nw, ns, out):
                _index_scores[(n, triton.cdiv(T, btt))](q, w, ic, pos, out, T, 0, NH=NH, D=D, BTT=btt, WSCALE=NH ** -0.5,
                                                        QSCALE=D ** -0.5, PACK=pack, num_warps=nw, num_stages=ns)
            dt = torch.int64 if pack else torch.int32
            ref = torch.empty((n, T), dtype=dt, device="cuda"); go(BT, 4, 3, ref); torch.cuda.synchronize()
            res = []
            for btt in (32, 64, 128, 256):
                for nw in (2, 4, 8):
                    for ns in (1, 2, 3):
                        out = torch.empty_like(ref)
                        try:
                            go(btt, nw, ns, out); torch.cuda.synchronize(); t = time.perf_counter()
                            for _ in range(5): go(btt, nw, ns, out)
                            torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 5 * 1e3
                            res.append((ms, (btt, nw, ns), torch.equal(out, ref)))
                        except Exception:  # noqa: BLE001
                            pass
            res.sort(); base = next(r for r in res if r[1] == (BT, 4, 3)) if any(r[1] == (BT, 4, 3) for r in res) else None
            best = next(r for r in res if r[2])
            print(f"T={T} pack={pack}: default (BT={BT},4w,3s) {base[0] if base else float('nan'):.3f} ms | best exact {best[1]} "
                  f"{best[0]:.3f} ms ({(base[0] / best[0]) if base else 0:.2f}x) | fastest {res[0][1]} {res[0][0]:.3f} exact={res[0][2]}", flush=True)
if __name__ == "__main__":
    main()
