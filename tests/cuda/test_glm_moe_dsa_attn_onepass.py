"""One-pass prompt attention (DIRECT) == one chunk + _merge bit for bit; close to the 2-chunk default; timings."""
import time
import torch
from tensorfold.families.glm_moe_dsa.cuda.fused import _attn_chunks
from tensorfold.families.glm5_next.cuda import latent

H, LW, RD, K = 16, 512, 64, 2048


def main():
    torch.manual_seed(0)
    dev = "cuda"
    for R, ctx in ((2048, 32768), (2048, 131072)):
        qa = (torch.randn(R, H, LW, device=dev) * 0.3).bfloat16()
        qr = (torch.randn(R, H, RD, device=dev) * 0.3).bfloat16()
        lc = (torch.randn(ctx, LW + RD, device=dev) * 0.3).bfloat16()
        pos = torch.tensor([ctx - R], dtype=torch.int64, device=dev)
        tok = torch.sort(torch.stack([torch.randperm(ctx - R, device=dev)[:K] for _ in range(R)]), -1).values.int()
        scale = (192 + 64) ** -0.5

        def chunks(chk, kt, nw, ns, direct):
            nch = max(1, K // chk)
            ol = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
            po = torch.empty((nch * R * H * LW,), dtype=torch.float32, device=dev)
            pm = torch.empty((nch * R * H,), dtype=torch.float32, device=dev)
            pl = torch.empty_like(pm)
            if direct:
                go = lambda: _attn_chunks[(R, 1)](qa, qr, lc, tok, pos, ol, pm, pl, R, H=H, LW=LW, RD=RD, K=K, CHK=chk,
                                                  KTT=kt, SCALE=scale, DIRECT=True, num_warps=nw, num_stages=ns)
            else:
                def go():
                    _attn_chunks[(R, nch)](qa, qr, lc, tok, pos, po, pm, pl, R, H=H, LW=LW, RD=RD, K=K, CHK=chk,
                                           KTT=kt, SCALE=scale, num_warps=nw, num_stages=ns)
                    latent._merge[(R, H)](po, pm, pl, ol, pm, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
            go(); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(3): go()
            torch.cuda.synchronize()
            return ol.clone(), (time.perf_counter() - t0) / 3 * 1e3

        base, tb = chunks(1024, 64, 8, 2, False)
        one_m, t1m = chunks(2048, 64, 8, 2, False)
        rows = [("1 chunk + merge", one_m, t1m)]
        for kt, nw, ns in ((64, 8, 2), (64, 8, 3), (32, 4, 2), (64, 4, 2), (32, 8, 2), (32, 4, 3)):
            try:
                d, td = chunks(2048, kt, nw, ns, True)
            except Exception as e:  # noqa: BLE001  (shared memory)
                print(f"  direct kt{kt} w{nw} s{ns}: skipped ({type(e).__name__})"); continue
            rows.append((f"direct kt{kt} w{nw} s{ns}", d, td))
        print(f"R={R} ctx={ctx}: default 2x1024+merge {tb:.2f} ms")
        for name, o, t in rows:
            exact = torch.equal(o, one_m)
            rel = ((o.float() - base.float()).norm() / base.float().norm()).item()
            print(f"  {name:22s} {t:7.2f} ms ({tb / t:.2f}x)  ==1chunk+merge {exact}  rel vs default {rel:.2e}", flush=True)


if __name__ == "__main__":
    main()
