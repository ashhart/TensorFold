"""Kimi K3's round on a TP x EP cluster from real shapes, our kernels' measured efficiencies and the fabric's link."""
import argparse

import k3_reference as ref

# Our kernels on the M5 Max (tf-k3-bench), as fractions of its measured stream bandwidth or simdgroup MMA peak.
EXPERT_EFF = ((1.0, 0.97), (1.3, 0.87), (1.67, 0.655), (2.54, 0.54), (4.62, 0.355))
SCALAR_BF16_EFF = 0.95
MMA_EFF = 0.72
KDA_EFF = 0.70
MLA_US_PER_ROW_HEAD_4K = 2.37


def interp(points, x):
    """Linear interpolation over (x, y) pairs sorted by x, clamped at the ends."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def weights(c: ref.Config):
    """Bytes a token reads: BF16 by group, and one routed expert's MXFP4 (packed + E8M0 scales)."""
    H, W, heads = c.hidden, c.kda_heads * c.kda_dim, c.mla_heads
    kda = 2 * (4 * W * H + H * W + c.kda_dim * H + W * c.kda_dim + c.kda_heads * H) + 4 * (3 * W * c.conv + W)
    mla = 2 * (c.q_lora * H + heads * (c.nope + c.rope) * c.q_lora + (c.kv_lora + c.rope) * H
               + heads * (c.nope + c.v_dim) * c.kv_lora + 2 * heads * c.v_dim * H)
    router = 2 * c.experts * H
    moe = 2 * (2 * c.latent * H + 3 * c.moe_inter * c.shared * H)
    n_kda = sum(c.is_kda(i) for i in range(c.layers))
    groups = {"KDA attention": n_kda * kda, "MLA attention": (c.layers - n_kda) * mla,
              "MoE latent + shared": (c.layers - 1) * moe, "routers (replicated)": (c.layers - 1) * router,
              "dense MLP (layer 0)": 2 * 3 * c.dense_inter * H, "LM head": 2 * c.vocab * H}
    expert = 3 * (c.moe_inter * c.latent // 2 + c.moe_inter * c.latent // 32)
    return groups, expert, n_kda


def distinct(c: ref.Config, rows: int) -> float:
    """Expected distinct experts a layer for `rows` independently, uniformly routed rows."""
    return c.experts * (1 - (1 - c.topk / c.experts) ** rows)


def allreduce_s(rows, width, a) -> float:
    """Reduce-scatter + all-gather of fp32 rows over a full mesh: each phase sends a quarter on every link at once."""
    per_link = rows * width * 4 / a.nodes
    return 2 * (per_link / a.link + a.one_way) + a.handoff


def model(c: ref.Config, rows: int, a) -> dict:
    groups, expert, n_kda = weights(c)
    split = sum(v for k, v in groups.items() if k != "routers (replicated)") / a.nodes + groups["routers (replicated)"]
    t = {}
    eff = SCALAR_BF16_EFF if rows < 16 else 1.0
    t["bf16"] = max(split / (a.stream * eff), rows * split / (a.mma * MMA_EFF))
    d = distinct(c, rows)
    m = rows * c.topk / d
    t["experts"] = d / a.nodes * (c.layers - 1) * expert / (a.stream * interp(EXPERT_EFF, m))
    heads = c.kda_heads // a.nodes
    t["kda state"] = rows * n_kda * heads * c.kda_dim * c.kda_dim * 4 * 2 / (a.stream * KDA_EFF)
    t["mla"] = rows * (c.mla_heads // a.nodes) * (c.layers - n_kda) * MLA_US_PER_ROW_HEAD_4K * 1e-6 / a.alu_scale \
        * a.context / 4096
    ar = allreduce_s(rows, c.hidden, a)
    t["collectives"] = (c.layers - 1) * 2 * (ar / a.chunks + a.handoff) + 2 * ar
    return {"rows": rows, "distinct": d, "per_expert_rows": m, "t": t, "round": sum(t.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4, help="TP and EP degree")
    ap.add_argument("--stream", type=float, default=700e9, help="a node's stream bandwidth, B/s (M3 Ultra, assumed)")
    ap.add_argument("--mma", type=float, default=22.5e12, help="a node's simdgroup MMA peak, FLOP/s (M3 Ultra ledger)")
    ap.add_argument("--alu-scale", type=float, default=1.65, help="M3 Ultra / M5 Max scalar ALU throughput")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--link", type=float, default=9.38e9, help="one link, one way (LINK-QUAL)")
    ap.add_argument("--one-way", type=float, default=3.5e-6, help="small-message one-way latency (LINK-QUAL)")
    ap.add_argument("--handoff", type=float, default=10e-6, help="GPU-CPU-GPU hand-off a collective (polled)")
    ap.add_argument("--chunks", type=int, default=4, help="output chunks a collective overlaps with")
    a = ap.parse_args()
    c = ref.Config()
    groups, expert, _ = weights(c)
    print("Bytes a token reads (one row):")
    for k, v in groups.items():
        print(f"  {k:30s} {v / 1e9:7.2f} GB")
    xp = c.topk * (c.layers - 1) * expert
    print(f"  {'routed experts (16 x 92)':30s} {xp / 1e9:7.2f} GB, one expert {expert / 1e6:.2f} MB")
    print(f"  {'total':30s} {(sum(groups.values()) + xp) / 1e9:7.2f} GB")
    print(f"\nPer node per round, TP{a.nodes} x EP{a.nodes}: stream {a.stream / 1e9:.0f} GB/s, MMA {a.mma / 1e12:.1f} TFLOPS, "
          f"{a.context} keys, link {a.link / 1e9:.2f} GB/s, hand-off {a.handoff * 1e6:.0f} us, {a.chunks} chunks")
    print("  rows  experts  rows/expert  bf16  experts  KDA   MLA  coll  round ms  tok/s  tokens/round for 80 / 100")
    for rows in (1, 4, 8, 16, 32, 64, 128, 256):
        r = model(c, rows, a)
        t = {k: v * 1e3 for k, v in r["t"].items()}
        print(f"  {rows:4d}  {r['distinct']:7.0f}  {r['per_expert_rows']:11.2f}  {t['bf16']:4.0f}  {t['experts']:7.0f} "
              f"{t['kda state']:5.1f} {t['mla']:5.1f} {t['collectives']:5.1f}  {r['round'] * 1e3:8.1f}  {rows / r['round']:5.1f}"
              f"  {80 * r['round']:5.1f} / {100 * r['round']:5.1f}")


if __name__ == "__main__":
    main()
