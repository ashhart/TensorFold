"""Copy a dump with the hyper-connection row kernels' product stage reading weights in order (same products, same
ordered sums: the same bits); their staged inputs padded one float in 32 against bank conflicts."""
import os, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=False)
for name in os.listdir(src):
    if name != "kernels":
        os.link(src / name, dst / name)
(dst / "kernels").mkdir()
done = {"down": 0, "up": 0}


def swap(text, pairs, kind):
    for a, b in pairs:
        assert a in text, (kind, a)
        text = text.replace(a, b)
    done[kind] += 1
    return text


for f in os.listdir(src / "kernels"):
    text = (src / "kernels" / f).read_text()
    if f.startswith("qa_hc_down_row"):
        text = swap(text, [
            ("threadgroup float xs[1024];", "threadgroup float xs[1056];"),
            ("    xs[t] = float(bfloat((float(HN[size_t(r) * W + e]) * ri) * NW[e]));",
             "    xs[t + t / 32] = float(bfloat((float(HN[size_t(r) * W + e]) * ri) * NW[e]));"),
            ("if (t < 32) vs[t] = scalar_sum(xs + 32 * t);", "if (t < 32) vs[t] = scalar_sum(xs + 33 * t);"),
            ("  ps[gl][ol] = scalar_chunk<BITS>(QW + size_t(o) * (W * BITS / 32) + (32 * k + gl) * BITS, xs + 32 * gl);",
             "  {\n    const int pgl = int(t) % 32, pol = int(t) / 32;\n    const int po = min(ob + pol, ND - 1);\n"
             "    ps[pgl][pol] = scalar_chunk<BITS>(QW + size_t(po) * (W * BITS / 32) + (32 * k + pgl) * BITS, xs + 33 * pgl);\n  }"),
        ], "down")
    elif f.startswith("qa_hc_up_row"):
        text = swap(text, [
            ("threadgroup float act[LOW];", "threadgroup float act[LOW + LOW / 32];"),
            ("if (cc < LOW) act[cc] = bsilu(v4);", "if (cc < LOW) act[cc + cc / 32] = bsilu(v4);"),
            ("if (int(t) < GL) vs[t] = scalar_sum(act + 32 * t);", "if (int(t) < GL) vs[t] = scalar_sum(act + 33 * t);"),
            ("  ps[g][n] = scalar_chunk<BITS>(QW + size_t(o) * (LOW * BITS / 32) + g * BITS, act + 32 * g);",
             "  {\n    const int pg = int(t) % GL, pn = int(t) / GL;\n    const int po = (pn / 8) * D + d0 + pn % 8;\n"
             "    ps[pg][pn] = scalar_chunk<BITS>(QW + size_t(po) * (LOW * BITS / 32) + pg * BITS, act + 33 * pg);\n  }"),
        ], "up")
    (dst / "kernels" / f).write_text(text)
print(done)
