"""Copy a dump with every lane_qmm bytes kernel computing its own group sums (xsum's loop, same order)."""
import os, re, shutil, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=False)
for name in os.listdir(src):
    if name == "kernels":
        continue
    os.link(src / name, dst / name)
(dst / "kernels").mkdir()
helper = """
inline float fz_xsum(const device bfloat16_t* X, int m, int g, int M, int K) {
  float acc = 0.0f;
  if (m < M) for (int i = 0; i < 32; i++) acc += float(X[m * K + g * 32 + i]);
  return acc;
}
"""
changed = 0
for f in os.listdir(src / "kernels"):
    text = (src / "kernels" / f).read_text()
    if f.startswith("lane_qmm_bytes_grouped"):
        a = "const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;"
        b = "const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;"
        assert a in text and b in text and "constexpr int GS = 32;" in text, f
        text = text.replace(a, "const float xs0 = live ? fz_xsum(X, rb + t * 16 + fm, g, M, K) : 0.0f;")
        text = text.replace(b, "const float xs1 = live ? fz_xsum(X, rb + t * 16 + fm + 8, g, M, K) : 0.0f;")
        k = text.index("[[kernel]]")
        text = text[:k] + helper + text[k:]
        changed += 1
    (dst / "kernels" / f).write_text(text)
print("fused", changed, "lane kernels")
