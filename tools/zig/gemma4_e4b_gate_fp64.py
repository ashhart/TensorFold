"""The E4B gate with its gelu epilogue against float64: x W^T, staged bf16 tanh GELU times the per-layer input."""
import os, sys
import numpy as np
import mlx.core as mx

# `dumps`: tf-gemma4-e4b-projections dumps of old and new builds; their distance to the reference, the worst column
model, *dumps = sys.argv[1:]
w = mx.load(os.path.join(model, 'model.safetensors'))
P = 'language_model.model.'

def bf(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)

def geglu(g, up):
    x = bf(g); up = bf(up)
    cube = bf(x * x * x)
    inner = bf(x + bf(bf(0.044715) * cube))
    th = bf(np.tanh(bf(bf(0.7978845608028654) * inner)))
    return bf(bf(bf(0.5 * x) * bf(1 + th)) * up)

layers = sorted({int(f.split('_L')[1].split('.')[0]) for f in os.listdir(dumps[0]) if f.startswith('gatefused_L')})
res = {d: dict(n=0, differ=0, sq=0.0, worst=(0.0, None, None), rel=0.0) for d in dumps}
for L in layers:
    Wd = np.array(w[f'{P}layers.{L}.per_layer_input_gate.weight'].astype(mx.float32)).astype(np.float64)
    n = Wd.shape[0]
    for d in dumps:  # each dump's own inputs (a build's earlier layers shift the later ones), its own reference
        x = np.load(f'{d}/gatefused_L{L}.x.npy').astype(np.float64)
        pli = np.load(f'{d}/gatefused_L{L}.pli.npy').astype(np.float64)
        dot = x @ Wd.T
        ref = geglu(dot, pli[:, L * n:(L + 1) * n])
        y = np.load(f'{d}/gatefused_L{L}.y.npy').astype(np.float64)
        r = res[d]
        err = np.abs(y - ref)
        r['n'] += err.size; r['differ'] += int((err > 0).sum()); r['sq'] += float((err ** 2).sum())
        c = int(err.max(0).argmax())
        if err.max(0)[c] > r['worst'][0]: r['worst'] = (float(err.max(0)[c]), c, L)
print(f"{'dump':<28}{'values':>9}{'differ from reference':>24}{'rms |err|':>12}{'worst col (layer)':>22}{'worst |err|':>12}")
for d, r in res.items():
    print(f"{d:<28}{r['n']:>9}{r['differ']:>14} ({100 * r['differ'] / r['n']:.3f}%){np.sqrt(r['sq'] / r['n']):>12.3e}{('%d (L%d)' % (r['worst'][1], r['worst'][2])):>22}{r['worst'][0]:>12.4e}")
