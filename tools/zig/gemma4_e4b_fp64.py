"""FP64 reference for tf-gemma4-e4b-projections: |native - ref| over sum |x||w| in 2^-8 units, worst column and rms."""
import os, sys
import numpy as np
import mlx.core as mx

model, dump = sys.argv[1], sys.argv[2]
# weights dequantized in float64 (scale * code + bias), one float64 sum, against the decode path's and tiles' outputs
w = mx.load(os.path.join(model, 'model.safetensors'))
P = 'language_model.model.'
GROUP = 64

def f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)

def dequant(prefix):
    words = np.array(w[prefix + '.weight']).astype(np.uint32)
    n = words.shape[0]
    codes = np.stack([(words >> (4 * j)) & 0xF for j in range(8)], -1).reshape(n, -1).astype(np.float64)
    s = f64(w[prefix + '.scales']).repeat(GROUP, 1)
    b = f64(w[prefix + '.biases']).repeat(GROUP, 1)
    return codes * s + b

def bf16_ulp(v):
    a = np.abs(v)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    return np.where(a > 0, 2.0 ** (e - 7), 2.0 ** -133)

def load(name):
    return np.load(os.path.join(dump, name + '.npy')).astype(np.float64)

classes = {}  # class -> {'win': [...], 'tile': [...]} of (layer, error in ulp [rows, n], abs error)

def add(cls, mode, layer, y, ref, mag):
    d = np.abs(y - ref)
    classes.setdefault(cls, {}).setdefault(mode, []).append((layer, d / mag * 256.0, d, ref))

layers = sorted({int(f.split('_L')[1].split('.')[0]) for f in os.listdir(dump) if '_L' in f})
for L in layers:
    base = f'{P}layers.{L}.'
    x = load(f'qkv_L{L}.x')
    members = [base + 'self_attn.q_proj'] + ([base + 'self_attn.k_proj', base + 'self_attn.v_proj'] if (base + 'self_attn.k_proj.weight') in w else [])
    W = np.concatenate([dequant(m) for m in members], 0)
    ref = x @ W.T
    mag = np.abs(x) @ np.abs(W).T
    for mode in ('win', 'tile'):
        add('qkv', mode, L, load(f'qkv_L{L}.{mode}'), ref, mag)
    x = load(f'o_L{L}.x'); Wm = dequant(base + 'self_attn.o_proj'); ref = x @ Wm.T; mag = np.abs(x) @ np.abs(Wm).T
    for mode in ('win', 'tile'): add('o', mode, L, load(f'o_L{L}.{mode}'), ref, mag)
    x = load(f'gateup_L{L}.x')
    g, u = dequant(base + 'mlp.gate_proj'), dequant(base + 'mlp.up_proj')
    n = g.shape[0]
    inter = np.empty((2 * n, g.shape[1]))
    for blk in range(n // 16):  # the row path's layout: 16 gate rows, 16 up rows, alternately
        inter[32 * blk:32 * blk + 16] = g[16 * blk:16 * blk + 16]
        inter[32 * blk + 16:32 * blk + 32] = u[16 * blk:16 * blk + 16]
    ref = x @ inter.T; mag = np.abs(x) @ np.abs(inter).T
    for mode in ('win', 'tile'): add('gate_up', mode, L, load(f'gateup_L{L}.{mode}'), ref, mag)
    x = load(f'down_L{L}.x'); Wm = dequant(base + 'mlp.down_proj'); ref = x @ Wm.T; mag = np.abs(x) @ np.abs(Wm).T
    for mode in ('win', 'tile'): add('down', mode, L, load(f'down_L{L}.{mode}'), ref, mag)
    for cls, key, tag in (('ple_gate', 'per_layer_input_gate', 'plegate'), ('ple_projection', 'per_layer_projection', 'pleproj')):
        x = load(f'{tag}_L{L}.x'); Wd = f64(w[base + key + '.weight'])
        add(cls, 'dense', L, load(f'{tag}_L{L}.y'), x @ Wd.T, np.abs(x) @ np.abs(Wd).T)
x = load('model_projection.x'); Wd = f64(w[P + 'per_layer_model_projection.weight'])
add('ple_model_projection', 'dense', -1, load('model_projection.y'), x @ Wd.T, np.abs(x) @ np.abs(Wd).T)

def summary(entries):
    ulp = np.concatenate([e[1].reshape(-1) for e in entries])
    percol = np.max(np.stack([np.pad(e[1].max(0), (0, 0)) for e in entries if e[1].shape[1] == entries[0][1].shape[1]]), 0)
    worst_layer, wc = None, None
    best = -1
    for L, e, d, ref in entries:
        c = int(e.max(0).argmax())
        if e.max(0)[c] > best:
            best, worst_layer, wc = float(e.max(0)[c]), L, c
    absmax = max(float(d.max()) for _, _, d, _ in entries)
    rel = np.concatenate([(d / np.maximum(np.abs(r), 1e-30)).reshape(-1) for _, _, d, r in entries])
    return dict(rms=float(np.sqrt((ulp ** 2).mean())), mean=float(ulp.mean()), mx=float(ulp.max()), col=wc, layer=worst_layer, absmax=absmax, n=len(ulp))

print(f"{'class':<22}{'path':<7}{'values':>10}{'mean':>10}{'rms':>9}{'worst col (layer)':>20}{'worst':>10}{'max |err|':>11}")
verdict = []
for cls, modes in classes.items():
    res = {m: summary(e) for m, e in modes.items()}
    for m, r in res.items():
        print(f"{cls:<22}{m:<7}{r['n']:>10}{r['mean']:>10.4f}{r['rms']:>9.4f}{('%d (L%d)' % (r['col'], r['layer'])):>20}{r['mx']:>10.2f}{r['absmax']:>11.4f}")
    if 'win' in res:
        t, v = res['tile'], res['win']
        verdict.append((cls, t['rms'] <= v['rms'] * 1.0000001, t['mx'] <= v['mx'] * 1.0000001, t['rms'] / v['rms'], t['mx'] / v['mx']))
print()
for cls, a, b, r1, r2 in verdict:
    print(f"{cls:<10} tiles' aggregate no worse than decode: {a} (rms ratio {r1:.2f}); worst no worse: {b} (ratio {r2:.2f})")
