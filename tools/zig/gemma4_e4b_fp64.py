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

EMULATE = os.environ.get('EMULATE') == '1'

def bf16r(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)

def dequant32(prefix):
    # scale * code + bias in fp32 (a bf16 scale times a code is exact in fp32: one rounding, as the kernel's fma)
    words = np.array(w[prefix + '.weight']).astype(np.uint32)
    n = words.shape[0]
    codes = np.stack([(words >> (4 * j)) & 0xF for j in range(8)], -1).reshape(n, -1).astype(np.float32)
    s_ = np.array(w[prefix + '.scales'].astype(mx.float32)).repeat(GROUP, 1)
    b_ = np.array(w[prefix + '.biases'].astype(mx.float32)).repeat(GROUP, 1)
    return codes * s_ + b_

def chunk32(x, W, blk=2048):
    # fp32 sums over 16-deep chunks in k order, the tiles' MMA slices (each chunk's dot in fp32, added to a running sum)
    x = x.astype(np.float32)
    R, K = x.shape
    C = K // 16
    xr = x.reshape(R, C, 16)
    out = np.empty((R, W.shape[0]), np.float32)
    for a in range(0, W.shape[0], blk):
        Wr = W[a:a + blk].reshape(-1, C, 16)
        acc = np.zeros((R, Wr.shape[0]), np.float32)
        for c_ in range(C):
            acc += xr[:, c_, :] @ Wr[:, c_, :].T
        out[:, a:a + blk] = acc
    return out.astype(np.float64)

def emulations(cls, L, x, W32, ref, mag, native_tile):
    # E0: exact sum rounded once to bf16; E1: bf16 weights; E2: fp32 chunks; E3: both. Compared with 'tile' native.
    Wb = bf16r(W32)
    outs = {'E0 exact': bf16r(ref), 'E1 w-bf16': bf16r(x @ Wb.T), 'E2 fp32-chunks': bf16r(chunk32(x, W32)), 'E3 both': bf16r(chunk32(x, Wb.astype(np.float32)))}
    for m, y in outs.items():
        add(cls, m, L, y, ref, mag)
        recolle.setdefault((cls, m), []).append(((y == native_tile).mean(), (np.abs(y - native_tile) / mag * 256.0)))

recolle = {}

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
    if EMULATE: emulations('qkv', L, x, np.concatenate([dequant32(m) for m in members], 0), ref, mag, load(f'qkv_L{L}.tile'))
    x = load(f'o_L{L}.x'); Wm = dequant(base + 'self_attn.o_proj'); ref = x @ Wm.T; mag = np.abs(x) @ np.abs(Wm).T
    for mode in ('win', 'tile'): add('o', mode, L, load(f'o_L{L}.{mode}'), ref, mag)
    if EMULATE: emulations('o', L, x, dequant32(base + 'self_attn.o_proj'), ref, mag, load(f'o_L{L}.tile'))
    x = load(f'gateup_L{L}.x')
    g, u = dequant(base + 'mlp.gate_proj'), dequant(base + 'mlp.up_proj')
    n = g.shape[0]
    inter = np.empty((2 * n, g.shape[1]))
    for blk in range(n // 16):  # the row path's layout: 16 gate rows, 16 up rows, alternately
        inter[32 * blk:32 * blk + 16] = g[16 * blk:16 * blk + 16]
        inter[32 * blk + 16:32 * blk + 32] = u[16 * blk:16 * blk + 16]
    ref = x @ inter.T; mag = np.abs(x) @ np.abs(inter).T
    for mode in ('win', 'tile'): add('gate_up', mode, L, load(f'gateup_L{L}.{mode}'), ref, mag)
    if EMULATE:
        g32, u32 = dequant32(base + 'mlp.gate_proj'), dequant32(base + 'mlp.up_proj')
        i32 = np.empty((2 * n, g32.shape[1]), np.float32)
        for blk in range(n // 16):
            i32[32 * blk:32 * blk + 16] = g32[16 * blk:16 * blk + 16]
            i32[32 * blk + 16:32 * blk + 32] = u32[16 * blk:16 * blk + 16]
        emulations('gate_up', L, x, i32, ref, mag, load(f'gateup_L{L}.tile'))
    x = load(f'down_L{L}.x'); Wm = dequant(base + 'mlp.down_proj'); ref = x @ Wm.T; mag = np.abs(x) @ np.abs(Wm).T
    for mode in ('win', 'tile'): add('down', mode, L, load(f'down_L{L}.{mode}'), ref, mag)
    if EMULATE: emulations('down', L, x, dequant32(base + 'mlp.down_proj'), ref, mag, load(f'down_L{L}.tile'))
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
    if 'win' in res and 'tile' in res:
        t, v = res['tile'], res['win']
        verdict.append((cls, t['rms'] <= v['rms'] * 1.0000001, t['mx'] <= v['mx'] * 1.0000001, t['rms'] / v['rms'], t['mx'] / v['mx']))
print()
for cls, a, b, r1, r2 in verdict:
    print(f"{cls:<10} tiles' aggregate no worse than decode: {a} (rms ratio {r1:.2f}); worst no worse: {b} (ratio {r2:.2f})")

if EMULATE:
    print()
    print('how closely each emulation reproduces the native tile outputs (share bit-equal; rms of the difference in the same units)')
    for (cls, m), v in recolle.items():
        eq = np.mean([a for a, _ in v])
        rms = float(np.sqrt(np.mean(np.concatenate([d.reshape(-1) ** 2 for _, d in v]))))
        print(f'{cls:<10}{m:<16} bit-equal {eq:7.3%}   rms diff {rms:.4f}')
