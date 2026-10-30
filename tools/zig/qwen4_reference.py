"""Independent binary64 formula oracle for a synthetic dense Qwen4, with explicit BF16 storage nodes."""

import argparse
import json
import math
import struct
from pathlib import Path


def bf(x):
    """Round binary64 directly to BF16, avoiding a binary32 double rounding."""
    if not math.isfinite(x) or x == 0:
        return x
    sign = -1 if x < 0 else 1
    x = abs(x)
    exponent = math.frexp(x)[1] - 1
    if exponent > 127:
        return math.copysign(math.inf, sign)
    unit = max(-133, exponent - 7)
    numerator, denominator = x.as_integer_ratio()
    if unit < 0:
        numerator <<= -unit
    else:
        denominator <<= unit
    q, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and q & 1):
        q += 1
    y = math.ldexp(float(q), unit)
    return sign * (math.inf if y >= math.ldexp(255.5, 120) else y)


def f32(x):
    """An explicit scalar FP32 node; all formula evaluation on either side still uses binary64."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def silu(x):
    return x * sigmoid(x)


def splitmix(x):
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


def prime_after(x):
    def prime(n):
        return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))
    x += 1
    while not prime(x):
        x += 1
    return x


def values(name, count, norm=False, gated_norm=False):
    """Exactly representable deterministic BF16 source weights, independent of Python's randomized hash."""
    h = 2166136261
    for byte in name.encode():
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    divisor = 512 if norm else 128
    base = 1 if gated_norm else 0
    return [bf(base + ((h + 17 * i + 3 * i * i) % 31 - 15) / divisor) for i in range(count)]


class Oracle:
    def __init__(self, config, probability="bf16", nodes=False, native_scalars=False):
        self.native_scalars = native_scalars
        self.probability = probability
        self.capture_nodes = nodes
        self.nodes = []
        self.c = config
        self.h = config['hidden_size']
        self.s = config['hc_count']
        self.w = self.h * self.s
        self.nk = config['linear_num_key_heads']
        self.nv = config['linear_num_value_heads']
        self.dk = config['linear_key_head_dim']
        self.dv = config['linear_value_head_dim']
        self.kw, self.vw = self.nk * self.dk, self.nv * self.dv
        self.channels = 2 * self.kw + self.vw
        self.eps = config['rms_norm_eps']
        self.cache = [{} for _ in config['layer_types']]
        self.tokens = []
        self.trace = []
        self.ple_order = {layer - 1: i for i, layer in enumerate(config['ple_layer_ids'])}

    def record(self, name, values, kind="bf16"):
        if self.capture_nodes and values:
            self.nodes.append(dict(name=name, position=len(self.tokens), kind=kind, values=list(values)))
        return values

    def matrix(self, x, name, n):
        k = len(x)
        weights = values(name + '.weight', n * k)
        return [bf(sum(x[j] * weights[i * k + j] for j in range(k))) for i in range(n)]

    def norm(self, x, name, group=None, gemma=True):
        group = group or len(x)
        gamma = values(name + '.weight', len(x), norm=gemma, gated_norm=not gemma)
        out = []
        for start in range(0, len(x), group):
            inv = 1 / math.sqrt(sum(v * v for v in x[start:start + group]) / group + self.eps)
            if self.native_scalars:
                mean = f32(f32(sum(v * v for v in x[start:start + group])) / group)
                root = math.sqrt(f32(mean + f32(self.eps)))
                inv = f32(1 / (f32(root) if self.probability == "cuda" else root))
                out.extend(bf(f32(f32(x[i] * inv) * f32(1 + gamma[i] if gemma else gamma[i])))
                           for i in range(start, start + group))
            else:
                out.extend(bf(x[i] * inv * (1 + gamma[i] if gemma else gamma[i]))
                           for i in range(start, start + group))
        return out

    def hc(self, hidden, name, inject=True):
        self.record(name + '.sums', [sum(v*v for v in hidden[i:i+256]) for i in range(0,len(hidden),256)], 'f32')
        normed = self.norm(hidden, name + '.hc_norm', self.h)
        down = self.matrix(normed, name + '.input_mix_weight_down', self.c['hc_lowrank'])
        low = [bf(silu(bf(v / self.s))) for v in down]
        up = self.matrix(low, name + '.input_mix_weight_up', self.w)
        mixed = [bf(sum(bf(bf(sigmoid(up[s * self.h + d])) * normed[s * self.h + d])
                        for s in range(self.s)) / self.s) for d in range(self.h)]
        gates = None
        if inject:
            raw = self.matrix(normed, name + '.block_inject_weight', self.s)
            gates = [bf(2 * bf(sigmoid(bf(v / self.s)))) for v in raw]
        for node, vector in (("norm", normed), ("down", down), ("low", low), ("up", up), ("mix", mixed)):
            self.record(name + '.' + node, vector)
        if inject:
            self.record(name + '.raw_inject', raw)
            self.record(name + '.injection', gates)
        return mixed, gates

    def writeback(self, hidden, branch, gates):
        return [bf(x + bf(branch[i % self.h] * gates[i // self.h])) for i, x in enumerate(hidden)]

    def dense(self, x, name):
        n = self.c['intermediate_size']
        gate = self.matrix(x, name + '.gate_proj', n)
        up = self.matrix(x, name + '.up_proj', n)
        act = [bf(bf(silu(g)) * u) for g, u in zip(gate, up)]
        for node, vector in (("gate", gate), ("up", up), ("act", act)):
            self.record(name + '.' + node, vector)
        return self.record(name + '.output', self.matrix(act, name + '.down_proj', self.h))

    def delta(self, x, layer, name):
        c, state = self.c, self.cache[layer]
        qkv = self.matrix(x, name + '.in_proj_qkv', self.channels)
        z = self.matrix(x, name + '.in_proj_z', self.vw)
        a = self.matrix(x, name + '.in_proj_a', self.nv)
        b = self.matrix(x, name + '.in_proj_b', self.nv)
        self.record(name + '.projections', qkv + z + b + a)
        taps = c['linear_conv_kernel_dim']
        history = state.setdefault('conv', [[0.0] * self.channels for _ in range(taps - 1)])
        rows = history + [qkv]
        conv = values(name + '.conv1d.weight', self.channels * taps)
        convolved = [bf(silu(bf(sum(rows[t][i] * conv[i * taps + t] for t in range(taps)))))
                     for i in range(self.channels)]
        state['conv'] = rows[1:]
        self.record(name + '.conv_tail', sum(state['conv'], []))
        queries = convolved[:self.kw]
        keys = convolved[self.kw:2 * self.kw]
        vals = convolved[2 * self.kw:]
        for head in range(self.nk):
            for vector, scale in ((queries, self.dk ** -0.5), (keys, 1.0)):
                at = head * self.dk
                inv = scale / math.sqrt(sum(v * v for v in vector[at:at + self.dk]) + 1e-6)
                vector[at:at + self.dk] = [v * inv for v in vector[at:at + self.dk]]
        alog = values(name + '.A_log', self.nv)
        dt = values(name + '.dt_bias', self.nv)
        recurrent = state.setdefault('recurrent', [0.0] * (self.nv * self.dv * self.dk))
        y = [0.0] * self.vw
        for head in range(self.nv):
            kh = head // (self.nv // self.nk)
            k = keys[kh * self.dk:(kh + 1) * self.dk]
            q = queries[kh * self.dk:(kh + 1) * self.dk]
            decay = math.exp(-math.exp(alog[head]) * math.log1p(math.exp(a[head] + dt[head])))
            beta = bf(sigmoid(b[head]))
            for col in range(self.dv):
                start = (head * self.dv + col) * self.dk
                old = [s * decay for s in recurrent[start:start + self.dk]]
                correction = (vals[head * self.dv + col] - sum(s * t for s, t in zip(old, k))) * beta
                new = [s + t * correction for s, t in zip(old, k)]
                recurrent[start:start + self.dk] = new
                y[head * self.dv + col] = bf(sum(s * t for s, t in zip(new, q)))
        self.record(name + '.recurrent', recurrent, 'f32')
        norm_weight = values(name + '.norm.weight', self.dv, gated_norm=True)
        out = []
        for head in range(self.nv):
            row = y[head * self.dv:(head + 1) * self.dv]
            inv = 1 / math.sqrt(sum(v * v for v in row) / self.dv + self.eps)
            for d, v in enumerate(row):
                normalized = bf(bf(v * inv) * norm_weight[d])
                gate = z[head * self.dv + d]
                activated = sigmoid(gate) if c['output_gate_type'] == 'sigmoid' else silu(gate)
                out.append(bf(normalized * activated))
        self.record(name + '.gated', out)
        return self.record(name + '.output', self.matrix(out, name + '.out_proj', self.h))

    def rope(self, x, pos):
        params = self.c['rope_parameters']
        width = int(self.c['head_dim'] * params['partial_rotary_factor'])
        half = width // 2
        out = list(x)
        for i in range(half):
            frequency = params['rope_theta'] ** (-2 * i / width)
            angle = f32(pos * f32(frequency)) if self.native_scalars else pos * frequency
            cosine, sine = math.cos(angle), math.sin(angle)
            if self.native_scalars:
                cosine, sine = f32(cosine), f32(sine)
                out[i] = bf(f32(f32(x[i] * cosine) - f32(x[i + half] * sine)))
                out[i + half] = bf(f32(f32(x[i + half] * cosine) + f32(x[i] * sine)))
            else:
                out[i] = bf(x[i] * cosine - x[i + half] * sine)
                out[i + half] = bf(x[i + half] * cosine + x[i] * sine)
        return out

    def attention(self, x, layer, name):
        c, state = self.c, self.cache[layer]
        hd, nh, nk = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
        ih, di, ratio = c['indexer_n_heads'], c['indexer_head_dim'], c['indexer_compress_ratio']
        pos = len(self.tokens)
        qraw = self.matrix(x, name + '.q_proj', 2 * nh * hd)
        kraw = self.matrix(x, name + '.k_proj', nk * hd)
        vraw = self.matrix(x, name + '.v_proj', nk * hd)
        ikraw = self.matrix(x, name + '.indexer.index_qk_proj', (ih + 1) * di)
        self.record(name + '.projections', qraw + kraw + vraw + ikraw)
        q = [self.rope(self.norm(qraw[h * 2 * hd:h * 2 * hd + hd], name + '.q_norm'), pos)
             for h in range(nh)]
        k = [self.rope(self.norm(kraw[h * hd:(h + 1) * hd], name + '.k_norm'), pos) for h in range(nk)]
        state.setdefault('keys', []).append(k)
        state.setdefault('values', []).append(vraw)
        state.setdefault('index_keys', []).append(ikraw[ih * di:])
        iq = [self.rope(self.norm(ikraw[h * di:(h + 1) * di], name + '.indexer.q_layernorm'), pos)
              for h in range(ih)]
        self.record(name + '.query', sum(q, []))
        self.record(name + '.key', sum(k, []))
        if self.probability == 'cuda':
            self.record(name + '.value', vraw)
        self.record(name + '.index_query', sum(iq, []))
        blocks = (pos + 1) // ratio
        scores = []
        for block in range(blocks):
            start = block * ratio
            pooled = [bf(sum(state['index_keys'][start + j][d] for j in range(ratio)) / ratio)
                      for d in range(di)]
            pooled = self.rope(self.norm(pooled, name + '.indexer.k_layernorm'), start)
            if block == blocks - 1 and (pos + 1) % ratio == 0:
                self.record(name + '.pooled', pooled)
            scores.append(sum(max(0.0, sum(u * v for u, v in zip(head, pooled))) for head in iq) / math.sqrt(di))
        top = c['indexer_budget'] // ratio
        selected = sorted(sorted(range(blocks), key=lambda i: (-scores[i], i))[:top])
        ids = [i * ratio + j for i in selected for j in range(ratio)] + list(range(blocks * ratio, pos + 1))
        self.record(name + '.scores', scores, 'f32')
        out, attended_rows = [], []
        for head in range(nh):
            kh = head // (nh // nk)
            logits = [sum(a * b for a, b in zip(q[head], state['keys'][i][kh])) / math.sqrt(hd) for i in ids]
            maximum = max(logits)
            weights = [math.exp(v - maximum) for v in logits]
            total = sum(weights)
            if self.probability == "cuda":
                probability = [bf(v) / total for v in weights]
            else:
                probability = [bf(v / total) if self.probability == "bf16" else v / total for v in weights]
            for d in range(hd):
                attended = bf(sum(p * state['values'][i][kh * hd + d] for i, p in zip(ids, probability)))
                attended_rows.append(attended)
                gate = qraw[head * 2 * hd + hd + d]
                out.append(bf(attended * bf(sigmoid(gate))))
        self.record(name + '.attention', attended_rows)
        self.record(name + '.gated', out)
        return self.record(name + '.output', self.matrix(out, name + '.o_proj', self.h))

    def ple(self, hidden, token, layer, name):
        c, state = self.c, self.cache[layer]
        ordinal = self.ple_order[layer]
        heads = (c['ngram_size'] - 1) * c['heads_per_ngram']
        sizes, prime = [], c['ngram_vocab_size_base'] - 1
        for i in range((ordinal + 1) * heads):
            prime = prime_after(prime)
            if i >= ordinal * heads:
                sizes.append(prime)
        divisor = c['make_ngram_vocab_size_divisible_by']
        rows = ((sum(sizes) + divisor - 1) // divisor) * divisor
        width = c['ple_embed_dim'] // heads
        table = values(name + '.ple_embedding.ngram_embedding.weight', rows * width)
        half_bound = max(1, ((1 << 63) - 1) // c['vocab_size'] // 2)
        multipliers = [2 * (splitmix((c.get('seed', 1234) + 10007 * ordinal +
                                    0x9E3779B97F4A7C15 * (i + 1)) & ((1 << 64) - 1)) % half_bound) + 1
                       for i in range(c['ngram_size'])]
        eos = c['eos_token_id']
        eos = eos[0] if isinstance(eos, list) else eos
        history = state.setdefault('history', [eos] * (c['ngram_size'] - 1))
        shifted = [token]
        for t in reversed(history):
            shifted.append(t if eos not in shifted[1:] else eos)
        embed, offset = [], 0
        for head, size in enumerate(sizes):
            mixed = 0
            for j in range(2 + head // c['heads_per_ngram']):
                mixed ^= (shifted[j] * multipliers[j]) & ((1 << 64) - 1)
            signed = mixed if mixed < 1 << 63 else mixed - (1 << 64)
            row = offset + signed % size
            embed.extend(table[row * width:(row + 1) * width])
            offset += size
        state['history'] = (history + [token])[1:]
        self.record(name + '.embedding', embed)
        key_raw = self.matrix(embed, name + '.key_proj', self.w)
        key = self.norm(key_raw, name + '.norm_key', self.h)
        value = self.matrix(embed, name + '.value_proj', self.h)
        self.record(name + '.projections', key_raw + value)
        query = self.norm(hidden, name + '.norm_query', self.h)
        gated = []
        for s in range(self.s):
            at = s * self.h
            dot = bf(sum(bf(key[at + d] * query[at + d]) for d in range(self.h))) / math.sqrt(self.h)
            dot = bf(dot)
            gate = bf(sigmoid(bf(math.copysign(math.sqrt(max(abs(dot), 1e-6)), dot) if dot else 0.0)))
            gated.extend(bf(gate * v) for v in value)
        normed = self.norm(gated, name + '.norm_conv', self.h)
        taps, dilation = c['ple_conv_kernel_size'], c['ngram_size']
        tail_size = (taps - 1) * dilation
        tail = state.setdefault('ple_conv', [[0.0] * self.w for _ in range(tail_size)])
        conv = values(name + '.conv1d.weight', self.w * taps)
        window = tail + [normed]
        branch = [bf(silu(bf(sum(window[t * dilation][d] * conv[d * taps + t] for t in range(taps)))))
                  for d in range(self.w)]
        state['ple_conv'] = window[1:]
        self.record(name + '.gated', gated)
        self.record(name + '.normed', normed)
        return self.record(name + '.residual', [bf(x + bf(g + b)) for x, g, b in zip(hidden, gated, branch)])

    def step(self, token):
        c = self.c
        embedding = values('model.embed_tokens.weight', c['vocab_size'] * self.h)
        hidden = embedding[token * self.h:(token + 1) * self.h] * self.s
        self.record('embed', hidden)
        for layer, kind in enumerate(c['layer_types']):
            pre = f'model.layers.{layer}'
            if layer in self.ple_order:
                hidden = self.ple(hidden, token, layer, pre + '.ple')
            x, gates = self.hc(hidden, pre + '.attn_hyper_connection')
            branch = self.delta(x, layer, pre + '.linear_attn') if kind == 'linear_attention' else self.attention(x, layer, pre + '.self_attn')
            hidden = self.writeback(hidden, branch, gates)
            self.record(pre + '.attn_hyper_connection.residual', hidden)
            x, gates = self.hc(hidden, pre + '.mlp_hyper_connection')
            hidden = self.writeback(hidden, self.dense(x, pre + '.mlp'), gates)
            self.record(pre + '.mlp_hyper_connection.residual', hidden)
            self.trace.append({'position': len(self.tokens), 'layer': layer, 'hidden': hidden})
        mixed, _ = self.hc(hidden, 'model.hyper_connection_mixer', inject=False)
        logits = self.matrix(mixed, 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head', c['vocab_size'])
        self.record('head', logits)
        self.tokens.append(token)
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tokens', default='3,5,8,2,13,21,7,9,11,4,6,15')
    parser.add_argument('--probability', choices=('bf16', 'f32', 'cuda'), default='bf16',
                        help='Legacy normalized BF16, Metal FP32, or CUDA BF16 exponential weights before division; binary64 sums')
    parser.add_argument('--native-scalars', action='store_true', help='Declare FP32 RMS and rotary scalar nodes')
    parser.add_argument('--nodes', action='store_true', help='Include every observable storage node')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    oracle = Oracle(config, args.probability, args.nodes, args.native_scalars)
    tokens = [int(x) for x in args.tokens.split(',')]
    logits = [oracle.step(token) for token in tokens]
    result = {'schema': 1, 'oracle': 'binary64 with explicit BF16 nodes', 'tokens': tokens,
                      'logits': logits, 'picks': [max(range(len(row)), key=lambda i: (row[i], -i)) for row in logits],
                      'trace': oracle.trace}
    if args.native_scalars:
        result['oracle'] += '; FP32 RMS and rotary scalar nodes'
    if args.probability == 'cuda':
        result['oracle'] += '; BF16 unnormalized exponentials before PV, separate denominator'
    if args.probability == 'f32':
        result['oracle'] += '; unrounded split-softmax probabilities'
    if args.nodes:
        result['nodes'] = oracle.nodes
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
