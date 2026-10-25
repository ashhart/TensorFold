//! Numeric instruments bind real packed projections and per-row native state, independently of the f64 oracle.
const std = @import("std");
const n = @import("numeric_gpu.zig");
const m = @import("math.zig");
pub fn delta(g: *n.G, w: n.Model.Linear, state: anytype, layer: usize, x: n.Tensor, y: n.Tensor) !void {
    const c = g.c;
    const l = state.layout.layers[layer];
    const conv = n.view(u16, try state.current(l.conv));
    const recurrent = n.view(f32, try state.current(l.recurrent));
    const cw = n.vector(w.conv);
    const nw = n.vector(w.norm.buffer);
    const alog = n.vector(w.alog);
    const dt = n.vector(w.dt);
    for (0..x.rows) |r| {
        const p = n.row(x, r);
        const front = try g.a.alloc(f32, c.convWidth());
        for (front, 0..) |*out, i| {
            var acc: f32 = 0;
            for (0..c.conv) |tap| acc += cw[i * c.conv + tap] * m.f(if (tap == c.conv - 1) p[i] else conv[tap * front.len + i]);
            out.* = m.round(m.silu(if (w.conv_bf16) m.round(acc) else acc));
        }
        std.mem.copyForwards(u16, conv[0 .. conv.len - front.len], conv[front.len..]);
        @memcpy(conv[conv.len - front.len ..], p[0..front.len]);
        const q = front[0..c.keyWidth()];
        const k = front[c.keyWidth()..][0..c.keyWidth()];
        for (0..c.key_heads) |head| for ([_][]f32{ q[head * c.key_dim ..][0..c.key_dim], k[head * c.key_dim ..][0..c.key_dim] }, 0..) |v, which| {
            var ss: f32 = 0;
            for (v) |vv| ss += vv * vv;
            const inv = (if (which == 0) 1 / @sqrt(@as(f32, @floatFromInt(c.key_dim))) else @as(f32, 1)) / @sqrt(ss + 1e-6);
            for (v) |*vv| vv.* *= inv;
        };
        const z = p[c.convWidth()..][0..c.valueWidth()];
        const bp = p[c.convWidth() + c.valueWidth() ..][0..c.value_heads];
        const ap = p[c.convWidth() + c.valueWidth() + c.value_heads ..];
        for (0..c.value_heads) |head| {
            const kh = head / (c.value_heads / c.key_heads);
            const keys = k[kh * c.key_dim ..][0..c.key_dim];
            const queries = q[kh * c.key_dim ..][0..c.key_dim];
            const a = m.f(ap[head]) + dt[head];
            const decay = @exp(-@exp(alog[head]) * @log(1 + @exp(a)));
            const beta = m.round(m.sig(m.f(bp[head])));
            const out = n.row(y, r)[head * c.value_dim ..][0..c.value_dim];
            for (out, 0..) |*outv, d| {
                const s = recurrent[(head * c.value_dim + d) * c.key_dim ..][0..c.key_dim];
                var predicted: f32 = 0;
                for (s, keys) |*v, key| {
                    v.* *= decay;
                    predicted += v.* * key;
                }
                const correction = (front[2 * c.keyWidth() + head * c.value_dim + d] - predicted) * beta;
                var value: f32 = 0;
                for (s, keys, queries) |*v, key, query| {
                    v.* += key * correction;
                    value += v.* * query;
                }
                outv.* = m.bf(value);
            }
            var ss: f32 = 0;
            for (out) |v| ss += m.f(v) * m.f(v);
            const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(c.value_dim)) + c.eps);
            for (out, 0..) |*v, d| {
                const weighted = m.round(m.f(v.*) * inv) * nw[d];
                const normed = if (w.norm.mid_bf16) m.round(weighted) else weighted;
                const gate = m.f(z[head * c.value_dim + d]);
                v.* = m.bf(normed * (if (c.gate == .sigmoid) m.sig(gate) else m.silu(gate)));
            }
        }
        @memcpy(n.view(u16, try state.row(r, l.conv)), conv);
        @memcpy(n.view(f32, try state.row(r, l.recurrent)), recurrent);
    }
}
fn normalized(g: *n.G, x: []const u16, w: n.Model.Norm, pos: usize) ![]u16 {
    const out = try n.norm(g, w, x, x.len, g.c.eps);
    return m.rope(g.a, out, g.c.rotary, g.c.theta, pos);
}
pub fn indexed(g: *n.G, w: n.Model.Indexed, state: anytype, layer: usize, x: n.Tensor, y: n.Tensor) !void {
    const c = g.c;
    const l = state.layout.layers[layer];
    const keys = n.view(u16, try state.kv(l.key));
    const values = n.view(u16, try state.kv(l.value));
    const raw = n.view(u16, try state.kv(l.index));
    const pool = n.view(u16, try state.kv(l.pooled));
    const hd = c.head_dim;
    const di = c.index_dim;
    const kw = c.kv_heads * hd;
    for (0..x.rows) |r| {
        const pos = state.base_pos + r;
        const p = n.row(x, r);
        const qr = p[0 .. 2 * c.heads * hd];
        const kr = p[qr.len..][0..kw];
        const vr = p[qr.len + kw ..][0..kw];
        const ir = p[qr.len + 2 * kw ..];
        for (0..c.kv_heads) |head| @memcpy(keys[pos * kw + head * hd ..][0..hd], try normalized(g, kr[head * hd ..][0..hd], w.kn, pos));
        @memcpy(values[pos * kw ..][0..kw], vr);
        @memcpy(raw[pos * di ..][0..di], ir[c.index_heads * di ..]);
        const blocks = (pos + 1) / c.index_ratio;
        if ((pos + 1) % c.index_ratio == 0) {
            const b = blocks - 1;
            const pooled = try g.a.alloc(u16, di);
            for (pooled, 0..) |*v, d| {
                var sum: f32 = 0;
                for (0..c.index_ratio) |j| sum += m.f(raw[(b * c.index_ratio + j) * di + d]);
                v.* = m.bf(sum / @as(f32, @floatFromInt(c.index_ratio)));
            }
            @memcpy(pool[b * di ..][0..di], try normalized(g, pooled, w.ikn, b * c.index_ratio));
        }
        const scores = try g.a.alloc(f32, blocks);
        @memset(scores, 0);
        for (0..c.index_heads) |head| {
            const iq = try normalized(g, ir[head * di ..][0..di], w.iqn, pos);
            for (scores, 0..) |*score, b| {
                var dot: f32 = 0;
                for (iq, pool[b * di ..][0..di]) |q, k| dot += m.f(q) * m.f(k);
                score.* += @max(@as(f32, 0), dot) / @sqrt(@as(f32, @floatFromInt(di)));
            }
        }
        const chosen = try g.a.alloc(usize, blocks);
        for (chosen, 0..) |*v, i| v.* = i;
        std.mem.sort(usize, chosen, scores, struct {
            fn less(sc: []f32, a: usize, b: usize) bool {
                return sc[a] > sc[b] or (sc[a] == sc[b] and a < b);
            }
        }.less);
        const take = @min(blocks, c.index_budget / c.index_ratio);
        std.mem.sort(usize, chosen[0..take], {}, std.sort.asc(usize));
        var ids: std.ArrayList(usize) = .empty;
        for (chosen[0..take]) |b| for (0..c.index_ratio) |j| try ids.append(g.a, b * c.index_ratio + j);
        for (blocks * c.index_ratio..pos + 1) |i| try ids.append(g.a, i);
        for (0..c.heads) |head| {
            const q = try normalized(g, qr[head * 2 * hd ..][0..hd], w.qn, pos);
            const kh = head / (c.heads / c.kv_heads);
            const logits = try g.a.alloc(f32, ids.items.len);
            var maximum: f32 = -std.math.inf(f32);
            for (ids.items, logits) |id, *logit| {
                var dot: f32 = 0;
                for (q, keys[id * kw + kh * hd ..][0..hd]) |qv, k| dot += m.f(qv) * m.f(k);
                logit.* = dot / @sqrt(@as(f32, @floatFromInt(hd)));
                maximum = @max(maximum, logit.*);
            }
            var sum: f32 = 0;
            for (logits) |*v| {
                v.* = @exp(v.* - maximum);
                sum += v.*;
            }
            for (logits) |*v| v.* = m.round(v.* / sum);
            for (0..hd) |d| {
                var value: f32 = 0;
                for (logits, ids.items) |prob, id| value += prob * m.f(values[id * kw + kh * hd + d]);
                n.row(y, r)[head * hd + d] = m.bf(m.round(value) * m.round(m.sig(m.f(qr[head * 2 * hd + hd + d]))));
            }
        }
    }
}
