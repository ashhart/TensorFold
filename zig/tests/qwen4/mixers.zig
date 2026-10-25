//! Independent FP32 test execution of recurrent and indexed mixers, with all persistent state explicit.
const std = @import("std");
const m = @import("math.zig");
const State = @import("state.zig").State;
const Tensor = m.Tensor;
fn part(a: std.mem.Allocator, stem: []const u8, tail: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}.{s}", .{ stem, tail });
}
pub fn delta(o: anytype, layer: usize, state: *State, x: Tensor) !Tensor {
    const c = o.c;
    const pre = try o.name(layer, "linear_attn");
    const qkv = try m.project(o.a, x, try part(o.a, pre, "in_proj_qkv"), c.convWidth());
    const z = try m.project(o.a, x, try part(o.a, pre, "in_proj_z"), c.valueWidth());
    const ap = try m.project(o.a, x, try part(o.a, pre, "in_proj_a"), c.value_heads);
    const bp = try m.project(o.a, x, try part(o.a, pre, "in_proj_b"), c.value_heads);
    const conv = try m.source(o.a, try part(o.a, pre, "conv1d.weight"), c.convWidth() * c.conv, false, false);
    const alog = try m.source(o.a, try part(o.a, pre, "A_log"), c.value_heads, false, false);
    const dt = try m.source(o.a, try part(o.a, pre, "dt_bias"), c.value_heads, false, false);
    const nw = try m.source(o.a, try part(o.a, pre, "norm.weight"), c.value_dim, false, true);
    const y = try m.tensor(o.a, x.rows, c.valueWidth());
    const cache = &state.cache[layer];
    for (0..x.rows) |row| {
        const ch = c.convWidth();
        const front = try o.a.alloc(f32, ch);
        for (front, 0..) |*v, i| {
            var acc: f32 = 0;
            for (0..c.conv) |tap| acc += m.f(if (tap == c.conv - 1) qkv.row(row)[i] else cache.conv[tap * ch + i]) * m.f(conv[i * c.conv + tap]);
            v.* = m.round(m.silu(m.round(acc)));
        }
        std.mem.copyForwards(u16, cache.conv[0 .. cache.conv.len - ch], cache.conv[ch..]);
        @memcpy(cache.conv[cache.conv.len - ch ..], qkv.row(row));
        const q = front[0..c.keyWidth()];
        const k = front[c.keyWidth()..][0..c.keyWidth()];
        for (0..c.key_heads) |head| {
            for ([_][]f32{ q[head * c.key_dim ..][0..c.key_dim], k[head * c.key_dim ..][0..c.key_dim] }, 0..) |v, which| {
                var ss: f32 = 0;
                for (v) |vv| ss += vv * vv;
                const inv = (if (which == 0) 1 / @sqrt(@as(f32, @floatFromInt(c.key_dim))) else @as(f32, 1)) / @sqrt(ss + 1e-6);
                for (v) |*vv| vv.* *= inv;
            }
        }
        for (0..c.value_heads) |head| {
            const kh = head / (c.value_heads / c.key_heads);
            const keys = k[kh * c.key_dim ..][0..c.key_dim];
            const queries = q[kh * c.key_dim ..][0..c.key_dim];
            const a = m.f(ap.row(row)[head]) + m.f(dt[head]);
            const decay = @exp(-@exp(m.f(alog[head])) * @log(1 + @exp(a)));
            const beta = m.round(m.sig(m.f(bp.row(row)[head])));
            for (0..c.value_dim) |d| {
                const s = cache.recurrent[(head * c.value_dim + d) * c.key_dim ..][0..c.key_dim];
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
                y.row(row)[head * c.value_dim + d] = m.bf(value);
            }
            const ys = y.row(row)[head * c.value_dim ..][0..c.value_dim];
            var ss: f32 = 0;
            for (ys) |v| ss += m.f(v) * m.f(v);
            const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(c.value_dim)) + c.eps);
            for (ys, 0..) |*v, d| {
                const norm_value = m.round(m.round(m.f(v.*) * inv) * m.f(nw[d]));
                const gate = m.f(z.row(row)[head * c.value_dim + d]);
                v.* = m.bf(norm_value * (if (c.gate == .sigmoid) m.sig(gate) else m.silu(gate)));
            }
        }
    }
    return m.project(o.a, y, try part(o.a, pre, "out_proj"), c.hidden);
}
fn normalized(o: anytype, bits: []u16, stem: []const u8, pos: usize) ![]u16 {
    const t = try m.norm(o.a, .{ .bits = bits, .rows = 1, .width = bits.len }, stem, bits.len, o.c.eps);
    return m.rope(o.a, t.bits, o.c.rotary, o.c.theta, pos);
}
pub fn indexed(o: anytype, layer: usize, state: *State, x: Tensor) !Tensor {
    const c = o.c;
    const pre = try o.name(layer, "self_attn");
    const hd = c.head_dim;
    const nh = c.heads;
    const nk = c.kv_heads;
    const ih = c.index_heads;
    const di = c.index_dim;
    const qraw = try m.project(o.a, x, try part(o.a, pre, "q_proj"), 2 * nh * hd);
    const kraw = try m.project(o.a, x, try part(o.a, pre, "k_proj"), nk * hd);
    const vraw = try m.project(o.a, x, try part(o.a, pre, "v_proj"), nk * hd);
    const iraw = try m.project(o.a, x, try part(o.a, pre, "indexer.index_qk_proj"), (ih + 1) * di);
    const out = try m.tensor(o.a, x.rows, nh * hd);
    const cache = &state.cache[layer];
    for (0..x.rows) |row| {
        const pos = state.pos + row;
        const queries = try o.a.alloc([]u16, nh);
        for (queries, 0..) |*q, head| q.* = try normalized(o, qraw.row(row)[head * 2 * hd ..][0..hd], try part(o.a, pre, "q_norm"), pos);
        const keys = try o.a.alloc(u16, nk * hd);
        for (0..nk) |head| @memcpy(keys[head * hd ..][0..hd], try normalized(o, kraw.row(row)[head * hd ..][0..hd], try part(o.a, pre, "k_norm"), pos));
        try cache.keys.append(o.a, keys);
        try cache.values.append(o.a, try o.a.dupe(u16, vraw.row(row)));
        try cache.index.append(o.a, try o.a.dupe(u16, iraw.row(row)[ih * di ..]));
        const iq = try o.a.alloc([]u16, ih);
        for (iq, 0..) |*q, head| q.* = try normalized(o, iraw.row(row)[head * di ..][0..di], try part(o.a, pre, "indexer.q_layernorm"), pos);
        const blocks = (pos + 1) / c.index_ratio;
        const scores = try o.a.alloc(f32, blocks);
        for (scores, 0..) |*score, block| {
            const pooled = try o.a.alloc(u16, di);
            for (pooled, 0..) |*v, d| {
                var total: f32 = 0;
                for (0..c.index_ratio) |j| total += m.f(cache.index.items[block * c.index_ratio + j][d]);
                v.* = m.bf(total / @as(f32, @floatFromInt(c.index_ratio)));
            }
            const rotated = try normalized(o, pooled, try part(o.a, pre, "indexer.k_layernorm"), block * c.index_ratio);
            var total: f32 = 0;
            for (iq) |head| {
                var dot: f32 = 0;
                for (head, rotated) |q, k| dot += m.f(q) * m.f(k);
                total += @max(@as(f32, 0), dot);
            }
            score.* = total / @sqrt(@as(f32, @floatFromInt(di)));
        }
        const chosen = try o.a.alloc(usize, blocks);
        for (chosen, 0..) |*v, i| v.* = i;
        std.mem.sort(usize, chosen, scores, struct {
            fn less(sc: []f32, lhs: usize, rhs: usize) bool {
                return sc[lhs] > sc[rhs] or (sc[lhs] == sc[rhs] and lhs < rhs);
            }
        }.less);
        const take = @min(blocks, c.index_budget / c.index_ratio);
        std.mem.sort(usize, chosen[0..take], {}, std.sort.asc(usize));
        var ids: std.ArrayList(usize) = .empty;
        for (chosen[0..take]) |block| for (0..c.index_ratio) |j| {
            try ids.append(o.a, block * c.index_ratio + j);
        };
        for (blocks * c.index_ratio..pos + 1) |i| try ids.append(o.a, i);
        for (queries, 0..) |query, head| {
            const kh = head / (nh / nk);
            const logits = try o.a.alloc(f32, ids.items.len);
            var maximum: f32 = -std.math.inf(f32);
            for (ids.items, logits) |id, *logit| {
                var dot: f32 = 0;
                for (query, cache.keys.items[id][kh * hd ..][0..hd]) |q, k| dot += m.f(q) * m.f(k);
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
                for (logits, ids.items) |p, id| value += p * m.f(cache.values.items[id][kh * hd + d]);
                out.row(row)[head * hd + d] = m.bf(m.round(value) * m.round(m.sig(m.f(qraw.row(row)[head * 2 * hd + hd + d]))));
            }
        }
    }
    return m.project(o.a, out, try part(o.a, pre, "o_proj"), c.hidden);
}
