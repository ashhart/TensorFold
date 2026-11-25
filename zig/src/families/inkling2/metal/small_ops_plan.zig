//! Pure Inkling 2 Small (H=4608, E=384, 8 blocks/taps) launch geometry and FP64 references.
const std = @import("std");

pub const hidden: usize = 4608;
pub const expert_count: usize = 384;
pub const topk: usize = 8;
pub const block_slots: usize = 8;
pub const tap_channels: usize = 8;
pub const max_sources: usize = block_slots + 1;
pub const max_rows: usize = 256;
pub const vocabulary: usize = 200058;
pub const eps: f64 = 1e-6;
pub const ar_scale: f64 = @as(f32, @bitCast(@as(u32, 0x3c715bef))); // 0x1.e2b7dep-7f = binary32(1/sqrt(4608))

pub const Kind = enum {
    embed_norm,
    ar,
    conv_dynamic,
    router,
    logits,
    attention_partial,
    capture_tap,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .embed_norm => "tf_ink2_small_embed_norm",
            .ar => "tf_ink2_small_ar",
            .conv_dynamic => "tf_ink2_small_conv_dynamic",
            .router => "tf_ink2_small_router",
            .logits => "tf_ink2_small_logits",
            .attention_partial => "tf_ink2_small_attention_partial",
            .capture_tap => "tf_ink2_small_capture_tap",
        };
    }

    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            .embed_norm => "metal inkling2 small embed_norm tf_ink2_small_embed_norm",
            .ar => "metal inkling2 small ar tf_ink2_small_ar",
            .conv_dynamic => "metal inkling2 small conv_dynamic tf_ink2_small_conv_dynamic",
            .router => "metal inkling2 small router tf_ink2_small_router",
            .logits => "metal inkling2 small logits tf_ink2_small_logits",
            .attention_partial => "metal inkling2 small attention_partial tf_ink2_small_attention_partial",
            .capture_tap => "metal inkling2 small capture_tap tf_ink2_small_capture_tap",
        };
    }

    pub fn threads(k: Kind) u32 {
        return switch (k) {
            .logits => 32,
            else => 256,
        };
    }

    pub fn tgMemory(k: Kind) u32 {
        return switch (k) {
            .embed_norm => 48,
            .ar => 656,
            .conv_dynamic => 128,
            .router => 4688,
            .logits, .attention_partial, .capture_tap => 0,
        };
    }
};

pub const Call = struct { rows: usize, n: usize, k: usize };

pub fn valid(kind: Kind, c: Call) bool {
    if (c.rows == 0 or c.rows > max_rows or c.n == 0) return false;
    return switch (kind) {
        .embed_norm => c.n == hidden and c.k == hidden,
        .ar => c.rows >= 2 and c.rows <= max_sources and c.n == hidden and c.k == hidden,
        .conv_dynamic => c.n <= 1024 and c.n % 128 == 0 and c.k == 4,
        .router => c.n == topk and c.k == hidden,
        .logits => c.n <= vocabulary and c.k == hidden,
        .attention_partial => c.n == hidden and c.k <= block_slots,
        .capture_tap => c.n == hidden and c.k < tap_channels,
    };
}

pub fn outputCount(kind: Kind, c: Call) usize {
    return switch (kind) {
        .capture_tap => c.rows * tap_channels * hidden,
        else => c.rows * c.n,
    };
}

pub fn roundBf16(v: f64) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}

pub fn sigmoid(x: f64) f64 {
    const e = @exp(-@abs(x));
    return if (x < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
}

pub fn softcap(x: f64, c: f64) f64 {
    return c * std.math.tanh(x / c);
}

pub fn embedNormRef(ids: []const u32, table: []const f32, weight: []const f32, out: []f64) void {
    for (ids, 0..) |id, r| {
        const row = table[@as(usize, id) * hidden ..][0..hidden];
        var ss: f64 = 0;
        for (row) |v| ss += @as(f64, v) * @as(f64, v);
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(hidden)) + eps);
        for (0..hidden) |d| out[r * hidden + d] = @as(f64, weight[d]) * (@as(f64, row[d]) * inv);
    }
}

pub const ArAccum = enum { none, f32, f16, bf16 };
pub const ArOrder = enum { sequential, pairwise, strided };
pub const ArSum = struct { accum: ArAccum, order: ArOrder };

fn arRound(v: f64, accum: ArAccum) f64 {
    return switch (accum) {
        .none => v,
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => roundBf16(v),
    };
}

fn arChain(terms: []const f64, sim: ArSum, stride: usize, first: usize) f64 {
    var total: f64 = 0;
    var i = first;
    while (i < terms.len) : (i += stride) total = arRound(total + arRound(terms[i], sim.accum), sim.accum);
    return total;
}

fn arHalves(terms: []const f64, sim: ArSum) f64 {
    if (terms.len == 0) return 0;
    if (terms.len == 1) return arRound(terms[0], sim.accum);
    const mid = terms.len / 2;
    return arRound(arHalves(terms[0..mid], sim) + arHalves(terms[mid..], sim), sim.accum);
}

fn arSumIn(terms: []const f64, sim: ?ArSum) f64 {
    if (sim) |s| return switch (s.order) {
        .sequential => arChain(terms, s, 1, 0),
        .pairwise => arHalves(terms, s),
        .strided => blk: {
            var lanes: [32]f64 = @splat(0);
            for (&lanes, 0..) |*lane, i| lane.* = arChain(terms, s, 32, i);
            break :blk arHalves(&lanes, s);
        },
    };
    var total: f64 = 0;
    for (terms) |term| total += term;
    return total;
}

pub fn arIn(blocks: []const f32, partial: []const f32, queries: []const f32, sources: usize, q_count: usize, sim: ?ArSum, out: []f64) void {
    var inv: [max_sources]f64 = undefined;
    var terms: [hidden]f64 = undefined;
    for (0..sources) |s| {
        for (&terms, 0..) |*term, d| {
            const v: f64 = if (s + 1 == sources) partial[d] else blocks[s * hidden + d];
            term.* = v * v;
        }
        const ss = arSumIn(&terms, sim);
        inv[s] = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(hidden)) + eps);
    }
    for (0..q_count) |q| {
        var w: [max_sources]f64 = undefined;
        var peak: f64 = -std.math.inf(f64);
        for (0..sources) |s| {
            for (&terms, 0..) |*term, d| {
                const v: f64 = if (s + 1 == sources) partial[d] else blocks[s * hidden + d];
                term.* = @as(f64, queries[q * hidden + d]) * (v * inv[s]);
            }
            w[s] = arSumIn(&terms, sim) * ar_scale;
            if (w[s] > peak) peak = w[s];
        }
        var den_terms: [max_sources]f64 = undefined;
        for (0..sources) |s| {
            den_terms[s] = @exp(w[s] - peak);
            w[s] = den_terms[s];
        }
        const den = arSumIn(den_terms[0..sources], sim);
        for (0..sources) |s| w[s] /= den;
        var src_terms: [max_sources]f64 = undefined;
        for (0..hidden) |d| {
            for (0..sources) |s| {
                const v: f64 = if (s + 1 == sources) partial[d] else blocks[s * hidden + d];
                src_terms[s] = w[s] * (v * inv[s]);
            }
            out[q * hidden + d] = arSumIn(src_terms[0..sources], sim);
        }
    }
}

pub fn arRef(blocks: []const f32, partial: []const f32, queries: []const f32, sources: usize, q_count: usize, out: []f64) void {
    arIn(blocks, partial, queries, sources, q_count, null, out);
}

pub fn convDynamicRef(taps: []const f32, input: []const f32, static_w: []const f32, proj: []const f32, rows: usize, channels: usize, out: []f64) void {
    var projected: [32]f64 = undefined;
    for (0..rows) |r| {
        for (0..32) |t_i| {
            var dot: f64 = 0;
            for (0..hidden) |d| dot += @as(f64, proj[t_i * hidden + d]) * @as(f64, input[r * hidden + d]);
            projected[t_i] = roundBf16(dot);
        }
        for (0..channels) |c| {
            var sum: f64 = taps[(r + 3) * channels + c];
            for (0..4) |lag| {
                const w = @as(f64, static_w[lag * channels + c]) + projected[4 * (c / 128) + lag];
                sum += w * @as(f64, taps[(r + 3 - lag) * channels + c]);
            }
            out[r * channels + c] = sum / 3.0;
        }
    }
}

pub fn routerRef(x: []const f32, weight: []const f32, bias: []const f32, rows: usize, ids_out: []u32, weights_out: []f64) void {
    var scores: [expert_count]f64 = undefined;
    var keys: [expert_count]f64 = undefined;
    for (0..rows) |r| {
        for (0..expert_count) |e| {
            var dot: f64 = 0;
            for (0..hidden) |d| dot += @as(f64, weight[e * hidden + d]) * @as(f64, x[r * hidden + d]);
            scores[e] = sigmoid(dot);
            keys[e] = scores[e] + @as(f64, bias[e]);
        }
        var used: [expert_count]bool = @splat(false);
        var picked: [topk]u32 = undefined;
        var chosen: [topk]f64 = undefined;
        for (0..topk) |j| {
            var best: usize = expert_count;
            for (0..expert_count) |e| {
                if (used[e]) continue;
                if (best == expert_count or keys[e] > keys[best]) best = e;
            }
            used[best] = true;
            picked[j] = @intCast(best);
            chosen[j] = scores[best];
        }
        for (1..topk) |j| {
            const id = picked[j];
            const sc = chosen[j];
            var k = j;
            while (k > 0 and picked[k - 1] > id) : (k -= 1) {
                picked[k] = picked[k - 1];
                chosen[k] = chosen[k - 1];
            }
            picked[k] = id;
            chosen[k] = sc;
        }
        var den: f64 = 1e-20;
        for (chosen) |sc| den += sc;
        for (0..topk) |j| {
            ids_out[r * topk + j] = picked[j];
            weights_out[r * topk + j] = (chosen[j] / den) * 8.0;
        }
    }
}

pub fn logitsRef(h: []const f32, unembed: []const f32, rows: usize, vocab: usize, out: []f64) void {
    for (0..rows) |r| {
        for (0..vocab) |v| {
            var dot: f64 = 0;
            for (0..hidden) |d| dot += @as(f64, unembed[v * hidden + d]) * @as(f64, h[r * hidden + d]);
            out[r * vocab + v] = softcap(roundBf16(dot), 30.0);
        }
    }
}

pub fn attentionPartialRef(p: f32, x: f32, boundary: usize) f64 {
    std.debug.assert(boundary <= block_slots);
    return if (boundary != 0) x else @as(f32, p + x);
}

pub fn blockOffset(row: usize, boundary: usize) usize {
    std.debug.assert(row < max_rows and boundary >= 1 and boundary <= block_slots);
    return (row * block_slots + boundary - 1) * hidden;
}

pub fn tapOffset(row: usize, channel: usize) usize {
    std.debug.assert(row < max_rows and channel < tap_channels);
    return (row * tap_channels + channel) * hidden;
}
