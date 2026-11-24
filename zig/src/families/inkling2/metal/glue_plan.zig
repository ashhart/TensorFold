//! Pure Inkling 2 Mini glue launch shapes and independent FP64 arithmetic.
const std = @import("std");

pub const hidden = 2304;
pub const expert_count = 128;
pub const intermediate = 1152;
pub const top_k = 8;
pub const ar_scale: f64 = @as(f32, @bitCast(@as(u32, 0x3caaaaab)));

pub const Kind = enum {
    embed_norm,
    ar,
    conv_static,
    conv_dynamic,
    router,
    swiglu,
    logits,
    expert_pick,
    expert_combine,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .embed_norm => "tf_inkling2_embed_norm",
            .ar => "tf_inkling2_ar",
            .conv_static => "tf_inkling2_conv_static",
            .conv_dynamic => "tf_inkling2_conv_dynamic",
            .router => "tf_inkling2_router",
            .swiglu => "tf_inkling2_swiglu",
            .logits => "tf_inkling2_logits",
            .expert_pick => "tf_inkling2_expert_pick",
            .expert_combine => "tf_inkling2_expert_combine",
        };
    }
    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            .embed_norm => "metal inkling2 glue embed_norm tf_inkling2_embed_norm",
            .ar => "metal inkling2 glue ar tf_inkling2_ar",
            .conv_static => "metal inkling2 glue conv_static tf_inkling2_conv_static",
            .conv_dynamic => "metal inkling2 glue conv_dynamic tf_inkling2_conv_dynamic",
            .router => "metal inkling2 glue router tf_inkling2_router",
            .swiglu => "metal inkling2 glue swiglu tf_inkling2_swiglu",
            .logits => "metal inkling2 glue logits tf_inkling2_logits",
            .expert_pick => "metal inkling2 glue expert_pick tf_inkling2_expert_pick",
            .expert_combine => "metal inkling2 glue expert_combine tf_inkling2_expert_combine",
        };
    }
};

pub const Call = struct { rows: usize, n: usize, k: usize };
pub fn valid(kind: Kind, c: Call) bool {
    const key_depth: usize = if (kind == .expert_combine) top_k else 4;
    const max_width: usize = if (kind == .conv_dynamic) 1024 else hidden;
    if (c.rows == 0 or c.rows > 64) return false;
    return switch (kind) {
        .embed_norm => c.k == hidden and c.n == hidden,
        .router => c.k == hidden and c.n == top_k,
        .logits => c.k == hidden and c.n > 0 and c.n <= 200058,
        .ar => c.k == hidden and c.n == hidden and c.rows >= 2 and c.rows <= 5,
        .conv_static, .conv_dynamic, .expert_combine => c.k == key_depth and c.n > 0 and c.n <= max_width,
        .swiglu => c.k == 2 and c.n > 0 and c.n <= intermediate,
        .expert_pick => c.k == top_k and c.n > 0 and c.n <= hidden,
    };
}
pub fn count(kind: Kind, c: Call) usize {
    const copies: usize = if (kind == .expert_pick) top_k else 1;
    return c.rows * c.n * copies;
}

pub fn roundBf16(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}
pub fn cap(x: f64, c: f64) f64 {
    return c * std.math.tanh(x / c);
}
pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + std.math.exp(-x));
}

/// One normalized BF16 embedding result, before its final BF16 store.
pub fn embedValue(x: []const f32, weight: []const f32, d: usize) f64 {
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + 1e-6);
    return @as(f64, weight[d]) * (@as(f64, x[d]) * inv);
}

/// Accumulation alternatives for the AR host model. Null means the independent FP64 oracle.
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

/// Squares for one AR RMS reduction, optionally under a precision/order model.
pub fn arSourceSquares(blocks: []const f32, partial: []const f32, sources: usize, s: usize, sim: ?ArSum) f64 {
    var terms: [hidden]f64 = undefined;
    for (&terms, 0..) |*term, d| {
        const x = if (s + 1 == sources) partial[d] else blocks[s * hidden + d];
        term.* = @as(f64, x) * x;
    }
    return arSumIn(&terms, sim);
}

/// One normalized AR query dot. Its terms are in ascending hidden-coordinate order.
pub fn arQueryDot(blocks: []const f32, partial: []const f32, queries: []const f32, sources: usize, q: usize, s: usize, inverse: f64, sim: ?ArSum) f64 {
    var terms: [hidden]f64 = undefined;
    for (&terms, 0..) |*term, d| {
        const x = if (s + 1 == sources) partial[d] else blocks[s * hidden + d];
        term.* = @as(f64, queries[q * hidden + d]) * (@as(f64, x) * inverse);
    }
    return arSumIn(&terms, sim);
}

/// AR with explicitly modeled reductions; non-reduction arithmetic remains FP64.
pub fn blockResidualIn(blocks: []const f32, partial: []const f32, queries: []const f32, sources: usize, q: usize, sim: ?ArSum, out: []f64) void {
    var inv: [5]f64 = undefined;
    var weight: [5]f64 = undefined;
    var peak = -std.math.inf(f64);
    for (0..sources) |s| {
        const ss = arSourceSquares(blocks, partial, sources, s, sim);
        inv[s] = 1 / @sqrt(ss / hidden + 1e-6);
        const dot = arQueryDot(blocks, partial, queries, sources, q, s, inv[s], sim);
        weight[s] = dot * ar_scale;
        peak = @max(peak, weight[s]);
    }
    var den_terms: [5]f64 = undefined;
    for (0..sources) |s| {
        den_terms[s] = std.math.exp(weight[s] - peak);
        weight[s] = den_terms[s];
    }
    const den = arSumIn(den_terms[0..sources], sim);
    var terms: [5]f64 = undefined;
    for (0..hidden) |d| {
        for (0..sources) |s| {
            const x = if (s + 1 == sources) partial[d] else blocks[s * hidden + d];
            terms[s] = weight[s] / den * (@as(f64, x) * inv[s]);
        }
        out[d] = arSumIn(terms[0..sources], sim);
    }
}

/// AR over BF16 blocks widened to f32 and an unrounded f32 partial. FP64 is only the oracle.
pub fn blockResidual(blocks: []const f32, partial: []const f32, queries: []const f32, sources: usize, q: usize, out: []f64) void {
    blockResidualIn(blocks, partial, queries, sources, q, null, out);
}

pub fn staticConv(taps: []const f32, weight: []const f32, channels: usize, r: usize, c: usize) f64 {
    var sum: f64 = taps[(r + 3) * channels + c];
    for (0..4) |tap| sum += @as(f64, weight[c * 4 + tap]) * taps[(r + tap) * channels + c];
    return sum / @sqrt(5.0);
}

pub fn dynamicConv(taps: []const f32, input: []const f32, static_w: []const f32, projection: []const f32, channels: usize, r: usize, c: usize) f64 {
    var sum: f64 = taps[(r + 3) * channels + c];
    for (0..4) |lag| {
        var dot: f64 = 0;
        for (0..hidden) |d| dot += @as(f64, projection[(4 * (c / 128) + lag) * hidden + d]) * input[r * hidden + d];
        const w = @as(f64, static_w[lag * channels + c]) + roundBf16(dot);
        sum += w * taps[(r + 3 - lag) * channels + c];
    }
    return sum / 3;
}

pub const Route = struct { ids: [top_k]u32, weights: [top_k]f64 };
pub fn route(x: []const f32, matrix: []const f32, bias: []const f32) Route {
    var score: [expert_count]f64 = undefined;
    var used: [expert_count]bool = @splat(false);
    var out: Route = undefined;
    for (0..expert_count) |e| {
        var dot: f64 = 0;
        for (0..hidden) |d| dot += @as(f64, matrix[e * hidden + d]) * x[d];
        score[e] = sigmoid(dot);
    }
    for (0..top_k) |j| {
        var best: ?usize = null;
        for (0..expert_count) |e| if (!used[e]) {
            if (best == null or score[e] + bias[e] > score[best.?] + bias[best.?]) best = e;
        };
        out.ids[j] = @intCast(best.?);
        out.weights[j] = score[best.?];
        used[best.?] = true;
    }
    std.mem.sort(u32, &out.ids, {}, std.sort.asc(u32));
    // Recover scores after sorting IDs, not route-order positions.
    var den: f64 = 1e-20;
    for (out.ids, 0..) |id, j| {
        out.weights[j] = score[id];
        den += score[id];
    }
    for (&out.weights) |*w| w.* = w.* / den * 8;
    return out;
}

pub fn swiglu(gate: f32, up: f32) f64 {
    return cap(gate, 4) * sigmoid(gate) * cap(up, 25);
}
pub fn logit(h: []const f32, unembed: []const f32, v: usize) f64 {
    var dot: f64 = 0;
    for (0..hidden) |d| dot += @as(f64, unembed[v * hidden + d]) * h[d];
    return cap(roundBf16(dot), 30);
}
pub fn combine(picked: []const f32, weights: []const f32, channels: usize, d: usize) f64 {
    var sum: f64 = 0;
    for (0..top_k) |j| sum += @as(f64, weights[j]) * picked[j * channels + d];
    return sum;
}

pub fn pick(all_experts: []const f32, ids: []const u32, channels: usize, row: usize, j: usize, d: usize) f32 {
    return all_experts[(row * expert_count + ids[row * top_k + j]) * channels + d];
}
