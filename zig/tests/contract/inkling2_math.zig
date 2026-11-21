//! Float64 formulas of Inkling 2's short convolutions over the whole stream: no ring, no history buffer, no chunking.
const std = @import("std");
const contract = @import("contract");
const g = contract.gates;
const c = contract.spec;

pub const taps = 4;

/// `stream` rows at p*channels; conv: out = (sum_j w[ch][j] x[t-3+j][ch] + x[t][ch]) / sqrt(5); `mag` = sum |terms|.
pub fn staticConv(stream: []const f64, w: []const f64, channels: usize, first: usize, rows: usize, out: []f64, mag: []f64) void {
    const root = @sqrt(@as(f64, taps + 1));
    for (0..rows) |r| for (0..channels) |ch| {
        const t = first + r;
        var sum: f64 = 0;
        var m: f64 = 0;
        for (0..taps) |j| {
            // before position 0 the input is the zero padding: its product with the weight still happens (NaN * 0)
            const x = if (t + j < taps - 1) 0 else stream[(t + j - (taps - 1)) * channels + ch];
            const term = w[ch * taps + j] * x;
            sum += term;
            m += @abs(term);
        }
        const cur = stream[t * channels + ch];
        out[r * channels + ch] = (sum + cur) / root;
        mag[r * channels + ch] = (m + @abs(cur)) / root;
    };
}

/// The same sums with the terms replayed in `acc` (gate 4): the product of each tap, one chain in `order`.
pub fn staticConvIn(acc: c.Accum, order: g.Order, stream: []const f64, w: []const f64, channels: usize, first: usize, rows: usize, out: []f64) void {
    const root = @sqrt(@as(f64, taps + 1));
    for (0..rows) |r| for (0..channels) |ch| {
        const t = first + r;
        var terms: [taps]f64 = undefined;
        for (0..taps) |j| terms[j] = w[ch * taps + j] * (if (t + j < taps - 1) 0 else stream[(t + j - (taps - 1)) * channels + ch]);
        out[r * channels + ch] = (contract.rows.sumIn(acc, order, &terms) + stream[t * channels + ch]) / root;
    };
}

/// weight = static[lag][ch] + dyn[r][head][lag], head = ch / (channels/kv_heads), out = (sum_lag w x[t-lag]+x[t]) / 3.
pub fn dynamicConv(stream: []const f64, static_w: []const f64, dyn: []const f64, channels: usize, kv_heads: usize, first: usize, rows: usize, out: []f64, mag: []f64) void {
    for (0..rows) |r| for (0..channels) |ch| {
        const t = first + r;
        const head = ch / (channels / kv_heads);
        var sum: f64 = 0;
        var m: f64 = 0;
        for (0..taps) |lag| {
            const x = if (t < lag) 0 else stream[(t - lag) * channels + ch];
            const weight = static_w[lag * channels + ch] + dyn[r * kv_heads * taps + head * taps + lag];
            const term = weight * x;
            sum += term;
            m += @abs(term);
        }
        const cur = stream[t * channels + ch];
        out[r * channels + ch] = (sum + cur) / 3;
        mag[r * channels + ch] = (m + @abs(cur)) / 3;
    };
}

pub fn dynamicConvIn(acc: c.Accum, order: g.Order, stream: []const f64, static_w: []const f64, dyn: []const f64, channels: usize, kv_heads: usize, first: usize, rows: usize, out: []f64) void {
    for (0..rows) |r| for (0..channels) |ch| {
        const t = first + r;
        const head = ch / (channels / kv_heads);
        var terms: [taps]f64 = undefined;
        for (0..taps) |lag| {
            terms[lag] = (static_w[lag * channels + ch] + dyn[r * kv_heads * taps + head * taps + lag]) * (if (t < lag) 0 else stream[(t - lag) * channels + ch]);
        }
        out[r * channels + ch] = (contract.rows.sumIn(acc, order, &terms) + stream[t * channels + ch]) / 3;
    };
}

test "static conv: zero before position 0 and tap 3 on the current row" {
    const stream = [_]f64{ 1, 2, 3, 4, 5 };
    const w = [_]f64{ 10, 100, 1000, 10000 };
    var out: [5]f64 = undefined;
    var mag: [5]f64 = undefined;
    staticConv(&stream, &w, 1, 0, 5, &out, &mag);
    const root = @sqrt(5.0);
    try std.testing.expectApproxEqAbs((10000.0 * 1 + 1) / root, out[0], 1e-9);
    try std.testing.expectApproxEqAbs((1000.0 * 1 + 10000.0 * 2 + 2) / root, out[1], 1e-9);
    try std.testing.expectApproxEqAbs((10.0 * 1 + 100.0 * 2 + 1000.0 * 3 + 10000.0 * 4 + 4) / root, out[3], 1e-9);
}

test "dynamic conv: lag 0 is the current row and heads share their weights" {
    const stream = [_]f64{ 1, 1, 2, 2, 3, 3 }; // 3 positions x 2 channels
    const static_w = [_]f64{ 1, 1, 0, 0, 0, 0, 0, 0 };
    const dyn: [24]f64 = @splat(0); // 3 rows x 2 heads x 4 lags
    var out: [6]f64 = undefined;
    var mag: [6]f64 = undefined;
    dynamicConv(&stream, &static_w, &dyn, 2, 2, 0, 3, &out, &mag);
    // row 0: lag 0 weighs 1 + 0 -> 1 * x0, plus the residual x0, over 3
    try std.testing.expectApproxEqAbs(2.0 / 3.0, out[0], 1e-12);
    // row 1, channel 0: lag 0 weighs 1 -> x1 = 2, lag 1 weighs 0, residual 2: (2 + 2) / 3
    try std.testing.expectApproxEqAbs(4.0 / 3.0, out[2], 1e-12);
}
