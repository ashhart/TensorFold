//! Layer-0 hyper-connection host checks: mode 0 writes each 256-wide chunk's sum of squares.

const std = @import("std");

/// glue._hc_writeback mode 0: pss[c, s] is a pairwise sum of squares of stream s, chunk c.
pub fn partialSums(h: []const u16, dims: usize, streams: usize, out: []f32) !void {
    const block: usize = 256;
    if (dims == 0 or dims % block != 0 or h.len != streams * dims) return error.UnexpectedTensor;
    const nc = dims / block;
    if (out.len != nc * streams) return error.UnexpectedTensor;
    for (0..nc) |c| {
        for (0..streams) |s| {
            var buf: [256]f32 = undefined;
            for (0..block) |i| {
                const bits: u32 = h[s * dims + c * block + i];
                const v: f32 = @bitCast(bits << 16);
                buf[i] = v * v;
            }
            var n: usize = block;
            while (n > 1) {
                n /= 2;
                for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
            }
            out[c * streams + s] = buf[0];
        }
    }
}

test "mode 0 partial sums are the squares of one 256-wide chunk" {
    var h: [256]u16 = @splat(0);
    h[0] = 0x3f80;
    var out: [1]f32 = undefined;
    try partialSums(&h, 256, 1, &out);
    try std.testing.expectEqual(@as(f32, 1), out[0]);
}

/// Steps between two positive finite floats. Triton's block sum lands a couple of steps from a pairwise sum.
pub fn ulps(a: f32, b: f32) u32 {
    const xa: i32 = @bitCast(a);
    const xb: i32 = @bitCast(b);
    const d = xa - xb;
    return @intCast(if (d < 0) -d else d);
}

test "ulps counts steps between positive floats" {
    const a: f32 = 1;
    const b: f32 = @bitCast(@as(u32, @bitCast(a)) + 2);
    try std.testing.expectEqual(@as(u32, 2), ulps(a, b));
}

pub const tile_bn: usize = 64;

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// MLX (n, k8) words to [n/64][k/32][64][4], n padded up to 64. qmm.tile_words.
pub fn tileWords(gpa: std.mem.Allocator, words: []const u8, n: usize, k8: usize) ![]u8 {
    if (k8 == 0 or k8 % 4 != 0 or words.len != n * k8 * 4) return error.UnexpectedTensor;
    const npad = (n + tile_bn - 1) / tile_bn * tile_bn;
    const kg = k8 / 4;
    const out = try gpa.alloc(u8, (npad / tile_bn) * kg * tile_bn * 4 * 4);
    @memset(out, 0);
    for (0..n) |r| {
        const tile = r / tile_bn;
        const local = r % tile_bn;
        for (0..kg) |g| {
            for (0..4) |w| {
                const src = ((r * k8) + g * 4 + w) * 4;
                const dst = (((tile * kg + g) * tile_bn + local) * 4 + w) * 4;
                @memcpy(out[dst..][0..4], words[src..][0..4]);
            }
        }
    }
    return out;
}

/// (rows, cols) bf16 to (cols, rows), the tiled Q4 scale layout.
pub fn transposeBf16(gpa: std.mem.Allocator, src: []const u8, rows: usize, cols: usize) ![]u8 {
    if (src.len != rows * cols * 2) return error.UnexpectedTensor;
    const out = try gpa.alloc(u8, src.len);
    for (0..rows) |r| {
        for (0..cols) |c| {
            const from = (r * cols + c) * 2;
            const to = (c * rows + r) * 2;
            @memcpy(out[to..][0..2], src[from..][0..2]);
        }
    }
    return out;
}

/// The normed row _qmm_hcdown writes: bf16(h * rinv * scale), one rinv per stream from that stream's partial sums.
pub fn normed(h: []const u16, pss: []const f32, scale: []const f32, dims: usize, streams: usize, eps: f32, out: []u16) !void {
    const nc = dims / 256;
    if (dims == 0 or dims % 256 != 0 or h.len != streams * dims or pss.len != nc * streams or scale.len != h.len or out.len != h.len) return error.UnexpectedTensor;
    for (0..streams) |s| {
        var total: f32 = 0;
        for (0..nc) |c| total += pss[c * streams + s];
        const rinv = 1.0 / @sqrt(total / @as(f32, @floatFromInt(dims)) + eps);
        for (0..dims) |i| {
            const hv: f32 = @bitCast(@as(u32, h[s * dims + i]) << 16);
            out[s * dims + i] = toBf16(hv * rinv * scale[s * dims + i]);
        }
    }
}

/// fp32 dot of the first K-slice against output column 0, groups of 32, from the MLX row (not the tiled copy).
pub fn firstPartial(row_normed: []const u16, row_words: []const u8, scale_row: []const u8, bias_row: []const u8, groups: usize) !f32 {
    if (row_normed.len < groups * 32 or row_words.len < groups * 16 or scale_row.len < groups * 2 or bias_row.len < groups * 2) return error.UnexpectedTensor;
    var acc: f32 = 0;
    for (0..groups) |g| {
        var buf: [32]f32 = undefined;
        var p: f32 = 0;
        for (0..4) |i| {
            const word = std.mem.readInt(u32, row_words[(g * 4 + i) * 4 ..][0..4], .little);
            for (0..8) |j| {
                const q: f32 = @floatFromInt((word >> @intCast(j * 4)) & 0xF);
                const xv: f32 = @bitCast(@as(u32, row_normed[g * 32 + i * 8 + j]) << 16);
                buf[i * 8 + j] = xv;
                p += xv * q;
            }
        }
        var n: usize = 32;
        while (n > 1) {
            n /= 2;
            for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
        }
        const s: f32 = @bitCast(@as(u32, std.mem.readInt(u16, scale_row[g * 2 ..][0..2], .little)) << 16);
        const b: f32 = @bitCast(@as(u32, std.mem.readInt(u16, bias_row[g * 2 ..][0..2], .little)) << 16);
        acc += p * s + buf[0] * b;
    }
    return acc;
}

test "tiled words keep row 0 in the first tile's local row 0" {
    var words: [16]u8 = @splat(0);
    std.mem.writeInt(u32, words[0..4], 0x21, .little);
    const out = try tileWords(std.testing.allocator, &words, 1, 4);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqual(@as(u32, 0x21), std.mem.readInt(u32, out[0..4], .little));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, out[16..20], .little));
}

test "the down norm writes bf16 of h times rinv times scale" {
    var h: [256]u16 = @splat(0);
    h[0] = 0x3f80;
    const pss = [_]f32{1};
    var scale: [256]f32 = @splat(1);
    var out: [256]u16 = undefined;
    try normed(&h, &pss, &scale, 256, 1, 1e-6, &out);
    const rinv = 1.0 / @sqrt(1.0 / 256.0 + 1e-6);
    try std.testing.expectEqual(toBf16(rinv), out[0]);
    try std.testing.expectEqual(@as(u16, 0), out[1]);
}

fn roundBf16(v: f32) f32 {
    return @bitCast(@as(u32, toBf16(v)) << 16);
}

/// glue._hc_reduce_act: sum the K slices in order, round, divide by the stream count, SiLU, and the inject gates.
pub fn reduceAct(part: []const f32, sk: usize, ndn: usize, low: usize, streams: usize, act: []u16, inj: []u16, xs: []f32) !void {
    if (low == 0 or low % 32 != 0 or ndn < low + streams or part.len != sk * ndn or act.len != low or inj.len != streams or xs.len != low / 32) return error.UnexpectedTensor;
    for (0..low) |i| {
        var sum: f32 = 0;
        for (0..sk) |s| sum += part[s * ndn + i];
        const y = roundBf16(roundBf16(sum) / @as(f32, @floatFromInt(streams)));
        act[i] = toBf16(y / (1.0 + @exp(-y)));
    }
    var buf: [32]f32 = undefined;
    for (0..low / 32) |g| {
        for (0..32) |j| buf[j] = @bitCast(@as(u32, act[g * 32 + j]) << 16);
        var n: usize = 32;
        while (n > 1) {
            n /= 2;
            for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
        }
        xs[g] = buf[0];
    }
    for (0..streams) |s| {
        var sum: f32 = 0;
        for (0..sk) |k| sum += part[k * ndn + low + s];
        const y = roundBf16(roundBf16(sum) / @as(f32, @floatFromInt(streams)));
        const sig = roundBf16(1.0 / (1.0 + @exp(-y)));
        inj[s] = toBf16(2.0 * sig);
    }
}

test "a zero down-projection sum is SiLU zero and an inject gate of one" {
    var part: [33]f32 = @splat(0);
    var act: [32]u16 = undefined;
    var inj: [1]u16 = undefined;
    var xs: [1]f32 = undefined;
    try reduceAct(&part, 1, 33, 32, 1, &act, &inj, &xs);
    try std.testing.expectEqual(@as(u16, 0), act[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), inj[0]);
    try std.testing.expectEqual(@as(f32, 0), xs[0]);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// One mixed hidden row: up-dot each stream, sigmoid-gate it by that stream's normed value, then average.
pub fn mix(act: []const u16, xs: []const f32, normed_row: []const u16, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k8: usize, dims: usize, streams: usize, out: []u16) !void {
    const kg = k8 / 4;
    if (dims == 0 or k8 == 0 or k8 % 4 != 0 or act.len < kg * 32 or xs.len < kg or normed_row.len != streams * dims or out.len != dims) return error.UnexpectedTensor;
    if (words.len != n * k8 * 4 or scales.len != n * kg * 2 or biases.len != scales.len) return error.UnexpectedTensor;
    for (0..dims) |col| {
        var total: f32 = 0;
        for (0..streams) |s| {
            const row = s * dims + col;
            var acc: f32 = 0;
            const word_row = words[row * k8 * 4 ..][0 .. k8 * 4];
            const scale_row = scales[row * kg * 2 ..][0 .. kg * 2];
            const bias_row = biases[row * kg * 2 ..][0 .. kg * 2];
            for (0..kg) |g| {
                var p: f32 = 0;
                for (0..4) |i| {
                    const word = std.mem.readInt(u32, word_row[(g * 4 + i) * 4 ..][0..4], .little);
                    for (0..8) |j| {
                        const q: f32 = @floatFromInt((word >> @intCast(j * 4)) & 0xF);
                        p += promote(act[g * 32 + i * 8 + j]) * q;
                    }
                }
                const sc = promote(std.mem.readInt(u16, scale_row[g * 2 ..][0..2], .little));
                const bias = promote(std.mem.readInt(u16, bias_row[g * 2 ..][0..2], .little));
                acc += p * sc + xs[g] * bias;
            }
            const up = roundBf16(acc);
            const sig = roundBf16(1.0 / (1.0 + @exp(-up)));
            total += roundBf16(sig * promote(normed_row[row]));
        }
        out[col] = toBf16(total / @as(f32, @floatFromInt(streams)));
    }
}

fn bf16Steps(a: u16, b: u16) u32 {
    const oa: i32 = if (a >= 0x8000) ~@as(i32, a) else a;
    const ob: i32 = if (b >= 0x8000) ~@as(i32, b) else b;
    const d = oa - ob;
    return @intCast(if (d < 0) -d else d);
}

pub fn mixedSteps(gpu: u16, host: u16) u32 {
    return bf16Steps(gpu, host);
}

test "a zero activation mixes to a zero hidden value" {
    var act: [32]u16 = @splat(0);
    const xs = [_]f32{0};
    var row: [1]u16 = @splat(0);
    var words: [16]u8 = @splat(0);
    var scale: [2]u8 = @splat(0);
    var bias: [2]u8 = @splat(0);
    var mixed: [1]u16 = undefined;
    try mix(&act, &xs, &row, &words, &scale, &bias, 1, 4, 1, 1, &mixed);
    try std.testing.expectEqual(@as(u16, 0), mixed[0]);
}
