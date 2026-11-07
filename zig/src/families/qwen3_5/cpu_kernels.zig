//! The Qwen3.5-2B affine projection and RMS norm on the host's cores: the Metal kernels' rounding points and orders.
const std = @import("std");
const bf16_round = @import("bf16_round");

pub const f32of = bf16_round.value;
pub const bf16of = bf16_round.bits;
/// Inputs an affine group shares one scale and bias over.
pub const group = 64;
/// Rows one `project` call takes at most (a prompt chunk).
pub const max_rows = 128;
/// Threads `projectOn` and `rmsnormOn` split a call over at most.
pub const max_threads = 64;
const V8 = @Vector(8, f32);
const U8 = @Vector(8, u32);

/// Affine words in tiles of eight columns: (tile, group, word j) holds the eight columns' word j; tables likewise.
pub const Linear = struct {
    tiles: []const u32,
    scales: []const u16,
    biases: []const u16,
    inputs: usize,
    outputs: usize,
    /// The accumulation chains the Metal kernel of this projection class runs (its `S`), reduced pairwise at the end.
    chains: usize,

    /// Code `e` of output row `n` (the checkpoint's nibble), read through the tiles.
    pub fn code(l: Linear, n: usize, e: usize) u32 {
        const groups = l.inputs / group;
        const g = e / group;
        const j = (e % group) / 8;
        const word = l.tiles[((n / 8) * groups + g) * 8 * 8 + j * 8 + n % 8];
        return (word >> @intCast(4 * (e % 8))) & 0xF;
    }

    /// The group table entry of output row `n`, group `g`.
    pub fn scale(l: Linear, n: usize, g: usize) u16 {
        return l.scales[((n / 8) * (l.inputs / group) + g) * 8 + n % 8];
    }

    pub fn bias(l: Linear, n: usize, g: usize) u16 {
        return l.biases[((n / 8) * (l.inputs / group) + g) * 8 + n % 8];
    }
};

/// The checkpoint's row-major words [N][K/8] and tables [N][K/64] into the tile order; N a multiple of eight.
pub fn pack(a: std.mem.Allocator, words: []const u32, scales: []const u16, biases: []const u16, outputs: usize, inputs: usize, chains: usize) !Linear {
    if (outputs % 8 != 0 or inputs % group != 0 or chains == 0 or chains > 32) return error.UnexpectedQwenTensor;
    const groups = inputs / group;
    const words_row = inputs / 8;
    if (words.len != outputs * words_row or scales.len != outputs * groups or biases.len != outputs * groups) return error.UnexpectedQwenTensor;
    const tiles = try a.alloc(u32, outputs * words_row);
    errdefer a.free(tiles);
    const ts = try a.alloc(u16, outputs * groups);
    errdefer a.free(ts);
    const tb = try a.alloc(u16, outputs * groups);
    for (0..outputs / 8) |t| for (0..groups) |g| {
        for (0..8) |j| for (0..8) |col| {
            tiles[((t * groups + g) * 8 + j) * 8 + col] = words[(t * 8 + col) * words_row + g * 8 + j];
        };
        for (0..8) |col| {
            ts[(t * groups + g) * 8 + col] = scales[(t * 8 + col) * groups + g];
            tb[(t * groups + g) * 8 + col] = biases[(t * 8 + col) * groups + g];
        }
    };
    return .{ .tiles = tiles, .scales = ts, .biases = tb, .inputs = inputs, .outputs = outputs, .chains = chains };
}

/// A row for `project`: xf as fp32, xs each group's sum as qmm sums it (eight blocks of eight, then the butterfly).
pub fn prepareRow(x: []const u16, xf: []f32, xs: []f32) void {
    for (x, xf) |b, *f| f.* = f32of(b);
    for (0..x.len / group) |g| {
        const src = xf[g * group ..][0..group];
        var block: [8]f32 = undefined;
        for (&block, 0..) |*b, j| {
            var t = src[8 * j];
            for (1..8) |e| t = src[8 * j + e] + t;
            b.* = t;
        }
        xs[g] = ((block[0] + block[1]) + (block[2] + block[3])) + ((block[4] + block[5]) + (block[6] + block[7]));
    }
}

/// The chains' pairwise reduction, `v[k] = v[k + w] + v[k]` for w = 1, 2, 4, ... (simd_sum's order too).
fn reduceChains(v: []V8) V8 {
    var width: usize = 1;
    while (width < v.len) : (width *= 2) {
        var k: usize = 0;
        while (k + width < v.len) : (k += 2 * width) v[k] = v[k + width] + v[k];
    }
    return v[0];
}

/// TB tiles over RB rows: per group the MMA's own 64-term fma chain (step outer, word inner), then qmm's chain update.
fn tileRows(comptime TB: usize, comptime RB: usize, l: Linear, tile: usize, xf: []const f32, xs: []const f32, rb: usize, y: []u16) void {
    const groups = l.inputs / group;
    const chains = l.chains;
    var acc: [TB][RB][32]V8 = undefined;
    for (&acc) |*t| for (t) |*row| @memset(row[0..chains], @splat(0));
    const mask: U8 = @splat(0xF);
    for (0..groups) |g| {
        var p: [TB][RB]V8 = @splat(@splat(@splat(0)));
        inline for (0..8) |s| {
            const shift: @Vector(8, u5) = @splat(@intCast(4 * s));
            inline for (0..8) |j| {
                inline for (0..TB) |t| {
                    const w: U8 = l.tiles[(((tile + t) * groups + g) * 8 + j) * 8 ..][0..8].*;
                    const codes: V8 = @floatFromInt((w >> shift) & mask);
                    inline for (0..RB) |r| p[t][r] = @mulAdd(V8, codes, @splat(xf[(rb + r) * l.inputs + g * group + 8 * j + s]), p[t][r]);
                }
            }
        }
        const chain = g % chains;
        inline for (0..TB) |t| {
            const at = ((tile + t) * groups + g) * 8;
            var sc: V8 = undefined;
            var bi: V8 = undefined;
            inline for (0..8) |col| {
                sc[col] = f32of(l.scales[at + col]);
                bi[col] = f32of(l.biases[at + col]);
            }
            inline for (0..RB) |r| acc[t][r][chain] = @mulAdd(V8, bi, @splat(xs[(rb + r) * groups + g]), @mulAdd(V8, sc, p[t][r], acc[t][r][chain]));
        }
    }
    inline for (0..TB) |t| inline for (0..RB) |r| {
        const sum = reduceChains(acc[t][r][0..chains]);
        inline for (0..8) |col| y[(rb + r) * l.outputs + (tile + t) * 8 + col] = bf16of(sum[col]);
    };
}

/// Columns [n0, n1) of y = x · Wᵀ: eight rows a pass over one tile, then single rows over four tiles (latency hidden).
pub fn project(l: Linear, xf: []const f32, xs: []const f32, rows: usize, y: []u16, n0: usize, n1: usize) void {
    std.debug.assert(rows <= max_rows and n0 % 8 == 0 and n1 % 8 == 0 and n1 <= l.outputs);
    const tiles = n1 / 8;
    var rb: usize = 0;
    while (rb + 8 <= rows) : (rb += 8) {
        var tile = n0 / 8;
        while (tile < tiles) : (tile += 1) tileRows(1, 8, l, tile, xf, xs, rb, y);
    }
    while (rb < rows) : (rb += 1) {
        var tile = n0 / 8;
        while (tile + 4 <= tiles) : (tile += 4) tileRows(4, 1, l, tile, xf, xs, rb, y);
        while (tile < tiles) : (tile += 1) tileRows(1, 1, l, tile, xf, xs, rb, y);
    }
}

/// The same projection over `threads` threads, the caller one of them, each over a run of whole tiles.
pub fn projectOn(l: Linear, xf: []const f32, xs: []const f32, rows: usize, y: []u16, threads: usize) !void {
    const tiles = l.outputs / 8;
    const n = @max(1, @min(threads, tiles, max_threads));
    var spawned: [max_threads]std.Thread = undefined;
    var made: usize = 0;
    defer for (spawned[0..made]) |t| t.join();
    for (1..n) |t| {
        spawned[made] = try std.Thread.spawn(.{}, project, .{ l, xf, xs, rows, y, 8 * (t * tiles / n), 8 * ((t + 1) * tiles / n) });
        made += 1;
    }
    project(l, xf, xs, rows, y, 0, 8 * (tiles / n));
}

/// Σ x² as the Metal norm sums it: 16 inputs a thread in an fma chain, 32 threads a simdgroup, groups of 512 in turn.
fn sumSquares(x: []const u16) f32 {
    var total: f32 = 0;
    var at: usize = 0;
    while (at < x.len) : (at += 512) {
        var lanes: [32]f32 = @splat(0);
        for (&lanes, 0..) |*lane, t| {
            var ss: f32 = 0;
            for (x[at + 16 * t ..][0..16]) |h| ss = @mulAdd(f32, f32of(h), f32of(h), ss);
            lane.* = ss;
        }
        var width: usize = 1;
        while (width < 32) : (width *= 2) {
            var k: usize = 0;
            while (k + width < 32) : (k += 2 * width) lanes[k] = lanes[k + width] + lanes[k];
        }
        total += lanes[0];
    }
    return total;
}

/// metal::rsqrt's portable form: 1 / sqrt(x) through float64, rounded once to fp32.
pub fn rsqrt(x: f32) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, x)));
}

/// qwen35_norm's row: out = bf16(w * (x * rsqrt(mean(x²) + eps))); the row a multiple of 512 wide.
pub fn rmsnorm(x: []const u16, w: []const u16, eps: f32, out: []u16) void {
    std.debug.assert(x.len % 512 == 0 and w.len == x.len and out.len == x.len);
    const inv = rsqrt(sumSquares(x) / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, w) |*o, h, ww| o.* = bf16of(f32of(ww) * (f32of(h) * inv));
}

const Norm = struct { x: []const u16, w: []const u16, eps: f32, out: []u16 };

fn normRows(n: Norm, first: usize, end: usize) void {
    const k = n.w.len;
    for (first..end) |r| rmsnorm(n.x[r * k ..][0..k], n.w, n.eps, n.out[r * k ..][0..k]);
}

/// `rows` rows of the norm over `threads` threads, the caller one of them, each over a run of whole rows.
pub fn rmsnormOn(x: []const u16, w: []const u16, eps: f32, rows: usize, out: []u16, threads: usize) !void {
    const n = @max(1, @min(threads, rows, max_threads));
    const norm: Norm = .{ .x = x, .w = w, .eps = eps, .out = out };
    var spawned: [max_threads]std.Thread = undefined;
    var made: usize = 0;
    defer for (spawned[0..made]) |t| t.join();
    for (1..n) |t| {
        spawned[made] = try std.Thread.spawn(.{}, normRows, .{ norm, t * rows / n, (t + 1) * rows / n });
        made += 1;
    }
    normRows(norm, 0, rows / n);
}

pub fn free(a: std.mem.Allocator, l: Linear) void {
    a.free(l.tiles);
    a.free(l.scales);
    a.free(l.biases);
}

/// Seeded weights of a projection class in the checkpoint's row-major order, packed.
fn synthetic(a: std.mem.Allocator, rnd: std.Random, outputs: usize, inputs: usize, chains: usize) !Linear {
    const words = try a.alloc(u32, outputs * inputs / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, outputs * inputs / group);
    defer a.free(scales);
    const biases = try a.alloc(u16, outputs * inputs / group);
    defer a.free(biases);
    for (words) |*w| w.* = rnd.int(u32);
    for (scales, biases) |*s, *b| {
        s.* = bf16of(rnd.float(f32) * 0.02 + 0.001);
        b.* = bf16of(-rnd.float(f32) * 0.2);
    }
    return pack(a, words, scales, biases, outputs, inputs, chains);
}

test "pack reads every code, scale and bias of the row-major source through the tiles" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const n = 24;
    const kw = 192;
    const words = try a.alloc(u32, n * kw / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, n * kw / group);
    defer a.free(scales);
    const biases = try a.alloc(u16, n * kw / group);
    defer a.free(biases);
    for (words) |*w| w.* = rnd.int(u32);
    for (scales, biases) |*s, *b| {
        s.* = rnd.int(u16);
        b.* = rnd.int(u16);
    }
    const l = try pack(a, words, scales, biases, n, kw, 16);
    defer free(a, l);
    for (0..n) |col| {
        for (0..kw) |i| try std.testing.expectEqual((words[col * kw / 8 + i / 8] >> @intCast(4 * (i % 8))) & 15, l.code(col, i));
        for (0..kw / group) |g| {
            try std.testing.expectEqual(scales[col * (kw / group) + g], l.scale(col, g));
            try std.testing.expectEqual(biases[col * (kw / group) + g], l.bias(col, g));
        }
    }
    try std.testing.expectError(error.UnexpectedQwenTensor, pack(a, words, scales, biases, n - 4, kw, 16));
}

test "the correctly rounded rsqrt at exact points, and apart from 1 / sqrt(x) in fp32" {
    try std.testing.expectEqual(@as(f32, 0.5), rsqrt(4));
    try std.testing.expectEqual(@as(f32, 0.125), rsqrt(64));
    try std.testing.expectEqual(@as(f32, 0), rsqrt(std.math.inf(f32)));
    try std.testing.expect(std.math.isNan(rsqrt(-1)));
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    var differs: usize = 0;
    for (0..100_000) |_| {
        const y = rnd.float(f32) * 100 + 1e-3;
        if (1.0 / @sqrt(y) != rsqrt(y)) differs += 1;
    }
    try std.testing.expect(differs > 0);
}

/// Wyhash of every output byte of the sweep's fixed inputs: x86_64 and arm64 give these bytes, at any thread count.
const portable_digest: u64 = 0xae93e2f573b522de;

test "the same bytes at 1 to 16 threads and at column splits every 8 columns to 256 and every 256 beyond, on every host" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const n = 2048;
    const kw = 2048;
    const rows = 128;
    const l = try synthetic(a, rnd, n, kw, 16);
    defer free(a, l);
    const x = try a.alloc(u16, rows * kw);
    defer a.free(x);
    for (x) |*v| v.* = bf16of(rnd.floatNorm(f32) * std.math.pow(f32, 2, rnd.float(f32) * 16 - 8));
    const xf = try a.alloc(f32, rows * kw);
    defer a.free(xf);
    const xs = try a.alloc(f32, rows * (kw / group));
    defer a.free(xs);
    for (0..rows) |r| prepareRow(x[r * kw ..][0..kw], xf[r * kw ..][0..kw], xs[r * (kw / group) ..][0 .. kw / group]);
    const want = try a.alloc(u16, rows * n);
    defer a.free(want);
    project(l, xf, xs, rows, want, 0, n);
    const got = try a.alloc(u16, rows * n);
    defer a.free(got);
    for (1..17) |threads| {
        @memset(got, 0xffff);
        try projectOn(l, xf, xs, rows, got, threads);
        try std.testing.expectEqualSlices(u16, want, got);
    }
    const part = 9;
    var split: usize = 8;
    while (split < n) : (split += if (split < 256) 8 else 256) {
        @memset(got, 0xffff);
        project(l, xf, xs, part, got, 0, split);
        project(l, xf, xs, part, got, split, n);
        try std.testing.expectEqualSlices(u16, want[0 .. part * n], got[0 .. part * n]);
    }
    const k = 1024;
    const h = try a.alloc(u16, rows * k);
    defer a.free(h);
    for (h) |*v| v.* = bf16of(rnd.floatNorm(f32) * 3);
    const w = try a.alloc(u16, k);
    defer a.free(w);
    for (w) |*v| v.* = bf16of(rnd.float(f32) + 0.5);
    const want_norm = try a.alloc(u16, rows * k);
    defer a.free(want_norm);
    try rmsnormOn(h, w, 1e-6, rows, want_norm, 1);
    const got_norm = try a.alloc(u16, rows * k);
    defer a.free(got_norm);
    for (2..17) |threads| {
        @memset(got_norm, 0xffff);
        try rmsnormOn(h, w, 1e-6, rows, got_norm, threads);
        try std.testing.expectEqualSlices(u16, want_norm, got_norm);
    }
    var hash = std.hash.Wyhash.init(0);
    hash.update(std.mem.sliceAsBytes(want));
    hash.update(std.mem.sliceAsBytes(want_norm));
    try std.testing.expectEqual(portable_digest, hash.final());
}
