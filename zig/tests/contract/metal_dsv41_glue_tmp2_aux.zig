//! The hc families' input fills and the four exact checks of the glue_tmp part-two suite, kept out of its main file.
const std = @import("std");
const contract = @import("contract");
const ref = @import("dsv41_glue_tmp_ref.zig");

const c = contract.spec;
const g = contract.gates;

pub const dim: usize = 5120;

pub const Kind = enum { mix, pre, post };

/// The slices of one hc family's prepared call that its fills write.
pub const Hc = struct {
    kind: Kind,
    rows: usize,
    x: []u16,
    fnw: []f32,
    basew: []f32,
    mixf: []f32,
    postf: []f32,
    comb: []f32,
    res: []u16,
    scale: *[3]f32,
    plants: *usize,
};

fn pow2(e: i32) f32 {
    return std.math.pow(f32, 2, @floatFromInt(e));
}

fn spread(rng: std.Random, lo: i32, hi: i32) f32 {
    return (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, lo, hi));
}

/// The random weights and stream, then gate 4's cancelling integers or gate 1's plants.
pub fn fillHc(h: Hc, rng: std.Random, fill: g.Fill) void {
    for (h.fnw) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.02;
    h.scale.* = .{ 0.5 + rng.float(f32), 0.5 + rng.float(f32), 0.5 + rng.float(f32) };
    for (h.basew) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.5;
    for (h.mixf) |*v| v.* = 0.25 + rng.float(f32);
    for (h.postf) |*v| v.* = 0.25 + rng.float(f32);
    for (h.comb) |*v| v.* = 0.25 + rng.float(f32);
    for (h.x) |*v| v.* = c.toBf16(spread(rng, -3, 3));
    for (h.res) |*v| v.* = c.toBf16(spread(rng, -3, 3));
    switch (fill) {
        .stress => stressHc(h),
        .nonfinite => plantHc(h),
        .random => {},
    }
}

/// Gate 4's inputs: integers whose partials pass 2^11 while the result stays small, so fp16 and bf16 lose it.
fn stressHc(h: Hc) void {
    const big: f32 = std.math.pow(f32, 2, 16);
    const half = dim / 2;
    @memset(h.x, c.toBf16(0));
    @memset(h.res, c.toBf16(0));
    switch (h.kind) {
        .mix => {
            for (0..half) |q| {
                const v: f32 = @floatFromInt(16 + (q % 8));
                h.x[q] = c.toBf16(v);
                h.x[half + q] = c.toBf16(v);
            }
            for (0..24) |m| {
                const f = h.fnw[m * dim ..][0..dim];
                for (0..half) |q| {
                    const w: f32 = @floatFromInt(1 + ((q + m) % 7));
                    f[q] = w;
                    f[half + q] = -w + if (q < 8) @as(f32, 1) else @as(f32, 0);
                }
            }
            h.scale.* = .{ 1, 1, 1 };
            @memset(h.basew, 0);
        },
        .pre => {
            @memset(h.mixf, 1);
            for (0..h.rows) |r| for (0..dim) |q| {
                const s: f32 = @floatFromInt(1 + (q % 7));
                h.x[(r * 4 + 0) * dim + q] = c.toBf16(big);
                h.x[(r * 4 + 1) * dim + q] = c.toBf16(s);
                h.x[(r * 4 + 2) * dim + q] = c.toBf16(-big);
            };
        },
        .post => {
            @memset(h.postf, 1);
            @memset(h.comb, 1);
            for (0..h.rows) |r| for (0..dim) |q| {
                const s: f32 = @floatFromInt(1 + (q % 7));
                h.res[(r * 4 + 0) * dim + q] = c.toBf16(big);
                h.res[(r * 4 + 1) * dim + q] = c.toBf16(s);
                h.res[(r * 4 + 2) * dim + q] = c.toBf16(-big);
            };
        },
    }
}

/// NaN and both infinities at known places, so the outputs' classes are proven, not assumed.
fn plantHc(h: Hc) void {
    const nf = std.math.nan(f32);
    const inf = std.math.inf(f32);
    switch (h.kind) {
        .mix => {
            h.x[0] = c.toBf16(nf);
            if (h.rows > 1) h.fnw[dim + 3] = nf;
            if (h.rows > 2) h.fnw[2 * dim + 5] = nf;
            if (h.rows > 3) h.x[3 * dim + 1] = c.toBf16(inf);
        },
        .pre => {
            h.x[1] = c.toBf16(inf);
            h.x[(4 + 2) * dim + 3] = c.toBf16(-inf);
            if (h.rows > 2) h.mixf[2 * 4 + 1] = nf;
        },
        .post => {
            h.x[1] = c.toBf16(inf);
            h.res[(4 + 2) * dim + 3] = c.toBf16(-inf);
            if (h.rows > 2) h.postf[2 * 4 + 1] = nf;
        },
    }
}

pub fn fail(msg: *[300]u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return try std.fmt.bufPrint(msg, fmt, args);
}

/// fp4pack's codes and scales against the math byte for byte, with the NaN blocks gate 1 needs.
pub fn pack(s: c.Shape, nb: usize, x: []const u16, dest: []const i32, codes: []const u8, scales: []const u8, plants: usize, gate: g.Gate, a: std.mem.Allocator, msg: *[300]u8) !?[]const u8 {
    const cw = try a.alloc(u8, s.rows * nb * 16);
    const sw = try a.alloc(u8, s.rows * nb);
    @memset(cw, 0xff);
    @memset(sw, 0xff);
    var nan_blocks: usize = 0;
    for (0..s.rows) |r| {
        if (dest[r] < 0) continue;
        const to: usize = @intCast(dest[r]);
        for (0..nb) |b| {
            const sb = ref.packBlock(x[r * s.n + 32 * b ..][0..32], cw[to * nb * 16 + 16 * b ..][0..16]);
            sw[to * nb + b] = sb;
            nan_blocks += @intFromBool(sb == 255);
        }
    }
    if (std.mem.indexOfDiff(u8, cw, codes)) |i| {
        return try fail(msg, "code byte {d} of {d}x{d}: got 0x{x:0>2}, the packing gives 0x{x:0>2}", .{ i, s.rows, s.n, codes[i], cw[i] });
    }
    if (std.mem.indexOfDiff(u8, sw, scales)) |i| return try fail(msg, "scale byte {d} of {d}x{d}: got 0x{x:0>2}, the packing gives 0x{x:0>2}", .{ i, s.rows, s.n, scales[i], sw[i] });
    if (gate == .nonfinite and (plants < 5 or nan_blocks < 4)) return try fail(msg, "{d} plants, {d} blocks at the byte 255: no proof", .{ plants, nan_blocks });
    return null;
}

/// fp4rt's bf16 bits against the reference's round trip, with the NaN blocks gate 1 needs.
pub fn rt(s: c.Shape, x: []const u16, dest: []const i32, got_bytes: []const u8, plants: usize, gate: g.Gate, a: std.mem.Allocator, msg: *[300]u8) !?[]const u8 {
    const want = try a.alloc(u16, s.rows * s.n);
    const written = try a.alloc(bool, s.rows * s.n);
    @memset(want, 0xffff);
    @memset(written, false);
    for (0..s.rows) |r| {
        if (dest[r] < 0) continue;
        const to: usize = @intCast(dest[r]);
        for (0..s.n / 16) |b| {
            ref.rtBlock(x[r * s.n + 16 * b ..][0..16], want[to * s.n + 16 * b ..][0..16]);
            @memset(written[to * s.n + 16 * b ..][0..16], true);
        }
    }
    const got = std.mem.bytesAsSlice(u16, got_bytes);
    for (want, got, written, 0..) |w, h, wr, i| {
        const nan = (w & 0x7fff) > 0x7f80 and (h & 0x7fff) > 0x7f80;
        if (!(if (wr) (w == h or nan) else h == 0xffff)) {
            return try fail(msg, "row {d} element {d} of {d}x{d}: got 0x{x:0>4}, the round trip gives 0x{x:0>4}", .{ i / s.n, i % s.n, s.rows, s.n, h, w });
        }
    }
    if (gate == .nonfinite and plants < 5) return try fail(msg, "{d} plants: no proof", .{plants});
    return null;
}

/// The pool's odd rows against float64 by column, its even rows unwritten, and the carry's last slot values.
pub fn pool(s: c.Shape, state0: []const f32, kv: []const f32, sc: []const f32, pos: []const i32, y_bytes: []const u8, st_bytes: []const u8, plants: usize, gate: g.Gate, msg: *[300]u8) !?[]const u8 {
    const n = s.n;
    const y = std.mem.bytesAsSlice(u16, y_bytes);
    const st = std.mem.bytesAsSlice(u32, st_bytes);
    var nonfinite: usize = 0;
    var pooled: usize = 0;
    for (0..n) |d| {
        var k2 = [2]f32{ state0[d], state0[n + d] };
        var s2 = [2]f32{ state0[2 * n + d], state0[3 * n + d] };
        for (0..s.rows) |r| {
            const slot: usize = @intCast(pos[r] & 1);
            k2[slot] = kv[r * n + d];
            s2[slot] = sc[r * n + d];
            const got = y[r * n + d];
            if (slot == 0) {
                if (got != 0xffff) return try fail(msg, "row {d} column {d}: an even position's row was written (0x{x:0>4})", .{ r, d, got });
                continue;
            }
            const v = ref.pool(.{ k2[0], k2[1] }, .{ s2[0], s2[1] });
            const have: f64 = c.fromBf16(got);
            pooled += 1;
            if (!std.math.isFinite(v.v)) {
                nonfinite += 1;
                if (c.classOf(have) != c.classOf(v.v)) return try fail(msg, "row {d} column {d} of {d}x{d}: {t} where float64 gives {t}", .{ r, d, s.rows, n, c.classOf(have), c.classOf(v.v) });
                continue;
            }
            const work = 10 * 0x1p-24 * v.mag;
            if (!(@abs(have - v.v) <= 0x1p-8 * (@abs(v.v) + work) + work + 0x1p-133)) return try fail(msg, "row {d} column {d} of {d}x{d}: {e:.6} where float64 gives {e:.6}", .{ r, d, s.rows, n, have, v.v });
        }
        for ([_]f32{ k2[0], k2[1], s2[0], s2[1] }, 0..) |v, i| {
            const h: f32 = @bitCast(st[i * n + d]);
            if (!(h == v or (std.math.isNan(h) and std.math.isNan(v)))) return try fail(msg, "carry word {d} of column {d}: {e:.6} where the last slot value is {e:.6}", .{ i, d, h, v });
        }
    }
    if (gate == .nonfinite and (plants < 4 or nonfinite < 3 or nonfinite == pooled)) return try fail(msg, "{d} plants, {d} of {d} pooled outputs non-finite: no proof", .{ plants, nonfinite, pooled });
    return null;
}

/// The candidate mask against the stable rank, with the non-finite keys gate 1 needs.
pub fn cand(s: c.Shape, scores: []const f32, vis: []const i32, got: []const u8, plants: usize, gate: g.Gate, a: std.mem.Allocator, msg: *[300]u8) !?[]const u8 {
    const bm = try a.alloc(f32, s.k);
    const mask = try a.alloc(u8, s.k);
    var nonfinite: usize = 0;
    for (0..s.rows) |r| {
        const row = scores[r * s.k * 8 ..][0 .. s.k * 8];
        for (row) |v| nonfinite += @intFromBool(!std.math.isFinite(v));
        ref.cand(row, vis[r], s.k, s.n, bm, mask);
        for (mask, 0..) |w, j| if (got[r * s.k + j] != w) return try fail(msg, "row {d} (visible {d} keys, {d} blocks, top {d}) block {d}: got {d}, the stable rank gives {d}", .{ r, vis[r], s.k, s.n, j, got[r * s.k + j], w });
    }
    if (gate == .nonfinite and (plants < 4 or nonfinite < 4)) return try fail(msg, "{d} plants, {d} non-finite keys: no proof", .{ plants, nonfinite });
    return null;
}
