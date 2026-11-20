//! Causal Ling MLA mixer with F32 accumulators and BF16 activation boundaries, without a graphics backend.
const std = @import("std");
const rows_ = @import("cpu_rows.zig");
const pool_mod = @import("cpu_gguf_pool.zig");
const lat: u32 = 512;
const rope_dim: u32 = 64;
const nope: u32 = 128;
const vd: u32 = 128;
const key: u32 = lat + rope_dim;
const qd: u32 = nope + rope_dim;
const nfreq: u32 = rope_dim / 2;
const round32 = rows_.round;
const sig = rows_.sigmoid;
pub const Shape = @import("cpu_mla.zig").Shape;
fn ropeTable(cos_tab: *[nfreq]f32, sin_tab: *[nfreq]f32, pos: u32, theta: f64) void {
    for (0..nfreq) |i| {
        const angle = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / rope_dim);
        cos_tab[i] = @floatCast(@cos(angle));
        sin_tab[i] = @floatCast(@sin(angle));
    }
}

fn rotateWithTable(out: []f32, z: []const f32, cos_tab: *const [nfreq]f32, sin_tab: *const [nfreq]f32) void {
    for (0..nfreq) |i| {
        const c = cos_tab[i];
        const sn = sin_tab[i];
        out[i] = round32(z[2 * i] * c - z[2 * i + 1] * sn);
        out[nfreq + i] = round32(z[2 * i + 1] * c + z[2 * i] * sn);
    }
}

inline fn loadF32x8(p: [*]const f32) @Vector(8, f32) {
    return @as(*align(1) const [8]f32, @ptrCast(p)).*;
}

fn projectLat4(lat_ptr: [*]const f32, w_base: [*]const f32, out_ptr: [*]f32, rows_out: usize) void {
    const F8 = @Vector(8, f32);
    var d: usize = 0;
    while (d +% 3 < rows_out) : (d +%= 4) {
        const w0 = w_base + d *% lat;
        const w1 = w0 + lat;
        const w2 = w1 + lat;
        const w3 = w2 + lat;
        var s0: F8 = @splat(0);
        var s1: F8 = @splat(0);
        var s2: F8 = @splat(0);
        var s3: F8 = @splat(0);
        var c: usize = 0;
        while (c < lat) : (c +%= 8) {
            const x = loadF32x8(lat_ptr + c);
            s0 = @mulAdd(F8, x, loadF32x8(w0 + c), s0);
            s1 = @mulAdd(F8, x, loadF32x8(w1 + c), s1);
            s2 = @mulAdd(F8, x, loadF32x8(w2 + c), s2);
            s3 = @mulAdd(F8, x, loadF32x8(w3 + c), s3);
        }
        out_ptr[d] = round32(@reduce(.Add, s0));
        out_ptr[d +% 1] = round32(@reduce(.Add, s1));
        out_ptr[d +% 2] = round32(@reduce(.Add, s2));
        out_ptr[d +% 3] = round32(@reduce(.Add, s3));
    }
}
pub const Cache = @import("cpu_mla.zig").Cache;

const MlaPoolCtx = struct {
    H: usize,
    T: usize,
    past: usize,
    total: usize,
    sc: f32,
    lat_t: []const f32,
    kvb: []const f32,
    q: []const f32,
    qrope: []const f32,
    krope: []const f32,
    g: []const f32,
    k_nope: []f32,
    v: []f32,
    scores: []f32,
    y: []f32,
};

fn mlaHeadRange(c: *MlaPoolCtx, h_lo: usize, h_hi: usize) void {
    const F8 = @Vector(8, f32);
    const H = c.H;
    const T = c.T;
    const past = c.past;
    const total = c.total;
    const sc = c.sc;
    for (h_lo..h_hi) |h| {
        const w_k = c.kvb.ptr + (h *% (nope + vd)) *% lat;
        const w_v = w_k + nope *% lat;
        for (0..T) |t| {
            const lat_ptr = c.lat_t.ptr + t *% lat;
            projectLat4(lat_ptr, w_k, c.k_nope.ptr + ((past + t) *% H +% h) *% nope, nope);
            projectLat4(lat_ptr, w_v, c.v.ptr + ((past + t) *% H +% h) *% vd, vd);
        }
        for (0..T) |t| {
            const qb = c.q.ptr + (t *% H +% h) *% qd;
            const qr = c.qrope.ptr + (t *% H +% h) *% rope_dim;
            const score_row = c.scores.ptr + (h *% T +% t) *% total;
            const valid = past + t + 1;
            var k: usize = 0;
            while (k < valid) : (k +%= 1) {
                const kb = c.k_nope.ptr + (k *% H +% h) *% nope;
                const kr = c.krope.ptr + k *% rope_dim;
                var acc: F8 = @splat(0);
                var d: usize = 0;
                while (d < nope) : (d +%= 8) {
                    acc = @mulAdd(F8, loadF32x8(qb + d), loadF32x8(kb + d), acc);
                }
                d = 0;
                while (d < rope_dim) : (d +%= 8) {
                    acc = @mulAdd(F8, loadF32x8(qr + d), loadF32x8(kr + d), acc);
                }
                score_row[k] = @reduce(.Add, acc) * sc;
            }
            while (k < total) : (k +%= 1) {
                score_row[k] = -std.math.inf(f32);
            }
        }
        for (0..T) |t| {
            var m: f32 = -std.math.inf(f32);
            const row = c.scores[(h * T + t) * total ..][0..total];
            const valid = past + t + 1;
            for (0..valid) |k| m = @max(m, row[k]);
            var den: f32 = 0;
            for (0..valid) |k| {
                row[k] = @exp(row[k] - m);
                den += row[k];
            }
            const inv_den = 1.0 / den;
            var o: [vd]f32 align(32) = @splat(0);
            const o_ptr: [*]f32 = &o;
            for (0..valid) |k| {
                const a_v: F8 = @splat(row[k] * inv_den);
                const vb = c.v.ptr + (k *% H +% h) *% vd;
                var d: usize = 0;
                while (d < vd) : (d +%= 8) {
                    const cur: F8 = @as(*align(32) const [8]f32, @ptrCast(@alignCast(o_ptr + d))).*;
                    @as(*align(32) [8]f32, @ptrCast(@alignCast(o_ptr + d))).* = @mulAdd(F8, a_v, loadF32x8(vb + d), cur);
                }
            }
            const gate = round32(sig(c.g[t * H + h]));
            for (0..vd) |d| c.y[(t * H + h) * vd + d] = round32(round32(o[d]) * gate);
        }
    }
}

fn mlaHeadWorker(raw: *anyopaque, h_lo: usize, h_hi: usize) void {
    const c: *MlaPoolCtx = @ptrCast(@alignCast(raw));
    mlaHeadRange(c, h_lo, h_hi);
}

fn mixImpl(
    pool: ?*pool_mod.Pool,
    alloc: std.mem.Allocator,
    y: []f32,
    q: []const f32,
    kv: []const f32,
    w_rms: []const f32,
    kvb: []const f32,
    g: []const f32,
    s: Shape,
    rows: u32,
    pos0: u32,
    cache: ?*Cache,
) !void {
    const H = s.heads;
    const T = rows;
    const past = if (cache) |c| c.position else 0;
    const total = past + T;
    if (cache) |c| if (past > c.capacity or T > c.capacity - past) return error.CpuContextCapacity;
    std.debug.assert(y.len == T * H * vd);
    std.debug.assert(q.len == T * H * qd);
    std.debug.assert(kv.len == T * key);
    std.debug.assert(w_rms.len == lat);
    std.debug.assert(kvb.len == H * (nope + vd) * lat);
    std.debug.assert(g.len == T * H);

    const lat_t = try alloc.alloc(f32, T * lat);
    defer alloc.free(lat_t);
    const krope = if (cache) |c| c.rope[0 .. total * rope_dim] else try alloc.alloc(f32, T * rope_dim);
    defer if (cache == null) alloc.free(krope);
    const qrope = try alloc.alloc(f32, T * H * rope_dim);
    defer alloc.free(qrope);
    const k_nope = if (cache) |c| c.k_nope[0 .. total * H * nope] else try alloc.alloc(f32, T * H * nope);
    defer if (cache == null) alloc.free(k_nope);
    const v = if (cache) |c| c.value[0 .. total * H * vd] else try alloc.alloc(f32, T * H * vd);
    defer if (cache == null) alloc.free(v);

    const eps: f32 = s.eps;
    for (0..T) |t| {
        var ss: f32 = 0;
        for (0..lat) |i| {
            const x: f32 = kv[t * key + i];
            ss += x * x;
        }
        const rstd = 1.0 / @sqrt(ss / @as(f32, lat) + eps);
        for (0..lat) |i| {
            const yn = @as(f32, kv[t * key + i]) * rstd;
            lat_t[t * lat + i] = round32(round32(yn) * w_rms[i]);
        }
        var cos_tab: [nfreq]f32 = undefined;
        var sin_tab: [nfreq]f32 = undefined;
        ropeTable(&cos_tab, &sin_tab, pos0 + @as(u32, @intCast(t)), s.theta);
        rotateWithTable(krope[(past + t) * rope_dim ..][0..rope_dim], kv[t * key + lat ..][0..rope_dim], &cos_tab, &sin_tab);
        for (0..H) |h| {
            rotateWithTable(qrope[(t * H + h) * rope_dim ..][0..rope_dim], q[(t * H + h) * qd + nope ..][0..rope_dim], &cos_tab, &sin_tab);
        }
    }

    const scores = try alloc.alloc(f32, H * T * total);
    defer alloc.free(scores);
    var ctx = MlaPoolCtx{
        .H = H,
        .T = T,
        .past = past,
        .total = total,
        .sc = s.scale(),
        .lat_t = lat_t,
        .kvb = kvb,
        .q = q,
        .qrope = qrope,
        .krope = krope,
        .g = g,
        .k_nope = k_nope,
        .v = v,
        .scores = scores,
        .y = y,
    };
    if (pool) |p| {
        p.runHeads(@ptrCast(&ctx), mlaHeadWorker, H, 1);
    } else {
        mlaHeadRange(&ctx, 0, H);
    }
    if (cache) |c| c.position = total;
}

/// Initial-history reference path retained for direct comparison with persistent decode.
pub fn mix(a: std.mem.Allocator, y: []f32, q: []const f32, kv: []const f32, w: []const f32, b: []const f32, g: []const f32, shape: Shape, rows: u32, pos0: u32) !void {
    return mixImpl(null, a, y, q, kv, w, b, g, shape, rows, pos0, null);
}

/// Initial-history MLA parallelized across independent heads through the GGUF worker pool; byte-equal to mix.
pub fn mixOnPool(pool: *pool_mod.Pool, a: std.mem.Allocator, y: []f32, q: []const f32, kv: []const f32, w: []const f32, b: []const f32, g: []const f32, shape: Shape, rows: u32, pos0: u32) !void {
    return mixImpl(pool, a, y, q, kv, w, b, g, shape, rows, pos0, null);
}

/// Append only new K/V rows; old projections and attention state are never recomputed.
pub fn append(a: std.mem.Allocator, y: []f32, q: []const f32, kv: []const f32, w: []const f32, b: []const f32, g: []const f32, shape: Shape, rows: u32, cache: *Cache) !void {
    return mixImpl(null, a, y, q, kv, w, b, g, shape, rows, cache.position, cache);
}

/// Append MLA parallelized across independent heads through the GGUF worker pool; byte-equal to append.
pub fn appendOnPool(pool: *pool_mod.Pool, a: std.mem.Allocator, y: []f32, q: []const f32, kv: []const f32, w: []const f32, b: []const f32, g: []const f32, shape: Shape, rows: u32, cache: *Cache) !void {
    return mixImpl(pool, a, y, q, kv, w, b, g, shape, rows, cache.position, cache);
}

test "MLA pool head-parallel appendOnPool is byte-equal to append across 2, 3, 4, and 8 threads" {
    const a = std.testing.allocator;
    const shape = Shape{ .heads = 16, .eps = 1e-6, .theta = 10000.0 };
    const H: usize = shape.heads;
    const past: u32 = 11;
    const cap: u32 = past + 8;

    const q = try a.alloc(f32, 4 * H * qd);
    defer a.free(q);
    const kv = try a.alloc(f32, 4 * key);
    defer a.free(kv);
    const w_rms = try a.alloc(f32, lat);
    defer a.free(w_rms);
    const kvb = try a.alloc(f32, H * (nope + vd) * lat);
    defer a.free(kvb);
    const g = try a.alloc(f32, 4 * H);
    defer a.free(g);

    for (q, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 19;
    for (kv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 17;
    for (w_rms, 0..) |*v, i| v.* = 0.75 + @as(f32, @floatFromInt(i % 11)) / 20;
    for (kvb, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 31;
    for (g, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 11;

    var init_cache = try Cache.init(a, shape.heads, cap);
    defer {
        a.free(init_cache.k_nope);
        a.free(init_cache.value);
        a.free(init_cache.rope);
    }
    init_cache.position = past;
    for (init_cache.k_nope[0 .. past * H * nope], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 15;
    for (init_cache.value[0 .. past * H * vd], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 21)) - 10)) / 17;
    for (init_cache.rope[0 .. past * rope_dim], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 15)) - 7)) / 13;

    const want_y = try a.alloc(f32, 4 * H * vd);
    defer a.free(want_y);
    const got_y = try a.alloc(f32, 4 * H * vd);
    defer a.free(got_y);

    for ([_]u32{ 1, 4 }) |rows| {
        var want_cache = try Cache.init(a, shape.heads, cap);
        defer {
            a.free(want_cache.k_nope);
            a.free(want_cache.value);
            a.free(want_cache.rope);
        }
        want_cache.position = past;
        @memcpy(want_cache.k_nope[0 .. past * H * nope], init_cache.k_nope[0 .. past * H * nope]);
        @memcpy(want_cache.value[0 .. past * H * vd], init_cache.value[0 .. past * H * vd]);
        @memcpy(want_cache.rope[0 .. past * rope_dim], init_cache.rope[0 .. past * rope_dim]);
        try append(a, want_y[0 .. rows * H * vd], q[0 .. rows * H * qd], kv[0 .. rows * key], w_rms, kvb, g[0 .. rows * H], shape, rows, &want_cache);

        const total = past + rows;
        for ([_]usize{ 2, 3, 4, 8 }) |threads| {
            const pool = try pool_mod.Pool.create(a, std.testing.io, threads);
            defer pool.destroy();
            var got_cache = try Cache.init(a, shape.heads, cap);
            defer {
                a.free(got_cache.k_nope);
                a.free(got_cache.value);
                a.free(got_cache.rope);
            }
            got_cache.position = past;
            @memcpy(got_cache.k_nope[0 .. past * H * nope], init_cache.k_nope[0 .. past * H * nope]);
            @memcpy(got_cache.value[0 .. past * H * vd], init_cache.value[0 .. past * H * vd]);
            @memcpy(got_cache.rope[0 .. past * rope_dim], init_cache.rope[0 .. past * rope_dim]);
            @memset(got_y[0 .. rows * H * vd], std.math.nan(f32));

            try appendOnPool(pool, a, got_y[0 .. rows * H * vd], q[0 .. rows * H * qd], kv[0 .. rows * key], w_rms, kvb, g[0 .. rows * H], shape, rows, &got_cache);
            try std.testing.expectEqual(want_cache.position, got_cache.position);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_y[0 .. rows * H * vd]), std.mem.sliceAsBytes(got_y[0 .. rows * H * vd]));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_cache.k_nope[0 .. total * H * nope]), std.mem.sliceAsBytes(got_cache.k_nope[0 .. total * H * nope]));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_cache.value[0 .. total * H * vd]), std.mem.sliceAsBytes(got_cache.value[0 .. total * H * vd]));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_cache.rope[0 .. total * rope_dim]), std.mem.sliceAsBytes(got_cache.rope[0 .. total * rope_dim]));
        }
    }
}
