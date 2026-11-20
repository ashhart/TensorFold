//! Causal Ling MLA mixer with F32 accumulators and BF16 activation boundaries, without a graphics backend.
const std = @import("std");
const rows_ = @import("cpu_rows.zig");
const lat: u32 = 512;
const rope_dim: u32 = 64;
const nope: u32 = 128;
const vd: u32 = 128;
const key: u32 = lat + rope_dim;
const qd: u32 = nope + rope_dim;
const nfreq: u32 = rope_dim / 2;
const round32 = rows_.round;
const sig = rows_.sigmoid;
pub const Shape = struct {
    heads: u32,
    eps: f32,
    theta: f64,
    pub fn scale(_: Shape) f32 {
        return 1 / @sqrt(@as(f32, qd));
    }
};
fn rotate(out: []f32, z: []const f32, pos: u32, theta: f64) void {
    for (0..nfreq) |i| {
        const angle = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / rope_dim);
        const c: f32 = @floatCast(@cos(angle));
        const sn: f32 = @floatCast(@sin(angle));
        out[i] = z[2 * i] * c - z[2 * i + 1] * sn;
        out[nfreq + i] = z[2 * i + 1] * c + z[2 * i] * sn;
    }
}
pub const Cache = struct {
    k_nope: []f32,
    value: []f32,
    rope: []f32,
    capacity: u32,
    position: u32 = 0,

    pub fn init(a: std.mem.Allocator, heads: u32, capacity: u32) !Cache {
        return .{ .k_nope = try a.alloc(f32, @as(usize, capacity) * heads * nope), .value = try a.alloc(f32, @as(usize, capacity) * heads * vd), .rope = try a.alloc(f32, @as(usize, capacity) * rope_dim), .capacity = capacity };
    }
};

fn mixImpl(
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
        var z: [rope_dim]f32 = undefined;
        var out: [rope_dim]f32 = undefined;
        for (0..rope_dim) |i| z[i] = kv[t * key + lat + i];
        rotate(&out, &z, pos0 + @as(u32, @intCast(t)), s.theta);
        for (0..rope_dim) |i| krope[(past + t) * rope_dim + i] = round32(out[i]);
        for (0..H) |h| {
            for (0..rope_dim) |i| z[i] = q[(t * H + h) * qd + nope + i];
            rotate(&out, &z, pos0 + @as(u32, @intCast(t)), s.theta);
            for (0..rope_dim) |i| qrope[(t * H + h) * rope_dim + i] = round32(out[i]);
        }
    }

    for (0..H) |h| {
        for (0..T) |t| {
            for (0..nope) |d| {
                var acc: f32 = 0;
                const w0 = ((h * (nope + vd) + d) * lat);
                for (0..lat) |c| acc += lat_t[t * lat + c] * kvb[w0 + c];
                k_nope[((past + t) * H + h) * nope + d] = round32(acc);
            }
            for (0..vd) |d| {
                var acc: f32 = 0;
                const w0 = ((h * (nope + vd) + nope + d) * lat);
                for (0..lat) |c| acc += lat_t[t * lat + c] * kvb[w0 + c];
                v[((past + t) * H + h) * vd + d] = round32(acc);
            }
        }
    }

    const sc = s.scale();
    const scores = try alloc.alloc(f32, H * T * total);
    defer alloc.free(scores);
    for (0..H) |h| {
        for (0..T) |t| {
            for (0..total) |k| {
                if (k > past + t) {
                    scores[(h * T + t) * total + k] = -std.math.inf(f32);
                    continue;
                }
                var p: f32 = 0;
                const qb = (t * H + h) * qd;
                const kb = (k * H + h) * nope;
                for (0..nope) |d| p += @as(f32, q[qb + d]) * k_nope[kb + d];
                const qr = (t * H + h) * rope_dim;
                for (0..rope_dim) |d| p += qrope[qr + d] * krope[k * rope_dim + d];
                scores[(h * T + t) * total + k] = p * sc;
            }
        }
    }

    for (0..H) |h| {
        for (0..T) |t| {
            var m: f32 = -std.math.inf(f32);
            const row = scores[(h * T + t) * total ..][0..total];
            for (0..past + t + 1) |k| m = @max(m, row[k]);
            var den: f32 = 0;
            for (0..past + t + 1) |k| {
                row[k] = @exp(row[k] - m);
                den += row[k];
            }
            for (0..past + t + 1) |k| row[k] /= den;
            var o: [vd]f32 = @splat(0);
            for (0..past + t + 1) |k| {
                const a = row[k];
                const vb = (k * H + h) * vd;
                for (0..vd) |d| o[d] += a * v[vb + d];
            }
            const gate = round32(sig(g[t * H + h]));
            for (0..vd) |d| y[(t * H + h) * vd + d] = round32(round32(o[d]) * gate);
        }
    }
    if (cache) |c| c.position = total;
}

/// Initial-history reference path retained for direct comparison with persistent decode.
pub fn mix(a: std.mem.Allocator, y: []f32, q: []const f32, kv: []const f32, w: []const f32, b: []const f32, g: []const f32, shape: Shape, rows: u32, pos0: u32) !void {
    return mixImpl(a, y, q, kv, w, b, g, shape, rows, pos0, null);
}

/// Append only new K/V rows; old projections and attention state are never recomputed.
pub fn append(a: std.mem.Allocator, y: []f32, q: []const f32, kv: []const f32, w: []const f32, b: []const f32, g: []const f32, shape: Shape, rows: u32, cache: *Cache) !void {
    return mixImpl(a, y, q, kv, w, b, g, shape, rows, cache.position, cache);
}
