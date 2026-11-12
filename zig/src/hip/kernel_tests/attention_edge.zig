//! The prompt tile's edge pass: a non-finite value at a key a row must not see adds nothing to that row.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const attention = @import("attention.zig");
const Cache = attention.Cache;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const heads = 32;
const kv_heads = 8;
const group = heads / kv_heads;
const d = 128;

/// A non-finite value planted in V: its KV head, key position and dim.
const Plant = struct { kv: usize, key: usize, dim: usize, value: f32 };

/// The output class float64 gives a row that sees keys up to `last`: the planted values it sees decide it.
const Class = enum { finite, pos_inf, neg_inf, nan };

fn expected(plants: []const Plant, kv: usize, dim: usize, last: usize) Class {
    var pos = false;
    var neg = false;
    for (plants) |p| {
        if (p.kv != kv or p.dim != dim or p.key > last) continue;
        if (std.math.isNan(p.value)) return .nan;
        if (p.value > 0) pos = true else neg = true;
    }
    if (pos and neg) return .nan;
    return if (pos) .pos_inf else if (neg) .neg_inf else .finite;
}

fn classOf(x: f32) Class {
    if (std.math.isNan(x)) return .nan;
    if (std.math.isInf(x)) return if (x > 0) .pos_inf else .neg_inf;
    return .finite;
}

fn planted(plants: []const Plant, dim: usize) bool {
    for (plants) |p| if (p.dim == dim) return true;
    return false;
}

fn plant(t: *const Rig, c: Cache, plants: []const Plant, span: usize) void {
    for (plants) |p| if (p.key < span) {
        c.v[(p.kv * span + p.key) * d + p.dim] = t.bits(p.value);
    };
}

/// Served and tile-only outputs of one prompt chunk, paged and flat; out (heads, qlen, d).
const Runs = struct { paged: [2][]f32, flat: [2][]f32 };

fn run(t: *Rig, c: Cache, hq: []const f32, qlen: usize, span: usize, q_pos0: usize) !Runs {
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(d)));
    const kind: c_int = if (t.bf16) 1 else 0;
    var q = try t.upload(hq);
    defer q.free();
    var k = try t.upload(c.k);
    defer k.free();
    var v = try t.upload(c.v);
    defer v.free();
    const pk = try c.pool(c.k, kv_heads, span, d);
    defer gpa.free(pk);
    const pv = try c.pool(c.v, kv_heads, span, d);
    defer gpa.free(pv);
    var pool_k = try t.upload(pk);
    defer pool_k.free();
    var pool_v = try t.upload(pv);
    defer pool_v.free();
    var table = try t.upload(c.table);
    defer table.free();
    var out = try t.alloc(heads * qlen * d * 4);
    defer out.free();
    const s = t.stream.handle;
    var r: Runs = undefined;
    // served (the edge pass where the launcher takes it), then the tile alone
    const edge = t.on.edge;
    defer t.on.edge = edge;
    for ([_]bool{ edge, false }, 0..) |e, i| {
        t.on.edge = e;
        try t.on.pagedCausal(at(q), at(pool_k), at(pool_v), at(table), at(out), qlen, span, heads, kv_heads, d, scale, q_pos0, c.table.len, kind, s);
        try t.stream.synchronize();
        r.paged[i] = try rig.download(f32, out);
        try t.on.tf_causal(@ptrFromInt(at(q)), @ptrFromInt(at(k)), @ptrFromInt(at(v)), @ptrFromInt(at(out)), 1, @intCast(qlen), @intCast(span), heads, kv_heads, d, scale, @intCast(q_pos0), 0, @intCast(span * d), d, 0, @intCast(span * d), d, kind, null, null, null, s, null);
        try t.stream.synchronize();
        r.flat[i] = try rig.download(f32, out);
    }
    return r;
}

/// Plants past rows' edges, odd start, partial row block, span past the last row: float64's class, tile's own bits.
fn edges(t: *Rig, rng: *rig.Rng) !void {
    const q_pos0 = 87;
    const qlen = 150;
    const span = 256; // the last tile, keys 192 to 255, holds keys past the last row's 236
    const inf = std.math.inf(f32);
    const plants = [_]Plant{
        .{ .kv = 0, .key = q_pos0 + 1, .dim = 7, .value = inf }, // past row 0's edge
        .{ .kv = 1, .key = q_pos0 + 71, .dim = 9, .value = -inf },
        .{ .kv = 2, .key = q_pos0 + 141, .dim = 11, .value = std.math.nan(f32) },
        .{ .kv = 3, .key = q_pos0 + 30, .dim = 13, .value = inf }, // row 30 sees +inf alone, later rows both
        .{ .kv = 3, .key = q_pos0 + 31, .dim = 13, .value = -inf },
        .{ .kv = 4, .key = 250, .dim = 15, .value = inf }, // past every row, inside the last tile
    };
    const c = try Cache.make(t, rng, kv_heads, span, d);
    defer c.free();
    plant(t, c, &plants, span);
    const hq = try gpa.alloc(f32, heads * qlen * d);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit();
    const r = try run(t, c, hq, qlen, span, q_pos0);
    defer for ([_][2][]f32{ r.paged, r.flat }) |pair| for (pair) |x| gpa.free(x);
    try std.testing.expectEqualSlices(u32, @ptrCast(r.flat[0]), @ptrCast(r.paged[0]));
    for (0..heads) |h| for (0..qlen) |i| for (0..d) |e| {
        const o = (h * qlen + i) * d + e;
        const want = expected(&plants, h / group, e, q_pos0 + i);
        if (classOf(r.paged[0][o]) != want) {
            std.debug.print("head {d} row {d} dim {d}: {d} where float64 gives {t}\n", .{ h, i, e, r.paged[0][o], want });
            return error.TestUnexpectedResult;
        }
        if (planted(&plants, e)) continue;
        // the same finite sums with and without the pass: a past-edge lane enters as +0 in one, as 0 x v in the other
        for ([_][2][]f32{ r.paged, r.flat }) |pair| {
            if (@as(u32, @bitCast(pair[0][o])) != @as(u32, @bitCast(pair[1][o]))) {
                std.debug.print("head {d} row {d} dim {d}: {d} served, {d} tile alone\n", .{ h, i, e, pair[0][o], pair[1][o] });
                return error.TestUnexpectedResult;
            }
        }
    };
}

/// A cache of `cap` positions on the device, flat (kv_heads, cap, d) and as a pool over its page table.
const Device = struct {
    k: DeviceBuffer,
    v: DeviceBuffer,
    pool_k: DeviceBuffer,
    pool_v: DeviceBuffer,
    table: DeviceBuffer,
    count: usize,

    fn upload(t: *Rig, c: Cache, cap: usize) !Device {
        const pk = try c.pool(c.k, kv_heads, cap, d);
        defer gpa.free(pk);
        const pv = try c.pool(c.v, kv_heads, cap, d);
        defer gpa.free(pv);
        return .{ .k = try t.upload(c.k), .v = try t.upload(c.v), .pool_k = try t.upload(pk), .pool_v = try t.upload(pv), .table = try t.upload(c.table), .count = c.table.len };
    }

    fn free(x: *Device) void {
        for ([_]*DeviceBuffer{ &x.k, &x.v, &x.pool_k, &x.pool_v, &x.table }) |b| b.free();
    }
};

/// Rows [0, cut) from 0, then [cut, len) from cut, one launch each, paged or flat; out (heads, len, d).
fn cutRuns(t: *Rig, hq: []const f32, dev: Device, cap: usize, len: usize, cut: usize, paged: bool) ![]f32 {
    const out = try gpa.alloc(f32, heads * len * d);
    errdefer gpa.free(out);
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(d)));
    const kind: c_int = if (t.bf16) 1 else 0;
    const s = t.stream.handle;
    for ([_][2]usize{ .{ 0, cut }, .{ cut, len } }) |seg| {
        const p = seg[0];
        const rows = seg[1] - p;
        if (rows == 0) continue;
        const q = try gpa.alloc(f32, heads * rows * d);
        defer gpa.free(q);
        for (0..heads) |h| @memcpy(q[h * rows * d ..][0 .. rows * d], hq[(h * len + p) * d ..][0 .. rows * d]);
        var dq = try t.upload(q);
        defer dq.free();
        var dout = try t.alloc(q.len * 4);
        defer dout.free();
        if (paged) {
            try t.on.pagedCausal(at(dq), at(dev.pool_k), at(dev.pool_v), at(dev.table), at(dout), rows, p + rows, heads, kv_heads, d, scale, p, dev.count, kind, s);
        } else {
            try t.on.tf_causal(@ptrFromInt(at(dq)), @ptrFromInt(at(dev.k)), @ptrFromInt(at(dev.v)), @ptrFromInt(at(dout)), 1, @intCast(rows), @intCast(p + rows), heads, kv_heads, d, scale, @intCast(p), 0, @intCast(cap * d), d, 0, @intCast(cap * d), d, kind, null, null, null, s, null);
        }
        try t.stream.synchronize();
        const got = try rig.download(f32, dout);
        defer gpa.free(got);
        for (0..heads) |h| @memcpy(out[(h * len + p) * d ..][0 .. rows * d], got[h * rows * d ..][0 .. rows * d]);
    }
    return out;
}

/// Planted values past some rows' edges: each row has the same bits cut at 0 and at `cut`, paged and flat.
fn shiftedPlanted(t: *Rig, rng: *rig.Rng, len: usize, cut: usize) !void {
    const cap = 256; // whole pages past len
    const inf = std.math.inf(f32);
    const plants = [_]Plant{
        .{ .kv = 0, .key = cut + 1, .dim = 7, .value = inf },
        .{ .kv = 1, .key = cut + 2, .dim = 9, .value = -inf },
        .{ .kv = 2, .key = len - 3, .dim = 11, .value = std.math.nan(f32) },
        .{ .kv = 3, .key = 40, .dim = 13, .value = inf },
        .{ .kv = 3, .key = 41, .dim = 13, .value = -inf },
    };
    const c = try Cache.make(t, rng, kv_heads, cap, d);
    defer c.free();
    plant(t, c, &plants, cap);
    const hq = try gpa.alloc(f32, heads * len * d);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit();
    var dev = try Device.upload(t, c, cap);
    defer dev.free();
    for ([_]bool{ true, false }) |paged| {
        const alone = try cutRuns(t, hq, dev, cap, len, 0, paged);
        defer gpa.free(alone);
        const after = try cutRuns(t, hq, dev, cap, len, cut, paged);
        defer gpa.free(after);
        try std.testing.expectEqualSlices(u32, @ptrCast(alone), @ptrCast(after));
    }
}

test "a non-finite value past a row's causal edge adds nothing to that row, in both cache types, paged and flat" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xED6E5 };
    const act = t.bf16;
    defer t.bf16 = act;
    for ([_]bool{ false, true }) |bf16| {
        if (bf16 and !act) continue; // RDNA2 has no bf16 dot2 and serves fp16 caches only
        t.bf16 = bf16;
        try edges(t, &rng);
        try shiftedPlanted(t, &rng, 237, 1);
        try shiftedPlanted(t, &rng, 237, 87);
    }
}
