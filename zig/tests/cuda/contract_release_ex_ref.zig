//! Host references for the release experts: routed problems, the exact grouping, FP64 dots and the SwiGLU interval.
const std = @import("std");
const core = @import("core");
const ref = @import("contract_release_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

/// The E2M1 magnitudes by code (bit 3 is the sign).
pub const e2m1 = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

pub fn e2m1Value(nib: u8) f64 {
    const v = e2m1[nib & 7];
    return if (nib & 8 != 0) -v else v;
}

/// Packed E2M1 [rows][k/2] (low nibble = even k) with [rows][k/32] scales: element (row, i)'s value.
pub fn packedValue(w: []const u8, s: []const u8, k: usize, row: usize, i: usize) f64 {
    const byte = w[row * (k / 2) + i / 2];
    return e2m1Value(if (i % 2 == 0) byte & 15 else byte >> 4) * ref.e8m0(s[row * (k / 32) + i / 32]);
}

/// FP8 codes [rows][k] with [rows][k/32] scales: element (row, i)'s value.
pub fn codes(x: []const u8, s: []const u8, k: usize, row: usize, i: usize) f64 {
    return ref.e4m3[x[row * k + i]] * ref.e8m0(s[row * (k / 32) + i / 32]);
}

/// A router's output for m rows: topk distinct ids of nids each, route weights; the rank holds ids [lo, lo + nslots).
pub const Routes = struct {
    m: usize,
    topk: usize,
    nids: usize,
    nslots: usize,
    lo: usize,
    ids: []i32,
    wts: []f32,
    slot_of: []i32,

    /// Row r takes row (r % uniq)'s ids; about one id in eight lies outside the held range.
    pub fn init(a: std.mem.Allocator, m: usize, topk: usize, nids: usize, nslots: usize, uniq: usize, rng: std.Random) !Routes {
        const lo = (nids - nslots) / 2;
        const ids = try a.alloc(i32, m * topk);
        errdefer a.free(ids);
        const wts = try a.alloc(f32, m * topk);
        errdefer a.free(wts);
        const slot_of = try a.alloc(i32, nids);
        for (slot_of, 0..) |*s, id| s.* = if (id >= lo and id < lo + nslots) @intCast(id - lo) else -1;
        for (0..@min(m, uniq)) |r| {
            var k: usize = 0;
            while (k < topk) {
                const held = (r == 0 and k == 0) or rng.uintLessThan(u32, 8) != 0 or nslots == nids;
                const id: i32 = @intCast(if (held) lo + rng.uintLessThan(usize, nslots) else rng.uintLessThan(usize, nids));
                if (std.mem.indexOfScalar(i32, ids[r * topk ..][0..k], id) != null) continue;
                ids[r * topk + k] = id;
                wts[r * topk + k] = std.math.ldexp(0.5 + rng.float(f32), -rng.intRangeAtMost(i32, 0, 3));
                k += 1;
            }
        }
        for (@min(m, uniq) * topk..m * topk) |i| {
            ids[i] = ids[i % (uniq * topk)];
            wts[i] = wts[i % (uniq * topk)];
        }
        return .{ .m = m, .topk = topk, .nids = nids, .nslots = nslots, .lo = lo, .ids = ids, .wts = wts, .slot_of = slot_of };
    }

    pub fn deinit(r: Routes, a: std.mem.Allocator) void {
        a.free(r.ids);
        a.free(r.wts);
        a.free(r.slot_of);
    }

    pub fn slot(r: Routes, pair: usize) i32 {
        const id = r.ids[pair];
        return if (id >= 0 and id < r.nids) r.slot_of[@intCast(id)] else -1;
    }
};

/// The grouping as tf_dsv41_rel_group writes it; unwritten words stay 0xffffffff (-1), as poisoned outputs read.
pub const Grouped = struct {
    offs: []i32,
    pair_tok: []i32,
    pair_of: []i32,
    active: []i32,
    nactive: i32,
    pairs: usize,

    pub fn init(a: std.mem.Allocator, r: Routes) !Grouped {
        const n = r.m * r.topk;
        var out: Grouped = .{ .offs = try a.alloc(i32, r.nslots + 1), .pair_tok = try a.alloc(i32, n), .pair_of = try a.alloc(i32, n), .active = try a.alloc(i32, r.nslots + 1), .nactive = 0, .pairs = 0 };
        @memset(out.pair_tok, -1);
        @memset(out.active, -1);
        var at: usize = 0;
        for (0..r.nslots) |s| {
            out.offs[s] = @intCast(at);
            const first = at;
            for (0..n) |p| if (r.slot(p) == @as(i32, @intCast(s))) {
                out.pair_tok[at] = @intCast(p);
                out.pair_of[p] = @intCast(at);
                at += 1;
            };
            if (at > first) {
                out.active[@intCast(out.nactive)] = @intCast(s);
                out.nactive += 1;
            }
        }
        out.offs[r.nslots] = @intCast(at);
        for (0..n) |p| if (r.slot(p) < 0) {
            out.pair_of[p] = -1;
        };
        out.pairs = at;
        return out;
    }

    pub fn deinit(gr: Grouped, a: std.mem.Allocator) void {
        a.free(gr.offs);
        a.free(gr.pair_tok);
        a.free(gr.pair_of);
        a.free(gr.active);
    }
};

/// Fills FP8 codes and scales for a gate's inputs: random, or gate 4's integers (2^4 scales, a mirrored half).
pub fn fillCodes(x: []u8, s: []u8, k: usize, fill: g.Fill, rng: std.Random) void {
    for (x) |*v| {
        const b = rng.int(u8);
        v.* = if (b & 0x7f == 0x7f) b ^ 1 else b;
    }
    for (s) |*v| v.* = 118 + rng.uintLessThan(u8, 16);
    if (fill != .stress) return;
    @memset(s, 131);
    const h = k / 2;
    for (0..x.len / k) |r| for (0..h) |i| {
        const v: f64 = @floatFromInt(1 + rng.uintLessThan(u8, 8));
        x[r * k + i] = ref.nearest(v);
        x[r * k + h + i] = ref.nearest(v + if (i < 8) @as(f64, 1) else 0);
    };
}

/// Fills packed E2M1 weights and scales: random, or gate 4's (the second half negated, 2^4 scales).
pub fn fillPacked(w: []u8, s: []u8, k: usize, fill: g.Fill, rng: std.Random) void {
    rng.bytes(w);
    for (s) |*v| v.* = 116 + rng.uintLessThan(u8, 12);
    if (fill != .stress) return;
    @memset(s, 131);
    const hb = k / 4;
    for (0..w.len / (k / 2)) |r| for (0..hb) |i| {
        const b = w[r * (k / 2) + i] & 0x77;
        w[r * (k / 2) + i] = b;
        w[r * (k / 2) + hb + i] = b | 0x88;
    };
}

/// A dot of FP8 row `xr` with packed column `wr` over k, its terms' magnitude, or its sum in `acc` and `order`.
pub fn dot(x: []const u8, xs: []const u8, xr: usize, w: []const u8, ws: []const u8, wr: usize, k: usize, t: []f64, acc: ?c.Accum, order: g.Order) [2]f64 {
    var sum: f64 = 0;
    var abs: f64 = 0;
    for (0..k) |i| {
        t[i] = codes(x, xs, k, xr, i) * packedValue(w, ws, k, wr, i);
        sum += t[i];
        abs += @abs(t[i]);
    }
    if (acc) |a| sum = core.contract.rows.sumIn(a, order, t[0..k]);
    return .{ sum, abs };
}

/// v's FP64 range from FP32 gate, up sums (bf16, clamps, expf within 2 ulp, a rounding a step); null: not finite.
pub fn swigluRange(gate: f32, up: f32, route: f32, routed: bool) ?[2]f64 {
    var gv: f64 = c.fromBf16(c.toBf16(gate));
    var uv: f64 = c.fromBf16(c.toBf16(up));
    if (!std.math.isFinite(gv) or !std.math.isFinite(uv)) return null;
    gv = @min(gv, 10);
    uv = std.math.clamp(uv, -10, 10);
    const e = @exp(-gv);
    const steps: f64 = if (routed) 4 else 3;
    // e / (1 + e) as 1 / (1 + exp(g)): finite where exp(-g) is not
    const rel = 1.01 * (steps * 0x1p-24 + 4 * 0x1p-24 / (1 + @exp(gv))) + 0x1p-40;
    const v = gv / (1 + e) * uv * (if (routed) @as(f64, route) else 1);
    const d = @abs(v) * rel + 0x1p-146;
    // FP32 expf overflows past -g = 88.72 (1 + e is then inf and v a signed zero)
    if (-gv > 88.5) return .{ @min(v - d, 0), @max(v + d, 0) };
    return .{ v - d, v + d };
}

/// Whether bf16 bits `got` are a round-to-nearest of some FP32 value in `range` (rounding is monotone).
pub fn inSpan(got: u16, range: [2]f64) bool {
    const v: f64 = c.fromBf16(got);
    const lo: f64 = c.fromBf16(c.toBf16(@floatCast(range[0])));
    const hi: f64 = c.fromBf16(c.toBf16(@floatCast(range[1])));
    return v >= lo and v <= hi;
}

/// The kernel's steps in FP32 with the host's expf, for non-finite g or u: only the result's class is compared.
pub fn swigluIeee(gate: f32, up: f32, route: f32, routed: bool) f32 {
    var gv = c.fromBf16(c.toBf16(gate));
    var uv = c.fromBf16(c.toBf16(up));
    gv = if (gv > 10) 10 else gv;
    uv = if (uv > 10) 10 else if (uv < -10) -10 else uv;
    var v = gv / (1 + @exp(-gv)) * uv;
    if (routed) v = v * route;
    return c.fromBf16(c.toBf16(v));
}

test "e2m1 values, packed nibbles and the grouping order (slot, row, k) with unheld ids skipped" {
    try std.testing.expectEqual(@as(f64, -6), e2m1Value(0xf));
    const w = [_]u8{0x2f};
    const s = [_]u8{127};
    try std.testing.expectEqual(@as(f64, -6), packedValue(&w, &s, 32, 0, 0));
    try std.testing.expectEqual(@as(f64, 1), packedValue(&w, &s, 32, 0, 1));
    const a = std.testing.allocator;
    var ids = [_]i32{ 5, 2, 7, 5, 2, 9 };
    var wts: [6]f32 = @splat(1);
    var slot_of = [_]i32{ -1, -1, 0, -1, -1, 1, -1, 2, -1, -1 };
    const r: Routes = .{ .m = 3, .topk = 2, .nids = 10, .nslots = 3, .lo = 2, .ids = &ids, .wts = &wts, .slot_of = &slot_of };
    const gr = try Grouped.init(a, r);
    defer gr.deinit(a);
    try std.testing.expectEqualSlices(i32, &.{ 0, 2, 4, 5 }, gr.offs);
    try std.testing.expectEqualSlices(i32, &.{ 1, 4, 0, 3, 2, -1 }, gr.pair_tok);
    try std.testing.expectEqualSlices(i32, &.{ 2, 0, 4, 3, 1, -1 }, gr.pair_of);
    try std.testing.expectEqual(@as(i32, 3), gr.nactive);
}
