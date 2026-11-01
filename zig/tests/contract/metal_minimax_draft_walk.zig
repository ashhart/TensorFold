//! DFlash2's picks under the contract: the chunked top-k (df_topk) and the greedy walk (df_walk), each by its rule.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const d = @import("metal_minimax_draft.zig");

const c = contract.spec;
const g = contract.gates;
const bf = d.bf;
const u = c.unit(.f32);
const Sim = struct { c.Accum, g.Order };

pub fn collect(s: *d.Suite, only: ?[]const u8) !void {
    for (std.enums.values(TopK.Variant)) |v| try s.add(TopK, only, v);
    try s.add(Walk, only, .dflash2);
}

// --- df_topk: each 2,048-value chunk's best 32 (value desc, id asc) by a bitonic sort ------------------------------

pub const TopK = struct {
    /// The first pass over a row's logits, then the two passes over the chunks' bests (ids carried).
    pub const Variant = enum { logits, bests, last };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    l: []u16 = &.{},
    vin: []f32 = &.{},
    iin: []u32 = &.{},
    ins: [3]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: the draft slots; n: the values a row (k unused).
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = switch (v) {
                .logits => "minimax_m3 draft dflash2 df_topk logits",
                .bests => "minimax_m3 draft dflash2 df_topk 98 chunks' bests",
                .last => "minimax_m3 draft dflash2 df_topk 2 chunks' bests",
            },
            .accum = .none,
            .depth = d.noSums,
            .out = .raw,
            .nan = .{ .stated = "finite rows: each chunk's 32 largest (value desc, id asc) as a host sort gives them; with NaN or infinities: the sorting network as Metal compares (NaN compares false), replayed bit for bit" },
            .tiles = .{ .{}, .{ .size = 2048, .step = 1, .min = 1 }, .{} },
            .served = switch (v) {
                .logits => &.{.{ .rows = 7, .n = 200064, .k = 1 }},
                .bests => &.{.{ .rows = 7, .n = 98 * 32, .k = 1 }},
                .last => &.{.{ .rows = 7, .n = 64, .k = 1 }},
            },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !TopK {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, .dflash2, "df_topk"), .v = v };
    }
    pub fn deinit(t: *TopK) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *TopK) void {
        inline for (.{ "l", "vin", "iin" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *TopK) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(TopK);

    fn chunks(t: *const TopK) usize {
        return (t.shape.n + 2047) / 2048;
    }
    fn fromIds(t: *const TopK) bool {
        return t.v != .logits;
    }

    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        t.l = try t.a.alloc(u16, if (t.fromIds()) 1 else n);
        t.vin = try t.a.alloc(f32, if (t.fromIds()) n else 1);
        t.iin = try t.a.alloc(u32, if (t.fromIds()) n else 1);
        // coarse logits, so equal values occur and ids must break the ties
        for (t.l) |*x| x.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 512))) / 16 - 16);
        for (t.vin) |*x| x.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 512))) / 16 - 16;
        for (t.iin, 0..) |*x, i| x.* = @intCast((i % shape.n) * 61 % 200064);
        if (fill == .nonfinite) for ([_]f32{ d.nan, d.inf, -d.inf, d.nan }, 0..) |x, k| {
            const i = (k % shape.rows) * shape.n + (k * 37) % shape.n;
            if (t.fromIds()) t.vin[i] = x else t.l[i] = c.toBf16(x);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.l));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.vin));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.iin));
        t.outs[0] = try t.rig.output(shape.rows * t.chunks() * 32 * 4);
        t.outs[1] = try t.rig.output(shape.rows * t.chunks() * 32 * 4);
        return true;
    }

    pub fn run(t: *TopK) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue([4]u32{ @intCast(t.shape.n), @intCast(t.shape.n), @intCast(t.chunks()), @intFromBool(t.fromIds()) }, 3);
        p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 4);
        p.e.setBuffer(t.outs[1].buf, t.outs[1].off, 5);
        p.e.dispatchGroups(mtl.Size.of(t.chunks(), t.shape.rows, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *TopK) usize {
        return 2 * t.shape.rows * t.chunks() * 32 * 4;
    }
    pub fn reference(_: *TopK, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    const P = struct { v: f32, id: u32 };
    fn first(_: void, a: P, b: P) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }

    pub fn check(t: *TopK, _: g.Gate) !?[]const u8 {
        const vals = std.mem.bytesAsSlice(f32, t.rig.bytes(t.outs[0], t.count() / 2));
        const ids = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[1], t.count() / 2));
        const buf = try t.a.alloc(P, 2048);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| for (0..t.chunks()) |ch| {
            var finite = true;
            for (buf, 0..) |*x, i| {
                const at = ch * 2048 + i;
                x.* = if (at < t.shape.n) .{ .v = if (t.fromIds()) t.vin[r * t.shape.n + at] else c.fromBf16(t.l[r * t.shape.n + at]), .id = if (t.fromIds()) t.iin[r * t.shape.n + at] else @intCast(at) } else .{ .v = -std.math.inf(f32), .id = 0xffffffff };
                if (at < t.shape.n and !std.math.isFinite(x.v)) finite = false;
            }
            if (finite) std.sort.pdq(P, buf, {}, first) else {
                var k: usize = 2;
                while (k <= 2048) : (k <<= 1) {
                    var j = k >> 1;
                    while (j > 0) : (j >>= 1) for (0..2048) |i| {
                        const l = i ^ j;
                        if (l <= i) continue;
                        const swap = if (i & k == 0) first({}, buf[l], buf[i]) else first({}, buf[i], buf[l]);
                        if (swap) std.mem.swap(P, &buf[i], &buf[l]);
                    };
                }
            }
            const o = (r * t.chunks() + ch) * 32;
            for (buf[0..32], vals[o..][0..32], ids[o..][0..32], 0..) |want, gv, gi, i| {
                if ((@as(u32, @bitCast(want.v)) != @as(u32, @bitCast(gv)) and !(std.math.isNan(want.v) and std.math.isNan(gv))) or want.id != gi)
                    return try std.fmt.bufPrint(&t.msg, "row {d} chunk {d} place {d}: ({d}, {d}) where the host has ({d}, {d})", .{ r, ch, i, gv, gi, want.v, want.id });
            }
        };
        return null;
    }
};

// --- df_walk: the selector's greedy path through each slot's top candidates -----------------------------------

pub const Walk = struct {
    pub const Variant = enum { dflash2 };
    /// The vocabulary only bounds ids (and clamps B's row); 8,192 rows keep A and B small.
    const vocab = 8192;
    const top = 32;
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    uv: []f32 = &.{},
    ui: []u32 = &.{},
    h: []u16 = &.{},
    pa: []u16 = &.{},
    pb: []u16 = &.{},
    seq: [8]u32 = undefined,
    ins: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: the draft slots; n: candidates a slot; k: the selector's rank.
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "minimax_m3 draft dflash2 df_walk",
            .accum = .f32,
            .depth = walkDepth,
            .out = .raw,
            .nan = .{ .stated = "each slot takes the first candidate with the largest score (a NaN never wins; ids past the vocabulary are skipped), within near ties of float64, and walks on from the id it took" },
            .tiles = .{ .{}, .{}, .{} },
            .served = &.{ .{ .rows = 7, .n = top, .k = 256 }, .{ .rows = 1, .n = top, .k = 256 } },
        };
    }
    /// A product, a 5-level simd_sum, the 8 simdgroups' parts in turn after the unary score.
    fn walkDepth(_: c.Shape) usize {
        return 15;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant) !Walk {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, .dflash2, "df_walk") };
    }
    pub fn deinit(t: *Walk) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Walk) void {
        inline for (.{ "uv", "ui", "h", "pa", "pb" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Walk) g.Kernel {
        return .{ .spec = spec(.dflash2), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Walk);

    pub fn prepare(t: *Walk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 7 or shape.n != top or shape.k != 256) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = shape.k;
        const s = shape.rows;
        t.uv = try t.a.alloc(f32, s * top);
        t.ui = try t.a.alloc(u32, s * top);
        t.h = try t.a.alloc(u16, s * k);
        t.pa = try t.a.alloc(u16, vocab * k);
        t.pb = try t.a.alloc(u16, vocab * k);
        for (t.uv) |*x| x.* = rng.float(f32) * 4 - 2;
        for (0..s) |e| {
            // distinct candidates, the last two past the vocabulary (the top-k's padding and the vocabulary's end)
            for (t.ui[e * top ..][0..top], 0..) |*x, cnd| x.* = @intCast((e * 977 + cnd * 251) % vocab);
            t.ui[e * top + top - 1] = 0xffffffff;
            t.ui[e * top + top - 2] = vocab;
        }
        d.fillBf(rng, t.h, -1, 1);
        d.fillBf(rng, t.pa, -1, 1);
        d.fillBf(rng, t.pb, -1, 1);
        if (fill == .stress) {
            // A integers times H = 1 (exact bf16 products), B's halves mirrored: partials near 2^21, small scores
            @memset(t.h, c.toBf16(1));
            for (0..vocab) |r| for (0..k / 2) |j| {
                t.pa[r * k + j] = c.toBf16(@floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 256))) - 128));
                t.pa[r * k + k / 2 + j] = t.pa[r * k + j];
                const b: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
                t.pb[r * k + j] = c.toBf16(b);
                t.pb[r * k + k / 2 + j] = c.toBf16(-b + (if (j < 4) @as(f32, @floatFromInt(rng.uintLessThan(u32, 3))) - 1 else 0));
            };
            for (t.uv) |*x| x.* = @floatFromInt(rng.uintLessThan(u32, 8));
        }
        if (fill == .nonfinite) {
            t.uv[0 * top + 3] = d.nan;
            if (s > 1) t.uv[1 * top + 5] = d.inf;
            if (s > 2) t.uv[2 * top + 7] = -d.inf;
        }
        @memset(&t.seq, 0);
        t.seq[0] = 1234;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.uv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ui));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.h));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.pa));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.pb));
        t.out = try t.rig.output(t.seq.len * 4);
        return true;
    }

    pub fn run(t: *Walk) !void {
        @memcpy(t.rig.bytes(t.out, t.seq.len * 4), std.mem.sliceAsBytes(&t.seq));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue([4]u32{ @intCast(t.shape.rows), @intCast(t.shape.k), top, vocab }, 5);
        p.e.setBuffer(t.out.buf, t.out.off, 6);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(t.shape.k, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Walk) usize {
        return t.seq.len * 4;
    }

    /// Slot e's candidate scores after `prev` in float64 (or with the rank's sum simulated) and their bounds.
    fn scores(t: *const Walk, e: usize, prev: u32, sim: ?Sim, out: *[top]f64, bnd: *[top]f64) void {
        const k = t.shape.k;
        var terms: [257]f64 = undefined;
        for (0..top) |cnd| {
            const id = t.ui[e * top + cnd];
            const row = @min(id, vocab - 1);
            var abs: f64 = @abs(t.uv[e * top + cnd]);
            var sum: f64 = t.uv[e * top + cnd];
            terms[k] = t.uv[e * top + cnd];
            for (0..k) |r| {
                const ah = bf(@as(f64, c.fromBf16(t.pa[@as(usize, prev) * k + r])) * c.fromBf16(t.h[e * k + r]));
                terms[r] = ah * c.fromBf16(t.pb[@as(usize, row) * k + r]);
                sum += terms[r];
                abs += @abs(terms[r]);
            }
            out[cnd] = if (sim) |o| contract.rows.sumIn(o[0], o[1], terms[0 .. k + 1]) else sum;
            bnd[cnd] = if (t.fill == .stress or sim != null) 0 else c.gamma(15, u) * abs;
        }
    }

    /// The kernel's choice on float64 scores: the first candidate strictly above all before it, valid ids only.
    fn choose(t: *const Walk, e: usize, s: *const [top]f64) usize {
        var best = -std.math.inf(f64);
        var bi: usize = 0;
        for (0..top) |cnd| if (t.ui[e * top + cnd] < vocab and s[cnd] > best) {
            best = s[cnd];
            bi = cnd;
        };
        return bi;
    }

    fn walk(t: *Walk, sim: ?Sim) [8]u32 {
        var seq = t.seq;
        var s: [top]f64 = undefined;
        var b: [top]f64 = undefined;
        for (0..t.shape.rows) |e| {
            t.scores(e, seq[e], sim, &s, &b);
            seq[1 + e] = t.ui[e * top + t.choose(e, &s)];
        }
        return seq;
    }

    /// Raw outputs: the seq buffer's bytes as the replay leaves them (gate 4's stress has no near ties).
    pub fn reference(t: *Walk, want: []f64, mag: []f64) !void {
        const seq = t.walk(null);
        for (want, std.mem.sliceAsBytes(&seq)) |*w, byte| w.* = @floatFromInt(byte);
        @memset(mag, 0);
    }
    pub fn simulate(t: *Walk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const seq = t.walk(.{ acc, order });
        for (out, std.mem.sliceAsBytes(&seq)) |*w, byte| w.* = @floatFromInt(byte);
    }
    pub fn bounds(_: *Walk, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }

    /// Each slot's id: a candidate tied near float64's choice (it, off finite scores), the walk going on from it.
    pub fn check(t: *Walk, _: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u32, t.rig.bytes(t.out, t.count()));
        if (got[0] != t.seq[0]) return try std.fmt.bufPrint(&t.msg, "seq[0] changed", .{});
        for (t.seq[t.shape.rows + 1 ..], got[t.shape.rows + 1 ..]) |w, x| if (w != x) return try std.fmt.bufPrint(&t.msg, "it wrote past its slots", .{});
        var s: [top]f64 = undefined;
        var b: [top]f64 = undefined;
        for (0..t.shape.rows) |e| {
            t.scores(e, got[e], null, &s, &b);
            const w = t.choose(e, &s);
            const id = got[1 + e];
            const cand = for (0..top) |cnd| {
                if (t.ui[e * top + cnd] == id) break cnd;
            } else return try std.fmt.bufPrint(&t.msg, "slot {d} took {d}, not one of its candidates", .{ e, id });
            if (cand == w) continue;
            if (!std.math.isFinite(s[w]) or !std.math.isFinite(s[cand]) or id >= vocab or !(s[cand] >= s[w] - b[cand] - b[w]))
                return try std.fmt.bufPrint(&t.msg, "slot {d} took candidate {d} ({d}) where float64 takes {d} ({d})", .{ e, cand, s[cand], w, s[w] });
        }
        return null;
    }
};
