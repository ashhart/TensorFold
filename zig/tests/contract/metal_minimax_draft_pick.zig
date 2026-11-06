//! DSpark's picks under the contract (mm_markov_step, mm_markov_pick), each by its rule; DFlash2's are in _walk.zig.
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
    try s.add(MarkovStep, only, .dspark);
    try s.add(MarkovPick, only, .dspark);
}

/// The lexicographic order the kernels' reductions keep: the larger value, the lower id on ties (NaN compares false).
fn before(v: f64, id: u32, w: f64, wid: u32) bool {
    return v > w or (v == w and id < wid);
}

// --- mm_markov_step: logits[step] + w2[v] . w1[prev] (w2 8-bit affine), each 256-id group's best ----------------

pub const MarkovStep = struct {
    pub const Variant = enum { dspark };
    const prev_id = 37;
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    l: []u16 = &.{},
    w1: []u16 = &.{},
    w2: []u32 = &.{},
    s2: []u16 = &.{},
    b2: []u16 = &.{},
    seq: [8]u32 = undefined,
    ins: [6]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: the block's logits rows (the step reads row rows - 2); n: the vocabulary; k: the Markov rank.
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "minimax_m3 draft dspark mm_markov_step",
            .accum = .f32,
            .depth = rankDepth,
            .out = .f32,
            .nan = .{ .stated = "each group's best follows its lane order (a NaN keeps its own place and loses where it is met); ids within near ties of the float64 scores" },
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 7, .n = 1000, .k = 256 }, .{ .rows = 7, .n = 200064, .k = 256 } },
        };
    }
    /// The rank's fma chain, plus the logit's add.
    fn rankDepth(s: c.Shape) usize {
        return s.k + 1;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant) !MarkovStep {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, .dspark, "mm_markov_step") };
    }
    pub fn deinit(t: *MarkovStep) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *MarkovStep) void {
        inline for (.{ "l", "w1", "w2", "s2", "b2" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *MarkovStep) g.Kernel {
        return .{ .spec = spec(.dspark), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(MarkovStep);

    fn groups(t: *const MarkovStep) usize {
        return (t.shape.n + 255) / 256;
    }
    fn step(t: *const MarkovStep) usize {
        return t.shape.rows - 2;
    }

    pub fn prepare(t: *MarkovStep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 2 or shape.rows > 8 or shape.n == 0 or shape.k != 256) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const v = shape.n;
        const k = shape.k;
        t.l = try t.a.alloc(u16, shape.rows * v);
        t.w1 = try t.a.alloc(u16, (prev_id + 1) * k);
        t.w2 = try t.a.alloc(u32, v * k / 4);
        t.s2 = try t.a.alloc(u16, v * k / 64);
        t.b2 = try t.a.alloc(u16, v * k / 64);
        d.fillBf(rng, t.l, -8, 8);
        d.fillBf(rng, t.w1, -1, 1);
        for (t.w2) |*w| w.* = rng.int(u32);
        d.fillBf(rng, t.s2, 0.001, 0.02);
        d.fillBf(rng, t.b2, -1.5, 1.5);
        if (fill == .stress) {
            // q - 128 exactly, w1 integers mirrored about the middle: fp32 sums exact, scores small, bf16 sums wrong
            @memset(t.s2, c.toBf16(1));
            @memset(t.b2, c.toBf16(-128));
            const e = t.w1[prev_id * k ..][0..k];
            for (0..k / 2) |r| {
                const x: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
                e[r] = c.toBf16(x);
                e[k / 2 + r] = c.toBf16(-x + (if (r < 3) @as(f32, @floatFromInt(rng.uintLessThan(u32, 3))) - 1 else 0));
            }
            for (0..v) |id| for (0..k / 8) |j| {
                t.w2[id * k / 4 + k / 8 + j] = t.w2[id * k / 4 + j]; // the second half's codes repeat the first's
            };
            for (t.l) |*x| x.* = c.toBf16(@floatFromInt(rng.uintLessThan(u32, 64)));
        }
        if (fill == .nonfinite) for ([_]f32{ d.nan, d.inf, -d.inf, d.nan }, [_]usize{ 3, 300, 517, 1000 }) |x, id| {
            if (id < v) t.l[t.step() * v + id] = c.toBf16(x);
        };
        @memset(&t.seq, 0);
        t.seq[t.step()] = prev_id;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.l));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w1));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.w2));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.s2));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.b2));
        t.ins[5] = try t.rig.input(std.mem.sliceAsBytes(&t.seq));
        t.outs[0] = try t.rig.output(t.groups() * 4);
        t.outs[1] = try t.rig.output(t.groups() * 4);
        return true;
    }

    pub fn run(t: *MarkovStep) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue([3]u32{ @intCast(t.shape.n), @intCast(t.shape.k), @intCast(t.step()) }, 6);
        p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 7);
        p.e.setBuffer(t.outs[1].buf, t.outs[1].off, 8);
        p.e.dispatchGroups(mtl.Size.of(t.groups(), 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *MarkovStep) usize {
        return 2 * t.groups();
    }

    /// Id v's float64 score and its bound (`sim`: the rank's sum in that accumulator and order instead).
    fn score(t: *const MarkovStep, v: usize, sim: ?Sim) struct { f64, f64 } {
        const k = t.shape.k;
        var terms: [256]f64 = undefined;
        var abs: f64 = 0;
        var sum: f64 = 0;
        var spread: f64 = 0;
        for (0..k) |r| {
            const sv: f64 = c.fromBf16(t.s2[v * k / 64 + r / 64]);
            const bv: f64 = c.fromBf16(t.b2[v * k / 64 + r / 64]);
            const q: f64 = @floatFromInt((t.w2[v * k / 4 + r / 4] >> @intCast(8 * (r % 4))) & 0xff);
            const e: f64 = c.fromBf16(t.w1[prev_id * k + r]);
            // bf16(s q + b) from an fp32 product and add (or one fma): either rounding's bf16
            const lo = bf(d.f32r(d.f32r(sv * q) + bv));
            const hi = bf(sv * q + bv);
            terms[r] = lo * e;
            sum += terms[r];
            abs += @abs(terms[r]);
            spread += @abs(hi - lo) * @abs(e);
        }
        const logit: f64 = c.fromBf16(t.l[t.step() * t.shape.n + v]);
        const s = if (sim) |o| contract.rows.sumIn(o[0], o[1], &terms) else sum;
        const exact = t.fill == .stress and sim == null;
        return .{ logit + s, if (exact) 0 else c.gamma(@floatFromInt(k + 1), u) * (@abs(logit) + abs) + spread };
    }

    const Pick = struct { v: f64, id: u32 };
    /// A group's pick by the kernel's own order: each simdgroup's xor butterfly, then thread 0 over the 8 parts.
    fn replay(scores: []const f64, base: usize, n: usize) Pick {
        var parts: [8]Pick = undefined;
        for (&parts, 0..) |*part, sg| {
            var lane: [32]Pick = undefined;
            for (&lane, 0..) |*l, i| {
                const v = base + sg * 32 + i;
                l.* = .{ .v = if (v < n) scores[v - base] else -std.math.inf(f64), .id = @intCast(v) };
            }
            var dist: usize = 16;
            while (dist > 0) : (dist /= 2) {
                const old = lane;
                for (&lane, 0..) |*l, i| {
                    const o = old[i ^ dist];
                    if (before(o.v, o.id, l.v, l.id)) l.* = o;
                }
            }
            part.* = lane[0];
        }
        var b = parts[0];
        for (parts[1..]) |p| if (before(p.v, p.id, b.v, b.id)) {
            b = p;
        };
        return b;
    }

    fn groupScores(t: *const MarkovStep, gi: usize, sim: ?Sim, scores: *[256]f64, bnd: *[256]f64) void {
        for (0..256) |i| {
            const v = gi * 256 + i;
            if (v >= t.shape.n) break;
            scores[i], bnd[i] = t.score(v, sim);
        }
    }

    fn outputs(t: *MarkovStep, out: []f64, allow: ?[]f64, sim: ?Sim) void {
        const ng = t.groups();
        var scores: [256]f64 = undefined;
        var bnd: [256]f64 = undefined;
        for (0..ng) |gi| {
            t.groupScores(gi, sim, &scores, &bnd);
            const p = replay(&scores, gi * 256, t.shape.n);
            out[gi] = p.v;
            out[ng + gi] = @as(f32, @bitCast(p.id));
            if (allow) |al| {
                al[gi] = bnd[p.id - gi * 256] + 0x1p-149;
                al[ng + gi] = 0;
            }
        }
    }

    pub fn reference(t: *MarkovStep, want: []f64, mag: []f64) !void {
        t.outputs(want, null, null);
        @memset(mag, 0);
    }
    pub fn bounds(t: *MarkovStep, _: []const f64, _: []const f64, allow: []f64) !void {
        const want = try t.a.alloc(f64, t.count());
        defer t.a.free(want);
        t.outputs(want, allow, null);
    }
    pub fn simulate(t: *MarkovStep, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(out, null, .{ acc, order });
    }

    /// Each group's best: its id in the group, its value within its id's bound, and the replay's pick or a near tie.
    pub fn check(t: *MarkovStep, _: g.Gate) !?[]const u8 {
        const ng = t.groups();
        const best = std.mem.bytesAsSlice(f32, t.rig.bytes(t.outs[0], ng * 4));
        const ids = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[1], ng * 4));
        var scores: [256]f64 = undefined;
        var bnd: [256]f64 = undefined;
        for (0..ng) |gi| {
            t.groupScores(gi, null, &scores, &bnd);
            const want = replay(&scores, gi * 256, t.shape.n);
            const id = ids[gi];
            if (id < gi * 256 or id >= @min(gi * 256 + 256, t.shape.n)) return try std.fmt.bufPrint(&t.msg, "group {d}'s best id {d} is outside it", .{ gi, id });
            const at = id - gi * 256;
            const w = want.id - gi * 256;
            const got: f64 = best[gi];
            if (std.math.isNan(want.v) or std.math.isNan(got) or std.math.isInf(want.v) or id == want.id) {
                if (id != want.id or c.classOf(got) != c.classOf(want.v)) return try std.fmt.bufPrint(&t.msg, "group {d} picks {d} ({d}) where the replay picks {d} ({d})", .{ gi, id, got, want.id, want.v });
            } else if (!(scores[at] >= scores[w] - bnd[at] - bnd[w])) return try std.fmt.bufPrint(&t.msg, "group {d} picks {d} ({d}), not within a near tie of {d} ({d})", .{ gi, id, scores[at], want.id, want.v });
            if (std.math.isFinite(got) and !(@abs(got - scores[at]) <= bnd[at] + 0x1p-149)) return try std.fmt.bufPrint(&t.msg, "group {d}'s best {d} is off its id's score {d}", .{ gi, got, scores[at] });
        }
        return null;
    }
};

// --- mm_markov_pick: the best of every group's best into seq[step + 1], bit for bit -------------------------------

pub const MarkovPick = struct {
    pub const Variant = enum { dspark };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    n: usize = 0,
    best: []f32 = &.{},
    ids: []u32 = &.{},
    old: [16]u32 = undefined,
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,
    const at = 5;

    /// Rows unused; n: the groups (782 for the vocabulary, and edges about 1,024 threads).
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "minimax_m3 draft dspark mm_markov_pick",
            .accum = .none,
            .depth = d.noSums,
            .out = .raw,
            .nan = .{ .stated = "the 1,024 threads' strided bests, then their xor butterflies, as Metal compares (a NaN compares false), replayed on the host bit for bit" },
            .tiles = .{ .{}, .{ .size = 1024, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 782, .k = 1 }, .{ .rows = 1, .n = 4, .k = 1 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant) !MarkovPick {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, .dspark, "mm_markov_pick") };
    }
    pub fn deinit(t: *MarkovPick) void {
        t.a.free(t.best);
        t.a.free(t.ids);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *MarkovPick) g.Kernel {
        return .{ .spec = spec(.dspark), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(MarkovPick);

    pub fn prepare(t: *MarkovPick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.best);
        t.a.free(t.ids);
        t.n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.best = try t.a.alloc(f32, shape.n);
        t.ids = try t.a.alloc(u32, shape.n);
        // coarse values, so ties between groups occur and the lower id must win
        for (t.best, t.ids, 0..) |*b, *i, k| {
            b.* = @floatFromInt(rng.uintLessThan(u32, 40));
            i.* = @intCast(k * 256 + rng.uintLessThan(u32, 256));
        }
        if (fill == .nonfinite) for ([_]f32{ d.nan, d.inf, -d.inf }, [_]usize{ 0, 2, 3 }) |v, k| {
            if (k < shape.n) t.best[k] = v;
        };
        for (&t.old) |*o| o.* = rng.int(u32);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.best));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.out = try t.rig.output(t.old.len * 4);
        return true;
    }

    pub fn run(t: *MarkovPick) !void {
        @memcpy(t.rig.bytes(t.out, t.old.len * 4), std.mem.sliceAsBytes(&t.old));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue([2]u32{ @intCast(t.n), at }, 2);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *MarkovPick) usize {
        return t.old.len * 4;
    }
    pub fn reference(_: *MarkovPick, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    const P = struct { v: f32, id: u32 };
    fn better(o: P, b: P) bool {
        return o.v > b.v or (o.v == b.v and o.id < b.id);
    }
    fn butterfly(lane: *[32]P) void {
        var dist: usize = 16;
        while (dist > 0) : (dist /= 2) {
            const old = lane.*;
            for (lane, 0..) |*l, i| if (better(old[i ^ dist], l.*)) {
                l.* = old[i ^ dist];
            };
        }
    }

    pub fn check(t: *MarkovPick, _: g.Gate) !?[]const u8 {
        var th: [1024]P = undefined;
        for (&th, 0..) |*x, i| {
            x.* = .{ .v = -std.math.inf(f32), .id = 0xffffffff };
            var k = i;
            while (k < t.n) : (k += 1024) {
                if (better(.{ .v = t.best[k], .id = t.ids[k] }, x.*)) x.* = .{ .v = t.best[k], .id = t.ids[k] };
            }
        }
        var firsts: [32]P = undefined;
        for (0..32) |sg| {
            butterfly(th[sg * 32 ..][0..32]);
            firsts[sg] = th[sg * 32];
        }
        butterfly(&firsts);
        var want = t.old;
        want[at + 1] = firsts[0].id;
        const got = t.rig.bytes(t.out, t.count());
        if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(&want))) return try std.fmt.bufPrint(&t.msg, "seq differs from the host's replay ({d} groups): {d} where it picks {d}", .{ t.n, std.mem.bytesAsSlice(u32, got)[at + 1], want[at + 1] });
        return null;
    }
};
