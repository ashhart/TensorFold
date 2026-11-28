//! tf_hy4_group_scores under the contract: each closed group's sum over 16 heads of w_h * scale * relu(q_h . k).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const ix = @import("metal_hy4_index.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const u = fl.u32_ulp;
const W = ix.width;
const H = ix.heads;
const recall = h.hy.index_recall;

/// A lane's 4 fmas then the 5 shuffle adds; the 16 heads' fma chain and the weight's rounding.
const dot_depth = 9;
const head_depth = H + 1;
fn depth(_: c.Shape) usize {
    return dot_depth + head_depth;
}

/// The rows' positions for a shape: consecutive, the last at 4 n + 3 (its own group the n-th, closed n).
pub fn lastPosition(shape: c.Shape) usize {
    return 4 * shape.n + 3;
}

const Variant = struct { nan_dot: bool, fault: ?ix.Control };

pub const Scores = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    base_buf: [192]u8 = undefined,
    label: [256]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    first: usize = 0,
    stride: usize = 0,
    q: []u16 = &.{},
    keys: []u16 = &.{},
    hw: []u16 = &.{},
    rows: []recall.Row = &.{},
    /// The score rows before the launch, restored before every run: an entry the kernel skips keeps its value.
    old: []f32 = &.{},
    allow: []f64 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn base(nan_dot: bool, buf: []u8) []const u8 {
        return ix.baseName(.recall, "tf_hy4_group_scores", if (nan_dot) "NaN dot" else "", buf);
    }
    /// Rows: the pass's rows; n: the last row's closed groups; k: the 16 x 128 products a score sums.
    pub fn spec(name: []const u8, v: Variant) c.Spec {
        const K = H * W;
        // DFly2's window at the budget's edge, decode at 16k, the first prompt chunk past it, decode at 64k
        const served = [_]c.Shape{ .{ .rows = h.window_rows, .n = ix.budget, .k = K }, .{ .rows = 1, .n = 4096, .k = K }, .{ .rows = h.prompt_rows, .n = 640, .k = K }, .{ .rows = 1, .n = 16384, .k = K } };
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 1,
            .out = .f32,
            .tiles = if (v.nan_dot) .{ .{}, .{}, .{} } else .{ .{}, .{ .size = ix.budget + 2, .step = 1, .min = ix.budget }, .{} },
            .served = if (v.nan_dot) served[0..1] else &served,
            .sums = .{ .exact = false, .why = "the dots exact; the heads' fma chain is in the derived bound" },
            .known = if (v.nan_dot) .{ .gate = .nonfinite, .finding = "a NaN dot scores 0 where float64 keeps the NaN: its relu is max(dot, 0.0f), which drops a NaN" } else null,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, nan_dot: bool, fault: ?ix.Control) !Scores {
        const pipe = try ix.pipeline(a, device, .recall, "tf_hy4_group_scores", if (fault) |f| f.edit else null);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .v = .{ .nan_dot = nan_dot, .fault = fault } };
    }
    pub fn deinit(t: *Scores) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Scores) void {
        inline for (.{ "q", "keys", "hw", "rows", "old", "allow" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Scores) g.Kernel {
        t.name = fl.named(base(t.v.nan_dot, &t.base_buf), if (t.v.fault) |f| f.drops else null, &t.label);
        return .{ .spec = spec(t.name, t.v), .ctx = t, .vt = &vt, .control = if (t.v.fault) |f| f.gate else null };
    }
    const vt = g.hooks(Scores);

    fn closed(t: *const Scores, r: usize) usize {
        return (t.first + r) / 4;
    }
    /// Whether row r scores: past the budget (recall then selects) and within the score rows.
    fn scores(t: *const Scores, r: usize) bool {
        return t.closed(r) >= ix.budget and t.closed(r) <= t.stride;
    }

    pub fn prepare(t: *Scores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != H * W or shape.n < ix.budget or shape.n > 65536 or 4 * shape.n + 4 < shape.rows) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const R = shape.rows;
        t.first = lastPosition(shape) + 1 - R;
        // the score rows' stride (the engine's capacity / 4) past the last row's closed groups
        t.stride = @min(shape.n + 3, 65536);
        const groups = shape.n + 2;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.q = try t.a.alloc(u16, R * H * W);
        t.keys = try t.a.alloc(u16, groups * W);
        t.hw = try t.a.alloc(u16, R * H);
        t.rows = try t.a.alloc(recall.Row, R);
        t.old = try t.a.alloc(f32, R * t.stride);
        t.allow = try t.a.alloc(f64, R * t.stride);
        for (t.rows, 0..) |*row, r| row.* = .{ .group_base = 0, .token = @intCast(t.first + r) };
        for (t.old) |*v| v.* = -40 - rng.float(f32) * 20;
        for (t.hw) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        if (fill == .stress) t.stress(rng) else {
            for (t.q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            for (t.keys) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        }
        if (fill == .nonfinite and R >= 4) {
            const a = R - 4;
            if (t.v.nan_dot) {
                // a NaN query and a NaN key column: their dots are NaN
                t.q[(a * H + 4) * W + 7] = c.toBf16(fl.nan32);
                t.keys[5 * W + 3] = c.toBf16(fl.nan32);
            } else {
                // the last four rows (all scoring): a NaN head weight, +Inf and -Inf queries, both in one row
                t.hw[a * H + 2] = c.toBf16(fl.nan32);
                t.q[((a + 1) * H + 3) * W + 5] = c.toBf16(fl.inf32);
                t.q[((a + 2) * H + 7) * W + 9] = c.toBf16(-fl.inf32);
                t.q[((a + 3) * H + 1) * W + 0] = c.toBf16(fl.inf32);
                t.q[((a + 3) * H + 11) * W + 100] = c.toBf16(-fl.inf32);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.keys));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.hw));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.rows));
        t.out = try t.rig.output(t.old.len * 4);
        return true;
    }

    /// Gate 4's dots: products 512 x integers past fp16's range, dims d and 127 - d cancelling but for two nudges.
    fn stress(t: *Scores, rng: std.Random) void {
        for (0..t.keys.len / W) |gi| for (0..W / 2) |d| {
            const v: f32 = if (rng.boolean()) 512 else -512;
            t.keys[gi * W + d] = c.toBf16(v);
            t.keys[gi * W + W - 1 - d] = c.toBf16(-v);
        };
        for (0..t.shape.rows * H) |rh| {
            const x = t.q[rh * W ..][0..W];
            for (0..W / 2) |d| {
                const v: u32 = 128 + rng.uintLessThan(u32, 125);
                const nudge: u32 = if (d == rh % 64 or d == (rh * 7 + 3) % 64) 1 + @as(u32, @intCast(d % 3)) else 0;
                x[d] = c.toBf16(@floatFromInt(v));
                x[W - 1 - d] = c.toBf16(@floatFromInt(v + nudge));
            }
        }
    }

    pub fn run(t: *Scores) !void {
        @memcpy(t.rig.bytes(t.out, t.old.len * 4), std.mem.sliceAsBytes(t.old));
        const R = t.shape.rows;
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(t.out.buf, t.out.off, 4);
        e.setValue(recall.Shape{ .rows = @intCast(R), .score_groups = @intCast(t.stride), .group_budget = ix.budget }, 5);
        e.setValue(recall.head_scale, 6);
        // as metal_index.zig: four groups a threadgroup up to the last row's closed groups, a row of them per row
        const last = t.first + R - 1;
        e.dispatchGroups(mtl.Size.of((last / 4 + 3) / 4, R, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Scores) usize {
        return t.shape.rows * t.stride;
    }

    fn relu(x: f64) f64 {
        return if (x > 0) x else if (x <= 0) 0 else x;
    }

    /// Row r's score of group gi (dots in `sim` if given) and its allowance: the dots' gamma(9), the chain's gamma(17).
    fn one(t: *const Scores, r: usize, gi: usize, sim: ?g.Order, acc: c.Accum) [2]f64 {
        const k = t.keys[gi * W ..][0..W];
        const scale: f64 = recall.head_scale;
        var s: f64 = 0;
        var dots: f64 = 0;
        var chain: f64 = 0;
        var heads: [H]f64 = undefined;
        var terms: [W]f64 = undefined;
        for (0..H) |hd| {
            const q = t.q[(r * H + hd) * W ..][0..W];
            var dot: f64 = 0;
            var m: f64 = 0;
            for (&terms, q, k) |*x, qv, kv| {
                x.* = @as(f64, c.fromBf16(qv)) * c.fromBf16(kv);
                dot += x.*;
                m += @abs(x.*);
            }
            if (sim) |o| dot = contract.rows.sumIn(acc, o, &terms);
            const w = @as(f64, c.fromBf16(t.hw[r * H + hd])) * scale;
            const ed = if (t.fill == .stress) 0 else c.gamma(dot_depth, u) * m;
            heads[hd] = w * relu(dot);
            s += heads[hd];
            dots += @abs(w) * ed;
            chain += @abs(w) * (relu(dot) + ed);
        }
        if (sim) |o| s = contract.rows.sumIn(acc, o, &heads);
        const e = (1 + u) * dots + c.gamma(head_depth, u) * (1 + u) * chain + (H + 1) * 0x1p-126;
        return .{ s, e };
    }

    const Job = struct {
        t: *const Scores,
        want: []f64,
        lo: usize,
        hi: usize,
        pub fn run(j: *Job) void {
            const t = j.t;
            for (j.lo..j.hi) |r| for (0..t.stride) |gi| {
                const i = r * t.stride + gi;
                if (!t.scores(r) or gi >= t.closed(r)) {
                    j.want[i] = t.old[i];
                    t.allow[i] = 0;
                    continue;
                }
                const x = t.one(r, gi, null, .f32);
                j.want[i] = x[0];
                t.allow[i] = x[1];
            };
        }
    };

    /// Float64 of the kernel's formula; an entry it skips (a row under the budget, a group not closed) keeps its value.
    pub fn reference(t: *Scores, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const R = t.shape.rows;
        var jobs: [contract.reference_threads.maximum]Job = undefined;
        const n = contract.reference_threads.count(contract.reference_threads.maximum, R);
        for (jobs[0..n], 0..) |*j, i| j.* = .{ .t = t, .want = want, .lo = R * i / n, .hi = R * (i + 1) / n };
        contract.reference_threads.run(Job, jobs[0..n], .normal);
    }

    pub fn bounds(t: *Scores, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, t.allow);
    }

    pub fn simulate(t: *Scores, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.shape.rows) |r| for (0..t.stride) |gi| {
            const i = r * t.stride + gi;
            out[i] = if (!t.scores(r) or gi >= t.closed(r)) t.old[i] else t.one(r, gi, order, acc)[0];
        };
    }
};
