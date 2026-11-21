//! tf_hy4_recall_groups under the contract: each row's top 511 closed groups and its own, by a stated rule for ties.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const ix = @import("metal_hy4_index.zig");
const score = @import("metal_hy4_index_scores.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const recall = h.hy.index_recall;
const keep_most = ix.budget - 1;
const stride_tokens = ix.budget * ix.slots;

const rule = "a row keeps min(closed, 511) closed groups, the highest scores by fp32's order with +0 and -0 equal and every " ++
    "NaN above +inf, ties to the lower group, listed in group order, then its own group's slots to its position; " ++
    "count 4 x kept + position % 4 + 1, the rest of its list untouched";

/// The rule's rank of a score: fp32's total order with +0 and -0 equal and every NaN one value above +inf.
pub fn rank(x: f32) u32 {
    if (std.math.isNan(x)) return 0xffc00000;
    const bits: u32 = @bitCast(x + 0.0);
    return if (bits & 0x80000000 != 0) ~bits else bits | 0x80000000;
}

const Pair = struct { key: u32, i: u32 };
fn higher(_: void, x: Pair, y: Pair) bool {
    return x.key > y.key or (x.key == y.key and x.i < y.i);
}

/// The groups a row keeps by the rule, in group order, into `out` (its length the count kept).
pub fn keptGroups(row: []const f32, closed: usize, pairs: []Pair, out: []u32) []u32 {
    const keep = @min(closed, keep_most);
    if (closed <= keep) {
        for (out[0..keep], 0..) |*o, i| o.* = @intCast(i);
        return out[0..keep];
    }
    for (pairs[0..closed], 0..) |*p, i| p.* = .{ .key = rank(row[i]), .i = @intCast(i) };
    std.mem.sort(Pair, pairs[0..closed], {}, higher);
    for (out[0..keep], pairs[0..keep]) |*o, p| o.* = p.i;
    std.mem.sort(u32, out[0..keep], {}, std.sort.asc(u32));
    return out[0..keep];
}

pub const Recall = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    fault: ?ix.Control,
    label: [256]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    first: usize = 0,
    stride: usize = 0,
    scores: []f32 = &.{},
    rows: []recall.Row = &.{},
    pairs: []Pair = &.{},
    kept: []u32 = &.{},
    ins: [2]rig_.Ref = undefined,
    tokens: rig_.Ref = undefined,
    counts: rig_.Ref = undefined,
    msg: [256]u8 = undefined,

    pub const base = "metal hy4 index_recall tf_hy4_recall_groups";
    /// The lists' bytes before the launch, restored before every run: what a row does not write must stay.
    const old_token: i32 = -77;
    const old_count: i32 = -5;

    /// Rows: the pass's rows; n: the last row's closed groups (positions as the scores'); k: unused.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .raw,
            .nan = .{ .stated = rule },
            // 256 threads stride over a row's closed groups
            .tiles = .{ .{}, .{ .size = 3 * 256, .step = 1, .min = ix.budget }, .{} },
            // DFly2's window at 6k positions and at the budget's edge; decode at 80k; a prompt chunk; the longest
            .served = &.{ .{ .rows = h.window_rows, .n = 1500, .k = 1 }, .{ .rows = h.window_rows, .n = ix.budget, .k = 1 }, .{ .rows = 1, .n = 20000, .k = 1 }, .{ .rows = h.prompt_rows, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 65536, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, fault: ?ix.Control) !Recall {
        const pipe = try ix.pipeline(a, device, .recall, "tf_hy4_recall_groups", if (fault) |f| f.edit else null);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .fault = fault };
    }
    pub fn deinit(t: *Recall) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Recall) void {
        inline for (.{ "scores", "rows", "pairs", "kept" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Recall) g.Kernel {
        t.name = fl.named(base, if (t.fault) |f| f.drops else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = if (t.fault) |f| f.gate else null };
    }
    const vt = g.hooks(Recall);

    fn closed(t: *const Recall, r: usize) usize {
        return (t.first + r) / 4;
    }

    /// Scores on a quarter grid (ties at every threshold), zeros of either sign; +Inf past each row's closed groups.
    pub fn prepare(t: *Recall, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n < 1 or shape.n > 65536 or 4 * shape.n + 4 < shape.rows) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const R = shape.rows;
        t.first = score.lastPosition(shape) + 1 - R;
        t.stride = @min(shape.n + 5, 65536);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.scores = try t.a.alloc(f32, R * t.stride);
        t.rows = try t.a.alloc(recall.Row, R);
        t.pairs = try t.a.alloc(Pair, t.stride);
        t.kept = try t.a.alloc(u32, ix.budget);
        for (t.rows, 0..) |*row, r| row.* = .{ .group_base = 0, .token = @intCast(t.first + r) };
        for (0..R) |r| {
            const row = t.scores[r * t.stride ..][0..t.stride];
            for (row, 0..) |*v, i| {
                const q = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8;
                v.* = if (i >= t.closed(r)) fl.inf32 else if (q == 0 and rng.boolean()) -0.0 else q;
            }
            if (fill == .nonfinite) plant(row[0..t.closed(r)], r, rng);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.rows));
        t.tokens = try t.rig.output(R * stride_tokens * 4);
        t.counts = try t.rig.output(R * 4);
        return true;
    }

    /// Gate 1's rows: NaN; both infinities; NaNs of each sign and payload; -Inf; zeros of either sign; all of them.
    fn plant(row: []f32, r: usize, rng: std.Random) void {
        const nan_neg: f32 = @bitCast(@as(u32, 0xffc00000));
        const nan_pay: f32 = @bitCast(@as(u32, 0x7fc00123));
        const n = row.len;
        switch (r % 6) {
            0 => for (0..40) |j| {
                row[(7 * j + 3) % n] = fl.nan32;
            },
            1 => {
                for (0..600) |j| row[(2 * j) % n] = -fl.inf32;
                for (0..30) |j| row[(11 * j + 1) % n] = fl.inf32;
            },
            // 700 NaNs of three kinds, so the threshold falls among them: the rule keeps the 511 lowest groups
            2 => for (0..700) |j| {
                row[(2 * j) % n] = switch (j % 3) {
                    0 => nan_neg,
                    1 => nan_pay,
                    else => fl.nan32,
                };
            },
            3 => for (row[0..n -| 500]) |*v| {
                v.* = -fl.inf32;
            },
            4 => for (row) |*v| {
                v.* = if (rng.boolean()) -0.0 else 0.0;
            },
            else => for (0..n / 3) |j| {
                const i = (j * 5 + 1) % n;
                row[i] = switch (j % 7) {
                    0 => fl.nan32,
                    1 => -fl.inf32,
                    2 => fl.inf32,
                    3 => nan_neg,
                    4 => -0.0,
                    else => 0.0,
                };
            },
        }
    }

    pub fn run(t: *Recall) !void {
        const R = t.shape.rows;
        @memset(std.mem.bytesAsSlice(i32, t.rig.bytes(t.tokens, R * stride_tokens * 4)), old_token);
        @memset(std.mem.bytesAsSlice(i32, t.rig.bytes(t.counts, R * 4)), old_count);
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setBuffer(t.tokens.buf, t.tokens.off, 2);
        e.setBuffer(t.counts.buf, t.counts.off, 3);
        e.setValue(recall.Shape{ .rows = @intCast(R), .score_groups = @intCast(t.stride), .group_budget = ix.budget }, 4);
        // as metal_index.zig: a threadgroup of 256 a row
        e.dispatchGroups(mtl.Size.of(R, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Recall) usize {
        return t.shape.rows * (stride_tokens + 1) * 4;
    }
    pub fn reference(_: *Recall, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Recall, _: g.Gate) !?[]const u8 {
        const R = t.shape.rows;
        const tokens = std.mem.bytesAsSlice(i32, t.rig.bytes(t.tokens, R * stride_tokens * 4));
        const counts = std.mem.bytesAsSlice(i32, t.rig.bytes(t.counts, R * 4));
        for (0..R) |r| {
            const pos = t.first + r;
            const cl = t.closed(r);
            const kept = keptGroups(t.scores[r * t.stride ..][0..t.stride], cl, t.pairs, t.kept);
            const want_count = kept.len * 4 + pos % 4 + 1;
            if (counts[r] != want_count) return try std.fmt.bufPrint(&t.msg, "row {d} (closed {d}) counts {d} where the rule gives {d}", .{ r, cl, counts[r], want_count });
            const list = tokens[r * stride_tokens ..][0..stride_tokens];
            for (list, 0..) |got, i| {
                const want: i32 = if (i >= want_count) old_token else @intCast((if (i / 4 < kept.len) kept[i / 4] else cl) * 4 + i % 4);
                if (got != want) return try std.fmt.bufPrint(&t.msg, "row {d} (closed {d}) slot {d} is {d} where the rule gives {d}", .{ r, cl, i, got, want });
            }
        }
        return null;
    }
};

test "the rule's picks: the highest by rank, ties and signed zeros to the lower group, NaN first, in group order" {
    var pairs: [1024]Pair = undefined;
    var out: [ix.budget]u32 = undefined;
    var row: [1024]f32 = @splat(1);
    row[3] = fl.nan32;
    row[600] = -0.0;
    for (row[700..]) |*v| v.* = 0.0;
    row[0] = -fl.inf32;
    // 1,024 closed, 511 kept: the NaN at 3 and the 510 lowest groups holding 1; -inf at 0 and every zero dropped
    const kept = keptGroups(&row, 1024, &pairs, &out);
    try std.testing.expectEqual(@as(usize, keep_most), kept.len);
    try std.testing.expectEqual(@as(u32, 1), kept[0]);
    try std.testing.expect(std.mem.indexOfScalar(u32, kept, 3) != null);
    try std.testing.expect(std.mem.indexOfScalar(u32, kept, 0) == null);
    try std.testing.expectEqual(rank(0.0), rank(-0.0));
    try std.testing.expect(rank(fl.nan32) > rank(fl.inf32));
}
