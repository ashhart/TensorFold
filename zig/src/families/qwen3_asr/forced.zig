//! The teacher-forced tie rule (#2648's): on the reference's history, a differing top token passes only as a swap.
const std = @import("std");
const case_ = @import("case.zig");

/// The widest top-1/top-2 margin a swapped step may have, on either side, in logits.
pub const bar: f32 = 0.25;

/// A step's two best tokens and the first one's lead.
pub const Top = struct { first: u32, second: u32, margin: f32 };

pub const Step = enum { same, swap, fail };

/// Same top token, or the two tokens swapped with both margins at most `max`; anything else (NaN too) fails.
pub fn judge(ours: Top, ref: Top, max: f32) Step {
    if (ours.first == ref.first and !std.math.isNan(ours.margin)) return .same;
    const swapped = ours.first == ref.second and ours.second == ref.first;
    return if (swapped and ours.margin <= max and ref.margin <= max) .swap else .fail;
}

/// The reference's step from its topk pair and its greedy id: topk may list an exact tie (margin 0) either way round.
pub fn reference(first: u32, second: u32, margin: f32, greedy: u32) error{TopIsNotGreedy}!Top {
    var r: Top = .{ .first = first, .second = second, .margin = margin };
    if (r.first != greedy and r.second == greedy and r.margin == 0) std.mem.swap(u32, &r.first, &r.second);
    return if (r.first == greedy) r else error.TopIsNotGreedy;
}

fn widen(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// The two best of bf16 logits, each the first of equal values (torch.argmax's rule); a NaN makes the margin NaN.
pub fn top(logits: []const u16) Top {
    var a: u32 = 0;
    var b: ?u32 = null;
    var nan = false;
    for (logits, 0..) |bits, i| {
        const v = widen(bits);
        if (std.math.isNan(v)) nan = true;
        if (i == 0) continue;
        if (v > widen(logits[a])) {
            b = a;
            a = @intCast(i);
        } else if (b == null or v > widen(logits[b.?])) b = @intCast(i);
    }
    const second = b orelse a;
    const m = widen(logits[a]) - widen(logits[second]);
    return .{ .first = a, .second = second, .margin = if (nan) std.math.nan(f32) else m };
}

/// The largest |a - b| over every column of every step, where it is, and the mean |a - b|.
pub const Dist = struct {
    worst: f64 = 0,
    step: usize = 0,
    col: usize = 0,
    sum: f64 = 0,
    n: usize = 0,

    pub fn add(d: *Dist, step: usize, a: []const f64, b: []const f32) void {
        for (a, b, 0..) |x, y, c| {
            const e = @abs(x - y);
            d.sum += e;
            d.n += 1;
            if (!(e <= d.worst)) {
                d.worst = e;
                d.step = step;
                d.col = c;
            }
        }
    }

    pub fn print(d: Dist, label: []const u8) void {
        std.debug.print("DIST {s}: worst column {d:.6} (step {d}, id {d}), mean {e:.4} over {d} logits\n", .{ label, d.worst, d.step, d.col, d.sum / @as(f64, @floatFromInt(@max(d.n, 1))), d.n });
    }
};

/// A clip's steps against the reference's, each swap or fail printed as it is met; then the FORCED line.
pub const Tally = struct {
    same: usize = 0,
    swaps: usize = 0,
    fails: usize = 0,

    /// Step k: our bf16 logits row against the case's top pair, margin and greedy id.
    pub fn step(t: *Tally, clip: []const u8, k: usize, logits: []const u16, first: u32, second: u32, margin: f32, greedy: u32) !void {
        const ours = top(logits);
        const ref = try reference(first, second, margin, greedy);
        switch (judge(ours, ref, bar)) {
            .same => t.same += 1,
            inline else => |v| {
                if (v == .swap) t.swaps += 1 else t.fails += 1;
                std.debug.print("{s} {s} step {d}: ours {d} (margin {d:.4}, runner-up {d}), reference {d} (margin {d:.4}, runner-up {d})\n", .{ if (v == .swap) "SWAP" else "FAIL", clip, k, ours.first, ours.margin, ours.second, ref.first, ref.margin, ref.second });
            },
        }
    }

    pub fn print(t: Tally, clip: []const u8, steps: usize, note: []const u8) void {
        std.debug.print("FORCED {s}: {d} steps, {d} same top token, {d} swaps within {d}, {d} fail{s}\n", .{ clip, steps, t.same, t.swaps, bar, t.fails, note });
    }
};

/// With logits_fp32 and logits_bf16 in the case: ours (`steps`, bf16 rows) and the package's bf16 to its fp32.
pub fn distances(gpa: std.mem.Allocator, case: *case_.Case, steps: []const u16, n: usize, vocab: usize) !void {
    const fp32 = case.floats(gpa, "logits_fp32") catch |e| return if (e == error.NoSuchStage) {} else e;
    defer gpa.free(fp32);
    const bf16 = try case.floats(gpa, "logits_bf16");
    defer gpa.free(bf16);
    if (fp32.len != n * vocab or bf16.len != fp32.len or steps.len < n * vocab) return error.StageSize;
    var ours: Dist = .{};
    var pkg: Dist = .{};
    const row = try gpa.alloc(f64, vocab);
    defer gpa.free(row);
    for (0..n) |k| {
        for (row, steps[k * vocab ..][0..vocab]) |*r, b| r.* = widen(b);
        ours.add(k, row, fp32[k * vocab ..][0..vocab]);
        for (row, bf16[k * vocab ..][0..vocab]) |*r, b| r.* = b;
        pkg.add(k, row, fp32[k * vocab ..][0..vocab]);
    }
    ours.print("ours to the package's fp32");
    pkg.print("the package's bf16 to its fp32");
}

fn bf(x: f32) u16 {
    return @intCast(@as(u32, @bitCast(x)) >> 16);
}

test "top: the two best, the first of equals, the lead" {
    const t = top(&.{ bf(1), bf(3), bf(2.5), bf(3) });
    try std.testing.expectEqual(Top{ .first = 1, .second = 3, .margin = 0 }, t);
    try std.testing.expectEqual(Top{ .first = 0, .second = 2, .margin = 0.5 }, top(&.{ bf(4), bf(1), bf(3.5) }));
    try std.testing.expect(std.math.isNan(top(&.{ bf(1), 0x7fc0 }).margin));
}

test "judge: a swap within 0.25 on both sides passes; a wider margin, a third token or a NaN fails" {
    const ref: Top = .{ .first = 10, .second = 20, .margin = 0.125 };
    try std.testing.expectEqual(Step.same, judge(.{ .first = 10, .second = 30, .margin = 3 }, ref, bar));
    try std.testing.expectEqual(Step.swap, judge(.{ .first = 20, .second = 10, .margin = 0.25 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 20, .second = 10, .margin = 0.375 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 20, .second = 10, .margin = 0 }, .{ .first = 10, .second = 20, .margin = 0.5 }, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 20, .second = 30, .margin = 0 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 30, .second = 10, .margin = 0 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 10, .second = 20, .margin = std.math.nan(f32) }, ref, bar));
}

test "reference: an exact tie listed the other way round is the greedy pick's; else the greedy id must lead" {
    try std.testing.expectEqual(Top{ .first = 20, .second = 10, .margin = 0 }, try reference(10, 20, 0, 20));
    try std.testing.expectEqual(Top{ .first = 10, .second = 20, .margin = 0 }, try reference(10, 20, 0, 10));
    try std.testing.expectError(error.TopIsNotGreedy, reference(10, 20, 0.125, 20));
    try std.testing.expectError(error.TopIsNotGreedy, reference(10, 20, 0, 30));
    // as on 1.7B 1089-134686-0000 step 21: the tie decides a swap, not a fail
    const ref = try reference(323, 11, 0, 11);
    try std.testing.expectEqual(Step.swap, judge(.{ .first = 323, .second = 11, .margin = 0.25 }, ref, bar));
}
