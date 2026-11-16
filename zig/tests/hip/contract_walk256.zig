//! The planned decode walk's d = 256 cases (the Qwen checkpoints' attention) under the contract, and their control.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const kit = @import("contract_kit.zig");

const c = core.contract.spec;

/// Spans the long-span score takes from (launches/plan.zig's score_span).
const long_span = 16384;
/// Steps a long-span score block walks at d = 256 (launches/plan.zig's score_steps, plan.hip's kScoreSteps).
const steps = 8;

/// Query heads over KV heads on short spans (two keys a wave) or long ones (four, eight steps a block).
pub const Case = struct { heads: usize, kv: usize, long: bool, served: []const c.Shape };

/// served[0] takes gates 1-2 (16 heads whole vector stores, 24 the scalar tail), gate 4 the largest; one has 8 rows.
pub const cases = [_]Case{
    .{ .heads = 16, .kv = 2, .long = false, .served = &.{
        .{ .rows = 4, .n = 16, .k = 4096 },
        .{ .rows = 2, .n = 16, .k = 1023 },
        .{ .rows = 8, .n = 16, .k = 8192 },
    } },
    .{ .heads = 16, .kv = 2, .long = true, .served = &.{
        .{ .rows = 4, .n = 16, .k = 16384 },
        .{ .rows = 8, .n = 16, .k = 16387 },
        .{ .rows = 2, .n = 16, .k = 32768 },
    } },
    .{ .heads = 24, .kv = 4, .long = false, .served = &.{
        .{ .rows = 4, .n = 24, .k = 4093 },
        .{ .rows = 2, .n = 24, .k = 1002 },
        .{ .rows = 8, .n = 24, .k = 8192 },
    } },
    .{ .heads = 24, .kv = 4, .long = true, .served = &.{
        .{ .rows = 4, .n = 24, .k = 16386 },
        .{ .rows = 8, .n = 24, .k = 20480 },
        .{ .rows = 2, .n = 24, .k = 32768 },
    } },
};

/// The case the control runs: 16 heads over 2 on long spans.
pub const control_case = 1;

/// Each case's registrations, q fp32 and in the cache's kind [case][act][bf16].
const names = blk: {
    var t: [cases.len][2][2][]const u8 = undefined;
    for (cases, 0..) |x, i| for (0..2) |act| for (0..2) |bf| {
        const call = if (act == 1) "planCausalAs" else "planCausal";
        const span = if (x.long) "long" else "short";
        const kind = if (bf == 1) "bf16" else "fp16";
        const what = if (act == 1) "q and rows " ++ kind else kind ++ " cache";
        t[i][act][bf] = std.fmt.comptimePrint("hip planned decode walk {s} d 256, {d} heads over {d}, {s} spans ({s})", .{ call, x.heads, x.kv, span, what });
    };
    break :blk t;
};

/// The real launches, or the control's: the long-span score grid counted for twice the steps its kernel walks.
pub const Fault = enum { none, steps };

/// Case `i`'s spec; tiles on the keys at the apply's tile (short) or a score block past the least long span (long).
pub fn spec(i: usize, k: kit.Kind, act: bool, fault: Fault, depth: *const fn (c.Shape) usize) c.Spec {
    const x = cases[i];
    var s: c.Spec = .{
        .name = names[i][@intFromBool(act)][@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = depth,
        .out = if (act) kit.elem(k) else .f32,
        .tiles = .{ .{}, .{}, if (x.long) .{ .size = long_span + 128, .min = long_span } else .{ .size = 128 } },
        .served = x.served,
    };
    if (act) s.replaces = "tf_cast, planCausal and tf_cast";
    if (fault == .steps) s.name = "control: hip planned decode walk d 256, its long-span score grid counted for 16 steps a block";
    return s;
}

/// Row r's visible keys at span k: the span's end, cuts inside a wave's keys, on a wave's, a step's and a block's edge.
pub fn visible(r: usize, k: usize) usize {
    const m = k / 2 / 128 * 128; // a long-span block's first key, so a step's and a wave's too
    const cut = switch (r % 8) {
        0 => k,
        1 => m + 5,
        2 => m,
        3 => m + 7,
        4 => 1,
        5 => m + 16,
        6 => k - 1,
        else => m + 4,
    };
    return std.math.clamp(cut, 1, k);
}

/// The control's walk: planCausal's four launches at d = 256 (q fp32), the score grid for 2 * steps steps a block.
pub fn faulty(l: *const hip.Launcher, st: hip.Stream, q: u64, out: u64, scratch: [3]u64, p: hip.launches.PlanRef, heads: usize, kv: usize, span: usize, scale: f32, kind: kit.Kind) !void {
    const which: usize = @intCast(@backingInt(kind) - 1);
    const rows: u32 = @intCast(p.rows);
    const tiles: c_int = @intCast((span + 127) / 128);
    const fixed = [_]c_int{ @intCast(span), @intCast(heads), @intCast(kv), 256 };
    var a: hip.launch.Args = .{};
    try a.add(q);
    try a.add(scratch[0]);
    for (fixed) |x| try a.add(x);
    try a.add(scale);
    try a.add(p.args);
    try a.add(@as(c_int, 0));
    const blocks: u32 = @intCast((span + 4 * 4 * 2 * steps - 1) / (4 * 4 * 2 * steps));
    const lds: u32 = @intCast(heads / kv * 256 * 4);
    try hip.launch.launch(l.plan.score[which], .{ .grid = .{ .x = blocks, .y = @intCast(kv), .z = rows }, .block = .{ .x = 128 }, .shared = lds }, st, &a);
    var b: hip.launch.Args = .{};
    try b.add(scratch[0]);
    try b.add(scratch[1]);
    try b.add(fixed[0]);
    try b.add(@as(c_int, 0));
    try b.add(p.args.pos);
    try hip.launch.launch(l.softmax_stats, .{ .grid = .{ .x = @intCast(heads), .y = rows }, .block = .{ .x = 256 } }, st, &b);
    var e: hip.launch.Args = .{};
    for (scratch) |x| try e.add(x);
    for (fixed) |x| try e.add(x);
    try e.add(p.args);
    try e.add(@as(c_int, 0));
    try hip.launch.launch(l.plan.apply[which], .{ .grid = .{ .x = @intCast(tiles), .y = @intCast(kv), .z = rows }, .block = .{ .x = 256 } }, st, &e);
    var f: hip.launch.Args = .{};
    try f.add(scratch[2]);
    try f.add(out);
    try f.add(tiles);
    try f.add(fixed[3]);
    try hip.launch.launch(l.sum_partials, .{ .grid = .{ .x = @intCast(heads), .y = rows }, .block = .{ .x = 256 } }, st, &f);
}
