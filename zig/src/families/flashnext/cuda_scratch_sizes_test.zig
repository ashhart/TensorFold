//! part's rows against every consumer in the pass's scratch: the out projection, a legacy out lane and the head.
const std = @import("std");
const sizes = @import("cuda_scratch_sizes.zig");
const plan = @import("cuda_mtp_plan.zig");
const mtp = @import("cuda_mtp.zig");
const wt = @import("cuda_prompt_wt.zig");

const streams = wt.streams;
const out_sk = wt.out_sk;
const fc_sk = mtp.fc_sk;
const piece_rows = mtp.piece_rows;

/// part's rows a lane over m rows writes (cuda_tiles.matmul): two of the native lane, sk below 64, one when K folds.
fn laneRows(native: bool, m: usize, sk: usize) usize {
    if (sk == 1) return 0;
    if (native) return 2 * m;
    return if (m >= 64) m else sk * m;
}

/// An out projection over a pass of `n` rows: its own slices, and the writeback reading the zero slice at cap rows.
fn outRows(cap: usize, native: bool, n: usize) usize {
    return @max(laneRows(native, n, out_sk), cap + n);
}

/// The head's widest reachable pass at `cap` rows: the plan's piece, or one grouped lane's rows (a verify window).
fn headPass(cap: usize) usize {
    return @min(cap, @max(plan.plan(cap, cap).piece, mtp.max_rows));
}

/// The head's fc_h lane over streams * rows rows, whichever lane the projection gets.
fn headNeed(cap: usize) usize {
    const m = streams * headPass(cap);
    return @max(laneRows(true, m, fc_sk), laneRows(false, m, fc_sk));
}

/// The head's widest pass the lane itself admits: forwardHeadMany refuses only a lane over piece_rows rows.
fn headWidest(cap: usize) usize {
    const m = streams * @min(cap, piece_rows);
    return @max(laneRows(true, m, fc_sk), laneRows(false, m, fc_sk));
}

/// Caps at or above this one carry a head pass: the plan's reserveRows floors the scratch at 32 rows.
const head_floor: usize = 32;

/// A sweep's record of the caps whose requirement left part: the count, the first and last, the first's numbers.
const Short = struct {
    caps: usize = 0,
    first: usize = 0,
    last: usize = 0,
    need: usize = 0,
    have: usize = 0,
};

fn one(s: *Short, cap: usize, need: usize, have: usize) void {
    if (need <= have) return;
    s.caps += 1;
    if (s.caps == 1) {
        s.first = cap;
        s.need = need;
        s.have = have;
    }
    s.last = cap;
}

fn report(what: []const u8, s: Short) void {
    if (s.caps == 0) return;
    std.debug.print("part rows: {s} is short at {d} caps, {d} to {d}\n", .{ what, s.caps, s.first, s.last });
    std.debug.print("part rows: first cap {d} needs {d} rows, partRows gives {d}\n", .{ s.first, s.need, s.have });
}

/// The caps every sweep ends with: a full window's cache and the long ones a paged draft admits.
const far_caps = [_]usize{ 4096, 8192, 16384, 32768, 65536, 131072, 262144 };

fn sweep(comptime need: fn (usize) usize, from: usize) Short {
    var s: Short = .{};
    var cap = from;
    while (cap < far_caps[0]) : (cap += 1) one(&s, cap, need(cap), sizes.partRows(cap));
    for (far_caps) |c| one(&s, c, need(c), sizes.partRows(c));
    return s;
}

test "Flash Next part rows: the out projection and a legacy out lane fit at every cap" {
    var s: Short = .{};
    var cap: usize = 1;
    while (cap < far_caps[0]) : (cap += 1) oneOut(&s, cap);
    for (far_caps) |c| oneOut(&s, c);
    report("the out projection", s);
    try std.testing.expectEqual(@as(usize, 0), s.caps);
}

/// A pass's rows where a lane's write changes: one row, the 16-row tile's last (63), the prompt tile's first, all.
fn oneOut(s: *Short, cap: usize) void {
    const have = sizes.partRows(cap);
    for ([_]usize{ 1, @min(cap, 63), @min(cap, 64), cap }) |n| {
        one(s, cap, outRows(cap, true, n), have);
        one(s, cap, outRows(cap, false, n), have);
    }
}

test "Flash Next part rows: the MTP head's fc_h lane fits part at every cap it runs" {
    // a head pass runs on a scratch of at least the plan's floor, whatever chunk ran before it
    try std.testing.expectEqual(head_floor, plan.reserveRows(1));
    try std.testing.expectEqual(head_floor, plan.reserveRows(head_floor / 4));
    // a pass of `head_floor` rows carries no head piece yet: the lane's own rows are the floor there
    try std.testing.expectEqual(head_floor, headPass(head_floor));
    try std.testing.expectEqual(@as(usize, 2 * head_floor), headPass(4 * head_floor));
    const reachable = sweep(headNeed, head_floor);
    report("the head's fc_h lane over a pass", reachable);
    try std.testing.expectEqual(@as(usize, 0), reachable.caps);
    const widest = sweep(headWidest, head_floor);
    report("the head's fc_h lane over piece_rows", widest);
    try std.testing.expectEqual(@as(usize, 0), widest.caps);
}

fn expectLeanPartFits(parallel: usize, admitted: usize, prompt_tokens: usize) !usize {
    const cap = @max(admitted, plan.reservePrefillRows(prompt_tokens - 1));
    const piece = @min(prompt_tokens - 1, plan.prefillPiece(cap));
    const have = sizes.partRows(cap);
    const need = 2 * streams * piece;
    std.debug.print("lean part rows: parallel {d}, prompt {d}, cap {d}, piece {d}: need {d}, have {d}\n", .{ parallel, prompt_tokens, cap, piece, need, have });
    try std.testing.expect(need <= have);
    return piece;
}

test "Flash Next MTP lean prefill: part fits long prompts at one stream" {
    for ([_]usize{ 1367, 3461, 14327 }) |prompt| _ = try expectLeanPartFits(1, 8192, prompt);
}

test "Flash Next MTP lean prefill: part fits long prompts at five streams" {
    for ([_]usize{ 909, 1367, 3461, 14327 }) |prompt| _ = try expectLeanPartFits(5, 3072, prompt);
}

test "Flash Next MTP lean prefill: sufficient capacity keeps a piece wider than 256 rows" {
    const piece = try expectLeanPartFits(1, 8192, 3461);
    try std.testing.expectEqual(@as(usize, 2048), piece);
    try std.testing.expect(piece > plan.piece_rows);
}

test "Flash Next part rows: a long cache's part is still two cap-stride slices" {
    try std.testing.expectEqual(@as(usize, 2 * 32768), sizes.partRows(32768));
    try std.testing.expectEqual(@as(usize, 2 * 131072), sizes.partRows(131072));
    // and the head's term does not grow it: the piece stops at piece_rows
    try std.testing.expectEqual(@as(usize, 2 * 262144), sizes.partRows(262144));
}
