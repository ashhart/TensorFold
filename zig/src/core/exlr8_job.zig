//! The EXLR8 projection kernels' arguments: one layout for the Metal and CUDA kernels.

const std = @import("std");

pub const max_rows = 16;
/// "No table entry" in a job: no input scales, output scales, signs or block scale.
pub const none: u32 = 0xFFFFFFFF;
/// Row capacities of the projection kernels; a window runs on the smallest that holds it.
pub const capacities = [_]usize{ 1, 2, 4, 8, 16 };

/// The capacity kernel for `rows` (1-16) by its index in `capacities`.
pub fn capacity(rows: usize) usize {
    for (capacities, 0..) |c, i| if (rows <= c) return i;
    unreachable;
}

/// (row capacity, sets) of the split kernels: a job's blocks round-robin over S sets of 8 simdgroups, the same bits.
pub const splits = [_][2]usize{ .{ 1, 2 }, .{ 1, 4 }, .{ 2, 2 }, .{ 2, 4 }, .{ 4, 2 } };

/// Sets for `count` jobs of `blocks` blocks at `rows` rows: few long jobs leave cores idle, so their blocks split.
pub fn sets(count: usize, blocks: usize, rows: usize) usize {
    if (count == 0 or blocks < 32 or rows > 4) return 1;
    if (count <= 24 and rows <= 2) return 4;
    return if (count <= 160) 2 else 1;
}

/// The split kernel for `rows` (1-16) over `s` sets by its index in `splits`; null runs the one-set kernel.
pub fn split(rows: usize, s: usize) ?usize {
    if (s < 2) return null;
    const cap = capacities[capacity(rows)];
    for (splits, 0..) |k, i| if (k[0] == cap and k[1] == s) return i;
    return null;
}

/// 128 output columns over `blocks` input blocks of 128; block b's 64 tiles start at byte bases[base + b].
pub const Job = extern struct {
    k: u32,
    blocks: u32,
    base: u32,
    in_scale: u32 = none,
    out_scale: u32 = none,
    out_sign: u32 = none,
    out_block: u32 = none,
    x_col: u32 = 0,
    y_col: u32 = 0,
    raw: u32 = 0,
    /// With `count` members, row i reads input row members[members + i] >> 16 and writes row (that & 0xFFFF).
    members: u32 = 0,
    count: u32 = 0,
};
pub const Rows = extern struct { rows: u32, x_stride: u32, y_stride: u32, y_f32: u32 };
/// r = (x * signs * scale) H0 per 128-block; slots sit slot_cols apart; with `scales` each reads its own scale offset.
pub const Rotate = extern struct {
    rows: u32,
    x_stride: u32,
    x_col: u32 = 0,
    x_f32: u32 = 0,
    r_stride: u32,
    r_col: u32 = 0,
    sign: u32 = none,
    scale: u32 = none,
    slot_cols: u32 = 0,
    scales: u32 = 0,
    /// Nonzero: slot s of row r reads its scale offset at slot_scales[r row_slots + s].
    row_slots: u32 = 0,
    pad1: u32 = 0,
};
/// y = signs 2^-7 ((sum_e w_e sv_e z_e) H0) per row and 128-block; z rows hold the slots z_slot floats apart.
pub const Combine = extern struct {
    rows: u32,
    slots: u32,
    z_stride: u32,
    y_stride: u32,
    y_f32: u32 = 0,
    sign: u32 = none,
    z_slot: u32,
    /// Nonzero: slot e of row r reads its block-scale offset at sv[r slots + e].
    per_row: u32 = 0,
};

/// Dense formats by row as the kernels number them: the record's format, or 12 for int4 g32 sharded by input.
pub const dense_int4_in: u32 = 12;

/// Rows [out0, out0 + outs) of a matrix over inputs [in0, in0 + ins); w, s: byte offsets of its data and scales.
pub const DenseJob = extern struct {
    format: u32,
    out0: u32,
    outs: u32,
    in0: u32,
    ins: u32,
    out_total: u32,
    in_total: u32,
    x_col: u32 = 0,
    y_col: u32 = 0,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    w: u64,
    s: u64 = 0,
};

/// A dense dispatch's threadgroups: (job, 32-row block) pairs, every job's blocks in order.
pub fn denseGroups(gpa: std.mem.Allocator, jobs: []const DenseJob) ![][2]u32 {
    var n: usize = 0;
    for (jobs) |j| n += (j.outs + 31) / 32;
    const out = try gpa.alloc([2]u32, n);
    var at: usize = 0;
    for (jobs, 0..) |j, i| for (0..(j.outs + 31) / 32) |b| {
        out[at] = .{ @intCast(i), @intCast(b) };
        at += 1;
    };
    return out;
}

test "the argument structs keep the kernels' layouts, and windows find their capacity" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(DenseJob));
    const g = try denseGroups(std.testing.allocator, &.{ .{ .format = 3, .out0 = 0, .outs = 64, .in0 = 0, .ins = 128, .out_total = 64, .in_total = 128, .w = 0 }, .{ .format = 6, .out0 = 0, .outs = 33, .in0 = 0, .ins = 128, .out_total = 33, .in_total = 128, .w = 0 } });
    defer std.testing.allocator.free(g);
    try std.testing.expectEqual(@as(usize, 4), g.len);
    try std.testing.expectEqual([2]u32{ 1, 1 }, g[3]);
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Job));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Rows));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Rotate));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Combine));
    try std.testing.expectEqual(@as(usize, 0), capacity(1));
    try std.testing.expectEqual(@as(usize, 2), capacity(3));
    try std.testing.expectEqual(@as(usize, 4), capacity(9));
    try std.testing.expectEqual(@as(usize, 2), sets(48, 64, 1));
    try std.testing.expectEqual(@as(usize, 4), sets(24, 64, 2));
    try std.testing.expectEqual(@as(usize, 2), sets(48, 64, 3));
    try std.testing.expectEqual(@as(usize, 1), sets(432, 8, 1));
    try std.testing.expectEqual(@as(usize, 2), sets(144, 48, 1));
    try std.testing.expectEqual(@as(usize, 1), sets(288, 48, 1));
    try std.testing.expectEqual(@as(usize, 1), sets(0, 48, 1));
    try std.testing.expectEqual(@as(?usize, 4), split(3, 2));
    try std.testing.expectEqual(@as(?usize, null), split(3, 4));
    try std.testing.expectEqual(@as(?usize, null), split(1, 1));
}
