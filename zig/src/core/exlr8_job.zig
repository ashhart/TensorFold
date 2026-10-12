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
    pad0: u32 = 0,
    pad1: u32 = 0,
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
    pad0: u32 = 0,
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
    pad: u32 = 0,
};

test "the argument structs keep the kernels' layouts, and windows find their capacity" {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Job));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Rows));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Rotate));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Combine));
    try std.testing.expectEqual(@as(usize, 0), capacity(1));
    try std.testing.expectEqual(@as(usize, 2), capacity(3));
    try std.testing.expectEqual(@as(usize, 4), capacity(9));
}
