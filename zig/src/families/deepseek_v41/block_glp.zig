//! Additive GLPv3 paired fold/writer calls; original site and format: Matt Suiche/weightless.
const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const glp_plan = @import("glp_plan.zig");
const moe = @import("block_moe.zig");
const E = block.Emitter;
const Arg = calls.Arg;
const ns = "tf_dsv41_glp_add_v1.";

pub fn steers(e: *const E, layer: u32) bool {
    const p = e.o.glp orelse return false;
    return p.has(layer);
}

/// The paired site needs every backbone block; replay, TBO and traced unsteered folds are refused.
pub fn check(e: *const E, layers: []const u32, whole: bool) !void {
    if (e.o.glp == null) return;
    if (e.o.trace or e.marks != null or !whole or layers.len != e.cfg.layers) return error.InvalidBlock;
    for (layers, 0..) |layer, i| if (layer != i) return error.InvalidBlock;
}

fn dirOf(e: *E, layer: u32) !Arg {
    const p = e.o.glp.?;
    const width: i64 = p.width;
    return e.view(.{ .buf = glp_plan.dirs_role }, .f32, &.{width}, &.{1}, @as(i64, p.slot(layer).?) * width * 4);
}

/// DSpark observes the edited mean at a target boundary, not the preceding unsteered collapse.
pub fn tap(e: *E, layer: u32) !?Arg {
    if (!e.o.taps) return null;
    const targets = e.cfg.dspark_targets.items();
    const slot = std.mem.indexOfScalar(u32, targets, layer) orelse return null;
    const width: i64 = e.cfg.hidden;
    return try e.view(.{ .buf = "w.taps" }, .bf16, &.{ e.n, width }, &.{ @as(i64, @intCast(targets.len)) * width, 1 }, @as(i64, @intCast(slot)) * width * 2);
}

/// The completed FFN post is already in the four-copy stream; edit it before the next Engram/site.
pub fn fold(e: *E, layer: u32, tap_to: ?Arg) !void {
    const p = e.o.glp.?;
    try e.ext(ns ++ "stream", &.{ try e.streams(), tap_to orelse try e.empty(.bf16), try dirOf(e, layer), .{ .f = p.alpha }, .{ .i = e.n }, .{ .i = p.width } });
}

/// Reduce every attention partial before adding the same direction, then enter the mHC post.
pub fn writer(e: *E, layer: u32, gathered: Arg) !void {
    const p = e.o.glp.?;
    try e.ext(ns ++ "writer", &.{ gathered, try dirOf(e, layer), .{ .f = p.alpha }, .{ .i = e.n }, .{ .i = p.width } });
}

pub fn final(e: *E) !void {
    const width: i64 = e.cfg.hidden;
    try e.ext(ns ++ "final", &.{
        try e.streams(),
        try e.buf("w.c{d}.pre", .{e.coef}, .f32, &.{ e.n, 4 }),
        try e.weight("norm", .{}, .f32, &.{width}),
        try e.buf("w.out", .{}, .bf16, &.{ e.n, width }),
        .{ .f = E.dec(e.cfg.eps) },
        .{ .i = e.n },
        .{ .i = width },
    });
}

pub fn finish(e: *E, layer: u32) !void {
    try moe.postOnly(e, null);
    try fold(e, layer, null);
    try final(e);
    try moe.head(e);
}
