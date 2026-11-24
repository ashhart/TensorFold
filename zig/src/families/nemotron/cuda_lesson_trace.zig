//! Read the bound CUDA sequence's lesson buffers and optional developer trace, and apply a lesson's adapter.
const std = @import("std");
const Engine = @import("cuda_engine.zig").Engine;
const weights = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const train_ops = @import("cuda_train_ops.zig");
const lesson_route = @import("living").lesson_route;

/// Any block's change, for the buffers every block shares (null: no lesson is attached).
pub fn lessons(e: *const Engine) ?weights.Adapter {
    return adapterOf(&e.w);
}

/// The first block's change in `w` (null: no lesson is attached, by Sliding Weights or saved lessons).
pub fn adapterOf(w: *const weights.Weights) ?weights.Adapter {
    for (w.blocks) |blk| switch (blk.kind) {
        .mamba => if (blk.mamba.adapter) |ad| return ad,
        .attention => if (blk.attn.adapter) |ad| return ad,
        .moe => if (blk.moe.adapter) |ad| return ad,
    };
    return null;
}

/// Dev trace (TF_SLIDE_TRACE): the bound sequence's router scores at its decision row and which lessons act on it.
pub fn traceLessons(e: *const Engine) void {
    const ad = e.lessons() orelse return;
    if (e.decide == lesson_route.none) return;
    const n = lesson_route.max_blocks;
    var score: [n]f32 = undefined;
    var need: [n]f32 = undefined;
    var flags: [n]u32 = undefined;
    const o = e.ops();
    o.download(std.mem.sliceAsBytes(&score), ad.score + e.slot * n * 4) catch return;
    o.download(std.mem.sliceAsBytes(&need), ad.need) catch return;
    o.download(std.mem.sliceAsBytes(&flags), ad.latch + e.slot * n * 4) catch return;
    e.stream.synchronize() catch return;
    var line: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    for (0..ad.rank / 16) |b| w.print(" {d:.2}/{d:.2}{s}", .{ score[b], need[b], if (flags[b] != 0) "*" else "" }) catch break;
    std.log.info("slide trace: decision row {d}: scores{s} (* acting)", .{ e.decide, w.buffered() });
}

/// y += scale (x a^T) b over the blocks `route` opens on each row (x and y rows their strides apart, y bf16 or fp32).
pub fn adapt(o: kern.Ops, ad: weights.Adapter, x: u64, x_stride: usize, y: u64, y_stride: usize, y_f32: bool, rows: usize, route: *const lesson_route.Route) !void {
    const t: train_ops.Ops = .{ .k = &o.k.train, .s = o.s };
    try t.loraIn(x, x_stride, ad.a, ad.xa, rows, ad.in, ad.rank, ad.xn);
    var r = route.*;
    if (ad.whole) r.whole = 1;
    try t.gate(ad.xa, ad.xn, ad.tau, ad.gates, rows, ad.rank, ad.unit, &r, ad.latch, ad.score);
    try t.loraOut(ad.xa, ad.b, y, y_stride, y_f32, rows, ad.out, ad.rank, ad.scale);
}
