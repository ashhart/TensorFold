//! Read the bound CUDA sequence's lesson buffers and optional developer trace.
const std = @import("std");
const Engine = @import("cuda_engine.zig").Engine;
const weights = @import("cuda_weights.zig");
const lesson_route = @import("living").lesson_route;

/// Any block's change, for the buffers every block shares (null: no lesson is attached).
pub fn lessons(e: *const Engine) ?weights.Adapter {
    for (e.w.blocks) |blk| switch (blk.kind) {
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
