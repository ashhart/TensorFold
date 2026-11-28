//! The Qwen3.8-27B-class Metal startup budget, using the shared current ownership and context checks.
const std = @import("std");
const tf = @import("tensorfold");
const q27_eng = @import("qwen27_engine.zig");
const Allocator = std.mem.Allocator;

pub const Qwen27Budget = @import("qwen27_reserve.zig").Budget;
/// What a Qwen3.8-27B-class start keeps free of its working set.
pub const qwen27_margin = tf.qwen27.prompt_memory.margin;

/// `serve`'s startup charge for a Qwen3.8-27B-class `dir` (no drafter or images); null for others or a refused context.
pub fn qwen27Budget(a: Allocator, io: std.Io, t: []const u8, dir: []const u8, weights: u64, context: ?i64, streams: u32, fixed: bool, problem: *[]const u8) !?Qwen27Budget {
    if (!q27_eng.ownsQwen27(a, io, t, dir)) return null;
    const n = try q27_eng.qwen27Context(a, context, problem) orelse return null;
    return try @import("qwen27_reserve.zig").budget(a, io, dir, weights, n, streams, fixed);
}
