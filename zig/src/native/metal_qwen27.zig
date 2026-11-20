//! The Qwen3.8-27B-class start on Macs: which checkpoints it owns, the context it serves, and serve's startup charge.
const std = @import("std");
const tf = @import("tensorfold");
const Allocator = std.mem.Allocator;

/// Whether `dir` opens on the Qwen3.8-27B-class engine: the 27B-class qwen3_5 checkpoints, and Qwen3.5-MoE.
pub fn ownsQwen27(a: Allocator, io: std.Io, t: []const u8, dir: []const u8) bool {
    return (std.mem.eql(u8, t, "qwen3_5") and @import("qwen27_engine.zig").owns(a, io, dir)) or std.mem.eql(u8, t, "qwen3_5_moe");
}

/// The context a Qwen3.8-27B-class start serves (--context, else 32,768); null with `problem` set past 1 to 262,144.
pub fn qwen27Context(a: Allocator, context: ?i64, problem: *[]const u8) !?u32 {
    const n = context orelse 32768;
    if (n <= 0 or n > 262144) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside this Qwen checkpoint's 1 to 262,144 tokens", .{n});
        return null;
    }
    return @intCast(n);
}

pub const Qwen27Budget = @import("qwen27_reserve.zig").Budget;
/// What a Qwen3.8-27B-class start keeps free of its working set.
pub const qwen27_margin = tf.qwen27.prompt_memory.margin;

/// `serve`'s startup charge for a Qwen3.8-27B-class `dir` (no drafter or images); null for others or a refused context.
pub fn qwen27Budget(a: Allocator, io: std.Io, t: []const u8, dir: []const u8, weights: u64, context: ?i64, streams: u32, fixed: bool, problem: *[]const u8) !?Qwen27Budget {
    if (!ownsQwen27(a, io, t, dir)) return null;
    const n = try qwen27Context(a, context, problem) orelse return null;
    return try @import("qwen27_reserve.zig").budget(a, io, dir, weights, n, streams, fixed);
}
