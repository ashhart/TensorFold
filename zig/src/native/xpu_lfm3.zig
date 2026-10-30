//! The LFM3 entry of the Intel GPU lane host: its family table row, loader and prompt window.
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const lfm3 = @import("lfm3_xpu");
const Allocator = std.mem.Allocator;

pub const model_type = "lfm3";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const name = "xpu-lfm3";
pub const Engine = lfm3.engine.Engine;

/// Rows of one prompt window (pdense above one row; the scratch is sized for it): 128 beat 32 and 64 on Iris Xe.
const prompt_rows: u32 = 128;

pub fn promptRows(_: *const Engine) u32 {
    return prompt_rows;
}

/// What the startup line adds about the engine's settings.
pub fn note(a: Allocator, e: *const Engine) ![]const u8 {
    return std.fmt.allocPrint(a, ", prompt windows of {d} rows, no draft head", .{e.window_rows});
}

/// The engine for `o.dir`, or null with `problem` set (the caller releases the device).
pub fn init(a: Allocator, io: std.Io, ctx: *const xpu.Context, o: api.Open, window: usize, problem: *[]const u8) !?*Engine {
    return Engine.init(std.heap.page_allocator, io, ctx, o.dir, .{ .context = window, .window_rows = prompt_rows }) catch |e| {
        problem.* = if (e == error.UnsupportedFormat)
            try std.fmt.allocPrint(a, "{s}: the Intel GPU engine reads LFM3 text checkpoints in MLX affine 4-bit, group 64", .{o.dir})
        else
            try std.fmt.allocPrint(a, "the Intel GPU engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
}
