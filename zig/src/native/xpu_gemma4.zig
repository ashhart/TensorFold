//! The Gemma 4 E4B entry of the Intel GPU lane host: its family table row, loader and prompt window.
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const gemma = @import("gemma4_xpu");
const Allocator = std.mem.Allocator;

pub const model_type = "gemma4";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const name = "xpu-gemma4";
pub const Engine = gemma.Engine;

/// Rows of one prompt window handed to the engine (it cuts them in forwards of 16 rows).
const prompt_rows: u32 = 512;

pub fn promptRows(_: *const Engine) u32 {
    return prompt_rows;
}

pub fn note(_: Allocator, _: *const Engine) ![]const u8 {
    return ", Gemma 4 E4B (MLX 4/64), top-64 on the card";
}

/// The engine for `o.dir`, or null with `problem` set (the caller releases the device).
pub fn init(a: Allocator, _: std.Io, _: *const xpu.Context, o: api.Open, window: usize, problem: *[]const u8) !?*Engine {
    return Engine.init(o.dir, @intCast(window)) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the Intel GPU engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
}
