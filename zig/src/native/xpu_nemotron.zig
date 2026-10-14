//! The Nemotron-H entry of the Intel GPU lane host: its family table row, loader and prompt window.
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const nemotron = @import("nemotron_xpu");
const Allocator = std.mem.Allocator;

pub const model_type = "nemotron_h";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const name = "xpu-nemotron";
pub const Engine = nemotron.Engine;

/// Rows of one prompt window (above 16: the prefill GEMMs).
const prompt_rows: u32 = 512;

pub fn promptRows(_: *const Engine) u32 {
    return prompt_rows;
}

/// What the startup line adds about the engine's settings.
pub fn note(_: Allocator, e: *const Engine) ![]const u8 {
    return if (e.head != null) ", MTP drafts on" else ", no draft head";
}

/// The MTP head loads when the checkpoint has mtp-4bit.safetensors, unless --no-drafts or TENSORFOLD_XPU_MTP=0.
fn wantsHead(a: Allocator, dir: []const u8, drafts: bool) bool {
    if (!drafts) return false;
    if (std.c.getenv("TENSORFOLD_XPU_MTP")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return false;
    const path = std.fmt.allocPrintSentinel(a, "{s}/mtp-4bit.safetensors", .{dir}, 0) catch return false;
    return std.c.access(path, 0) == 0;
}

/// The engine for `o.dir`, or null with `problem` set (the caller releases the device).
pub fn init(a: Allocator, io: std.Io, ctx: *const xpu.Context, o: api.Open, window: usize, problem: *[]const u8) !?*Engine {
    return nemotron.Engine.init(std.heap.page_allocator, io, ctx, o.dir, .{ .context = window, .window_rows = prompt_rows, .mtp = wantsHead(a, o.dir, o.drafts) }) catch |e| {
        problem.* = if (e == error.UnsupportedFormat)
            try std.fmt.allocPrint(a, "{s}: NVFP4 checkpoints are not supported on the XPU backend yet; use the MLX 4-bit checkpoint", .{o.dir})
        else
            try std.fmt.allocPrint(a, "the Intel GPU engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
}
