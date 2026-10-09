//! The Qwen3.8-27B entry of the Intel GPU lane host: MLX 4-bit and EXL3 directories, KV format from the environment.
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const qwen = @import("qwen_xpu");
const Allocator = std.mem.Allocator;

pub const model_type = "qwen3_5";
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "exl3" };
pub const name = "xpu-qwen";
pub const Engine = qwen.Engine;

/// Env TENSORFOLD_XPU_KV picks the KV cache format; q8 holds twice the context of bf16 in the same memory.
const kv_env = "TENSORFOLD_XPU_KV";

fn kvMode(problem: *[]const u8, a: Allocator) !?qwen.attn_long.Mode {
    const v = std.c.getenv(kv_env) orelse return .q8;
    return std.meta.stringToEnum(qwen.attn_long.Mode, std.mem.span(v)) orelse {
        problem.* = try std.fmt.allocPrint(a, "{s}={s} is not a KV cache format (bf16, q8 or q4)", .{ kv_env, std.mem.span(v) });
        return null;
    };
}

/// Rows of one prompt window: chosen from the memory left when the engine loaded.
pub fn promptRows(_: *const Engine) u32 {
    return qwen.win.chosen_rows;
}

pub fn note(a: Allocator, _: *const Engine) ![]const u8 {
    return std.fmt.allocPrint(a, ", KV {s}", .{@tagName(qwen.attn_long.kvMode())});
}

/// Qwen3.8-27B only: the kernels are sized for its configuration, so another Qwen3.5 checkpoint is refused.
fn checkSize(a: Allocator, o: api.Open, problem: *[]const u8) !bool {
    const text = xpu.loader.readFile(a, o.dir, "config.json") catch return true;
    const cfg = (qwen.config.parse(a, text) catch return true).value;
    cfg.validate() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "{s}: the Intel GPU engine reads the Qwen3.8-27B configuration only ({s})", .{ o.dir, @errorName(e) });
        return false;
    };
    return true;
}

/// The engine for `o.dir`, or null with `problem` set (the caller releases the device).
pub fn init(a: Allocator, io: std.Io, ctx: *const xpu.Context, o: api.Open, window: usize, problem: *[]const u8) !?*Engine {
    if (!try checkSize(a, o, problem)) return null;
    const kv = (try kvMode(problem, a)) orelse return null;
    return qwen.Engine.init(std.heap.page_allocator, io, ctx, o.dir, .{ .context = window, .kv = kv, .window_rows = qwen.engine.auto_window }) catch |e| {
        problem.* = if (e == error.ContextTooLarge)
            try std.fmt.allocPrint(a, "--context {d} is more than the Intel GPU engine holds for Qwen3.8 ({d} positions)", .{ window, qwen.attn_long.max_ctx })
        else
            try std.fmt.allocPrint(a, "the Intel GPU engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
}
