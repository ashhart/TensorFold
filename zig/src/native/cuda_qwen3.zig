//! Qwen3 dense embedding and reranker checkpoints on the native CUDA server: the family's flag refusals, then its open.
const std = @import("std");
const api = @import("engine_api");
const qwen3 = @import("qwen3_cuda");
const refusals = @import("cuda_refusal.zig");
const device = @import("cuda_device.zig");
const environment = @import("sys").environment;

const kernels_setting = "TENSORFOLD_CUDA_KERNELS";

/// The refusal for a captured kernel set (`raw` is the setting): the family runs on the engine's embedded images only.
pub fn kernelsRefusal(a: std.mem.Allocator, raw: ?[]const u8) !?[]const u8 {
    const set = environment.path(kernels_setting, raw) catch return try environment.describe(a, kernels_setting, raw.?);
    if (set) |dir| if (!std.mem.eql(u8, dir, "native")) return "Qwen3 uses embedded native images: unset TENSORFOLD_CUDA_KERNELS or set native";
    return null;
}

/// The engine for a `qwen3` checkpoint, or null with `problem` set; refusals come before checkpoint or device access.
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8, comptime budget: type, comptime windows: type) !?api.Opened {
    if (refusals.family(qwen3.native, o)) |why| {
        problem.* = why;
        return null;
    }
    if (o.precision != null) {
        problem.* = "--precision: this CUDA family reads no NVFP4 checkpoint; serve it without --precision";
        return null;
    }
    if (try kernelsRefusal(a, environment.Process.get(.{}, kernels_setting))) |why| {
        problem.* = why;
        return null;
    }
    const ordinal = try device.setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    return qwen3.native.open(a, gpa, io, o, ordinal, problem, budget, windows, @import("sys").optional_libraries);
}
