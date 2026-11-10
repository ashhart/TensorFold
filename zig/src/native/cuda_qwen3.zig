//! Qwen3 dense embedding checkpoints on the native CUDA server: the flags and settings its family does not take, then its open.
const std = @import("std");
const api = @import("engine_api");
const qwen3 = @import("qwen3_cuda");
const refusals = @import("cuda_refusal.zig");
const device = @import("cuda_device.zig");

/// The engine for a `qwen3` checkpoint, or null with `problem` set; refusals come before any checkpoint or device access.
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8, comptime budget: type, comptime windows: type) !?api.Opened {
    if (refusals.family(qwen3.native, o)) |why| {
        problem.* = why;
        return null;
    }
    if (o.precision != null) {
        problem.* = "--precision: this CUDA family reads no NVFP4 checkpoint; serve it without --precision";
        return null;
    }
    if (device.getenv("TENSORFOLD_CUDA_KERNELS")) |v| if (v.len > 0 and !std.mem.eql(u8, v, "native")) {
        problem.* = "Qwen3 uses embedded native images: unset TENSORFOLD_CUDA_KERNELS or set native";
        return null;
    };
    const ordinal = try device.setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    return qwen3.native.open(a, gpa, io, o, ordinal, problem, budget, windows, @import("sys").optional_libraries);
}
