//! Limits-only execution never constructs numerical fixtures or opens a command queue.
const std = @import("std");
const mtl = @import("metal");

pub fn gate(a: std.mem.Allocator, io: std.Io, device: mtl.Device, offline: ?[]const u8, only: ?[]const u8) !usize {
    var problems = try @import("metal_limits.zig").gate(a, io, device, offline, only);
    problems += try @import("metal_gemma4.zig").limits(a, device, only);
    problems += try @import("metal_gemma4_embed.zig").limits(a, device, only);
    problems += try @import("metal_gemma4_mixed.zig").limits(a, device, only);
    problems += try @import("metal_llama_all.zig").limits(a, device, device.tensorUnits(), only);
    return problems;
}

pub fn run(a: std.mem.Allocator, io: std.Io, device: mtl.Device, enabled: bool, offline: ?[]const u8, only: ?[]const u8) !u8 {
    std.debug.print("tf-kernel-contract limits only on {s} (applegpu_g{d}, tensor units {})\n", .{ device.name(), device.generation(), device.tensorUnits() });
    const start = std.Io.Clock.awake.now(io);
    const problems = if (enabled) try gate(a, io, device, offline, only) else 0;
    const end = std.Io.Clock.awake.now(io);
    const ms = @divTrunc(start.durationTo(end).toNanoseconds(), std.time.ns_per_ms);
    std.debug.print("SUMMARY 0 kernels: 0 pass gates 1-4, 0 fail, 0 known findings; 0 controls: 0 red as planted, 0 missed; {d} limit problems; gates 0 ms, limits {d} ms\n", .{ problems, ms });
    return if (problems == 0) 0 else 1;
}
