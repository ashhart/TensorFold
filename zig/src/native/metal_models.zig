//! Model-type tests the Metal loader shares: Gemma 4, the 27B-class engine, and the host's memory counts.
const std = @import("std");
const api = @import("engine_api");
const cache_fit = @import("cache_fit.zig");

/// Gemma 4 31B (gemma4) and 12B (gemma4_unified, its text model): one native family, gemma4.zig's host.
pub fn isGemma4(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "gemma4") or std.mem.eql(u8, model_type, "gemma4_unified");
}

/// The host's memory counts: `cache_fit.hostMemory` from the process footprint (no configured load limit).
pub fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    return cache_fit.hostMemory(0, reset_peak);
}

/// Whether `dir` opens on the Qwen3.8-27B-class engine: the 27B-class qwen3_5 checkpoints, and Qwen3.5-MoE.
pub fn ownsQwen27(a: std.mem.Allocator, io: std.Io, t: []const u8, dir: []const u8) bool {
    const tied = @import("sys").environment.read(bool, "TF_QWEN27_TIED_HEAD", false) catch false;
    return (std.mem.eql(u8, t, "qwen3_5") and @import("qwen27_engine.zig").owns(a, io, dir, tied)) or std.mem.eql(u8, t, "qwen3_5_moe");
}
