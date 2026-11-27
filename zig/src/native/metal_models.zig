//! Model-type tests the Metal loader shares: Gemma 4 and the host's memory counts.
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
