//! The native prompt-cache option bounds retained payload; the KV pool and live buffers have separate admission.
const std = @import("std");

pub const default_bytes: u64 = 16 << 30;

pub fn bytes(gib: ?f64) !u64 {
    const g = gib orelse return default_bytes;
    const b = g * (1 << 30);
    if (!std.math.isFinite(b) or b < 0 or b >= 9223372036854775808.0) return error.BadCacheBudget;
    return @intFromFloat(b);
}

pub fn options(o: @import("sessions_gpu.zig").Options, cap: u64) @import("sessions_gpu.zig").Options {
    var out = o;
    out.cache_bytes = cap;
    if (cap == 0) out.enabled = false;
    return out;
}
