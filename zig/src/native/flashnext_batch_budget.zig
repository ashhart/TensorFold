//! Flash Next reserves prompt states before session sizing; startup uses this same plan's named budgets.
const std = @import("std");
const GiB = @import("cache_budget.zig").GiB;
pub const Reservation = struct { cache: u64, reserved: u64, sessions: u64 };

pub fn reserve(cap: u64, ready: u64, margin: u64, cache: u64, explicit: bool, over: bool) Reservation {
    const reserved = if (over and explicit) 0 else cache;
    return .{ .cache = cache, .reserved = reserved, .sessions = cap -| ready -| margin -| reserved };
}

pub const Ready = struct { cache: u64, room: u64, over_cap: bool };

pub fn afterSessions(r: Reservation, cap: u64, allocated: u64, margin: u64) error{OverMemoryLimit}!Ready {
    const room = cap -| allocated -| margin;
    if (allocated > cap or margin > cap - allocated or r.reserved > room) return error.OverMemoryLimit;
    return .{ .cache = r.cache, .room = room, .over_cap = r.cache > room };
}

pub fn startup(w: *std.Io.Writer, slots: u32, tokens: []const u8, r: Reservation, serial: bool, lone: bool) !void {
    try w.print("Flash Next: {d} session slots ({s} tokens), session budget {d:.1} GiB; prompt cache {d:.1} GiB, shared target forwards, {s}{s}", .{ slots, tokens, @as(f64, @floatFromInt(r.sessions)) / GiB, @as(f64, @floatFromInt(r.cache)) / GiB, if (serial) "serial MTP drafts" else "every stream's drafts in one command buffer", if (lone) "; a lone request on the engine's own rounds" else "" });
}
