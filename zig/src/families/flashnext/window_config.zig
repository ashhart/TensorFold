//! One load owns its capacity and rotary policy; planning a later load changes no live engine.
const std = @import("std");
pub const trained: usize = @import("flashnext_memory").trained;
pub const maximum: usize = 4 * trained + 64;
pub const Settings = struct {
    capacity: usize = trained,
    factor: f64 = 0,
    startup_extra: u64 = 0,
    pub fn validate(s: Settings, paired: bool) !void {
        if (s.capacity < trained or s.capacity > maximum or s.capacity % 4 != 0) return error.ContextTooLong;
        if (s.factor != (if (s.capacity > trained) @as(f64, 4) else 0)) return error.BadRotaryPolicy;
        if (paired and s.factor > 0) return error.LongWindowPairUnsupported;
    }
};
pub const Choice = struct { settings: Settings = .{}, refusal: ?[]const u8 = null };
pub fn plan(window: i64, native: i64, margin: i64, paired: bool) !Choice {
    if (window < 0 or margin < 0) return error.ContextTooLong;
    if (native <= 0 or window <= native) return .{};
    if (paired) return .{ .refusal = "--context past the trained window is served on one Mac; the --speed-up pair reads up to it" };
    const n = std.math.cast(usize, window) orelse return error.ContextTooLong;
    const extra = std.math.cast(usize, margin) orelse return error.ContextTooLong;
    const requested = std.math.add(usize, n, extra) catch return error.ContextTooLong;
    if (requested > maximum) return error.ContextTooLong;
    const cap = std.mem.alignForward(usize, @max(requested, trained), 4);
    const settings: Settings = .{ .capacity = cap, .factor = if (cap > trained) 4 else 0 };
    try settings.validate(false);
    return .{ .settings = settings };
}

pub fn request(window: i64, native: i64, margin: i64, paired: bool, why: *[]const u8) !Settings {
    const choice = try plan(window, native, margin, paired);
    if (choice.refusal) |text| {
        why.* = text;
        return error.LongWindowPairUnsupported;
    }
    return choice.settings;
}
