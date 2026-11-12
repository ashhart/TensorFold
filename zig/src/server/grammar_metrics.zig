//! Grammar totals accompany round counters only on replies whose lane thread measured grammar windows.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");

pub fn put(a: std.mem.Allocator, runtime: anytype, timing: @FieldType(api.Stats, "grammar")) !void {
    const t = timing orelse return;
    try runtime.put(a, "grammar_windows", try json.intValue(a, t.windows));
    try runtime.put(a, "grammar_fills", try json.intValue(a, t.fills));
    try runtime.put(a, "grammar_window_seconds", .{ .float = @as(f64, @floatFromInt(t.window_ns)) / 1e9 });
    try runtime.put(a, "grammar_fill_seconds", .{ .float = @as(f64, @floatFromInt(t.fill_ns)) / 1e9 });
}
