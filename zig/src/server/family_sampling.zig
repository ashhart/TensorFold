//! Served sampling: the family's defaults, then generation_config.json, then the serve flags.
const std = @import("std");
const json = @import("json");

/// The serve flags' sampling values; each set one replaces what the layers below gave.
pub const Flags = struct {
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
};

/// DeepSeek-V4.x: temperature 1.0, top_p 1.0, no top_k (the V4 and V4.1 model cards' recommended sampling).
pub fn put(a: std.mem.Allocator, out: *json.Object, model_type: []const u8) !void {
    if (!std.mem.eql(u8, model_type, "deepseek_v41") and !std.mem.eql(u8, model_type, "deepseek_v4")) return;
    try out.put(a, "temperature", .{ .float = 1.0 });
    try out.put(a, "top_p", .{ .float = 1.0 });
}

/// The three layers over each other (``do_sample`` false is greedy); null when none sets a value.
pub fn resolve(a: std.mem.Allocator, model_type: []const u8, generation_config: ?[]const u8, flags: Flags) !?json.Value {
    const out = try json.newObject(a);
    try put(a, out, model_type);
    if (generation_config) |bytes| if ((try json.parse(a, bytes)) == .ok) {
        const cfg = (try json.parse(a, bytes)).ok;
        for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
        if (cfg.get("do_sample")) |d| if (d == .bool) {
            if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
        };
    };
    if (flags.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (flags.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (flags.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (flags.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return if (out.count() > 0) json.Value{ .object = out } else null;
}

test "DeepSeek-V4.x serves its cards' sampling defaults; other families keep none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "deepseek_v41", "deepseek_v4" }) |kind| {
        const out = (try resolve(a, kind, null, .{})).?;
        try std.testing.expectEqual(@as(f64, 1.0), out.get("temperature").?.float);
        try std.testing.expectEqual(@as(f64, 1.0), out.get("top_p").?.float);
        try std.testing.expect(out.get("top_k") == null);
    }
    try std.testing.expect((try resolve(a, "glm5_next", null, .{})) == null);
}

test "family sampling defaults sit under generation_config.json and the serve flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try resolve(a, "deepseek_v41", "{\"temperature\": 0.6, \"top_k\": 40}", .{});
    try std.testing.expectEqual(@as(f64, 0.6), cfg.?.get("temperature").?.float);
    try std.testing.expectEqual(@as(?i64, 40), cfg.?.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 1.0), cfg.?.get("top_p").?.float);
    const flag = try resolve(a, "deepseek_v41", "{\"temperature\": 0.6}", .{ .temperature = 0.3, .top_p = 0.9 });
    try std.testing.expectEqual(@as(f64, 0.3), flag.?.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0.9), flag.?.get("top_p").?.float);
    const greedy = try resolve(a, "deepseek_v41", "{\"do_sample\": false}", .{});
    try std.testing.expectEqual(@as(f64, 0), greedy.?.get("temperature").?.float);
}
