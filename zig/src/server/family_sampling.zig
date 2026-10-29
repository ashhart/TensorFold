//! A family's own default sampling, under generation_config.json and the serve flags.
const std = @import("std");
const json = @import("json");

/// DeepSeek-V4.x: its reference server's temperature 1.0, top_k 20 and top_p 0.95 (no generation_config).
pub fn put(a: std.mem.Allocator, out: *json.Object, model_type: []const u8) !void {
    if (!std.mem.eql(u8, model_type, "deepseek_v41") and !std.mem.eql(u8, model_type, "deepseek_v4")) return;
    try out.put(a, "temperature", .{ .float = 1.0 });
    try out.put(a, "top_k", try json.intValue(a, 20));
    try out.put(a, "top_p", .{ .float = 0.95 });
}

test "DeepSeek-V4.x serves its reference sampling defaults; other families keep none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "deepseek_v41", "deepseek_v4" }) |kind| {
        const out = try json.newObject(a);
        try put(a, out, kind);
        try std.testing.expectEqual(@as(f64, 1.0), out.get("temperature").?.float);
        try std.testing.expectEqual(@as(?i64, 20), out.get("top_k").?.int64());
        try std.testing.expectEqual(@as(f64, 0.95), out.get("top_p").?.float);
    }
    const other = try json.newObject(a);
    try put(a, other, "glm5_next");
    try std.testing.expectEqual(@as(usize, 0), other.count());
}
