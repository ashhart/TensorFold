const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const translate = @import("responses_translate.zig");
const cli = @import("cli.zig");
const t = std.testing;
const NoHistory = struct {
    pub fn conversation(_: @This(), _: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return &.{};
    }
};
fn turn(cx: *errors.Cx, raw: []const u8, strict: bool) !translate.Request {
    return translate.translateMode(cx, (try json.parse(cx.a, raw)).ok, NoHistory{}, strict);
}
test "Responses strict flag defaults off and requires an explicit serve switch" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    try t.expect(!(try cli.parse(a, &.{"model"}, &usage)).responses_strict);
    try t.expect((try cli.parse(a, &.{ "model", "--responses-strict" }, &usage)).responses_strict);
    try t.expectError(error.Usage, cli.parse(a, &.{ "model", "--responses-strict=false" }, &usage));
    const config: @import("server.zig").Config = .{ .served_name = "m", .model_ids = &.{"m"} };
    try t.expect(!config.responses_strict);
}
test "Responses strict on: allowed order and deferred prompt visibility require opt-in" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const raw = "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"beta\",\"defer_loading\":true},{\"type\":\"function\",\"name\":\"alpha\"}]";
    const off = try turn(&cx, raw ++ "}", false);
    const on = try turn(&cx, raw ++ "}", true);
    try t.expectEqual(@as(usize, 2), off.chat.get("tools").?.array.len);
    try t.expectEqual(@as(usize, 1), on.chat.get("tools").?.array.len);
    const picked = try turn(&cx, raw ++ ",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\",\"name\":\"alpha\"},{\"type\":\"function\",\"name\":\"beta\"}]}}", true);
    const tools = picked.chat.get("tools").?.array;
    try t.expectEqualStrings("alpha", tools[0].get("function").?.strField("name").?);
    try t.expectEqualStrings("beta", tools[1].get("function").?.strField("name").?);
}
test "Responses strict on: items 3 through 6 add their refusals only with opt-in" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    for ([_][]const u8{
        "{\"input\":[{\"type\":\"function_call\",\"call_id\":\"c\",\"name\":\"f\",\"arguments\":true}]}",
        "{\"input\":\"x\",\"tool_choice\":{\"type\":\"allowed_tools\"}}",
        "{\"input\":\"x\",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":false}}",
        "{\"input\":\"x\",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\",\"name\":\"missing\"}]}}",
        "{\"input\":\"x\",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\"}]}}",
        "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"defer_loading\":1}]}",
    }) |raw| {
        _ = try turn(&cx, raw, false);
        try t.expectError(error.Refused, turn(&cx, raw, true));
    }
}
test "Responses strict off: duplicate declared tools remain in their original allowed order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const raw = "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"f\"},{\"type\":\"function\",\"name\":\"f\"}],\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\",\"name\":\"f\"}]}}";
    try t.expectEqual(@as(usize, 2), (try turn(&cx, raw, false)).chat.get("tools").?.array.len);
    try t.expectEqual(@as(usize, 1), (try turn(&cx, raw, true)).chat.get("tools").?.array.len);
}
test "Responses strict off: trailing and interrupted reasoning retain main translation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const prefix = "{\"input\":[{\"type\":\"reasoning\",\"content\":[{\"type\":\"reasoning_text\",\"text\":\"thought\"}]}";
    const tail = try turn(&cx, prefix ++ "]}", false);
    try t.expectEqual(@as(usize, 0), tail.added.len);
    const interrupted = try turn(&cx, prefix ++ ",{\"role\":\"user\",\"content\":\"next\"},{\"role\":\"assistant\",\"content\":\"answer\"}]}", false);
    try t.expectEqual(@as(usize, 2), interrupted.added.len);
    try t.expectEqualStrings("thought", interrupted.added[1].strField("reasoning_content").?);
    const strict_tail = try turn(&cx, prefix ++ "]}", true);
    try t.expectEqual(@as(usize, 1), strict_tail.added.len);
}
test "Responses strict off: per-request custom tools preserve exact replay input" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const request = try turn(&cx, "{\"input\":[{\"type\":\"custom_tool_call\",\"call_id\":\"c\",\"name\":\"f\",\"input\":\" raw {\"}],\"tools\":[{\"type\":\"custom\",\"name\":\"f\"}]}", false);
    const call = request.added[0].get("tool_calls").?.array[0];
    try t.expectEqualStrings(" raw {", call.get("custom").?.get("input").?.string);
}
