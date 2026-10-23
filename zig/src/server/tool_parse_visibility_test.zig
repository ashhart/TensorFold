//! Rejected tool envelopes stay visible without consuming the accepted-call limit.
const std = @import("std");
const json = @import("json");
const parse = @import("tool_parse.zig").parse;

test "tool visibility: a first unoffered call remains content in single-call mode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const forms = [_][]const u8{
        "<tool_call><function=tools.session_move><parameter=folder>project</parameter></function></tool_call>",
        "<tool_call>{\"name\":\"tools.session_move\",\"arguments\":{}}</tool_call>",
        "<tool_call>not a tool payload</tool_call>",
    };
    for (forms) |text| for ([_]?usize{ null, 1 }) |limit| {
        const result = try parse(a, text, tools, limit);
        try std.testing.expect(result.calls == null);
        try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
        try std.testing.expectEqualStrings(text, result.content);
    };
}

test "tool visibility: rejected text does not consume a valid call and the second accepted call stays limited" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const bad = "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>";
    const good = "<tool_call>{\"name\":\"execute\",\"arguments\":{}}</tool_call>";
    const result = try parse(a, bad ++ good ++ good, tools, 1);
    try std.testing.expectEqualStrings(bad, result.content);
    try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
    try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
    try std.testing.expectEqualStrings("execute", result.calls.?[0].get("function").?.get("name").?.string);
}
