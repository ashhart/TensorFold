const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const translate = @import("responses_translate.zig");
const response_tools = @import("responses_tools.zig");
const tool_specs = @import("tool_specs.zig");
const t = std.testing;

const NoHistory = struct {
    pub fn conversation(_: @This(), cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no earlier turn");
    }
};

fn turn(a: std.mem.Allocator, cx: *errors.Cx, source: []const u8) !translate.Request {
    return translate.translateMode(cx, (try json.parse(a, source)).ok, NoHistory{}, true);
}

test "Responses custom tool replay preserves call ID, exact input, result and namespaces" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const request = try turn(a, &cx, "{\"input\":[{\"type\":\"custom_tool_call\",\"call_id\":\"custom-1\",\"name\":\"shared\",\"input\":\" raw\\n{\\\"incomplete\"},{\"type\":\"custom_tool_call_output\",\"call_id\":\"custom-1\",\"output\":\"kept\"}],\"tools\":[{\"type\":\"function\",\"name\":\"shared\"},{\"type\":\"custom\",\"name\":\"shared\",\"format\":{\"type\":\"text\"}}]" ++
        ",\"tool_choice\":{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"shared\"},{\"type\":\"custom\",\"name\":\"shared\"}]}}");
    const messages = request.chat.get("messages").?.array;
    const call = messages[0].get("tool_calls").?.array[0];
    try t.expectEqualStrings("custom-1", call.get("id").?.string);
    try t.expectEqualStrings("custom", call.get("type").?.string);
    try t.expectEqualStrings(" raw\n{\"incomplete", call.get("custom").?.get("input").?.string);
    try t.expectEqualStrings("custom-1", messages[1].get("tool_call_id").?.string);
    try t.expectEqualStrings("kept", messages[1].get("content").?.string);
    const tools = request.chat.get("tools").?.array;
    try t.expectEqual(@as(usize, 2), tools.len);
    try t.expectEqualStrings("function", tools[0].get("type").?.string);
    try t.expectEqualStrings("custom", tools[1].get("type").?.string);
}

test "Responses deferred tools activate only through a resolved same-namespace reference" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const base = "\"input\":\"next\",\"tools\":[{\"type\":\"function\",\"name\":\"ready\"},{\"type\":\"custom\",\"name\":\"later\",\"defer_loading\":true}]";
    const automatic = try turn(a, &cx, "{" ++ base ++ "}");
    const automatic_tools = automatic.chat.get("tools").?.array;
    try t.expectEqual(@as(usize, 1), automatic_tools.len);
    try t.expectEqualStrings("ready", automatic_tools[0].get("function").?.get("name").?.string);

    const selected = try turn(a, &cx, "{" ++ base ++ ",\"tool_choice\":{\"type\":\"custom\",\"name\":\"later\"}}");
    const selected_tools = selected.chat.get("tools").?.array;
    try t.expectEqual(@as(usize, 1), selected_tools.len);
    try t.expectEqualStrings("custom", selected.chat.get("tool_choice").?.strField("type").?);
    try t.expectEqualStrings("later", selected_tools[0].get("name").?.string);

    var invalid_cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, turn(a, &invalid_cx, "{" ++ base ++ ",\"tool_choice\":{\"type\":\"custom\",\"name\":\"missing\"}}"));
    try t.expect(std.mem.indexOf(u8, invalid_cx.message, "unknown custom tool 'missing'") != null);
}

test "Responses malformed freeform format and missing input fail explicitly" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const bad_format = (try json.parse(a, "[{\"type\":\"custom\",\"name\":\"x\",\"format\":{\"type\":\"grammar\",\"syntax\":\"lark\",\"definition\":\"\"}}]")).ok;
    try t.expectError(error.Refused, response_tools.tools(&cx, bad_format));
    const missing_input = (try json.parse(a, "[{\"type\":\"custom_tool_call\",\"call_id\":\"c\",\"name\":\"x\"}]")).ok.array;
    try t.expectError(error.Refused, translate.messages(&cx, missing_input));
}

test "Responses drops only foreign opaque compaction and refuses a missing original reference" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const items = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"before\"},{\"type\":\"compaction\",\"encrypted_content\":\"foreign opaque\"},{\"role\":\"user\",\"content\":\"after\"}]")).ok.array;
    const messages = try translate.messages(&cx, items);
    try t.expectEqual(@as(usize, 2), messages.len);
    try t.expectEqualStrings("before", messages[0].get("content").?.string);
    try t.expectEqualStrings("after", messages[1].get("content").?.string);
    const missing = (try json.parse(a, "[{\"type\":\"item_reference\",\"id\":\"fc_missing\"}]")).ok.array;
    try t.expectError(error.Refused, translate.messages(&cx, missing));
    try t.expect(std.mem.indexOf(u8, cx.message, "original response history") != null);
}

test "Responses named tool choice deactivates other names and same-name opposite namespaces" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const tools = "\"tools\":[{\"type\":\"function\",\"name\":\"shared\"},{\"type\":\"custom\",\"name\":\"shared\"},{\"type\":\"function\",\"name\":\"other\"}]";
    inline for ([_][]const u8{ "function", "custom" }) |selected_kind| {
        const body = "{\"input\":\"next\"," ++ tools ++ ",\"tool_choice\":{\"type\":\"" ++ selected_kind ++ "\",\"name\":\"shared\"}}";
        const request = try turn(a, &cx, body);
        const offered = request.chat.get("tools").?.array;
        try t.expectEqual(@as(usize, 1), offered.len);
        try t.expectEqualStrings(selected_kind, offered[0].strField("type").?);
        const active = try tool_specs.active(&cx, request.chat.get("tools"), request.chat.get("tool_choice"));
        try t.expectEqual(@as(usize, 1), active.len);
        try t.expectEqualStrings(selected_kind, active[0].strField("type").?);
        try t.expect(try tool_specs.choiceRequiresCall(a, request.chat.get("tool_choice")));
    }
}

test "Responses continuation flushes trailing reasoning without adding an empty control message" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const items = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"next\"},{\"type\":\"reasoning\",\"content\":[{\"type\":\"reasoning_text\",\"text\":\"trailing thought\"}]}]")).ok.array;
    const messages = try translate.messagesMode(&cx, items, true);
    try t.expectEqual(@as(usize, 2), messages.len);
    try t.expectEqualStrings("assistant", messages[1].strField("role").?);
    try t.expectEqualStrings("", messages[1].strField("content").?);
    try t.expectEqualStrings("trailing thought", messages[1].strField("reasoning_content").?);
    const plain = try translate.messages(&cx, items[0..1]);
    try t.expectEqual(@as(usize, 1), plain.len);
}
