//! DeepSeek-V4.1's own Chat Completions rules (template.zig's `recipe`) where the recipe fixture has no case.
const std = @import("std");
const json = @import("json");
const template = @import("template.zig");
const t = std.testing;

fn render(a: std.mem.Allocator, messages: []const u8, tools: []const u8, o: template.Options) ![]u8 {
    var opts = o;
    if (tools.len > 0) opts.tools = (try json.parseText(a, tools)).ok.array;
    var problem: template.Problem = .{};
    return template.render(a, (try json.parseText(a, messages)).ok, opts, &problem);
}

test "V4.1 keeps the first of consecutive reminders and closes an assistant turn only for a generation prompt" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reminders = try render(a, "[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"latest_reminder\",\"content\":\"first\"},{\"role\":\"latest_reminder\",\"content\":\"second\"}]", "", .{ .thinking = false, .recipe = true });
    try t.expect(std.mem.endsWith(u8, reminders, template.USER ++ "a" ++ template.REMINDER ++ "first" ++ template.ASSISTANT ++ template.THINK_END));
    const turn = "[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}]";
    const closed = template.ASSISTANT ++ template.THINK_END ++ "b" ++ template.EOS;
    try t.expect(std.mem.endsWith(u8, try render(a, turn, "", .{ .recipe = true }), closed ++ template.ASSISTANT ++ template.THINK_OPEN));
    try t.expect(std.mem.endsWith(u8, try render(a, turn, "", .{ .thinking = false, .recipe = true }), closed ++ template.ASSISTANT ++ template.THINK_END));
    try t.expect(std.mem.endsWith(u8, try render(a, turn, "", .{ .recipe = true, .add_generation_prompt = false }), closed));
    try t.expect(std.mem.endsWith(u8, try render(a, turn, "", .{}), "b" ++ template.EOS)); // V4: complete, no prompt
}

test "V4.1 drops a namespace from schema lines and calls, fills a missing description and keeps blank arguments" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = "[{\"type\":\"function\",\"namespace\":{\"name\":\"git\",\"description\":\"Git.\"},\"function\":{\"name\":\"commit\",\"strict\":true}}]";
    const messages = "[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c\",\"type\":\"function\",\"function\":{\"name\":\"commit\",\"namespace\":\"git\",\"arguments\":\"\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"c\",\"content\":\"ok\"}]";
    const v41 = try render(a, messages, tools, .{ .thinking = false, .recipe = true });
    try t.expect(std.mem.indexOf(u8, v41, "{\"name\": \"commit\", \"description\": \"\", \"parameters\": null}\n\n") != null);
    try t.expect(std.mem.indexOf(u8, v41, "invoke name=\"commit\">\n<" ++ template.DSML ++ " parameter name=\"arguments\" string=\"true\"></") != null);
    const v4 = try render(a, messages, tools, .{ .thinking = false });
    try t.expect(std.mem.indexOf(u8, v4, "{\"name\": \"git::commit\", \"strict\": true, \"description\": \"Git.\\n\"}") != null);
    try t.expect(std.mem.indexOf(u8, v4, "invoke name=\"git::commit\">\n\n</") != null);
}
