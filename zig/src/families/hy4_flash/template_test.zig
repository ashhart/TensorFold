//! The checkpoint's chat template through the engine's template core, against the reference's rendered prompts.
const std = @import("std");
const template = @import("template");

const source = @embedFile("fixtures/chat_template.jinja");
const cases = @embedFile("fixtures/prompt_cases.json");
const t = std.testing;

test "the template core renders each reference prompt, thinking switch included" {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, cases, .{});
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 5), parsed.value.array.items.len);
    for (parsed.value.array.items) |case| {
        const o = case.object;
        var diag: template.Diag = .{};
        defer if (diag.msg.len > 0) t.allocator.free(diag.msg);
        const got = template.renderChat(t.allocator, source, o.get("messages").?, .null, o.get("context").?, true, &diag) catch |err| {
            std.debug.print("{s}: {s}\n", .{ o.get("name").?.string, diag.msg });
            return err;
        };
        defer t.allocator.free(got);
        try t.expectEqualStrings(o.get("text").?.string, got);
    }
}

test "the default reasoning effort is high and no_think is spelled in the system turn" {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, cases, .{});
    defer parsed.deinit();
    const items = parsed.value.array.items;
    try t.expect(std.mem.indexOf(u8, items[0].object.get("text").?.string, "reasoning_effort:high") != null);
    try t.expect(std.mem.indexOf(u8, items[1].object.get("text").?.string, "reasoning_effort:no_think") != null);
}
