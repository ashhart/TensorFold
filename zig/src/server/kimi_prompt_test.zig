//! Native XTML segments against the release-generated fixtures, without importing or executing publisher Python.
const std = @import("std");
const json = @import("json");
const kimi = @import("kimi_prompt.zig");

test "Kimi native prompts equal all fifteen release goldens segment by segment" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = (try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok;
    const cases = fixture.get("cases").?.array;
    try std.testing.expectEqual(@as(usize, 15), cases.len);
    for (cases) |c| {
        const kw = c.get("kwargs").?;
        const out = try kimi.render(a, c.get("messages").?, .{
            .thinking = if (kw.get("thinking")) |v| v.bool else true,
            .add_generation_prompt = if (kw.get("add_generation_prompt")) |v| v.bool else true,
            .thinking_effort = kw.strField("thinking_effort"),
            .tools = kw.get("tools") orelse .null,
            .tool_choice = kw.get("tool_choice") orelse .null,
            .response_format = kw.get("response_format") orelse .null,
        });
        const expected = c.get("segments").?.array;
        try std.testing.expectEqual(expected.len, out.len);
        var joined: std.ArrayList(u8) = .empty;
        for (expected, out) |want, got| {
            try std.testing.expectEqualStrings(want.strField("text").?, got.text);
            try std.testing.expectEqual(want.get("allow_special").?.bool, got.allow_special);
            try std.testing.expect(!got.request or !got.allow_special);
            try joined.appendSlice(a, got.text);
        }
        try std.testing.expectEqualStrings(c.strField("joined").?, joined.items);
    }
}

test "Kimi malformed content and unsupported thinking effort are named refusals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"x\"}]")).ok;
    try std.testing.expectError(error.BadKimiEffort, kimi.render(a, messages, .{ .thinking_effort = "xhigh" }));
    const vision = (try json.parse(a, "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":\"unused\"}]}]")).ok;
    try std.testing.expectError(error.KimiVisionUnsupported, kimi.render(a, vision, .{}));
    const role = (try json.parse(a, "[{\"role\":\"unknown\",\"content\":\"x\"}]")).ok;
    try std.testing.expectError(error.BadKimiRole, kimi.render(a, role, .{}));
}
