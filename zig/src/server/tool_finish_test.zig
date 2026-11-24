//! A reply the token cap ended keeps finish_reason length with its calls; one that ends on its own reports tool_calls.
const std = @import("std");
const pre = @import("stream_preflight_test.zig");

const head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";
const first = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"a\"}}\n</tool_call>";
const second = "\n<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"b\"}}\n</tool_call>";

/// Reply `text` under max_tokens `cap` (0: none), whole and streamed: by characters, at once, at strides 1, 2, 3, 256.
fn expectFinish(text: []const u8, cap: usize, calls: usize, finish: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try pre.TestText.encode(undefined, a, text, false);
    const limit = if (cap > 0) try std.fmt.allocPrint(a, ",\"max_tokens\":{d}", .{cap}) else "";
    const body = try std.mem.concat(a, u8, &.{ head, limit });
    for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| for ([_]bool{ false, true }) |batched| for ([_]bool{ false, true }) |stream| {
        var backend: pre.CallEngine = .{ .tokens = tokens, .batched = batched, .stride = stride };
        const seen = try pre.sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, if (stream) ",\"stream\":true}" else ",\"stream\":false}" }), .{});
        try std.testing.expectEqual(calls, seen.calls);
        try std.testing.expectEqualStrings(finish, @tagName(seen.finish.?));
    };
}

test "tool finish: a complete call when the token cap ends the reply keeps finish_reason length" {
    try expectFinish(first ++ second, first.len, 1, "length");
}

test "tool finish: a complete call then a call the cap cuts keeps finish_reason length, both returned" {
    try expectFinish(first ++ second, first.len + 53, 2, "length");
}

test "tool finish: calls in a reply that ends before the cap report tool_calls, and plain text stop" {
    try expectFinish(first ++ second, 0, 2, "tool_calls");
    try expectFinish(first ++ second, first.len + second.len + 16, 2, "tool_calls");
    try expectFinish("No tool is needed.", 0, 0, "stop");
}
