//! A reply that max_tokens cuts inside a tool call: the whole reply gives what the stream gave (#1238).
const std = @import("std");
const pre = @import("stream_preflight_test.zig");
const TestText = pre.TestText;
const CallEngine = pre.CallEngine;
const sendWith = pre.sendWith;

const head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}]";
const opened = "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\nThe history of";
const rest = " the bicycle\n</parameter>\n</function>\n</tool_call>";

/// ``cut`` then ``rest``, max_tokens at ``cut``'s end: whole and streamed replies agree (null ``content``: unchecked).
fn expectCut(cut: []const u8, calls: usize, arguments: []const u8, content: ?[]const u8, extra: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try TestText.encode(undefined, a, try std.mem.concat(a, u8, &.{ cut, rest }), false);
    const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}{s}", .{ head, cut.len, extra });
    for ([_]bool{ false, true }) |batched| {
        var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
        const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
        const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(streamed.opened and streamed.status == null);
        for ([_]@TypeOf(whole){ whole, streamed }) |seen| {
            try std.testing.expectEqual(calls, seen.calls);
            try std.testing.expectEqualStrings(arguments, seen.arguments[0..seen.arguments_len]);
            if (content) |c| try std.testing.expectEqualStrings(c, seen.content());
            if (std.mem.indexOf(u8, seen.content(), "<function=") != null) return error.CallMarkupInContent;
            try std.testing.expectEqual(.length, seen.finish.?);
        }
    }
}

test "a call max_tokens cuts is the whole reply's tool call, cut, as the stream's is" {
    const args = "{\"path\":\"notes.txt\",\"content\":\"The history of";
    try expectCut(opened, 1, args, "", "");
    try expectCut("Writing it.\n" ++ opened, 1, args, "Writing it.", "");
    // a whole call first: both calls, the second cut
    const first = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n";
    try expectCut(first ++ opened, 2, "{\"path\":\"a.txt\"}" ++ args, "", "");
    // cut between parameters, and after the function closed but before the envelope did
    try expectCut("<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n", 1, "{\"path\":\"notes.txt\"", "", "");
    try expectCut("<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n", 1, "{\"path\":\"notes.txt\"}", "", "");
}

test "the one-call policy and a cut before any call is named keep their replies" {
    // parallel_tool_calls false hides the cut call in both, as before
    try expectCut("Writing it.\n" ++ opened, 0, "", null, ",\"parallel_tool_calls\":false");
    // an envelope with no function named yet is no call in either reply
    try expectCut("<tool_call>\n", 0, "", "<tool_call>\n", "");
}
