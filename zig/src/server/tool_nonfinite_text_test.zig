//! NaN and Infinity inside a tool call's text values are text in every format, whole and streamed (1.3.4).
const std = @import("std");
const pre = @import("stream_preflight_test.zig");
const TestText = pre.TestText;
const CallEngine = pre.CallEngine;
const sendWith = pre.sendWith;

const body = "{\"model\":\"test-model\",\"max_tokens\":2000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"text\":{\"type\":\"string\"},\"n\":{\"type\":\"number\"}}}}}]";

const mm = "]<]minimax[>[";

/// One call to `edit` with `text` in each format the server reads, then the arguments it must give.
fn calls(a: std.mem.Allocator, text: []const u8) ![7][2][]const u8 {
    const want = try std.mem.concat(a, u8, &.{ "{\"text\":\"", text, "\"}" });
    return .{
        .{ try std.mem.concat(a, u8, &.{ "<tool_call>\n<function=edit>\n<parameter=text>\n", text, "\n</parameter>\n</function>\n</tool_call>" }), want },
        .{ try std.mem.concat(a, u8, &.{ "<tool_call>edit<arg_key>text</arg_key><arg_value>", text, "</arg_value></tool_call>" }), want },
        .{ try std.mem.concat(a, u8, &.{ "<|tool_call>call:edit{text:<|\"|>", text, "<|\"|>}<tool_call|>" }), want },
        .{ try std.mem.concat(a, u8, &.{ mm, "<tool_call>", mm, "<invoke name=\"edit\">", mm, "<text>", text, mm, "</text>", mm, "</invoke>", mm, "</tool_call>" }), want },
        .{ try std.mem.concat(a, u8, &.{ "<|tool_call_start|>[edit(text='", text, "')]<|tool_call_end|>" }), want },
        .{ try std.mem.concat(a, u8, &.{ "<|tool_call_start|>[{\"name\":\"edit\",\"arguments\":{\"text\":\"", text, "\"}}]<|tool_call_end|>" }), want },
        .{ try std.mem.concat(a, u8, &.{ "<tool_call>{\"name\":\"edit\",\"arguments\":{\"text\":\"", text, "\"}}</tool_call>" }), want },
    };
}

fn expectCall(a: std.mem.Allocator, reply: []const u8, want: ?[]const u8) !void {
    const tokens = try TestText.encode(undefined, a, reply, false);
    for ([_]bool{ false, true }) |batched| {
        var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
        const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
        const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(streamed.opened and streamed.status == null);
        const n: usize = if (want != null) 1 else 0;
        std.testing.expectEqual(n, whole.calls) catch |err| {
            std.debug.print("whole reply of {s}: {d} calls\n", .{ reply, whole.calls });
            return err;
        };
        std.testing.expectEqual(n, streamed.calls) catch |err| {
            std.debug.print("streamed reply of {s}: {d} calls\n", .{ reply, streamed.calls });
            return err;
        };
        if (want) |w| {
            try std.testing.expectEqualStrings(w, whole.arguments[0..whole.arguments_len]);
            try std.testing.expectEqualStrings(w, streamed.arguments[0..streamed.arguments_len]);
        }
    }
}

test "tool text values: NaN, Infinity and InfinityPool in a value are text in every format, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "fix NaN handling", "Infinity and beyond", "InfinityPool", "NaN" }) |text| {
        for (try calls(a, text)) |case| try expectCall(a, case[0], case[1]);
    }
}

test "tool text values: a JSON NaN or Infinity literal still refuses, as a MiniMax number's does" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "NaN", "Infinity", "-Infinity" }) |lit| {
        try expectCall(a, try std.mem.concat(a, u8, &.{ "<tool_call>{\"name\":\"edit\",\"arguments\":{\"n\":", lit, "}}</tool_call>" }), null);
        try expectCall(a, try std.mem.concat(a, u8, &.{ "<|tool_call_start|>[{\"name\":\"edit\",\"arguments\":{\"n\":", lit, "}}]<|tool_call_end|>" }), null);
        try expectCall(a, try std.mem.concat(a, u8, &.{ mm, "<tool_call>", mm, "<invoke name=\"edit\">", mm, "<n>", lit, mm, "</n>", mm, "</invoke>", mm, "</tool_call>" }), null);
    }
    // a MiniMax number that is one stays a call, as before
    try expectCall(a, mm ++ "<tool_call>" ++ mm ++ "<invoke name=\"edit\">" ++ mm ++ "<n>3" ++ mm ++ "</n>" ++ mm ++ "</invoke>" ++ mm ++ "</tool_call>", "{\"n\":3}");
}
