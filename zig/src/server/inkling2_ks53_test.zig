//! KS53's two Inkling 2 findings and the stop boundaries around them, whole and streamed at strides 1, 2, 3 and 256.
const std = @import("std");
const h = @import("inkling2_chat_test.zig");
const t = std.testing;

const strides = [_]usize{ 1, 2, 3, 256 };

/// A raw completion's reply, whole and at every stride: the same content, finish and completion tokens.
fn raw(a: std.mem.Allocator, reply: []const u32, extra: []const u8) !h.Seen {
    const whole = try h.run(a, "release", reply, extra, false, 1, "\"prompt\":\"Hi\"", false);
    for (strides) |stride| {
        const s = try h.run(a, "release", reply, extra, true, stride, "\"prompt\":\"Hi\"", false);
        try t.expectEqualStrings(whole.content, s.content);
        try t.expectEqualStrings(whole.finish, s.finish);
        try t.expectEqual(whole.usage.completion, s.usage.completion);
    }
    return whole;
}

test "Inkling 2 reply: KS53 1, a raw reply opening on an invalid byte meets a replacement stop at offset zero" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 222 is byte 80 and 32 is A: no character before the reply owns 80, so it reads as U+FFFD and stops there
    const first = try raw(a, &.{ 222, 32 }, ",\"stop\":\"\u{fffd}\"");
    try t.expectEqualStrings("", first.content);
    try t.expectEqualStrings("stop", first.finish);
    try t.expectEqual(@as(i64, 1), first.usage.completion);
    const spelled = try raw(a, &.{ 222, 50, 51, 46, 47 }, ",\"stop\":\"\u{fffd}STOP\"");
    try t.expectEqualStrings("", spelled.content);
    try t.expectEqualStrings("stop", spelled.finish);
    try t.expectEqual(@as(i64, 5), spelled.usage.completion);
}

test "Inkling 2 reply: KS53 2, a stop right after a bare envelope keeps its call, as the cut reply reads it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const envelope = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
    const got = try h.probeIds(a, "release", try h.encodeReply(a, envelope ++ "<|message_model|>"), h.lookup ++ ",\"stop\":\"<|message_model|>\"");
    try t.expectEqualStrings("lookup", got.names);
    try t.expectEqualStrings("{\"q\":\"x\"}", got.arguments);
    try t.expectEqualStrings("", got.content);
    try t.expectEqualStrings("tool_calls", got.finish);
}

test "Inkling 2 reply: stops at offset zero, at and inside controls, and at envelope edges, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const envelope = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
    const Case = struct { reply: []const u8, stop: []const u8, tools: bool = false, content: []const u8 = "", reasoning: []const u8 = "", names: []const u8 = "", finish: []const u8 = "stop" };
    const cases = [_]Case{
        // offset zero of a block's text, and the reply's first control itself
        .{ .reply = "<|message_model|><|content_text|>STOP more<|end_message|>", .stop = "STOP" },
        .{ .reply = "<|message_model|><|content_text|>hi<|end_message|>", .stop = "<|message_model|>" },
        // inside a control's spelling, a control with the text after it, and a thinking block's opener
        .{ .reply = "<|message_model|><|content_text|>hi<|end_message|>", .stop = "content_text" },
        .{ .reply = "<|message_model|><|content_text|>hi<|end_message|>", .stop = "<|content_text|>hi" },
        .{ .reply = "<|message_model|><|content_thinking|>abc<|end_message|>", .stop = "<|content_thinking|>" },
        // right after a block's close
        .{ .reply = "<|message_model|><|content_text|>hi<|end_message|><|message_model|>", .stop = "<|end_message|>", .content = "hi" },
        // a bare envelope's opening edge, and text right after its closing edge
        .{ .reply = envelope, .stop = "<tool_call>", .tools = true },
        .{ .reply = envelope ++ "done", .stop = "done", .tools = true, .names = "lookup", .finish = "tool_calls" },
    };
    for (cases) |case| {
        const extra = try std.fmt.allocPrint(a, "{s},\"stop\":{f}", .{ if (case.tools) h.lookup else "", std.json.fmt(case.stop, .{}) });
        const got = try h.probeIds(a, "release", try h.encodeReply(a, case.reply), extra);
        try t.expectEqualStrings(case.content, got.content);
        try t.expectEqualStrings(case.reasoning, got.reasoning);
        try t.expectEqualStrings(case.names, got.names);
        try t.expectEqualStrings(case.finish, got.finish);
    }
    // raw completions: an ASCII stop at offset zero, and windows opening on owned and unowned continuation bytes
    const at_zero = try raw(a, &.{ 50, 51, 46, 47 }, ",\"stop\":\"STOP\"");
    try t.expectEqualStrings("", at_zero.content);
    try t.expectEqual(@as(i64, 4), at_zero.usage.completion);
    // 13 tokens against a 12-token window: the last check's window opens on A9, which C3 before it owns
    var owned: [13]u32 = @splat(32);
    owned[0] = 127;
    owned[1] = 102;
    const whole_char = try raw(a, &owned, ",\"stop\":\"\u{fffd}A\"");
    try t.expectEqualStrings("\u{e9}AAAAAAAAAAA", whole_char.content);
    try t.expectEqualStrings("length", whole_char.finish);
    try t.expectEqual(@as(i64, 13), whole_char.usage.completion);
    const unowned = try raw(a, &.{ 32, 222, 32 }, ",\"stop\":\"\u{fffd}\"");
    try t.expectEqualStrings("A", unowned.content);
    try t.expectEqualStrings("stop", unowned.finish);
    try t.expectEqual(@as(i64, 2), unowned.usage.completion);
}
