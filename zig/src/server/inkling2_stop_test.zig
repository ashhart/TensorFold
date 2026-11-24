//! The partial second-call trigger at the route and production lane stop boundary, with token-id controls.
const std = @import("std");
const h = @import("inkling2_chat_test.zig");
const t = std.testing;
const call_x = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
const open_y = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"y\"}}";
const required = h.lookup ++ ",\"tool_choice\":\"required\",\"return_token_ids\":true";

fn partialSecond(lane: bool) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try std.mem.concat(a, u32, &.{ try h.encodeReply(a, call_x ++ open_y ++ "<|end_message|></tool_call>"), &.{2} });
    const want = try std.json.Stringify.valueAlloc(a, reply, .{});
    for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| {
        const got = if (lane) try h.askLane(a, "release", reply, required, stride > 0, @max(stride, 1)) else try h.ask(a, "release", reply, required, stride > 0, @max(stride, 1));
        try t.expectEqualStrings("lookuplookup", got.names);
        try t.expectEqualStrings("{\"q\":\"x\"}{\"q\":\"y\"}", got.arguments);
        try t.expectEqualStrings("tool_calls", got.finish);
        try t.expectEqualStrings(want, got.token_ids);
        try t.expectEqual(@as(usize, 1), got.eos_len);
        try t.expectEqual(@as(usize, 0), got.grammars);
        try t.expectEqual(@as(usize, 1), got.tool_grammars);
    }
}

test "Inkling 2 reply: KS65 request EOS cannot cut an open second call at the route harness" {
    try partialSecond(false);
}

test "Inkling 2 reply: KS65 request EOS cannot cut an open second call at the production lane commit" {
    try partialSecond(true);
}

test "Inkling 2 reply: plain auto and JSON-schema token ids stay exact through both stop boundaries" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { text: []const u8, extra: []const u8, grammars: usize = 0 };
    const cases = [_]Case{
        .{ .text = "left<|end_message|>right", .extra = "" },
        .{ .text = "left<|end_message|>right", .extra = h.lookup },
        .{ .text = "{\"a\":1}<|end_message|>", .extra = ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}", .grammars = 1 },
    };
    for (cases) |case| {
        const reply = try std.mem.concat(a, u32, &.{ try h.encodeReply(a, case.text), &.{2} });
        const want = try std.json.Stringify.valueAlloc(a, reply, .{});
        const extra = try std.mem.concat(a, u8, &.{ case.extra, ",\"return_token_ids\":true" });
        for ([_]bool{ false, true }) |lane| for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| {
            const got = if (lane) try h.askLane(a, "release", reply, extra, stride > 0, @max(stride, 1)) else try h.ask(a, "release", reply, extra, stride > 0, @max(stride, 1));
            try t.expectEqualStrings(want, got.token_ids);
            try t.expectEqualStrings("stop", got.finish);
            try t.expectEqual(@as(usize, 1), got.eos_len);
            try t.expectEqual(case.grammars, got.grammars);
            try t.expectEqual(@as(usize, 0), got.tool_grammars);
        };
    }
}
