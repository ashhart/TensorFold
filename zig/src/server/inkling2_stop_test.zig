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
    const plain = try h.ask(a, "release", &.{2}, "", false, 1);
    var failures: usize = 0;
    for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| {
        const got = if (lane) try h.askLane(a, "release", reply, required, stride > 0, @max(stride, 1)) else try h.ask(a, "release", reply, required, stride > 0, @max(stride, 1));
        const exact = std.mem.eql(u8, "lookuplookup", got.names) and
            std.mem.eql(u8, "{\"q\":\"x\"}{\"q\":\"y\"}", got.arguments) and
            std.mem.eql(u8, "tool_calls", got.finish) and std.mem.eql(u8, want, got.token_ids) and
            got.eos_len == plain.eos_len and got.grammars == 0 and got.tool_grammars == 1;
        std.debug.print("INKLING_STOP_CELL lane={any} stride={d} exact={any} calls={s} eos={d}/{d}\n", .{ lane, stride, exact, got.names, got.eos_len, plain.eos_len });
        if (!exact) failures += 1;
    }
    try t.expectEqual(@as(usize, 0), failures);
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
    const plain = try h.ask(a, "release", &.{2}, "", false, 1);
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
            try t.expectEqual(plain.eos_len, got.eos_len);
            try t.expectEqual(case.grammars, got.grammars);
            try t.expectEqual(@as(usize, 0), got.tool_grammars);
        };
    }
}

test "Inkling 2 reply: non-Inkling required tools keep all token ids and the shared compiler" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try std.mem.concat(a, u32, &.{ try h.encodeReply(a, call_x ++ "<|end_message|>" ++ call_x), &.{2} });
    const want = try std.json.Stringify.valueAlloc(a, reply, .{});
    const plain = try h.ask(a, "not_inkling2", &.{2}, "", false, 1);
    for ([_]bool{ false, true }) |lane| for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| {
        const got = if (lane) try h.askLane(a, "not_inkling2", reply, required, stride > 0, @max(stride, 1)) else try h.ask(a, "not_inkling2", reply, required, stride > 0, @max(stride, 1));
        try t.expectEqualStrings(want, got.token_ids);
        try t.expectEqualStrings("lookuplookup", got.names);
        try t.expectEqual(plain.eos_len, got.eos_len);
        try t.expectEqual(@as(usize, 1), got.grammars);
        try t.expectEqual(@as(usize, 0), got.tool_grammars);
    };
}

test "Inkling 2 reply: complete-call stop requires a surviving required-tool grammar" {
    const callEnd = @import("chat_inkling2.zig").callEnd;
    const api = @import("engine_api");
    var srv: struct { text: struct { native_inkling2: bool } } = .{ .text = .{ .native_inkling2 = true } };
    const tool: api.Structure = .{ .kind = .structural_tag, .field = "tool_choice" };
    try t.expect(callEnd(&srv, true, tool));
    try t.expect(!callEnd(&srv, false, tool));
    try t.expect(!callEnd(&srv, true, null));
    try t.expect(!callEnd(&srv, true, .{ .kind = .json_schema, .field = "response_format" }));
    try t.expect(!callEnd(&srv, true, .{ .kind = .structural_tag, .field = "guided_structural_tag" }));
    srv.text.native_inkling2 = false;
    try t.expect(!callEnd(&srv, true, tool));
}
