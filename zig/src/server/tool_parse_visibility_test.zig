//! Rejected tool envelopes stay visible without consuming the accepted-call limit.
const std = @import("std");
const json = @import("json");
const tp = @import("tool_parse.zig");
const rc = @import("reply_code.zig");
const rt = @import("reply_text.zig");

test {
    _ = rc;
}
const tail_mod = @import("reply_tool_tail.zig");
const lfm2 = @import("tool_lfm2.zig");
const stream = @import("tool_stream.zig");
const parse = tp.parse;

fn testTools(a: std.mem.Allocator) ![]const json.Value {
    return (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
}

fn streamedArgs(a: std.mem.Allocator, tools: []const json.Value, text: []const u8, split: usize) !struct { headers: usize, args: []const u8 } {
    var s = try stream.Streamer.init(a, tools);
    var out: std.ArrayList(json.Value) = .empty;
    try s.feed(text[0..split], &out);
    try s.finishCut(text, &out);
    var headers: usize = 0;
    var args: std.ArrayList(u8) = .empty;
    for (out.items) |delta| for (delta.get("tool_calls").?.array) |one| {
        if (one.get("id") != null) headers += 1;
        if (one.get("function").?.strField("arguments")) |part| try args.appendSlice(a, part);
    };
    return .{ .headers = headers, .args = args.items };
}

test "tool visibility: a first unoffered call remains content in single-call mode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const forms = [_][]const u8{
        "<tool_call><function=tools.session_move><parameter=folder>project</parameter></function></tool_call>",
        "<tool_call>{\"name\":\"tools.session_move\",\"arguments\":{}}</tool_call>",
        "<tool_call>not a tool payload</tool_call>",
    };
    for (forms) |text| for ([_]?usize{ null, 1 }) |limit| {
        const result = try parse(a, text, tools, limit);
        try std.testing.expect(result.calls == null);
        try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
        try std.testing.expectEqualStrings(text, result.content);
    };
}

test "tool visibility: rejected text does not consume a valid call and the second accepted call stays limited" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const bad = "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>";
    const good = "<tool_call>{\"name\":\"execute\",\"arguments\":{}}</tool_call>";
    const result = try parse(a, bad ++ good ++ good, tools, 1);
    try std.testing.expectEqualStrings(bad, result.content);
    try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
    try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
    try std.testing.expectEqualStrings("execute", result.calls.?[0].get("function").?.get("name").?.string);
}

test "quoted openers in inline code or fenced blocks do not swallow a following call" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try testTools(a);
    const call = "<tool_call>lookup<arg_key>q</arg_key><arg_value>ok</arg_value></tool_call>";
    const cases = [_]struct { reply: []const u8, content: []const u8 }{
        .{ .reply = "Raw `<tool_call>` markup before call.\n" ++ call, .content = "Raw `<tool_call>` markup before call." },
        .{ .reply = "Doubled `` `<tool_call>` `` markup.\n" ++ call, .content = "Doubled `` `<tool_call>` `` markup." },
        .{ .reply = "Fenced:\n```xml\n<tool_call>\n```\n" ++ call, .content = "Fenced:\n```xml\n<tool_call>\n```" },
        .{ .reply = "Tilde:\n~~~\n<|tool_call>call:lookup{}\n~~~\n" ++ call, .content = "Tilde:\n~~~\n<|tool_call>call:lookup{}\n~~~" },
        .{ .reply = "Note `<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">` first.\n" ++ call, .content = "Note `<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">` first." },
    };
    for (cases) |c| {
        const res = try parse(a, c.reply, tools, null);
        try std.testing.expectEqual(@as(usize, 0), res.rejected_envelopes);
        try std.testing.expectEqualStrings(c.content, res.content);
        try std.testing.expectEqual(@as(usize, 1), res.calls.?.len);
        try std.testing.expectEqualStrings("{\"q\":\"ok\"}", res.calls.?[0].get("function").?.get("arguments").?.string);
    }
    // An unclosed fence inside a real call's value stays part of that call and must not suppress the next call.
    const nested = "<tool_call>lookup<arg_key>q</arg_key><arg_value>```md\n<tool_call>\n</arg_value></tool_call>\n" ++ call;
    const nested_res = try parse(a, nested, tools, null);
    try std.testing.expectEqual(@as(usize, 2), nested_res.calls.?.len);
    try std.testing.expectEqualStrings("{\"q\":\"```md\\n<tool_call>\\n\"}", nested_res.calls.?[0].get("function").?.get("arguments").?.string);
    try std.testing.expectEqualStrings("{\"q\":\"ok\"}", nested_res.calls.?[1].get("function").?.get("arguments").?.string);
    const cut = "Raw `<tool_call>` and `<｜DSML｜tool_calls>` before:\n<tool_call>lookup<arg_key>q</arg_key><arg_value>ok";
    try std.testing.expectEqualStrings("</arg_value></tool_call>", try tp.closeCall(a, cut, tools));
    const quoted_lfm = "Note `<|tool_call_start|>[lookup(q=\"x\")]<|tool_call_end|>` prose.";
    try std.testing.expectEqualStrings(quoted_lfm, try lfm2.dsml(a, quoted_lfm, tools, true));
}

test "streamed quoted GLM opener naming an offered tool stays prose at every byte boundary" {
    const text = "The tag `<tool_call>lookup<arg_key>bad` broke; running:\n" ++
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>ok</arg_value></tool_call>";
    const want_prose = "The tag `<tool_call>lookup<arg_key>bad` broke; running:\n";
    var outer = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer outer.deinit();
    const whole = try rt.hideToolCalls(outer.allocator(), text, true);
    try std.testing.expectEqualStrings(want_prose, whole);
    for (0..text.len + 1) |b| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try testTools(a);
        const partial = try rt.hideToolCalls(a, text[0..b], false);
        try std.testing.expect(std.mem.startsWith(u8, whole, partial));
        const got = try streamedArgs(a, tools, text, b);
        try std.testing.expectEqual(@as(usize, 1), got.headers);
        try std.testing.expectEqualStrings("{\"q\":\"ok\"}", got.args);
    }
}

test "complete calls inside inline code or fenced blocks stay prose whole and streamed" {
    const cases = [_][]const u8{
        "Use `<tool_call>lookup<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>` here.",
        "Example:\n```xml\n<tool_call>lookup<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>\n```\nDone.",
        "Example:\n~~~\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>\n~~~",
        "Gemma form: `<|tool_call>call:lookup{q:<|\"|>x<|\"|>}<tool_call|>`.",
        "LFM form: `<|tool_call_start|>[lookup(q=\"x\")]<|tool_call_end|>`.",
    };
    const single: stream.Policy = .{ .single = true };
    for (cases) |text| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try testTools(a);
        const res = try parse(a, text, tools, null);
        try std.testing.expect(res.calls == null);
        try std.testing.expectEqual(@as(usize, 0), res.rejected_envelopes);
        try std.testing.expectEqualStrings(text, res.content);
        try std.testing.expectEqualStrings(text, try rt.hideToolCalls(a, text, true));
        try std.testing.expectEqualStrings(text, try rt.finishedProse(a, text));
        try std.testing.expectEqualStrings(text, try single.parsedContent(a, text));
        for (0..text.len + 1) |b| {
            const partial = try rt.hideToolCalls(a, text[0..b], false);
            try std.testing.expect(std.mem.startsWith(u8, text, partial));
        }
    }
    const tail = "Thought.\n```xml\n<tool_call>lookup<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>\n```";
    try std.testing.expect(tail_mod.start(tail, rt.shared_openers) == null);
}

test "single-call TextFilter keeps unclosed quoted openers and stray backtick does not block later call" {
    const prose = "Prose quotes `<tool_call>` and `<TOOL_CALL>` and `<ns:tool_call>` and keeps going.";
    var outer = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer outer.deinit();
    const oa = outer.allocator();
    var f: stream.TextFilter = .{};
    var joined: std.ArrayList(u8) = .empty;
    for (0..prose.len) |i| try joined.appendSlice(oa, try f.feed(oa, prose[i .. i + 1]));
    try joined.appendSlice(oa, f.finish());
    try std.testing.expectEqualStrings(prose, joined.items);
    const single: stream.Policy = .{ .single = true };
    try std.testing.expectEqualStrings(prose, try single.parsedContent(oa, prose));
    const stray_cases = [_]struct { reply: []const u8, prose: []const u8 }{
        .{ .reply = "Format is `<tool_call>lookup<arg_key>q</arg_key><arg_value>ok</arg_value></tool_call>", .prose = "Format is `" },
        .{ .reply = "Format is `\n\n<tool_call>lookup<arg_key>q</arg_key><arg_value>ok</arg_value></tool_call>", .prose = "Format is `\n\n" },
    };
    for (stray_cases) |c| {
        const tools = try testTools(oa);
        const res = try parse(oa, c.reply, tools, null);
        try std.testing.expectEqual(@as(usize, 1), res.calls.?.len);
        try std.testing.expectEqualStrings(std.mem.trimEnd(u8, c.prose, "\n"), res.content);
        try std.testing.expectEqualStrings(c.prose, try rt.hideToolCalls(oa, c.reply, true));
        for (0..c.reply.len + 1) |b| {
            var run = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer run.deinit();
            const ra = run.allocator();
            const rtools = try testTools(ra);
            const partial = try rt.hideToolCalls(ra, c.reply[0..b], false);
            try std.testing.expect(std.mem.startsWith(u8, c.prose, partial));
            const got = try streamedArgs(ra, rtools, c.reply, b);
            try std.testing.expectEqual(@as(usize, 1), got.headers);
            try std.testing.expectEqualStrings("{\"q\":\"ok\"}", got.args);
        }
    }
}
