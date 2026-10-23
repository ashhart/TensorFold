const std = @import("std");
const json = @import("json");
const tool_params = @import("tool_params.zig");
const witness = @import("tool_union_witness.zig");
const stream = @import("tool_stream.zig");
const Streamer = stream.Streamer;
const Policy = stream.Policy;
const Value = json.Value;
const Allocator = std.mem.Allocator;

test "a streamed call its end token left open completes as its closing markup completes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars\n</parameter>\n<parameter=n>\n5\n</parameter>\n";
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    const ended = try streamed(a, tools, &.{ open, try std.mem.concat(a, u8, &.{ open, try tool_parse.closeCall(a, open, tools) }) });
    const marked = try streamed(a, tools, &.{ open, open ++ "</function>\n</tool_call>" });
    // the arguments sent before the end, closed by closedJson, are the ones the client ends with
    try std.testing.expectEqualStrings((try tool_params.closedJson(a, sent)).?, try sentArguments(a, ended));
    try std.testing.expectEqual(marked.len, ended.len);
    for (marked[1..], ended[1..]) |x, y| try std.testing.expectEqualStrings(try json.stringify(a, x, .{}), try json.stringify(a, y, .{}));
}

test "a streamed call cut inside a value stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars and more";
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, open, tools));
    // its opening delta went out and cannot be taken back; what it sent does not close under closedJson
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    try std.testing.expectEqualStrings("{\"q\":\"fast c", sent);
    try std.testing.expect((try tool_params.closedJson(a, sent)) == null);
}

test "a streamed parameter whose schema is a union comes back as the value the model wrote" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed quoted string under a union stays the text the model wrote" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.quoted_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed parameter whose schema points at a definition comes back as the value the model wrote" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.ref_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed parameter whose schema combines branches comes back as the value the model wrote" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.combined_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed parameter whose schema names a number and an integer takes the integer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.number_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed parameter whose schema constrains the value with every keyword comes back as the value the model wrote" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.keyword_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed parameter whose schema points at a definition by pointer comes back as the value the model wrote" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.pointer_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

test "a streamed parameter whose schema names no kinds constrains nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (witness.unconstrained_cases) |c| try std.testing.expectEqualStrings(try canonical(a, c.expected), try canonical(a, try streamedArguments(a, c)));
}

/// The arguments the stream gives the client for one case.
fn streamedArguments(a: Allocator, c: witness.Case) ![]const u8 {
    const tools = try witness.toolsWith(a, c.schema, c.defs);
    const deltas = try streamed(a, tools, &.{try witness.reply(a, c.value)});
    return sentArguments(a, deltas);
}

/// The arguments as one canonical JSON text.
fn canonical(a: Allocator, text: []const u8) ![]const u8 {
    return json.stringify(a, (try json.parse(a, text)).ok, .{ .compact = true });
}

test "a one-call reply retains an unoffered call and still hides an unclosed one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const single: Policy = .{ .single = true };
    const text = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    // what attachCalls makes of the reply: the parse, then the policy's content
    const r = try tool_parse.parse(a, text, tools, single.maxCalls());
    try std.testing.expect(r.calls == null);
    try std.testing.expectEqualStrings(text, r.content);
    try std.testing.expectEqualStrings(text, try single.parsedContent(a, r.content));
    try std.testing.expectEqualStrings("Launching. ", try single.parsedContent(a, "Launching. <tool_call>{\"name\""));
    // a stream that sent the prose before the block ends with the block
    try std.testing.expectEqualStrings(text["Launching. ".len..], single.kept(text, "\nLaunching. ").?);
    try std.testing.expect(single.kept("Launching.", "Launching. ") == null);
    const parallel: Policy = .{};
    try std.testing.expect(parallel.kept(text, "Launching. ") == null);
}

const tool_parse = @import("tool_parse.zig");

/// The deltas one streamer sends while the reply grows through ``texts``.
fn streamed(a: Allocator, tools: []const Value, texts: []const []const u8) Allocator.Error![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (texts) |t| try s.feed(t, &out);
    return out.items;
}

/// The arguments a client joins from streamed deltas.
fn sentArguments(a: Allocator, deltas: []const Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    };
    return out.items;
}

test "DSML containment survives every byte boundary and preserves adjacent call order" {
    const prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
        "<｜DSML｜parameter name=\"q\" string=\"true\">";
    const suffix = "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
    const legacy = "<tool_call><function=lookup></function></tool_call>";
    for ([_][]const u8{ legacy, "<tool_call><function=lookup></function>" }) |literal| {
        var outer_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer outer_arena.deinit();
        const outer_a = outer_arena.allocator();
        const text = try std.mem.concat(outer_a, u8, &.{ legacy, prefix, literal, suffix, legacy });
        const expected = try std.mem.concat(outer_a, u8, &.{ "{}{\"q\":", try json.quote(outer_a, literal, .{}), "}{}" });
        for (0..text.len + 1) |boundary| {
            var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
            var s = try Streamer.init(a, tools);
            var out: std.ArrayList(Value) = .empty;
            try s.feed(text[0..boundary], &out);
            try s.feed(text, &out);
            try s.feed(text, &out);
            var calls: i64 = 0;
            for (out.items) |delta| for (delta.get("tool_calls").?.array) |call| {
                if (call.get("id") == null) continue;
                try std.testing.expectEqual(calls, call.get("index").?.int64().?);
                try std.testing.expectEqualStrings("lookup", call.get("function").?.strField("name").?);
                calls += 1;
            };
            try std.testing.expectEqual(@as(i64, 3), calls);
            try std.testing.expectEqualStrings(expected, try sentArguments(a, out.items));
        }
    }
}

test "unfinished DSML retains containment through byte feeds and final closeCall" {
    const prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
        "<｜DSML｜parameter name=\"q\" string=\"true\">";
    const literals = [_][]const u8{
        "<tool_call><function=lookup></function></tool_call>",
        "<tool_call><function=lookup></function>",
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{}",
    };
    for (literals) |literal| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
        const text = try std.mem.concat(a, u8, &.{ prefix, literal });
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        for (0..text.len + 1) |n| {
            try s.feed(text[0..n], &out);
            try std.testing.expectEqual(@as(usize, 0), out.items.len);
        }
        const tail = try tool_parse.closeCall(a, text, tools);
        try std.testing.expectEqualStrings("", tail);
        try s.feed(try std.mem.concat(a, u8, &.{ text, tail }), &out);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
        try std.testing.expect(!s.streamed);
    }
}

test "malformed closed DSML masks its nested XML and resumes the next legitimate call" {
    const malformed = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
        "<｜DSML｜parameter name=\"q\" string=\"true\">" ++
        "<tool_call><function=lookup></function></tool_call>" ++
        "</｜DSML｜parameter></｜DSML｜tool_calls>";
    const legacy = "<tool_call><function=lookup></function></tool_call>";
    const text = malformed ++ legacy;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (0..malformed.len + 1) |n| {
        try s.feed(text[0..n], &out);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
    }
    for (malformed.len + 1..text.len + 1) |n| try s.feed(text[0..n], &out);
    var calls: usize = 0;
    for (out.items) |delta| for (delta.get("tool_calls").?.array) |call| {
        if (call.get("id") != null) calls += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), calls);
    try std.testing.expectEqualStrings("{}", try sentArguments(a, out.items));
}
