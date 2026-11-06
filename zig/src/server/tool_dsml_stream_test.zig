//! DSML's streamed arguments join to the whole reply's calls byte for byte; a block that breaks early sends nothing.
const std = @import("std");
const json = @import("json");
const tool_dsml = @import("tool_dsml.zig");
const tool_parse = @import("tool_parse.zig");
const Streamer = @import("tool_stream.zig").Streamer;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Spelling = tool_dsml.Spelling;
const spellings = &tool_dsml.spellings;

/// The tools these cases offer.
pub const tools_text = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},\"tags\":{\"type\":\"array\"}}}}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}]";

fn toolsOf(a: Allocator) ![]const Value {
    return (try json.parse(a, tools_text)).ok.array;
}

/// A parameter as written: its key, whether it is a string (``string="true"``), and its value.
pub const Param = struct { []const u8, bool, []const u8 };

/// One invoke of ``name`` in spelling ``sp``, a newline after each tag as DeepSeek writes them.
pub fn invoke(a: Allocator, sp: *const Spelling, name: []const u8, params: []const Param) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s}{s}\">\n", .{ sp.invoke_open, name });
    for (params) |p| try out.print(a, "{s}{s}\" string=\"{s}\">{s}{s}\n", .{ sp.param_open, p[0], if (p[1]) "true" else "false", p[2], sp.param_close });
    try out.appendSlice(a, sp.invoke_close);
    return out.items;
}

/// A block of ``invokes`` in spelling ``sp``.
pub fn block(a: Allocator, sp: *const Spelling, invokes: []const []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ sp.open, "\n", try std.mem.join(a, "\n", invokes), "\n", sp.close });
}

/// The deltas a streamer sends while the reply grows through ``pieces``.
fn streamed(a: Allocator, tools: []const Value, pieces: []const []const u8) ![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (pieces) |p| try s.feed(p, &out);
    return out.items;
}

/// The deltas of a reply fed one more byte at a time.
fn bytewise(a: Allocator, tools: []const Value, text: []const u8) ![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (1..text.len + 1) |n| try s.feed(text[0..n], &out);
    return out.items;
}

/// The arguments a client joins from streamed deltas.
fn joined(a: Allocator, deltas: []const Value) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| try out.appendSlice(a, one.get("function").?.strField("arguments").?);
    return out.items;
}

/// The names the stream's headers carry, joined by spaces.
fn names(a: Allocator, deltas: []const Value) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| if (one.get("id") != null) {
        try out.print(a, "{s} ", .{one.get("function").?.strField("name").?});
    };
    return out.items;
}

/// The whole reply's calls: their joined arguments and their names, as ``joined`` and ``names`` write them.
fn whole(a: Allocator, tools: []const Value, text: []const u8) !struct { args: []const u8, names: []const u8 } {
    const calls = (try tool_parse.parse(a, text, tools, null)).calls orelse return .{ .args = "", .names = "" };
    var args: std.ArrayList(u8) = .empty;
    var called: std.ArrayList(u8) = .empty;
    for (calls) |c| {
        try args.appendSlice(a, c.get("function").?.strField("arguments").?);
        try called.print(a, "{s} ", .{c.get("function").?.strField("name").?});
    }
    return .{ .args = args.items, .names = called.items };
}

test "a DSML call's two thousand byte value streams while it is written, in every spelling, and joins as the whole's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const value = try a.alloc(u8, 2000);
    for (value, 0..) |*c, i| c.* = "abcdefghij klmnopqrstuvwxyz\n"[i % 28];
    for (spellings) |*sp| {
        const text = try block(a, sp, &.{try invoke(a, sp, "write_file", &.{ .{ "path", true, "notes.txt" }, .{ "content", true, value } })});
        const closer = std.mem.lastIndexOf(u8, text, sp.param_close).?;
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        var early: usize = 0;
        var at: usize = 0;
        while (at < text.len) : (at += 3) {
            try s.feed(text[0..@min(at + 3, text.len)], &out);
            if (at + 3 < closer) early = out.items.len;
        }
        try std.testing.expectEqualStrings("write_file ", try names(a, out.items));
        try std.testing.expect(out.items.len > 100);
        try std.testing.expect(early > 100); // the header, the path and the content's pieces went out before it closed
        try std.testing.expectEqualStrings((try whole(a, tools, text)).args, try joined(a, out.items));
        try std.testing.expect(!s.open());
    }
}

test "DSML calls stream the whole reply's arguments at every byte boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    for (spellings) |*sp| {
        const texts = [_][]const u8{
            // a string the JSON escapes, then an integer
            try block(a, sp, &.{try invoke(a, sp, "lookup", &.{ .{ "q", true, "h\u{e9}llo < 2 \"fast\" \\ \x01\n\u{1f600}" }, .{ "n", false, "3" } })}),
            // typed values keep the whole reply's compact JSON; a string holds the starts of closers and other markup
            try block(a, sp, &.{try invoke(a, sp, " LOOKUP ", &.{
                .{ "tags", false, "[\"a\", \"b\"]" },
                .{ "obj", false, " {\"k\": 1, \"j\": [1, 2.50, 1e400]} " },
                .{ "q", true, "</\u{ff5c}DSML\u{ff5c} par </\u{ff5c}DSML\u{ff5c}invo <\u{ff5c}DSML\u{ff5c} invoke name=\"x\"> </tool_call>" },
            })}),
            // two invokes, the second without a parameter; a key outside ASCII; wide whitespace between tags
            try std.mem.replaceOwned(u8, a, try block(a, sp, &.{
                try invoke(a, sp, "write_file", &.{ .{ "k\u{e9}y", true, "" }, .{ "content", true, "x" } }),
                try invoke(a, sp, "lookup", &.{}),
            }), "\n", "\n\u{3000} \t"),
            // a single string parameter
            try block(a, sp, &.{try invoke(a, sp, "lookup", &.{.{ "q", true, "only" }})}),
        };
        for (texts) |text| {
            const want = try whole(a, tools, text);
            try std.testing.expect(want.names.len > 0);
            const one = try bytewise(a, tools, text);
            try std.testing.expectEqualStrings(want.names, try names(a, one));
            try std.testing.expectEqualStrings(want.args, try joined(a, one));
            for (0..text.len + 1) |boundary| {
                const pieces = try streamed(a, tools, &.{ text[0..boundary], text });
                try std.testing.expectEqualStrings(want.names, try names(a, pieces));
                try std.testing.expectEqualStrings(want.args, try joined(a, pieces));
            }
        }
    }
}

test "seeded DSML blocks stream every call before their closer and join as the whole reply's" {
    var prng: std.Random.DefaultPrng = .init(0x1975d5);
    const r = prng.random();
    const gaps = [_][]const u8{ "", " ", "\n", "\t\n", "\u{3000}", "\u{a0}" };
    const pieces = [_][]const u8{ "a", " ", "\n", "\"", "\\", "<", ">", "\u{e9}", "\u{1f600}", "\u{ff5c}", "</", "</\u{ff5c}DSML\u{ff5c}", "<tool_call>", "\x01", "\t", "}" };
    const typed = [_][]const u8{ "3", "-0.5", "true", "null", "[1, 2]", "{\"a\": [1, {\"b\": null}]}", "\"s\\u00e9\"", " 7 ", "1E3", "[]", "{}" };
    const keys = [_][]const u8{ "q", "n", "path", "content", "k\u{e9}", " a b " };
    const called = [_][]const u8{ "lookup", "LOOKUP", " write_file " };
    var valid: usize = 0;
    for (0..400) |_| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try toolsOf(a);
        const sp = &spellings[r.uintLessThan(usize, spellings.len)];
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(a, sp.open);
        for (0..1 + r.uintLessThan(usize, 3)) |_| {
            try text.print(a, "{s}{s}{s}\">", .{ gaps[r.uintLessThan(usize, gaps.len)], sp.invoke_open, called[r.uintLessThan(usize, called.len)] });
            const first_key = r.uintLessThan(usize, keys.len); // distinct keys: a repeated one is the gap tested above
            for (0..r.uintLessThan(usize, 4)) |k| {
                const string = r.boolean();
                try text.print(a, "{s}{s}{s}\" string=\"{s}\">", .{ gaps[r.uintLessThan(usize, gaps.len)], sp.param_open, keys[(first_key + k) % keys.len], if (string) "true" else "false" });
                if (string) {
                    for (0..r.uintLessThan(usize, 12)) |_| try text.appendSlice(a, pieces[r.uintLessThan(usize, pieces.len)]);
                } else try text.appendSlice(a, typed[r.uintLessThan(usize, typed.len)]);
                try text.appendSlice(a, sp.param_close);
            }
            try text.print(a, "{s}{s}", .{ gaps[r.uintLessThan(usize, gaps.len)], sp.invoke_close });
        }
        try text.print(a, "{s}", .{gaps[r.uintLessThan(usize, gaps.len)]});
        const before_close = text.items.len;
        try text.appendSlice(a, sp.close);
        const want = try whole(a, tools, text.items);
        if (want.names.len == 0) continue; // the pieces happened to spell a closer
        valid += 1;
        // every invoke closed before the block's closer, so every call has gone out whole by then
        const early = try streamed(a, tools, &.{text.items[0..before_close]});
        try std.testing.expectEqualStrings(want.names, try names(a, early));
        try std.testing.expectEqualStrings(want.args, try joined(a, early));
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        var at: usize = 0;
        while (at < text.items.len) {
            at = @min(text.items.len, at + 1 + r.uintLessThan(usize, 9));
            try s.feed(text.items[0..at], &out);
        }
        try std.testing.expectEqualStrings(want.names, try names(a, out.items));
        try std.testing.expectEqualStrings(want.args, try joined(a, out.items));
        try std.testing.expect(s.state == .outside);
    }
    try std.testing.expect(valid > 360);
}

/// Blocks that break before their first call's header could go out: the whole reply keeps each as text.
pub fn earlyBreaks(a: Allocator, sp: *const Spelling) ![]const []const u8 {
    const q = "\" string=\"true\">";
    // the grammar's spelling shares tool_calls' invoke tags and calls' parameter tags: another spelling per tag
    const other_invoke = &spellings[if (sp == &spellings[1]) 0 else 1];
    const other = &spellings[if (sp == &spellings[0]) 1 else 0];
    const first = try std.mem.concat(a, u8, &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q", q, "a", sp.param_close });
    var out: std.ArrayList([]const u8) = .empty;
    for ([_][]const []const u8{
        &.{ sp.open, sp.invoke_open, "launch\">", sp.param_open, "q", q, "a", sp.param_close, sp.invoke_close, sp.close }, // not offered
        &.{ sp.open, "x", sp.invoke_open, "lookup\">", sp.invoke_close, sp.close }, // text before the invoke
        &.{ sp.open, sp.invoke_open, "lookup\" >", sp.invoke_close, sp.close }, // the name's quote without >
        &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q\" string=\"yes\">a", sp.param_close, sp.invoke_close, sp.close },
        &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q\">a", sp.param_close, sp.invoke_close, sp.close },
        &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "n\" string=\"false\">{oops", sp.param_close, sp.param_open, "q", q, "b", sp.param_close, sp.invoke_close, sp.close },
        &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q", q, "<tool_call><function=lookup></function></tool_call>" }, // never closed
        &.{ first, sp.close }, // the block closed with its invoke open
        &.{ first, other_invoke.invoke_close, sp.close }, // another spelling's invoke closer
        &.{ first, "x", sp.param_open, "n", q, "b", sp.param_close, sp.invoke_close, sp.close }, // text between parameters
        &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q", q, "a", sp.invoke_close, "b", sp.param_close, sp.invoke_close, sp.close },
        &.{ sp.open, "\n", sp.close }, // no invoke
        &.{ sp.open, sp.invoke_open, "lookup\">", other.param_open, "q", q, "a", other.param_close, sp.invoke_close, sp.close },
        &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q", q, "a", other.param_close, other_invoke.invoke_close, other.close },
    }) |parts| try out.append(a, try std.mem.concat(a, u8, parts));
    return out.items;
}

test "a DSML block that breaks before its first header sends nothing at any byte, as the whole reply makes no call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    for (spellings) |*sp| for (try earlyBreaks(a, sp)) |text| {
        try std.testing.expect((try tool_parse.parse(a, text, tools, null)).calls == null);
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        for (1..text.len + 1) |n| try s.feed(text[0..n], &out);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
        try std.testing.expect(!s.streamed and !s.open());
        // a later call still streams
        const xml = "<tool_call>\n<function=lookup>\n<parameter=q>\nb\n</parameter>\n</function>\n</tool_call>";
        const after = try std.mem.concat(a, u8, &.{ text, xml });
        try std.testing.expectEqualStrings((try whole(a, tools, after)).args, try joined(a, try bytewise(a, tools, after)));
    };
}

test "a DSML block that breaks after a header went out: the whole reply keeps it as text, the stream sent the call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const sp = &spellings[1];
    const head = try std.mem.concat(a, u8, &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q\" string=\"true\">a", sp.param_close, sp.param_open });
    const cases = [_]struct { []const u8, []const u8 }{
        .{ try std.mem.concat(a, u8, &.{ head, "n\" string=\"yes\">3", sp.param_close, sp.invoke_close, sp.close }), "{\"q\":\"a\"" },
        .{ try std.mem.concat(a, u8, &.{ head, "n\" string=\"false\">{oops", sp.param_close, sp.invoke_close, sp.close }), "{\"q\":\"a\",\"n\":" },
        .{ try std.mem.concat(a, u8, &.{ head, "n\" string=\"true\">b", sp.param_close, sp.invoke_close, sp.invoke_open, "launch\">", sp.invoke_close, sp.close }), "{\"q\":\"a\",\"n\":\"b\"}" },
    };
    for (cases) |case| {
        try std.testing.expect((try tool_parse.parse(a, case[0], tools, null)).calls == null);
        const deltas = try bytewise(a, tools, case[0]);
        try std.testing.expectEqualStrings("lookup ", try names(a, deltas));
        try std.testing.expectEqualStrings(case[1], try joined(a, deltas));
    }
}

test "a repeated DSML key streams both pairs, as XML and GLM do; the whole reply keeps the last" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const text = try block(a, &spellings[1], &.{try invoke(a, &spellings[1], "lookup", &.{ .{ "q", true, "a" }, .{ "q", true, "b" } })});
    try std.testing.expectEqualStrings("{\"q\":\"a\",\"q\":\"b\"}", try joined(a, try bytewise(a, tools, text)));
    try std.testing.expectEqualStrings("{\"q\":\"b\"}", (try whole(a, tools, text)).args);
}

test "a cut DSML value replays from the stream; before its first header the stream makes no call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const sp = &spellings[1];
    const head = try std.mem.concat(a, u8, &.{ sp.open, "\n", sp.invoke_open, "write_file\">\n", sp.param_open, "path\" string=\"true\">notes.txt", sp.param_close, "\n", sp.param_open });
    const cases = [_]struct { []const u8, []const u8, bool }{
        .{ try std.mem.concat(a, u8, &.{ head, "content\" string=\"true\">The history of</\u{ff5c}DSML\u{ff5c} par" }), "{\"path\":\"notes.txt\",\"content\":\"The history of", true },
        .{ try std.mem.concat(a, u8, &.{ head, "limit\" string=\"false\">12" }), "{\"path\":\"notes.txt\",\"limit\":12", true },
        .{ try std.mem.concat(a, u8, &.{ head, "content\" str" }), "{\"path\":\"notes.txt\"", true },
        .{ try std.mem.concat(a, u8, &.{ sp.open, sp.invoke_open, "write_file\">", sp.param_open, "path\" string=\"true\">notes" }), "", false },
    };
    for (cases) |case| {
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        try s.finishCut(case[0], &out);
        try std.testing.expectEqual(case[2], s.open());
        try std.testing.expectEqualStrings(case[1], try joined(a, out.items));
    }
}
