//! tool_grammar's auto grammars through a real libtfgrammar (zig build test-grammar-tools), a character a token.
const std = @import("std");
const sys = @import("sys");
const lanes = @import("lanes");
const json = @import("json");
const built = @import("grammar_library");
const tool_grammar = @import("tool_grammar.zig");
const offered = @import("tool_grammar_auto_test.zig").offered;

const ascii = blk: {
    var s: [95]u8 = undefined;
    for (&s, 0..) |*c, i| c.* = @intCast(0x20 + i);
    break :blk s;
};

/// The stop token, then printable ASCII, blanks and DSML's fullwidth bar: every reply here spells one per character.
const vocab = blk: {
    var v: [ascii.len + 4][]const u8 = undefined;
    v[0] = "<eos>";
    for (v[1 .. ascii.len + 1], 0..) |*t, i| t.* = ascii[i..][0..1];
    v[ascii.len + 1] = "\n";
    v[ascii.len + 2] = "\t";
    v[ascii.len + 3] = "｜";
    break :blk v;
};
const words = (vocab.len + 31) / 32;

const Lib = struct {
    f: lanes.grammar.Library,
    h: *anyopaque,

    fn open(a: std.mem.Allocator) !Lib {
        var dl = try sys.DynLib.open(built.library);
        var f: lanes.grammar.Library = undefined;
        const info = @typeInfo(lanes.grammar.Library).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(f, name) = if (@typeInfo(T) == .optional) dl.lookup(@typeInfo(T).optional.child, "tfg_" ++ name) else dl.lookup(T, "tfg_" ++ name) orelse return error.MissingSymbol;
        }
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(a, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (vocab, 0..) |v, i| {
            if (i > 0) try text.append(a, ',');
            try text.print(a, "{f}:{d}", .{ std.json.fmt(v, .{}), i });
        }
        try text.appendSlice(a, "}},\"decoder\":{\"type\":\"Fuse\"}}");
        var err: [512]u8 = undefined;
        const stops = [_]i32{0};
        const h = f.open(text.items.ptr, text.items.len, vocab.len, &stops, 1, &err, err.len) orelse return error.GrammarSetup;
        return .{ .f = f, .h = h };
    }

    /// Whether the grammar takes `reply` a character a token and may then stop.
    fn admits(l: Lib, g: *anyopaque, reply: []const u8) !bool {
        const m = l.f.matcher(g) orelse return error.Matcher;
        defer l.f.matcher_free(m);
        var i: usize = 0;
        while (i < reply.len) {
            const n = try std.unicode.utf8ByteSequenceLength(reply[i]);
            const id = for (vocab[1..], 1..) |v, t| {
                if (std.mem.eql(u8, v, reply[i..][0..n])) break t;
            } else return error.NotInVocab;
            switch (l.f.accept(m, @intCast(id))) {
                1 => {},
                0 => return false,
                else => return error.Accept,
            }
            i += n;
        }
        var bits: [words]u32 = @splat(0);
        if (l.f.fill(m, @ptrCast(&bits), words) != 0) return error.Fill;
        return bits[0] & 1 != 0;
    }
};

const Case = struct { style: tool_grammar.Style, reply: []const u8, ok: bool, what: []const u8 };

fn dsml(comptime name: []const u8, comptime params: []const u8) []const u8 {
    return "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"" ++ name ++ "\">\n" ++ params ++ "</｜DSML｜invoke>\n</｜DSML｜function_calls>";
}

fn param(comptime key: []const u8, comptime string: bool, comptime v: []const u8) []const u8 {
    return "<｜DSML｜ parameter name=\"" ++ key ++ "\" string=\"" ++ (if (string) "true" else "false") ++ "\">" ++ v ++ "</｜DSML｜ parameter>\n";
}

fn hermes(comptime name: []const u8, comptime args: []const u8) []const u8 {
    return "<tool_call>\n{\"name\": \"" ++ name ++ "\", \"arguments\": " ++ args ++ "}\n</tool_call>";
}

fn qwen(comptime name: []const u8, comptime params: []const u8) []const u8 {
    return "<tool_call>\n<function=" ++ name ++ ">\n" ++ params ++ "</function>\n</tool_call>";
}

fn qp(comptime key: []const u8, comptime v: []const u8) []const u8 {
    return "<parameter=" ++ key ++ ">\n" ++ v ++ "\n</parameter>\n";
}

const nested = "{\"a\":[1,{\"b\":null}]}";

const cases = [_]Case{
    .{ .style = .deepseek, .ok = true, .what = "now tz", .reply = dsml("now", param("tz", true, "UTC")) },
    .{ .style = .deepseek, .ok = true, .what = "now after text", .reply = "Let me check.\n\n" ++ dsml("now", param("tz", true, "UTC")) },
    .{ .style = .deepseek, .ok = true, .what = "now nested", .reply = dsml("now", param("tz", true, "UTC") ++ param("opts", false, nested)) },
    .{ .style = .deepseek, .ok = true, .what = "lookup valid", .reply = dsml("lookup", param("q", true, "hi") ++ param("n", false, "2")) },
    .{ .style = .deepseek, .ok = false, .what = "lookup n as text", .reply = dsml("lookup", param("q", true, "hi") ++ param("n", true, "two")) },
    .{ .style = .deepseek, .ok = false, .what = "lookup extra", .reply = dsml("lookup", param("q", true, "hi") ++ param("n", false, "2") ++ param("x", true, "1")) },
    .{ .style = .deepseek, .ok = false, .what = "lookup no n", .reply = dsml("lookup", param("q", true, "hi")) },
    .{ .style = .json, .ok = true, .what = "now tz", .reply = hermes("now", "{\"tz\": \"UTC\"}") },
    .{ .style = .json, .ok = true, .what = "now nested", .reply = hermes("now", "{\"tz\": \"UTC\", \"opts\": " ++ nested ++ "}") },
    .{ .style = .json, .ok = true, .what = "lookup valid", .reply = hermes("lookup", "{\"q\": \"hi\", \"n\": 2}") },
    .{ .style = .json, .ok = false, .what = "lookup n as text", .reply = hermes("lookup", "{\"q\": \"hi\", \"n\": \"two\"}") },
    .{ .style = .json, .ok = false, .what = "lookup extra", .reply = hermes("lookup", "{\"q\": \"hi\", \"n\": 2, \"x\": 1}") },
    .{ .style = .json, .ok = false, .what = "lookup no n", .reply = hermes("lookup", "{\"q\": \"hi\"}") },
    .{ .style = .xml, .ok = true, .what = "now tz", .reply = qwen("now", qp("tz", "UTC")) },
    .{ .style = .xml, .ok = true, .what = "now nested", .reply = qwen("now", qp("tz", "UTC") ++ qp("opts", nested)) },
    .{ .style = .xml, .ok = true, .what = "lookup valid", .reply = qwen("lookup", qp("q", "hi") ++ qp("n", "2")) },
    .{ .style = .xml, .ok = false, .what = "lookup n as text", .reply = qwen("lookup", qp("q", "hi") ++ qp("n", "two")) },
    .{ .style = .xml, .ok = false, .what = "lookup extra", .reply = qwen("lookup", qp("q", "hi") ++ qp("n", "2") ++ qp("x", "1")) },
    .{ .style = .xml, .ok = false, .what = "lookup no n", .reply = qwen("lookup", qp("q", "hi")) },
};

test "auto tools through xgrammar: non-strict now takes any object, strict lookup only its schema" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const l = try Lib.open(a);
    defer l.f.close(l.h);
    const tools = (try json.parse(a, offered)).ok.array;
    var wrong: usize = 0;
    for ([_]bool{ false, true }) |parallel| for (cases) |c| {
        const text = try tool_grammar.optional(a, c.style, tools, false, parallel);
        var err: [512]u8 = undefined;
        err[0] = 0;
        const g = l.f.compile(l.h, @backingInt(lanes.grammar.Kind.structural_tag), text.ptr, text.len, &err, err.len) orelse {
            std.debug.print("compile {s}: {s}\n", .{ @tagName(c.style), std.mem.sliceTo(&err, 0) });
            return error.Grammar;
        };
        defer l.f.free(g);
        const got = try l.admits(g, c.reply);
        if (got == c.ok) continue;
        std.debug.print("{s} {s} (parallel {}): want {}, got {}\n", .{ @tagName(c.style), c.what, parallel, c.ok, got });
        wrong += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), wrong);
}

/// One object of `count` optional string properties: xgrammar's compile time grows faster than the square of `count`.
fn optionalObject(a: std.mem.Allocator, count: usize) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "{\"type\":\"object\",\"properties\":{");
    for (0..count) |i| try text.print(a, "{s}\"p{d}\":{{\"type\":\"string\"}}", .{ if (i > 0) "," else "", i });
    try text.appendSlice(a, "}}");
    return text.items;
}

fn now() i64 {
    return std.Io.Clock.awake.now(std.testing.io).toMilliseconds();
}

const Schema = struct {
    l: Lib,
    text: []const u8,
    started: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    g: ?*anyopaque = null,
    err: [512]u8 = @splat(0),
    ms: i64 = 0,

    fn compile(s: *Schema) void {
        s.started.store(true, .release);
        const began = now();
        s.g = s.l.f.compile(s.l.h, @backingInt(lanes.grammar.Kind.json_schema), s.text.ptr, s.text.len, &s.err, s.err.len);
        s.ms = now() - began;
        s.done.store(true, .release);
    }
};

test "grammar compile through xgrammar: 8,000 optional properties are refused at the time limit, others compile meanwhile" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const l = try Lib.open(a);
    defer l.f.close(l.h);
    const json_schema = @backingInt(lanes.grammar.Kind.json_schema);
    var large: Schema = .{ .l = l, .text = try optionalObject(a, 8000) };
    try std.testing.expectEqual(@as(usize, 206_922), large.text.len);
    const thread = try std.Thread.spawn(.{}, Schema.compile, .{&large});
    const small_ok, const small_ms = blk: {
        while (!large.started.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
        const small = "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}},\"required\":[\"x\"]}";
        var err: [512]u8 = @splat(0);
        const began = now();
        const g = l.f.compile(l.h, json_schema, small.ptr, small.len, &err, err.len);
        defer if (g) |done| l.f.free(done);
        break :blk .{ g != null, now() - began };
    };
    const meanwhile = !large.done.load(.acquire);
    thread.join();
    if (large.g) |g| l.f.free(g);
    try std.testing.expect(small_ok and meanwhile and small_ms < 1000);
    try std.testing.expect(large.g == null and large.ms < 3000);
    try std.testing.expect(std.mem.indexOf(u8, std.mem.sliceTo(&large.err, 0), "2000 ms time limit") != null);
    // xgrammar cannot stop it: it runs on in its slot, and asking again joins it instead of compiling it twice
    const compiling = l.f.compiling orelse return error.MissingSymbol;
    try std.testing.expectEqual(@as(c_int, 1), compiling(l.h));
    var err: [512]u8 = @splat(0);
    const again = now();
    try std.testing.expect(l.f.compile(l.h, json_schema, large.text.ptr, large.text.len, &err, err.len) == null);
    try std.testing.expect(now() - again < 500 and std.mem.indexOf(u8, std.mem.sliceTo(&err, 0), "time limit") != null);
    try std.testing.expectEqual(@as(c_int, 1), compiling(l.h));
    const waited = now();
    while (compiling(l.h) != 0) {
        if (now() - waited > 600_000) return error.CompileNeverReturned;
        try std.Io.sleep(io, .fromMilliseconds(50), .awake);
    }
    // once it returns its slot is free, and xgrammar's cache serves the grammar it finished
    const cached = now();
    const g = l.f.compile(l.h, json_schema, large.text.ptr, large.text.len, &err, err.len) orelse return error.Grammar;
    defer l.f.free(g);
    try std.testing.expect(now() - cached < 1000);
    try std.testing.expect(try l.admits(g, "{}"));
}
