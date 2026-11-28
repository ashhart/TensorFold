//! response_format structural_tag through a real libtfgrammar (zig build test-grammar-tools), a character a token.
const std = @import("std");
const lanes = @import("lanes");
const json = @import("json");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const Lib = @import("tool_grammar_xgrammar_test.zig").Lib;

const weather = "{\"begin\":\"<weather>\",\"schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"additionalProperties\":false},\"end\":\"</weather>\"}";
const time = "{\"begin\":\"<time>\",\"schema\":{\"type\":\"object\",\"properties\":{\"tz\":{\"type\":\"string\"}},\"required\":[\"tz\"],\"additionalProperties\":false},\"end\":\"</time>\"}";
const wide = "{\"begin\":\"<w\",\"schema\":{\"type\":\"object\"},\"end\":\"</w>\"}";
const head = "{\"response_format\":{\"type\":\"structural_tag\",\"structures\":[";
const both = head ++ weather ++ "," ++ time ++ "]}}";
const angle = head ++ weather ++ "," ++ time ++ "],\"triggers\":[\"<\"]}}";
const nested = head ++ weather ++ "," ++ wide ++ "]}}";

const oslo = "<weather>{\"city\":\"Oslo\"}</weather>";
const utc = "<time>{\"tz\":\"UTC\"}</time>";

const Case = struct { body: []const u8, reply: []const u8 = "", tokens: ?[]const []const u8 = null, ok: bool, what: []const u8 };

const cases = [_]Case{
    .{ .body = both, .ok = true, .what = "no tags: free text alone", .reply = "No tool needed today." },
    .{ .body = both, .ok = true, .what = "free text outside tags is unconstrained", .reply = "Braces {not: json}, [1, 2] and \"quotes\" are fine." },
    .{ .body = both, .ok = true, .what = "a lone < and a begin cut short are free text", .reply = "1 < 2, and <weathe is no tag." },
    .{ .body = both, .ok = true, .what = "one region", .reply = "Checking. " ++ oslo ++ " Done." },
    .{ .body = both, .ok = true, .what = "several regions with text between", .reply = oslo ++ "\nand\n" ++ utc ++ "\n" ++ oslo },
    .{ .body = both, .ok = true, .what = "a few blanks inside a region's JSON", .reply = "<weather>{ \"city\" : \"Oslo\" }</weather>" },
    .{ .body = both, .ok = false, .what = "a run of 33 blanks inside a region's JSON", .reply = "<weather>{" ++ @as([33]u8, @splat(' ')) ++ "\"city\":\"Oslo\"}</weather>" },
    .{ .body = both, .ok = false, .what = "a region against its schema: wrong key", .reply = "<weather>{\"town\":\"Oslo\"}</weather>" },
    .{ .body = both, .ok = false, .what = "a region against its schema: not JSON", .reply = "<weather>sunny</weather>" },
    .{ .body = both, .ok = false, .what = "a region against its schema: the other structure's", .reply = "<time>{\"city\":\"Oslo\"}</time>" },
    .{ .body = both, .ok = false, .what = "text between a region's JSON and its end", .reply = "<weather>{\"city\":\"Oslo\"} ok</weather>" },
    .{ .body = both, .ok = false, .what = "a region left open", .reply = "<weather>{\"city\":\"Oslo\"}" },
    .{ .body = both, .ok = true, .what = "multi-character tokens across every edge of a region", .tokens = &.{ "C", "h", "e", "c", "k", "i", "ng <wea", "ther>{\"ci", "ty\":\"Oslo\"}</wea", "ther> ok" } },
    .{ .body = both, .ok = false, .what = "a begin split across tokens still opens its region", .tokens = &.{ "ng <wea", "ther> ok" } },
    .{ .body = angle, .ok = true, .what = "a shared trigger opens either structure", .reply = "a " ++ utc ++ " b " ++ oslo },
    .{ .body = angle, .ok = false, .what = "a trigger in free text must open a structure", .reply = "a < b" },
    .{ .body = nested, .ok = true, .what = "a begin another begin starts: both open", .reply = "<w{}</w> and " ++ oslo },
};

test "structural_tag through xgrammar: text free outside tags, each region held to its schema until its end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const l = try Lib.open(a);
    defer l.f.close(l.h);
    var wrong: usize = 0;
    for (cases) |c| {
        var cx: errors.Cx = .{ .a = a };
        const s = (try grammar.spec(&cx, (try json.parse(a, c.body)).ok)).?;
        var err: [512]u8 = undefined;
        err[0] = 0;
        const g = l.f.compile(l.h, @backingInt(lanes.grammar.Kind.structural_tag), s.text.ptr, s.text.len, &err, err.len) orelse {
            std.debug.print("compile {s}: {s}\n", .{ c.what, std.mem.sliceTo(&err, 0) });
            return error.Grammar;
        };
        defer l.f.free(g);
        const got = if (c.tokens) |t| try l.takes(g, t) else try l.admits(g, c.reply);
        if (got == c.ok) continue;
        std.debug.print("{s}: want {}, got {}\n", .{ c.what, c.ok, got });
        wrong += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), wrong);
}
