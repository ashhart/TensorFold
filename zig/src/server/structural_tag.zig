//! response_format structural_tag: free text; each structure's begin opens a region its schema holds to its end.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const node = @import("tool_grammar.zig").node;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The most structures one request may list, and the most triggers.
pub const most = 64;

/// The longest begin, end or trigger, in bytes.
pub const longest = 256;

/// The most grammar text the structures may compile to (their tags and schemas), in bytes.
pub const most_bytes = 1 << 20;

/// The most blanks a region's JSON takes between two tokens: the json_schema format's cap (tf_grammar.cc BLANKS).
const blanks = 32;

const field = "response_format structural_tag";
const no_structures = field ++ " needs structures: a non-empty list of {\"begin\", \"schema\", \"end\"}";
const bad_triggers = field ++ " triggers must be a non-empty list of non-empty strings";

/// xgrammar's structural tag for `rf` (triggered_tags: text, and a tag wherever a trigger starts), or its refusal.
pub fn build(cx: *Cx, rf: Value) errors.Refused![]const u8 {
    const a = cx.a;
    const list = rf.field("structures") orelse {
        if (rf.field("format") != null) return cx.refuse(field ++ " takes structures and triggers, not xgrammar's format object");
        return cx.refuse(no_structures);
    };
    if (list != .array or list.array.len == 0) return cx.refuse(no_structures);
    if (list.array.len > most) return cx.fail(.request, field ++ " takes at most {d} structures", .{most});
    const tags = try a.alloc(Value, list.array.len);
    const begins = try a.alloc([]const u8, list.array.len);
    const ws = try json.intValue(a, blanks);
    for (list.array, tags, begins, 0..) |s, *tag, *begin, i| {
        if (s != .object) return cx.fail(.request, field ++ " structures[{d}] must be an object with begin, schema and end", .{i});
        begin.* = try tagText(cx, s.get("begin"), i, "begin");
        const end = try tagText(cx, s.get("end"), i, "end");
        const raw = s.field("schema") orelse return cx.fail(.request, field ++ " structures[{d}] needs schema (a JSON schema object)", .{i});
        const where = try std.fmt.allocPrint(a, field ++ " structures[{d}].schema", .{i});
        const content = try node(a, "json_schema", &.{ .{ "json_schema", try grammar.schemaValue(cx, raw, where) }, .{ "max_whitespace_cnt", ws } });
        tag.* = try node(a, "tag", &.{ .{ "begin", .{ .string = begin.* } }, .{ "content", content }, .{ "end", .{ .string = end } } });
    }
    const format = try node(a, "triggered_tags", &.{
        .{ "triggers", if (rf.field("triggers")) |t| try triggers(cx, t, begins) else try defaults(a, begins) },
        .{ "tags", .{ .array = tags } },
        .{ "stop_after_first", .{ .bool = false } },
    });
    const text = try json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
    if (text.len > most_bytes) return cx.fail(.request, field ++ " compiles to {d} bytes of grammar, past the {d}-byte limit: send fewer or smaller schemas", .{ text.len, most_bytes });
    return text;
}

/// A structure's begin or end: a non-empty string of at most `longest` bytes.
fn tagText(cx: *Cx, v: ?Value, i: usize, key: []const u8) errors.Refused![]const u8 {
    const s = v orelse Value.null;
    if (s != .string or s.string.len == 0) return cx.fail(.request, field ++ " structures[{d}].{s} must be a non-empty string", .{ i, key });
    if (s.string.len > longest) return cx.fail(.request, field ++ " structures[{d}].{s} is {d} bytes, past the {d}-byte limit", .{ i, key, s.string.len, longest });
    return s.string;
}

/// Without triggers: each distinct begin no shorter begin starts, so every begin starts with exactly one trigger.
fn defaults(a: Allocator, begins: []const []const u8) Allocator.Error!Value {
    var out: std.ArrayList(Value) = .empty;
    for (begins, 0..) |b, i| {
        const held = for (begins, 0..) |x, j| {
            if (std.mem.startsWith(u8, b, x) and (x.len < b.len or j < i)) break true;
        } else false;
        if (!held) try out.append(a, .{ .string = b });
    }
    return .{ .array = out.items };
}

/// Given triggers: distinct, each starting some begin, and every begin starting with exactly one (as xgrammar needs).
fn triggers(cx: *Cx, v: Value, begins: []const []const u8) errors.Refused!Value {
    if (v != .array or v.array.len == 0) return cx.refuse(bad_triggers);
    if (v.array.len > most) return cx.fail(.request, field ++ " takes at most {d} triggers", .{most});
    for (v.array, 0..) |t, k| {
        if (t != .string or t.string.len == 0) return cx.refuse(bad_triggers);
        if (t.string.len > longest) return cx.fail(.request, field ++ " triggers[{d}] is {d} bytes, past the {d}-byte limit", .{ k, t.string.len, longest });
        for (v.array[0..k]) |x| if (std.mem.eql(u8, x.string, t.string)) return cx.fail(.request, field ++ " trigger {s} is listed twice", .{t.string});
        const used = for (begins) |b| {
            if (std.mem.startsWith(u8, b, t.string)) break true;
        } else false;
        if (!used) return cx.fail(.request, field ++ " trigger {s} starts no structure's begin", .{t.string});
    }
    for (begins, 0..) |b, i| {
        var first: ?[]const u8 = null;
        for (v.array) |t| {
            if (!std.mem.startsWith(u8, b, t.string)) continue;
            if (first) |f| return cx.fail(.request, field ++ " structures[{d}].begin {s} starts with two triggers, {s} and {s}", .{ i, b, f, t.string });
            first = t.string;
        }
        if (first == null) return cx.fail(.request, field ++ " structures[{d}].begin {s} starts with none of the triggers", .{ i, b });
    }
    return v;
}

fn specOf(cx: *Cx, body: []const u8) errors.Refused!?grammar.Spec {
    return grammar.spec(cx, (try json.parse(cx.a, body)).ok);
}

const weather = "{\"begin\":\"<weather>\",\"schema\":{\"type\":\"object\",\"required\":[\"city\"]},\"end\":\"</weather>\"}";
const time = "{\"begin\":\"<time>\",\"schema\":\"{\\\"type\\\": \\\"object\\\"}\",\"end\":\"</time>\"}";
const short = "{\"begin\":\"<w\",\"schema\":{\"type\":\"object\"},\"end\":\"</w\"}";
const head = "{\"response_format\":{\"type\":\"structural_tag\"";

test "structural_tag: one region, several regions, default and given triggers, in xgrammar's triggered_tags form" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const cases = [_]struct { body: []const u8, triggers: []const []const u8, begins: []const []const u8 }{
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "]}}", .triggers = &.{"<weather>"}, .begins = &.{"<weather>"} },
        // a begin listed twice is one trigger; a schema given as JSON text is parsed
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "," ++ time ++ "," ++ weather ++ "]}}", .triggers = &.{ "<weather>", "<time>" }, .begins = &.{ "<weather>", "<time>", "<weather>" } },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "," ++ time ++ "],\"triggers\":[\"<\"]}}", .triggers = &.{"<"}, .begins = &.{ "<weather>", "<time>" } },
        // a begin a shorter begin starts is no trigger of its own: xgrammar takes one trigger a begin
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "," ++ short ++ "],\"triggers\":null}}", .triggers = &.{"<w"}, .begins = &.{ "<weather>", "<w" } },
    };
    for (cases) |c| {
        const s = (try specOf(&cx, c.body)).?;
        try std.testing.expectEqual(grammar.Kind.structural_tag, s.kind);
        try std.testing.expectEqualStrings("response_format", s.field);
        const v = (try json.parse(cx.a, s.text)).ok;
        try std.testing.expectEqualStrings("structural_tag", v.strField("type").?);
        const f = v.get("format").?;
        try std.testing.expectEqualStrings("triggered_tags", f.strField("type").?);
        try std.testing.expect(!f.get("stop_after_first").?.bool); // regions may follow one another
        const ts = f.get("triggers").?.array;
        try std.testing.expectEqual(c.triggers.len, ts.len);
        for (ts, c.triggers) |t, want| try std.testing.expectEqualStrings(want, t.string);
        const tags = f.get("tags").?.array;
        try std.testing.expectEqual(c.begins.len, tags.len);
        for (tags, c.begins) |t, b| {
            try std.testing.expectEqualStrings("tag", t.strField("type").?);
            try std.testing.expectEqualStrings(b, t.strField("begin").?);
            try std.testing.expectEqualStrings(try std.mem.concat(cx.a, u8, &.{ "</", b[1..] }), t.strField("end").?);
            const content = t.get("content").?;
            try std.testing.expectEqualStrings("json_schema", content.strField("type").?);
            try std.testing.expectEqualStrings("object", content.get("json_schema").?.strField("type").?);
            try std.testing.expectEqual(@as(?i64, blanks), content.get("max_whitespace_cnt").?.int64());
        }
    }
}

test "structural_tag: an invalid region or trigger list is refused with what is wrong" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const long = "<" ++ @as([longest]u8, @splat('a')) ++ ">";
    var many: std.ArrayList(u8) = .empty;
    try many.appendSlice(a, head ++ ",\"structures\":[" ++ weather);
    for (0..most) |_| try many.appendSlice(a, "," ++ weather);
    try many.appendSlice(a, "]}}");
    var lots: std.ArrayList(u8) = .empty;
    try lots.appendSlice(a, head ++ ",\"structures\":[" ++ weather ++ "],\"triggers\":[\"<weather>\"");
    for (0..most) |k| try lots.print(a, ",\"<w{d}\"", .{k});
    try lots.appendSlice(a, "]}}");
    const filler = try a.alloc(u8, most_bytes);
    @memset(filler, 'x');
    const big = try std.mem.concat(a, u8, &.{ head, ",\"structures\":[{\"begin\":\"<a>\",\"schema\":{\"description\":\"", filler, "\"},\"end\":\"</a>\"}]}}" });
    const refused = [_]struct { body: []const u8, says: []const u8 }{
        .{ .body = head ++ "}}", .says = "needs structures" },
        .{ .body = head ++ ",\"structures\":[]}}", .says = "needs structures" },
        .{ .body = head ++ ",\"format\":{\"type\":\"any_text\"}}}", .says = "not xgrammar's format object" },
        .{ .body = head ++ ",\"structures\":[\"<a>\"]}}", .says = "structures[0] must be an object" },
        .{ .body = head ++ ",\"structures\":[{\"begin\":\"\",\"schema\":{},\"end\":\"</a>\"}]}}", .says = "structures[0].begin must be" },
        .{ .body = head ++ ",\"structures\":[{\"begin\":\"<a>\",\"schema\":{}}]}}", .says = "structures[0].end must be" },
        .{ .body = head ++ ",\"structures\":[{\"begin\":\"" ++ long ++ "\",\"schema\":{},\"end\":\"</a>\"}]}}", .says = "structures[0].begin is 258 bytes, past the 256-byte limit" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ ",{\"begin\":\"<a>\",\"end\":\"</a>\"}]}}", .says = "structures[1] needs schema" },
        .{ .body = head ++ ",\"structures\":[{\"begin\":\"<a>\",\"schema\":\"{nope\",\"end\":\"</a>\"}]}}", .says = "structures[0].schema is not valid JSON" },
        .{ .body = head ++ ",\"structures\":[{\"begin\":\"<a>\",\"schema\":[1],\"end\":\"</a>\"}]}}", .says = "structures[0].schema must be a JSON schema object" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "],\"triggers\":[]}}", .says = "triggers must be a non-empty list" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "],\"triggers\":[\"<weather>\",\"[\"]}}", .says = "trigger [ starts no structure" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "," ++ time ++ "],\"triggers\":[\"<w\"]}}", .says = "structures[1].begin <time> starts with none" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "],\"triggers\":[\"<\",\"<\"]}}", .says = "trigger < is listed twice" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "],\"triggers\":[\"<\",\"<w\"]}}", .says = "structures[0].begin <weather> starts with two triggers, < and <w" },
        .{ .body = head ++ ",\"structures\":[" ++ weather ++ "],\"triggers\":[\"" ++ long ++ "\"]}}", .says = "triggers[0] is 258 bytes" },
        .{ .body = many.items, .says = "takes at most 64 structures" },
        .{ .body = lots.items, .says = "takes at most 64 triggers" },
        .{ .body = big, .says = "past the 1048576-byte limit" },
    };
    for (refused) |c| {
        var cx: Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, specOf(&cx, c.body));
        std.testing.expect(std.mem.indexOf(u8, cx.message, c.says) != null) catch |e| {
            std.debug.print("want {s}, got {s}\n", .{ c.says, cx.message });
            return e;
        };
    }
}

test "structural_tag: requests without it get the spec they get today" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    try std.testing.expect(try specOf(&cx, "{\"messages\":[]}") == null);
    try std.testing.expect(try specOf(&cx, "{\"response_format\":{\"type\":\"text\"}}") == null);
    try std.testing.expectEqual(grammar.Kind.json, (try specOf(&cx, "{\"response_format\":{\"type\":\"json_object\"}}")).?.kind);
    // a json_schema's text is the schema stringified, as before schemaValue was split out of schemaText
    for ([_][]const u8{ "{\"type\":\"object\",\"properties\":{\"é\":{}}}", "\"{\\\"type\\\": \\\"object\\\"}\"", "true" }) |schema| {
        const body = try std.mem.concat(cx.a, u8, &.{ "{\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":", schema, "}}}" });
        const s = (try specOf(&cx, body)).?;
        const raw = (try json.parse(cx.a, schema)).ok;
        const value = if (raw == .string) (try json.parse(cx.a, raw.string)).ok else if (raw == .bool) Value{ .object = try json.newObject(cx.a) } else raw;
        try std.testing.expectEqualStrings(try json.stringify(cx.a, value, .{ .ascii = false }), s.text);
    }
    try std.testing.expectError(error.Refused, specOf(&cx, "{\"response_format\":{\"type\":\"bogus\"}}"));
    try std.testing.expectEqualStrings("response_format type must be text, json_object or json_schema, not 'bogus'", cx.message);
}
