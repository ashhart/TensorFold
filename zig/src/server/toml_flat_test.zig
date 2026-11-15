//! toml_flat.zig: what the flat TOML reader accepts, how it decodes it, and each refusal with its line.
const std = @import("std");
const toml = @import("toml_flat.zig");
const t = std.testing;

fn read(a: std.mem.Allocator, text: []const u8) ![]const toml.Entry {
    var problem: toml.Problem = .{};
    return toml.parse(a, text, &problem) catch |e| {
        std.debug.print("refused at line {d}: {s}\n", .{ problem.line, problem.message });
        return e;
    };
}

/// `text` is refused on `line` with a message that holds `needle`.
fn refused(a: std.mem.Allocator, text: []const u8, line: u32, needle: []const u8) !void {
    var problem: toml.Problem = .{};
    try t.expectError(error.Invalid, toml.parse(a, text, &problem));
    errdefer std.debug.print("text: {s}\nline {d}: {s}\n", .{ text, problem.line, problem.message });
    try t.expectEqual(line, problem.line);
    try t.expect(std.mem.indexOf(u8, problem.message, needle) != null);
}

test "flat TOML: scalars, comments and line ends decode in file order" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try read(a,
        \\# a launch
        \\model = "org/Model"   # trailing comment
        \\
        \\port=8080
        \\  temperature   =   0.6
        \\thinking = false
        \\dashboard = true
        \\path = 'C:\raw\n'
        \\"quoted key" = "a # not a comment"
        \\'lit-key' = ""
    );
    try t.expectEqual(@as(usize, 8), entries.len);
    try t.expectEqualStrings("model", entries[0].key);
    try t.expectEqual(@as(u32, 2), entries[0].line);
    try t.expectEqualStrings("org/Model", entries[0].value.scalar.string);
    try t.expectEqualStrings("8080", entries[1].value.scalar.integer);
    try t.expectEqual(@as(u32, 4), entries[1].line);
    try t.expectEqualStrings("0.6", entries[2].value.scalar.float);
    try t.expect(!entries[3].value.scalar.boolean);
    try t.expect(entries[4].value.scalar.boolean);
    try t.expectEqualStrings("C:\\raw\\n", entries[5].value.scalar.string);
    try t.expectEqualStrings("quoted key", entries[6].key);
    try t.expectEqualStrings("a # not a comment", entries[6].value.scalar.string);
    try t.expectEqualStrings("lit-key", entries[7].key);
    try t.expectEqualStrings("", entries[7].value.scalar.string);
    try t.expectEqual(@as(usize, 0), (try read(a, "")).len);
    try t.expectEqual(@as(usize, 0), (try read(a, "  # only a comment")).len);
    const crlf = try read(a, "a = 1\r\nb = [\r\n  \"2\",\r\n]\r\n# end\r\n");
    try t.expectEqual(@as(usize, 2), crlf.len);
    try t.expectEqual(@as(u32, 2), crlf[1].line);
    try t.expectEqualStrings("2", crlf[1].value.array[0]);
}

test "flat TOML: every escape of a basic string decodes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try read(a, "s = \"\\b\\t\\n\\f\\r\\\"\\\\ \\u00e9 \\U0001F600 tab\traw \xc3\xa9\"\n");
    try t.expectEqualStrings("\x08\t\n\x0c\r\"\\ \xc3\xa9 \xf0\x9f\x98\x80 tab\traw \xc3\xa9", entries[0].value.scalar.string);
}

test "flat TOML: numbers keep their text without underscores or an integer's plus" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { text: []const u8, want: []const u8, float: bool = false };
    for ([_]Case{
        .{ .text = "0", .want = "0" },
        .{ .text = "-0", .want = "-0" },
        .{ .text = "1_000", .want = "1000" },
        .{ .text = "+7", .want = "7" },
        .{ .text = "-17", .want = "-17" },
        .{ .text = "-0.5", .want = "-0.5", .float = true },
        .{ .text = "+1.0", .want = "+1.0", .float = true },
        .{ .text = "6.02e+23", .want = "6.02e+23", .float = true },
        .{ .text = "1e3", .want = "1e3", .float = true },
        .{ .text = "1E-03", .want = "1E-03", .float = true },
        .{ .text = "9_224.617_445e1_0", .want = "9224.617445e10", .float = true },
        .{ .text = "inf", .want = "inf", .float = true },
        .{ .text = "-nan", .want = "-nan", .float = true },
    }) |c| {
        const entries = try read(a, try std.fmt.allocPrint(a, "n = {s}\n", .{c.text}));
        const got = entries[0].value.scalar;
        try t.expectEqualStrings(c.want, if (c.float) got.float else got.integer);
    }
}

test "flat TOML: arrays take strings on one line or several, with comments and a trailing comma" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try read(a,
        \\empty = []
        \\one = ["a", 'b', "c, d]"]
        \\many = [ # the aliases
        \\  "x",
        \\
        \\  "y", # second
        \\]
        \\after = 1
    );
    try t.expectEqual(@as(usize, 0), entries[0].value.array.len);
    const one = entries[1].value.array;
    try t.expectEqual(@as(usize, 3), one.len);
    try t.expectEqualStrings("a", one[0]);
    try t.expectEqualStrings("b", one[1]);
    try t.expectEqualStrings("c, d]", one[2]);
    try t.expectEqual(@as(usize, 2), entries[2].value.array.len);
    try t.expectEqualStrings("y", entries[2].value.array[1]);
    try t.expectEqual(@as(u32, 3), entries[2].line);
    try t.expectEqual(@as(u32, 8), entries[3].line);
}

test "flat TOML: what the subset leaves out is refused by line and by name" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refused(a, "a = 1\n[server]\nb = 2\n", 2, "tables are not read");
    try refused(a, "[[servers]]\n", 1, "tables are not read");
    try refused(a, "a.b = 1\n", 1, "dotted keys");
    try refused(a, "\"a\" . b = 1\n", 1, "dotted keys");
    try refused(a, "a = { b = 1 }\n", 1, "inline tables");
    try refused(a, "a = [[1], [2]]\n", 1, "nested arrays");
    try refused(a, "a = [\n  { b = 1 },\n]\n", 2, "nested arrays and inline tables");
    try refused(a, "a = \"\"\"\nx\n\"\"\"\n", 1, "multi-line strings");
    try refused(a, "a = '''x'''\n", 1, "multi-line strings");
    for ([_][]const u8{ "0x10", "0o7", "0b1", "1979-05-27", "07:32:00", "1979-05-27T07:32:00Z", "01", "-01", "1__0", "1.", ".5", "1_", "_1", "1e", "1.5e", "1.e5", "1.2.3", "+", "Inf", "NaN", "True", "yes", "bare" }) |token|
        try refused(a, try std.fmt.allocPrint(a, "ok = 1\nn = {s}\n", .{token}), 2, "not a string, a decimal number, true or false");
}

test "flat TOML: malformed text is refused by line" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refused(a, "a = 1\nb = 2\na = 3\n", 3, "key defined twice");
    for ([_][]const u8{ "\"a\"", "'a'", "\"\\u0061\"" }) |spelling| // one key under each quoted spelling
        try refused(a, try std.fmt.allocPrint(a, "a = 1\n{s} = 2\n", .{spelling}), 2, "key defined twice");
    try refused(a, "a = \"open\nb = 1\n", 1, "unterminated string");
    try refused(a, "a = 'open", 1, "unterminated string");
    try refused(a, "a = [\"1\", \"2\"\nb = 3\n", 2, "expected ',' or ']'");
    try refused(a, "a = [\"1\",\n  \"2\",\n", 3, "unterminated array");
    try refused(a, "a = [\"1\" \"2\"]\n", 1, "expected ',' or ']'");
    for ([_][]const u8{ "[,]", "[1]", "[\"a\", 2.5]", "[true]", "[\"a\",\n\n  bare]" }) |list| // an array holds strings only
        try refused(a, try std.fmt.allocPrint(a, "a = {s}\n", .{list}), if (std.mem.indexOfScalar(u8, list, '\n') != null) 3 else 1, "an array holds strings only");
    try refused(a, "a = 1 2\n", 1, "unexpected text after the value");
    try refused(a, "a = \"x\" y\n", 1, "unexpected text after the value");
    try refused(a, "\n\na 1\n", 3, "expected '='");
    try refused(a, "a\n", 1, "expected '='");
    try refused(a, "a =\n", 1, "expected a value");
    try refused(a, "a = # nothing\n", 1, "expected a value");
    try refused(a, "= 1\n", 1, "expected a key");
    try refused(a, "a b = 1\n", 1, "expected '='");
    try refused(a, "a = \"\\q\"\n", 1, "unknown escape");
    try refused(a, "a = \"\\uD800\"\n", 1, "Unicode scalar value");
    try refused(a, "a = \"\\u12\"\n", 1, "Unicode scalar value");
    try refused(a, "a = \"\\u+123\"\n", 1, "Unicode scalar value");
    try refused(a, "a = \"\\U00110000\"\n", 1, "Unicode scalar value");
    try refused(a, "a = \"\\u00e", 1, "unterminated string");
    try refused(a, "a = \"bell\x07\"\n", 1, "control character in a string");
    try refused(a, "a = 'del\x7f'\n", 1, "control character in a string");
    try refused(a, "a = 1 # bell\x07\n", 1, "control character in a comment");
    try refused(a, "a = 1\r# x\n", 1, "carriage return without a line feed");
    try refused(a, "a = 1\n# x\ry\n", 2, "control character in a comment");
    try refused(a, "a = \x00\n", 1, "not a string, a decimal number, true or false");
    try refused(a, "\xef\xbb\xbfa = 1\n", 1, "byte order mark");
    try refused(a, "a = 1\nb = 2\nc = \"\xff\"\n", 3, "invalid UTF-8");
    try refused(a, "a = \"\xc3\"", 1, "invalid UTF-8");
}
