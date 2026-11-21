//! The tokenizer regex against what the tokenizers library's Oniguruma finds, and the limits a pattern is held to.
const std = @import("std");
const regex = @import("regex.zig");
const Regex = regex.Regex;
const Matcher = regex.Matcher;

fn splitAll(re: *const Regex, s: []const u8) ![]const []const u8 {
    const a = std.testing.allocator;
    var m = Matcher{ .a = a };
    defer m.deinit();
    var spans: std.ArrayList([2]usize) = .empty;
    defer spans.deinit(a);
    try re.findAll(&m, s, &spans);
    var out: std.ArrayList([]const u8) = .empty;
    for (spans.items) |span| try out.append(a, s[span[0]..span[1]]);
    return out.toOwnedSlice(a);
}

test "qwen pre-tokenizer pattern splits like Oniguruma" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const re = try Regex.compile(arena.allocator(), "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
    const got = try splitAll(&re, "It'S x'\u{17F} 12 caf\u{E9}  \n\n  hi!!\r\n  ");
    defer std.testing.allocator.free(got);
    const want = [_][]const u8{ "It", "'S", " x", "'\u{17F}", " ", "1", "2", " caf\u{E9}", "  \n\n", " ", " hi", "!!\r\n", "  " };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
}

test "LFM2's pre-tokenizer pattern folds its (?i:) class as the tokenizers library splits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const re = try Regex.compile(arena.allocator(), "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s");
    const got = try splitAll(&re, "It'S x'\u{17F} we'LL they'Ve 1234 caf\u{E9}  \n\n  hi!!\r\n  ");
    defer std.testing.allocator.free(got);
    const want = [_][]const u8{ "It", "'S", " x", "'\u{17F}", " we", "'LL", " they", "'Ve", " ", "123", "4", " caf\u{E9}", "  \n\n", " ", " hi", "!!\r\n", "  " };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
    for ([_][]const u8{ "(?i:[^a])", "(?i:[\\p{L}])", "(?i:[\u{E9}])" }) |bad|
        try std.testing.expectError(error.UnsupportedRegex, Regex.compile(arena.allocator(), bad));
}

test "Ling's pre-tokenizer pattern: possessive quantifiers never give back, as the tokenizers library matches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ling = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}|" ++
        " ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
    const re = try Regex.compile(a, ling);
    const got = try splitAll(&re, "It'S x'\u{17F} we'LL they'Ve 1234 caf\u{E9}  \n\n  hi!!\r\n  ");
    defer std.testing.allocator.free(got);
    const want = [_][]const u8{
        "It", "'S", " x", "'\u{17F}",   " we",    "'LL", " they", "'Ve",    " ",  "1",
        "2",  "3",  "4",  " caf\u{E9}", "  \n\n", " ",   " hi",   "!!\r\n", "  ",
    };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
    // where giving back would match more: what the tokenizers library's Oniguruma finds
    const Case = struct { pattern: []const u8, text: []const u8, found: []const []const u8 };
    const cases = [_]Case{
        .{ .pattern = "a*+a", .text = "aaa", .found = &.{} },
        .{ .pattern = "[0-9]?+5", .text = "15 5", .found = &.{"15"} },
        .{ .pattern = "b?+\\p{L}+", .text = "bb b1", .found = &.{"bb"} },
        .{ .pattern = "(?:ab)++ab", .text = "abab ab", .found = &.{} },
        .{ .pattern = "(?:a++)+a", .text = "aaa", .found = &.{} },
        // the atomic step keeps its body's first match, and only the quantified element is atomic
        .{ .pattern = "(?:a|ab)?+b", .text = "ab", .found = &.{"ab"} },
        .{ .pattern = "a?a?+a", .text = "aa", .found = &.{"aa"} },
        // Ruby syntax reads `{n,m}+` as two quantifiers, so this one gives back
        .{ .pattern = "a{1,2}+a", .text = "aaaa", .found = &.{"aaaa"} },
        // a plain lookahead still matches empty: only the atomic step keeps its match
        .{ .pattern = "a(?=b)", .text = "ab ab", .found = &.{ "a", "a" } },
    };
    for (cases) |c| {
        const found = try splitAll(&try Regex.compile(a, c.pattern), c.text);
        defer std.testing.allocator.free(found);
        try std.testing.expectEqual(c.found.len, found.len);
        for (c.found, found) |w, g| try std.testing.expectEqualStrings(w, g);
    }
    // an unbounded repeat of what can match empty is refused, possessive inside or out, as `(a*)*` is
    try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, "(?:a*+)*"));
    try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, "(?:a?)++"));
}

test "counted repetition, ranges and refusals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const digits = try Regex.compile(a, "\\p{N}{1,3}");
    const got = try splitAll(&digits, "a12345b6");
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(usize, 3), got.len);
    try std.testing.expectEqualStrings("45", got[1]);
    const kana = try Regex.compile(a, "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}]+|[!\"#\\-./\\[\\\\\\]^_`{|}~][A-Za-z]+");
    const k = try splitAll(&kana, "x\u{3042}\u{4e00}y-abc");
    defer std.testing.allocator.free(k);
    try std.testing.expectEqualStrings("\u{3042}\u{4e00}", k[0]);
    try std.testing.expectEqualStrings("-abc", k[1]);
    for ([_][]const u8{ "\\w+", "(?i:ss)", "a(?<=b)", "\\bx", "(?i)a", "[a[b]]", "(a*)*", "\\p{Latin}", "(?>a)" }) |bad|
        try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, bad));
}

test "security: a pattern that backtracks without end stops at its text's allowance" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // every way to cut a run of 64 into groups is tried before `b` is found missing: 2^63 ways
    const re = try Regex.compile(a, "(?:a+)+b");
    const run: [64]u8 = @splat('a');
    var m = Matcher{ .a = a };
    var out: std.ArrayList([2]usize) = .empty;
    try std.testing.expectError(error.RegexBudget, re.findAll(&m, &run, &out));
    try std.testing.expectError(error.RegexBudget, re.find(&m, &run, 0));
    // a text the same pattern can decide is split as before, and the next search starts with its own allowance
    const got = try splitAll(&re, "aaab aab");
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("aaab", got[0]);
    try std.testing.expectEqualStrings("aab", got[1]);
}

test "security: a pattern nested, repeated or quantified past the compiler's limits is refused, not compiled" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // groups: 64 deep compiles, one more does not, and 100,000 do not run the parser off its stack
    for ([_]usize{ 64, 65, 100_000 }) |depth| {
        const pattern = try a.alloc(u8, 2 * depth + 1);
        @memset(pattern[0..depth], '(');
        pattern[depth] = 'a';
        @memset(pattern[depth + 1 ..], ')');
        if (depth <= 64) {
            _ = try Regex.compile(a, pattern);
        } else try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, pattern));
    }
    // counted repeats multiply the program: 10,000 instructions fit, a billion do not
    _ = try Regex.compile(a, "(?:a{100}){100}");
    try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, "(?:(?:a{1000}){1000}){1000}"));
    // quantifiers on one element: eight compile, a ninth does not
    _ = try Regex.compile(a, "a{1}{1}{1}{1}{1}{1}{1}{1}");
    try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, "a{1}{1}{1}{1}{1}{1}{1}{1}{1}"));
}

test "the split patterns checkpoints ship use a small part of their allowance on long runs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const patterns = [_][]const u8{
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    };
    const units = [_][]const u8{ " ", "a", "7", "!", "\n", " \n", "a b ", "\u{4e2d}", "'s't", "    x", "a\u{301}", "! " };
    for (patterns) |pattern| {
        const re = try Regex.compile(a, pattern);
        for (units) |unit| {
            const text = try a.alloc(u8, 8192 - 8192 % unit.len);
            for (0..text.len / unit.len) |i| @memcpy(text[i * unit.len ..][0..unit.len], unit);
            var m = Matcher{ .a = a };
            var out: std.ArrayList([2]usize) = .empty;
            try re.findAll(&m, text, &out);
            const allowed = regex.work_base + regex.work_per_byte * text.len;
            try std.testing.expect((allowed - m.left) * 16 < allowed);
        }
    }
}
