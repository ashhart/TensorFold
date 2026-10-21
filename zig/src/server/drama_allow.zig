//! Opt-in speech control filtering; callers own the arena used for changed input.
const std = @import("std");
const unicode = @import("tokenizer").unicode;
const case = @import("template").unicode;
const Allocator = std.mem.Allocator;

pub const Policy = struct { list: []const u8 = "" };

// The policy borrows LIST, as the other string-valued serve flags borrow their arguments.
pub fn parse(list: []const u8) error{InvalidControl}!Policy {
    if (!std.unicode.utf8ValidateSlice(list) or std.mem.indexOfAny(u8, list, "[]") != null) return error.InvalidControl;
    return .{ .list = std.mem.trim(u8, list, " \t\r\n") };
}

const max_control = 120;
const scoped_tags = [_][]const u8{ "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful" };
const Section = enum { cues, modifiers };
const vocabulary = vocabulary: {
    @setEvalBranchQuota(200_000);
    const Entry = struct { []const u8, Section };
    var entries: []const Entry = &.{};
    var section: ?Section = null;
    var lines = std.mem.tokenizeScalar(u8, @embedFile("drama_cues.txt"), '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "## ")) {
            section = std.meta.stringToEnum(Section, line[3..]) orelse @compileError("unknown section " ++ line);
        } else if (line[0] != '#') entries = entries ++ &[_]Entry{.{ line, section orelse @compileError("a word before any section") }};
    }
    break :vocabulary std.StaticStringMap(Section).initComptime(entries);
};

pub fn filter(a: Allocator, policy: ?Policy, input: []const u8) Allocator.Error![]const u8 {
    const active = policy orelse return input;
    // The request layer reports invalid UTF-8 after filtering; normalization needs valid code points.
    if (!std.unicode.utf8ValidateSlice(input)) return input;
    var names: std.ArrayList([]const u8) = .empty;
    var parts = std.mem.splitScalar(u8, active.list, ',');
    while (parts.next()) |part| {
        const name = try normalize(a, part);
        if (name.len > 0) try names.append(a, name);
    }
    return filterNames(a, input, names.items);
}

fn filterNames(a: Allocator, input: []const u8, names: []const []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var done: usize = 0;
    var stacked: ?usize = null;
    var text_until: usize = 0;
    var closes: ?[]const usize = null;
    var i: usize = 0;
    while (i < input.len) {
        const tag = tagAt(input[i..]);
        if (tag > 0) {
            try out.appendSlice(a, input[done..i]);
            done = drop(&out, input, i + tag);
            i = done;
            continue;
        }
        const end = (if (i < text_until) null else candidateAt(input, i, stacked)) orelse {
            i += 1;
            continue;
        };
        stacked = end;
        const inner = input[i + 1 .. end - 1];
        const name = try normalize(a, try untagged(a, inner));
        const listed = for (names) |kept| {
            if (std.mem.eql(u8, name, kept)) break true;
        } else false;
        if (listed and !std.mem.eql(u8, name, inner)) {
            try out.appendSlice(a, input[done..i]);
            try out.print(a, "[{s}]", .{name});
            done = end;
            i = done;
        } else if (!listed and isControl(name)) {
            try out.appendSlice(a, input[done..i]);
            done = drop(&out, input, end);
            i = done;
        } else {
            const close = (closes orelse made: {
                closes = try matching(a, input);
                break :made closes.?;
            })[i];
            // A stray '[' must not shield the rest of the input.
            text_until = if (close < input.len) close + 1 else end;
            stacked = text_until;
            i += 1;
        }
    }
    if (done == 0) return input;
    try out.appendSlice(a, input[done..]);
    return out.items;
}

fn matching(a: Allocator, input: []const u8) Allocator.Error![]const usize {
    const closes = try a.alloc(usize, input.len);
    @memset(closes, input.len);
    var open: std.ArrayList(usize) = .empty;
    for (input, 0..) |c, k| switch (c) {
        '[' => try open.append(a, k),
        ']' => if (open.pop()) |start| {
            closes[start] = k;
        },
        else => {},
    };
    return closes;
}

fn normalize(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    const lower = try case.lower(a, try unicode.normalize(a, text, .nfc));
    var out: std.ArrayList(u8) = .empty;
    var space = false;
    var it = std.unicode.Utf8View.initUnchecked(lower).iterator();
    while (it.nextCodepointSlice()) |bytes| {
        if (unicode.isSpace(std.unicode.utf8Decode(bytes) catch unreachable)) {
            space = out.items.len > 0;
            continue;
        }
        if (space) try out.append(a, ' ');
        space = false;
        try out.appendSlice(a, bytes);
    }
    var end = out.items.len;
    while (end > 0) {
        if (std.mem.indexOfScalar(u8, "!?. ", out.items[end - 1]) != null) {
            end -= 1;
        } else if (std.mem.endsWith(u8, out.items[0..end], "\u{2026}")) {
            end -= "\u{2026}".len;
        } else break;
    }
    return out.items[0..end];
}

fn isControl(name: []const u8) bool {
    var cue = false;
    var words = std.mem.tokenizeAny(u8, name, " ,-");
    while (words.next()) |word| switch (vocabulary.get(word) orelse return false) {
        .cues => cue = true,
        .modifiers => {},
    };
    return cue;
}

fn candidateAt(input: []const u8, i: usize, stacked: ?usize) ?usize {
    if (input[i] != '[') return null;
    if (i > 0 and stacked != i) {
        var start = i - 1;
        while (start > 0 and input[start] & 0xc0 == 0x80) start -= 1;
        const before = std.unicode.utf8Decode(input[start..i]) catch unreachable;
        if (before == ']' or before == '_' or before == '`') return null;
        if (unicode.bit(unicode.category(before)) & (unicode.letter | unicode.number) != 0) return null;
    }
    const close = std.mem.indexOfScalar(u8, input[i..@min(input.len, i + max_control)], ']') orelse return null;
    const end = i + close + 1;
    if (end < input.len and input[end] == '(') return null;
    return end;
}

fn tagAt(rest: []const u8) usize {
    if (rest[0] != '<') return 0;
    const open: usize = if (rest.len > 1 and rest[1] == '/') 2 else 1;
    for (scoped_tags) |tag| {
        const close = open + tag.len;
        if (rest.len > close and rest[close] == '>' and std.ascii.eqlIgnoreCase(rest[open..close], tag)) return close + 1;
    }
    return 0;
}

fn untagged(a: Allocator, inner: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, inner, '<') == null) return inner;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < inner.len) {
        const n = tagAt(inner[i..]);
        if (n == 0) try out.append(a, inner[i]);
        i += @max(n, 1);
    }
    return out.items;
}

fn drop(out: *std.ArrayList(u8), input: []const u8, end: usize) usize {
    var after = end;
    while (after < input.len and std.ascii.isWhitespace(input[after])) after += 1;
    var before = out.items.len;
    while (before > 0 and std.ascii.isWhitespace(out.items[before - 1])) before -= 1;
    if (before == 0) {
        out.clearRetainingCapacity();
        return after;
    }
    const lines_before = std.mem.count(u8, out.items[before..], "\n");
    if (std.mem.count(u8, input[end..after], "\n") > lines_before) {
        out.shrinkRetainingCapacity(before);
        return end;
    }
    return if (before < out.items.len) after else end;
}

test "allowed controls accepts normalized open vocabulary and rejects brackets or invalid UTF-8" {
    for ([_][]const u8{ "unknown", "Sighs", "gentle robot", "low-pitched", "two  words", "two\twords", "CAFÉ!", ",sighs,," }) |name| _ = try parse(name);
    for ([_][]const u8{ "[sighs]", "sighs]", "\xff" }) |bad| try std.testing.expectError(error.InvalidControl, parse(bad));
}

test "allowed controls off returns identical input without allocating" {
    const input = "[unknown] <sighs>你好</sighs> [unclosed";
    const result = try filter(std.testing.failing_allocator, null, input);
    try std.testing.expectEqualStrings(input, result);
    try std.testing.expectEqual(input.ptr, result.ptr);
    try std.testing.expectError(error.OutOfMemory, filter(std.testing.failing_allocator, try parse("sighs"), "Hi [sighs]"));
}

test {
    _ = @import("drama_allow_golden_test.zig");
}
