//! Code point helpers with Python str semantics: indexing, whitespace, case mapping and printability.
const std = @import("std");
const data = @import("unicode_data.zig");
const Allocator = std.mem.Allocator;

/// Decodes the code point at `i` and advances past it; bytes are valid UTF-8 from JSON or the template.
pub fn next(s: []const u8, i: *usize) u21 {
    const n = std.unicode.utf8ByteSequenceLength(s[i.*]) catch 1;
    const end = @min(i.* + n, s.len);
    defer i.* = end;
    return std.unicode.utf8Decode(s[i.*..end]) catch s[i.*];
}

pub fn count(s: []const u8) usize {
    var n: usize = 0;
    for (s) |c| n += @intFromBool(c & 0xC0 != 0x80);
    return n;
}

/// Byte offset of code point `cp_index`, clamped to the end.
pub fn offset(s: []const u8, cp_index: usize) usize {
    var i: usize = 0;
    var k: usize = 0;
    while (i < s.len and k < cp_index) : (k += 1) _ = next(s, &i);
    return i;
}

pub fn isSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Whether `cp` falls in one of a table's sorted inclusive [first, last] pairs.
fn inTable(t: []const u21, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = t.len / 2;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (cp < t[2 * mid]) hi = mid else if (cp > t[2 * mid + 1]) lo = mid + 1 else return true;
    }
    return false;
}

pub fn isPrintable(cp: u21) bool {
    return !inTable(&data.nonprintable, cp);
}

/// CPython's Final_Sigma context for the capital sigma at bytes [at, end) of the original string.
fn finalSigma(s: []const u8, at: usize, end: usize) bool {
    var j = at;
    const before: u21 = while (j > 0) {
        var b = j - 1;
        while (b > 0 and s[b] & 0xC0 == 0x80) b -= 1;
        var k = b;
        const c = next(s, &k);
        j = b;
        if (!inTable(&data.case_ignorable, c)) break c;
    } else return false;
    if (!inTable(&data.cased, before)) return false;
    var k = end;
    while (k < s.len) {
        const c = next(s, &k);
        if (!inTable(&data.case_ignorable, c)) return !inTable(&data.cased, c);
    }
    return true;
}

pub fn encode(out: *std.ArrayList(u8), a: Allocator, cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch return out.appendSlice(a, "\u{FFFD}");
    try out.appendSlice(a, buf[0..n]);
}

fn mapped(runs: []const [4]i32, special: []const [4]u21, cp: u21, out: *[3]u21) usize {
    for (special) |row| if (row[0] == cp) {
        var n: usize = 0;
        while (n < 3 and row[n + 1] != 0) : (n += 1) out[n] = row[n + 1];
        return n;
    };
    var lo: usize = 0;
    var hi: usize = runs.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const r = runs[mid];
        if (cp < r[0]) hi = mid else if (cp > r[1]) lo = mid + 1 else {
            out[0] = if (@mod(@as(i32, cp) - r[0], r[3]) == 0) @intCast(@as(i32, cp) + r[2]) else cp;
            return 1;
        }
    }
    out[0] = cp;
    return 1;
}

pub const CaseError = error{OutOfMemory};

/// Python str.lower(), including the Final_Sigma rule for capital sigma.
pub fn lower(a: Allocator, s: []const u8) CaseError![]u8 {
    return convert(a, s, &data.lower, &data.lower_special, true);
}

pub fn upper(a: Allocator, s: []const u8) CaseError![]u8 {
    return convert(a, s, &data.upper, &data.upper_special, false);
}

fn convert(a: Allocator, s: []const u8, runs: []const [4]i32, special: []const [4]u21, to_lower: bool) CaseError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, s.len);
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] < 0x80) {
            try out.append(a, if (to_lower) std.ascii.toLower(s[i]) else std.ascii.toUpper(s[i]));
            i += 1;
            continue;
        }
        const at = i;
        const cp = next(s, &i);
        if (to_lower and cp == 0x3A3) {
            try encode(&out, a, if (finalSigma(s, at, i)) 0x3C2 else 0x3C3);
            continue;
        }
        var buf: [3]u21 = undefined;
        for (buf[0..mapped(runs, special, cp, &buf)]) |m| try encode(&out, a, m);
    }
    return out.toOwnedSlice(a);
}

test "python whitespace, printability and case tables" {
    const a = std.testing.allocator;
    try std.testing.expect(isSpace(0x3000) and isSpace(0x1C) and !isSpace(0x200B));
    try std.testing.expect(isPrintable('a') and isPrintable(0x1F44B) and !isPrintable(0xA0) and !isPrintable(0x200D) and !isPrintable(0x7F));
    const up = try upper(a, "straße ǆ ﬁ");
    defer a.free(up);
    try std.testing.expectEqualStrings("STRASSE Ǆ FI", up);
    const low = try lower(a, "ÆØÅ İ ǅ");
    defer a.free(low);
    try std.testing.expectEqualStrings("æøå i\u{307} ǆ", low);
    for ([_][2][]const u8{ .{ "ΟΔΥΣΣΕΥΣ", "οδυσσευς" }, .{ "Σ", "σ" }, .{ "AΣ", "aς" }, .{ "AΣB", "aσb" }, .{ "A'Σ", "a'ς" }, .{ "AΣ'", "aς'" }, .{ "AΣ'b", "aσ'b" } }) |pair| {
        const got = try lower(a, pair[0]);
        defer a.free(got);
        try std.testing.expectEqualStrings(pair[1], got);
    }
    try std.testing.expectEqual(@as(usize, 3), count("a世👋"));
    try std.testing.expectEqual(@as(usize, 4), offset("a世👋", 2));
}
