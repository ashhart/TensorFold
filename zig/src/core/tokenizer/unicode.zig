//! Unicode for the tokenizer: general categories, White_Space, ASCII case folds, canonical normalization and UTF-8.
const std = @import("std");
const data = @import("unicode_data.zig");

pub const Category = enum(u5) { Cn, Lu, Ll, Lt, Lm, Lo, Mn, Mc, Me, Nd, Nl, No, Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So, Zs, Zl, Zp, Cc, Cf, Cs, Co };

comptime {
    for (data.categories, 0..) |name, i| std.debug.assert(std.mem.eql(u8, name, @tagName(@as(Category, @fromBackingInt(@intCast(i))))));
}

pub fn bit(c: Category) u32 {
    return @as(u32, 1) << @backingInt(c);
}

fn bits(comptime names: []const Category) u32 {
    var m: u32 = 0;
    for (names) |c| m |= bit(c);
    return m;
}

pub const letter = bits(&.{ .Lu, .Ll, .Lt, .Lm, .Lo });
pub const mark = bits(&.{ .Mn, .Mc, .Me });
pub const number = bits(&.{ .Nd, .Nl, .No });
pub const punctuation = bits(&.{ .Pc, .Pd, .Ps, .Pe, .Pi, .Pf, .Po });
pub const symbol = bits(&.{ .Sm, .Sc, .Sk, .So });
pub const separator = bits(&.{ .Zs, .Zl, .Zp });
pub const other = bits(&.{ .Cc, .Cf, .Cs, .Co, .Cn });

/// General category mask for a property name as Oniguruma spells it (\p{L}, \p{Lu}, \p{LC}); case-insensitive.
pub fn propertyMask(name: []const u8) ?u32 {
    if (name.len == 1) return switch (std.ascii.toUpper(name[0])) {
        'L' => letter,
        'M' => mark,
        'N' => number,
        'P' => punctuation,
        'S' => symbol,
        'Z' => separator,
        'C' => other,
        else => null,
    };
    if (name.len != 2) return null;
    if (std.ascii.eqlIgnoreCase(name, "LC")) return bits(&.{ .Lu, .Ll, .Lt });
    for (std.enums.values(Category)) |c| if (std.ascii.eqlIgnoreCase(name, @tagName(c))) return bit(c);
    return null;
}

const bmp = blk: {
    @setEvalBranchQuota(400_000);
    var table: [0x10000]Category = undefined;
    for (data.category_runs, 0..) |run, i| {
        const start = run >> 5;
        if (start >= 0x10000) break;
        const end = if (i + 1 < data.category_runs.len) @min(data.category_runs[i + 1] >> 5, 0x10000) else 0x10000;
        @memset(table[start..end], @fromBackingInt(@intCast(run & 31)));
    }
    break :blk table;
};

pub fn category(cp: u21) Category {
    if (cp < 0x10000) return bmp[cp];
    var lo: usize = 0;
    var hi: usize = data.category_runs.len;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (data.category_runs[mid] >> 5 <= cp) lo = mid else hi = mid;
    }
    return @fromBackingInt(@intCast(data.category_runs[lo] & 31));
}

/// White_Space: Oniguruma's \s and Rust's char::is_whitespace agree on this set.
pub fn isSpace(cp: u21) bool {
    if (cp < 0x80) return (cp >= 9 and cp <= 13) or cp == ' ';
    return cp == 0x85 or bit(category(cp)) & separator != 0;
}

/// Code points whose simple case folding equals that of an ASCII letter, including it.
pub fn asciiCaseVariants(c: u8, out: *[4]u21) []const u21 {
    const lower = std.ascii.toLower(c);
    out[0] = lower;
    var n: usize = 1;
    if (std.ascii.isAlphabetic(c)) {
        out[1] = std.ascii.toUpper(c);
        n = 2;
        for (data.ascii_folds) |fold| if (fold[1] == lower) {
            out[n] = fold[0];
            n += 1;
        };
    }
    return out[0..n];
}

/// Whether the ASCII pair can start a multi-character case folding (Oniguruma matches "ss" to U+00DF).
pub fn isMultiFoldPair(a: u8, b: u8) bool {
    for (data.ascii_multi_folds) |pair| if (pair[0] == std.ascii.toLower(a) and pair[1] == std.ascii.toLower(b)) return true;
    return false;
}

pub fn combiningClass(cp: u21) u8 {
    if (cp < 0x300) return 0;
    var lo: usize = 0;
    var hi: usize = data.combining.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const run = data.combining[mid];
        if (cp < run[0]) hi = mid else if (cp > run[1]) lo = mid + 1 else return @intCast(run[2]);
    }
    return 0;
}

const s_base = 0xAC00;
const l_base = 0x1100;
const v_base = 0x1161;
const t_base = 0x11A7;
const t_count = 28;
const n_count = 21 * t_count;
const s_count = 19 * n_count;

fn decomposition(cp: u21) ?[2]u21 {
    var lo: usize = 0;
    var hi: usize = data.decompositions.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const entry = data.decompositions[mid];
        if (cp < entry[0]) hi = mid else if (cp > entry[0]) lo = mid + 1 else return .{ entry[1], entry[2] };
    }
    return null;
}

fn composition(first: u21, second: u21) ?u21 {
    if (first >= l_base and first < l_base + 19 and second >= v_base and second < v_base + 21)
        return s_base + ((first - l_base) * 21 + (second - v_base)) * t_count;
    if (first >= s_base and first < s_base + s_count and (first - s_base) % t_count == 0 and second > t_base and second < t_base + t_count)
        return first + (second - t_base);
    const key = @as(u64, first) << 21 | second;
    var lo: usize = 0;
    var hi: usize = data.compositions.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const entry = data.compositions[mid];
        const at = @as(u64, entry[0]) << 21 | entry[1];
        if (key < at) hi = mid else if (key > at) lo = mid + 1 else return entry[2];
    }
    return null;
}

const Mark = struct { cp: u21, ccc: u8 };

fn decompose(cp: u21, out: *std.ArrayList(Mark), a: std.mem.Allocator) !void {
    if (cp >= s_base and cp < s_base + s_count) {
        const index = cp - s_base;
        try out.append(a, .{ .cp = l_base + index / n_count, .ccc = 0 });
        try out.append(a, .{ .cp = v_base + index % n_count / t_count, .ccc = 0 });
        if (index % t_count != 0) try out.append(a, .{ .cp = t_base + index % t_count, .ccc = 0 });
    } else if (decomposition(cp)) |parts| {
        try decompose(parts[0], out, a);
        if (parts[1] != 0) try decompose(parts[1], out, a);
    } else try out.append(a, .{ .cp = cp, .ccc = combiningClass(cp) });
}

pub const Form = enum { nfc, nfd };

/// Canonical normalization of valid UTF-8 (UAX #15); the result is owned by `a`.
pub fn normalize(a: std.mem.Allocator, text: []const u8, form: Form) ![]u8 {
    // Below U+0300 every character is NFC-stable and never the second half of a composition.
    const limit: u8 = if (form == .nfc) 0xCC else 0x80;
    for (text) |b| {
        if (b >= limit) break;
    } else return a.dupe(u8, text);
    var marks: std.ArrayList(Mark) = .empty;
    defer marks.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        const at = decodeAt(text, i);
        try decompose(at.cp, &marks, a);
        i += at.len;
    }
    const items = marks.items;
    var start: usize = 0;
    while (start < items.len) : (start += 1) {
        if (items[start].ccc == 0) continue;
        var end = start;
        while (end < items.len and items[end].ccc != 0) end += 1;
        std.sort.insertion(Mark, items[start..end], {}, struct {
            fn lt(_: void, x: Mark, y: Mark) bool {
                return x.ccc < y.ccc;
            }
        }.lt);
        start = end;
    }
    var len = items.len;
    if (form == .nfc and len > 0) {
        var starter: usize = 0;
        var last: u16 = if (items[0].ccc == 0) 0 else 256;
        len = 1;
        for (items[1..]) |item| {
            if (last < item.ccc or last == 0) if (composition(items[starter].cp, item.cp)) |composite| {
                items[starter].cp = composite;
                continue;
            };
            if (item.ccc == 0) starter = len;
            last = item.ccc;
            items[len] = item;
            len += 1;
        }
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, text.len + 8);
    for (items[0..len]) |item| try appendCodepoint(&out, a, item.cp);
    return out.toOwnedSlice(a);
}

pub const Decoded = struct { cp: u21, len: u3 };

/// Decodes one character of valid UTF-8.
pub inline fn decodeAt(s: []const u8, i: usize) Decoded {
    const b = s[i];
    if (b < 0x80) return .{ .cp = b, .len = 1 };
    if (b < 0xE0) return .{ .cp = @as(u21, b & 0x1F) << 6 | (s[i + 1] & 0x3F), .len = 2 };
    if (b < 0xF0) return .{ .cp = @as(u21, b & 0x0F) << 12 | @as(u21, s[i + 1] & 0x3F) << 6 | (s[i + 2] & 0x3F), .len = 3 };
    return .{ .cp = @as(u21, b & 0x07) << 18 | @as(u21, s[i + 1] & 0x3F) << 12 | @as(u21, s[i + 2] & 0x3F) << 6 | (s[i + 3] & 0x3F), .len = 4 };
}

/// Length of the well-formed sequence at `i`, or of its maximal invalid subpart (Unicode 3.9, Table 3-7) when `cp` is null.
pub fn scan(s: []const u8, i: usize) struct { len: usize, cp: ?u21 } {
    const b = s[i];
    if (b < 0x80) return .{ .len = 1, .cp = b };
    var lo: u8 = 0x80;
    var hi: u8 = 0xBF;
    const need: usize, var cp: u21 = switch (b) {
        0xC2...0xDF => .{ 1, b & 0x1F },
        0xE0...0xEF => .{ 2, b & 0x0F },
        0xF0...0xF4 => .{ 3, b & 0x07 },
        else => return .{ .len = 1, .cp = null },
    };
    if (b == 0xE0) lo = 0xA0;
    if (b == 0xED) hi = 0x9F;
    if (b == 0xF0) lo = 0x90;
    if (b == 0xF4) hi = 0x8F;
    for (1..need + 1) |j| {
        if (i + j >= s.len or s[i + j] < lo or s[i + j] > hi) return .{ .len = j, .cp = null };
        cp = cp << 6 | (s[i + j] & 0x3F);
        lo = 0x80;
        hi = 0xBF;
    }
    return .{ .len = need + 1, .cp = cp };
}

/// Appends `bytes` as UTF-8, replacing each maximal invalid subpart with U+FFFD like Rust's from_utf8_lossy.
pub fn appendLossy(out: *std.ArrayList(u8), a: std.mem.Allocator, bytes: []const u8) !void {
    if (std.unicode.utf8ValidateSlice(bytes)) return out.appendSlice(a, bytes);
    var i: usize = 0;
    while (i < bytes.len) {
        const step = scan(bytes, i);
        if (step.cp == null) try out.appendSlice(a, "\u{FFFD}") else try out.appendSlice(a, bytes[i .. i + step.len]);
        i += step.len;
    }
}

pub fn appendCodepoint(out: *std.ArrayList(u8), a: std.mem.Allocator, cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
    try out.appendSlice(a, buf[0..n]);
}

test "categories and spaces follow Unicode 16" {
    try std.testing.expectEqual(Category.Lu, category('A'));
    try std.testing.expectEqual(Category.Lo, category(0x4E00));
    try std.testing.expectEqual(Category.Lo, category(0x105C0));
    try std.testing.expectEqual(Category.Cn, category(0x323B0));
    try std.testing.expectEqual(Category.So, category(0x1F600));
    try std.testing.expectEqual(Category.Co, category(0x10FFFD));
    try std.testing.expect(isSpace(0x3000) and isSpace(0x85) and isSpace(0x2029) and !isSpace(0x200B) and !isSpace(0x1C));
    try std.testing.expectEqual(letter, propertyMask("l").?);
    try std.testing.expectEqual(bit(.Lo), propertyMask("lo").?);
    try std.testing.expect(propertyMask("Letter") == null);
}

test "canonical normalization" {
    const a = std.testing.allocator;
    const cases = [_][3][]const u8{
        .{ "e\u{301}", "\u{E9}", "e\u{301}" },
        .{ "\u{1E0B}\u{323}", "\u{1E0D}\u{307}", "d\u{323}\u{307}" },
        .{ "\u{1100}\u{1161}\u{11A8}", "\u{AC01}", "\u{1100}\u{1161}\u{11A8}" },
        .{ "\u{212B}", "\u{C5}", "A\u{30A}" },
        .{ "\u{301}e", "\u{301}e", "\u{301}e" },
        .{ "a\u{315}\u{300}\u{5AE}\u{300}b", "\u{E0}\u{5AE}\u{300}\u{315}b", "a\u{5AE}\u{300}\u{300}\u{315}b" },
    };
    for (cases) |case| {
        const c = try normalize(a, case[0], .nfc);
        defer a.free(c);
        try std.testing.expectEqualStrings(case[1], c);
        const d = try normalize(a, case[0], .nfd);
        defer a.free(d);
        try std.testing.expectEqualStrings(case[2], d);
    }
}

test "lossy UTF-8 replaces maximal subparts" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try appendLossy(&out, a, "a\xF0\x9F\x98\xE4\xBD\xA0\xC0\xAF\xED\xA0\x80z\xE4");
    try std.testing.expectEqualStrings("a\u{FFFD}\u{4F60}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}z\u{FFFD}", out.items);
    var buf: [4]u21 = undefined;
    try std.testing.expectEqualSlices(u21, &.{ 's', 'S', 0x17F }, asciiCaseVariants('s', &buf));
    try std.testing.expect(isMultiFoldPair('S', 's') and !isMultiFoldPair('r', 'e'));
}
