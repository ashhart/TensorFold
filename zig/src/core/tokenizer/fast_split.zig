//! Extensible streaming pre-tokenizer scanners; currently implements o200k / Inkling-2 Split bit-identically.
const std = @import("std");
const unicode = @import("unicode.zig");
const regex = @import("regex.zig");

/// Supported fast pre-tokenizer regex families.
pub const Kind = enum { o200k };

/// The exact Split regex used by Inkling 2 Mini and o200k checkpoints.
pub const o200k_pattern =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?" ++
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?" ++
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Returns the matching fast scanner kind if `pattern` has an exact verified fast implementation.
pub fn detect(pattern: []const u8) ?Kind {
    if (std.mem.eql(u8, pattern, o200k_pattern)) return .o200k;
    return null;
}

const Flags = packed struct(u8) {
    letter: bool = false,
    number: bool = false,
    space: bool = false,
    head: bool = false, // Lu, Lt, Lm, Lo, M
    tail: bool = false, // Ll, Lm, Lo, M
    newline: bool = false, // \r or \n
    _pad: u2 = 0,
};

const ascii_table: [128]Flags = blk: {
    var tbl: [128]Flags = undefined;
    for (0..128) |b| {
        const is_upper = b >= 'A' and b <= 'Z';
        const is_lower = b >= 'a' and b <= 'z';
        const is_num = b >= '0' and b <= '9';
        const is_sp = (b >= 9 and b <= 13) or b == ' ';
        tbl[b] = .{
            .letter = is_upper or is_lower,
            .number = is_num,
            .space = is_sp,
            .head = is_upper,
            .tail = is_lower,
            .newline = b == '\r' or b == '\n',
        };
    }
    break :blk tbl;
};

const CharInfo = struct { len: usize, cp: u21, f: Flags };

inline fn classify(s: []const u8, i: usize) CharInfo {
    const b = s[i];
    if (b < 0x80) return .{ .len = 1, .cp = b, .f = ascii_table[b] };
    const d = unicode.decodeAt(s, i);
    const cat = unicode.category(d.cp);
    const m = unicode.bit(cat);
    return .{
        .len = d.len,
        .cp = d.cp,
        .f = .{
            .letter = (m & unicode.letter) != 0,
            .number = (m & unicode.number) != 0,
            .space = unicode.isSpace(d.cp),
            .head = (m & (unicode.bit(.Lu) | unicode.bit(.Lt) | unicode.bit(.Lm) | unicode.bit(.Lo) | unicode.mark)) != 0,
            .tail = (m & (unicode.bit(.Ll) | unicode.bit(.Lm) | unicode.bit(.Lo) | unicode.mark)) != 0,
        },
    };
}

inline fn isS(cp: u21) bool {
    return cp == 's' or cp == 'S' or cp == 0x017F;
}

inline fn asciiEqLower(b: u8, lower: u8) bool {
    return b == lower or b == lower - ('a' - 'A');
}

/// Case-insensitive `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` matching regex.zig (`unicode.asciiCaseVariants`).
fn matchContraction(s: []const u8, at: usize) usize {
    if (at + 1 >= s.len or s[at] != '\'') return 0;
    const d = unicode.decodeAt(s, at + 1);
    if (isS(d.cp)) return 1 + d.len;
    const b = s[at + 1];
    if (b >= 0x80) return 0;
    if (asciiEqLower(b, 't') or asciiEqLower(b, 'm') or asciiEqLower(b, 'd')) return 2;
    if (at + 2 >= s.len) return 0;
    const c = s[at + 2];
    if ((asciiEqLower(b, 'r') and asciiEqLower(c, 'e')) or
        (asciiEqLower(b, 'v') and asciiEqLower(c, 'e')) or
        (asciiEqLower(b, 'l') and asciiEqLower(c, 'l'))) return 3;
    return 0;
}

/// Branch 1: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(contraction)?`.
fn tryBranch1At(s: []const u8, p1: usize) ?usize {
    var i = p1;
    var last_tail: ?usize = null;
    while (i < s.len) {
        const ch = classify(s, i);
        if (!ch.f.head) break;
        if (ch.f.tail) last_tail = i;
        i += ch.len;
    }
    var end: usize = i;
    if (i < s.len and classify(s, i).f.tail) {
        while (end < s.len) {
            const ch = classify(s, end);
            if (!ch.f.tail) break;
            end += ch.len;
        }
    } else if (last_tail) |lt| {
        // Backtrack greedy head-class across Lu/Lt chars so `+` captures the last shared Lm/Lo/M char.
        end = lt + unicode.decodeAt(s, lt).len;
    } else return null;
    return end + matchContraction(s, end);
}

/// Branch 2: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(contraction)?`.
fn tryBranch2At(s: []const u8, p1: usize) ?usize {
    var i = p1;
    var count: usize = 0;
    while (i < s.len) : (count += 1) {
        const ch = classify(s, i);
        if (!ch.f.head) break;
        i += ch.len;
    }
    if (count == 0) return null;
    while (i < s.len) {
        const ch = classify(s, i);
        if (!ch.f.tail) break;
        i += ch.len;
    }
    return i + matchContraction(s, i);
}

/// Optional prefix `[^\r\n\p{L}\p{N}]?` (greedy: try consuming first, then empty).
fn tryWord(s: []const u8, pos: usize, comptime tryAt: fn ([]const u8, usize) ?usize) ?usize {
    const ch = classify(s, pos);
    const can_prefix = !ch.f.newline and !ch.f.letter and !ch.f.number;
    if (can_prefix) if (tryAt(s, pos + ch.len)) |end| return end;
    return tryAt(s, pos);
}

/// Next match span `[start, end]` starting at `pos.*` for `kind`, advancing `pos.*` to `end`; null at EOF.
pub fn next(kind: Kind, s: []const u8, pos: *usize) ?[2]usize {
    return switch (kind) {
        .o200k => nextO200k(s, pos),
    };
}

fn nextO200k(s: []const u8, pos: *usize) ?[2]usize {
    const start = pos.*;
    if (start >= s.len) return null;
    const ch0 = classify(s, start);

    // Branches 1 and 2: letters/marks with optional non-letter/number/newline prefix and contraction.
    if (tryWord(s, start, tryBranch1At)) |end| {
        pos.* = end;
        return .{ start, end };
    }
    if (tryWord(s, start, tryBranch2At)) |end| {
        pos.* = end;
        return .{ start, end };
    }

    // Branch 3: `\p{N}{1,3}`.
    if (ch0.f.number) {
        var end = start + ch0.len;
        var n: usize = 1;
        while (n < 3 and end < s.len) : (n += 1) {
            const ch = classify(s, end);
            if (!ch.f.number) break;
            end += ch.len;
        }
        pos.* = end;
        return .{ start, end };
    }

    // Branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`.
    const p4_start = if (ch0.cp == ' ') start + 1 else start;
    if (p4_start < s.len) {
        const c4 = classify(s, p4_start);
        if (!c4.f.space and !c4.f.letter and !c4.f.number) {
            var end = p4_start + c4.len;
            while (end < s.len) {
                const ch = classify(s, end);
                if (ch.f.space or ch.f.letter or ch.f.number) break;
                end += ch.len;
            }
            while (end < s.len and (s[end] == '\r' or s[end] == '\n' or s[end] == '/')) end += 1;
            pos.* = end;
            return .{ start, end };
        }
    }

    // Branches 5, 6, 7: `\s*[\r\n]+ | \s+(?!\S) | \s+`.
    if (ch0.f.space) {
        var end = start + ch0.len;
        var last_rn_end: ?usize = if (ch0.f.newline) end else null;
        var prev_start = start;
        while (end < s.len) {
            const ch = classify(s, end);
            if (!ch.f.space) break;
            prev_start = end;
            end += ch.len;
            if (ch.f.newline) last_rn_end = end;
        }
        if (last_rn_end) |rn| {
            pos.* = rn;
            return .{ start, rn };
        }
        const split_end = if (end < s.len and prev_start > start) prev_start else end;
        pos.* = split_end;
        return .{ start, split_end };
    }

    // Every valid UTF-8 codepoint belongs to \p{L}, \p{N}, \s, or [^\s\p{L}\p{N}].
    pos.* = start + ch0.len;
    return .{ start, start + ch0.len };
}

/// Finds a safe parallel-chunk split `i` in `[min_pos, max_pos]` for `kind`, else null.
pub fn findSafeCut(kind: Kind, s: []const u8, target: usize, min_pos: usize, max_pos: usize) ?usize {
    return switch (kind) {
        .o200k => findSafeCutO200k(s, target, min_pos, max_pos),
    };
}

fn findSafeCutO200k(s: []const u8, target: usize, min_pos: usize, max_pos: usize) ?usize {
    const lo = @max(min_pos, 1);
    const hi = @min(max_pos, s.len -| 1);
    if (lo > hi) return null;
    const t = std.math.clamp(target, lo, hi);
    var delta: usize = 0;
    while (t + delta <= hi or t >= lo + delta) : (delta += 1) {
        if (t + delta <= hi) {
            const p = t + delta;
            if (s[p - 1] == '\n' and isAsciiAlnum(s[p])) return p;
        }
        if (delta > 0 and t >= lo + delta) {
            const p = t - delta;
            if (s[p - 1] == '\n' and isAsciiAlnum(s[p])) return p;
        }
    }
    return null;
}

inline fn isAsciiAlnum(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z') or (b >= '0' and b <= '9');
}

fn checkParity(a: std.mem.Allocator, re: *const regex.Regex, m: *regex.Matcher, s: []const u8) !void {
    var expected: std.ArrayList([2]usize) = .empty;
    defer expected.deinit(m.a);
    try re.findAll(m, s, &expected);
    var got: std.ArrayList([2]usize) = .empty;
    defer got.deinit(a);
    var pos: usize = 0;
    while (next(.o200k, s, &pos)) |span| try got.append(a, span);
    try std.testing.expectEqualSlices([2]usize, expected.items, got.items);
}

test "fast_split matches regex.zig across edge cases and 20k fuzzed strings" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var m = regex.Matcher{ .a = arena.allocator() };
    var prog_arena = std.heap.ArenaAllocator.init(a);
    defer prog_arena.deinit();
    const compiled = try regex.Regex.compile(prog_arena.allocator(), o200k_pattern);

    const fixed = [_][]const u8{
        "",
        "hello world",
        "   foo   bar  ",
        "  \r\n  \n  baz",
        "foo's bar'T baz'RE qux've I'm we'll He'd it'\u{017F} ",
        "AbCDeF1234567!@#   ///\r\n\n\t ",
        "cafe\u{0301} あいうA \u{0301}b \u{0301} A\u{3042}B \u{0661}\u{0662}\u{0663}\u{0664}",
        " ! ? \r\n/// !a !A !1 ! ",
        "\"Hello\" (world) [1234] \t\n\n  \n ",
    };
    for (fixed) |s| try checkParity(a, &compiled, &m, s);

    const pool = [_][]const u8{
        "a",         "A",        "z", "Z",  "0", "9", " ", "\t", "\r", "\n", "/", "'", "s", "S", "t", "r", "e", "v", "m", "l", "d",
        "!",         ".",        ",", "\"", "(", ")",
        "é",
        "e\u{0301}", "\u{0301}",
        "あ",
        "漢",
        "\u{017F}",  "\u{00A0}",
        "🚀",
        "\u{0661}",
    };
    var prng = std.Random.DefaultPrng.init(0x5eed1234);
    const rng = prng.random();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    for (0..20_000) |_| {
        buf.clearRetainingCapacity();
        const pieces = rng.uintLessThan(usize, 18);
        for (0..pieces) |_| try buf.appendSlice(a, pool[rng.uintLessThan(usize, pool.len)]);
        try checkParity(a, &compiled, &m, buf.items);
    }
}
