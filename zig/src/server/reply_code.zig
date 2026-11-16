//! Linear-time Markdown inline code-span and fenced code-block scanner so quoted call markup stays prose.
const std = @import("std");

pub const Status = union(enum) {
    outside,
    skip: usize,
    pending,
};

const LineEnd = enum { yes, no, wait };

pub const Scanner = struct {
    pos: usize = 0,
    skip_end: usize = 0,
    open_kind: enum { none, fence_line, fence_body, inline_span } = .none,
    open_at: usize = 0,
    open_ch: u8 = 0,
    open_len: usize = 0,
    open_scan: usize = 0,
    once: u128 = 0,
    twice: u128 = 0,
    long_runs: usize = 0,
    dead_until: usize = 0,
    dead_runs: u128 = 0,
    dead_long: bool = false,

    /// Advances past an already consumed envelope so markup inside tool arguments is not scanned as Markdown.
    pub fn jumpTo(c: *Scanner, pos: usize) void {
        if (c.pos < pos or c.open_kind != .none) c.* = .{ .pos = pos, .skip_end = pos };
    }

    /// Classifies `at` in `text` against preceding Markdown code spans and fences; monotone `at` stays O(N).
    pub fn check(c: *Scanner, text: []const u8, at: usize, finished: bool) Status {
        if (at >= text.len) return .outside;
        if (at < c.skip_end) c.* = .{};
        if (c.open_kind == .none and at < c.pos) return .outside;
        if (c.open_kind != .none and at < c.open_at) return .outside;
        if (c.open_kind != .none) if (c.resumeOpen(text, at, finished)) |st| return st;
        while (c.pos <= at) {
            const i = std.mem.indexOfAnyPos(u8, text[0 .. at + 1], c.pos, "`~") orelse {
                c.pos = at + 1;
                return .outside;
            };
            c.pos = i;
            const ch = text[i];
            const e = runEnd(text, i, ch);
            const n = e - i;
            if (ch == '~') {
                if (!atFenceStart(text, i)) {
                    c.pos = e;
                    continue;
                }
                if (e == text.len and !finished and n < 3) return .pending;
                if (n < 3) {
                    c.pos = e;
                    continue;
                }
                c.open_kind = .fence_line;
            } else {
                if (e == text.len and !finished) return .pending;
                if (e == text.len) {
                    c.pos = e;
                    continue;
                }
                if (n >= 3 and atFenceStart(text, i)) {
                    c.open_kind = .fence_line;
                } else {
                    if (i < c.dead_until and c.isDead(n)) {
                        c.pos = e;
                        continue;
                    }
                    c.open_kind = .inline_span;
                    c.once = 0;
                    c.twice = 0;
                    c.long_runs = 0;
                }
            }
            c.open_at = i;
            c.open_ch = ch;
            c.open_len = n;
            c.open_scan = e;
            if (c.resumeOpen(text, at, finished)) |st| return st;
        }
        return .outside;
    }

    fn resumeOpen(c: *Scanner, text: []const u8, at: usize, finished: bool) ?Status {
        if (c.open_kind == .fence_line) {
            const needles: []const u8 = if (c.open_ch == '`') "\n`" else "\n";
            if (std.mem.indexOfAnyPos(u8, text, c.open_scan, needles)) |hit| {
                if (text[hit] == '`') {
                    c.open_kind = .inline_span;
                    c.open_scan = hit;
                    c.once = 0;
                    c.twice = 0;
                    c.long_runs = 0;
                } else {
                    c.open_kind = .fence_body;
                    c.open_scan = hit + 1;
                }
            } else {
                if (finished) {
                    c.open_kind = .none;
                    c.pos = c.open_at + c.open_len;
                    return null;
                }
                c.open_scan = text.len;
                return .pending;
            }
        }
        if (c.open_kind == .fence_body) return c.resumeFenceBody(text, at, finished);
        return c.resumeInline(text, at, finished);
    }

    fn resumeFenceBody(c: *Scanner, text: []const u8, at: usize, finished: bool) ?Status {
        var line = c.open_scan;
        while (true) {
            c.open_scan = line;
            if (line == text.len) {
                if (!finished) return .{ .skip = text.len };
                c.open_kind = .none;
                c.pos = text.len;
                c.skip_end = text.len;
                return .{ .skip = text.len };
            }
            var q = line;
            var sp: usize = 0;
            while (sp < 4 and q < text.len and text[q] == ' ') : ({
                q += 1;
                sp += 1;
            }) {}
            if (sp <= 3 and q < text.len and text[q] == c.open_ch) {
                const r = runEnd(text, q, c.open_ch);
                if (r - q >= c.open_len) {
                    var k = r;
                    while (k < text.len and (text[k] == ' ' or text[k] == '\t' or text[k] == '\r')) k += 1;
                    if (k < text.len and text[k] == '\n') {
                        const end = k + 1;
                        c.open_kind = .none;
                        c.pos = end;
                        c.skip_end = end;
                        return if (at < end) .{ .skip = end } else null;
                    }
                    if (k == text.len) {
                        if (!finished) return .{ .skip = text.len };
                        c.open_kind = .none;
                        c.pos = text.len;
                        c.skip_end = text.len;
                        return .{ .skip = text.len };
                    }
                }
            }
            if (std.mem.indexOfScalarPos(u8, text, line, '\n')) |nl| {
                line = nl + 1;
                continue;
            }
            if (!finished) return .{ .skip = text.len };
            c.open_kind = .none;
            c.pos = text.len;
            c.skip_end = text.len;
            return .{ .skip = text.len };
        }
    }

    fn resumeInline(c: *Scanner, text: []const u8, at: usize, finished: bool) ?Status {
        var p = c.open_scan;
        while (std.mem.indexOfAnyPos(u8, text, p, "\n`")) |hit| {
            if (text[hit] == '\n') {
                switch (nlEndsParagraph(text, hit, finished)) {
                    .yes => {
                        c.markDead(hit);
                        c.open_kind = .none;
                        c.pos = c.open_at + c.open_len;
                        return null;
                    },
                    .wait => {
                        c.open_scan = hit;
                        return .pending;
                    },
                    .no => {
                        p = hit + 1;
                        continue;
                    },
                }
            }
            const je = runEnd(text, hit, '`');
            if (!finished and je == text.len) {
                c.open_scan = hit;
                return .pending;
            }
            const m = je - hit;
            if (m == c.open_len) {
                c.open_kind = .none;
                c.pos = je;
                c.skip_end = je;
                return if (at < je) .{ .skip = je } else null;
            }
            c.recordRun(m);
            p = je;
        }
        if (finished) {
            c.markDead(text.len);
            c.open_kind = .none;
            c.pos = c.open_at + c.open_len;
            return null;
        }
        c.open_scan = text.len;
        return .pending;
    }

    fn recordRun(c: *Scanner, n: usize) void {
        if (n < 128) {
            const bit = @as(u128, 1) << @intCast(n);
            if ((c.once & bit) != 0) c.twice |= bit;
            c.once |= bit;
        } else c.long_runs +|= 1;
    }

    fn markDead(c: *Scanner, until: usize) void {
        c.dead_until = until;
        c.dead_runs = ~c.twice;
        c.dead_long = c.long_runs < 2;
    }

    fn isDead(c: *const Scanner, n: usize) bool {
        if (n < 128) return (c.dead_runs & (@as(u128, 1) << @intCast(n))) != 0;
        return c.dead_long;
    }
};

fn runEnd(text: []const u8, from: usize, ch: u8) usize {
    var i = from + 1;
    while (i < text.len and text[i] == ch) i += 1;
    return i;
}

fn atFenceStart(text: []const u8, i: usize) bool {
    var s: usize = 0;
    while (s < 4 and i > s and text[i - 1 - s] == ' ') s += 1;
    return s <= 3 and (i == s or text[i - 1 - s] == '\n');
}

fn nlEndsParagraph(text: []const u8, nl: usize, finished: bool) LineEnd {
    var q = nl + 1;
    while (q < text.len and (text[q] == ' ' or text[q] == '\t' or text[q] == '\r')) q += 1;
    if (q == text.len) return if (finished) .no else .wait;
    if (text[q] == '\n') return .yes;
    var s = nl + 1;
    var sp: usize = 0;
    while (sp < 4 and s < text.len and text[s] == ' ') : ({
        s += 1;
        sp += 1;
    }) {}
    if (sp <= 3 and s < text.len and (text[s] == '`' or text[s] == '~')) {
        const ch = text[s];
        const e = runEnd(text, s, ch);
        if (!finished and e == text.len and e - s < 3) return .wait;
        if (e - s >= 3) {
            var u = e;
            while (u < text.len) : (u += 1) {
                if (ch == '`' and text[u] == '`') return .no;
                if (text[u] == '\n') return .yes;
            }
            return if (finished) .no else .wait;
        }
    }
    return .no;
}

test "reply code scanner agrees between one-pass finished and incremental streaming queries" {
    var prng: std.Random.DefaultPrng = .init(0x5eedc0de);
    const r = prng.random();
    const parts = [_][]const u8{ "`", "``", "```", "~~~", "<", "x", " ", "\n", "\n\n", "a<b" };
    for (0..10_000) |_| {
        var buf: [96]u8 = undefined;
        var len: usize = 0;
        const count = r.uintLessThan(usize, 18);
        for (0..count) |_| {
            const p = parts[r.uintLessThan(usize, parts.len)];
            if (len + p.len > buf.len) break;
            @memcpy(buf[len..][0..p.len], p);
            len += p.len;
        }
        const text = buf[0..len];
        var whole_sc: Scanner = .{};
        var stream_sc: Scanner = .{};
        for (0..text.len) |i| {
            const ch = text[i];
            if (ch == '`' or ch == '~') continue;
            var resolved = false;
            var n = i + 1;
            while (n <= text.len) : (n += 1) {
                const st = stream_sc.check(text[0..n], i, false);
                if (st != .pending) {
                    const whole = whole_sc.check(text, i, true);
                    switch (st) {
                        .outside => try std.testing.expectEqual(Status.outside, whole),
                        .skip => |s| switch (whole) {
                            .skip => |w| try std.testing.expect(s <= w),
                            else => return error.TestUnexpectedResult,
                        },
                        .pending => unreachable,
                    }
                    resolved = true;
                    break;
                }
            }
            if (!resolved) {
                const st = stream_sc.check(text, i, true);
                const whole = whole_sc.check(text, i, true);
                try std.testing.expectEqual(whole, st);
            }
        }
    }
}
