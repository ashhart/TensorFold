//! DSML markup in DeepSeek-V4.1 replies: Python-exact whitespace, tag and header matchers over a growing text.
const std = @import("std");

pub const THINK_END = "</think>";
const MARK = "<｜DSML｜"; // every DSML tag starts here
const MARK_CLOSE = "</｜DSML｜";
const BLOCK_NAMES = [_][]const u8{ "calls", "tool_calls", "function_calls" };
/// Text that may still become a calls block's opening tag (with its blank line) is held back.
const OPEN_TAGS = blk: {
    var tags: [8][]const u8 = undefined;
    var i: usize = 0;
    for ([_][]const u8{ "", "\n\n" }) |sep| for ([_][]const u8{ " calls", "calls", "tool_calls", "function_calls" }) |name| {
        tags[i] = sep ++ MARK ++ name ++ ">";
        i += 1;
    };
    break :blk tags;
};
/// The longest a tag can be (an isspace character is up to 3 bytes): searches resume this far back.
pub const tag_window = 64;

// -- text helpers --------------------------------------------------------------------------------------------------

/// The byte length of the ``str.isspace`` whitespace character at ``i``, else 0.
fn pySpace(s: []const u8, i: usize) usize {
    if (i >= s.len) return 0;
    const b = s[i];
    if (b < 0x80) return if ((b >= 9 and b <= 13) or (b >= 0x1c and b <= 0x20)) 1 else 0;
    const n = std.unicode.utf8ByteSequenceLength(b) catch return 0;
    if (i + n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return 0;
    return switch (cp) {
        0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => n,
        else => 0,
    };
}

/// ``str.strip()``: strips isspace characters at both ends.
pub fn pyStrip(s: []const u8) []const u8 {
    var a: usize = 0;
    while (a < s.len) {
        const n = pySpace(s, a);
        if (n == 0) break;
        a += n;
    }
    var b = s.len;
    while (b > a) {
        var c = b - 1;
        while (c > a and s[c] & 0xC0 == 0x80) c -= 1;
        if (pySpace(s, c) != b - c) break;
        b = c;
    }
    return s[a..b];
}

/// The longest proper prefix of ``tag`` that ``text`` ends with.
pub fn partial(text: []const u8, tag: []const u8) usize {
    var k = @min(tag.len - 1, text.len);
    while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, text, tag[0..k])) return k;
    return 0;
}

/// Trailing text that may still become a calls block's opening tag.
pub fn hold(text: []const u8) usize {
    var most: usize = 0;
    for (OPEN_TAGS) |t| most = @max(most, partial(text, t));
    return most;
}

pub const Match = struct { start: usize, end: usize };

/// ``<｜DSML｜`` (or ``</｜DSML｜``) at ``i``, then ``\s?``: where the tag's name starts.
fn markAt(s: []const u8, i: usize, close: bool) ?usize {
    const m = if (close) MARK_CLOSE else MARK;
    if (!std.mem.startsWith(u8, s[i..], m)) return null;
    return i + m.len;
}

fn word(s: []const u8, at: usize, w: []const u8) ?usize {
    return if (std.mem.startsWith(u8, s[at..], w)) at + w.len else null;
}

/// ``\s?`` + one of ``names``: the end of the name, else null (the optional space given back when the name fails).
fn spacedWord(s: []const u8, at: usize, names: []const []const u8) ?usize {
    const sp = pySpace(s, at);
    for ([_]usize{ at + sp, at }) |from| {
        for (names) |n| if (word(s, from, n)) |e| return e;
        if (sp == 0) break;
    }
    return null;
}

fn spaces(s: []const u8, at: usize) usize {
    var k = at;
    while (true) {
        const n = pySpace(s, k);
        if (n == 0) return k;
        k += n;
    }
}

/// The first ``<｜DSML｜\s?(calls|tool_calls|function_calls)>`` (or its ``</`` form) in ``s[from..to]``.
pub fn blockTag(s: []const u8, from: usize, to: usize, close: bool) ?Match {
    const t = s[0..to];
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, if (close) MARK_CLOSE else MARK)) |p| : (i = p + 1) {
        const n = markAt(t, p, close).?;
        const e = spacedWord(t, n, &BLOCK_NAMES) orelse continue;
        if (e < t.len and t[e] == '>') return .{ .start = p, .end = e + 1 };
    }
    return null;
}

/// A header read at a position: matched, refused, or cut off by the end of the text (it may match later).
const Got = enum { ok, no, more };
const Step = struct { got: Got, at: usize = 0 };

fn wordStep(t: []const u8, at: usize, w: []const u8) Step {
    const have = t[at..];
    if (std.mem.startsWith(u8, have, w)) return .{ .got = .ok, .at = at + w.len };
    return .{ .got = if (have.len < w.len and std.mem.startsWith(u8, w, have)) .more else .no };
}

/// ``\s?`` + one of ``names``, three-way.
fn spacedStep(t: []const u8, at: usize, names: []const []const u8) Step {
    if (at >= t.len) return .{ .got = .more };
    const sp = pySpace(t, at);
    var more = false;
    for (names) |n| {
        const st = wordStep(t, at + sp, n);
        if (st.got == .ok) return st;
        more = more or st.got == .more;
    }
    if (sp > 0) for (names) |n| if (wordStep(t, at, n).got == .ok) return wordStep(t, at, n);
    return .{ .got = if (more) .more else .no };
}

/// ``\s+`` (``plus``) or ``\s*`` then ``c``.
fn spacedChar(t: []const u8, at: usize, plus: bool, c: u8) Step {
    const sp = spaces(t, at);
    if (sp >= t.len) return .{ .got = .more };
    if (plus and sp == at) return .{ .got = .no };
    return if (t[sp] == c) .{ .got = .ok, .at = sp + 1 } else .{ .got = .no };
}

/// ``\s+`` then ``w``.
fn spacedThen(t: []const u8, at: usize, w: []const u8) Step {
    const sp = spaces(t, at);
    if (sp >= t.len) return .{ .got = .more };
    if (sp == at) return .{ .got = .no };
    return wordStep(t, sp, w);
}

/// ``<｜DSML｜\s?invoke\s+name="([^"]*)"\s*>``: the match and the name.
pub const Invoke = struct { m: Match, name: []const u8 };

fn Probe(comptime T: type) type {
    return union(Got) { ok: T, no, more };
}

pub fn invokeAt(t: []const u8, p: usize) Probe(Invoke) {
    const n = markAt(t, p, false) orelse return .no;
    const e = spacedStep(t, n, &.{"invoke"});
    if (e.got != .ok) return if (e.got == .more) .more else .no;
    const q = spacedThen(t, e.at, "name=\"");
    if (q.got != .ok) return if (q.got == .more) .more else .no;
    const close = std.mem.indexOfScalarPos(u8, t, q.at, '"') orelse return .more;
    const gt = spacedChar(t, close + 1, false, '>');
    if (gt.got != .ok) return if (gt.got == .more) .more else .no;
    return .{ .ok = .{ .m = .{ .start = p, .end = gt.at }, .name = t[q.at..close] } };
}

/// The first whole header at/after ``from``, and where to search again (first cut-off header, or before a late tag).
fn Search(comptime T: type) type {
    return struct { hit: ?T, again: usize };
}

pub fn searchHeads(comptime T: type, t: []const u8, from: usize, comptime at: fn ([]const u8, usize) Probe(T)) Search(T) {
    var i = from;
    var first_more: ?usize = null;
    while (std.mem.indexOfPos(u8, t, i, MARK)) |p| : (i = p + 1) switch (at(t, p)) {
        .ok => |h| return .{ .hit = h, .again = p },
        .more => if (first_more == null) {
            first_more = p;
        },
        .no => {},
    };
    return .{ .hit = null, .again = first_more orelse @max(from, t.len -| (MARK.len - 1)) };
}

pub fn invokeOpen(s: []const u8, from: usize, to: usize) ?Invoke {
    if (from > to) return null;
    return searchHeads(Invoke, s[0..to], from, invokeAt).hit;
}

pub fn invokeEnd(s: []const u8, from: usize, to: usize) ?Match {
    if (from > to) return null;
    const t = s[0..to];
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, MARK_CLOSE)) |p| : (i = p + 1) {
        const e = spacedWord(t, markAt(t, p, true).?, &.{"invoke>"}) orelse continue;
        return .{ .start = p, .end = e };
    }
    return null;
}

/// A parameter's opening tag: ``<｜DSML｜\s?parameter\s+name="([^"]*)"\s+string="(true|false)"\s*>``.
pub const ParamHead = struct { start: usize, end: usize, name: []const u8, string: bool };

pub fn paramHeadAt(t: []const u8, p: usize) Probe(ParamHead) {
    const n = markAt(t, p, false) orelse return .no;
    const e = spacedStep(t, n, &.{"parameter"});
    if (e.got != .ok) return if (e.got == .more) .more else .no;
    const q = spacedThen(t, e.at, "name=\"");
    if (q.got != .ok) return if (q.got == .more) .more else .no;
    const close = std.mem.indexOfScalarPos(u8, t, q.at, '"') orelse return .more;
    const v = spacedThen(t, close + 1, "string=\"");
    if (v.got != .ok) return if (v.got == .more) .more else .no;
    const yes = wordStep(t, v.at, "true\"");
    const no = wordStep(t, v.at, "false\"");
    const flag = if (yes.got == .ok) yes else if (no.got == .ok) no else return if (yes.got == .more or no.got == .more) .more else .no;
    const gt = spacedChar(t, flag.at, false, '>');
    if (gt.got != .ok) return if (gt.got == .more) .more else .no;
    return .{ .ok = .{ .start = p, .end = gt.at, .name = t[q.at..close], .string = yes.got == .ok } };
}

/// The first ``</｜DSML｜\s?parameter>`` in ``t[from..]``.
pub fn paramClose(t: []const u8, from: usize) ?Match {
    if (from > t.len) return null;
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, MARK_CLOSE)) |p| : (i = p + 1) {
        const e = spacedWord(t, markAt(t, p, true).?, &.{"parameter>"}) orelse continue;
        return .{ .start = p, .end = e };
    }
    return null;
}

const Param = struct { name: []const u8, string: bool, raw: []const u8, end: usize };

/// The leftmost whole parameter in ``s[from..to]`` (the lazy value up to the first closing tag).
pub fn paramIn(s: []const u8, from: usize, to: usize) ?Param {
    if (from > to) return null;
    const t = s[0..to];
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, MARK)) |p| : (i = p + 1) {
        const h = switch (paramHeadAt(t, p)) {
            .ok => |h| h,
            else => continue,
        };
        const c = paramClose(t, h.end) orelse continue;
        return .{ .name = h.name, .string = h.string, .raw = t[h.end..c.start], .end = c.end };
    }
    return null;
}
