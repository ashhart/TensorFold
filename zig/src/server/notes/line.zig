//! A note's line as LLWZ's NoteStore.line() renders it (zengine/notes.py): its id, its text quoted, and its label.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

/// lwlogic.TPD: ticks per day; a day's first tick is its ordinal (0001-01-01 is 1) times this.
pub const tpd: i64 = 1_000_000;

/// The snapshot's "takeback" setting: how a current take-back's label reads.
pub const Takeback = enum { plain, explicit };

/// notes.TAKEBACK: the label of a current take-back, for an attribute of one value or of several.
pub fn takebackWords(t: Takeback, multi: bool) []const u8 {
    return switch (t) {
        .plain => "this takes back something said earlier",
        .explicit => if (multi) "a take-back of this one item only: it no longer holds, and the other notes still do" else "a take-back: the earlier value no longer holds and no new one was given, so it is not known now",
    };
}

/// notes.UNKNOWN_RULE: joined to the block's head when the snapshot's unknown_rule is true.
pub const unknown_rule = "When the notes leave a fact about the user open, or do not mention it, say you do not know; do not guess.";

/// What follows a note's quoted text: nothing (a current fact), a take-back's words, or what retired it.
pub const Label = union(enum) {
    none,
    takeback: bool, // multi: the attribute holds several values
    superseded: Retired,
    withdrawn: Retired,
};

/// The note that retired another, and its told tick (the label names that day).
pub const Retired = struct { by: u32, told: i64 };

/// notes.quote(): json.dumps(text, ensure_ascii=False), and U+2028, U+2029 and U+0085 escaped as well.
pub fn quote(a: Allocator, out: *std.ArrayList(u8), text: []const u8) Allocator.Error!void {
    const q = try json.quote(a, text, .{ .ascii = false });
    defer a.free(q);
    var i: usize = 0;
    while (i < q.len) {
        const rest = q[i..];
        if (std.mem.startsWith(u8, rest, "\u{2028}")) {
            try out.appendSlice(a, "\\u2028");
            i += 3;
        } else if (std.mem.startsWith(u8, rest, "\u{2029}")) {
            try out.appendSlice(a, "\\u2029");
            i += 3;
        } else if (std.mem.startsWith(u8, rest, "\u{85}")) {
            try out.appendSlice(a, "\\u0085");
            i += 2;
        } else {
            try out.append(a, q[i]);
            i += 1;
        }
    }
}

const months = [_][]const u8{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };

pub const Civil = struct { year: i64, month: u8, day: u8 };

/// date.fromordinal(n): the proleptic Gregorian date of an ordinal day (Howard Hinnant's civil_from_days).
pub fn civil(ordinal: i64) Civil {
    const z = ordinal - 719163 + 719468; // days since 0000-03-01
    const era = @divFloor(z, 146097);
    const doe = z - era * 146097;
    const yoe = @divFloor(doe - @divFloor(doe, 1460) + @divFloor(doe, 36524) - @divFloor(doe, 146096), 365);
    const doy = doe - (365 * yoe + @divFloor(yoe, 4) - @divFloor(yoe, 100));
    const mp = @divFloor(5 * doy + 2, 153);
    const d = doy - @divFloor(153 * mp + 2, 5) + 1;
    const m = if (mp < 10) mp + 3 else mp - 9;
    return .{ .year = yoe + era * 400 + @as(i64, if (m <= 2) 1 else 0), .month = @intCast(m), .day = @intCast(d) };
}

/// date(y, m, d).toordinal().
pub fn ordinalOf(year: i64, month: u8, day: u8) i64 {
    const y = if (month <= 2) year - 1 else year;
    const era = @divFloor(y, 400);
    const yoe = y - era * 400;
    const mp: i64 = if (month > 2) month - 3 else month + 9;
    const doy = @divFloor(153 * mp + 2, 5) + day - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468 + 719163;
}

/// L.say_date(L.iso_of(told // L.TPD)): "12 October 2026".
pub fn sayDate(buf: []u8, told: i64) []const u8 {
    const c = civil(@divFloor(told, tpd));
    return std.fmt.bufPrint(buf, "{d} {s} {d}", .{ c.day, months[c.month - 1], c.year }) catch buf[0..0];
}

/// NoteStore.line(): "[id] quote" and the label in parentheses.
pub fn render(a: Allocator, id: u32, text: []const u8, label: Label, takeback: Takeback) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "[{d}] ", .{id});
    try quote(a, &out, text);
    var day: [40]u8 = undefined;
    switch (label) {
        .none => {},
        .takeback => |multi| try out.print(a, " ({s})", .{takebackWords(takeback, multi)}),
        .superseded => |r| try out.print(a, " (no longer current: replaced on {s} by [{d}])", .{ sayDate(&day, r.told), r.by }),
        .withdrawn => |r| try out.print(a, " (the user withdrew this on {s}, in [{d}])", .{ sayDate(&day, r.told), r.by }),
    }
    return out.toOwnedSlice(a);
}

/// A snapshot's line read back: the note's text and what its label says.
pub const Parsed = struct {
    text: []u8,
    kind: enum { none, takeback, superseded, withdrawn },
    multi: bool = false,
    by: u32 = 0,
};

/// The end of the JSON string that starts at `s[0]` ('"'), past its closing quote, or null.
fn stringEnd(s: []const u8) ?usize {
    if (s.len == 0 or s[0] != '"') return null;
    var i: usize = 1;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\') {
            i += 1;
        } else if (s[i] == '"') return i + 1;
    }
    return null;
}

/// A snapshot note's line as line() gives it for `id`: its text, which must quote back to the same bytes, and label.
pub fn parse(a: Allocator, id: u32, line: []const u8, takeback: Takeback) error{ BadLine, OutOfMemory }!Parsed {
    var head: [16]u8 = undefined;
    const prefix = std.fmt.bufPrint(&head, "[{d}] ", .{id}) catch return error.BadLine;
    if (!std.mem.startsWith(u8, line, prefix)) return error.BadLine;
    const body = line[prefix.len..];
    const end = stringEnd(body) orelse return error.BadLine;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const v = switch (try json.parse(arena.allocator(), body[0..end])) {
        .ok => |x| x,
        .err => return error.BadLine,
    };
    if (v != .string or !std.unicode.utf8ValidateSlice(v.string)) return error.BadLine;
    var again: std.ArrayList(u8) = .empty;
    defer again.deinit(a);
    try quote(a, &again, v.string);
    if (!std.mem.eql(u8, again.items, body[0..end])) return error.BadLine; // render() must give these bytes back
    var out: Parsed = .{ .text = undefined, .kind = .none };
    const rest = body[end..];
    if (rest.len > 0) {
        if (!std.mem.startsWith(u8, rest, " (") or !std.mem.endsWith(u8, rest, ")")) return error.BadLine;
        const words = rest[2 .. rest.len - 1];
        if (std.mem.eql(u8, words, takebackWords(takeback, false))) {
            out.kind = .takeback;
        } else if (std.mem.eql(u8, words, takebackWords(takeback, true))) {
            out = .{ .text = undefined, .kind = .takeback, .multi = true };
        } else if (retiredBy(words, "no longer current: replaced on ", " by [")) |n| {
            out = .{ .text = undefined, .kind = .superseded, .by = n };
        } else if (retiredBy(words, "the user withdrew this on ", ", in [")) |n| {
            out = .{ .text = undefined, .kind = .withdrawn, .by = n };
        } else return error.BadLine;
    }
    out.text = try a.dupe(u8, v.string);
    return out;
}

/// The note id in "LEAD<date>MID<n>]", or null when `words` is not that label.
fn retiredBy(words: []const u8, lead: []const u8, mid: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, words, lead) or !std.mem.endsWith(u8, words, "]")) return null;
    const at = std.mem.lastIndexOf(u8, words, mid) orelse return null;
    if (at < lead.len) return null;
    return std.fmt.parseInt(u32, words[at + mid.len .. words.len - 1], 10) catch null;
}

test "dates as Python's date.fromordinal and say_date give them" {
    try std.testing.expectEqual(@as(i64, 1), ordinalOf(1, 1, 1));
    try std.testing.expectEqual(@as(i64, 739901), ordinalOf(2026, 10, 12)); // date(2026, 10, 12).toordinal()
    try std.testing.expectEqual(@as(i64, 719163), ordinalOf(1970, 1, 1));
    var buf: [40]u8 = undefined;
    try std.testing.expectEqualStrings("12 October 2026", sayDate(&buf, 739901 * tpd + 2900));
    try std.testing.expectEqualStrings("29 February 2024", sayDate(&buf, ordinalOf(2024, 2, 29) * tpd));
    try std.testing.expectEqualStrings("1 January 1", sayDate(&buf, tpd));
    var n: i64 = 1;
    while (n < 800_000) : (n += 997) {
        const c = civil(n);
        try std.testing.expectEqual(n, ordinalOf(c.year, c.month, c.day));
    }
}

test "a line renders as Python's and reads back" {
    const a = std.testing.allocator;
    const cases = [_]struct { id: u32, text: []const u8, label: Label, want: []const u8 }{
        .{ .id = 2, .text = "My cat's eyes are green.", .label = .none, .want = "[2] \"My cat's eyes are green.\"" },
        .{ .id = 1, .text = "My dog is named Luna.", .label = .{ .superseded = .{ .by = 13, .told = 739901001300 } }, .want = "[1] \"My dog is named Luna.\" (no longer current: replaced on 12 October 2026 by [13])" },
        .{ .id = 12, .text = "I really like cheese.", .label = .{ .withdrawn = .{ .by = 29, .told = 739901002900 } }, .want = "[12] \"I really like cheese.\" (the user withdrew this on 12 October 2026, in [29])" },
        .{ .id = 29, .text = "I no longer like cheese.", .label = .{ .takeback = false }, .want = "[29] \"I no longer like cheese.\" (this takes back something said earlier)" },
        .{ .id = 7, .text = "a \"q\"\\\n\u{2028}x\u{85}\u{1}é", .label = .none, .want = "[7] \"a \\\"q\\\"\\\\\\n\\u2028x\\u0085\\u0001é\"" },
    };
    for (cases) |c| {
        const got = try render(a, c.id, c.text, c.label, .plain);
        defer a.free(got);
        try std.testing.expectEqualStrings(c.want, got);
        const back = try parse(a, c.id, got, .plain);
        defer a.free(back.text);
        try std.testing.expectEqualStrings(c.text, back.text);
    }
    try std.testing.expectError(error.BadLine, parse(a, 3, "[2] \"x\"", .plain));
    try std.testing.expectError(error.BadLine, parse(a, 2, "[2] \"x\" (something else)", .plain));
    try std.testing.expectError(error.BadLine, parse(a, 2, "[2] \"\\u00e9\"", .plain)); // json.dumps keeps é raw
}

test "explicit take-back labels match Python's TAKEBACK and read back" {
    const a = std.testing.allocator;
    const one = "a take-back: the earlier value no longer holds and no new one was given, so it is not known now";
    const many = "a take-back of this one item only: it no longer holds, and the other notes still do";
    const cases = [_]struct { id: u32, text: []const u8, multi: bool, want: []const u8 }{
        .{ .id = 2, .text = "My favourite colour is no longer green.", .multi = false, .want = "[2] \"My favourite colour is no longer green.\" (a take-back: the earlier value no longer holds and no new one was given, so it is not known now)" },
        .{ .id = 5, .text = "I no longer like cheese.", .multi = true, .want = "[5] \"I no longer like cheese.\" (a take-back of this one item only: it no longer holds, and the other notes still do)" },
    };
    for (cases) |c| {
        const got = try render(a, c.id, c.text, .{ .takeback = c.multi }, .explicit);
        defer a.free(got);
        try std.testing.expectEqualStrings(c.want, got);
        const back = try parse(a, c.id, got, .explicit);
        defer a.free(back.text);
        try std.testing.expectEqualStrings(c.text, back.text);
        try std.testing.expectEqual(back.kind, .takeback);
        try std.testing.expectEqual(c.multi, back.multi);
        try std.testing.expectError(error.BadLine, parse(a, c.id, got, .plain)); // plain does not accept the explicit words
    }
    try std.testing.expectEqualStrings(one, takebackWords(.explicit, false));
    try std.testing.expectEqualStrings(many, takebackWords(.explicit, true));
    try std.testing.expectEqualStrings(unknown_rule, "When the notes leave a fact about the user open, or do not mention it, say you do not know; do not guess.");
}
