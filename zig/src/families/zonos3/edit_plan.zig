//! What a speech edit decides before the model runs (zonos3.tts.edit_align): regions, windows, fill lengths.
const std = @import("std");
const difflib = @import("difflib.zig");
const tables = @import("unicode_words.zig");
const Span = @import("edit.zig").Span;

const A = std.mem.Allocator;

/// A word of the original with its place in the clip (seconds) and its edit tokens.
pub const Word = struct { word: []const u8, start: f64, end: f64, confidence: ?f64 = null, tokens: []const []const u8 = &.{} };

pub const group_open = "\u{e000}";
pub const group_close = "\u{e001}";

fn inRanges(table: []const u32, cp: u32) bool {
    var lo: usize = 0;
    var hi: usize = table.len / 2;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (cp > table[2 * mid + 1]) lo = mid + 1 else hi = mid;
    }
    return lo < table.len / 2 and cp >= table[2 * lo];
}

/// str.upper of one code point into `out`; the count written (at most 3).
fn upper(cp: u21, out: *[3]u21) usize {
    var lo: usize = 0;
    var hi: usize = tables.upper.len / 2;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (tables.upper[2 * mid] < cp) lo = mid + 1 else hi = mid;
    }
    out[0] = cp;
    if (lo < tables.upper.len / 2 and tables.upper[2 * lo] == cp) {
        out[0] = @intCast(tables.upper[2 * lo + 1]);
        return 1;
    }
    var i: usize = 0;
    while (i < tables.upper_many.len) : (i += 2 + tables.upper_many[i + 1]) {
        if (tables.upper_many[i] != cp) continue;
        const n = tables.upper_many[i + 1];
        for (out[0..n], tables.upper_many[i + 2 ..][0..n]) |*o, x| o.* = @intCast(x);
        return n;
    }
    return 1;
}

/// Han and kana, whose runs the reference cuts into words with ICU: text this port does not tokenize.
fn cjk(cp: u21) bool {
    return switch (cp) {
        0x3005, 0x3007, 0x3400...0x4dbf, 0x4e00...0x9fff, 0xf900...0xfaff, 0x3040...0x30ff, 0x31f0...0x31ff, 0xff66...0xff9f => true,
        else => false,
    };
}

/// textmatch.tokens: upper-cased, every letter or digit of any script and `'` kept, everything else a word break.
pub fn tokens(a: A, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var word: std.ArrayList(u8) = .empty;
    var it = (std.unicode.Utf8View.init(text) catch return error.BadUtf8).iterator();
    while (it.nextCodepoint()) |raw| {
        const cp: u21 = switch (raw) {
            0x2018, 0x2019, 0x201b, 0x02bc => '\'',
            '_' => ' ',
            else => raw,
        };
        var up: [3]u21 = undefined;
        for (up[0..upper(cp, &up)]) |c| {
            if (cjk(c)) return error.EditTextNeedsWordBreaks;
            if (c == '\'' or inRanges(&tables.word, c)) {
                var buf: [4]u8 = undefined;
                try word.appendSlice(a, buf[0 .. std.unicode.utf8Encode(c, &buf) catch return error.BadUtf8]);
            } else if (word.items.len != 0) try out.append(a, try word.toOwnedSlice(a));
        }
    }
    if (word.items.len != 0) try out.append(a, try word.toOwnedSlice(a));
    return out.toOwnedSlice(a);
}

fn isSpace(c: u21) bool {
    return switch (c) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// str.split(): the whitespace words of `text`, as slices of it.
pub fn words(a: A, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = (std.unicode.Utf8View.init(text) catch return error.BadUtf8).iterator();
    var start: ?usize = null;
    while (true) {
        const at = it.i;
        const cp = it.nextCodepoint();
        if (cp != null and !isSpace(cp.?)) {
            if (start == null) start = at;
            continue;
        }
        if (start) |s| try out.append(a, text[s..at]);
        start = null;
        if (cp == null) break;
    }
    return out.toOwnedSlice(a);
}

fn isGroup(token: []const u8) bool {
    return std.mem.startsWith(u8, token, group_open) and std.mem.endsWith(u8, token, group_close) and token.len >= group_open.len + group_close.len;
}

/// token_chars: a token's length in characters; a pronunciation group's symbols count 1.4 letters each.
fn tokenChars(token: []const u8) f64 {
    const n: f64 = @floatFromInt(std.unicode.utf8CountCodepoints(token) catch token.len);
    return if (isGroup(token)) 1.4 * (n - 2) else n;
}

/// target_tokens: the edit tokens of each whitespace word of the target; a pronunciation group is one token.
pub fn targetTokens(a: A, target: []const []const u8) ![]const []const []const u8 {
    const out = try a.alloc([]const []const u8, target.len);
    for (out, target) |*o, word| {
        var list: std.ArrayList([]const u8) = .empty;
        var rest = word;
        while (std.mem.indexOf(u8, rest, group_open)) |open| {
            const close = std.mem.indexOfPos(u8, rest, open, group_close) orelse break;
            // a group's body holds no second opening mark: the last one before the close starts it
            const from = std.mem.lastIndexOf(u8, rest[0..close], group_open).?;
            try list.appendSlice(a, try tokens(a, rest[0..from]));
            try list.append(a, rest[from .. close + group_close.len]);
            rest = rest[close + group_close.len ..];
        }
        try list.appendSlice(a, try tokens(a, rest));
        o.* = try list.toOwnedSlice(a);
    }
    return out;
}

fn flat(a: A, lists: []const []const []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (lists) |l| try out.appendSlice(a, l);
    return out.toOwnedSlice(a);
}

/// Two token lists as ids: equal tokens share an id.
fn intern(a: A, x: []const []const u8, y: []const []const u8) ![2][]u32 {
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    var out: [2][]u32 = undefined;
    for (&out, [_][]const []const u8{ x, y }) |*o, list| {
        o.* = try a.alloc(u32, list.len);
        for (o.*, list) |*id, token| {
            const gop = try ids.getOrPut(a, token);
            if (!gop.found_existing) gop.value_ptr.* = ids.count() - 1;
            id.* = gop.value_ptr.*;
        }
    }
    return out;
}

/// seconds_to_spans: seconds rounded outward to frames, clipped to the audio; `edit.valid` judges the result.
pub fn secondsToSpans(a: A, pairs: []const [2]f64, total: usize, rate: f64) ![]Span {
    const out = try a.alloc(Span, pairs.len);
    for (out, pairs) |*s, p| {
        if (!std.math.isFinite(p[0]) or !std.math.isFinite(p[1]) or p[0] * rate > 1e9 or p[1] * rate > 1e9) return error.BadEditSpans;
        const first: usize = @intFromFloat(@max(0, @floor(p[0] * rate + 1e-6)));
        const last: usize = @min(total, @as(usize, @intFromFloat(@max(0, @ceil(p[1] * rate - 1e-6)))));
        s.* = .{ .start = first, .len = last -| first };
    }
    return out;
}

/// diff_spans: the regions a token diff touches, widened by the margin; regions under `min_gap` frames apart are one.
pub fn diffSpans(a: A, original: []const Word, target: []const []const u8, total: usize, rate: f64, margin: f64, max_spans: usize) ![]Span {
    var owner: std.ArrayList(usize) = .empty;
    var spelled: std.ArrayList([]const u8) = .empty;
    for (original, 0..) |w, k| for (w.tokens) |t| {
        try owner.append(a, k);
        try spelled.append(a, t);
    };
    const ids = try intern(a, spelled.items, target);
    const ops = try difflib.opcodes(a, ids[0], ids[1]);
    const min_gap = 5;
    const frames: f64 = @floatFromInt(total);
    var spans: std.ArrayList(Span) = .empty;
    for (ops) |op| {
        if (op.tag == .equal) continue;
        var t0: f64 = undefined;
        var t1: f64 = undefined;
        if (op.i1 != op.i2) {
            t0 = original[owner.items[op.i1]].start;
            t1 = original[owner.items[op.i2 - 1]].end;
        } else {
            t0 = if (op.i1 > 0) original[owner.items[op.i1 - 1]].end else 0;
            t1 = if (op.i1 < owner.items.len) original[owner.items[op.i1]].start else frames / rate;
        }
        const start: usize = @intFromFloat(@min(frames - 1, @max(0, @floor((t0 - margin) * rate))));
        const end: usize = @max(start + 1, @as(usize, @intFromFloat(@min(frames, @max(0, @ceil((t1 + margin) * rate))))));
        if (spans.items.len != 0 and @as(i64, @intCast(start)) - @as(i64, @intCast(spans.items[spans.items.len - 1].start + spans.items[spans.items.len - 1].len)) < min_gap) {
            const last = &spans.items[spans.items.len - 1];
            last.len = @max(end, last.start + last.len) - last.start;
        } else try spans.append(a, .{ .start = start, .len = end - start });
    }
    if (spans.items.len == 0) return error.EditChangesNothing;
    if (spans.items.len > max_spans) return error.EditTouchesTooManyRegions;
    return spans.toOwnedSlice(a);
}

/// edit_words' checks of a caller's timings (finite, ordered, in the clip, spelling the text); `at`: the word at fault.
pub fn checkWords(given: []const Word, original: []const []const u8, duration: f64, at: *usize) !void {
    if (given.len == 0) return error.NoOriginalWords;
    var previous: f64 = 0;
    var spelled: usize = 0;
    for (given, 0..) |w, k| {
        at.* = k;
        if (!std.math.isFinite(w.start) or !std.math.isFinite(w.end)) return error.WordTimeNotFinite;
        if (w.start < 0) return error.WordBeforeClip;
        if (!(w.start < w.end)) return error.WordEndsBeforeItStarts;
        if (w.start < previous - 1e-6) return error.WordsOverlap;
        if (w.end > duration + 1e-3) return error.WordAfterClip;
        previous = w.end;
        for (w.tokens) |t| {
            if (spelled >= original.len or !std.mem.eql(u8, t, original[spelled])) return error.WordsDoNotSpellText;
            spelled += 1;
        }
    }
    if (spelled != original.len) return error.WordsDoNotSpellText;
}

/// WordDiff: the original's words against the target's, diffed over their tokens.
const WordDiff = struct {
    orig_start: []usize,
    tgt_start: []usize,
    ops: []difflib.Op,

    fn init(a: A, original: []const Word, target_tokens: []const []const []const u8) !WordDiff {
        const o = try a.alloc(usize, original.len + 1);
        const t = try a.alloc(usize, target_tokens.len + 1);
        var mine: std.ArrayList([]const u8) = .empty;
        o[0] = 0;
        for (original, 0..) |w, k| {
            o[k + 1] = o[k] + w.tokens.len;
            try mine.appendSlice(a, w.tokens);
        }
        t[0] = 0;
        for (target_tokens, 0..) |ts, k| t[k + 1] = t[k] + ts.len;
        const ids = try intern(a, mine.items, try flat(a, target_tokens));
        return .{ .orig_start = o, .tgt_start = t, .ops = try difflib.opcodes(a, ids[0], ids[1]) };
    }

    /// The target token position of a cut before original token `i`; null inside a changed block.
    fn cut(d: *const WordDiff, i: usize, left: bool) ?usize {
        for (0..d.ops.len) |n| {
            const op = d.ops[if (left) n else d.ops.len - 1 - n];
            if (op.i1 < i and i < op.i2) return if (op.tag == .equal) op.j1 + i - op.i1 else null;
            if ((if (left) op.i1 else op.i2) == i) return if (left) op.j1 else op.j2;
        }
        return if (i == 0) 0 else d.tgt_start[d.tgt_start.len - 1];
    }

    /// The target word index of a cut before original word `k`; null inside a changed block or a target word.
    fn targetWord(d: *const WordDiff, k: usize, left: bool) ?usize {
        const j = d.cut(d.orig_start[k], left) orelse return null;
        var found: ?usize = null;
        for (d.tgt_start, 0..) |start, t| if (start == j) {
            if (left) return t;
            found = t;
        };
        return found;
    }

    /// The original word range [a, b) widened until both edges are target word boundaries.
    fn clean(d: *const WordDiff, a0: usize, b0: usize) [2]usize {
        var a = a0;
        var b = b0;
        while (a > 0 and d.targetWord(a, true) == null) a -= 1;
        while (b < d.orig_start.len - 1 and d.targetWord(b, false) == null) b += 1;
        return .{ a, b };
    }

    /// The target words of the clean original range [a, b), as Python slices them: an edge with no boundary is open.
    fn targetRange(d: *const WordDiff, a: usize, b: usize) [2]usize {
        const n = d.tgt_start.len - 1;
        const ja = d.targetWord(a, true) orelse 0;
        const jb = d.targetWord(b, false) orelse n;
        return .{ @min(ja, n), @max(@min(jb, n), @min(ja, n)) };
    }
};

/// The frames [start, end) of the clip a fill's prompt holds, the regions it fills and its stretch of the target.
pub const Window = struct { start: usize, end: usize, regions: []const usize, text: []const u8 };

/// whole_clip_window.
pub fn wholeClip(a: A, spans: []const Span, total: usize, target_text: []const u8) ![]Window {
    const regions = try a.alloc(usize, spans.len);
    for (regions, 0..) |*r, k| r.* = k;
    return a.dupe(Window, &.{.{ .start = 0, .end = total, .regions = regions, .text = target_text }});
}

/// Python's round(): half to even.
fn roundHalfEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

const Draft = struct { a: usize, b: usize, regions: std.ArrayList(usize) };

fn draftBefore(_: void, x: Draft, y: Draft) bool {
    return if (x.a != y.a) x.a < y.a else x.b < y.b;
}

/// plan_edit_windows: each region with `context` seconds of words, edges out of changed blocks, overlaps merged.
pub fn windows(a: A, spans: []const Span, original: []const Word, target: []const []const u8, target_tokens: []const []const []const u8, total: usize, rate: f64, context: f64, target_text: []const u8) ![]Window {
    const n = original.len;
    if (n == 0) return error.NoWordsToWindowBy;
    const diff = try WordDiff.init(a, original, target_tokens);
    const Ctx = struct {
        original: []const Word,
        spans: []const Span,
        total: usize,
        rate: f64,

        fn boundary(c: @This(), k: usize) usize {
            if (k == 0) return 0;
            if (k == c.original.len) return c.total;
            const at = roundHalfEven((c.original[k - 1].end + c.original[k].start) / 2 * c.rate);
            return @min(c.total, @as(usize, @intFromFloat(@max(0, at))));
        }

        fn frames(c: @This(), w: *const Draft) [2]usize {
            var first: usize = std.math.maxInt(usize);
            var last: usize = 0;
            for (w.regions.items) |k| {
                first = @min(first, c.spans[k].start);
                last = @max(last, c.spans[k].start + c.spans[k].len);
            }
            return .{ @min(c.boundary(w.a), first), @max(c.boundary(w.b), last) };
        }
    };
    const c: Ctx = .{ .original = original, .spans = spans, .total = total, .rate = rate };
    const drafts = try a.alloc(Draft, spans.len);
    for (drafts, spans, 0..) |*d, s, index| {
        const t0 = @as(f64, @floatFromInt(s.start)) / rate - context;
        const t1 = @as(f64, @floatFromInt(s.start + s.len)) / rate + context;
        var first: usize = n;
        for (original, 0..) |w, k| if (w.end > t0) {
            first = k;
            break;
        };
        var after: usize = 0;
        for (original, 0..) |w, k| {
            if (w.start < t1) after = k + 1;
        }
        const edges = diff.clean(first, @max(first, after));
        d.* = .{ .a = edges[0], .b = edges[1], .regions = .empty };
        try d.regions.append(a, index);
    }
    std.mem.sort(Draft, drafts, {}, draftBefore);
    var merged: std.ArrayList(Draft) = .empty;
    for (drafts) |d| {
        if (merged.items.len != 0) {
            const current = &merged.items[merged.items.len - 1];
            if (d.a < current.b or c.frames(&d)[0] < c.frames(current)[1]) {
                current.a = @min(current.a, d.a);
                current.b = @max(current.b, d.b);
                try current.regions.appendSlice(a, d.regions.items);
                std.mem.sort(usize, current.regions.items, {}, std.sort.asc(usize));
                continue;
            }
        }
        try merged.append(a, d);
    }
    const out = try a.alloc(Window, merged.items.len);
    for (out, merged.items) |*w, *d| {
        const f = c.frames(d);
        var text = target_text;
        if (!(d.a == 0 and d.b == n)) {
            const r = diff.targetRange(d.a, d.b);
            text = try std.mem.join(a, " ", target[r[0]..r[1]]);
        }
        w.* = .{ .start = f[0], .end = f[1], .regions = d.regions.items, .text = text };
    }
    return out;
}

fn chars(lists: []const []const []const u8) f64 {
    var n: f64 = 0;
    for (lists) |l| for (l) |t| {
        n += tokenChars(t);
    };
    return n;
}

/// expected_fill_seconds: a region's seconds scaled by target over original text length; none: the clip's pace.
pub fn expectedSeconds(a: A, spans: []const Span, original: []const Word, target_tokens: []const []const []const u8, rate: f64) ![]f64 {
    const diff = try WordDiff.init(a, original, target_tokens);
    const own = try a.alloc([]const []const u8, original.len);
    var spoken: f64 = 0;
    for (own, original) |*o, w| {
        o.* = w.tokens;
        spoken += w.end - w.start;
    }
    const pace = if (spoken > 0) chars(own) / spoken else 0;
    const out = try a.alloc(f64, spans.len);
    for (out, spans) |*o, s| {
        const r0 = @as(f64, @floatFromInt(s.start)) / rate;
        const r1 = @as(f64, @floatFromInt(s.start + s.len)) / rate;
        var first: ?usize = null;
        var last: usize = 0;
        var from: usize = original.len;
        for (original, 0..) |w, k| {
            const mid = (w.start + w.end) / 2;
            if (from == original.len and mid >= r0) from = k;
            if (r0 <= mid and mid < r1) {
                if (first == null) first = k;
                last = k + 1;
            }
        }
        const at = first orelse from;
        const edges = diff.clean(at, if (first != null) last else at);
        const r = diff.targetRange(edges[0], edges[1]);
        const was = chars(own[edges[0]..edges[1]]);
        const now = chars(target_tokens[r[0]..r[1]]);
        o.* = if (was != 0) (r1 - r0) * now / was else if (pace > 0) now / pace else r1 - r0;
    }
    return out;
}

/// An edit as asked: the clip's length, the target, and what is known of the original.
pub const Request = struct {
    target_text: []const u8,
    /// The original's words with their tokens (a caller's, checked, or an alignment's); null: not known.
    original: ?[]const Word = null,
    /// Explicit regions in seconds; null: the word diff's.
    spans: ?[]const [2]f64 = null,
    frames: usize,
    rate: f64 = 15,
    max_spans: usize,
    margin_seconds: f64 = 0.08,
    max_span_seconds: f64 = 30,
    max_context_seconds: f64 = 16,
    context_seconds: f64 = 4,
    min_fill_ratio: f64 = 0.7,

    /// Whether the clip is edited on windows, which needs the original's words.
    pub fn windowed(r: Request) bool {
        return @as(f64, @floatFromInt(r.frames)) / r.rate > r.max_context_seconds;
    }
};

pub const Plan = struct {
    spans: []Span,
    windows: []Window,
    windowed: bool,
    /// Each region's expected fill length in seconds, and the frames its span-end stays masked for.
    expected: []f64,
    min_frames: []usize,
    /// A fill's row budget: max_span_seconds of frames and the flush.
    max_rows: usize,
};

/// EditRuntime.edit up to the first fill: the regions, the windows, the expected and minimum fill lengths.
pub fn plan(a: A, r: Request) !Plan {
    const target = try words(a, r.target_text);
    const target_tokens = try targetTokens(a, target);
    const windowed = r.windowed();
    if (r.original == null and (r.spans == null or windowed)) return error.EditNeedsOriginalWords;
    const spans = if (r.spans) |pairs| try secondsToSpans(a, pairs, r.frames, r.rate) else try diffSpans(a, r.original.?, try flat(a, target_tokens), r.frames, r.rate, r.margin_seconds, r.max_spans);
    if (!@import("edit.zig").valid(spans, r.frames, r.max_spans)) return error.BadEditSpans;
    const wins = if (windowed) try windows(a, spans, r.original.?, target, target_tokens, r.frames, r.rate, r.context_seconds, r.target_text) else try wholeClip(a, spans, r.frames, r.target_text);
    const expected = if (r.original) |original| try expectedSeconds(a, spans, original, target_tokens, r.rate) else blk: {
        const own = try a.alloc(f64, spans.len);
        for (own, spans) |*o, s| o.* = @as(f64, @floatFromInt(s.len)) / r.rate;
        break :blk own;
    };
    const cap: usize = @intFromFloat(r.max_span_seconds * r.rate);
    const min_frames = try a.alloc(usize, spans.len);
    for (min_frames, expected) |*m, seconds| m.* = @min(@as(usize, @intFromFloat(@max(0, r.min_fill_ratio * seconds * r.rate))), cap);
    return .{ .spans = spans, .windows = wins, .windowed = windowed, .expected = expected, .min_frames = min_frames, .max_rows = cap + @import("config.zig").codebooks + 1 };
}

test "edit tokens: upper case, letters and digits of any script and the apostrophe kept, the rest a break" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try tokens(a, "Don\u{2019}t stop\u{2014}it's na\u{ef}ve, stra\u{df}e_3 \u{43f}\u{440}\u{438}\u{432}\u{435}\u{442}!");
    const want = [_][]const u8{ "DON'T", "STOP", "IT'S", "NA\u{cf}VE", "STRASSE", "3", "\u{41f}\u{420}\u{418}\u{412}\u{415}\u{422}" };
    try t.expectEqual(want.len, got.len);
    for (want, got) |w, g| try t.expectEqualStrings(w, g);
    try t.expectError(error.EditTextNeedsWordBreaks, tokens(a, "\u{6771}\u{4eac}"));
    const tt = try targetTokens(a, try words(a, " say \u{e000}f\u{252}ks\u{e001}, ok "));
    try t.expectEqual(@as(usize, 3), tt.len);
    try t.expectEqualStrings("\u{e000}f\u{252}ks\u{e001}", tt[1][0]);
    try t.expectEqual(@as(f64, 1.4 * 4), tokenChars(tt[1][0]));
    try t.expectEqual(@as(f64, 2), roundHalfEven(2.5));
    try t.expectEqual(@as(f64, 4), roundHalfEven(3.5));
}
