//! Normalizers, pre-tokenizers and decoders of the Hugging Face tokenizer.json format; unknown kinds fail to load.
const std = @import("std");
const unicode = @import("unicode.zig");
const regex = @import("regex.zig");
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

pub const Error = error{ UnsupportedNormalizer, UnsupportedPreTokenizer, UnsupportedDecoder, InvalidTokenizer } || regex.Error;

/// Per-call scratch: `a` is an arena freed when the call returns.
pub const Context = struct {
    a: Allocator,
    matcher: regex.Matcher,
    raw_bytes: bool = false,
    spans: std.ArrayList(Span) = .empty,
    hits: std.ArrayList([2]usize) = .empty,
};

pub const Span = struct { start: usize, end: usize, matched: bool };
pub const Piece = struct { text: []const u8, first: bool };

pub fn field(v: Value, key: []const u8) ?Value {
    if (v != .object) return null;
    const x = v.object.get(key) orelse return null;
    return if (x == .null) null else x;
}

pub fn string(v: Value, key: []const u8) ?[]const u8 {
    const x = field(v, key) orelse return null;
    return if (x == .string) x.string else null;
}

pub fn flag(v: Value, key: []const u8, default: bool) bool {
    const x = field(v, key) orelse return default;
    return if (x == .bool) x.bool else default;
}

fn count(v: Value, key: []const u8, default: usize) usize {
    const x = field(v, key) orelse return default;
    return if (x == .integer and x.integer >= 0) @intCast(x.integer) else default;
}

/// A Sequence step's `key` list, each item parsed as a `T`.
fn sequence(comptime T: type, a: Allocator, v: Value, key: []const u8) Error![]const T {
    const list = field(v, key) orelse return error.InvalidTokenizer;
    if (list != .array) return error.InvalidTokenizer;
    const items = try a.alloc(T, list.array.items.len);
    for (items, list.array.items) |*item, json| item.* = try T.parse(a, json);
    return items;
}

fn char(s: []const u8) Error!u21 {
    if (s.len == 0 or !std.unicode.utf8ValidateSlice(s)) return error.InvalidTokenizer;
    const d = unicode.decodeAt(s, 0);
    if (d.len != s.len) return error.InvalidTokenizer;
    return d.cp;
}

pub const Pattern = union(enum) {
    literal: []const u8,
    regex: regex.Regex,

    fn parse(a: Allocator, v: ?Value) Error!Pattern {
        const p = v orelse return error.InvalidTokenizer;
        if (string(p, "String")) |s| return .{ .literal = s };
        if (string(p, "Regex")) |s| return .{ .regex = try regex.Regex.compile(a, s) };
        return error.InvalidTokenizer;
    }

    /// Splits `s` into matched and unmatched spans like tokenizers' Pattern::find_matches.
    fn find(p: *const Pattern, ctx: *Context, s: []const u8) Error![]const Span {
        ctx.spans.clearRetainingCapacity();
        if (s.len == 0) {
            try ctx.spans.append(ctx.a, .{ .start = 0, .end = 0, .matched = false });
            return ctx.spans.items;
        }
        ctx.hits.clearRetainingCapacity();
        switch (p.*) {
            .literal => |lit| if (lit.len > 0) {
                var at: usize = 0;
                while (std.mem.indexOfPos(u8, s, at, lit)) |start| {
                    try ctx.hits.append(ctx.a, .{ start, start + lit.len });
                    at = start + lit.len;
                }
            },
            .regex => |*re| try re.findAll(&ctx.matcher, s, &ctx.hits),
        }
        var prev: usize = 0;
        for (ctx.hits.items) |hit| {
            if (prev != hit[0]) try ctx.spans.append(ctx.a, .{ .start = prev, .end = hit[0], .matched = false });
            try ctx.spans.append(ctx.a, .{ .start = hit[0], .end = hit[1], .matched = true });
            prev = hit[1];
        }
        if (prev != s.len) try ctx.spans.append(ctx.a, .{ .start = prev, .end = s.len, .matched = false });
        return ctx.spans.items;
    }

    fn replace(p: *const Pattern, ctx: *Context, s: []const u8, content: []const u8) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        for (try p.find(ctx, s)) |span| try out.appendSlice(ctx.a, if (span.matched) content else s[span.start..span.end]);
        return out.items;
    }
};

pub const Normalizer = union(enum) {
    sequence: []const Normalizer,
    nfc,
    nfd,
    replace: struct { pattern: Pattern, content: []const u8 },
    prepend: []const u8,
    strip: struct { left: bool, right: bool },

    pub fn parse(a: Allocator, v: Value) Error!Normalizer {
        const kind = string(v, "type") orelse return error.UnsupportedNormalizer;
        if (std.mem.eql(u8, kind, "Sequence")) return .{ .sequence = try sequence(Normalizer, a, v, "normalizers") };
        if (std.mem.eql(u8, kind, "NFC")) return .nfc;
        if (std.mem.eql(u8, kind, "NFD")) return .nfd;
        if (std.mem.eql(u8, kind, "Replace")) return .{ .replace = .{ .pattern = try Pattern.parse(a, field(v, "pattern")), .content = string(v, "content") orelse return error.InvalidTokenizer } };
        if (std.mem.eql(u8, kind, "Prepend")) return .{ .prepend = string(v, "prepend") orelse return error.InvalidTokenizer };
        if (std.mem.eql(u8, kind, "Strip")) return .{ .strip = .{ .left = flag(v, "strip_left", true), .right = flag(v, "strip_right", true) } };
        return error.UnsupportedNormalizer;
    }

    pub fn hasForms(n: *const Normalizer) bool {
        return switch (n.*) {
            .sequence => |steps| for (steps) |*step| {
                if (step.hasForms()) break true;
            } else false,
            .nfc, .nfd => true,
            else => false,
        };
    }

    /// Normalizes `s`; `zero` is the count of leading bytes aligned to original offset 0 (tokenizers' alignments).
    pub fn apply(n: *const Normalizer, ctx: *Context, s: []const u8, zero: *usize) Error![]const u8 {
        switch (n.*) {
            .sequence => |steps| {
                var text = s;
                for (steps) |*step| text = try step.apply(ctx, text, zero);
                return text;
            },
            .nfc => return unicode.normalize(ctx.a, s, .nfc),
            .nfd => return unicode.normalize(ctx.a, s, .nfd),
            .replace => |r| {
                // Replacement text takes the alignment of the text it replaces.
                var out: std.ArrayList(u8) = .empty;
                var lead = true;
                var kept: usize = 0;
                for (try r.pattern.find(ctx, s)) |span| {
                    const piece = if (span.matched) r.content else s[span.start..span.end];
                    try out.appendSlice(ctx.a, piece);
                    if (!lead) continue;
                    const covered = if (span.matched) (if (span.start < zero.*) piece.len else 0) else @min(piece.len, zero.* -| span.start);
                    kept += covered;
                    lead = covered == piece.len;
                }
                zero.* = kept;
                return out.items;
            },
            .prepend => |p| {
                if (s.len == 0) return s;
                if (zero.* > 0) zero.* += p.len;
                return std.mem.concat(ctx.a, u8, &.{ p, s });
            },
            .strip => |st| {
                var start: usize = 0;
                var end = s.len;
                if (st.left) while (start < end) {
                    const d = unicode.decodeAt(s, start);
                    if (!unicode.isSpace(d.cp)) break;
                    start += d.len;
                };
                if (st.right) while (end > start) {
                    var at = end - 1;
                    while (s[at] & 0xC0 == 0x80) at -= 1;
                    if (!unicode.isSpace(unicode.decodeAt(s, at).cp)) break;
                    end = at;
                };
                zero.* = @min(zero.* -| start, end - start);
                return s[start..end];
            },
        }
    }
};

pub const Behavior = enum { removed, isolated, merged_with_previous, merged_with_next, contiguous };
pub const Scheme = enum { always, first, never };

/// GPT-2's byte-to-character alphabet: printable Latin-1 maps to itself, the other 68 bytes to U+0100 onwards.
pub const byte_chars: [256]u21 = blk: {
    var table: [256]u21 = undefined;
    var next: u21 = 256;
    for (0..256) |b| {
        const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE;
        table[b] = if (printable) b else next;
        if (!printable) next += 1;
    }
    break :blk table;
};

const char_bytes: [0x144]i16 = blk: {
    var table: [0x144]i16 = @splat(-1);
    for (byte_chars, 0..) |cp, b| table[cp] = b;
    break :blk table;
};

const gpt2_pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

fn scheme(v: Value) Error!Scheme {
    if (string(v, "prepend_scheme")) |s| return std.meta.stringToEnum(Scheme, s) orelse error.InvalidTokenizer;
    return if (flag(v, "add_prefix_space", true)) .always else .never;
}

pub const PreTokenizer = union(enum) {
    sequence: []const PreTokenizer,
    split: struct { pattern: Pattern, behavior: Behavior, invert: bool },
    byte_level: struct { add_prefix_space: bool, regex: ?regex.Regex },
    metaspace: struct { replacement: []const u8, scheme: Scheme, split: bool },
    whitespace_split,

    pub fn parse(a: Allocator, v: Value) Error!PreTokenizer {
        const kind = string(v, "type") orelse return error.UnsupportedPreTokenizer;
        if (std.mem.eql(u8, kind, "Sequence")) return .{ .sequence = try sequence(PreTokenizer, a, v, "pretokenizers") };
        if (std.mem.eql(u8, kind, "Split")) {
            const names = [_][]const u8{ "Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous" };
            const name = string(v, "behavior") orelse return error.InvalidTokenizer;
            const index = for (names, 0..) |n, i| {
                if (std.mem.eql(u8, n, name)) break i;
            } else return error.InvalidTokenizer;
            return .{ .split = .{ .pattern = try Pattern.parse(a, field(v, "pattern")), .behavior = @fromBackingInt(@intCast(index)), .invert = flag(v, "invert", false) } };
        }
        if (std.mem.eql(u8, kind, "ByteLevel"))
            return .{ .byte_level = .{ .add_prefix_space = flag(v, "add_prefix_space", true), .regex = if (flag(v, "use_regex", true)) try regex.Regex.compile(a, gpt2_pattern) else null } };
        if (std.mem.eql(u8, kind, "Metaspace")) {
            const replacement = string(v, "replacement") orelse return error.InvalidTokenizer;
            _ = try char(replacement);
            return .{ .metaspace = .{ .replacement = replacement, .scheme = try scheme(v), .split = flag(v, "split", true) } };
        }
        if (std.mem.eql(u8, kind, "WhitespaceSplit")) return .whitespace_split;
        return error.UnsupportedPreTokenizer;
    }

    /// Where a Metaspace with the "first" scheme sits: on the whole segment, or after another split.
    pub fn firstScheme(p: *const PreTokenizer) enum { none, leading, nested } {
        return p.scan(true);
    }

    fn scan(p: *const PreTokenizer, leading: bool) @TypeOf(p.firstScheme()) {
        switch (p.*) {
            .metaspace => |m| return if (m.scheme != .first) .none else if (leading) .leading else .nested,
            .sequence => |steps| {
                var found: @TypeOf(p.firstScheme()) = .none;
                for (steps, 0..) |*step, i| switch (step.scan(leading and i == 0)) {
                    .nested => return .nested,
                    .leading => found = .leading,
                    .none => {},
                };
                return found;
            },
            else => return .none,
        }
    }

    /// Replaces `pieces` with their pre-tokenized parts; empty parts are dropped as tokenizers drops them.
    pub fn apply(p: *const PreTokenizer, ctx: *Context, pieces: *std.ArrayList(Piece)) Error!void {
        if (p.* == .sequence) {
            for (p.sequence) |*step| try step.apply(ctx, pieces);
            return;
        }
        const input = try ctx.a.dupe(Piece, pieces.items);
        pieces.clearRetainingCapacity();
        for (input) |piece| switch (p.*) {
            .sequence => unreachable,
            .split => |s| try splitPiece(ctx, pieces, piece, &s.pattern, s.behavior, s.invert),
            .byte_level => |b| {
                var text = piece.text;
                if (b.add_prefix_space and !std.mem.startsWith(u8, text, " ")) text = try std.mem.concat(ctx.a, u8, &.{ " ", text });
                const start = pieces.items.len;
                if (b.regex) |re| try splitPiece(ctx, pieces, .{ .text = text, .first = piece.first }, &.{ .regex = re }, .isolated, false) else try pieces.append(ctx.a, .{ .text = text, .first = piece.first });
                for (pieces.items[start..]) |*part| {
                    var out: std.ArrayList(u8) = .empty;
                    try out.ensureTotalCapacity(ctx.a, part.text.len * 2);
                    for (part.text) |byte| try unicode.appendCodepoint(&out, ctx.a, byte_chars[byte]);
                    part.text = out.items;
                }
            },
            .metaspace => |m| {
                var text = try (Pattern{ .literal = " " }).replace(ctx, piece.text, m.replacement);
                const prepend = switch (m.scheme) {
                    .always => true,
                    .first => piece.first,
                    .never => false,
                };
                if (prepend and !std.mem.startsWith(u8, text, m.replacement)) text = try std.mem.concat(ctx.a, u8, &.{ m.replacement, text });
                if (m.split) try splitPiece(ctx, pieces, .{ .text = text, .first = piece.first }, &.{ .literal = m.replacement }, .merged_with_next, false) else if (text.len > 0) try pieces.append(ctx.a, .{ .text = text, .first = piece.first });
            },
            .whitespace_split => {
                var i: usize = 0;
                var start: usize = 0;
                while (i <= piece.text.len) {
                    const d: unicode.Decoded = if (i < piece.text.len) unicode.decodeAt(piece.text, i) else .{ .cp = ' ', .len = 1 };
                    if (unicode.isSpace(d.cp)) {
                        if (i > start) try pieces.append(ctx.a, .{ .text = piece.text[start..i], .first = piece.first and start == 0 });
                        start = i + d.len;
                    }
                    i += d.len;
                }
            },
        };
    }
};

fn splitPiece(ctx: *Context, out: *std.ArrayList(Piece), piece: Piece, pattern: *const Pattern, behavior: Behavior, invert: bool) Error!void {
    const spans = try ctx.a.dupe(Span, try pattern.find(ctx, piece.text));
    if (invert) for (spans) |*s| {
        s.matched = !s.matched;
    };
    var merged: std.ArrayList(Span) = .empty;
    switch (behavior) {
        .isolated => for (spans) |s| try merged.append(ctx.a, .{ .start = s.start, .end = s.end, .matched = false }),
        .removed => try merged.appendSlice(ctx.a, spans),
        .contiguous, .merged_with_previous => {
            var previous = false;
            for (spans) |s| {
                const join = if (behavior == .contiguous) s.matched == previous else s.matched and !previous;
                if (join and merged.items.len > 0) merged.items[merged.items.len - 1].end = s.end else try merged.append(ctx.a, .{ .start = s.start, .end = s.end, .matched = false });
                previous = s.matched;
            }
        },
        .merged_with_next => {
            var previous = false;
            var i = spans.len;
            while (i > 0) {
                i -= 1;
                const s = spans[i];
                if (s.matched and !previous and merged.items.len > 0) merged.items[merged.items.len - 1].start = s.start else try merged.append(ctx.a, .{ .start = s.start, .end = s.end, .matched = false });
                previous = s.matched;
            }
            std.mem.reverse(Span, merged.items);
        },
    }
    for (merged.items) |s| if (!s.matched and s.end > s.start)
        try out.append(ctx.a, .{ .text = piece.text[s.start..s.end], .first = piece.first and s.start == 0 });
}

pub const Decoder = union(enum) {
    sequence: []const Decoder,
    byte_level,
    byte_fallback,
    fuse,
    replace: struct { pattern: Pattern, content: []const u8 },
    strip: struct { content: u21, start: usize, stop: usize },
    metaspace: struct { replacement: u21, scheme: Scheme },
    wordpiece: struct { prefix: []const u8, cleanup: bool },

    pub fn parse(a: Allocator, v: Value) Error!Decoder {
        const kind = string(v, "type") orelse return error.UnsupportedDecoder;
        if (std.mem.eql(u8, kind, "Sequence")) return .{ .sequence = try sequence(Decoder, a, v, "decoders") };
        if (std.mem.eql(u8, kind, "ByteLevel")) return .byte_level;
        if (std.mem.eql(u8, kind, "ByteFallback")) return .byte_fallback;
        if (std.mem.eql(u8, kind, "Fuse")) return .fuse;
        if (std.mem.eql(u8, kind, "Replace")) return .{ .replace = .{ .pattern = try Pattern.parse(a, field(v, "pattern")), .content = string(v, "content") orelse return error.InvalidTokenizer } };
        if (std.mem.eql(u8, kind, "Strip")) return .{ .strip = .{ .content = try char(string(v, "content") orelse return error.InvalidTokenizer), .start = count(v, "start", 0), .stop = count(v, "stop", 0) } };
        if (std.mem.eql(u8, kind, "Metaspace")) return .{ .metaspace = .{ .replacement = try char(string(v, "replacement") orelse return error.InvalidTokenizer), .scheme = try scheme(v) } };
        if (std.mem.eql(u8, kind, "WordPiece")) return .{ .wordpiece = .{ .prefix = string(v, "prefix") orelse "##", .cleanup = flag(v, "cleanup", true) } };
        return error.UnsupportedDecoder;
    }

    /// Only token-local chains can supply exact bytes independently of adjacent tokens.
    pub fn tokenLocal(d: *const Decoder) bool {
        var joined = false;
        return d.localChain(&joined);
    }

    fn localChain(d: *const Decoder, joined: *bool) bool {
        return switch (d.*) {
            .byte_level, .byte_fallback => blk: {
                if (joined.*) break :blk false;
                joined.* = true;
                break :blk true;
            },
            .fuse => blk: {
                joined.* = true;
                break :blk true;
            },
            .replace => |r| !joined.* and r.pattern == .literal,
            .sequence => |items| blk: {
                for (items) |*item| if (!item.localChain(joined)) break :blk false;
                break :blk true;
            },
            else => false,
        };
    }

    /// Runs the decoder chain and joins its output, like tokenizers' Decoder::decode.
    pub fn decode(d: *const Decoder, a: Allocator, ctx: *Context, tokens: []const []const u8) Error![]u8 {
        return std.mem.concat(a, u8, try d.chain(ctx, tokens));
    }

    fn chain(d: *const Decoder, ctx: *Context, tokens: []const []const u8) Error![]const []const u8 {
        const a = ctx.a;
        switch (d.*) {
            .sequence => |steps| {
                var current = tokens;
                for (steps) |*step| current = try step.chain(ctx, current);
                return current;
            },
            .byte_level => {
                var bytes: std.ArrayList(u8) = .empty;
                for (tokens) |token| {
                    const start = bytes.items.len;
                    var i: usize = 0;
                    const mapped = while (i < token.len) {
                        const c = unicode.decodeAt(token, i);
                        if (c.cp >= char_bytes.len or char_bytes[c.cp] < 0) break false;
                        try bytes.append(a, @intCast(char_bytes[c.cp]));
                        i += c.len;
                    } else true;
                    if (!mapped) {
                        bytes.shrinkRetainingCapacity(start);
                        try bytes.appendSlice(a, token);
                    }
                }
                if (ctx.raw_bytes) return a.dupe([]const u8, &.{bytes.items});
                var text: std.ArrayList(u8) = .empty;
                try unicode.appendLossy(&text, a, bytes.items);
                return a.dupe([]const u8, &.{text.items});
            },
            .byte_fallback => {
                var out: std.ArrayList([]const u8) = .empty;
                var pending: std.ArrayList(u8) = .empty;
                for (tokens, 0..) |token, i| {
                    const byte: ?u8 = if (token.len == 6 and std.mem.startsWith(u8, token, "<0x") and token[5] == '>') std.fmt.parseInt(u8, token[3..5], 16) catch null else null;
                    if (byte) |b| try pending.append(a, b);
                    if (byte == null or i + 1 == tokens.len) if (pending.items.len > 0) {
                        if (ctx.raw_bytes or std.unicode.utf8ValidateSlice(pending.items)) try out.append(a, try a.dupe(u8, pending.items)) else for (pending.items) |_| try out.append(a, "\u{FFFD}");
                        pending.clearRetainingCapacity();
                    };
                    if (byte == null) try out.append(a, token);
                }
                return out.items;
            },
            .fuse => return a.dupe([]const u8, &.{try std.mem.concat(a, u8, tokens)}),
            .replace => |r| {
                const out = try a.alloc([]const u8, tokens.len);
                for (out, tokens) |*o, t| o.* = try a.dupe(u8, try r.pattern.replace(ctx, t, r.content));
                return out;
            },
            .strip => |st| {
                const out = try a.alloc([]const u8, tokens.len);
                for (out, tokens) |*o, t| {
                    var lo: usize = 0;
                    var n: usize = 0;
                    while (n < st.start and lo < t.len) : (n += 1) {
                        const c = unicode.decodeAt(t, lo);
                        if (c.cp != st.content) break;
                        lo += c.len;
                    }
                    var hi = t.len;
                    n = 0;
                    while (n < st.stop and hi > lo) : (n += 1) {
                        var at = hi - 1;
                        while (t[at] & 0xC0 == 0x80) at -= 1;
                        if (unicode.decodeAt(t, at).cp != st.content) break;
                        hi = at;
                    }
                    o.* = t[lo..hi];
                }
                return out;
            },
            .metaspace => |m| {
                const out = try a.alloc([]const u8, tokens.len);
                for (out, tokens, 0..) |*o, t, n| {
                    var text: std.ArrayList(u8) = .empty;
                    var i: usize = 0;
                    while (i < t.len) {
                        const c = unicode.decodeAt(t, i);
                        if (c.cp != m.replacement) try text.appendSlice(a, t[i .. i + c.len]) else if (n != 0 or m.scheme == .never) try text.append(a, ' ');
                        i += c.len;
                    }
                    o.* = text.items;
                }
                return out;
            },
            .wordpiece => |w| {
                const out = try a.alloc([]const u8, tokens.len);
                for (out, tokens, 0..) |*o, t, n| {
                    var token: []const u8 = t;
                    if (n != 0) token = if (std.mem.startsWith(u8, t, w.prefix)) t[w.prefix.len..] else try std.mem.concat(a, u8, &.{ " ", t });
                    o.* = if (w.cleanup) try cleanup(a, token) else token;
                }
                return out;
            },
        }
    }
};

fn cleanup(a: Allocator, token: []const u8) Error![]const u8 {
    const pairs = [_][2][]const u8{ .{ " .", "." }, .{ " ?", "?" }, .{ " !", "!" }, .{ " ,", "," }, .{ " ' ", "'" }, .{ " n't", "n't" }, .{ " 'm", "'m" }, .{ " do not", " don't" }, .{ " 's", "'s" }, .{ " 've", "'ve" }, .{ " 're", "'re" } };
    var text = token;
    for (pairs) |pair| text = try std.mem.replaceOwned(u8, a, text, pair[0], pair[1]);
    return text;
}

test "byte-level alphabet" {
    try std.testing.expectEqual(@as(u21, 0x120), byte_chars[' ']);
    try std.testing.expectEqual(@as(u21, 0x10A), byte_chars['\n']);
    try std.testing.expectEqual(@as(u21, 0x143), byte_chars[0xAD]);
    try std.testing.expectEqual(@as(u21, 'A'), byte_chars['A']);
    try std.testing.expectEqual(@as(i16, 0xAD), char_bytes[0x143]);
}

test "split behaviors match tokenizers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var ctx = Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
    const cases = [_]struct { Behavior, []const []const u8 }{
        .{ .removed, &.{ "the", "final", "countdown" } },
        .{ .isolated, &.{ "the", "-", "final", "-", "-", "countdown" } },
        .{ .merged_with_previous, &.{ "the-", "final-", "-", "countdown" } },
        .{ .merged_with_next, &.{ "the", "-final", "-", "-countdown" } },
        .{ .contiguous, &.{ "the", "-", "final", "--", "countdown" } },
    };
    for (cases) |case| {
        var out: std.ArrayList(Piece) = .empty;
        try splitPiece(&ctx, &out, .{ .text = "the-final--countdown", .first = true }, &.{ .literal = "-" }, case[0], false);
        try std.testing.expectEqual(case[1].len, out.items.len);
        for (case[1], out.items) |want, got| try std.testing.expectEqualStrings(want, got.text);
    }
}
