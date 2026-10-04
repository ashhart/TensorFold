//! Kimi's tiktoken tokenizer: tiktoken.model's byte ranks, Kimi's split pattern by hand, specials matched anywhere.
const std = @import("std");
const unicode = @import("unicode.zig");
const Allocator = std.mem.Allocator;

/// Script=Han ranges (Unicode 17), for the pattern's \p{Han}.
const han = [_][2]u21{
    .{ 0x2E80, 0x2E99 },   .{ 0x2E9B, 0x2EF3 },   .{ 0x2F00, 0x2FD5 },   .{ 0x3005, 0x3005 },   .{ 0x3007, 0x3007 },
    .{ 0x3021, 0x3029 },   .{ 0x3038, 0x303B },   .{ 0x3400, 0x4DBF },   .{ 0x4E00, 0x9FFF },   .{ 0xF900, 0xFA6D },
    .{ 0xFA70, 0xFAD9 },   .{ 0x16FE2, 0x16FE3 }, .{ 0x16FF0, 0x16FF6 }, .{ 0x20000, 0x2A6DF }, .{ 0x2A700, 0x2B81D },
    .{ 0x2B820, 0x2CEAD }, .{ 0x2CEB0, 0x2EBE0 }, .{ 0x2EBF0, 0x2EE5D }, .{ 0x2F800, 0x2FA1D }, .{ 0x30000, 0x3134A },
    .{ 0x31350, 0x33479 },
};

fn isHan(cp: u21) bool {
    for (han) |r| if (cp >= r[0] and cp <= r[1]) return true;
    return false;
}

const Cat = unicode.Category;

fn inCats(cp: u21, mask: u32) bool {
    return unicode.bit(unicode.category(cp)) & mask != 0;
}

const upper_mask = unicode.bit(Cat.Lu) | unicode.bit(Cat.Lt) | unicode.bit(Cat.Lm) | unicode.bit(Cat.Lo) | unicode.mark;
const lower_mask = unicode.bit(Cat.Ll) | unicode.bit(Cat.Lm) | unicode.bit(Cat.Lo) | unicode.mark;

/// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]
fn upperish(cp: u21) bool {
    return inCats(cp, upper_mask) and !isHan(cp);
}

/// [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]
fn lowerish(cp: u21) bool {
    return inCats(cp, lower_mask) and !isHan(cp);
}

/// [^\r\n\p{L}\p{N}]
fn lead(cp: u21) bool {
    return cp != '\r' and cp != '\n' and !inCats(cp, unicode.letter | unicode.number);
}

/// [^\s\p{L}\p{N}]
fn punct(cp: u21) bool {
    return !unicode.isSpace(cp) and !inCats(cp, unicode.letter | unicode.number);
}

const Text = struct {
    s: []const u8,

    fn at(t: Text, i: usize) ?struct { cp: u21, len: usize } {
        if (i >= t.s.len) return null;
        const r = unicode.scan(t.s, i);
        return .{ .cp = r.cp orelse 0xFFFD, .len = r.len };
    }

    /// End of the longest run from i of code points satisfying f.
    fn run(t: Text, i: usize, comptime f: fn (u21) bool) usize {
        var j = i;
        while (t.at(j)) |c| {
            if (!f(c.cp)) break;
            j += c.len;
        }
        return j;
    }

    fn is(t: Text, i: usize, comptime f: fn (u21) bool) ?usize {
        const c = t.at(i) orelse return null;
        return if (f(c.cp)) i + c.len else null;
    }
};

/// (?i:'s|'t|'re|'ve|'m|'ll|'d)? after a word: the first alternative that matches, case-insensitively.
fn contraction(t: Text, i: usize) usize {
    if (i >= t.s.len or t.s[i] != '\'') return i;
    for ([_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" }) |alt| {
        var j = i + 1;
        const ok = for (alt) |ch| {
            const c = t.at(j) orelse break false;
            var buf: [4]u21 = undefined;
            const variants = unicode.asciiCaseVariants(ch, &buf);
            if (std.mem.indexOfScalar(u21, variants, c.cp) == null) break false;
            j += c.len;
        } else true;
        if (ok) return j;
    }
    return i;
}

/// U* L+ from j as the regex backtracks: past the U run if an L follows, else up to the run's last char that is also L.
fn upperThenLower(t: Text, j: usize) ?usize {
    var p = j;
    var last_both: ?usize = null;
    while (t.at(p)) |c| {
        if (!upperish(c.cp)) break;
        if (lowerish(c.cp)) last_both = p + c.len;
        p += c.len;
    }
    if (t.is(p, lowerish) != null) return t.run(p, lowerish);
    return last_both;
}

/// U+ L* from j; null when no U starts it.
fn upperRun(t: Text, j: usize) ?usize {
    const e = t.run(j, upperish);
    return if (e == j) null else t.run(e, lowerish);
}

/// The end of the piece Kimi's pattern matches at i (alternatives in pattern order), or null.
fn piece(t: Text, i: usize) ?usize {
    const c = t.at(i) orelse return null;
    if (isHan(c.cp)) return t.run(i, isHan);
    inline for (.{ upperThenLower, upperRun }) |word| {
        if (lead(c.cp)) if (word(t, i + c.len)) |e| return contraction(t, e);
        if (word(t, i)) |e| return contraction(t, e);
    }
    if (inCats(c.cp, unicode.number)) {
        var j = i;
        for (0..3) |_| j = t.is(j, struct {
            fn f(cp: u21) bool {
                return inCats(cp, unicode.number);
            }
        }.f) orelse break;
        return j;
    }
    const nl = struct {
        fn f(cp: u21) bool {
            return cp == '\r' or cp == '\n';
        }
    }.f;
    if (c.cp == ' ') if (t.is(i + 1, punct) != null) return t.run(t.run(i + 1, punct), nl);
    if (punct(c.cp)) return t.run(t.run(i, punct), nl);
    if (!unicode.isSpace(c.cp)) return null;
    const w = t.run(i, unicode.isSpace);
    var last: ?usize = null;
    var j = i;
    while (j < w) {
        const d = t.at(j).?;
        if (nl(d.cp)) last = j + d.len;
        j += d.len;
    }
    if (last) |e| return e;
    if (w == t.s.len) return w;
    var prev = i;
    j = i;
    while (j < w) {
        prev = j;
        j += t.at(j).?.len;
    }
    return if (prev > i) prev else w;
}

pub const TikToken = struct {
    a: Allocator,
    ranks: std.StringHashMapUnmanaged(u32) = .empty,
    tokens: std.ArrayList([]const u8) = .empty,
    specials: std.ArrayList([]const u8) = .empty,
    first_special: u32 = 0,
    bytes: []u8 = &.{},

    /// tiktoken.model (base64 token, rank a line) and tokenizer_config.json's special names, from `dir`.
    pub fn load(io: std.Io, a: Allocator, dir: []const u8) !TikToken {
        var t = TikToken{ .a = a };
        errdefer t.deinit();
        const model = try std.fs.path.join(a, &.{ dir, "tiktoken.model" });
        defer a.free(model);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, model, a, .limited(1 << 30));
        defer a.free(text);
        var size: usize = 0;
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| size += std.base64.standard.Decoder.calcSizeForSlice(line[0 .. std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadModel]) catch return error.BadModel;
        t.bytes = try a.alloc(u8, size);
        var at: usize = 0;
        lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const sp = std.mem.indexOfScalar(u8, line, ' ').?;
            const n = try std.base64.standard.Decoder.calcSizeForSlice(line[0..sp]);
            try std.base64.standard.Decoder.decode(t.bytes[at .. at + n], line[0..sp]);
            const rank = try std.fmt.parseInt(u32, std.mem.trim(u8, line[sp + 1 ..], " \r"), 10);
            if (rank != t.tokens.items.len) return error.BadModel;
            try t.tokens.append(a, t.bytes[at .. at + n]);
            try t.ranks.put(a, t.bytes[at .. at + n], rank);
            at += n;
        }
        t.first_special = @intCast(t.tokens.items.len);
        const cfg_path = try std.fs.path.join(a, &.{ dir, "tokenizer_config.json" });
        defer a.free(cfg_path);
        const cfg = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 26));
        defer a.free(cfg);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, cfg, .{});
        defer parsed.deinit();
        const named = if (parsed.value.object.get("added_tokens_decoder")) |v| v.object else null;
        for (0..256) |k| {
            const id = t.first_special + @as(u32, @intCast(k));
            var buf: [16]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "{d}", .{id});
            const name = if (named) |m| (if (m.get(key)) |e| e.object.get("content").?.string else null) else null;
            try t.specials.append(a, if (name) |s| try a.dupe(u8, s) else try std.fmt.allocPrint(a, "<|reserved_token_{d}|>", .{id}));
        }
        return t;
    }

    pub fn deinit(t: *TikToken) void {
        for (t.specials.items) |s| t.a.free(s);
        t.specials.deinit(t.a);
        t.ranks.deinit(t.a);
        t.tokens.deinit(t.a);
        t.a.free(t.bytes);
    }

    /// tiktoken's byte-pair merge of one piece: repeatedly the leftmost lowest-ranked adjacent pair.
    fn bpe(t: *const TikToken, p: []const u8, out: *std.ArrayList(u32)) !void {
        if (t.ranks.get(p)) |r| return out.append(t.a, r);
        const none = std.math.maxInt(u32);
        var starts: std.ArrayList(usize) = .empty;
        defer starts.deinit(t.a);
        var rank: std.ArrayList(u32) = .empty;
        defer rank.deinit(t.a);
        for (0..p.len + 1) |i| try starts.append(t.a, i);
        const get = struct {
            fn f(tt: *const TikToken, pp: []const u8, s: []const usize, i: usize) u32 {
                return if (i + 2 < s.len) tt.ranks.get(pp[s[i]..s[i + 2]]) orelse none else none;
            }
        }.f;
        for (0..p.len) |i| try rank.append(t.a, get(t, p, starts.items, i));
        while (true) {
            var best: u32 = none;
            var at: usize = 0;
            for (rank.items[0 .. starts.items.len - 2], 0..) |r, i| if (r < best) {
                best = r;
                at = i;
            };
            if (best == none) break;
            _ = starts.orderedRemove(at + 1);
            _ = rank.orderedRemove(at + 1);
            rank.items[at] = get(t, p, starts.items, at);
            if (at > 0) rank.items[at - 1] = get(t, p, starts.items, at - 1);
        }
        for (0..starts.items.len - 1) |i| try out.append(t.a, t.ranks.get(p[starts.items[i]..starts.items[i + 1]]) orelse return error.UnknownPiece);
    }

    fn ordinary(t: *const TikToken, s: []const u8, out: *std.ArrayList(u32)) !void {
        const txt = Text{ .s = s };
        var i: usize = 0;
        while (i < s.len) {
            if (piece(txt, i)) |e| {
                try t.bpe(s[i..e], out);
                i = e;
            } else i += txt.at(i).?.len;
        }
    }

    /// tiktoken's encode with every special allowed: the leftmost (then longest) special splits the text.
    fn withSpecials(t: *const TikToken, s: []const u8, out: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < s.len) {
            var hit: ?struct { at: usize, id: u32, len: usize } = null;
            var j = i;
            search: while (j < s.len) : (j += 1) {
                if (s[j] != '<' and s[j] != '[') continue;
                for (t.specials.items, 0..) |name, k| if (std.mem.startsWith(u8, s[j..], name)) {
                    if (hit == null or name.len > hit.?.len) hit = .{ .at = j, .id = t.first_special + @as(u32, @intCast(k)), .len = name.len };
                };
                if (hit != null) break :search;
            }
            const end = if (hit) |h| h.at else s.len;
            try t.ordinary(s[i..end], out);
            if (hit) |h| try out.append(t.a, h.id);
            i = if (hit) |h| h.at + h.len else s.len;
        }
    }

    /// Kimi's encode: 400k-character chunks, runs of over 25k spaces or non-spaces split, specials allowed.
    pub fn encode(t: *const TikToken, s: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(t.a);
        const txt = Text{ .s = s };
        var chunk: usize = 0;
        while (chunk < s.len) {
            var end = chunk;
            var chars: usize = 0;
            while (end < s.len and chars < 400_000) : (chars += 1) end += txt.at(end).?.len;
            var start = chunk;
            var i = chunk;
            var run: usize = 0;
            var space = if (txt.at(chunk)) |c| unicode.isSpace(c.cp) else false;
            while (i < end) {
                const c = txt.at(i).?;
                const now = unicode.isSpace(c.cp);
                if (now != space) {
                    run = 1;
                    space = now;
                } else {
                    run += 1;
                    if (run > 25_000) {
                        try t.withSpecials(s[start..i], &out);
                        start = i;
                        run = 1;
                    }
                }
                i += c.len;
            }
            try t.withSpecials(s[start..end], &out);
            chunk = end;
        }
        return out.toOwnedSlice(t.a);
    }

    pub fn decode(t: *const TikToken, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(t.a);
        for (ids) |id| {
            if (id < t.first_special) try out.appendSlice(t.a, t.tokens.items[id]) else if (id - t.first_special < t.specials.items.len) {
                try out.appendSlice(t.a, t.specials.items[id - t.first_special]);
            } else return error.UnknownToken;
        }
        return out.toOwnedSlice(t.a);
    }
};

test "pieces follow Kimi's pattern" {
    const cases = .{
        .{ " hello", 6 }, .{ "Hello's x", 7 },
        .{ "中文abc", 6 },
        .{ "12345", 3 },  .{ "  \n\nx", 4 },
        .{ "   x", 2 },   .{ " !!\n\nx", 5 },
    };
    inline for (cases) |c| try std.testing.expectEqual(@as(?usize, c[1]), piece(.{ .s = c[0] }, 0));
}
