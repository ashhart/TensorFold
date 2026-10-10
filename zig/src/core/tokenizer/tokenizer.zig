//! TensorFold's tokenizer for Hugging Face tokenizer.json files: encode equals tokenizers' encode(text, add_special_tokens=False).
const std = @import("std");
const unicode = @import("unicode.zig");
const steps = @import("steps.zig");
const models = @import("model.zig");
const Allocator = std.mem.Allocator;

pub const Kind = models.Kind;
pub const Error = models.Error || std.json.ParseError(std.json.Scanner) || error{UnsupportedAddedToken};

pub const AddedToken = struct {
    id: u32,
    content: []const u8,
    special: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = false,
};

/// Dense id-to-token table; entries put after loading stay owned by the caller.
pub const IdTable = struct {
    allocator: Allocator,
    tokens: std.ArrayList(?[]const u8) = .empty,

    pub fn put(t: *IdTable, id: u32, token: []const u8) Allocator.Error!void {
        if (id >= t.tokens.items.len) try t.tokens.appendNTimes(t.allocator, null, id + 1 - t.tokens.items.len);
        t.tokens.items[id] = token;
    }

    pub fn get(t: *const IdTable, id: u32) ?[]const u8 {
        return if (id < t.tokens.items.len) t.tokens.items[id] else null;
    }

    pub fn deinit(t: *IdTable) void {
        t.tokens.deinit(t.allocator);
    }
};

/// Byte trie over added-token contents for leftmost-longest matching, like tokenizers' Aho-Corasick split.
const Trie = struct {
    first: [256]u32 = @splat(0),
    edges: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    ends: std.ArrayList(?u32) = .empty,

    const Match = struct { start: usize, end: usize, token: u32 };

    fn insert(t: *Trie, a: Allocator, s: []const u8, token: u32) Allocator.Error!void {
        if (s.len == 0) return;
        if (t.ends.items.len == 0) try t.ends.append(a, null);
        var node: u32 = 0;
        for (s, 0..) |b, i| {
            const slot = if (i == 0) &t.first[b] else (try t.edges.getOrPutValue(a, @as(u64, node) << 8 | b, 0)).value_ptr;
            if (slot.* == 0) {
                try t.ends.append(a, null);
                slot.* = @intCast(t.ends.items.len - 1);
            }
            node = slot.*;
        }
        if (t.ends.items[node] == null) t.ends.items[node] = token;
    }

    fn next(t: *const Trie, s: []const u8, from: usize) ?Match {
        if (t.ends.items.len == 0) return null;
        for (from..s.len) |i| {
            var node = t.first[s[i]];
            if (node == 0) continue;
            var best: ?Match = null;
            var j = i + 1;
            while (true) {
                if (t.ends.items[node]) |token| best = .{ .start = i, .end = j, .token = token };
                if (j == s.len) break;
                node = t.edges.get(@as(u64, node) << 8 | s[j]) orelse break;
                j += 1;
            }
            if (best) |b| return b;
        }
        return null;
    }
};

const Segment = struct { start: usize, end: usize, id: ?u32 };

pub const Tokenizer = struct {
    allocator: Allocator,
    arena: ?*std.heap.ArenaAllocator = null,
    /// Token to id for the model vocabulary and every added token.
    vocab: std.StringHashMap(u32),
    id_to_token: IdTable,
    unk_id: ?u32 = null,
    added: []const AddedToken = &.{},
    flagged_specials: []const AddedToken = &.{},
    model: models.Model,
    normalizer: ?steps.Normalizer = null,
    pre_tokenizer: ?steps.PreTokenizer = null,
    decoder: ?steps.Decoder = null,
    raw: Trie = .{},
    normalized: Trie = .{},
    specials: std.StringHashMapUnmanaged(void) = .empty,

    pub fn deinit(t: *Tokenizer) void {
        t.vocab.deinit();
        t.id_to_token.deinit();
        if (t.arena) |arena| {
            arena.deinit();
            t.allocator.destroy(arena);
        }
        t.* = undefined;
    }

    /// Id of a token written out whole, such as a chat template's special token.
    pub fn specialTokenId(t: *const Tokenizer, text: []const u8) ?u32 {
        return t.vocab.get(text);
    }

    /// Token ids for `text`, without post-processor tokens; invalid UTF-8 reads as U+FFFD. Caller frees with `a`.
    pub fn encode(t: *const Tokenizer, a: Allocator, text: []const u8) Error![]u32 {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
        var valid = text;
        if (!std.unicode.utf8ValidateSlice(text)) {
            var fixed: std.ArrayList(u8) = .empty;
            try unicode.appendLossy(&fixed, ctx.a, text);
            valid = fixed.items;
        }
        var ids: std.ArrayList(u32) = .empty;
        for (try t.splitAdded(&ctx, &t.raw, valid)) |seg| {
            if (seg.id) |id| try ids.append(ctx.a, id) else if (seg.end > seg.start) try t.encodeText(&ctx, valid[seg.start..seg.end], seg.start == 0, &ids);
        }
        return a.dupe(u32, ids.items);
    }

    /// `encode` with no added token matched: text that spells one (a quoted marker) is ordinary text.
    pub fn encodePlain(t: *const Tokenizer, a: Allocator, text: []const u8) Error![]u32 {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
        var valid = text;
        if (!std.unicode.utf8ValidateSlice(text)) {
            var fixed: std.ArrayList(u8) = .empty;
            try unicode.appendLossy(&fixed, ctx.a, text);
            valid = fixed.items;
        }
        var ids: std.ArrayList(u32) = .empty;
        if (valid.len > 0) try t.encodeTextWith(&ctx, valid, true, &ids, false);
        return a.dupe(u32, ids.items);
    }

    fn encodeText(t: *const Tokenizer, ctx: *steps.Context, text: []const u8, first: bool, ids: *std.ArrayList(u32)) Error!void {
        return t.encodeTextWith(ctx, text, first, ids, true);
    }

    fn encodeTextWith(t: *const Tokenizer, ctx: *steps.Context, text: []const u8, first: bool, ids: *std.ArrayList(u32), added: bool) Error!void {
        var zero: usize = if (first) unicode.decodeAt(text, 0).len else 0;
        const normalized = if (t.normalizer) |*n| try n.apply(ctx, text, &zero) else text;
        const whole = [_]Segment{.{ .start = 0, .end = normalized.len, .id = null }};
        const segs: []const Segment = if (added) try t.splitAdded(ctx, &t.normalized, normalized) else &whole;
        for (segs) |seg| {
            if (seg.id) |id| {
                try ids.append(ctx.a, id);
                continue;
            }
            if (seg.end == seg.start) continue;
            var pieces: std.ArrayList(steps.Piece) = .empty;
            try pieces.append(ctx.a, .{ .text = normalized[seg.start..seg.end], .first = seg.start < zero });
            if (t.pre_tokenizer) |*p| try p.apply(ctx, &pieces);
            for (pieces.items) |piece| try t.model.tokenize(ctx, &t.vocab, piece.text, ids);
        }
    }

    /// Added-token split with tokenizers' lstrip and rstrip rules.
    fn splitAdded(t: *const Tokenizer, ctx: *steps.Context, trie: *const Trie, s: []const u8) Error![]const Segment {
        var out: std.ArrayList(Segment) = .empty;
        var done: usize = 0;
        var at: usize = 0;
        while (trie.next(s, at)) |hit| {
            at = hit.end;
            const token = t.added[hit.token];
            var start = hit.start;
            var stop = hit.end;
            if (token.lstrip) {
                var k = start;
                while (k > done) {
                    var c = k - 1;
                    while (s[c] & 0xC0 == 0x80) c -= 1;
                    if (!unicode.isSpace(unicode.decodeAt(s, c).cp)) break;
                    k = c;
                }
                start = k;
            }
            if (token.rstrip) while (stop < s.len) {
                const d = unicode.decodeAt(s, stop);
                if (!unicode.isSpace(d.cp)) break;
                stop += d.len;
            };
            if (start > done) try out.append(ctx.a, .{ .start = done, .end = start, .id = null });
            try out.append(ctx.a, .{ .start = start, .end = stop, .id = token.id });
            done = stop;
        }
        if (done < s.len) try out.append(ctx.a, .{ .start = done, .end = s.len, .id = null });
        return out.items;
    }

    /// Text for `ids` through the decoder chain; unknown ids are skipped. Caller frees with `a`.
    pub fn decode(t: *const Tokenizer, a: Allocator, ids: []const u32, skip_special_tokens: bool) Error![]u8 {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
        var tokens = try std.ArrayList([]const u8).initCapacity(ctx.a, ids.len);
        for (ids) |id| {
            const token = t.id_to_token.get(id) orelse continue;
            if (skip_special_tokens and t.specials.contains(token)) continue;
            tokens.appendAssumeCapacity(token);
        }
        if (t.decoder) |*d| return d.decode(a, &ctx, tokens.items);
        return std.mem.join(a, " ", tokens.items);
    }

    /// Exact, possibly incomplete UTF-8 bytes; null for context-dependent decoders. Caller frees with `a`.
    pub fn tokenBytes(t: *const Tokenizer, a: Allocator, id: u32) Error!?[]u8 {
        const d = if (t.decoder) |*decoder| decoder else return null;
        if (!d.tokenLocal()) return null;
        const piece = t.id_to_token.get(id) orelse return try a.dupe(u8, "");
        if (t.specials.contains(piece)) return try a.dupe(u8, piece);
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() }, .raw_bytes = true };
        return try d.decode(a, &ctx, &.{piece});
    }
};

/// Loads `path/tokenizer.json`, or `path` itself when it names a .json file.
pub fn loadTokenizer(io: std.Io, a: Allocator, path: []const u8) !Tokenizer {
    const file = if (std.mem.endsWith(u8, path, ".json")) try a.dupe(u8, path) else try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{path});
    defer a.free(file);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(1 << 30));
    defer a.free(bytes);
    return parse(a, bytes);
}

pub fn parse(a: Allocator, json: []const u8) Error!Tokenizer {
    const arena = try a.create(std.heap.ArenaAllocator);
    arena.* = .init(a);
    var t = Tokenizer{ .allocator = a, .arena = arena, .vocab = .init(a), .id_to_token = .{ .allocator = a }, .model = .empty(.bpe) };
    errdefer t.deinit();
    const keep = arena.allocator();
    var scratch_arena = std.heap.ArenaAllocator.init(a);
    defer scratch_arena.deinit();
    const scratch = scratch_arena.allocator();
    var scanner = std.json.Scanner.initCompleteInput(scratch, json);
    const options = std.json.ParseOptions{ .max_value_len = json.len, .allocate = .alloc_always };
    var added: std.json.Value = .null;
    var have_model = false;
    if (try scanner.next() != .object_begin) return error.InvalidTokenizer;
    while (true) {
        const key = switch (try scanner.nextAlloc(scratch, .alloc_if_needed)) {
            .object_end => break,
            .string => |s| s,
            .allocated_string => |s| s,
            else => return error.InvalidTokenizer,
        };
        if (std.mem.eql(u8, key, "model")) {
            t.model = try models.load(keep, scratch, &scanner, &t.vocab, &t.id_to_token);
            have_model = true;
        } else if (std.mem.eql(u8, key, "added_tokens")) {
            added = try std.json.innerParse(std.json.Value, scratch, &scanner, options);
        } else if (std.mem.eql(u8, key, "normalizer")) {
            const v = try std.json.innerParse(std.json.Value, keep, &scanner, options);
            if (v != .null) t.normalizer = try steps.Normalizer.parse(keep, v);
        } else if (std.mem.eql(u8, key, "pre_tokenizer")) {
            const v = try std.json.innerParse(std.json.Value, keep, &scanner, options);
            if (v != .null) t.pre_tokenizer = try steps.PreTokenizer.parse(keep, v);
        } else if (std.mem.eql(u8, key, "decoder")) {
            const v = try std.json.innerParse(std.json.Value, keep, &scanner, options);
            if (v != .null) t.decoder = try steps.Decoder.parse(keep, v);
        } else try scanner.skipValue();
    }
    if (!have_model) return error.InvalidTokenizer;
    // Metaspace's "first" scheme needs original offsets, tracked here through Prepend, Replace and Strip only.
    if (t.pre_tokenizer) |*p| if (p.firstScheme() == .nested or (p.firstScheme() == .leading and t.normalizer != null and t.normalizer.?.hasForms()))
        return error.UnsupportedPreTokenizer;
    try addTokens(&t, keep, scratch, added);
    t.unk_id = if (t.model.unk_token) |u| t.model.id(&t.vocab, u) else null;
    return t;
}

/// Registers added tokens the way tokenizers' AddedVocabulary does: vocabulary ids first, then new ids past the model.
fn addTokens(t: *Tokenizer, keep: Allocator, scratch: Allocator, list: std.json.Value) Error!void {
    if (list == .null) return;
    if (list != .array) return error.InvalidTokenizer;
    const items = list.array.items;
    var ids = std.StringHashMap(u32).init(scratch);
    var max: ?u32 = null;
    for (items) |item| {
        const content = steps.string(item, "content") orelse return error.InvalidTokenizer;
        // tokenizers' single_word uses Unicode \w (Alphabetic, from PropList), which these tables do not carry.
        if (steps.flag(item, "single_word", false)) return error.UnsupportedAddedToken;
        if (content.len == 0) continue;
        if (steps.flag(item, "special", false)) try t.specials.put(keep, try keep.dupe(u8, content), {});
        if (ids.contains(content)) continue;
        const id = t.model.id(&t.vocab, content) orelse if (max) |m| (if (m >= t.model.size or t.model.size == 0) m + 1 else t.model.size) else t.model.size;
        max = @max(max orelse id, id);
        try ids.put(content, id);
    }
    var tokens: std.ArrayList(AddedToken) = .empty;
    var seen = std.StringHashMap(void).init(scratch);
    // tokenizers lists special tokens before the others; a content's first special entry sets its flags.
    for ([_]bool{ true, false }) |special_pass| for (items) |item| {
        const content = steps.string(item, "content").?;
        const special = steps.flag(item, "special", false);
        if (content.len == 0 or special != special_pass or seen.contains(content)) continue;
        if (!special and t.specials.contains(content)) continue;
        try seen.put(content, {});
        try tokens.append(keep, .{
            .id = ids.get(content).?,
            .content = try keep.dupe(u8, content),
            .special = special,
            .lstrip = steps.flag(item, "lstrip", false),
            .rstrip = steps.flag(item, "rstrip", false),
            .normalized = steps.flag(item, "normalized", !special),
        });
    };
    t.added = tokens.items;
    var specials: usize = 0;
    while (specials < tokens.items.len and tokens.items[specials].special) specials += 1;
    t.flagged_specials = tokens.items[0..specials];
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
    for (tokens.items, 0..) |token, index| {
        if (t.model.id(&t.vocab, token.content) == null) {
            try t.vocab.put(token.content, token.id);
            try t.model.added_only.put(keep, token.content, {});
        }
        var zero: usize = 0;
        const text = if (token.normalized and t.normalizer != null) try keep.dupe(u8, try t.normalizer.?.apply(&ctx, token.content, &zero)) else token.content;
        // tokenizers matches and decodes a normalized added token by its normalized content.
        try t.id_to_token.put(token.id, text);
        try (if (token.normalized) &t.normalized else &t.raw).insert(keep, text, @intCast(index));
    }
}

test "byte-level BPE with merges, added tokens and decode" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "normalizer":{"type":"NFC"},
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    const ids = try t.encode(a, "ab ab<|end|>c");
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5, 7, 6 }, ids);
    const text = try t.decode(a, ids, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("ab ab<|end|>c", text);
    const skipped = try t.decode(a, ids, true);
    defer a.free(skipped);
    try std.testing.expectEqualStrings("ab abc", skipped);
    try std.testing.expectEqual(@as(?u32, 7), t.specialTokenId("<|end|>"));
    try std.testing.expectEqual(@as(usize, 1), t.flagged_specials.len);
    const marker = (try t.tokenBytes(a, 7)).?;
    defer a.free(marker);
    try std.testing.expectEqualStrings("<|end|>", marker);
    const unknown = (try t.tokenBytes(a, 99)).?;
    defer a.free(unknown);
    try std.testing.expectEqualStrings("", unknown);
}

test "sentencepiece-style BPE with byte fallback" {
    const json =
        \\{"added_tokens":[],"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
        \\ "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
        \\ "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
        \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
        \\  "vocab":{"<unk>":0,"<0xC3>":1,"<0xA9>":2,"h":3,"i":4,"▁":5,"hi":6,"▁h":7},"merges":["h i","▁ h"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    const ids = try t.encode(a, "hi h\u{E9}");
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7, 1, 2 }, ids);
    const text = try t.decode(a, ids, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("hi h\u{E9}", text);
    try std.testing.expectEqual(@as(?u32, 0), t.unk_id);
    const first = (try t.tokenBytes(a, 1)).?;
    defer a.free(first);
    const second = (try t.tokenBytes(a, 2)).?;
    defer a.free(second);
    try std.testing.expectEqualSlices(u8, &.{0xc3}, first);
    try std.testing.expectEqualSlices(u8, &.{0xa9}, second);
    const space = (try t.tokenBytes(a, 7)).?;
    defer a.free(space);
    try std.testing.expectEqualStrings(" h", space);
}

test "without a decoder, decode joins known ids' tokens with spaces" {
    const a = std.testing.allocator;
    var t = Tokenizer{ .allocator = a, .vocab = .init(a), .id_to_token = .{ .allocator = a }, .model = .empty(.wordpiece) };
    defer t.deinit();
    try t.vocab.put("t1", 1);
    try t.id_to_token.put(1, "t1");
    try t.id_to_token.put(4, "t4");
    const text = try t.decode(a, &.{ 1, 4, 9 }, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("t1 t4", text);
    try std.testing.expectEqual(@as(?u32, null), t.unk_id);
    try std.testing.expect((try t.tokenBytes(a, 1)) == null);
}

test "raw ByteLevel pieces preserve an incomplete UTF-8 character" {
    const a = std.testing.allocator;
    var t = Tokenizer{
        .allocator = a,
        .vocab = .init(a),
        .id_to_token = .{ .allocator = a },
        .model = .empty(.bpe),
        .decoder = .byte_level,
    };
    defer t.deinit();
    try t.id_to_token.put(0, "\u{c3}");
    try t.id_to_token.put(1, "\u{a9}");
    const first = (try t.tokenBytes(a, 0)).?;
    defer a.free(first);
    const second = (try t.tokenBytes(a, 1)).?;
    defer a.free(second);
    try std.testing.expectEqualSlices(u8, &.{0xc3}, first);
    try std.testing.expectEqualSlices(u8, &.{0xa9}, second);
    const decoded = try t.decode(a, &.{ 0, 1 }, false);
    defer a.free(decoded);
    try std.testing.expectEqualStrings("\u{e9}", decoded);
    const partial = try t.decode(a, &.{0}, false);
    defer a.free(partial);
    try std.testing.expectEqualStrings("\u{fffd}", partial);
}

test "raw token bytes refuse replacements after the decoder joins token boundaries" {
    const a = std.testing.allocator;
    const decoders = [_]steps.Decoder{
        .byte_level, .{ .replace = .{ .pattern = .{ .literal = "ab" }, .content = "x" } },
    };
    var t = Tokenizer{
        .allocator = a,
        .vocab = .init(a),
        .id_to_token = .{ .allocator = a },
        .model = .empty(.bpe),
        .decoder = .{ .sequence = &decoders },
    };
    defer t.deinit();
    try t.id_to_token.put(0, "a");
    try t.id_to_token.put(1, "b");
    const decoded = try t.decode(a, &.{ 0, 1 }, false);
    defer a.free(decoded);
    try std.testing.expectEqualStrings("x", decoded);
    try std.testing.expect((try t.tokenBytes(a, 0)) == null);
}

test "raw token bytes conservatively refuse context-sensitive decoder kinds" {
    const a = std.testing.allocator;
    var scratch = std.heap.ArenaAllocator.init(a);
    defer scratch.deinit();
    const pattern = try @import("regex.zig").Regex.compile(scratch.allocator(), "a+");
    const after_fuse = [_]steps.Decoder{
        .fuse, .{ .replace = .{ .pattern = .{ .literal = "ab" }, .content = "x" } },
    };
    const decoders = [_]steps.Decoder{
        .{ .strip = .{ .content = ' ', .start = 1, .stop = 0 } },
        .{ .metaspace = .{ .replacement = 0x2581, .scheme = .always } },
        .{ .wordpiece = .{ .prefix = "##", .cleanup = false } },
        .{ .replace = .{ .pattern = .{ .regex = pattern }, .content = "x" } },
        .{ .sequence = &after_fuse },
    };
    var t = Tokenizer{
        .allocator = a,
        .vocab = .init(a),
        .id_to_token = .{ .allocator = a },
        .model = .empty(.bpe),
    };
    defer t.deinit();
    try t.id_to_token.put(0, "a");
    for (decoders) |decoder| {
        t.decoder = decoder;
        try std.testing.expect((try t.tokenBytes(a, 0)) == null);
    }
}
