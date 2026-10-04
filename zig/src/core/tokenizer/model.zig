//! The word-level models of tokenizer.json: BPE (merge ranks, byte fallback, unk fusing), WordPiece and WordLevel.
const std = @import("std");
const unicode = @import("unicode.zig");
const steps = @import("steps.zig");
const Allocator = std.mem.Allocator;
const Vocab = std.StringHashMap(u32);

pub const Kind = enum { bpe, wordpiece, wordlevel };
pub const Error = error{ UnsupportedModel, InvalidTokenizer, UnknownTokenMissing } || steps.Error || std.json.Scanner.NextError || std.json.Scanner.AllocError || std.json.Scanner.SkipError;

pub const Merge = struct { rank: u32, id: u32 };

pub const Model = struct {
    kind: Kind,
    unk_token: ?[]const u8 = null,
    prefix: ?[]const u8 = null,
    suffix: ?[]const u8 = null,
    fuse_unk: bool = false,
    byte_fallback: bool = false,
    ignore_merges: bool = false,
    max_chars: usize = 100,
    merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty,
    byte_ids: [256]?u32 = @splat(null),
    /// Ids at or past this are added tokens unless the model's vocabulary holds them.
    size: u32 = 0,
    added_only: std.StringHashMapUnmanaged(void) = .empty,

    pub fn empty(kind: Kind) Model {
        return .{ .kind = kind, .prefix = if (kind == .wordpiece) "##" else null, .unk_token = if (kind == .wordpiece) "[UNK]" else null };
    }

    /// Model-vocabulary id of `s`; added tokens outside the model vocabulary are invisible to the model.
    pub fn id(m: *const Model, vocab: *const Vocab, s: []const u8) ?u32 {
        const found = vocab.get(s) orelse return null;
        if (found >= m.size and m.added_only.contains(s)) return null;
        return found;
    }

    fn unk(m: *const Model, vocab: *const Vocab) Error!u32 {
        return m.id(vocab, m.unk_token orelse return error.UnknownTokenMissing) orelse error.UnknownTokenMissing;
    }

    pub fn tokenize(m: *const Model, ctx: *steps.Context, vocab: *const Vocab, word: []const u8, out: *std.ArrayList(u32)) Error!void {
        if (word.len == 0) return;
        switch (m.kind) {
            .bpe => try m.bpe(ctx, vocab, word, out),
            .wordpiece => try m.wordpiece(ctx, vocab, word, out),
            .wordlevel => try out.append(ctx.a, m.id(vocab, word) orelse try m.unk(vocab)),
        }
    }

    const Symbol = struct { id: u32, prev: i32, next: i32, alive: bool = true };
    const Candidate = struct { rank: u32, pos: u32, id: u32 };

    fn order(_: void, x: Candidate, y: Candidate) std.math.Order {
        if (x.rank != y.rank) return std.math.order(x.rank, y.rank);
        return std.math.order(x.pos, y.pos);
    }

    fn pair(m: *const Model, left: u32, right: u32) ?Merge {
        return m.merges.get(@as(u64, left) << 32 | right);
    }

    fn bpe(m: *const Model, ctx: *steps.Context, vocab: *const Vocab, word: []const u8, out: *std.ArrayList(u32)) Error!void {
        const a = ctx.a;
        if (m.ignore_merges) if (m.id(vocab, word)) |whole| return out.append(a, whole);
        var symbols: std.ArrayList(Symbol) = .empty;
        var pending_unk: ?u32 = null;
        var piece: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        while (i < word.len) {
            const len = unicode.decodeAt(word, i).len;
            var s = word[i .. i + len];
            if ((i > 0 and m.prefix != null) or (i + len == word.len and m.suffix != null)) {
                piece.clearRetainingCapacity();
                if (i > 0) try piece.appendSlice(a, m.prefix orelse "");
                try piece.appendSlice(a, s);
                if (i + len == word.len) try piece.appendSlice(a, m.suffix orelse "");
                s = piece.items;
            }
            i += len;
            if (m.id(vocab, s)) |found| {
                if (pending_unk) |u| try symbols.append(a, .{ .id = u, .prev = 0, .next = 0 });
                pending_unk = null;
                try symbols.append(a, .{ .id = found, .prev = 0, .next = 0 });
                continue;
            }
            if (m.byte_fallback) {
                // tokenizers adds byte tokens without flushing a pending unk, so the unk lands after them.
                const all = for (s) |b| {
                    if (m.byte_ids[b] == null) break false;
                } else true;
                if (all) {
                    for (s) |b| try symbols.append(a, .{ .id = m.byte_ids[b].?, .prev = 0, .next = 0 });
                    continue;
                }
            }
            if (m.unk_token != null) {
                if (pending_unk != null and m.fuse_unk) continue;
                if (pending_unk) |u| try symbols.append(a, .{ .id = u, .prev = 0, .next = 0 });
                pending_unk = try m.unk(vocab);
            }
        }
        if (pending_unk) |u| try symbols.append(a, .{ .id = u, .prev = 0, .next = 0 });
        const syms = symbols.items;
        for (syms, 0..) |*s, k| {
            s.prev = @as(i32, @intCast(k)) - 1;
            s.next = if (k + 1 < syms.len) @intCast(k + 1) else -1;
        }
        var queue = std.PriorityQueue(Candidate, void, order).empty;
        for (0..syms.len -| 1) |k| if (m.pair(syms[k].id, syms[k + 1].id)) |hit|
            try queue.push(a, .{ .rank = hit.rank, .pos = @intCast(k), .id = hit.id });
        while (queue.pop()) |top| {
            const s = &syms[top.pos];
            if (!s.alive or s.next < 0) continue;
            const right: usize = @intCast(s.next);
            const current = m.pair(s.id, syms[right].id) orelse continue;
            if (current.id != top.id) continue;
            s.id = top.id;
            s.next = syms[right].next;
            syms[right].alive = false;
            if (s.next >= 0) syms[@intCast(s.next)].prev = @intCast(top.pos);
            if (s.prev >= 0) if (m.pair(syms[@intCast(s.prev)].id, s.id)) |hit|
                try queue.push(a, .{ .rank = hit.rank, .pos = @intCast(s.prev), .id = hit.id });
            if (s.next >= 0) if (m.pair(s.id, syms[@intCast(s.next)].id)) |hit|
                try queue.push(a, .{ .rank = hit.rank, .pos = top.pos, .id = hit.id });
        }
        for (syms) |s| if (s.alive) try out.append(a, s.id);
    }

    fn wordpiece(m: *const Model, ctx: *steps.Context, vocab: *const Vocab, word: []const u8, out: *std.ArrayList(u32)) Error!void {
        const a = ctx.a;
        if ((std.unicode.utf8CountCodepoints(word) catch return error.InvalidTokenizer) > m.max_chars) return out.append(a, try m.unk(vocab));
        const mark = out.items.len;
        var piece: std.ArrayList(u8) = .empty;
        var start: usize = 0;
        while (start < word.len) {
            var end = word.len;
            const found = while (start < end) {
                piece.clearRetainingCapacity();
                if (start > 0) try piece.appendSlice(a, m.prefix orelse "");
                try piece.appendSlice(a, word[start..end]);
                if (m.id(vocab, piece.items)) |hit| break hit;
                end -= 1;
                while (word[end] & 0xC0 == 0x80) end -= 1;
            } else {
                out.shrinkRetainingCapacity(mark);
                return out.append(a, try m.unk(vocab));
            };
            try out.append(a, found);
            start = end;
        }
    }
};

fn nextString(scanner: *std.json.Scanner, a: Allocator) Error!?[]const u8 {
    return switch (try scanner.nextAlloc(a, .alloc_if_needed)) {
        .string => |s| s,
        .allocated_string => |s| s,
        .null => null,
        else => error.InvalidTokenizer,
    };
}

fn nextBool(scanner: *std.json.Scanner) Error!bool {
    return switch (try scanner.next()) {
        .true => true,
        .false => false,
        .null => false,
        else => error.InvalidTokenizer,
    };
}

fn nextCount(scanner: *std.json.Scanner) Error!?u64 {
    return switch (try scanner.next()) {
        .number => |n| std.fmt.parseInt(u64, n, 10) catch null,
        .null => null,
        else => error.InvalidTokenizer,
    };
}

/// Streams the "model" object: the vocabulary goes into `vocab` and `tokens`, strings into `keep`.
pub fn load(keep: Allocator, scratch: Allocator, scanner: *std.json.Scanner, vocab: *Vocab, tokens: anytype) Error!Model {
    if (try scanner.next() != .object_begin) return error.InvalidTokenizer;
    var kind: ?[]const u8 = null;
    var model = Model{ .kind = .bpe };
    var prefix: ?[]const u8 = null;
    var have_prefix = false;
    var unk: ?[]const u8 = null;
    var have_unk = false;
    var merges: std.ArrayList([2][]const u8) = .empty;
    while (true) {
        const key = switch (try scanner.nextAlloc(scratch, .alloc_if_needed)) {
            .object_end => break,
            .string => |s| s,
            .allocated_string => |s| s,
            else => return error.InvalidTokenizer,
        };
        if (std.mem.eql(u8, key, "type")) {
            kind = try nextString(scanner, scratch);
        } else if (std.mem.eql(u8, key, "vocab")) {
            if (try scanner.next() != .object_begin) return error.UnsupportedModel;
            while (true) {
                const token = switch (try scanner.nextAlloc(keep, .alloc_always)) {
                    .object_end => break,
                    .allocated_string => |s| s,
                    else => return error.InvalidTokenizer,
                };
                const value = (try nextCount(scanner)) orelse return error.InvalidTokenizer;
                const at: u32 = std.math.cast(u32, value) orelse return error.InvalidTokenizer;
                try vocab.put(token, at);
                try tokens.put(at, token);
            }
        } else if (std.mem.eql(u8, key, "merges")) {
            if (try scanner.next() != .array_begin) return error.InvalidTokenizer;
            while (true) {
                switch (try scanner.peekNextTokenType()) {
                    .array_end => {
                        _ = try scanner.next();
                        break;
                    },
                    .string => {
                        const line = (try nextString(scanner, scratch)).?;
                        if (std.mem.startsWith(u8, line, "#version")) continue;
                        var parts = std.mem.splitScalar(u8, line, ' ');
                        const left = parts.next().?;
                        const right = parts.next() orelse return error.InvalidTokenizer;
                        if (parts.next() != null) return error.InvalidTokenizer;
                        try merges.append(scratch, .{ left, right });
                    },
                    .array_begin => {
                        _ = try scanner.next();
                        const left = (try nextString(scanner, scratch)) orelse return error.InvalidTokenizer;
                        const right = (try nextString(scanner, scratch)) orelse return error.InvalidTokenizer;
                        if (try scanner.next() != .array_end) return error.InvalidTokenizer;
                        try merges.append(scratch, .{ left, right });
                    },
                    else => return error.InvalidTokenizer,
                }
            }
        } else if (std.mem.eql(u8, key, "unk_token")) {
            unk = if (try nextString(scanner, scratch)) |s| try keep.dupe(u8, s) else null;
            have_unk = true;
        } else if (std.mem.eql(u8, key, "continuing_subword_prefix")) {
            prefix = if (try nextString(scanner, scratch)) |s| try keep.dupe(u8, s) else null;
            have_prefix = true;
        } else if (std.mem.eql(u8, key, "end_of_word_suffix")) {
            model.suffix = if (try nextString(scanner, scratch)) |s| try keep.dupe(u8, s) else null;
        } else if (std.mem.eql(u8, key, "fuse_unk")) {
            model.fuse_unk = try nextBool(scanner);
        } else if (std.mem.eql(u8, key, "byte_fallback")) {
            model.byte_fallback = try nextBool(scanner);
        } else if (std.mem.eql(u8, key, "ignore_merges")) {
            model.ignore_merges = try nextBool(scanner);
        } else if (std.mem.eql(u8, key, "max_input_chars_per_word")) {
            model.max_chars = (try nextCount(scanner)) orelse 100;
        } else if (std.mem.eql(u8, key, "dropout")) {
            // BPE dropout samples merges at random, so no exact encoding exists.
            switch (try scanner.next()) {
                .null => {},
                .number => |n| if ((std.fmt.parseFloat(f64, n) catch 1) > 0) return error.UnsupportedModel,
                else => return error.InvalidTokenizer,
            }
        } else try scanner.skipValue();
    }
    const name = kind orelse return error.UnsupportedModel;
    model.kind = if (std.mem.eql(u8, name, "BPE")) .bpe else if (std.mem.eql(u8, name, "WordPiece")) .wordpiece else if (std.mem.eql(u8, name, "WordLevel")) .wordlevel else return error.UnsupportedModel;
    const defaults = Model.empty(model.kind);
    model.prefix = if (have_prefix) prefix else defaults.prefix;
    model.unk_token = if (have_unk) unk else if (model.kind == .wordlevel) "<unk>" else defaults.unk_token;
    model.size = @intCast(vocab.count());
    if (model.kind != .bpe) return model;
    if (model.prefix != null and model.prefix.?.len == 0) model.prefix = null;
    if (model.suffix != null and model.suffix.?.len == 0) model.suffix = null;
    if (model.byte_fallback) for (&model.byte_ids, 0..) |*slot, b| {
        var name_buf: [6]u8 = undefined;
        slot.* = vocab.get(std.fmt.bufPrint(&name_buf, "<0x{X:0>2}>", .{b}) catch unreachable);
    };
    const skip = if (model.prefix) |p| p.len else 0;
    var joined: std.ArrayList(u8) = .empty;
    try model.merges.ensureTotalCapacity(keep, @intCast(merges.items.len));
    for (merges.items, 0..) |m, rank| {
        const left = vocab.get(m[0]) orelse return error.InvalidTokenizer;
        const right = vocab.get(m[1]) orelse return error.InvalidTokenizer;
        if (m[1].len < skip) return error.InvalidTokenizer;
        joined.clearRetainingCapacity();
        try joined.appendSlice(scratch, m[0]);
        try joined.appendSlice(scratch, m[1][skip..]);
        const merged = vocab.get(joined.items) orelse return error.InvalidTokenizer;
        model.merges.putAssumeCapacity(@as(u64, left) << 32 | right, .{ .rank = @intCast(rank), .id = merged });
    }
    return model;
}
