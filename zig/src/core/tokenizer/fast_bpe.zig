//! Fast single-thread ByteLevel-BPE encoder with reusable scratch, short-word merges and piece cache.
const std = @import("std");
const models = @import("model.zig");
const steps = @import("steps.zig");
const fast_split = @import("fast_split.zig");
const Allocator = std.mem.Allocator;

const cache_slots = 512;
const small_bpe_limit = 24;

const Symbol = struct { id: u32, prev: i32, next: i32, alive: bool = true };
const Candidate = struct { rank: u32, pos: u32, id: u32 };

fn order(_: void, x: Candidate, y: Candidate) std.math.Order {
    if (x.rank != y.rank) return std.math.order(x.rank, y.rank);
    return std.math.order(x.pos, y.pos);
}

const ByteUtf8 = struct { bytes: [2]u8, len: u2 };

const byte_utf8: [256]ByteUtf8 = blk: {
    @setEvalBranchQuota(10_000);
    var table: [256]ByteUtf8 = undefined;
    for (0..256) |b| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(steps.byte_chars[b], &buf) catch unreachable;
        std.debug.assert(n <= 2);
        table[b] = .{ .bytes = .{ buf[0], if (n == 2) buf[1] else 0 }, .len = @intCast(n) };
    }
    break :blk table;
};

/// Precomputed ByteLevel fast-path state for a Tokenizer with a verified fast regex split.
pub const FastPath = struct {
    split: fast_split.Kind,
    byte_ids: [256]u32,
};

/// Detects whether `model`, `normalizer` and `pre_tokenizer` qualify for the fused ByteLevel BPE fast path.
pub fn detect(
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    normalizer: ?steps.Normalizer,
    pre_tokenizer: ?steps.PreTokenizer,
) ?FastPath {
    if (model.kind != .bpe or model.prefix != null or model.suffix != null or normalizer != null) return null;
    const pre = pre_tokenizer orelse return null;
    if (pre != .sequence or pre.sequence.len != 2) return null;
    const first = pre.sequence[0];
    const second = pre.sequence[1];
    if (first != .split or first.split.behavior != .isolated or first.split.invert) return null;
    if (first.split.pattern != .regex) return null;
    const split_kind = first.split.fast_split orelse return null;
    if (second != .byte_level or second.byte_level.add_prefix_space or second.byte_level.regex != null) return null;
    var byte_ids: [256]u32 = undefined;
    for (0..256) |b| {
        const u = byte_utf8[b];
        byte_ids[b] = model.id(vocab, u.bytes[0..u.len]) orelse return null;
    }
    return .{ .split = split_kind, .byte_ids = byte_ids };
}

const CacheSlot = struct {
    hash: u64 = 0,
    src_start: u32 = 0,
    src_len: u32 = 0,
    out_start: u32 = 0,
    out_len: u16 = 0,
};

/// Per-thread reusable buffers and piece cache; avoids per-word allocations on normal pieces.
pub const Scratch = struct {
    bl_buf: [512]u8 = undefined,
    bl_heap: std.ArrayList(u8) = .empty,
    symbols: std.ArrayList(Symbol) = .empty,
    queue: std.PriorityQueue(Candidate, void, order) = .empty,
    cache: [cache_slots]CacheSlot = @splat(.{}),

    pub fn deinit(s: *Scratch, a: Allocator) void {
        s.bl_heap.deinit(a);
        s.symbols.deinit(a);
        s.queue.deinit(a);
    }

    pub fn resetCache(s: *Scratch) void {
        s.cache = @splat(.{});
    }
};

fn byteLevelEncode(s: *Scratch, a: Allocator, piece: []const u8) Allocator.Error![]const u8 {
    if (piece.len * 2 <= s.bl_buf.len) {
        var n: usize = 0;
        for (piece) |b| {
            const u = byte_utf8[b];
            s.bl_buf[n] = u.bytes[0];
            if (u.len == 2) s.bl_buf[n + 1] = u.bytes[1];
            n += u.len;
        }
        return s.bl_buf[0..n];
    }
    s.bl_heap.clearRetainingCapacity();
    try s.bl_heap.ensureTotalCapacity(a, piece.len * 2);
    for (piece) |b| {
        const u = byte_utf8[b];
        s.bl_heap.appendAssumeCapacity(u.bytes[0]);
        if (u.len == 2) s.bl_heap.appendAssumeCapacity(u.bytes[1]);
    }
    return s.bl_heap.items;
}

inline fn pairMerge(model: *const models.Model, left: u32, right: u32) ?models.Merge {
    return model.merges.get(@as(u64, left) << 32 | right);
}

fn bpeSmall(model: *const models.Model, syms: []Symbol, a: Allocator, out: *std.ArrayList(u32)) Allocator.Error!void {
    while (true) {
        var best_rank: u32 = std.math.maxInt(u32);
        var best_id: u32 = 0;
        var best_left: usize = 0;
        var best_right: usize = 0;
        var i: i32 = 0;
        while (i >= 0) {
            const li: usize = @intCast(i);
            const ni = syms[li].next;
            if (ni < 0) break;
            const ri: usize = @intCast(ni);
            if (pairMerge(model, syms[li].id, syms[ri].id)) |hit| {
                if (hit.rank < best_rank) {
                    best_rank = hit.rank;
                    best_id = hit.id;
                    best_left = li;
                    best_right = ri;
                }
            }
            i = ni;
        }
        if (best_rank == std.math.maxInt(u32)) break;
        syms[best_left].id = best_id;
        syms[best_left].next = syms[best_right].next;
        syms[best_right].alive = false;
        if (syms[best_left].next >= 0) syms[@intCast(syms[best_left].next)].prev = @intCast(best_left);
    }
    var i: i32 = 0;
    while (i >= 0) {
        const idx: usize = @intCast(i);
        try out.append(a, syms[idx].id);
        i = syms[idx].next;
    }
}

fn bpeHeap(model: *const models.Model, s: *Scratch, syms: []Symbol, a: Allocator, out: *std.ArrayList(u32)) Allocator.Error!void {
    s.queue.clearRetainingCapacity();
    for (0..syms.len - 1) |k| {
        if (pairMerge(model, syms[k].id, syms[k + 1].id)) |hit|
            try s.queue.push(a, .{ .rank = hit.rank, .pos = @intCast(k), .id = hit.id });
    }
    while (s.queue.pop()) |top| {
        const sym = &syms[top.pos];
        if (!sym.alive or sym.next < 0) continue;
        const right: usize = @intCast(sym.next);
        const current = pairMerge(model, sym.id, syms[right].id) orelse continue;
        if (current.id != top.id) continue;
        sym.id = top.id;
        sym.next = syms[right].next;
        syms[right].alive = false;
        if (sym.next >= 0) syms[@intCast(sym.next)].prev = @intCast(top.pos);
        if (sym.prev >= 0) if (pairMerge(model, syms[@intCast(sym.prev)].id, sym.id)) |hit|
            try s.queue.push(a, .{ .rank = hit.rank, .pos = @intCast(sym.prev), .id = hit.id });
        if (sym.next >= 0) if (pairMerge(model, sym.id, syms[@intCast(sym.next)].id)) |hit|
            try s.queue.push(a, .{ .rank = hit.rank, .pos = top.pos, .id = hit.id });
    }
    for (syms) |sym| if (sym.alive) try out.append(a, sym.id);
}

/// Encodes one regex-split `span` from `text`, checking the local direct-mapped piece cache first.
pub fn encodePiece(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    s: *Scratch,
    a: Allocator,
    text: []const u8,
    span: [2]usize,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    const piece = text[span[0]..span[1]];
    if (piece.len == 0) return;
    if (piece.len == 1) {
        try out.append(a, fast.byte_ids[piece[0]]);
        return;
    }

    const can_cache = piece.len <= 64 and text.len <= std.math.maxInt(u32) and out.items.len <= std.math.maxInt(u32);
    var h: u64 = 0;
    if (can_cache) {
        h = std.hash.Wyhash.hash(0, piece) | 1;
        const slot = &s.cache[h & (cache_slots - 1)];
        if (slot.hash == h and slot.src_len == piece.len and std.mem.eql(u8, text[slot.src_start..][0..slot.src_len], piece)) {
            try out.ensureUnusedCapacity(a, slot.out_len);
            out.appendSliceAssumeCapacity(out.items[slot.out_start..][0..slot.out_len]);
            return;
        }
    }

    const out_before = out.items.len;
    if (model.ignore_merges) {
        const bl = try byteLevelEncode(s, a, piece);
        if (model.id(vocab, bl)) |whole| {
            try out.append(a, whole);
            if (can_cache) {
                s.cache[h & (cache_slots - 1)] = .{
                    .hash = h,
                    .src_start = @intCast(span[0]),
                    .src_len = @intCast(piece.len),
                    .out_start = @intCast(out_before),
                    .out_len = 1,
                };
            }
            return;
        }
    }

    s.symbols.clearRetainingCapacity();
    try s.symbols.ensureTotalCapacity(a, piece.len);
    for (piece, 0..) |b, k| {
        s.symbols.appendAssumeCapacity(.{
            .id = fast.byte_ids[b],
            .prev = @as(i32, @intCast(k)) - 1,
            .next = if (k + 1 < piece.len) @intCast(k + 1) else -1,
        });
    }
    const syms = s.symbols.items;
    if (syms.len <= small_bpe_limit) {
        try bpeSmall(model, syms, a, out);
    } else {
        try bpeHeap(model, s, syms, a, out);
    }
    if (can_cache) {
        const produced = out.items.len - out_before;
        if (produced <= std.math.maxInt(u16)) {
            s.cache[h & (cache_slots - 1)] = .{
                .hash = h,
                .src_start = @intCast(span[0]),
                .src_len = @intCast(piece.len),
                .out_start = @intCast(out_before),
                .out_len = @intCast(produced),
            };
        }
    }
}

/// Scans and encodes a raw `text` slice sequentially with one reusable `Scratch`.
pub fn encodeSlice(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    a: Allocator,
    text: []const u8,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    var s: Scratch = .{};
    defer s.deinit(a);
    try out.ensureUnusedCapacity(a, text.len / 3 + 16);
    var pos: usize = 0;
    while (fast_split.next(fast.split, text, &pos)) |span| {
        try encodePiece(fast, model, vocab, &s, a, text, span, out);
    }
}

/// Encodes pre-computed regex `spans` inside `text` sequentially with one reusable `Scratch`.
pub fn encodeSpans(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    a: Allocator,
    text: []const u8,
    spans: []const [2]usize,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    var s: Scratch = .{};
    defer s.deinit(a);
    s.resetCache();
    try out.ensureUnusedCapacity(a, spans.len + 16);
    for (spans) |span| {
        try encodePiece(fast, model, vocab, &s, a, text, span, out);
    }
}

/// Contiguous slice of an added-token split pass (`id != null` emits a literal token id).
pub const Segment = struct { start: usize, end: usize, id: ?u32 };

/// Encodes already-split added/text `segs` sequentially while sharing `Scratch` across text runs.
pub fn encodeSegmentsSerial(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    a: Allocator,
    text: []const u8,
    segs: []const Segment,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    var s: Scratch = .{};
    defer s.deinit(a);
    var total: usize = 0;
    for (segs) |seg| total += if (seg.id != null) 3 else seg.end - seg.start;
    try out.ensureUnusedCapacity(a, total / 3 + segs.len + 16);
    for (segs) |seg| {
        if (seg.id) |id| {
            try out.append(a, id);
            continue;
        }
        if (seg.end == seg.start) continue;
        var pos: usize = 0;
        const slice = text[seg.start..seg.end];
        while (fast_split.next(fast.split, slice, &pos)) |sp| {
            try encodePiece(fast, model, vocab, &s, a, text, .{ seg.start + sp[0], seg.start + sp[1] }, out);
        }
    }
}
