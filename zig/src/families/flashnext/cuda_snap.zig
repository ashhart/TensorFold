//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const pages = @import("cuda_pages.zig");
const sizes = @import("cuda_seq_bytes.zig");
const Heads = @import("cuda_heads.zig").Heads;

const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;
/// The MTP head's cache: its own layer, whole on every rank.
const head_heads: Heads = .full;

/// A state's shape for its disk image (cuda_disk.zig, one rank, the whole layer): layer counts and bytes a row.
pub const shape = .{ .linear = wt.linear_n, .attn = wt.attn_n, .conv = sizes.convBytes(.full), .state = sizes.stateBytes(.full), .kv_row = Heads.full.kvRow(), .index_row = index_row, .ratio = wt.index_ratio, .tail = tail_bytes };

/// One attention cache's first `at` key, value and indexer-key rows at heads `h`, and the pooled blocks they complete.
fn attnBytes(at: usize, h: Heads) usize {
    return at * (2 * h.kvRow() + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state at the model's heads `h` after `at` tokens takes: fixed DeltaNet pieces, attention by token.
pub fn bytes(at: usize, head: bool, h: Heads) usize {
    const own = wt.linear_n * (sizes.convBytes(h) + sizes.stateBytes(h)) + wt.attn_n * attnBytes(at, h) + tail_bytes;
    return own + if (head) attnBytes(at, head_heads) else 0;
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs.
pub const Snap = struct {
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    at: usize,
    history: [2]i64,
    head: bool,
    /// The model's heads its pieces hold (the saved sequence's): a sequence of other heads cannot take it.
    heads: Heads = .full,
};

const Way = enum { save, restore };

/// A piece `move` copies: `layers` buffers of `n` bytes, rows of `row` bytes (0: whole) a `per`-position block each.
pub const Piece = struct { layers: usize, n: usize, row: usize = 0, per: usize = 1 };

/// `move`'s pieces in order at `at` and the model's heads `h` (the disk image's at full): DeltaNet, attention, tail.
pub fn order(at: usize, head: bool, h: Heads) struct { p: [11]Piece, len: usize } {
    var o: [11]Piece = undefined;
    o[0] = .{ .layers = wt.linear_n, .n = sizes.convBytes(h) };
    o[1] = .{ .layers = wt.linear_n, .n = sizes.stateBytes(h) };
    o[2..6].* = attnOrder(wt.attn_n, at, h);
    o[6] = .{ .layers = 1, .n = tail_bytes };
    if (head) o[7..11].* = attnOrder(1, at, head_heads);
    return .{ .p = o, .len = if (head) 11 else 7 };
}

/// An attention cache's pieces for `at` rows: complete pooled blocks only (the rest pool again as later rows land).
fn attnOrder(layers: usize, at: usize, h: Heads) [4]Piece {
    return .{
        .{ .layers = layers, .n = at * h.kvRow(), .row = h.kvRow() },
        .{ .layers = layers, .n = at * h.kvRow(), .row = h.kvRow() },
        .{ .layers = layers, .n = at * index_row, .row = index_row },
        .{ .layers = layers, .n = at / wt.index_ratio * index_row, .row = index_row, .per = wt.index_ratio },
    };
}

const W = @import("core").state_walk.Device;

/// A state's parts at most: every layer of the 11 pieces `order` names.
const max_parts = 2 * wt.linear_n + 4 * wt.attn_n + 1 + 4;

/// The sequence's pieces (the head's too, with `d`) as walk parts in `order`'s order: whole, or rows on its pages.
fn parts(seq: *prompt.Seq, d: ?*Draft, at: usize, out: *[max_parts]W.Part) ![]const W.Part {
    const o = order(at, d != null, seq.heads);
    var bufs: [11][]cuda.DeviceBuffer = undefined;
    bufs[0..7].* = .{ seq.conv, seq.state, seq.kc, seq.vc, seq.ikc, seq.pooled, (&seq.tail)[0..1] };
    if (d) |h| bufs[7..11].* = .{ (&h.kc)[0..1], (&h.vc)[0..1], (&h.ikc)[0..1], (&h.pooled)[0..1] };
    var n: usize = 0;
    for (o.p[0..o.len], bufs[0..o.len], 0..) |p, layers, pi| {
        if (layers.len != p.layers) return error.SnapShape;
        for (layers, 0..) |b, layer| {
            if (p.n == 0) continue;
            // the model's caches on the stream's pages; the head's too in a pool lane (the layer after the model's)
            const head = pi >= 7 and d != null and d.?.pg.pt != 0;
            if ((pi >= 2 and pi < 6) or head) if (seq.paged) |*pg| {
                const part: pages.Part = @fromBackingInt(@intCast(if (head) pi - 7 else pi - 2));
                const row = pages.rowBytes(part);
                if (row != p.row) return error.SnapShape;
                const span = pages.rows(part, pg.tokens) * row;
                const live = pg.at(if (head) wt.attn_n else layer, part);
                out[n] = .{ .live = live, .len = p.n, .span = span, .page = pages.pageBytesOf(pg.tokens, pg.layers) };
                n += 1;
                continue;
            };
            out[n] = .{ .live = b.ptr, .len = p.n };
            n += 1;
        }
    }
    return out[0..n];
}

/// The parts to (`save`) or from the buffer at `base`, queued on `stream`: the core walk, a copy a piece.
fn move(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, base: u64, at: usize, way: Way) !void {
    var store: [max_parts]W.Part = undefined;
    const ps = try parts(seq, d, at, &store);
    std.debug.assert(W.total(ps) == bytes(at, d != null, seq.heads));
    const ids: []const u32 = if (seq.paged) |*pg| pg.ids else &.{};
    try W.each(ps, ids, .{ .base = base }, Copy{ .s = stream, .save = way == .save }, Copy.run);
}

const Copy = struct {
    s: cuda.Stream,
    save: bool,
    fn run(c: Copy, live: u64, kept: u64, n: usize) anyerror!void {
        const dst, const src = if (c.save) .{ kept, live } else .{ live, kept };
        try c.s.d.check(c.s.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, c.s.handle), "cuMemcpyDtoDAsync");
    }
};

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    const at = seq.fed;
    if (d) |h| if (h.len != at or h.drafted != 0) return error.NotAtMark;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, bytes(at, d != null, seq.heads)), .at = at, .history = seq.history, .head = d != null, .heads = seq.heads };
    errdefer snap.buf.free();
    errdefer cuda.completion.drain(driver, stream, "a failed snapshot save did not drain");
    try move(stream, seq, d, snap.buf.ptr, at, .save);
    try stream.synchronize();
    return snap;
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (!snap.heads.eql(seq.heads)) return error.SnapShape;
    if (d) |h| if (!snap.head or snap.at > h.cap) return error.NoHeadState;
    // a state with the head's rows serves a stream without drafts too: those rows stay in the buffer
    try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
    }
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    cuda.completion.drain(stream.d, stream, "a snapshot drop did not drain");
    snap.buf.free();
    gpa.destroy(snap);
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, false, .full) - bytes(0, false, .full));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, false, .full) - bytes(0, false, .full));
    // the head's cache is one more attention layer's rows
    try std.testing.expectEqual(@as(usize, 13 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true, .full) - bytes(0, true, .full));
    try std.testing.expect(bytes(0, false, .full) > 100 << 20 and bytes(0, false, .full) < 120 << 20);
}

test "a state at a rank's half of the heads: half the conv windows, states and KV rows, the indexer and the head's whole" {
    const h = Heads.half;
    try std.testing.expectEqual(@as(usize, 36 * (3 * 5120 * 2 + 24 * 128 * 128 * 4) + 9 * 4 * 2560 * 2), bytes(0, false, h));
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2)), bytes(1, false, h) - bytes(0, false, h));
    // the head's cache keeps its two KV heads beside the model's one
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2) + (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true, h) - bytes(0, true, h));
    const o = order(8, true, h);
    try std.testing.expectEqual(@as(usize, 3 * 5120 * 2), o.p[0].n);
    try std.testing.expectEqual(@as(usize, 24 * 128 * 128 * 4), o.p[1].n);
    for (o.p[2..4]) |p| try std.testing.expectEqual(@as(usize, 512), p.row);
    for (o.p[7..9]) |p| try std.testing.expectEqual(@as(usize, 1024), p.row);
    var sum: usize = 0;
    for (o.p[0..o.len]) |p| sum += p.layers * p.n;
    try std.testing.expectEqual(bytes(8, true, h), sum);
}

test "a state of the whole layer restores into no sequence of half the heads, and the reverse" {
    var driver: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_][2]Heads{ .{ .full, .half }, .{ .half, .full } }) |pair| {
        const kept: Snap = .{ .buf = .{ .d = &driver, .ptr = 0x100, .len = 1, .borrowed = true }, .at = 1, .history = .{ 0, 0 }, .head = false, .heads = pair[0] };
        var seq: prompt.Seq = .{ .heads = pair[1], .conv = &.{}, .state = &.{}, .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = .{ .d = &driver, .ptr = 0, .len = 0 }, .history = .{ 0, 0 }, .fed = 0, .logits = &.{}, .cap = 8 };
        // refused before any copy: no driver call is made
        try std.testing.expectError(error.SnapShape, restore(stream, &seq, null, &kept));
        try std.testing.expectEqual(@as(usize, 0), seq.fed);
    }
}

test {
    _ = @import("cuda_paging_snap_test.zig");
}
