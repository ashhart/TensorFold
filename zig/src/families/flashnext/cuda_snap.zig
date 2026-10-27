//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const kv = @import("exl3/kv/cuda_exl3_kv.zig");
const pages = @import("cuda_pages.zig");

const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
const state_bytes = wt.state_n * 4;
const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;

/// One attention cache's first `at` key, value and indexer-key rows, and the pooled blocks they complete.
fn attnBytes(at: usize) usize {
    return at * (2 * prompt.kv_row_bytes + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state after `at` tokens takes: the DeltaNet's are fixed, each attention cache grows by the token.
pub fn bytes(at: usize, head: bool) usize {
    return wt.linear_n * (conv_bytes + state_bytes) + (wt.attn_n + @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// `bytes` for a sequence whose K/V rows are `r` (int8 codes and scales, or one KV head a rank: cuda_exl3_kv.zig).
pub fn bytesKv(at: usize, head: bool, r: kv.Rows) usize {
    const layer = at * (2 * (r.kv + r.scales) + index_row) + at / wt.index_ratio * index_row;
    return wt.linear_n * (conv_bytes + state_bytes) + wt.attn_n * layer + @as(usize, @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// A sequence's K/V rows a token, read from its buffers (cap rows each).
fn seqRows(seq: *const prompt.Seq) kv.Rows {
    // borrowed page views carry no length: their rows are upstream's (bf16, two KV heads)
    const per = if (seq.paged != null or seq.cap == 0) 0 else seq.kc[0].len / seq.cap;
    if (per == 0) return kv.snapRows(false, 2);
    return .{ .kv = per, .scales = if (seq.kv8) |c| c.ks[0].len / seq.cap else 0 };
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs.
pub const Snap = struct {
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    /// The pinned host memory `buf` borrows (states parked on the host: `on_host`), or null (device memory).
    host: ?cuda.HostBuffer = null,
    at: usize,
    history: [2]i64,
    head: bool,
};

const Way = enum { save, restore };

const Piece = struct { bufs: []cuda.DeviceBuffer, n: usize };

/// An attention cache's pieces for `at` rows: complete pooled blocks only (the rest pool again as later rows land).
fn attnPieces(kc: []cuda.DeviceBuffer, vc: []cuda.DeviceBuffer, ikc: []cuda.DeviceBuffer, pooled: []cuda.DeviceBuffer, at: usize) [4]Piece {
    return attnPiecesRow(kc, vc, ikc, pooled, at, prompt.kv_row_bytes);
}

fn attnPiecesRow(kc: []cuda.DeviceBuffer, vc: []cuda.DeviceBuffer, ikc: []cuda.DeviceBuffer, pooled: []cuda.DeviceBuffer, at: usize, kv_row: usize) [4]Piece {
    return .{
        .{ .bufs = kc, .n = at * kv_row },
        .{ .bufs = vc, .n = at * kv_row },
        .{ .bufs = ikc, .n = at * index_row },
        .{ .bufs = pooled, .n = at / wt.index_ratio * index_row },
    };
}

/// Each piece of the sequence (and the head's cache, with `d`) to or from its place in the buffer at `base`.
fn move(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, base: u64, at: usize, way: Way) !void {
    const dr = stream.d;
    var off: u64 = 0;
    var pieces: [13]Piece = undefined;
    pieces[0] = .{ .bufs = seq.conv, .n = conv_bytes };
    pieces[1] = .{ .bufs = seq.state, .n = state_bytes };
    const rows = seqRows(seq);
    pieces[2..6].* = attnPiecesRow(seq.kc, seq.vc, seq.ikc, seq.pooled, at, rows.kv);
    pieces[6] = .{ .bufs = (&seq.tail)[0..1], .n = tail_bytes };
    var count: usize = 7;
    if (seq.kv8) |c| {
        // the int8 cache's scales (its codes went with kc/vc)
        pieces[7] = .{ .bufs = c.ks, .n = at * rows.scales };
        pieces[8] = .{ .bufs = c.vs, .n = at * rows.scales };
        count = 9;
    }
    if (d) |h| {
        pieces[count..][0..4].* = attnPieces((&h.kc)[0..1], (&h.vc)[0..1], (&h.ikc)[0..1], (&h.pooled)[0..1], at);
        count += 4;
    }
    for (pieces[0..count], 0..) |p, pi| for (p.bufs, 0..) |b, layer| {
        if (p.n == 0) continue;
        // the model's caches on the stream's pages; the head's too in a pool lane (the layer after the model's)
        const head = pi >= 7 and d != null and d.?.pg.pt != 0;
        if ((pi >= 2 and pi < 6) or head) if (seq.paged) |*pg| {
            // the same bytes in the same order as a contiguous cache's, a run a page
            const part: pages.Part = @fromBackingInt(@intCast(if (head) pi - 7 else pi - 2));
            const at_layer = if (head) wt.attn_n else layer;
            const row = pages.rowBytes(part);
            const per = pages.rows(part, pg.tokens);
            const n = p.n / row;
            var r: usize = 0;
            while (r < n) {
                const run = @min(per - r % per, n - r);
                const at_row = pg.rowAt(at_layer, part, r);
                const dst, const src = if (way == .save) .{ base + off, at_row } else .{ at_row, base + off };
                try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, run * row, stream.handle), "cuMemcpyDtoDAsync");
                off += run * row;
                r += run;
            }
            continue;
        };
        const dst, const src = if (way == .save) .{ base + off, b.ptr } else .{ b.ptr, base + off };
        try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
        off += p.n;
    };
    std.debug.assert(off == bytesKv(at, d != null, rows));
}

fn discard(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    cuda.completion.drain(stream.d, stream, "snapshot copies did not drain");
    if (snap.host) |*h| h.free() else snap.buf.free();
    gpa.destroy(snap);
}

/// Kept states go to mapped pinned host memory, set once at load (TF_FLASHNEXT_SNAP_HOST); the host budgets the cache.
pub var on_host: bool = false;

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    const at = seq.fed;
    if (at > seq.cap) return error.SequenceTooShort;
    if (d) |h| if (h.len != at or h.drafted != 0 or h.open or at > h.cap) return error.NotAtMark;
    const n = bytesKv(at, d != null, seqRows(seq));
    const snap = try allocate(gpa, driver, n, at, seq.history, d != null);
    errdefer discard(gpa, stream, snap);
    try move(stream, seq, d, snap.buf.ptr, at, .save);
    try stream.synchronize();
    return snap;
}

fn allocate(gpa: std.mem.Allocator, driver: *const cuda.Driver, n: usize, at: usize, history: [2]i64, head: bool) !*Snap {
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = undefined, .at = at, .history = history, .head = head };
    if (on_host) {
        var host = try cuda.HostBuffer.allocMapped(driver, @max(n, 256));
        errdefer host.free();
        snap.buf = .{ .d = driver, .ptr = try host.device(), .len = n, .borrowed = true };
        snap.host = host;
    } else snap.buf = try cuda.DeviceBuffer.alloc(driver, n);
    return snap;
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (d) |h| if (!snap.head or snap.at > h.cap) return error.NoHeadState;
    // a state with the head's rows serves a stream without drafts too: those rows stay in the buffer
    try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
        h.open = false;
        h.fed0 = snap.at;
    }
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    discard(gpa, stream, snap);
}

test "a placed state's bytes: int8 codes and scales or one KV head a token, the head's rows upstream's" {
    const r = kv.snapRows(true, 2);
    try std.testing.expectEqual(@as(usize, 12 * (2 * (512 + 32) + 128 * 2)), bytesKv(1, false, r) - bytesKv(0, false, r));
    try std.testing.expectEqual(bytesKv(1, false, r) - bytesKv(0, false, r) + (2 * 2 * 256 * 2 + 128 * 2), bytesKv(1, true, r) - bytesKv(0, true, r));
    try std.testing.expectEqual(bytes(7, true), bytesKv(7, true, kv.snapRows(false, 2)));
    const half = kv.snapRows(false, 1);
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2)), bytesKv(1, false, half) - bytesKv(0, false, half));
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, false) - bytes(0, false));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, false) - bytes(0, false));
    // the head's cache is one more attention layer's rows
    try std.testing.expectEqual(@as(usize, 13 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true) - bytes(0, true));
    try std.testing.expect(bytes(0, false) > 100 << 20 and bytes(0, false) < 120 << 20);
}

test {
    _ = @import("cuda_paging_snap_test.zig");
}
