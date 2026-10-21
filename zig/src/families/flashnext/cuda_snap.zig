//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.
//!
//! BF16 paged snapshot format and ordering contract (version 1, exact pager geometry).
//!
//! A paged sequence is not a dense kc/vc allocation. Its kc/vc views alias compact device windows. Dividing their
//! lengths by seq.cap, or copying at * row_bytes from them, cannot describe the logical history. The host store
//! alone is also insufficient: rows still in each device window have not necessarily been written back.
//!
//! Format: keep the dense/int8 format separate, with an explicit version and paging discriminator in Snap. The
//! paged header records at, history, head presence, BF16 row width, attention-layer count, index ratio, cap,
//! window/rows/slots, wr and nblocks, plus checked offsets/lengths for every section. No stored GPU pointers are
//! executable after restore. The initial exact-state format requires equal pager geometry and capacity on the
//! destination; reject a mismatch before touching it. A future capacity-changing format must explicitly re-encode
//! slot2blk's layer * nblocks keys and rebuild both inverse tables, rather than copying those keys unchanged.
//!
//! Payload, at one completed sequence boundary:
//! - Existing convolution, recurrent state, PLE tail, IKC [0,at), complete pooled blocks and optional MTP head
//!   state. Include partially pooled raw IKC rows; they are needed when the next block completes.
//! - Canonical BF16 host K/V [layer][K|V][0,at), after flushing every pending window interval [win0[layer],at).
//! - The entire compact device K/V allocation, including all windows and shared slot payloads, and the entire
//!   tabs allocation: blk2slot, mark, ctl (including global clock, sticky status and per-layer epoch), slot2blk,
//!   stamp. Copying only host history and resetting tables would lose the requested mappings and LRU state.
//! - Host win0, rebases and written arrays. graph_prepared is transient execution state and restores as false.
//!   work lists and status_host are scratch: rebuild/overwrite them before use; never restore stale completion
//!   status as proof that an operation completed. Logical rows >= at are invalid irrespective of their bytes.
//!
//! Save ordering:
//! 1. Exclude scheduling/replay for this sequence. Establish an event dependency from every producer (including
//!    prefill reduction and copy streams) onto the snapshot stream; synchronizing this stream alone is insufficient
//!    unless the caller already joined those producers. Reject an outstanding draft or inconsistent head mark.
//! 2. Validate ALL layer ranges, geometry and checked sizes before any copy; require win0 aligned to index_ratio,
//!    win0 <= at <= min(cap,win0+wr), and reject a sticky paging failure after a completed status transfer.
//! 3. Allocate the budgeted snapshot, enqueue window writebacks, then canonical host-prefix copies, device K/V,
//!    tables and ordinary state on the ordered stream. Snapshot storage owns its copies; it never aliases a live
//!    pager. Preserve counters as well as data; flush itself does not advance epochs or rebase windows.
//! 4. Wait for the copies before publishing the immutable snapshot or releasing source/storage ownership. On a
//!    submission failure, drain already submitted copies before freeing their targets. If draining fails, retain
//!    ownership until the device/session is quiesced. A partial snapshot must never reach the prompt cache.
//!
//! Restore ordering:
//! 1. Validate format, bounds, byte counts, pager geometry, sequence capacity and optional MTP head availability
//!    before changing the destination. Drain prior uses and pending graph instantiation before overwriting state.
//! 2. Invalidate MTP verification generation before the first mutation, so previously published or delayed results
//!    cannot be consumed. Restore into the destination's own allocations; never transplant snapshot addresses.
//! 3. Copy canonical host rows, compact device payload, tables, IKC/pooled/recurrent/tail/head state, then wait.
//!    Restore host origins/counters only after device completion; clear graph_prepared and refresh paging status.
//!    Matching tables, slot payloads and clock must become visible together before another attention call.
//! 4. Commit seq.fed/history and reset Draft's transient drafted/rows state and len only after successful completion.
//!    On a failure after mutation, discard/poison the destination rather than publishing its previous position over
//!    partially restored memory. The snapshot remains immutable and reusable. Destroy snapshots only after their
//!    last asynchronous reader completes; dropping a source sequence must not invalidate saved storage.
//!
//! Integration: the engine freezes pager configuration at load; serving lanes use the configured, budgeted capacity.
//! snapBytes charges the canonical prefix, windows/slots, tables, ordinary state, next logits, optional head and header.
//! Snapshots own mapped host storage. The scheduler serializes operations; context completion joins all CUDA producers.
//! cuda_rank_engine uses prepare/copy/commit acknowledgements before returning a restored pair to scheduling.
//! releaseSeq joins pending graph instantiations and discards captures that reference the destination sequence.
//! Failed submissions drain through CUDA's completion policy before freeing storage; an undrainable context terminates
//! the rank through the existing completion handler. A partially restored sequence remains invalid for execution.
//!
//! Acceptance gates are specified in zig/tests/cuda/exl3_kvpage_tests.zig; host fakes alone do not establish GPU or
//! serving correctness. The snapshot command checks device lifecycle; flashnext-exl3-graph snapshot checks TP2 parity.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const kv = @import("cuda_exl3_kv.zig");
const pager = @import("cuda_exl3_kvpage.zig");

const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
const state_bytes = wt.state_n * 4;
const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;

test "exl3: paged snapshot host cases" {
    _ = @import("cuda_snap_test.zig");
}

/// One attention cache's first `at` key, value and indexer-key rows, and the pooled blocks they complete.
fn attnBytes(at: usize) usize {
    return at * (2 * prompt.kv_row_bytes + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state after `at` tokens takes: the DeltaNet's are fixed, each attention cache grows by the token.
pub fn bytes(at: usize, head: bool) usize {
    return wt.linear_n * (conv_bytes + state_bytes) + (wt.attn_n + @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// `bytes` for a sequence whose K/V rows are `r` (int8 codes and scales, or one KV head a rank: cuda_exl3_kv.zig);
/// the head's cache stays upstream's.
pub fn bytesKv(at: usize, head: bool, r: kv.Rows) usize {
    const layer = at * (2 * (r.kv + r.scales) + index_row) + at / wt.index_ratio * index_row;
    return wt.linear_n * (conv_bytes + state_bytes) + wt.attn_n * layer + @as(usize, @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// A sequence's K/V rows a token, read from its buffers (cap rows each).
fn seqRows(seq: *const prompt.Seq) kv.Rows {
    return .{ .kv = seq.kc[0].len / seq.cap, .scales = if (seq.kv8) |c| c.ks[0].len / seq.cap else 0 };
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs.
pub const Snap = struct {
    version: u32 = 1,
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    /// The pinned host memory `buf` borrows (states parked on the host: `on_host`), or null (device memory).
    host: ?cuda.HostBuffer = null,
    at: usize,
    history: [2]i64,
    head: bool,
    paged: ?pager.Snapshot = null,
    owned_blocks: ?pager.shared_contract.OwnedBlocks = null,
};

pub fn pagedBytes(cap: usize, cfg: pager.Config, at: usize, head: bool) !usize {
    const layout = try pager.SnapshotLayout.init(cap, cfg, at);
    const raw = try std.math.mul(usize, at, wt.attn_n * index_row);
    const pooled = try std.math.mul(usize, at / wt.index_ratio, wt.attn_n * index_row);
    const ordinary = try std.math.add(usize, wt.linear_n * (conv_bytes + state_bytes) + tail_bytes, try std.math.add(usize, raw, pooled));
    const head_bytes = if (head) try std.math.add(usize, try std.math.add(usize, try std.math.mul(usize, at, 2 * prompt.kv_row_bytes + index_row), try std.math.mul(usize, at / wt.index_ratio, index_row)), if (at == 0) 0 else wt.wide * 2) else 0;
    return try std.math.add(usize, try pager.ownedSnapshotBytes(cfg), try std.math.add(usize, @sizeOf(Snap) + prompt.vocabulary * 2, try std.math.add(usize, layout.total, try std.math.add(usize, ordinary, head_bytes))));
}

fn payloadBytes(image: pager.Snapshot, head: bool) !usize {
    return try pagedBytes(image.layout.cap, image.layout.cfg, image.layout.at, head) - @sizeOf(Snap) - try pager.ownedSnapshotBytes(image.layout.cfg);
}

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
    if (seq.paged != null) return error.KvPageSnapshot; // paged K/V (cuda_exl3_kvpage.zig) has no snapshot yet
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
    for (pieces[0..count]) |p| for (p.bufs) |b| {
        if (p.n == 0) continue;
        const dst, const src = if (way == .save) .{ base + off, b.ptr } else .{ b.ptr, base + off };
        try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
        off += p.n;
    };
    std.debug.assert(off == bytesKv(at, d != null, rows));
}

fn movePaged(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap, way: Way) !void {
    const image = snap.paged.?;
    const pg = seq.paged.?;
    try pg.moveSnapshot(stream, snap.buf.ptr, image, way == .save);
    var offset = image.layout.total;
    const pieces = [_]Piece{
        .{ .bufs = seq.conv, .n = conv_bytes },
        .{ .bufs = seq.state, .n = state_bytes },
        .{ .bufs = seq.ikc, .n = snap.at * index_row },
        .{ .bufs = seq.pooled, .n = snap.at / wt.index_ratio * index_row },
        .{ .bufs = (&seq.tail)[0..1], .n = tail_bytes },
    };
    for (pieces) |piece| for (piece.bufs) |buffer| {
        try copyPiece(stream, snap.buf.ptr + offset, buffer.ptr, piece.n, way);
        offset += piece.n;
    };
    if (snap.head) {
        const head_n = attnBytes(snap.at) + (if (snap.at == 0) @as(usize, 0) else wt.wide * 2);
        if (d) |head| {
            const head_pieces = attnPieces((&head.kc)[0..1], (&head.vc)[0..1], (&head.ikc)[0..1], (&head.pooled)[0..1], snap.at);
            for (head_pieces) |piece| for (piece.bufs) |buffer| {
                try copyPiece(stream, snap.buf.ptr + offset, buffer.ptr, piece.n, way);
                offset += piece.n;
            };
            if (snap.at > 0) {
                try copyPiece(stream, snap.buf.ptr + offset, if (way == .save) head.prompt_last else head.streams.ptr, wt.wide * 2, way);
                offset += wt.wide * 2;
            }
        } else offset += head_n;
    }
    std.debug.assert(offset + seq.logits.len == snap.buf.len);
}

fn copyPiece(stream: cuda.Stream, stored: u64, live: u64, len: usize, way: Way) !void {
    if (len == 0) return;
    const dst, const src = if (way == .save) .{ stored, live } else .{ live, stored };
    try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, stream.handle), "cuMemcpyDtoDAsync");
}

fn quiesce(stream: cuda.Stream) !void {
    try stream.d.synchronize();
}

fn validatePagedSeq(seq: *const prompt.Seq, d: ?*Draft, at: usize) !void {
    if (seq.cap != seq.paged.?.cap or seq.conv.len != wt.linear_n or seq.state.len != wt.linear_n or seq.ikc.len != wt.attn_n or seq.pooled.len != wt.attn_n or seq.tail.len < tail_bytes) return error.KvPageSnapshotLayout;
    const pieces = [_]Piece{ .{ .bufs = seq.conv, .n = conv_bytes }, .{ .bufs = seq.state, .n = state_bytes }, .{ .bufs = seq.ikc, .n = at * index_row }, .{ .bufs = seq.pooled, .n = at / wt.index_ratio * index_row } };
    for (pieces) |piece| for (piece.bufs) |buffer| if (buffer.len < piece.n) return error.KvPageSnapshotLayout;
    if (d) |head| {
        const head_pieces = attnPieces((&head.kc)[0..1], (&head.vc)[0..1], (&head.ikc)[0..1], (&head.pooled)[0..1], at);
        for (head_pieces) |piece| for (piece.bufs) |buffer| if (buffer.len < piece.n) return error.NoHeadState;
    }
}

fn discard(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    cuda.completion.drain(stream.d, stream, "snapshot copies did not drain");
    if (snap.owned_blocks) |*owned| owned.deinit(gpa);
    if (snap.host) |*h| h.free() else snap.buf.free();
    gpa.destroy(snap);
}

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
/// Kept states go to mapped pinned host memory (copied at ~14 GB/s by the same DtoD copies; the device keeps none of
/// them): set once at load (cuda_lanes_load.zig, TF_FLASHNEXT_SNAP_HOST), and the native host then budgets the prompt
/// cache in host memory (Loaded.host_snaps).
pub var on_host: bool = false;

pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    const at = seq.fed;
    if (seq.invalidated) return error.SnapshotDestinationInvalid;
    if (at > seq.cap) return error.SequenceTooShort;
    const shared_lease = try acquireShared(seq);
    defer finishShared(seq, shared_lease);
    errdefer failShared(stream, seq, shared_lease);
    const image = if (seq.paged) |pg| try pg.snapshot(at) else null;
    if (d) |h| if (h.len != at or h.drafted != 0 or h.open or at > h.cap) return error.NotAtMark;
    if (image != null) {
        try validatePagedSeq(seq, d, at);
        if (seq.kv8 != null or seq.media != null or seq.logits.len != prompt.vocabulary * 2) return error.KvPageSnapshotLayout;
        if (d) |h| if (at > 0 and (h.prompt_last < h.streams.ptr or h.streams.len < wt.wide * 2 or h.prompt_last - h.streams.ptr > h.streams.len - wt.wide * 2)) return error.NotAtMark;
        errdefer {
            cuda.completion.drain(stream.d, stream, "paged snapshot preparation did not drain");
            seq.invalidated = true;
        }
        try quiesce(stream);
        try seq.paged.?.stageStatus(stream);
        try stream.synchronize();
        try seq.paged.?.checkStatus();
    }
    const n = if (image) |pg| try payloadBytes(pg, d != null) else bytesKv(at, d != null, seqRows(seq));
    const snap = try allocate(gpa, driver, n, at, seq.history, d != null, image);
    errdefer discard(gpa, stream, snap);
    errdefer if (image != null) {
        seq.invalidated = true;
    };
    if (image != null) {
        if (shared_lease) |lease| try seq.paged.?.shared_arena.?.submitted(lease);
        try seq.paged.?.flush(stream, at);
        try movePaged(stream, seq, d, snap, .save);
        if (shared_lease) |lease| snap.owned_blocks = try seq.paged.?.shared_arena.?.saveOwnedBlocks(gpa, lease);
    } else try move(stream, seq, d, snap.buf.ptr, at, .save);
    try stream.synchronize();
    if (image != null) {
        const logits = snap.host.?.bytes[n - seq.logits.len .. n];
        if (at == 0) @memset(logits, 0) else @memcpy(logits, seq.logits);
    }
    return snap;
}

fn allocate(gpa: std.mem.Allocator, driver: *const cuda.Driver, n: usize, at: usize, history: [2]i64, head: bool, image: ?pager.Snapshot) !*Snap {
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = undefined, .at = at, .history = history, .head = head, .paged = image };
    if (image) |pg| if (pg.layout.cfg.shared_lanes != 0) {
        snap.version = 2;
    };
    if (on_host or image != null) {
        var host = try cuda.HostBuffer.allocMapped(driver, @max(n, 256));
        errdefer host.free();
        snap.buf = .{ .d = driver, .ptr = try host.device(), .len = n, .borrowed = true };
        snap.host = host;
    } else snap.buf = try cuda.DeviceBuffer.alloc(driver, n);
    return snap;
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn validateRestore(seq: *const prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    const shared = if (snap.paged) |image| image.layout.cfg.shared_lanes != 0 else false;
    if (snap.version != @as(u32, if (shared) 2 else 1)) return error.SnapshotVersion;
    if (shared != (snap.owned_blocks != null)) return error.KvPageSnapshotLayout;
    if (seq.invalidated) return error.SnapshotDestinationInvalid;
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (d) |h| if (!snap.head or snap.at > h.cap) return error.NoHeadState;
    if (snap.paged) |image| {
        const pg = seq.paged orelse return error.KvPageSnapshotLayout;
        try pg.validateSnapshot(image);
        if (snap.owned_blocks) |*owned| {
            try pg.shared_arena.?.validateOwnedBlocks(owned);
        }
        try validatePagedSeq(seq, d, snap.at);
        if (snap.at != image.layout.at or snap.host == null or snap.buf.len != try payloadBytes(image, snap.head) or snap.host.?.bytes.len < snap.buf.len or seq.kv8 != null or seq.media != null or seq.logits.len != prompt.vocabulary * 2) return error.KvPageSnapshotLayout;
        if (d) |h| if (h.streams.len < wt.wide * 2) return error.NoHeadState;
    } else if (seq.paged != null) return error.KvPageSnapshotLayout;
}

pub fn restoreData(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    try validateRestore(seq, d, snap);
    const shared_lease = try acquireShared(seq);
    defer finishShared(seq, shared_lease);
    errdefer failShared(stream, seq, shared_lease);
    if (snap.paged != null) try quiesce(stream);
    @import("cuda_exl3_graph.zig").releaseSeq(seq);
    if (d) |h| h.verify_cache.invalidate();
    seq.invalidated = true;
    if (shared_lease) |lease| try seq.paged.?.shared_arena.?.submitted(lease);
    errdefer cuda.completion.drain(stream.d, stream, "partial snapshot restore did not drain");
    if (shared_lease) |lease| try seq.paged.?.shared_arena.?.restoreOwnedBlocks(lease, &snap.owned_blocks.?);
    if (snap.paged != null) try movePaged(stream, seq, d, snap, .restore) else try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    if (seq.paged) |pg| try pg.stageStatus(stream);
    try stream.synchronize();
    if (seq.paged) |pg| try pg.checkStatus();
}

fn acquireShared(seq: *const prompt.Seq) !?pager.shared_device.Lease {
    const pg = seq.paged orelse return null;
    if (pg.cfg.shared_lanes == 0) return null;
    return try pg.shared_arena.?.acquire(pg.shared_sequence);
}

fn finishShared(seq: *const prompt.Seq, lease: ?pager.shared_device.Lease) void {
    if (lease) |held| {
        const arena = seq.paged.?.shared_arena.?;
        if (arena.active != null) arena.completeAfterConsumers(held) catch unreachable;
    }
}

fn failShared(stream: cuda.Stream, seq: *prompt.Seq, lease: ?pager.shared_device.Lease) void {
    if (lease) |held| {
        const arena = seq.paged.?.shared_arena.?;
        if (!arena.active.?.submitted) {
            arena.cancel(held) catch unreachable;
            return;
        }
        if (!arena.active.?.failed) arena.fail(held) catch unreachable;
        seq.invalidated = true;
        cuda.completion.drain(stream.d, stream, "shared snapshot failed to drain");
        arena.drainFailed(held) catch @panic("shared snapshot still owns consumers");
    }
}

pub fn publishRestore(seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) void {
    std.debug.assert(seq.invalidated);
    if (snap.paged) |image| {
        seq.paged.?.publishSnapshot(image);
        @memcpy(seq.logits, snap.host.?.bytes[snap.buf.len - seq.logits.len .. snap.buf.len]);
    }
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
        h.open = false;
        h.fed0 = snap.at;
        h.verify_pending = null;
        if (snap.paged != null) h.prompt_last = if (snap.at == 0) 0 else h.streams.ptr;
    }
    seq.invalidated = false;
}

pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    try restoreData(stream, seq, d, snap);
    publishRestore(seq, d, snap);
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
