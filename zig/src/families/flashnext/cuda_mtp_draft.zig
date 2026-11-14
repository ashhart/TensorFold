//! A drafting stream's state: the head's cache, a window's buffers and what a partly kept window rolls back to.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");
const mtp = @import("cuda_mtp.zig");
const pages = @import("cuda_pages.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const vocab = wt.vocab;
const eps = wt.eps;
const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const out_sk = wt.out_sk;
const ple_tail_n = wt.ple_tail_n;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const max_chain = mtp.max_chain;
const max_rows = mtp.max_rows;
/// best: [max_rows, 4] fp32 best values, then [max_rows, topk_max] bf16 candidates and i32 columns (cuda_mtp_draw.zig).
pub const best_bytes: usize = max_rows * 16 + max_rows * @import("cuda_mtp_draw.zig").topk_max * 6;
const sizes = @import("cuda_seq_bytes.zig");
const Heads = @import("cuda_heads.zig").Heads;

/// One draft and its temperature-1 probability under the head.
pub const Pick = struct { token: u32, prob: f64 };

/// A chain of drafts the head holds for the next window.
pub const Chain = struct {
    tokens: [max_chain]u32 = undefined,
    probs: [max_chain]f64 = undefined,
    n: usize = 0,

    pub fn slice(c: *const Chain) []const u32 {
        return c.tokens[0..c.n];
    }
};

/// A sequence's head cache and the buffers a window needs: its rows' scores and streams, and what it rolls back to.
pub const Draft = struct {
    /// Its graphs' keys (cuda_window_graph.nextGen).
    gen: u64 = 0,
    label: []const u8 = "unbound",
    pooled_lane: bool = false,
    stream: ?cuda.Stream = null,
    media: ?@import("cuda_media.zig").View = null,
    gpa: std.mem.Allocator,
    kc: cuda.DeviceBuffer,
    vc: cuda.DeviceBuffer,
    ikc: cuda.DeviceBuffer,
    /// Its pooled indexer keys [cap / 4, 128] bf16, for sparse attention past 2,048 keys (Python State.mtp_pooled).
    pooled: cuda.DeviceBuffer,
    /// The cache's pages (a pool lane's: its stream's, the layer after the model's), or contiguous.
    pg: @import("cuda_glue.zig").Pg = .{},
    /// Entries the head's cache holds (Python mtp_len), the last `drafted` of them a chain's.
    len: usize = 0,
    drafted: usize = 0,
    cap: usize,
    /// The last pass's rows of main-model streams [rows_cap, wide]: a prompt chunk's, then each window's.
    streams: cuda.DeviceBuffer,
    rows_cap: usize,
    /// The prompt's last row of streams in `streams`, which the first draft reads with the first token.
    prompt_last: u64 = 0,
    /// The head's last output row (its streams after the MoE write-back): the next chained draft's input.
    out: cuda.DeviceBuffer,
    /// A window's scores [max_rows, vocab], the draft head's [ids], best values [max_rows, 4].
    logits: cuda.DeviceBuffer,
    dlogits: cuda.DeviceBuffer,
    best: cuda.DeviceBuffer,
    /// Before a window: each DeltaNet layer's conv window and (from the pass) projection rows; the n-gram tail.
    saved: cuda.DeviceBuffer,
    /// A window's DeltaNet end states: swapped in when every row is kept, else the kept rows replay on the sequence's.
    spare: [linear_n]cuda.DeviceBuffer = undefined,
    /// Where `saved` holds each piece and how large the spares are, at the model's heads (its sequence's).
    layout: Layout,
    /// A window ran with the states written to `spare` and is not yet kept or rolled back.
    open: bool = false,
    /// Pending DeltaNet rows: a control word, then each layer's log of batch.late_rows rows; null: rollbacks replay.
    log: ?cuda.DeviceBuffer = null,
    log_first: u32 = 0,
    pending: u32 = 0,
    keep: batch.Keep = .{},
    history: [2]i64 = .{ 0, 0 },
    unused_history: [2]i64 = .{ 0, 0 },
    fed0: usize = 0,
    rows: usize = 0,
    ids: [max_rows]u32 = undefined,
    /// The scores a sampled draw reads on the host (a window's rows, or the draft head's row).
    host: []u16,
    best_host: [max_rows * 4]f32 = undefined,

    /// `saved`'s pieces and the spares' size at the model's heads: conv windows, projection rows, tail, n-gram rows.
    pub const Layout = struct {
        heads: Heads,
        conv: usize,
        proj: usize,
        hist_at: usize = 0,
        proj_at: usize,
        tail_at: usize,
        /// The window's n-gram rows (the pass's ple_nrow), saved for a rollback to shift the kept ones into the tail.
        nrow_at: usize,
        saved: usize,
        state: usize,

        pub fn of(h: Heads) Layout {
            const conv = sizes.convBytes(h);
            const proj = h.projWidth() * 2;
            const proj_at = linear_n * conv;
            const tail_at = proj_at + linear_n * max_rows * proj;
            const nrow_at = tail_at + ple_tail_n * wide * 2;
            return .{ .heads = h, .conv = conv, .proj = proj, .proj_at = proj_at, .tail_at = tail_at, .nrow_at = nrow_at, .saved = nrow_at + max_rows * wide * 2, .state = sizes.stateBytes(h) };
        }
    };

    /// The head's own cache: its layer is whole on every rank.
    const head_heads: Heads = .full;

    /// Device bytes `init(cap, draft_n, rows, h)` allocates: the head's cache at full KV rows, the rest at `h`.
    pub fn bytes(cap: usize, draft_n: usize, rows: usize, h: Heads) usize {
        const l = Layout.of(h);
        const head = 2 * sizes.kvBytes(cap, head_heads) + sizes.indexBytes(cap) + sizes.pooledBytes(cap);
        const log = if (late()) logBytes() else 0;
        return head + @max(rows, max_rows) * wide * 2 + wide * 2 + max_rows * vocab * 2 + draft_n * 2 + best_bytes + l.saved + linear_n * l.state + log;
    }

    /// Drafting streams keep a partly kept window's DeltaNet rows pending (set at load; TF_FLASHNEXT_KEEP_LATE=0: off).
    pub var late_on: bool = false;

    pub fn late() bool {
        return late_on;
    }

    /// TF_FLASHNEXT_KEEP_LATE: "1" asked for, "0" refused, null: not set.
    pub fn lateWanted() ?bool {
        // the startup check refused every malformed value already, so a plain default stands here
        return @import("sys").environment.read(?bool, "TF_FLASHNEXT_KEEP_LATE", null) catch null;
    }

    const log_ctl_bytes: usize = 256;
    const layer_log_bytes: usize = batch.late_rows * batch.late_row_bytes;

    /// The pending-row log's bytes: its first and pending rows, then every DeltaNet layer's rows.
    pub fn logBytes() usize {
        return log_ctl_bytes + linear_n * layer_log_bytes;
    }

    /// A partly kept window's rows join the pending ones; past late_store the chain stored its state, so they restart.
    pub fn keptLate(d: *Draft, kept: usize) void {
        if (d.pending > batch.late_store) {
            d.log_first = @intCast((d.log_first + d.pending) % batch.late_rows);
            d.pending = @intCast(kept);
        } else d.pending += @intCast(kept);
    }

    /// A head cache of `cap` entries; `rows` the widest prompt chunk whose streams it keeps; `h` the model's heads.
    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize, h: Heads) !Draft {
        var d: Draft = .{ .gen = @import("cuda_window_graph.zig").nextGen(), .stream = stream, .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = undefined, .vc = undefined, .ikc = undefined, .pooled = undefined, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined, .layout = .of(h) };
        d.kc = try cuda.DeviceBuffer.alloc(driver, sizes.kvBytes(cap, head_heads));
        errdefer d.kc.free();
        d.vc = try cuda.DeviceBuffer.alloc(driver, sizes.kvBytes(cap, head_heads));
        errdefer d.vc.free();
        d.ikc = try cuda.DeviceBuffer.alloc(driver, sizes.indexBytes(cap));
        errdefer d.ikc.free();
        d.pooled = try cuda.DeviceBuffer.alloc(driver, sizes.pooledBytes(cap));
        errdefer d.pooled.free();
        d.streams = try cuda.DeviceBuffer.alloc(driver, d.rows_cap * wide * 2);
        errdefer d.streams.free();
        d.out = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        errdefer d.out.free();
        d.logits = try cuda.DeviceBuffer.alloc(driver, max_rows * vocab * 2);
        errdefer d.logits.free();
        d.dlogits = try cuda.DeviceBuffer.alloc(driver, draft_n * 2);
        errdefer d.dlogits.free();
        d.best = try cuda.DeviceBuffer.alloc(driver, best_bytes);
        errdefer d.best.free();
        d.saved = try cuda.DeviceBuffer.alloc(driver, d.layout.saved);
        errdefer d.saved.free();
        if (late()) d.log = try cuda.DeviceBuffer.alloc(driver, logBytes());
        errdefer if (d.log) |*b| b.free();
        var made: usize = 0;
        errdefer for (d.spare[0..made]) |*b| b.free();
        while (made < linear_n) : (made += 1) d.spare[made] = try cuda.DeviceBuffer.alloc(driver, d.layout.state);
        d.host = try gpa.alloc(u16, @max(max_rows * vocab, draft_n));
        errdefer gpa.free(d.host);
        errdefer cuda.completion.drain(driver, stream, "Flash Next draft fills did not drain before rollback");
        // zeroed on the compute stream, ahead of the passes that read them
        for ([_]cuda.DeviceBuffer{ d.kc, d.vc, d.ikc, d.pooled }) |b| try b.fill8(0, stream.handle);
        d.point();
        return d;
    }

    /// The fixed-size buffers of a draft in one block (initParts) at the model's heads `h`: all but the head's caches.
    pub fn partsBytes(draft_n: usize, rows: usize, h: Heads) usize {
        const a = std.mem.alignForward;
        const rows_cap = @max(rows, max_rows);
        const l = Layout.of(h);
        var n: usize = 0;
        for ([_]usize{ rows_cap * wide * 2, wide * 2, max_rows * vocab * 2, draft_n * 2, best_bytes, l.saved, if (late()) logBytes() else 0 }) |x| n += a(usize, x, 256);
        return n + linear_n * a(usize, l.state, 256);
    }

    /// init's draft whose fixed-size buffers (partsBytes) are zeroed views of `block`; its caches allocated as init's.
    pub fn initParts(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize, h: Heads, block: cuda.DeviceBuffer) !Draft {
        const bytes_in = partsBytes(draft_n, rows, h);
        if (block.len < bytes_in) return error.FlashNextRunTooShort;
        var d: Draft = .{ .gen = @import("cuda_window_graph.zig").nextGen(), .stream = stream, .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = undefined, .vc = undefined, .ikc = undefined, .pooled = undefined, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .spare = undefined, .host = undefined, .layout = .of(h) };
        d.kc = try cuda.DeviceBuffer.alloc(driver, sizes.kvBytes(cap, head_heads));
        errdefer d.kc.free();
        d.vc = try cuda.DeviceBuffer.alloc(driver, sizes.kvBytes(cap, head_heads));
        errdefer d.vc.free();
        d.ikc = try cuda.DeviceBuffer.alloc(driver, sizes.indexBytes(cap));
        errdefer d.ikc.free();
        d.pooled = try cuda.DeviceBuffer.alloc(driver, sizes.pooledBytes(cap));
        errdefer d.pooled.free();
        const Carve = struct {
            fn take(dr: *const cuda.Driver, p: *u64, n: usize) cuda.DeviceBuffer {
                const b: cuda.DeviceBuffer = .{ .d = dr, .ptr = p.*, .len = n, .borrowed = true };
                p.* += std.mem.alignForward(usize, n, 256);
                return b;
            }
        };
        var at = block.ptr;
        d.streams = Carve.take(driver, &at, d.rows_cap * wide * 2);
        d.out = Carve.take(driver, &at, wide * 2);
        d.logits = Carve.take(driver, &at, max_rows * vocab * 2);
        d.dlogits = Carve.take(driver, &at, draft_n * 2);
        d.best = Carve.take(driver, &at, best_bytes);
        d.saved = Carve.take(driver, &at, d.layout.saved);
        if (late()) d.log = Carve.take(driver, &at, logBytes());
        for (&d.spare) |*b| b.* = Carve.take(driver, &at, d.layout.state);
        std.debug.assert(at - block.ptr == bytes_in);
        d.host = try gpa.alloc(u16, @max(max_rows * vocab, draft_n));
        errdefer gpa.free(d.host);
        errdefer cuda.completion.drain(driver, stream, "Flash Next draft fills did not drain before rollback");
        for ([_]cuda.DeviceBuffer{ d.kc, d.vc, d.ikc, d.pooled }) |b| try b.fill8(0, stream.handle);
        try stream.d.check(stream.d.api.cuMemsetD8Async(block.ptr, 0, bytes_in, stream.handle), "cuMemsetD8Async");
        d.point();
        return d;
    }

    /// The keep's history and projection rows in `saved`, and the end states into the spares.
    fn point(d: *Draft) void {
        var li: usize = 0;
        for (&d.keep.layers, 0..) |*lk, layer| {
            if (!wt.linearLayer(layer)) continue;
            lk.hist = d.saved.ptr + d.layout.hist_at + li * d.layout.conv;
            lk.proj = d.saved.ptr + d.layout.proj_at + li * max_rows * d.layout.proj;
            if (d.log) |b| {
                lk.log = b.ptr + log_ctl_bytes + li * layer_log_bytes;
                lk.log_ctl = b.ptr;
            }
            li += 1;
        }
        d.aim();
    }

    /// Device bytes `initIn` takes from a run: every buffer but the head's cache, each on a 256-byte boundary.
    pub fn runBytes(draft_n: usize, rows: usize) usize {
        return bytes(0, draft_n, rows, pool_heads) + (7 + linear_n) * 256;
    }

    /// A pool lane's model heads: the pages hold the whole layer (one rank; cuda_lanes_load.zig refuses pages at --tp).
    const pool_heads: Heads = .full;

    /// A pool lane's head: its cache on `paged`'s pages after the model's layers, its buffers from `base`'s run.
    pub fn initIn(gpa: std.mem.Allocator, driver: *cuda.Driver, cap: usize, draft_n: usize, rows: usize, paged: *const pages.Paged, base: u64, len: usize) !Draft {
        if (len < runBytes(draft_n, rows)) return error.FlashNextRunTooShort;
        var at = base;
        const Carve = struct {
            fn take(d: *const cuda.Driver, p: *u64, n: usize) cuda.DeviceBuffer {
                const b: cuda.DeviceBuffer = .{ .d = d, .ptr = p.*, .len = n, .borrowed = true };
                p.* += std.mem.alignForward(usize, n, 256);
                return b;
            }
            fn page(d: *const cuda.Driver, ptr: u64) cuda.DeviceBuffer {
                return .{ .d = d, .ptr = ptr, .len = 0, .borrowed = true };
            }
        };
        const head = wt.attn_n;
        var d: Draft = .{ .gen = @import("cuda_window_graph.zig").nextGen(), .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = Carve.page(driver, paged.at(head, .key)), .vc = Carve.page(driver, paged.at(head, .value)), .ikc = Carve.page(driver, paged.at(head, .index)), .pooled = Carve.page(driver, paged.at(head, .pooled)), .pg = paged.pg(), .pooled_lane = true, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined, .layout = .of(pool_heads) };
        d.streams = Carve.take(driver, &at, d.rows_cap * wide * 2);
        d.out = Carve.take(driver, &at, wide * 2);
        d.logits = Carve.take(driver, &at, max_rows * vocab * 2);
        d.dlogits = Carve.take(driver, &at, draft_n * 2);
        d.best = Carve.take(driver, &at, best_bytes);
        d.saved = Carve.take(driver, &at, d.layout.saved);
        if (late()) d.log = Carve.take(driver, &at, logBytes());
        for (&d.spare) |*b| b.* = Carve.take(driver, &at, d.layout.state);
        std.debug.assert(at - base <= len);
        d.host = try gpa.alloc(u16, @max(max_rows * vocab, draft_n));
        d.point();
        return d;
    }

    /// The head's one layer as a pass reads its cache: contiguous, or the lane's pages (`pg`) after the model's.
    pub fn headSeq(d: *Draft, media: ?@import("cuda_media.zig").View, scores: u64) batch.SeqState {
        return .{ .media = media, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .pg = d.pg, .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = scores };
    }

    /// Each DeltaNet layer's window end state into its spare buffer.
    fn aim(d: *Draft) void {
        var li: usize = 0;
        for (&d.keep.layers, 0..) |*lk, layer| {
            if (!wt.linearLayer(layer)) continue;
            lk.state_out = d.spare[li].ptr;
            li += 1;
        }
    }

    /// An open window kept whole: its end states become the sequence's (buffers swapped), no rows left pending.
    pub fn settle(d: *Draft, seq: *prompt.Seq) void {
        if (!d.open) return;
        for (seq.state, &d.spare) |*a, *b| std.mem.swap(cuda.DeviceBuffer, a, b);
        d.aim();
        d.open = false;
        d.log_first = 0;
        d.pending = 0;
    }

    pub fn deinit(d: *Draft) void {
        if (d.stream) |stream| cuda.completion.drain(stream.d, stream, "Flash Next draft work did not drain before release");
        d.gpa.free(d.host);
        for (&d.spare) |*b| b.free();
        if (d.log) |*b| b.free();
        d.saved.free();
        d.best.free();
        d.dlogits.free();
        d.logits.free();
        d.out.free();
        d.streams.free();
        d.pooled.free();
        d.ikc.free();
        d.vc.free();
        d.kc.free();
    }

    /// Drop the head's cache (a new prompt).
    pub fn reset(d: *Draft) void {
        d.len = 0;
        d.drafted = 0;
        d.rows = 0;
        d.log_first = 0;
        d.pending = 0;
    }
};

test "a draft at 65,536 tokens and 1,536-row chunks: the head's cache whole, saved rows and spares at the model's heads" {
    const cap = 65536;
    const rows = 1536;
    const draft_n = 32768;
    // the head's cache: two KV heads of 256, 128 indexer keys a position, a pooled row a block of 4
    const head: usize = cap * (2 * 2 * 256 * 2 + 128 * 2) + cap / 4 * 128 * 2;
    // streams rows, the head's output row, a window's scores, the draft head's, best values
    const buffers: usize = rows * 4 * 2560 * 2 + 4 * 2560 * 2 + 32 * 248320 * 2 + draft_n * 2 + best_bytes;
    // saved: 36 conv windows, 36 layers' 32 projection rows, the n-gram tail and 32 n-gram rows; 36 spare states
    const tail: usize = 9 * 4 * 2560 * 2 + 32 * 4 * 2560 * 2;
    const full: usize = 36 * 3 * 10240 * 2 + 36 * 32 * 16480 * 2 + tail + 36 * 48 * 128 * 128 * 4;
    const half: usize = 36 * 3 * 5120 * 2 + 36 * 32 * 8240 * 2 + tail + 36 * 24 * 128 * 128 * 4;
    try std.testing.expectEqual(head + buffers + full, Draft.bytes(cap, draft_n, rows, .full));
    try std.testing.expectEqual(head + buffers + half, Draft.bytes(cap, draft_n, rows, .half));
    try std.testing.expectEqual(@as(usize, 356_905_472), Draft.bytes(cap, draft_n, rows, .full));
    try std.testing.expectEqual(@as(usize, 280_191_488), Draft.bytes(cap, draft_n, rows, .half));
    // with the stream's own caches at half the heads
    try std.testing.expectEqual(@as(usize, 1_395_069_440), sizes.seqBytes(cap, .half) + Draft.bytes(cap, draft_n, rows, .half));
}

test "blank TF_FLASHNEXT_KEEP_LATE: unset, empty or whitespace-only text asks for nothing, 1 and 0 keep their meaning" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const t = std.testing;
    const env = @import("sys").env;
    const name = "TF_FLASHNEXT_KEEP_LATE";
    const before = if (std.c.getenv(name)) |v| try t.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (before) |v| {
            _ = env.setenv(name, v, 1);
            t.allocator.free(v);
        } else _ = env.unsetenv(name);
    }
    _ = env.unsetenv(name);
    try t.expectEqual(@as(?bool, null), Draft.lateWanted());
    for ([_][:0]const u8{ "", " ", "\t", " \t\r\n" }) |v| {
        try t.expectEqual(@as(c_int, 0), env.setenv(name, v, 1));
        try t.expectEqual(@as(?bool, null), Draft.lateWanted());
    }
    for ([_][:0]const u8{ "1", "0" }, [_]?bool{ true, false }) |v, wanted| {
        try t.expectEqual(@as(c_int, 0), env.setenv(name, v, 1));
        try t.expectEqual(wanted, Draft.lateWanted());
    }
}

test "the saved pieces follow one another at the model's heads, and the spares hold its states" {
    for ([_]Heads{ .full, .half }) |h| {
        const l = Draft.Layout.of(h);
        try std.testing.expectEqual(@as(usize, 36 * 3 * h.convDim() * 2), l.proj_at);
        try std.testing.expectEqual(l.proj_at + 36 * 32 * h.projWidth() * 2, l.tail_at);
        try std.testing.expectEqual(l.tail_at + 9 * 4 * 2560 * 2, l.nrow_at);
        try std.testing.expectEqual(l.nrow_at + 32 * 4 * 2560 * 2, l.saved);
        try std.testing.expectEqual(sizes.stateBytes(h), l.state);
        try std.testing.expect(l.heads.eql(h));
    }
    try std.testing.expectEqual(@as(usize, 41_021_440), Draft.Layout.of(.full).saved);
    try std.testing.expectEqual(@as(usize, 20_930_560), Draft.Layout.of(.half).saved);
}
