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
        return head + @max(rows, max_rows) * wide * 2 + wide * 2 + max_rows * vocab * 2 + draft_n * 2 + max_rows * 16 + l.saved + linear_n * l.state;
    }

    /// A head cache of `cap` entries; `rows` the widest prompt chunk whose streams it keeps; `h` the model's heads.
    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize, h: Heads) !Draft {
        var d: Draft = .{ .stream = stream, .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = undefined, .vc = undefined, .ikc = undefined, .pooled = undefined, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined, .layout = .of(h) };
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
        d.best = try cuda.DeviceBuffer.alloc(driver, max_rows * 16);
        errdefer d.best.free();
        d.saved = try cuda.DeviceBuffer.alloc(driver, d.layout.saved);
        errdefer d.saved.free();
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

    /// The keep's history and projection rows in `saved`, and the end states into the spares.
    fn point(d: *Draft) void {
        var li: usize = 0;
        for (&d.keep.layers, 0..) |*lk, layer| {
            if (!wt.linearLayer(layer)) continue;
            lk.hist = d.saved.ptr + d.layout.hist_at + li * d.layout.conv;
            lk.proj = d.saved.ptr + d.layout.proj_at + li * max_rows * d.layout.proj;
            li += 1;
        }
        d.aim();
    }

    /// Device bytes `initIn` takes from a run: every buffer but the head's cache, each on a 256-byte boundary.
    pub fn runBytes(draft_n: usize, rows: usize) usize {
        return bytes(0, draft_n, rows, pool_heads) + (6 + linear_n) * 256;
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
        var d: Draft = .{ .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = Carve.page(driver, paged.at(head, .key)), .vc = Carve.page(driver, paged.at(head, .value)), .ikc = Carve.page(driver, paged.at(head, .index)), .pooled = Carve.page(driver, paged.at(head, .pooled)), .pg = paged.pg(), .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined, .layout = .of(pool_heads) };
        d.streams = Carve.take(driver, &at, d.rows_cap * wide * 2);
        d.out = Carve.take(driver, &at, wide * 2);
        d.logits = Carve.take(driver, &at, max_rows * vocab * 2);
        d.dlogits = Carve.take(driver, &at, draft_n * 2);
        d.best = Carve.take(driver, &at, max_rows * 16);
        d.saved = Carve.take(driver, &at, d.layout.saved);
        for (&d.spare) |*b| b.* = Carve.take(driver, &at, d.layout.state);
        std.debug.assert(at - base <= len);
        d.host = try gpa.alloc(u16, @max(max_rows * vocab, draft_n));
        d.point();
        return d;
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

    /// An open window kept whole: its end states become the sequence's (buffers swapped, nothing copied).
    pub fn settle(d: *Draft, seq: *prompt.Seq) void {
        if (!d.open) return;
        for (seq.state, &d.spare) |*a, *b| std.mem.swap(cuda.DeviceBuffer, a, b);
        d.aim();
        d.open = false;
    }

    pub fn deinit(d: *Draft) void {
        if (d.stream) |stream| cuda.completion.drain(stream.d, stream, "Flash Next draft work did not drain before release");
        d.gpa.free(d.host);
        for (&d.spare) |*b| b.free();
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
    }
};

test "a draft at 65,536 tokens and 1,536-row chunks: the head's cache whole, saved rows and spares at the model's heads" {
    const cap = 65536;
    const rows = 1536;
    const draft_n = 32768;
    // the head's cache: two KV heads of 256, 128 indexer keys a position, a pooled row a block of 4
    const head: usize = cap * (2 * 2 * 256 * 2 + 128 * 2) + cap / 4 * 128 * 2;
    // streams rows, the head's output row, a window's scores, the draft head's, best values
    const buffers: usize = rows * 4 * 2560 * 2 + 4 * 2560 * 2 + 32 * 248320 * 2 + draft_n * 2 + 32 * 16;
    // saved: 36 conv windows, 36 layers' 32 projection rows, the n-gram tail and 32 n-gram rows; 36 spare states
    const tail: usize = 9 * 4 * 2560 * 2 + 32 * 4 * 2560 * 2;
    const full: usize = 36 * 3 * 10240 * 2 + 36 * 32 * 16480 * 2 + tail + 36 * 48 * 128 * 128 * 4;
    const half: usize = 36 * 3 * 5120 * 2 + 36 * 32 * 8240 * 2 + tail + 36 * 24 * 128 * 128 * 4;
    try std.testing.expectEqual(head + buffers + full, Draft.bytes(cap, draft_n, rows, .full));
    try std.testing.expectEqual(head + buffers + half, Draft.bytes(cap, draft_n, rows, .half));
    try std.testing.expectEqual(@as(usize, 356_893_184), Draft.bytes(cap, draft_n, rows, .full));
    try std.testing.expectEqual(@as(usize, 280_179_200), Draft.bytes(cap, draft_n, rows, .half));
    // with the stream's own caches at half the heads
    try std.testing.expectEqual(@as(usize, 1_395_057_152), sizes.seqBytes(cap, .half) + Draft.bytes(cap, draft_n, rows, .half));
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
