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
const state_n = wt.state_n;
const out_sk = wt.out_sk;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;
const ple_tail_n = wt.ple_tail_n;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const max_chain = mtp.max_chain;
const max_rows = mtp.max_rows;
const conv_bytes = mtp.conv_bytes;
const proj_bytes = mtp.proj_bytes;

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

    const hist_at: usize = 0;
    const proj_at: usize = hist_at + linear_n * conv_bytes;
    pub const tail_at: usize = proj_at + linear_n * max_rows * proj_bytes;
    /// The window's n-gram rows (the pass's ple_nrow), saved for a rollback to shift the kept ones into the tail.
    pub const nrow_at: usize = tail_at + ple_tail_n * wide * 2;
    const saved_bytes: usize = nrow_at + max_rows * wide * 2;

    /// Device bytes `init(cap, draft_n, rows)` allocates.
    pub fn bytes(cap: usize, draft_n: usize, rows: usize) usize {
        return cap * (2 * prompt.kv_row_bytes + index_dim * 2) + prompt.pooledBytes(cap) + @max(rows, max_rows) * wide * 2 + wide * 2 + max_rows * vocab * 2 + draft_n * 2 + max_rows * 16 + saved_bytes + linear_n * state_n * 4;
    }

    /// A head cache of `cap` entries; `rows` the widest prompt chunk whose streams it keeps.
    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize) !Draft {
        var d: Draft = .{ .stream = stream, .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = undefined, .vc = undefined, .ikc = undefined, .pooled = undefined, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined };
        d.kc = try cuda.DeviceBuffer.alloc(driver, cap * kv_heads * head_dim * 2);
        errdefer d.kc.free();
        d.vc = try cuda.DeviceBuffer.alloc(driver, cap * kv_heads * head_dim * 2);
        errdefer d.vc.free();
        d.ikc = try cuda.DeviceBuffer.alloc(driver, cap * index_dim * 2);
        errdefer d.ikc.free();
        d.pooled = try cuda.DeviceBuffer.alloc(driver, prompt.pooledBytes(cap));
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
        d.saved = try cuda.DeviceBuffer.alloc(driver, saved_bytes);
        errdefer d.saved.free();
        var made: usize = 0;
        errdefer for (d.spare[0..made]) |*b| b.free();
        while (made < linear_n) : (made += 1) d.spare[made] = try cuda.DeviceBuffer.alloc(driver, state_n * 4);
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
            lk.hist = d.saved.ptr + hist_at + li * conv_bytes;
            lk.proj = d.saved.ptr + proj_at + li * max_rows * proj_bytes;
            li += 1;
        }
        d.aim();
    }

    /// Device bytes `initIn` takes from a run: every buffer but the head's cache, each on a 256-byte boundary.
    pub fn runBytes(draft_n: usize, rows: usize) usize {
        return bytes(0, draft_n, rows) + (6 + linear_n) * 256;
    }

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
        var d: Draft = .{ .gpa = gpa, .cap = cap, .rows_cap = @max(rows, max_rows), .kc = Carve.page(driver, paged.at(head, .key)), .vc = Carve.page(driver, paged.at(head, .value)), .ikc = Carve.page(driver, paged.at(head, .index)), .pooled = Carve.page(driver, paged.at(head, .pooled)), .pg = paged.pg(), .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined };
        d.streams = Carve.take(driver, &at, d.rows_cap * wide * 2);
        d.out = Carve.take(driver, &at, wide * 2);
        d.logits = Carve.take(driver, &at, max_rows * vocab * 2);
        d.dlogits = Carve.take(driver, &at, draft_n * 2);
        d.best = Carve.take(driver, &at, max_rows * 16);
        d.saved = Carve.take(driver, &at, saved_bytes);
        for (&d.spare) |*b| b.* = Carve.take(driver, &at, state_n * 4);
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
