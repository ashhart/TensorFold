//! prompt_logprobs at TP: a chunk's rows through each rank's head columns, made whole (Sampler.gather) for rank 0.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const b16 = @import("cuda_b16.zig");
const projection = @import("cuda_proj.zig");
const fw = @import("cuda_forward.zig");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// The words the kernel writes for one row (lanes.logprob's).
pub const row_words = lanes.logprob.words;

/// Rows a head pass takes with buffers of its own (the head's columns read once for all); else a verify window's.
pub const batch_rows = 128;

/// A prompt pass with prompt_logprobs: every rank runs the head on its rows; rank 0 has the reader of their words.
pub const Want = struct {
    k: u8 = 0,
    reader: ?Reader = null,
};

/// Rank 0's: `words` (rows of row_words) for `picks`, the prompt tokens the rows score, in prompt order.
pub const Reader = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, words: []const u32, picks: []const u32) anyerror!void };

/// This rank's columns of batch_rows rows and the whole rows, made at the first pass that asks.
pub const Buffers = struct {
    columns: cuda.DeviceBuffer,
    full: cuda.DeviceBuffer,

    fn make(e: *const Engine) !Buffers {
        var columns = try cuda.DeviceBuffer.alloc(e.d, batch_rows * e.share.vocab.len() * 2);
        errdefer columns.free();
        return .{ .columns = columns, .full = try cuda.DeviceBuffer.alloc(e.d, batch_rows * e.c.vocab * 2) };
    }

    pub fn free(b: *Buffers) void {
        b.full.free();
        b.columns.free();
    }
};

/// The host-mapped words of a pass's rows, then their picks (Engine.prompt_out).
pub const out_bytes = batch_rows * (row_words + 1) * 4;

/// Owns the mapped score words from engine startup until engine teardown.
pub const Output = struct {
    host: cuda.HostBuffer,
    address: u64,

    pub fn init(d: *const cuda.Driver) !Output {
        var host = try cuda.HostBuffer.allocMapped(d, out_bytes);
        errdefer host.free();
        return .{ .host = host, .address = try host.device() };
    }

    pub fn free(self: *Output) void {
        self.host.free();
    }
    pub fn slice(self: Output, comptime T: type) []T {
        return self.host.slice(T);
    }
};

/// The chunk's rows in e.bp.fnormed that score `next` (row r scores next[r]), a head pass at a time.
pub fn run(e: *Engine, want: Want, next: []const u32) !void {
    const own = try buffers(e);
    const rows: usize = if (own != null) batch_rows else fw.window_rows;
    const columns = if (own) |b| b.columns.ptr else e.bd.logits;
    const full = if (own) |b| b.full.ptr else e.smp.full.ptr;
    const D: usize = e.c.hidden;
    const words = e.prompt_out.slice(u32);
    const picks = words[batch_rows * row_words ..][0..batch_rows];
    const out = e.prompt_out.address;
    var r: usize = 0;
    while (r < next.len) : (r += rows) {
        const n = @min(rows, next.len - r);
        // as the head's last-row pass (Forward.finish); its column count takes no K slices (the test below)
        try projection.matmul(projectionSet(e), e.stream, e.w.head, e.bp.fnormed + r * D * 2, D, columns, e.bp.kda.mm_part, n, e.share.vocab.len(), D, false);
        try e.smp.gather(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = columns }}, n, full);
        const reader = want.reader orelse continue;
        @memcpy(picks[0..n], next[r..][0..n]);
        try e.smp.logprobRows(e.stream, full, n, out + batch_rows * row_words * 4, want.k, out);
        try e.stream.synchronize();
        try reader.call(reader.ctx, words[0 .. n * row_words], picks[0..n]);
    }
}

/// The passes' own buffers from the first that asks; if a rank has no room, every rank takes the window's (8 rows).
fn buffers(e: *Engine) !?*Buffers {
    if (!e.prompt_tried) {
        e.prompt_tried = true;
        e.prompt_bufs = Buffers.make(e) catch null;
        if (try e.shared(@floatFromInt(@intFromBool(e.prompt_bufs == null))) > 0) {
            if (e.prompt_bufs) |*b| b.free();
            e.prompt_bufs = null;
            std.log.warn("prompt_logprobs: a rank has no room for {d}-row head passes; {d} rows a pass", .{ batch_rows, fw.window_rows });
        }
    }
    return if (e.prompt_bufs) |*b| b else null;
}

comptime {
    std.debug.assert(batch_rows >= fw.window_rows); // the window's passes use the same host-mapped words
}

test "the head's columns at TP=2 and TP=3 take no K slices: a pass of any rows needs no partial sums" {
    const c: config.Config = .{};
    for ([_]usize{ 2, 3 }) |world| for (0..world) |rank| {
        const s = try shard.Share.of(&c, rank, world);
        try std.testing.expectEqual(@as(usize, 1), b16.splitK(s.vocab.len(), c.hidden));
    };
}

/// The head passes' projection policy (no indexer launches, so no .index); an MXFP8 head runs on the lane.
pub fn projectionSet(e: anytype) @import("cuda_glue.zig").Set {
    return .{ .glue = &e.m.glue, .mm = if (e.m.tiled) &e.m.mm else null, .packs = e.m.packs, .prompt = true, .linear = .{ .lane = &e.m.lane } };
}
