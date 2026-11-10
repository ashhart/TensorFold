//! The batched pass's n-gram gather: the rows' embeddings on a host thread while the GPU runs layer 0.
const std = @import("std");
const ple = @import("cuda_ple.zig");
const wt = @import("cuda_prompt_wt.zig");
const Lane = @import("cuda_batch_state.zig").Lane;

const dims = wt.dims;

/// The rows' n-gram embeddings, gathered on a host thread while the GPU embeds and runs layer 0.
pub const Gather = struct {
    ngram: *const ple.Ngram,
    history: []i64,
    ids: []const u32,
    emb: []u16,
    xs: []f32,
    failed: ?anyerror = null,
    ns: i96 = 0,
    io: std.Io,
    /// A round's rows, each gathered after its own stream's history (empty: every row after `history`).
    lanes: []const Lane = &.{},

    pub fn main(g: *Gather) void {
        const t0 = std.Io.Clock.awake.now(g.io).toNanoseconds();
        defer g.ns = std.Io.Clock.awake.now(g.io).toNanoseconds() - t0;
        if (g.lanes.len == 0) {
            g.ngram.rows(g.history, g.ids, g.emb, g.xs) catch |err| {
                g.failed = err;
            };
            return;
        }
        var row0: usize = 0;
        for (g.lanes) |l| {
            g.ngram.rows(l.seq.history[0..], g.ids[row0..][0..l.rows], g.emb[row0 * dims ..][0 .. l.rows * dims], g.xs[row0 * (dims / 32) ..][0 .. l.rows * (dims / 32)]) catch |err| {
                g.failed = err;
                return;
            };
            row0 += l.rows;
        }
    }
};
