//! Flash Next's sparse attention a stream at a time on a HIP rank: the indexer's scores and picks, then the attention.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const Rows = @import("mixers.zig").Rows;

const Ops = hip.ops.Ops;

/// Rows of a prompt's attention a slice: the indexer's scores and the attention's parts stay bounded.
const slice_rows = 64;
const parts = 16;

/// One stream's `rows` rows from `pos` in slices; `graphed`: every row scored and selected over `staged.bound` blocks.
pub fn attend(
    o: Ops,
    m: *const model.Model,
    d: model.Dims,
    kv: cache.SparseView,
    cap: usize,
    q32: u64,
    iq: u64,
    att: u64,
    rows: usize,
    pos: usize,
    host: *forward.Host,
    staged: ?Rows,
    graphed: bool,
) !void {
    const hd = d.head_dim;
    const di = d.index_dim;
    const kind = @backingInt(m.act);
    const key_stride = d.index_top * d.index_ratio + d.index_ratio;
    var r0: usize = 0;
    while (r0 < rows) : (r0 += slice_rows) {
        const n = @min(slice_rows, rows - r0);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        var complete: [slice_rows]i32 = undefined;
        var ends: [slice_rows]i32 = undefined;
        var sparse_rows = graphed;
        for (0..n) |r| {
            ends[r] = @intCast(pos + r0 + r + 1);
            complete[r] = @divFloor(ends[r], @as(i32, @intCast(d.index_ratio)));
            sparse_rows = sparse_rows or complete[r] > d.index_top;
        }
        const dev_complete = if (staged) |st| st.complete + r0 * 4 else try host.put(
            o,
            std.mem.sliceAsBytes(complete[0..n]),
        );
        const dev_ends = if (staged) |st| st.ends + r0 * 4 else try host.put(o, std.mem.sliceAsBytes(ends[0..n]));
        var ids: ?u64 = null;
        var sparse_flags: ?u64 = null;
        var count = dev_ends;
        if (sparse_rows) {
            const nb: usize = if (graphed) staged.?.bound else @intCast(complete[n - 1]);
            const scores = try o.arena.of(f32, n * nb);
            try o.l.fn_idx_scores(
                @ptrFromInt(iq + r0 * d.index_heads * di * 4),
                @ptrFromInt(kv.pooled),
                @ptrFromInt(dev_complete),
                @ptrFromInt(scores),
                kind,
                @intCast(n),
                @intCast(d.index_heads),
                @intCast(di),
                @intCast(nb),
                @intCast(nb),
                @intCast(d.index_top),
                o.stream,
            );
            const list = try o.arena.of(i32, n * key_stride);
            const counts = try o.arena.of(i32, n);
            const flags = try o.arena.of(i32, n);
            try o.l.fn_idx_select(
                @ptrFromInt(scores),
                @intCast(nb),
                @ptrFromInt(dev_complete),
                @ptrFromInt(dev_ends),
                @ptrFromInt(list),
                @intCast(key_stride),
                @ptrFromInt(counts),
                @ptrFromInt(flags),
                @intCast(n),
                @intCast(d.index_top),
                @intCast(d.index_ratio),
                o.stream,
            );
            ids = list;
            sparse_flags = flags;
            count = counts;
        }
        const po = try o.arena.of(f32, n * d.heads * parts * hd);
        const pm = try o.arena.of(f32, n * d.heads * parts * 2);
        const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(hd))));
        try o.l.fn_sparse_attention(
            @ptrFromInt(q32 + r0 * d.heads * hd * 4),
            @ptrFromInt(kv.k),
            @ptrFromInt(kv.v),
            @backingInt(m.act),
            if (ids) |i| @ptrFromInt(i) else null,
            @intCast(key_stride),
            @ptrFromInt(count),
            if (sparse_flags) |f| @ptrFromInt(f) else null,
            @ptrFromInt(po),
            @ptrFromInt(pm),
            @ptrFromInt(att + r0 * d.heads * hd * 4),
            @intCast(n),
            @intCast(d.heads),
            @intCast(d.kv_heads),
            @intCast(hd),
            @intCast(cap),
            scale,
            parts,
            o.stream,
        );
    }
}
