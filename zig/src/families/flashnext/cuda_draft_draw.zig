//! Draft lanes reduce the same gathered rows once, then draw with each lane's original seed and position.
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const nrows = mtp.max_rows;
const vals_at = nrows * 16;
const cols_at = vals_at + nrows * draw.topk_max * 2;
const bytes = @import("cuda_mtp_draft.zig").best_bytes;
pub const lanes_max = 16;
pub fn scoreBytes(v: usize) usize {
    return lanes_max * v * 2;
}

pub const Ready = struct {
    words: [bytes / 8]u64 = undefined,
    k: usize,
    fn best(r: *const Ready, row: usize) [4]f32 {
        return std.mem.bytesToValue([4]f32, std.mem.asBytes(&r.words)[row * 16 ..][0..16]);
    }
    pub fn pick(r: *const Ready, h: *const mtp.Head, st: cuda.Stream, a: std.mem.Allocator, d: *mtp.Draft, row: usize, position: u64, sm: ?mtp.Sampling, batched: bool) !mtp.Pick {
        const b = r.best(row);
        if (mtp.isSampled(sm) and (!draw.usable(sm.?) or sm.?.top_k != r.k)) return draw.pick(h, st, a, d, b, position, sm, batched);
        const data = std.mem.asBytes(&r.words);
        const vals: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data[vals_at + row * r.k * 2 ..][0 .. r.k * 2]));
        const cols: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, data[cols_at + row * r.k * 4 ..][0 .. r.k * 4]));
        return draw.pickFrom(h, a, b, vals, cols, position, sm) catch |err| {
            if (err == error.ProposalNonfinite) draw.badRow(st, d, b, batched, d.dlogits.ptr, h.ids.len);
            return err;
        };
    }
    fn load(r: *Ready, st: cuda.Stream, ptr: u64, rows: usize) !void {
        const n = if (r.k == 0) rows * 16 else cols_at + rows * r.k * 4;
        if (n > bytes) return error.ProposalScoreShape;
        try st.synchronize();
        try cuda.memory.download(st.d, ptr, std.mem.asBytes(&r.words)[0..n]);
    }
};

pub fn single(h: *const mtp.Head, st: cuda.Stream, a: std.mem.Allocator, d: *mtp.Draft, position: u64, sm: ?mtp.Sampling, batch_on: bool) !mtp.Pick {
    const n = h.ids.len;
    try h.best(st, d.dlogits.ptr, n, n, 1, d.best.ptr);
    if (!batch_on) {
        try st.synchronize();
        try cuda.memory.download(st.d, d.best.ptr, std.mem.sliceAsBytes(d.best_host[0..4]));
        return draw.pick(h, st, a, d, d.best_host[0..4].*, position, sm, false);
    }
    var ready = Ready{ .k = if (mtp.isSampled(sm) and draw.usable(sm.?)) sm.?.top_k else 0 };
    if (ready.k != 0) try h.topk(st, d.dlogits.ptr, n, n, 1, ready.k, d.best.ptr + vals_at, d.best.ptr + cols_at);
    try ready.load(st, d.best.ptr, 1);
    return ready.pick(h, st, a, d, 0, position, sm, false);
}

pub fn gather(h: *const mtp.Head, st: cuda.Stream, ds: []const *mtp.Draft, active: []const usize, sms: []const ?mtp.Sampling) !Ready {
    const n = h.ids.len;
    if (active.len > lanes_max or active.len > nrows) return error.ProposalScoreShape;
    var ready = Ready{ .k = 0 };
    for (active, 0..) |i, row| {
        try st.d.check(st.d.api.cuMemcpyDtoDAsync_v2(h.draw_batch.scores.ptr + row * n * 2, ds[i].dlogits.ptr, n * 2, st.handle), "cuMemcpyDtoDAsync");
        if (ready.k == 0 and mtp.isSampled(sms[i]) and draw.usable(sms[i].?)) ready.k = sms[i].?.top_k;
    }
    try h.best(st, h.draw_batch.scores.ptr, n, n, active.len, ds[0].best.ptr);
    if (ready.k != 0) try h.topk(st, h.draw_batch.scores.ptr, n, n, active.len, ready.k, ds[0].best.ptr + vals_at, ds[0].best.ptr + cols_at);
    try ready.load(st, ds[0].best.ptr, active.len);
    return ready;
}
