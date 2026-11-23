const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const types = @import("cuda_mtp_types.zig");
const mtp_draw = @import("cuda_mtp_draw.zig");

const max_lanes = types.max_lanes;
const ranks = types.ranks;
const most_window = types.most_window;
const anc_stride = 1 + max_lanes;

pub fn holdTreeSampled(d: anytype, ctx: anytype, pos0: u64, l: *const lanes.shape.Shape, sampling: ?lanes.Sampling, anc_at: usize) !void {
    const e = d.e;
    const g = e.g;
    const n = l.parents.len;
    if (n > max_lanes) return error.TooManyLanes;
    var cands: [max_lanes][ranks]u32 = undefined;
    var internal: [max_lanes]bool = @splat(false);
    for (l.parents, l.depths, l.ranks, 0..) |par, depth, rank, i| {
        if (rank >= ranks or par >= @as(i32, @intCast(i)) or depth >= most_window) return error.BadLanes;
        if (par >= 0) internal[@intCast(par)] = true else ctx.held[i] = ctx.cands[rank];
    }
    var done: [max_lanes]bool = @splat(false);
    var depth: u8 = 0;
    while (depth < most_window) : (depth += 1) {
        var batch: [max_lanes]usize = undefined;
        var rows: usize = 0;
        for (l.depths, 0..) |dd, i| if (dd == depth and internal[i]) {
            batch[rows] = i;
            rows += 1;
        };
        if (rows == 0) {
            var any = false;
            for (l.depths, 0..) |dd, i| any = any or (dd > depth and internal[i]);
            if (!any) break;
            continue;
        }
        const st = d.staged.bytes;
        const anc: []i32 = @alignCast(std.mem.bytesAsSlice(i32, st[anc_at..][0 .. rows * anc_stride * 4]));
        var toks: [max_lanes]u32 = undefined;
        for (batch[0..rows], 0..) |i, m| {
            const par = l.parents[i];
            const src = if (par < 0) ctx.last.ptr else d.lane_state + @as(u64, @intCast(par)) * g.hidden * 2;
            try d.scratch.copyFrom(d.hin - d.scratch.ptr + m * g.hidden * 2, src, g.hidden * 2, e.stream.handle);
            toks[m] = ctx.held[i];
            var count: usize = 0;
            var ancestor = par;
            while (ancestor >= 0) : (ancestor = l.parents[@intCast(ancestor)]) {
                anc[m * anc_stride + 1 + count] = ancestor;
                count += 1;
            }
            std.mem.reverse(i32, anc[m * anc_stride + 1 ..][0..count]);
            anc[m * anc_stride] = @intCast(count);
        }
        try d.putIds(toks[0..rows]);
        try cuda.copy_source.enqueue(e.d, d.anc, st[anc_at..][0 .. rows * anc_stride * 4], e.stream);
        try d.layer(ctx, rows, pos0 + depth, .{ .tree = d.anc });
        const kw = d.kvWidth();
        for (batch[0..rows], 0..) |i, m| {
            try d.scratch.copyFrom(d.lane_state - d.scratch.ptr + i * g.hidden * 2, d.x + m * g.hidden * 2, g.hidden * 2, e.stream.handle);
            try d.scratch.copyFrom(d.lane_k - d.scratch.ptr + i * kw * 2, d.kb + m * kw * 2, kw * 2, e.stream.handle);
            try d.scratch.copyFrom(d.lane_v - d.scratch.ptr + i * kw * 2, d.vb + m * kw * 2, kw * 2, e.stream.handle);
            done[i] = true;
        }
        const specs: [max_lanes]mtp_draw.Spec = @splat(.{ .sampling = sampling, .position = pos0 + depth + 2 });
        try d.drawSpecs(0, specs[0..rows], true);
        var got: [max_lanes][ranks]u32 = undefined;
        try d.fetchSpecs(specs[0..rows], &.{}, got[0..rows]);
        for (batch[0..rows], 0..) |i, m| cands[i] = got[m];
        for (l.parents, l.ranks, 0..) |par, rank, i| if (par >= 0 and done[@intCast(par)]) {
            ctx.held[i] = cands[@intCast(par)][rank];
        };
    }
    ctx.held_n = n;
}
