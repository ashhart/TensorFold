//! Ling's KDA mixer on CUDA: seven projections in their loaded formats feeding GLM's chain, then o_proj.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");

const kda = graph.kda;
const Ops = graph.Ops;

/// One sequence's rows of a forward: rows row0 .. row0 + rows on `st` from its position.
pub const Seg = struct { st: *state.State, row0: usize, rows: usize };

/// Where segment `sg`'s projection rows and saves go: the State's window rows, or the Buffers' prompt rows.
fn place(b: *const state.Buffers, sg: Seg, li: usize) struct { p: u64, k: u64, v: u64, g: u64, b: u64 } {
    const st = sg.st;
    if (!b.prompt) {
        const s = st.saves(li);
        return .{ .p = s.p, .k = s.k, .v = s.v, .g = s.g, .b = s.b };
    }
    const n = st.heads * kda.dk;
    const r = sg.row0;
    return .{ .p = b.kp + r * st.stride * 2, .k = b.saves.k + r * n * 4, .v = b.saves.v + r * n * 2, .g = b.saves.g + r * n * 4, .b = b.saves.b + r * st.heads * 4 };
}

/// The block on b.normed into b.y: layer `li` among the KDA layers; prompt rows commit their states and conv windows.
pub fn block(o: Ops, b: *const state.Buffers, a: weights.Kda, c: *const @import("config.zig").Config, li: usize, segs: []const Seg, rows: usize) !void {
    const D: usize = c.hidden;
    const H: usize = c.kda_heads;
    const n = H * kda.dk;
    const ps = segs[0].st.stride;
    for ([_]weights.Proj{ a.q, a.k, a.v, a.b }, [_]usize{ 0, n, 2 * n, 3 * n }) |w, col0| {
        if (rows == 1) { // one row is its own columns: no copy
            try o.mm(w, b.normed, D, 1, place(b, segs[0], li).p + col0 * 2);
            continue;
        }
        try o.mm(w, b.normed, D, rows, b.tmp);
        const w_n = w.outputs();
        for (segs) |sg| try o.copy(b.tmp + sg.row0 * w_n * 2, w_n * 2, place(b, sg, li).p + col0 * 2, ps * 2, sg.rows, w_n * 2);
    }
    try o.mm(a.f, b.normed, D, rows, b.ka);
    try o.mm(a.g, b.normed, D, rows, b.kg);
    for (segs) |sg| {
        const st = sg.st;
        const at = place(b, sg, li);
        const cur = st.cur[li];
        const ch: kda.Chain = .{
            .p = at.p,
            .p_stride = ps,
            .b_off = 3 * n,
            .a = b.ka + sg.row0 * n * 2,
            .a_stride = n,
            .g = b.kg + sg.row0 * n * 2,
            .g_stride = n,
            .conv_state = st.convAt(li),
            .conv_w = a.conv,
            .state_in = st.recAt(cur, li),
            .a_log = a.a_log,
            .dt_bias = a.dt_bias,
            .norm_w = a.o_norm,
            .eps = c.eps,
            .lower = c.lower_bound,
            .heads = H,
        };
        const sv: kda.Saves = .{ .out = b.saves.out + sg.row0 * n * 2, .state_out = st.recAt(1 - cur, li), .k = at.k, .v = at.v, .g = at.g, .b = at.b };
        const wide: ?kda.Wide = if (if (b.prompt) sg.rows >= kda.wide_rows else o.kda_wide) .{ .q = b.wide.q + sg.row0 * n * 4, .y = b.wide.y + sg.row0 * n * 2 } else null;
        try kda.chain(&o.m.kda, o.s, ch, sg.rows, sv, wide);
        if (!b.prompt) continue;
        try graph.glue.convShift(o.set(), o.s, st.convAt(li), at.p, sg.rows, sg.rows * ps, ps, 1, 3 * n);
        st.cur[li] = 1 - cur; // a prompt keeps every row: the layer commits now
    }
    try o.mm(a.o, b.saves.out, n, rows, b.y);
}

/// A window's commit of `keep` rows on every KDA layer: the states replayed when rows are dropped, the conv windows.
pub fn commit(o: Ops, st: *state.State, keep: usize) !void {
    if (st.kda_layers == 0) return;
    const cur = st.cur[0];
    for (st.cur) |x| if (x != cur) return error.KdaLayersOutOfStep;
    const h = st.heads;
    if (keep < st.pending) {
        var a: cuda.Args = .{};
        a.add(@as(c_int, @intCast(h)));
        a.add(st.recAt(cur, 0));
        a.add(@as(usize, st.rec_layer / 4));
        for ([_]u64{ st.save_k, st.save_v, st.save_g, st.save_b }) |x| a.add(x);
        a.add(@as(usize, state.window_rows * h * kda.dk));
        a.add(@as(usize, state.window_rows * h));
        a.add(@as(c_int, @intCast(keep)));
        a.add(st.recAt(1 - cur, 0));
        try cuda.launch.launch(o.m.kda.replay_layers, .{ .grid = .{ .x = @intCast(st.kda_layers * h) }, .block = .{ .x = 1024 } }, o.s, &a);
    }
    for (st.cur) |*x| x.* = 1 - x.*;
    try graph.glue.convShift(o.set(), o.s, st.conv, st.proj, keep, state.window_rows * st.stride, st.stride, st.kda_layers, st.convWidth());
}
