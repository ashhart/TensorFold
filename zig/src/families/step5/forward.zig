//! Up to 16 rows through every layer in one command buffer: dense attention over the cache, routed experts.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

/// The glue kernels' constants (glue.metal's s5_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, span: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, theta: f32 = 0, limit: f32 = 0 };

const Enc = struct {
    e: mtl.ComputeEncoder,
    m: *const Model,

    fn glue(x: Enc, g: Glue, args: Args, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        x.e.setPipeline(x.m.pipe(g));
        var slot: usize = 0;
        for (bufs) |r| {
            if (r.args) x.e.setValue(args, slot) else x.e.setBuffer(r.buf, r.off, slot);
            slot += 1;
        }
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    fn project(x: Enc, w: wts.Linear, in: mtl.Buffer, in_off: usize, out: mtl.Buffer, out_off: usize, n_rows: usize) !void {
        const call = try rows.call(&x.m.rows, w, n_rows, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, in_off, 0);
        x.e.setBuffer(w.w, w.w_off, 1);
        x.e.setBuffer(w.scales, w.s_off, 2);
        x.e.setBuffer(w.biases, w.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, out_off, 5);
        x.e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    fn experts(x: Enc, w: rows.Experts, in: mtl.Buffer, out: mtl.Buffer, ids: mtl.Buffer, slots: usize) !void {
        const call = try rows.callIndexed(&x.m.rows, w, slots, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, 0, 0);
        x.e.setBuffer(w.one.w, w.one.w_off, 1);
        x.e.setBuffer(w.one.scales, w.one.s_off, 2);
        x.e.setBuffer(w.one.biases, w.one.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, 0, 5);
        x.e.setBuffer(ids, 0, 6);
        x.e.dispatchGroups(mtl.Size.of(call.groups, slots, 1), mtl.Size.of(call.threads, 1, 1));
    }
};

const B = struct { buf: mtl.Buffer = undefined, off: usize = 0, args: bool = false };
fn b(buf: mtl.Buffer) B {
    return .{ .buf = buf };
}
fn bo(buf: mtl.Buffer, off: usize) B {
    return .{ .buf = buf, .off = off };
}
fn t(x: wts.Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}
const A = B{ .args = true };

/// `ids` at positions cache.len.. through the model: keys and values into the cache (len unmoved), last logits out.
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32) !void {
    const c = &m.config;
    const n = ids.len;
    const start = cache.len;
    if (n == 0 or n > st.window_rows) return error.Step5RowsExceeded;
    if (start + n > cache.capacity or start + n > c.dense_limit) return error.Step5ContextFull;
    for (ids) |id| if (id >= c.vocab) return error.InvalidStep5Token;
    @memcpy(s.ids.slice(u32, n), ids);
    const h: usize = c.hidden;
    const kvw: usize = c.kvWidth();
    const qw: usize = @as(usize, c.heads) * c.head_dim;
    const nr: u32 = @intCast(n);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    const views = [2]View{ masks(s, .sliding, c.window, start, n, 0), masks(s, .full, c.window, start, n, st.window_rows * cache.capacity) };

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m };
    const W = m.weights;
    const embed = Args{ .rows = nr, .width = c.hidden, .bits = W.embed.bits, .group = @intCast(W.embed.group) };
    x.glue(.embed, embed, &.{ b(s.ids), .{ .buf = W.embed.w, .off = W.embed.w_off }, .{ .buf = W.embed.scales, .off = W.embed.s_off }, .{ .buf = W.embed.biases, .off = W.embed.b_off }, A, b(s.resid) }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    const normed = Args{ .rows = nr, .width = c.hidden, .eps = c.eps };
    for (W.layers, 0..) |l, i| {
        const lay = c.layer[i];
        x.glue(.norm, normed, &.{ b(s.resid), t(l.input_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(l.attn.q, s.x, 0, s.q, 0, n);
        try x.project(l.attn.k, s.x, 0, s.k, 0, n);
        try x.project(l.attn.v, s.x, 0, s.v, 0, n);
        try x.project(l.attn.g, s.x, 0, s.g, 0, n);
        const rot = Args{ .rows = nr, .dim = c.head_dim, .span = lay.span, .pos = @intCast(start), .eps = c.eps, .theta = lay.theta };
        var qa = rot;
        qa.heads = c.heads;
        qa.stride = @intCast(qw);
        x.glue(.qk, qa, &.{ b(s.q), t(l.attn.q_norm), A, b(s.qn) }, .{ c.heads, n, 1 }, .{ 32, 1, 1 }, true);
        var ka = rot;
        ka.heads = c.kv_heads;
        ka.stride = @intCast(kvw);
        x.glue(.qk, ka, &.{ b(s.k), t(l.attn.k_norm), A, bo(cache.keys, cache.at(c, i, start)) }, .{ c.kv_heads, n, 1 }, .{ 32, 1, 1 }, true);
        x.glue(.copy, .{ .rows = nr, .width = @intCast(kvw), .stride = @intCast(kvw) }, &.{ b(s.v), A, bo(cache.values, cache.at(c, i, start)) }, .{ kvw, n, 1 }, .{ 256, 1, 1 }, false);
        const v = views[@intFromEnum(lay.kind)];
        try x.m.attention.encode(x.e, n, v.keys, v.keys, scale, .{ .buf = s.qn }, .{ .buf = cache.keys, .off = cache.at(c, i, v.lo) }, .{ .buf = cache.values, .off = cache.at(c, i, v.lo) }, .{ .buf = s.mask, .off = v.off }, .{ .buf = s.o });
        x.glue(.gate, .{ .rows = nr, .heads = c.heads, .dim = c.head_dim }, &.{ b(s.o), b(s.g), A }, .{ qw, n, 1 }, .{ 256, 1, 1 }, false);
        try x.project(l.attn.o, s.o, 0, s.y, 0, n);
        x.glue(.add, .{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        x.glue(.norm, normed, &.{ b(s.resid), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
        switch (l.ffn) {
            .dense => |d| {
                try mlp(x, d, s, s.y, n, c.intermediate, lay.limit);
                x.glue(.add, .{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
            },
            .moe => |e| {
                route.logits(x.e, m.route[0], .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk }, b(s.x), b(e.router), b(s.logits32), nr);
                route.select(x.e, m.route[1], b(s.logits32), t(e.bias), c.routed_scale, nr, .{ 0, c.experts }, .{
                    .pick = b(s.pick), .wts = b(s.wts), .ids = b(s.route[0]), .members = b(s.route[1]), .count = b(s.route[2]),
                    .mine = b(s.route[3]), .theirs = b(s.route[4]), .counts = b(s.route[5]), .word = b(s.route[6]),
                });
                const slots = n * c.topk;
                var gate = e.gate;
                gate.repeat = c.topk;
                var up = e.up;
                up.repeat = c.topk;
                try x.experts(gate, s.x, s.eg, s.pick, slots);
                try x.experts(up, s.x, s.eu, s.pick, slots);
                x.glue(.swiglu, .{ .rows = @intCast(slots), .width = c.moe_intermediate, .limit = lay.limit }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_intermediate, slots, 1 }, .{ 256, 1, 1 }, false);
                try x.experts(e.down, s.ea, s.ey, s.pick, slots);
                try mlp(x, e.shared, s, s.shared, n, c.shared_intermediate, lay.shared_limit);
                x.glue(.moe_add, .{ .rows = nr, .width = c.hidden, .heads = c.topk }, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
            },
        }
    }
    x.glue(.final_norm, normed, &.{ bo(s.resid, (n - 1) * h * 4), t(W.norm), A, b(s.x) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, true);
    try x.project(W.head, s.x, 0, cache.logits, 0, 1);
    x.e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("step5 forward: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// gate and up from s.x, the clamped SwiGLU, down into `out`.
fn mlp(x: Enc, w: wts.Mlp, s: *st.Scratch, out: mtl.Buffer, n: usize, inter: u32, limit: f32) !void {
    try x.project(w.gate, s.x, 0, s.gate, 0, n);
    try x.project(w.up, s.x, 0, s.up, 0, n);
    x.glue(.swiglu, .{ .rows = @intCast(n), .width = inter, .limit = limit }, &.{ b(s.gate), b(s.up), A, b(s.act) }, .{ inter, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(w.down, s.act, 0, out, 0, n);
}

const View = struct { lo: usize, keys: usize, off: usize };

/// Row r (position start + r) sees key lo + j when it is not later, and on sliding layers within the window.
fn masks(s: *st.Scratch, kind: cfg.Kind, window: u32, start: usize, n: usize, off: usize) View {
    const end = start + n;
    const lo = if (kind == .sliding and start + 1 > window) start + 1 - window else 0;
    const keys = end - lo;
    const bytes = (s.mask.contents() + off)[0 .. n * keys];
    for (0..n) |r| for (0..keys) |j| {
        const p = start + r;
        const k = lo + j;
        bytes[r * keys + j] = @intFromBool(k <= p and (kind == .full or p - k < window));
    };
    return .{ .lo = lo, .keys = keys, .off = off };
}
