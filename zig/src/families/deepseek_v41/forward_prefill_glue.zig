//! A prefill segment's glue steps and the tables and buffers they need once.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const longpf = @import("longpf.zig");
const hooks = @import("forward_hooks.zig");
const fwd = @import("forward.zig");
const ced = @import("ced.zig");
const Forward = fwd.Forward;
const pf = @import("forward_prefill.zig");
const pf_finish = @import("forward_prefill_finish.zig");
const Tail = pf.Tail;
const enterRun = pf_finish.enterRun;

/// glue pf_seg: segment j's slot current, its rows pending, its positions; -1: the run's view again (enterRun).
pub fn segGlue(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const m = f.multi orelse return error.BadGlue;
    const j = c.args[0].arg.i;
    if (j < 0) return enterRun(f, m);
    const sg = m.segs[@intCast(j)];
    const sw = f.pf_switch.?;
    try sw.run(sw.ctx, sg.slot);
    f.ids = sg.ids;
    f.slot.pending = .{ .start = sg.start, .n = @intCast(sg.ids.len) };
    const k = r.kernels.others(r.stream);
    const lo = try r.tensorAddr(c.args[3].arg.t);
    try k.positions(@intCast(sg.start), @intCast(sg.ids.len), try r.tensorAddr(c.args[1].arg.t), try r.tensorAddr(c.args[2].arg.t), lo);
    // a decoder replay: every row's window starts at the segment's first row (R0)
    if (c.args[4].arg.b) try r.d.check(r.d.api.cuMemsetD32Async(lo, @bitCast(@as(i32, @intCast(sg.start))), sg.ids.len, r.stream.handle), "cuMemsetD32Async");
}

/// glue mpositions: every segment's positions into the run's "w.mpos64" (rope over the run's rows).
pub fn runPositions(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const m = f.multi orelse return error.BadGlue;
    const k = r.kernels.others(r.stream);
    const mpos = try r.tensorAddr(c.args[0].arg.t);
    const pos = r.addressOf("w.pos") orelse return error.Unbound;
    const lo = r.addressOf("w.lo") orelse return error.Unbound;
    for (m.segs, m.row0) |sg, r0| try k.positions(@intCast(sg.start), @intCast(sg.ids.len), pos, mpos + 8 * r0, lo);
}

/// ced_stash (`store`): a segment's last rows into the ring at row p % ring; ced_load: ring rows back to rows 0 .. n.
pub fn stashCopy(f: *Forward, r: *run.Runner, c: *const calls.Call, store: bool) !void {
    const p = f.slot.pending orelse return error.BadGlue;
    if (f.pf_stash) |h| try h.run(h.ctx); // the ring holds this slot's rows (slots.zig swaps it lazily)
    const x = c.args;
    if (x.len != 2 * ced.roles.len) return error.BadGlue;
    const k: u64 = @min(p.n, ced.ring);
    const a0: u64 = p.start + p.n - k;
    var sp: [2]ced.Span = undefined;
    for (0..ced.roles.len) |j| {
        const t = x[j].arg.t;
        const rg = x[ced.roles.len + j].arg.t;
        const row = dim(t, 1) * buffers.dtSize(t.dt);
        if (dim(rg, 1) * buffers.dtSize(rg.dt) != row or ld(t) != dim(t, 1) or dim(rg, 0) != ced.ring) return error.BadGlue;
        const ta = try r.tensorAddr(t);
        const ra = try r.tensorAddr(rg);
        for (ced.spans(a0, a0 + k, &sp)) |s| {
            const tr = ta + (a0 - p.start + s.at) * row;
            const rr = ra + @as(u64, s.row) * row;
            if (store) try copy(r, rr, tr, s.n * row) else try copy(r, tr, rr, s.n * row);
        }
    }
}

/// Once: the tables no loader form builds and the buffers a segment's exchanges / Engram rows need.
pub fn ready(f: *Forward) !void {
    const st = pf.state(f);
    if (st.ready) return;
    const r = f.runner;
    // exl3.prefill's Hadamard: H[i, j] = (-1)^popcount(i & j), bf16
    if (r.addressOf("s.pf.had")) |p| {
        const h = try f.gpa.alloc(u16, 128 * 128);
        defer f.gpa.free(h);
        for (0..128) |i| for (0..128) |j| {
            h[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xBF80 else 0x3F80;
        };
        try f.upload(p, std.mem.sliceAsBytes(h));
    }
    // x3gm's pointer tables: the routed experts' trellis addresses (forms' tables hold the shared expert's last)
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "tg", "tp_g" }, .{ "tu", "tp_u" }, .{ "td", "tp_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoExpertTable;
        try copy(r, dst, src, 8 * @as(usize, f.cfg.expertsOf(L).count));
    };
    // x3gm v2's width tables (gm2pf.zig, x3gm.Ragged.k2g / k2d): the routed experts' K2s, then a 0 sentinel
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "k2g", "k2_g" }, .{ "k2d", "k2_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoK2Table;
        const Et: usize = f.cfg.expertsOf(L).count;
        try copy(r, dst, src, 4 * Et);
        try zero(r, dst + 4 * Et, 4);
    };
    // a segment's exchange send and Engram rows at their largest (decode's would grow them on the first segment)
    const rows: usize = @intCast(f.opts.prefill_rows);
    const W: usize = f.comm.world();
    _ = try f.sendBuffer(r, 2 * rows * f.cfg.hidden);
    if (f.engram) |h| {
        const cols = h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
        _ = try grow(&f.gathered, r.d, W * 2 * rows * cols);
        for (0..h.layer_ids.len) |li| _ = try f.rows_stage[li].take(r.d, 2 * rows * cols);
    }
    _ = try f.ids_stage.take(r.d, 8 * rows);
    try r.stream.synchronize();
    st.ready = true;
}

pub fn grow(b: *?cuda.DeviceBuffer, d: *const cuda.Driver, bytes: usize) !u64 {
    return @import("grow_device.zig").grow(b, d, @max(bytes, 256));
}

pub fn copy(r: *run.Runner, dst: u64, src: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, r.stream.handle), "cuMemcpyDtoDAsync");
}

pub fn zero(r: *run.Runner, dst: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemsetD8Async(dst, 0, bytes, r.stream.handle), "cuMemsetD8Async");
}

pub fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |x| n *= @intCast(x);
    return n;
}

pub fn bytesOf(t: calls.Tensor) usize {
    return numel(t) * buffers.dtSize(t.dt);
}

pub fn dim(t: calls.Tensor, i: usize) usize {
    return @intCast(t.shape[i]);
}

pub fn ld(t: calls.Tensor) usize {
    return @intCast(t.stride[0]);
}

/// A prefill segment's glue steps; the ones a decode window has too are the decode glue's.
pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = pf.state(f);
    const pr = st.tbo orelse return glueOne(ctx, r, c);
    // TF_DSV41_PF_TBO: the step's micro-batch (B's are the side stream's) on the forward while it runs
    const m = if (c.side) &pr.b else &pr.a;
    const keep_ids = f.ids;
    const keep_pending = f.slot.pending;
    var keep_tail: Tail = .{ .len = f.slot.tail_len };
    @memcpy(keep_tail.ids[0..keep_tail.len], f.slot.tail[0..f.slot.tail_len]);
    f.ids = m.ids;
    f.slot.pending = m.pending;
    f.slot.tail_len = m.tail.len;
    @memcpy(f.slot.tail[0..m.tail.len], m.tail.items());
    st.moe = m.moe;
    st.layer = m.layer;
    if (c.side) std.mem.swap(longpf.State, &st.long, &st.long_b);
    defer {
        if (c.side) std.mem.swap(longpf.State, &st.long, &st.long_b);
        m.moe = st.moe;
        m.layer = st.layer;
        f.ids = keep_ids;
        f.slot.pending = keep_pending;
        f.slot.tail_len = keep_tail.len;
        @memcpy(f.slot.tail[0..keep_tail.len], keep_tail.items());
    }
    return glueOne(ctx, r, c);
}

pub fn glueOne(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    if (try @import("prefill_heads.zig").glue(r, c)) return;
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = pf.state(f);
    const step = c.name["glue.".len..];
    const k = r.kernels.others(r.stream);
    const x = c.args;
    const eql = std.mem.eql;
    if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.BadGlue).glue(r, c, step);
    if (eql(u8, step, "proj_carry")) {
        // torch.cat([the slot's carry, the projection]): position start - 1's row, then the segment's
        const carry = x[0].arg.t;
        const buf = try r.tensorAddr(x[2].arg.t);
        try copy(r, buf, try r.tensorAddr(carry), bytesOf(carry));
        return copy(r, buf + bytesOf(carry), try r.tensorAddr(x[1].arg.t), bytesOf(x[1].arg.t));
    }
    if (eql(u8, step, "stream_merge")) return longpf.merge(&st.long, f.gpa, r, c);
    // TF_DSV41_PF_TBO: A's ratio-2 carry into the slot at its layer's end; the pair's closing join has nothing to run
    if (eql(u8, step, "copy_rows")) return copy(r, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "tbo_join")) return;
    if (eql(u8, step, "pf_seg")) return segGlue(f, r, c);
    if (eql(u8, step, "mpositions")) return runPositions(f, r, c);
    if (eql(u8, step, "ced_stash") or eql(u8, step, "ced_load")) return stashCopy(f, r, c, eql(u8, step, "ced_stash"));
    if (eql(u8, step, "zeros")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "image_keep")) return (try hooks.vision()).keep(f, r, c);
    if (eql(u8, step, "image_split") or eql(u8, step, "image_merge")) return (try hooks.vision()).split(f, r, c, eql(u8, step, "image_merge"));
    if (eql(u8, step, "gm_ticket_again")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t)); // vision.moe's text call: the same layer
    if (eql(u8, step, "cand_keys")) {
        const blocks = x[0].arg.t;
        return k.candKeys(try r.tensorAddr(blocks), ld(blocks), dim(blocks, 0), dim(blocks, 1), @intCast(x[2].arg.i), try r.tensorAddr(x[1].arg.t));
    }
    if (eql(u8, step, "proj")) {
        // blocks._projection: f32(kv), or ratio 2 [f32(kv) | f32(gate)]
        const kv = x[0].arg.t;
        const gate: u64 = if (x.len == 3) try r.tensorAddr(x[1].arg.t) else 0;
        return k.widenCat(try r.tensorAddr(kv), gate, dim(kv, 0), dim(kv, 1), try r.tensorAddr(x[x.len - 1].arg.t));
    }
    if (eql(u8, step, "ix_w")) return k.f64Bf16F32(try r.tensorAddr(x[0].arg.t), try r.tensorAddr(x[1].arg.t), numel(x[0].arg.t));
    if (eql(u8, step, "stage_in") or eql(u8, step, "stage_out")) {
        // blocks.stage_rows(dst, dst_ring, src, src_ring, lo, hi), a tensor at a time (values, scales)
        const dst_ring: usize = @intCast(x[2].arg.i);
        const src_ring: usize = @intCast(x[5].arg.i);
        const lo: u64 = @intCast(x[6].arg.i);
        const hi: u64 = @intCast(x[7].arg.i);
        for (0..2) |j| {
            const dst = x[j].arg.t;
            const src = x[3 + j].arg.t;
            const row = dim(dst, 1) * buffers.dtSize(dst.dt);
            if (dim(src, 1) * buffers.dtSize(src.dt) != row) return error.BadGlue;
            try k.ringCopy(try r.tensorAddr(src), src_ring, try r.tensorAddr(dst), dst_ring, row, lo, hi);
        }
        return;
    }
    if (eql(u8, step, "gm_picks")) {
        const pick = x[0].arg.t;
        return k.gmPicks(try r.tensorAddr(pick), try r.tensorAddr(x[1].arg.t), dim(pick, 1), dim(x[2].arg.t, 1), dim(pick, 0), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t));
    }
    if (eql(u8, step, "gm_ticket")) {
        // a layer's MoE starts here: the layer gm_plan's K2 tables are of
        if (st.moe >= st.layers.len) return error.BadGlue;
        st.layer = st.layers[st.moe];
        st.moe += 1;
        return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    }
    if (eql(u8, step, "gm_plan")) {
        // x3gm._run_ragged: the picks of one width (the others past the table), then x3gm.plan's passes
        const pk = x[0].arg.t;
        const P = numel(pk);
        const k2: u32 = @intCast(x[6].arg.i);
        const down = x[7].arg.b;
        const bm: usize = @intCast(x[8].arg.i);
        const Et: usize = @intCast(x[9].arg.i);
        // k2 0 (x3gm._run_v2's one plan, gm2pf.zig): every pick, no width mask
        if (k2 == 0) return r.kernels.experts(r.stream).gmPlan(try r.tensorAddr(pk), P, Et, bm, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t));
        var nb: [64]u8 = undefined;
        // forms' "s.L<i>.ex.k2_{g,d}": each routed expert's K2 (gate = up), the shared expert's past Et (never read)
        const tab = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.k2_{s}", .{ st.layer, if (down) "d" else "g" })) orelse return error.NoK2Table;
        const masked = try grow(&st.masked, r.d, 4 * P);
        try k.widthMask(try r.tensorAddr(pk), P, tab, Et, k2, masked);
        const t = [5]u64{ try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t) };
        return r.kernels.experts(r.stream).gmPlan(masked, P, Et, bm, t[0], t[1], t[2], t[3], t[4]);
    }
    if (eql(u8, step, "swiglu")) {
        // the shared expert's silu(min(g, limit)) * clamp(u, +-limit), fp32
        const g = x[0].arg.t;
        const u = x[1].arg.t;
        const act = x[2].arg.t;
        return k.swiglu(try r.tensorAddr(g), ld(g), try r.tensorAddr(u), ld(u), try r.tensorAddr(act), ld(act), dim(g, 0), dim(g, 1), @floatCast(x[3].arg.f));
    }
    if (eql(u8, step, "moe_sum")) {
        // bf16(routed + shared): one RNE of the fp32 sum; no shared expert: the routed partial's cast
        const routed = x[0].arg.t;
        if (x.len == 3) return k.addBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), numel(routed));
        return k.castBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), numel(routed));
    }
    return Forward.glueFn(ctx, r, c);
}
