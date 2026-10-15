//! `ling-layer-speed DIR FIXTURE [REPS]`: one real Ling-3.1 layer timed by CUDA events, whole and op by op, per phase.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_forward_rig.zig");
const bytes = @import("ling_layer_bytes.zig");

const F = rig.F;
const G = ling.graph;
const W = rig.W;
const kda = G.kda;
const latent = G.latent;
const glue = G.glue;

/// Events between op groups; each group's milliseconds by name, medians over reps.
const Timer = struct {
    ev: [48]cuda.Event,
    names: [48][]const u8 = undefined,
    n: usize = 0,
    s: cuda.Stream,
    on: bool,

    fn mark(t: *Timer, name: []const u8) !void {
        if (!t.on) return;
        try t.ev[t.n].record(t.s);
        t.names[t.n] = name;
        t.n += 1;
    }
};

const Seg = F.Seg;

fn kdaBlock(o: G.Ops, b: *const F.Buffers, a: W.Kda, c: *const ling.config.Config, li: usize, sg: Seg, rows: usize, t: *Timer) !void {
    const D: usize = c.hidden;
    const H: usize = c.kda_heads;
    const n = H * kda.dk;
    const st = sg.st;
    const ps = st.stride;
    const sv = st.saves(li);
    const p = if (b.prompt) b.kp else sv.p;
    for ([_]W.Proj{ a.q, a.k, a.v, a.b }, [_]usize{ 0, n, 2 * n, 3 * n }, [_][]const u8{ "proj q", "proj k", "proj v", "proj b (bf16)" }) |w, col0, name| {
        try o.mm(w, b.normed, D, rows, b.tmp);
        try t.mark(name);
        const w_n = w.outputs();
        try o.copy(b.tmp, w_n * 2, p + col0 * 2, ps * 2, rows, w_n * 2);
        try t.mark("copies into p");
    }
    try o.mm(a.f, b.normed, D, rows, b.ka);
    try t.mark("proj f");
    try o.mm(a.g, b.normed, D, rows, b.kg);
    try t.mark("proj g");
    const cur = st.cur[li];
    const ch: kda.Chain = .{ .p = p, .p_stride = ps, .b_off = 3 * n, .a = b.ka, .a_stride = n, .g = b.kg, .g_stride = n, .conv_state = st.convAt(li), .conv_w = a.conv, .state_in = st.recAt(cur, li), .a_log = a.a_log, .dt_bias = a.dt_bias, .norm_w = a.o_norm, .eps = c.eps, .lower = c.lower_bound, .heads = H };
    const k = if (b.prompt) b.saves.k else sv.k;
    const v = if (b.prompt) b.saves.v else sv.v;
    const g = if (b.prompt) b.saves.g else sv.g;
    const bb = if (b.prompt) b.saves.b else sv.b;
    const s: kda.Saves = .{ .out = b.saves.out, .state_out = st.recAt(1 - cur, li), .k = k, .v = v, .g = g, .b = bb };
    const wide: ?kda.Wide = if (if (b.prompt) rows >= kda.wide_rows else o.kda_wide) .{ .q = b.wide.q, .y = b.wide.y } else null;
    try kda.chain(&o.m.kda, o.s, ch, rows, s, wide);
    try t.mark("KDA chain");
    if (b.prompt) {
        try glue.convShift(o.set(), o.s, st.convAt(li), p, rows, rows * ps, ps, 1, 3 * n);
        try t.mark("conv shift");
    }
    try o.mm(a.o, b.saves.out, n, rows, b.y);
    try t.mark("proj o");
}

fn mlaBlock(o: G.Ops, b: *const F.Buffers, a: W.Mla, ab: G.Absorb, g: *const F.Graph, sg: Seg, rows: usize, t: *Timer) !void {
    const c = &g.w.config;
    const D: usize = c.hidden;
    const H: usize = c.mla_heads;
    const qk: usize = c.qkDim();
    const nope: usize = c.nope;
    const kw = F.cache_width;
    const lw = 512;
    const ql: usize = c.q_lora;
    try o.mm(a.q_a.?, b.normed, D, rows, b.qa_in);
    try t.mark("proj q_a");
    try glue.rmsnorm(o.set(), o.s, b.qa_in, ql, a.q_a_norm, b.qr, ql, null, c.eps, rows, ql);
    try t.mark("norms q_a, kv");
    try o.mm(a.q_b.?, b.qr, ql, rows, b.q);
    try t.mark("proj q_b");
    try o.mm(a.kv_a, b.normed, D, rows, b.kv);
    try t.mark("proj kv_a");
    try glue.rmsnorm(o.set(), o.s, b.kv, kw, a.kv_a_norm, b.lat, kw, null, c.eps, rows, lw);
    try t.mark("norms q_a, kv");
    try o.mm(a.gate.?, b.normed, D, rows, b.g);
    try t.mark("proj gate (bf16)");
    try latent.absorbQStrided(o.set(), o.s, b.q, qk, ab.wk, b.qa, kw, rows, H, nope, lw);
    try t.mark("absorb q");
    const st = sg.st;
    try latent.rope(o.set(), o.s, .{ .x = b.q + nope * 2, .x_row = H * qk, .x_head = qk, .out = b.qa + lw * 2, .o_row = H * kw, .o_head = kw }, rows, H, st.pos_dev, g.freq);
    try latent.rope(o.set(), o.s, .{ .x = b.kv + lw * 2, .x_row = kw, .x_head = 0, .out = b.lat + lw * 2, .o_row = kw, .o_head = 0 }, rows, 1, st.pos_dev, g.freq);
    try latent.latentWrite(o.set(), o.s, b.lat, kw, st.cacheAt(0), st.pos_dev, rows, kw);
    try t.mark("rope, cache write");
    var r0: usize = 0;
    while (r0 < rows) : (r0 += ling.state.att_rows) {
        const m = @min(ling.state.att_rows, rows - r0);
        var at = st.pos_dev;
        if (r0 > 0) {
            try o.addI32(st.pos_dev, r0, b.at);
            at = b.at;
        }
        const nch = latent.chunksFor(st.pos + r0 + m);
        try latent.attentionRoped(o.set(), o.s, b.qa + r0 * H * kw * 2, st.cacheAt(0), at, b.partials, b.ol + r0 * H * lw * 2, b.dummy, m, H, lw, nch, c.smScale());
    }
    try t.mark("latent attention");
    try latent.expandV(o.set(), o.s, b.ol, ab.wv, b.vn, rows, H, c.v_dim, lw);
    try t.mark("expand v");
    try o.gate(b.vn, b.g, H, rows, H, c.v_dim);
    try t.mark("head gate");
    try o.mm(a.o, b.vn, H * c.v_dim, rows, b.y);
    try t.mark("proj o");
}

fn mlp(o: G.Ops, b: *const F.Buffers, m: W.Mlp, rows: usize, limit: f32, out: u64, t: *Timer, comptime tag: []const u8) !void {
    const D = m.gate.inputs();
    const w = m.gate.outputs();
    for ([_]W.Proj{ m.gate, m.up }, [_]usize{ 0, w }) |p, col0| {
        try o.mm(p, b.normed, D, rows, b.tmp);
        try t.mark(tag ++ " gate, up");
        try o.copy(b.tmp, w * 2, b.gu + col0 * 2, 4 * w, rows, w * 2);
        try t.mark(tag ++ " copies");
    }
    try glue.swigluWith(o.m.swiglu, o.s, b.gu, b.act, b.xs, limit, rows, w);
    try t.mark(tag ++ " swiglu");
    try o.mm(m.down, b.act, w, rows, out);
    try t.mark(tag ++ " down");
}

fn moe(o: G.Ops, b: *const F.Buffers, e: W.Moe, g: *const F.Graph, rows: usize, t: *Timer) !void {
    const c = &g.w.config;
    const D: usize = c.hidden;
    const E: usize = c.experts;
    const top: usize = c.topk;
    const slots = top + 1;
    try glue.router(o.set(), o.s, b.normed, D, e.router, b.mlog, b.rpart, rows, D, E);
    try t.mark("router");
    try glue.select(o.set(), o.s, b.mlog, if (e.bias != 0) e.bias else g.zero_bias, b.pick, b.wts, rows, top, E, c.routed_scale, c.norm_topk, ling.groups(c));
    try t.mark("select");
    const l = e.routed.grouped;
    const tile: usize = if (o.prompt) cuda.experts.promptTile(l.format) else 16;
    try o.m.router.route(o.s, b.pick, rows * slots, E + 1, tile, b.plan);
    try t.mark("expert plan");
    const items = cuda.grouped.maxItems(rows * slots, E + 1, tile);
    const gr = o.m.grouped();
    const up: cuda.experts.Rows = .{ .x = b.normed, .stride = l.dims, .slots = slots };
    const down: cuda.experts.Rows = .{ .x = b.eact, .stride = l.width };
    const skip: i32 = @intCast(E);
    if (o.prompt) try gr.prompt(o.s, .up, up, l, b.plan, b.eact, items, skip) else try gr.decode(o.s, .up, up, l, b.plan, b.eact, items, skip);
    try t.mark("experts gate-up");
    if (o.prompt) try gr.prompt(o.s, .down_bf16, down, l, b.plan, b.ey, items, skip) else try gr.decode(o.s, .down_bf16, down, l, b.plan, b.ey, items, skip);
    try t.mark("experts down");
    if (o.prompt) { // the decode kernels on 16-pair items over the same rows, for comparison
        try o.m.router.route(o.s, b.pick, rows * slots, E + 1, 16, b.plan);
        try t.mark("alt plan 16");
        const it16 = cuda.grouped.maxItems(rows * slots, E + 1, 16);
        try gr.decode(o.s, .up, up, l, b.plan, b.eact, it16, skip);
        try t.mark("alt gate-up decode kernel");
        try gr.decode(o.s, .down_bf16, down, l, b.plan, b.ey, it16, skip);
        try t.mark("alt down decode kernel");
    }
    try mlp(o, b, e.shared.?, rows, e.shared_limit, b.sh, t, "shared");
    var a: cuda.Args = .{};
    inline for (.{ b.ey, b.pick, b.wts, b.sh, b.y, @as(i64, @intCast(rows)), @as(i64, @intCast(D)), @as(c_int, @intCast(slots)), @as(c_int, @intCast(top)), @as(c_int, 0) }) |v| a.add(v);
    try cuda.launch.launch(o.m.combine, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((D + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    try t.mark("combine");
}

/// The layer as forward.layer runs it, op groups marked.
fn layerOps(f: *const F.Forward, sg: Seg, rows: usize, t: *Timer) !void {
    const o = f.ops();
    const b = f.b;
    const g = f.g;
    const c = &g.w.config;
    const D: usize = c.hidden;
    const l = g.w.layers[0];
    try t.mark("start");
    try glue.rmsnorm(o.set(), o.s, b.x, D, l.attn_norm, b.normed, D, null, c.eps, rows, D);
    try t.mark("norms + residual adds");
    switch (l.mixer) {
        .kda => |a| try kdaBlock(o, b, a, c, 0, sg, rows, t),
        .mla => |a| try mlaBlock(o, b, a, g.absorb[0], g, sg, rows, t),
    }
    try o.add(b.x, b.y, b.x, rows * D);
    try glue.rmsnorm(o.set(), o.s, b.x, D, l.mlp_norm, b.normed, D, null, c.eps, rows, D);
    try t.mark("norms + residual adds");
    switch (l.ffn) {
        .dense => |m| try mlp(o, b, m, rows, 0, b.y, t, "dense"),
        .moe => |e| try moe(o, b, e, g, rows, t),
    }
    try o.add(b.x, b.y, b.x, rows * D);
    try t.mark("norms + residual adds");
}

const Row = struct { name: []const u8, ms: [32]f64 = undefined, n: usize = 0 };

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// `ling-layer-speed DIR FIXTURE [REPS]`: decode at position 2048, windows 2..16, prompts of 512 and 2048 rows.
pub fn run(gpu: check.Gpu, dir: []const u8, path: []const u8, reps_arg: ?[]const u8) !void {
    const reps: usize = if (reps_arg) |x| try std.fmt.parseInt(usize, x, 10) else 15;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k: bytes.Kit = undefined;
    try k.init(gpu, a, s, dir, path, 2048, 4200);
    defer k.deinit();
    var t: Timer = .{ .ev = undefined, .s = s, .on = true };
    for (&t.ev) |*e| e.* = try cuda.Event.init(gpu.d, true);
    defer for (&t.ev) |*e| e.deinit();
    const st = try k.r.state(k.cap);
    defer st.deinit();
    const c = &k.w.config;
    std.debug.print("RESULT ling-layer-speed layer {d} ({t}, {s}, {t}), {d} reps, median and least us (other GPU contexts' time slices land in the medians)\n", .{ k.L, c.kind(k.L), if (c.isMoe(k.L)) "MoE" else "dense", c.quant, reps });
    for ([_]usize{ 1, 2, 4, 8, 16, 512, 2048 }) |rows| {
        const b = if (rows > F.window_rows) &k.pb else &k.db;
        const pos: usize = if (rows > F.window_rows) 0 else 2048;
        try st.setPos(s, pos);
        const d = gpu.d;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(b.x, k.x.ptr, rows * k.D * 2, s.handle), "cuMemcpyHtoDAsync");
        const f = k.r.fwd(b, null);
        const sg: Seg = .{ .st = st, .row0 = 0, .rows = rows };
        var whole: [64]f64 = undefined;
        var table: [32]Row = undefined;
        var nt: usize = 0;
        for (0..reps + 2) |i| {
            // whole layer through forward.layer, then the same launches op by op
            try t.ev[46].record(s);
            try f.layer(&.{sg}, 0, rows, null);
            try t.ev[47].record(s);
            try s.synchronize();
            if (i >= 2) whole[i - 2] = try t.ev[46].elapsedMs(t.ev[47]);
            t.n = 0;
            try layerOps(&f, sg, rows, &t);
            try s.synchronize();
            if (i < 2) continue;
            var acc: [32]f64 = @splat(0);
            for (1..t.n) |j| {
                const name = t.names[j];
                var at: usize = 0;
                while (at < nt and !std.mem.eql(u8, table[at].name, name)) at += 1;
                if (at == nt) {
                    table[nt] = .{ .name = name };
                    nt += 1;
                }
                acc[at] += try t.ev[j - 1].elapsedMs(t.ev[j]);
            }
            for (table[0..nt], acc[0..nt]) |*r, v| {
                r.ms[r.n] = v;
                r.n += 1;
            }
        }
        var sum: f64 = 0;
        var low: f64 = 0;
        std.debug.print("RESULT   rows {d:>4}: layer {d:.1} {d:.1}", .{ rows, median(whole[0..reps]) * 1000, std.mem.min(f64, whole[0..reps]) * 1000 });
        for (table[0..nt]) |*r| {
            const lo = std.mem.min(f64, r.ms[0..r.n]) * 1000;
            const m = median(r.ms[0..r.n]) * 1000;
            sum += m;
            low += lo;
            std.debug.print(" | {s} {d:.1} {d:.1}", .{ r.name, m, lo });
        }
        std.debug.print(" | sum of ops {d:.1} {d:.1}\n", .{ sum, low });
    }
    check.pass("ling-layer-speed layer {d}: timed", .{k.L});
}
