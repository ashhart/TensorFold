//! One Mini forward's launches: embedding, each visit from the plan (core/visit_plan.zig), the head.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const attn = @import("kernels.zig");
const ops = @import("ops.zig");
const loop = @import("loop.zig");
const mm = @import("mini_model.zig");
const wts = @import("mini_weights.zig");

const Model = mm.Model;
const Kernels = mm.Kernels;
const Site = mm.Site;

/// A forward's rows as staged on the device: counts, the call's attention rows and its streams.
pub const Rows = struct {
    n: usize,
    streams: usize,
    prompt: bool,
    last_pos: u32,
    n_tiles: u32,

    /// Ids, positions, segments and tiles of `chains` uploaded on `s`.
    pub fn stage(m: *Model, s: cuda.Stream, chains: []const mm.Chain, rows: usize, prompt: bool) !Rows {
        const gpa = m.gpa;
        const ints = try gpa.alloc(i32, 3 * rows);
        defer gpa.free(ints);
        var segs: [2][mm.round_rows]attn.Seg = undefined;
        var cs: [mm.round_rows][4]i32 = undefined;
        const kvw: i64 = @as(i64, m.c.local.kv) * m.c.local.dim;
        var at: usize = 0;
        var last: u32 = 0;
        for (chains, 0..) |ch, i| {
            for (ch.tokens, 0..) |tok, j| {
                ints[at + j] = @intCast(tok);
                ints[rows + at + j] = @intCast(ch.p0 + j);
                ints[2 * rows + at + j] = @intCast(i);
            }
            const slot: i64 = @intCast(ch.slot);
            for ([_]attn.Layer{ m.sliding, m.full }, 0..) |l, side| {
                segs[side][i] = .{ .row0 = @intCast(at), .n = @intCast(ch.tokens.len), .p0 = @intCast(ch.p0), .off = slot * kvw * l.ring };
            }
            cs[i] = .{ @intCast(at), @intCast(ch.tokens.len), @intCast(ch.slot), 0 };
            last = @max(last, @as(u32, @intCast(ch.p0 + ch.tokens.len - 1)));
            at += ch.tokens.len;
        }
        const group = m.c.local.q / m.c.local.kv;
        var tiles: [attn.tileCount(mm.prompt_rows, 3) + mm.round_rows]attn.Tile = undefined;
        const n_tiles = try attn.tiles(segs[0][0..chains.len], group, &tiles);
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ints[0..rows]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(ints[rows .. 2 * rows]));
        try m.upload(s, m.s.seg_of, std.mem.sliceAsBytes(ints[2 * rows ..]));
        for (0..2) |side| try m.upload(s, m.s.segs[side], std.mem.sliceAsBytes(segs[side][0..chains.len]));
        try m.upload(s, m.s.conv_segs, std.mem.sliceAsBytes(cs[0..chains.len]));
        try m.upload(s, m.s.tiles, std.mem.sliceAsBytes(tiles[0..n_tiles]));
        return .{ .n = rows, .streams = chains.len, .prompt = prompt, .last_pos = last, .n_tiles = @intCast(n_tiles) };
    }
};

fn kvWidth(m: *const Model) usize {
    return @as(usize, m.c.local.kv) * m.c.local.dim;
}

/// The whole forward: embedding, every visit, then the head's rows [head0, head0 + heads).
pub fn run(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const Rows, head0: usize, heads: usize) !void {
    const t = m.c;
    const h: usize = t.hidden;
    try ops.embed(&k.ops, s, m.s.ids, r.n, m.w.embed, m.w.embed_norm, m.s.h[0], h, t.eps);
    var cur: usize = 0;
    for (0..m.w.plan.visits.len) |v| {
        try visitOne(m, k, s, r, @intCast(v), m.s.h[cur], m.s.h[1 - cur]);
        cur = 1 - cur;
    }
    if (heads == 0) return;
    const plane = r.n * h * 2;
    const nb = m.w.plan.final_sources - 1;
    var mix: ops.MixArgs = .{ .partial = m.s.h[cur], .query = m.w.output_query, .out = m.s.xf, .n_blocks = @intCast(nb), .partial_bf16 = 1, .nq = 1, .rows = @intCast(r.n), .c = @intCast(h), .eps = t.eps };
    for (0..nb) |j| mix.blocks[j] = m.s.blocks + j * plane;
    mix.norm[0] = m.w.norm;
    try ops.mix(&k.ops, s, mix);
    const v: usize = t.unpadded_vocab;
    try k.lane.matmulOut(s, m.s.xf + head0 * h * 2, h, heads, m.w.unembed, m.s.logits, .f32, m.s.lane_part);
    try ops.softcap(&k.ops, s, m.s.logits, heads * v, t.logit_cap);
}

/// The input projections' outputs: the scratch buffers, or a one-row forward's parts of the fused output.
const Out = struct { q: u64, k: u64, v: u64, rel: u64, gate: u64, sink: u64 };

fn proj(m: *const Model, k: *const Kernels, s: cuda.Stream, w: cuda.qmmf.Weight, x: u64, out: u64, n: usize) !void {
    try k.lane.matmulOut(s, x, w.k, n, w, out, .bf16, m.s.lane_part);
}

/// Bank the previous pass before a restart: its blocks past the entry, its output; the state restarts from the entry.
fn restart(m: *Model, s: cuda.Stream, r: *const Rows, hin: u64) !void {
    const p = &m.w.plan;
    const plane = r.n * m.c.hidden * 2;
    for (0..p.bank_blocks) |j| try m.copy(s, m.s.bank + j * plane, m.s.blocks + (p.pre_blocks + 1 + j) * plane, plane);
    try m.copy(s, m.s.bank_h, hin, plane);
    try m.copy(s, hin, m.s.blocks + p.pre_blocks * plane, plane);
}

/// The previous pass's messages for core layer `j`: four fp32 mixes over its banked blocks and output (q, k, v, mlp).
fn messages(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const Rows, j: u32) !void {
    const h: usize = m.c.hidden;
    const plane = r.n * h;
    var a: loop.MixArgs = .{ .partial = m.s.bank_h, .out = m.s.msg, .out_stride = @intCast(plane), .n_blocks = @intCast(m.w.plan.bank_blocks), .partial_f32 = 0, .nq = 4, .rows = @intCast(r.n), .h = @intCast(h), .eps = m.c.eps, .scale = scale(h) };
    for (0..m.w.plan.bank_blocks) |b| a.blocks[b] = m.s.bank + b * plane * 2;
    const row = @as(usize, j) * 2 * h * 4;
    a.query = .{ m.w.cross_main + row, m.w.cross_kv + row, m.w.cross_kv + row + h * 4, m.w.cross_main + row + h * 4 };
    try loop.mix(&k.loop, s, a);
}

fn scale(h: usize) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(h))));
}

/// A core input through its gates: the residual mix over blocks and partial, then each gate and its norm into `out`.
fn gated(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const Rows, nb: usize, partial: u64, f32_partial: bool, query: u64, nq: usize, ctl: []const wts.Controller, slots: [3]i32, norms: [3]u64, out: u64) !void {
    const h: usize = m.c.hidden;
    const plane = r.n * h;
    var a: loop.MixArgs = .{ .partial = partial, .out = m.s.mixf, .out_stride = @intCast(plane), .n_blocks = @intCast(nb), .partial_f32 = @intFromBool(f32_partial), .nq = @intCast(nq), .rows = @intCast(r.n), .h = @intCast(h), .eps = m.c.eps, .scale = scale(h) };
    for (0..nb) |b| a.blocks[b] = m.s.blocks + b * plane * 2;
    for (0..nq) |q| a.query[q] = query + q * h * 4;
    try loop.mix(&k.loop, s, a);
    var g: loop.GateArgs = .{ .msg = m.s.msg, .msg_stride = @intCast(plane), .down = m.s.gdown, .ys = m.s.gys, .out = out, .out_stride = @intCast(plane), .rows = @intCast(r.n), .h_dim = @intCast(h), .rank = @intCast(m.c.loop.?.rank), .eps_unit = m.c.eps, .eps = m.c.eps };
    for (0..ctl.len) |i| {
        g.h[i] = m.s.mixf + (if (nq == 3) i else 0) * plane * 4;
        g.slot[i] = slots[i];
        g.down_w[i] = ctl[i].down;
        g.forget[i] = ctl[i].forget;
        g.gain[i] = ctl[i].gain;
        g.norm[i] = norms[i];
    }
    try loop.gates(&k.loop, s, g, ctl.len);
}

/// A short conv at `site` of visit `v` over the call's streams (their states by slot).
fn conv(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const Rows, v: u32, site: Site, a: ops.ConvArgs) !void {
    var c = a;
    c.state = m.conv[v][@backingInt(site)];
    c.segs = m.s.conv_segs;
    c.state_stride = 3 * @as(i64, c.c);
    c.rows = @intCast(r.n);
    try ops.conv(&k.ops, s, c, r.streams);
}

fn visitOne(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const Rows, v: u32, hin: u64, hout: u64) !void {
    const t = m.c;
    const p = &m.w.plan;
    const step = p.visits.steps[v];
    const i: u32 = step.layer;
    const spec = p.at(i);
    const L = &m.w.layers[i];
    const a = &L.attn;
    const h: usize = t.hidden;
    const n = r.n;
    const plane = n * h;
    const global = spec.kind == .global;
    const msgs = p.messages(v);
    if (p.restarts(v)) try restart(m, s, r, hin);
    // the attention's inputs: layer 0 norms the embedding; later layers mix blocks; core passes after the first gate
    if (msgs) {
        try messages(m, k, s, r, p.coreIndex(v));
        try gated(m, k, s, r, spec.blocks_in, hin, false, L.attn_query, spec.attn_queries, L.controllers.?[0..3], .{ 0, 1, 2 }, a.norms, m.s.xin);
    } else {
        var mix: ops.MixArgs = .{ .partial = hin, .query = if (i == 0) 0 else L.attn_query, .out = m.s.xin, .n_blocks = @intCast(if (i == 0) 0 else spec.blocks_in), .partial_bf16 = 1, .nq = if (i == 0) 1 else spec.attn_queries, .rows = @intCast(n), .c = @intCast(h), .eps = t.eps };
        for (0..spec.blocks_in) |j| mix.blocks[j] = m.s.blocks + j * plane * 2;
        mix.norm = a.norms;
        try ops.mix(&k.ops, s, mix);
    }
    const separate = global or msgs;
    const xq = m.s.xin;
    const xk = if (separate) m.s.xin + plane * 2 else xq;
    const xv = if (separate) m.s.xin + 2 * plane * 2 else xq;
    // one row: the fused product (q, gate, rel, sink and k, v when they share the input), its parts read in place
    var o: Out = .{ .q = m.s.q, .k = m.s.kraw, .v = m.s.vraw, .rel = m.s.rel, .gate = m.s.gate, .sink = m.s.sink };
    if (n == 1) {
        const f = wts.Fused.of(t);
        const at = struct {
            fn col(base: u64, c: usize) u64 {
                return base + c * 2;
            }
        }.col;
        const fz = m.s.fused;
        o = .{ .q = at(fz, f.q), .k = at(fz, f.k), .v = at(fz, f.v), .rel = at(fz, f.rel), .gate = at(fz, f.gate), .sink = at(fz, f.sink) };
        try proj(m, k, s, if (separate) a.qx else a.all, xq, fz, 1);
        if (separate) {
            try proj(m, k, s, a.wk, xk, o.k, 1);
            try proj(m, k, s, a.wv, xv, o.v, 1);
        }
    } else {
        try proj(m, k, s, a.wq, xq, o.q, n);
        try proj(m, k, s, a.wk, xk, o.k, n);
        try proj(m, k, s, a.wv, xv, o.v, n);
        try proj(m, k, s, a.wr, xq, o.rel, n);
        try proj(m, k, s, a.gate, xq, o.gate, n);
        try proj(m, k, s, a.sink, xq, o.sink, n);
    }
    const kvw = kvWidth(m);
    const kvh: usize = t.local.kv;
    const taps = kvh * config.conv_taps;
    if (global) for ([_]u64{ xk, xv }, [_]u64{ a.k_taps, a.v_taps }, 0..) |x, w, side| {
        // the taps as torch's bf16 linear leaves them: float(bf16(sum)) times 1
        try k.dense.rowsScaled(s, x, h, w, m.w.ones, m.s.dyn[side], taps, n, taps, h);
    };
    for ([_]Site{ .k, .v }, [_]u64{ o.k, o.v }, [_]u64{ m.s.kc, m.s.vc }, [_]u64{ a.k_conv, a.v_conv }, 0..) |site, x, out, w, side| {
        try conv(m, k, s, r, v, site, .{ .x = x, .state = 0, .w = w, .dyn = if (global) m.s.dyn[side] else 0, .out = out, .x_bf16 = 1, .heads = @intCast(kvh), .c = @intCast(kvw), .rows = 0, .mode = 0 });
    }
    // q, rel and the caches' new K and V rows, the bias table, then attention through the sink and the output gate
    const l = if (global) m.full else m.sliding;
    const rows: attn.Rows = .{ .rows = @intCast(n), .heads = t.local.q, .kv_heads = t.local.kv, .pos = m.s.pos, .seg_of = m.s.seg_of, .segs = m.s.segs[@intFromBool(global)], .tiles = m.s.tiles, .n_tiles = r.n_tiles, .last_pos = r.last_pos, .streams = @intCast(r.streams) };
    const cache: attn.Cache = .{ .k = m.kv[0][v], .v = m.kv[1][v] };
    const prep: attn.Prep = .{ .qk_clip = t.qk_clip, .rel_clip = t.rel_clip, .floor = @floatFromInt(t.log_floor), .alpha = t.log_alpha, .eps = t.eps };
    try attn.prep(&k.attn, s, rows, l, prep, .{ o.q, m.s.kc, m.s.vc, o.rel }, .{ a.q_scale, a.rel_scale }, m.s.qp, m.s.relp, cache);
    try attn.relTable(&k.attn, s, rows, m.s.relp, a.rel_proj, m.s.tab);
    const stride = attn.slotsFor(l, m.context);
    try attn.attend(&k.attn, s, rows, l, stride, m.s.entries, m.s.qp, cache, m.s.tab, m.s.part_o, m.s.part_ml, o.sink, o.gate, m.s.att);
    try proj(m, k, s, a.wo, m.s.att, m.s.o, n);
    // the attention conv into the fp32 residual: a block boundary keeps only the attention output as the partial
    try conv(m, k, s, r, v, .attn, .{ .x = m.s.o, .state = 0, .w = L.attn_conv, .base = if (spec.boundary) 0 else hin, .resid = m.s.h32, .x_bf16 = 1, .c = @intCast(h), .rows = 0, .mode = 1 });
    if (spec.boundary and !p.restarts(v)) try m.copy(s, m.s.blocks + @as(usize, i / t.block) * plane * 2, hin, plane * 2);
    const nb = spec.blocks_in + @intFromBool(spec.boundary);
    if (msgs) {
        const mn = [3]u64{ L.mlp_norm, L.mlp_norm, L.mlp_norm };
        try gated(m, k, s, r, nb, m.s.h32, true, L.mlp_query, 1, L.controllers.?[3..4], .{ 3, 3, 3 }, mn, m.s.xm);
    } else {
        var mix: ops.MixArgs = .{ .partial = m.s.h32, .query = L.mlp_query, .out = m.s.xm, .n_blocks = @intCast(nb), .partial_bf16 = 0, .nq = 1, .rows = @intCast(n), .c = @intCast(h), .eps = t.eps };
        for (0..nb) |j| mix.blocks[j] = m.s.blocks + j * plane * 2;
        mix.norm[0] = L.mlp_norm;
        try ops.mix(&k.ops, s, mix);
    }
    try moe(m, k, s, r, L);
    // the MLP conv: the next visit's state is bf16(h32 + bf16(conv(moe)))
    try conv(m, k, s, r, v, .mlp, .{ .x = m.s.moe, .state = 0, .w = L.mlp_conv, .resid = m.s.h32, .out = hout, .x_bf16 = 1, .c = @intCast(h), .rows = 0, .mode = 2 });
}

/// The router's picks over sigmoid scores, the routed experts as stored (core experts), their weighted sum in bf16.
fn moe(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const Rows, L: *const wts.Layer) !void {
    const t = m.c;
    const h: usize = t.hidden;
    const n = r.n;
    const pairs = n * t.topk;
    try ops.rowdot(&k.ops, s, m.s.xm, L.router, m.s.scores, n, t.experts, h, false, true);
    try ops.topk(&k.ops, s, m.s.scores, L.router_bias, m.s.pick, m.s.rw, n, t.experts, t.topk, t.route_scale);
    const tile: usize = if (r.prompt) cuda.experts.promptTile(L.experts.format) else 16;
    const items = cuda.grouped.maxItems(pairs, t.experts, tile);
    try k.router.route(s, m.s.pick, pairs, t.experts, tile, m.s.plan);
    const gx = k.experts();
    const up_in: cuda.experts.Rows = .{ .x = m.s.xm, .stride = h, .slots = t.topk };
    const down_in: cuda.experts.Rows = .{ .x = m.s.act, .stride = t.expert_inter };
    if (r.prompt) {
        try gx.prompt(s, .up, up_in, L.experts, m.s.plan, m.s.act, items, pairs, tile, -1);
        try gx.prompt(s, .down_bf16, down_in, L.experts, m.s.plan, m.s.y, items, pairs, tile, -1);
        try k.glue.combine(s, m.s.y, .bf16, m.s.rw, m.s.moe, .bf16, n, t.topk, h);
    } else {
        try gx.decode(s, .up, up_in, L.experts, m.s.plan, m.s.act, items, pairs, tile, -1);
        try gx.decode(s, .down_f32, down_in, L.experts, m.s.plan, m.s.y, items, pairs, tile, -1);
        try k.glue.combine(s, m.s.y, .f32, m.s.rw, m.s.moe, .bf16, n, t.topk, h);
    }
}
