//! GLM-5.3-Flash on CUDA: the weights' views and the layers of a step (forward.py's blocks): KDA and DSA attention,
//! the dense MLP and the MoE, the hyper-connections around them, the final norm and the head; and the MTP head.
//! Row-wise work covers the step's rows at once; the KDA recurrences, the attention and the indexer read the step's
//! tables (forward.prepare).

const std = @import("std");
const cuda = @import("cuda");
const api = @import("api.zig");
const dflash = @import("dflash.zig");
const kern = @import("kern.zig");
const sq = @import("seq.zig");
const forward = @import("forward.zig");

const Engine = forward.Engine;
const Buffers = forward.Buffers;
const D = kern.D;
const HL = kern.HL;
const LL = kern.LL;
const PROJ_W = sq.PROJ_W;
const FA_OFF: i64 = 3 * LL * 128;
const GA_OFF: i64 = FA_OFF + 128;
const B_OFF: i64 = GA_OFF + 128;
const MOE_W = forward.MOE_W;
const DENSE_W = forward.DENSE_W;
const IDX_KW = forward.IDX_KW;

// -- weights ------------------------------------------------------------------------------------------------------------
const HCW = struct { fn_: u64, base: u64, scale: u64 };
const KdaW = struct { proj: kern.B16, fb: kern.B16, gb: kern.B16, conv: u64, a_log: u64, dt_bias: u64, norm: u64, o: kern.B16 };
const IndexW = struct { kw: kern.B16, qb: kern.B16, ln_w: u64, ln_b: u64, gate: u64, ape: u64 };
const DsaW = struct { proj: kern.B16, q_norm: u64, kv_norm: u64, q_b: kern.B16, wk: u64, wv: u64, o: kern.B16, index: IndexW };
const MlpW = struct { gu: kern.B16, down: kern.B16 };
const MoeW = struct { router: u64, bias: u64, up: u64 = 0, down: u64 = 0, exl3: ?kern.Exl3 = null, shared: MlpW };
const LayerW = struct {
    attn_hc: HCW = undefined,
    ffn_hc: HCW = undefined,
    in_norm: u64,
    post_norm: u64,
    kda: ?KdaW = null,
    dsa: ?DsaW = null,
    mlp: ?MlpW = null,
    moe: ?MoeW = null,
    li: usize = 0, // its KDA or DSA index
};
const MtpW = struct { enorm: u64, hnorm: u64, eh: kern.B16, norm: u64, layer: LayerW };

fn ptr(store: *const api.Store, comptime fmt: []const u8, args: anytype) !u64 {
    return (try store.getf(fmt, args)).ptr;
}

fn b16(store: *const api.Store, comptime fmt: []const u8, args: anytype) !kern.B16 {
    var buf: [192]u8 = undefined;
    const pre = try std.fmt.bufPrint(&buf, fmt, args);
    var nb: [224]u8 = undefined;
    if (store.map.get(try std.fmt.bufPrint(&nb, "{s}.qw", .{pre}))) |q| { // packed by the loader (weights.quantize)
        var nb2: [224]u8 = undefined;
        const sc = try store.get(try std.fmt.bufPrint(&nb2, "{s}.qs", .{pre}));
        const kk = sc.shape[1] * 64;
        return .{ .w = q.ptr, .n = q.shape[0], .k = kk, .s = sc.ptr, .bits = @intCast(@divExact(q.shape[1] * 8, kk)) };
    }
    const t = try store.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{pre}));
    return .{ .w = t.ptr, .n = t.shape[0], .k = t.shape[1] };
}

fn hcw(store: *const api.Store, p: []const u8, site: []const u8) !HCW {
    return .{ .fn_ = try ptr(store, "{s}.{s}.fn", .{ p, site }), .base = try ptr(store, "{s}.{s}.base", .{ p, site }), .scale = try ptr(store, "{s}.{s}.scale", .{ p, site }) };
}

fn mlpW(store: *const api.Store, p: []const u8) !MlpW {
    return .{ .gu = try b16(store, "{s}.gu", .{p}), .down = try b16(store, "{s}.down", .{p}) };
}

fn dsaW(store: *const api.Store, p: []const u8) !DsaW {
    return .{ .proj = try b16(store, "{s}.dsa.proj", .{p}), .q_norm = try ptr(store, "{s}.dsa.q_norm", .{p}), .kv_norm = try ptr(store, "{s}.dsa.kv_norm", .{p}),
        .q_b = try b16(store, "{s}.dsa.q_b", .{p}), .wk = try ptr(store, "{s}.dsa.absorb.wk", .{p}), .wv = try ptr(store, "{s}.dsa.absorb.wv", .{p}),
        .o = try b16(store, "{s}.dsa.o", .{p}), .index = .{ .kw = try b16(store, "{s}.dsa.index.kw", .{p}), .qb = try b16(store, "{s}.dsa.index.qb", .{p}),
            .ln_w = try ptr(store, "{s}.dsa.index.ln_w", .{p}), .ln_b = try ptr(store, "{s}.dsa.index.ln_b", .{p}), .gate = try ptr(store, "{s}.dsa.index.gate", .{p}),
            .ape = try ptr(store, "{s}.dsa.index.ape", .{p}) } };
}

fn moeW(store: *const api.Store, p: []const u8) !MoeW {
    var b: [160]u8 = undefined;
    var m: MoeW = .{ .router = try ptr(store, "{s}.moe.router", .{p}), .bias = try ptr(store, "{s}.moe.bias", .{p}),
        .shared = try mlpW(store, try std.fmt.bufPrint(&b, "{s}.moe.shared", .{p})) };
    if ((store.ints.get("exl3") orelse 0) != 0) {
        m.exl3 = .{ .gt = try ptr(store, "{s}.moe.experts.gt", .{p}), .ut = try ptr(store, "{s}.moe.experts.ut", .{p}), .dt = try ptr(store, "{s}.moe.experts.dt", .{p}),
            .suh_g = try ptr(store, "{s}.moe.experts.suh_g", .{p}), .suh_u = try ptr(store, "{s}.moe.experts.suh_u", .{p}),
            .svh_g = try ptr(store, "{s}.moe.experts.svh_g", .{p}), .svh_u = try ptr(store, "{s}.moe.experts.svh_u", .{p}),
            .suh_d = try ptr(store, "{s}.moe.experts.suh_d", .{p}), .svh_d = try ptr(store, "{s}.moe.experts.svh_d", .{p}) };
    } else {
        m.up = try ptr(store, "{s}.moe.experts.up", .{p});
        m.down = try ptr(store, "{s}.moe.experts.down", .{p});
    }
    return m;
}

fn layerW(store: *const api.Store, p: []const u8, kda: bool, dense: bool, plain: bool, li: usize) !LayerW {
    var l: LayerW = .{ .in_norm = try ptr(store, "{s}.in_norm", .{p}), .post_norm = try ptr(store, "{s}.post_norm", .{p}), .li = li };
    if (!plain) {
        l.attn_hc = try hcw(store, p, "attn_hc");
        l.ffn_hc = try hcw(store, p, "ffn_hc");
    }
    if (kda) {
        l.kda = .{ .proj = try b16(store, "{s}.kda.proj", .{p}), .fb = try b16(store, "{s}.kda.fb", .{p}), .gb = try b16(store, "{s}.kda.gb", .{p}),
            .conv = try ptr(store, "{s}.kda.conv", .{p}), .a_log = try ptr(store, "{s}.kda.a_log", .{p}), .dt_bias = try ptr(store, "{s}.kda.dt_bias", .{p}),
            .norm = try ptr(store, "{s}.kda.norm", .{p}), .o = try b16(store, "{s}.kda.o", .{p}) };
    } else l.dsa = try dsaW(store, p);
    var b: [160]u8 = undefined;
    if (dense) l.mlp = try mlpW(store, try std.fmt.bufPrint(&b, "{s}.mlp", .{p})) else l.moe = try moeW(store, p);
    return l;
}

pub const Weights = struct {
    layers: [46]LayerW = undefined,
    nlayers: usize = 0,
    kinds: [46]bool = @splat(false),
    nkda: usize = 0,
    ndsa: usize = 0,
    mtp: ?MtpW = null,
    embed: u64 = 0,
    norm: u64 = 0,
    head: kern.B16 = undefined,
    draft_head: kern.B16 = undefined,
    vocab_offset: i64 = 0,
    experts_sym: bool = false,
    experts3: bool = false,
    exl3: bool = false, // EXL3 routed experts (glm/exl3.cu)
    exl3_shx: bool = false, // every layer's experts share one input rotation for gate and up (one rotated row a token in prompts)

    pub fn caches(w: *const Weights) usize {
        return w.ndsa + @intFromBool(w.mtp != null);
    }

    /// The store's views, and every dense linear tiled in place for glm/mm.cu (once, at load).
    pub fn load(k: *kern.K, store: *const api.Store) !Weights {
        var w: Weights = .{};
        w.vocab_offset = store.ints.get("vocab_offset").?;
        const kinds = store.host.get("kinds").?;
        w.nlayers = kinds.len;
        const dense: usize = @intCast(store.ints.get("dense").?);
        for (kinds, 0..) |kv, i| {
            w.kinds[i] = kv != 0;
            var nb: [32]u8 = undefined;
            const li = if (w.kinds[i]) w.nkda else w.ndsa;
            w.layers[i] = try layerW(store, try std.fmt.bufPrint(&nb, "layers.{d}", .{i}), w.kinds[i], i < dense, false, li);
            if (w.kinds[i]) w.nkda += 1 else w.ndsa += 1;
        }
        if (store.ints.get("mtp").? != 0) {
            w.mtp = .{ .enorm = try ptr(store, "mtp.enorm", .{}), .hnorm = try ptr(store, "mtp.hnorm", .{}), .eh = try b16(store, "mtp.eh", .{}),
                .norm = try ptr(store, "mtp.norm", .{}), .layer = try layerW(store, "mtp.layer", false, false, true, w.ndsa) };
        }
        w.embed = try ptr(store, "embed", .{});
        w.norm = try ptr(store, "norm", .{});
        w.experts_sym = (store.ints.get("experts_sym") orelse 0) != 0;
        w.experts3 = (store.ints.get("experts3") orelse 0) != 0;
        w.exl3 = (store.ints.get("exl3") orelse 0) != 0;
        w.exl3_shx = (store.ints.get("exl3_shx") orelse 0) != 0;
        w.head = try b16(store, "head", .{});
        // drafts only propose (verification reads ``head``): a smaller copy when the loader made one
        w.draft_head = if (store.map.contains("draft_head.qw")) try b16(store, "draft_head", .{}) else w.head;
        try w.tile(k);
        return w;
    }

    pub fn deinit(_: *Weights, _: std.mem.Allocator) void {}

    fn linears(w: *const Weights, out: *std.ArrayList(kern.B16), gpa: std.mem.Allocator) !void {
        const add = struct {
            fn f(o: *std.ArrayList(kern.B16), a: std.mem.Allocator, x: kern.B16) !void {
                for (o.items) |y| if (y.w == x.w) return;
                try o.append(a, x);
            }
        }.f;
        const layer = struct {
            fn f(o: *std.ArrayList(kern.B16), a: std.mem.Allocator, l: LayerW) !void {
                if (l.kda) |x| for ([_]kern.B16{ x.proj, x.fb, x.gb, x.o }) |m| try add(o, a, m);
                if (l.dsa) |x| for ([_]kern.B16{ x.proj, x.q_b, x.o, x.index.kw, x.index.qb }) |m| try add(o, a, m);
                if (l.mlp) |x| for ([_]kern.B16{ x.gu, x.down }) |m| try add(o, a, m);
                if (l.moe) |x| for ([_]kern.B16{ x.shared.gu, x.shared.down }) |m| try add(o, a, m);
            }
        }.f;
        for (w.layers[0..w.nlayers]) |l| try layer(out, gpa, l);
        if (w.mtp) |m| {
            try add(out, gpa, m.eh);
            try layer(out, gpa, m.layer);
        }
        try add(out, gpa, w.head);
        try add(out, gpa, w.draft_head);
    }

    fn tile(w: *const Weights, k: *kern.K) !void {
        var all: std.ArrayList(kern.B16) = .empty;
        defer all.deinit(k.gpa);
        try w.linears(&all, k.gpa);
        var most: u64 = 0;
        for (all.items) |x| most = @max(most, wbytes(x));
        var tmp = try cuda.DeviceBuffer.alloc(k.d, most + (most / 32) + 512);
        defer tmp.free();
        for (all.items) |x| {
            if (@mod(x.n, 8) != 0 or @mod(x.k, 64) != 0) return error.LinearNotTileable;
            const wb = wbytes(x);
            try k.copy(tmp.ptr, x.w, wb);
            const sb: u64 = if (x.bits != 0) @intCast(@divExact(x.n * x.k, 64) * 2) else 0;
            const st = tmp.ptr + std.mem.alignForward(u64, wb, 256);
            if (sb > 0) try k.copy(st, x.s, sb);
            try kern.mmTile(k, x.bits, tmp.ptr, st, x.w, x.s, x.n, x.k);
            try k.stream.synchronize();
        }
    }

    fn wbytes(x: kern.B16) u64 {
        const bits: i64 = if (x.bits == 0) 16 else x.bits;
        return @intCast(@divExact(x.n * x.k * bits, 8));
    }
};

// -- the blocks -----------------------------------------------------------------------------------------------------------
pub fn mm(e: *Engine, b: *Buffers, x: u64, x_stride: i64, w: kern.B16, out: u64, f32_out: bool, R: i64) !void {
    if (e.skip.dense and w.n != forward.HEAD_N) return;
    try kern.mm(&e.k, x, x_stride, w, out, f32_out, R, b.sk);
}

fn outProj(e: *Engine, b: *Buffers, x: u64, x_stride: i64, w: kern.B16, R: i64) !u64 {
    try mm(e, b, x, x_stride, w, b.part, true, R);
    return forward.gather(e, b, R);
}

/// A KDA layer: a prompt chunk's rows through its one stream's states (a kept point inside the chunk cuts the
/// chain there), or a decode step's windows, each through its own stream's (the segmented chain).
fn kdaBlock(e: *Engine, b: *Buffers, l: LayerW, R: i64) !u64 {
    const k = l.kda.?;
    const li = l.li;
    const p = b.kproj;
    try mm(e, b, b.normed, D, k.proj, p, false, R);
    try mm(e, b, p + FA_OFF * 2, PROJ_W, k.fb, b.ka, false, R);
    try mm(e, b, p + GA_OFF * 2, PROJ_W, k.gb, b.kg, false, R);
    if (!b.prefill) {
        try kern.kdaChainSeg(&e.k, b.segs, @intCast(e.step.segs.len), li, p, PROJ_W, B_OFF, b.ka, LL * 128, b.kg, LL * 128, k.conv, k.a_log, k.dt_bias,
            k.norm, b.kout, sq.MAX_ROWS, PROJ_W);
        return outProj(e, b, b.kout, LL * 128, k.o, R);
    }
    const s = e.step.segs[0].s;
    const cur = s.cur;
    const conv = s.convAt(li);
    const sc = b.ks;
    const n = e.cut_at;
    if (n > 0 and n < R) { // a kept point inside the chunk: the state and window at it, then the rest from them
        const cut_rec = e.cut_rec + @as(u64, @intCast(li)) * @as(u64, @intCast(sq.REC * 4));
        const cut_conv = e.cut_conv + @as(u64, @intCast(li)) * @as(u64, @intCast(3 * sq.CONV_C * 2));
        try kern.kdaChain(&e.k, p, PROJ_W, B_OFF, b.ka, LL * 128, b.kg, LL * 128, conv, k.conv, s.recAt(cur, li), k.a_log, k.dt_bias, k.norm, n, sc,
            cut_rec, b.q_tmp, b.y_tmp);
        try kern.convShift(&e.k, cut_conv, b.kproj, n, 3 * sq.CONV_C, b.rows * PROJ_W, PROJ_W, 1, sq.CONV_C);
        const hw: u64 = @intCast(LL * 128);
        const nr: u64 = @intCast(n);
        const rest: kern.KdaScratch = .{ .out = sc.out + nr * hw * 2, .k = sc.k + nr * hw * 4, .v = sc.v + nr * hw * 2, .g = sc.g + nr * hw * 4, .b = sc.b + nr * @as(u64, @intCast(LL)) * 4 };
        try kern.kdaChain(&e.k, p + nr * @as(u64, @intCast(PROJ_W)) * 2, PROJ_W, B_OFF, b.ka + nr * hw * 2, LL * 128, b.kg + nr * hw * 2, LL * 128, cut_conv, k.conv, cut_rec,
            k.a_log, k.dt_bias, k.norm, R - n, rest, s.recAt(1 - cur, li), b.q_tmp, b.y_tmp);
    } else {
        try kern.kdaChain(&e.k, p, PROJ_W, B_OFF, b.ka, LL * 128, b.kg, LL * 128, conv, k.conv, s.recAt(cur, li), k.a_log, k.dt_bias, k.norm, R, sc,
            s.recAt(1 - cur, li), b.q_tmp, b.y_tmp);
    }
    // a prompt chunk keeps every row: the layer's conv window commits now (its state flips with the chunk)
    try kern.convShift(&e.k, conv, b.kproj, R, 3 * sq.CONV_C, b.rows * PROJ_W, PROJ_W, 1, sq.CONV_C);
    return outProj(e, b, sc.out, LL * 128, k.o, R);
}

/// Each row's attended tokens (ascending, -1 padded) and their count past the dense limit (0: a dense row).
fn selectTokens(e: *Engine, b: *Buffers, ci: usize, R: i64) !void {
    const t = b.rowsTab();
    const B = @min(R, kern.SELECT_ROWS);
    var a: i64 = 0;
    while (a < R) : (a += B) {
        const n = @min(B, R - a);
        try kern.scores(&e.k, b.qi + @as(u64, @intCast(a * kern.IDX_HEADS * kern.IDX_DIM * 2)), b.ikr + @as(u64, @intCast(a * IDX_KW + kern.IDX_DIM)) * 2, IDX_KW,
            t.at(a), ci, b.scores, n, e.step.np);
        try kern.selectRows(&e.k, b.scores, b.pools + @as(u64, @intCast(a * kern.TOPK_POOLS * 8)), e.step.np, t.at(a), n);
    }
    try kern.sparseTokens(&e.k, b.pools, t, R, b.tokens, b.counts);
}

/// A DSA layer on latent cache ``ci`` (the MTP layer's is the last): the indexer always (its keys and pools are kept
/// for every position), dense attention for the rows below the dense limit, sparse for the rest.
fn dsaBlock(e: *Engine, b: *Buffers, a: DsaW, ci: usize, R: i64) !u64 {
    const t = b.rowsTab();
    try mm(e, b, b.normed, D, a.proj, b.dp, false, R);
    try kern.rmsnorm(&e.k, b.dp, kern.Q_LORA + kern.KV_LORA, a.q_norm, b.qr, kern.Q_LORA, b.xs_qr, R, kern.Q_LORA);
    try kern.rmsnorm(&e.k, b.dp + kern.Q_LORA * 2, kern.Q_LORA + kern.KV_LORA, a.kv_norm, b.lat, kern.KV_LORA, b.xs_lat, R, kern.KV_LORA);
    try mm(e, b, b.qr, kern.Q_LORA, a.q_b, b.q, false, R);
    try kern.latentWrite(&e.k, b.lat, kern.KV_LORA, t, ci, R);
    try mm(e, b, b.normed, D, a.index.kw, b.ikr, false, R);
    try kern.router(&e.k, b.normed, D, a.index.gate, b.igr, R, kern.IDX_DIM);
    try kern.indexUpdate(&e.k, b.ikr, IDX_KW, b.igr, a.index.ln_w, a.index.ln_b, a.index.ape, t, ci, R);
    if (e.skip.attn) return outProj(e, b, b.vn, HL * kern.DV, a.o, R);
    try kern.absorbQ(&e.k, b.q, a.wk, b.qa, R);
    const att = b.attRows();
    const lat_row: u64 = @intCast(HL * kern.KV_LORA * 2);
    if (e.step.nch > 0) {
        var r0: i64 = 0;
        while (r0 < R) : (r0 += att) {
            const n = @min(att, R - r0);
            const off = @as(u64, @intCast(r0)) * lat_row;
            try kern.denseAttention(&e.k, b.qa + off, t.at(r0), ci, b.po, b.pm, b.pl, b.ol + off, n, e.step.nch);
        }
    }
    if (e.step.sparse) {
        try mm(e, b, b.qr, kern.Q_LORA, a.index.qb, b.qi, false, R);
        try selectTokens(e, b, ci, R);
        var r0: i64 = 0;
        while (r0 < R) : (r0 += att) {
            const n = @min(att, R - r0);
            const off = @as(u64, @intCast(r0)) * lat_row;
            try kern.sparseAttention(&e.k, b.qa + off, t.at(r0), ci, b.tokens + @as(u64, @intCast(r0 * kern.SEL_W * 4)), b.counts + @as(u64, @intCast(r0 * 4)),
                b.po, b.pm, b.pl, b.ol + off, n);
        }
    }
    try kern.expandV(&e.k, b.ol, a.wv, b.vn, R);
    return outProj(e, b, b.vn, HL * kern.DV, a.o, R);
}

fn mlpBlock(e: *Engine, b: *Buffers, m: MlpW, R: i64) !u64 {
    try mm(e, b, b.normed, D, m.gu, b.gu, false, R);
    try kern.swiglu(&e.k, b.gu, b.act, b.xs_act, R, DENSE_W);
    return outProj(e, b, b.act, DENSE_W, m.down, R);
}

/// The MoE: the routed slots through the grouped kernels (8 a row), the shared expert (slot 8) through the dense
/// linears, the slots combined by their weights.
fn moeBlock(e: *Engine, b: *Buffers, m: MoeW, R: i64) !u64 {
    try kern.router(&e.k, b.normed, D, m.router, b.mlog, R, kern.EXPERTS);
    try kern.topk(&e.k, b.mlog, m.bias, b.pick, b.wts, R);
    try kern.copyRows(&e.k, b.rpick, @intCast(kern.TOPK * 4), b.pick, @intCast(kern.SLOTS * 4), @intCast(kern.TOPK * 4), @intCast(R));
    const pairs = R * kern.TOPK;
    try kern.expertsPlan(&e.k, b.rpick, pairs, b.plan);
    const items = kern.maxItems(pairs, kern.EXPERTS, b.plan.tile);
    const gs: i64 = 64;
    const nbu = @divExact(MOE_W, 32);
    const nbd = @divExact(D, 32);
    if (m.exl3) |x| { // the routed slots straight into ey (fp32, rows of 9 slots), then the shared expert's slot
        if (!e.skip.experts) try kern.exl3Routed(&e.k, b.normed, D, b.rpick, b.plan, x, b.x3, b.ey, kern.SLOTS, R, MOE_W, b.prefill, e.w.exl3_shx);
        const dsz: usize = @intCast(D);
        try mm(e, b, b.normed, D, m.shared.gu, b.sgu, false, R);
        try kern.swiglu(&e.k, b.sgu, b.sact, b.sxs, R, MOE_W);
        try mm(e, b, b.sact, MOE_W, m.shared.down, b.sy, true, R);
        try kern.copyRows(&e.k, b.ey + @as(u64, @intCast(kern.TOPK * D * 4)), @as(usize, @intCast(kern.SLOTS)) * dsz * 4, b.sy, dsz * 4, dsz * 4, @intCast(R));
        try kern.combine(&e.k, b.ey, true, b.wts, b.part, R);
        return forward.gather(e, b, R);
    }
    if (b.prefill) {
        try kern.expertsCall(&e.k, true, 2, b.normed, D, kern.TOPK, m.up, @divExact(D, gs), nbu, b.plan, b.eact, MOE_W, items, kern.LIMIT);
        try kern.expertsCall(&e.k, true, 3, b.eact, MOE_W, 0, m.down, @divExact(MOE_W, gs), nbd, b.plan, b.rey, D, items, 0.0);
    } else {
        try kern.expertsCall(&e.k, false, 2, b.normed, D, kern.TOPK, m.up, @divExact(D, gs), nbu, b.plan, b.eact, MOE_W, items * nbu, kern.LIMIT);
        try kern.expertsCall(&e.k, false, 0, b.eact, MOE_W, 0, m.down, @divExact(MOE_W, gs), nbd, b.plan, b.rey, D, items * nbd, 0.0);
    }
    const elt: usize = if (b.prefill) 2 else 4;
    const dsz: usize = @intCast(D);
    const slots: usize = @intCast(kern.SLOTS);
    const topk: usize = @intCast(kern.TOPK);
    try kern.copyRows(&e.k, b.ey, slots * dsz * elt, b.rey, topk * dsz * elt, topk * dsz * elt, @intCast(R));
    try mm(e, b, b.normed, D, m.shared.gu, b.sgu, false, R);
    try kern.swiglu(&e.k, b.sgu, b.sact, b.sxs, R, MOE_W);
    try mm(e, b, b.sact, MOE_W, m.shared.down, b.sy, true, R);
    const slot8: u64 = @intCast(kern.TOPK * D * @as(i64, @intCast(elt)));
    if (b.prefill) {
        try kern.castBf16(&e.k, b.sy, D, b.ey + slot8, kern.SLOTS * D, R, D);
    } else {
        try kern.copyRows(&e.k, b.ey + slot8, slots * dsz * 4, b.sy, dsz * 4, dsz * 4, @intCast(R));
    }
    try kern.combine(&e.k, b.ey, !b.prefill, b.wts, b.part, R);
    return forward.gather(e, b, R);
}

fn layerForward(e: *Engine, b: *Buffers, l: LayerW, R: i64) !void {
    const h = l.attn_hc;
    if (e.skip.hc) {
        const g = if (l.kda != null) try kdaBlock(e, b, l, R) else try dsaBlock(e, b, l.dsa.?, l.li, R);
        _ = g;
        _ = if (l.mlp) |m| try mlpBlock(e, b, m, R) else try moeBlock(e, b, l.moe.?, R);
        return;
    }
    try kern.hcPre(&e.k, b.x, h.fn_, h.base, h.scale, l.in_norm, b.normed, b.xs, b.post, b.comb, b.hcpart, R);
    const g = if (l.kda != null) try kdaBlock(e, b, l, R) else try dsaBlock(e, b, l.dsa.?, l.li, R);
    try kern.hcPost(&e.k, b.x, g, b.post, b.comb, R);
    const f = l.ffn_hc;
    try kern.hcPre(&e.k, b.x, f.fn_, f.base, f.scale, l.post_norm, b.normed, b.xs, b.post, b.comb, b.hcpart, R);
    const g2 = if (l.mlp) |m| try mlpBlock(e, b, m, R) else try moeBlock(e, b, l.moe.?, R);
    try kern.hcPost(&e.k, b.x, g2, b.post, b.comb, R);
}

/// ``x``'s rows (positions first .. first + R, ``copies`` streams each) that fall in an image's run take its features.
fn injectMedia(e: *Engine, s: *const sq.Seq, first: i64, R: i64, x: u64, copies: i64) !void {
    const m = s.media orelse return;
    var row0: i64 = 0; // the run's first feature row
    for (m.spans) |sp| {
        const lo = @max(sp[0], first);
        const hi = @min(sp[1], first + R);
        if (lo < hi) {
            const f = m.feats + @as(u64, @intCast((row0 + lo - sp[0]) * D * 2));
            try kern.injectRows(&e.k, f, x + @as(u64, @intCast((lo - first) * copies * D * 2)), hi - lo, copies);
        }
        row0 += sp[1] - sp[0];
    }
}

/// The step's rows through every layer, the final norm, and the head (a prompt chunk: its last row only). The ids
/// are staged in b.ids, the tables prepared.
pub fn compute(e: *Engine, b: *Buffers) !void {
    const R = e.step.rows;
    const w = &e.w;
    try kern.embed(&e.k, b.ids, w.embed, b.x, R, kern.S);
    if (b.prefill) try injectMedia(e, e.step.segs[0].s, e.step.first_pos, R, b.x, kern.S);
    // DFlash2's taps: a verify step's rows and a prompt chunk's (MTP and the drafter's own runs take none)
    const taps: u64 = if (e.drafter) |dr| (if (b.prefill) dr.taps_prefill else if (b == &e.buf) dr.taps_step else 0) else 0;
    for (w.layers[0..w.nlayers], 0..) |l, li| {
        try layerForward(e, b, l, R);
        if (taps != 0 and dflash.isTap(e.drafter.?, li)) try dflash.tap(e.drafter.?, &e.k, b.x, taps, li, R);
    }
    try kern.streamMean(&e.k, b.x, b.hidden, R);
    try kern.rmsnorm(&e.k, b.hidden, D, w.norm, b.fnormed, D, b.fxs, R, D);
    if (b.prefill) {
        try mm(e, b, b.fnormed + @as(u64, @intCast((R - 1) * D * 2)), D, w.head, b.logits, false, 1);
    } else {
        try mm(e, b, b.fnormed, D, w.head, b.logits, false, R);
    }
}

/// The MTP head over the step's staged rows (ids in b.ids, input hidden rows in b.hin), then each stream's last row's
/// logits (in segment order).
pub fn mtpCompute(e: *Engine, b: *Buffers) !void {
    const m = e.w.mtp.?;
    const R = e.step.rows;
    const t = b.rowsTab();
    try kern.embed(&e.k, b.ids, e.w.embed, b.me, R, 1);
    if (b.prefill) try injectMedia(e, e.step.segs[0].s, e.step.first_pos + 1, R, b.me, 1); // MTP row i reads the token at mtp_len + 1 + i
    try zeroFirst(e, b.me, t.posAt(), R);
    try kern.rmsnorm(&e.k, b.me, D, m.enorm, b.mcat, 2 * D, null, R, D);
    try kern.rmsnorm(&e.k, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, R, D);
    try mm(e, b, b.mcat, 2 * D, m.eh, b.mx, false, R);
    const l = m.layer;
    try kern.rmsnorm(&e.k, b.mx, D, l.in_norm, b.normed, D, b.xs, R, D);
    const g = try dsaBlock(e, b, l.dsa.?, e.w.ndsa, R);
    try kern.residualAdd(&e.k, b.mx, g, R);
    try kern.rmsnorm(&e.k, b.mx, D, l.post_norm, b.normed, D, b.xs, R, D);
    const g2 = try moeBlock(e, b, l.moe.?, R);
    try kern.residualAdd(&e.k, b.mx, g2, R);
    const nseg: i64 = @intCast(e.step.segs.len);
    try gatherRows(e, b.mx, b.last, b.hrows, nseg);
    try kern.rmsnorm(&e.k, b.hrows, D, m.norm, b.fnormed, D, b.fxs, nseg, D);
    try mm(e, b, b.fnormed, D, e.w.draft_head, b.logits, false, nseg);
}

fn zeroFirst(e: *Engine, x: u64, pos: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(x); a.add(pos); a.add(@as(i32, @intCast(D)));
    try e.k.go(try e.k.ext(.rows, "glm_zero_first"), .{ .x = @intCast(R) }, 256, 0, &a);
}

fn gatherRows(e: *Engine, src: u64, idx: u64, dst: u64, n: i64) !void {
    var a: cuda.Args = .{};
    a.add(src); a.add(idx); a.add(dst); a.add(@as(i32, @intCast(D)));
    try e.k.go(try e.k.ext(.rows, "glm_gather_rows"), .{ .x = @intCast(n) }, 256, 0, &a);
}
