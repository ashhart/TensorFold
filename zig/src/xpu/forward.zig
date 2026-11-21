//! Qwen3.8-27B forward on the XPU across TP={1,2,4}: activations, prompt GEMMs (fp16, W4A8, W4A4) and the layer loop.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const loader = @import("load.zig");
const D = q.D;
const Dev = dev.Dev;
const EPS = q.EPS;
const LAYERS = q.LAYERS;
const MASK_WORDS = q.MASK_WORDS;
const Model = loader.Model;
const Proj = loader.Proj;
const SPLIT = q.SPLIT;
const limits = @import("limits.zig");
const cand = @import("cand.zig");
const tail = @import("prompt_tail.zig");
const TAP_LAYERS = q.TAP_LAYERS;
const VOCAB = q.VOCAB;
const alloc = q.alloc;
extern "c" fn close(fd: c_int) c_int;
const copy = dev.copy;
const off = dev.off;

pub const Acts = struct {
    x: *anyopaque, // fp32 [maxRows, D]
    h: *anyopaque, // fp16 [maxRows, max(D, ff, od)]
    ahi: *anyopaque, // fp16 [maxRows, max(D, ff, od) / 64]
    alo: *anyopaque,
    r: *anyopaque, // fp16 [maxRows, D]
    p: *anyopaque, // fp16 [maxRows, max(2*ff, gdn_in, att_in)]
    q: *anyopaque, // bf16 [maxRows, aq, 256]
    k: *anyopaque, // bf16 [maxRows, gk, 128]
    v: *anyopaque, // bf16 [maxRows, gv, 128]
    g: *anyopaque, // fp32 [maxRows, gv]
    beta: *anyopaque,
    y: *anyopaque, // bf16 [maxRows, od]
    la: *anyopaque,
    wb: *anyopaque,
    u0: *anyopaque,
    hs: *anyopaque,
    vn: *anyopaque,
    pm: *anyopaque,
    pl: *anyopaque,
    po: *anyopaque,
    pscr: *anyopaque, // fp32 [maxRows, D]
    logits: *anyopaque, // rank 0: fp16 [16, VOCAB] (peers: minimal stub)
    shard_logits: *anyopaque, // TP>1: fp16 [16, vocab_len]
    taps: *anyopaque, // rank 0: bf16 [maxRows, 5 D]
    tok: *anyopaque, // i32
    pv: *anyopaque,
    pi: *anyopaque,
    mrows: *anyopaque,
    mbits: *anyopaque,
    mtok: *anyopaque,
    tkv: *anyopaque, // topk_part's values f32 [16, 256, TOPK_MAX] …
    tki: *anyopaque, // … their ids i32, same shape …
    tkf: *anyopaque, // … and each group's not-finite flag i32 [16, 256]
    cand: *anyopaque, // u32 [16, CAND_WORDS]: topk_merge's rows
    ids: *anyopaque, // i32 [maxRows]
    zeros: *anyopaque,
    ones: *anyopaque,
    chain: *anyopaque,
    tree: *anyopaque,
    path: *anyopaque,
    mpath: *anyopaque,
    sl: *anyopaque, // fp32 [16, gv, 128, 128]
    mt: *anyopaque, // u64 [LAYERS, 16, 8]
    mpos: *anyopaque, // i32 [16]
    a8: *anyopaque,
    sa: *anyopaque,

    pub fn initRank(d: *Dev, rank: u32, vocab_len: u32) !Acts {
        const sh = q.tpShape();
        const rows = q.maxRows();
        const nc = limits.chunks(rows);
        const ns = limits.splits(q.MAX_CTX, SPLIT);
        const h_cols: usize = @max(@as(usize, D), @max(@as(usize, sh.ff), @as(usize, sh.od)));
        const max_g_stride: usize = gStride(h_cols / 64);
        const p_cols: usize = @max(@as(usize, 2 * sh.ff), @max(@as(usize, sh.gdn_in), @as(usize, sh.att_in)));
        return .{
            .x = try d.alloc_(rows * D * 4),
            .h = try d.alloc_(rows * h_cols * 2),
            .ahi = try d.alloc_(rows * max_g_stride * 2),
            .alo = try d.alloc_(rows * max_g_stride * 2),
            .r = try d.alloc_(rows * D * 2),
            .p = try d.alloc_(rows * p_cols * 2),
            .q = try d.alloc_(rows * @as(usize, sh.aq) * 256 * 2),
            .k = try d.alloc_(rows * @as(usize, sh.gk) * 128 * 2),
            .v = try d.alloc_(rows * @as(usize, sh.gv) * 128 * 2),
            .g = try d.alloc_(rows * @as(usize, sh.gv) * 4),
            .beta = try d.alloc_(rows * @as(usize, sh.gv) * 4),
            .y = try d.alloc_(rows * @as(usize, sh.od) * 2),
            .la = try d.alloc_(@as(usize, sh.gv) * nc * 64 * 4),
            .wb = try d.alloc_(3 * nc * @as(usize, sh.gv) * 64 * 128 * 2),
            .u0 = try d.alloc_(nc * @as(usize, sh.gv) * 64 * 128 * 4),
            .hs = try d.alloc_(3 * nc * @as(usize, sh.gv) * 128 * 128 * 2),
            .vn = try d.alloc_(3 * nc * @as(usize, sh.gv) * 128 * 64 * 2),
            .pm = try d.alloc_(16 * @as(usize, sh.aq) * ns * 4),
            .pl = try d.alloc_(16 * @as(usize, sh.aq) * ns * 4),
            .po = try d.alloc_(16 * @as(usize, sh.aq) * ns * 256 * 4),
            .pscr = try d.alloc_(rows * D * 4),
            .logits = try d.alloc_(if (rank == 0) 16 * VOCAB * 2 else 64),
            .shard_logits = try d.alloc_(if (q.TP > 1) 16 * @as(usize, vocab_len) * 2 else 64),
            .tok = try d.alloc_(16 * 4),
            .pv = try d.alloc_(16 * 1024),
            .pi = try d.alloc_(16 * 1024),
            .mrows = try d.alloc_(16 * 4),
            .mbits = try d.alloc_(if (rank == 0) 16 * MASK_WORDS * 4 else 64),
            .mtok = try d.alloc_(16 * 4),
            .tkv = try d.alloc_(16 * 256 * cand.TOPK_MAX * 4),
            .tki = try d.alloc_(16 * 256 * cand.TOPK_MAX * 4),
            .tkf = try d.alloc_(16 * 256 * 4),
            .cand = try d.alloc_(16 * cand.CAND_WORDS * 4),
            .taps = try d.alloc_(if (rank == 0) rows * 5 * D * 2 else 64),
            .ids = try d.alloc_(rows * 4),
            .zeros = try d.alloc_(rows * @as(usize, sh.gv) * 128 * 2),
            .ones = try d.alloc_(rows * @as(usize, sh.gv) * 4),
            .chain = try d.upload(std.mem.sliceAsBytes(&chainTable())),
            .tree = try d.alloc_(16 * 17 * 4),
            .path = try d.alloc_(16 * 4),
            .mpath = try d.alloc_(16 * 2 * 16 * 4),
            .sl = try d.alloc_(16 * @as(usize, sh.gv) * 128 * 128 * 4),
            .mt = try d.alloc_(LAYERS * 16 * 8 * 8),
            .mpos = try d.alloc_(16 * 4),
            .a8 = try d.alloc_(rows * h_cols),
            .sa = try d.alloc_((h_cols / 128) * (rows + 32) * 4),
        };
    }

    pub fn init(d: *Dev) !Acts {
        return initRank(d, 0, VOCAB);
    }
};

pub fn chainTable() [16 * 17]i32 {
    var t: [16 * 17]i32 = @splat(0);
    for (0..16) |r| {
        t[r * 17] = @intCast(r);
        for (0..r + 1) |dd| t[r * 17 + 1 + dd] = @intCast(dd);
    }
    return t;
}

pub fn mlpIn(d: *Dev, a: *Acts, w: Proj, T: u32, gs: i32, pre: bool, wide: bool) !*anyopaque {
    const sh = q.tpShape();
    if (q.prefill_int8 and wide and w.ilv and w.gs == 128) {
        if (!pre) try quantA8(d, a, a.h, w.k, T, 0);
        try d.launch(d.k.a8_swiglu, a8Groups(w, T), a8Args(a, w, a.p, T));
        return a.p;
    }
    if (wide and w.ilv and gemm_hook == null) {
        const M: i32 = @intCast(T);
        const tiles = (T + 63) / 64;
        const k = if (w.gs == 128) d.k.ngen_swiglu_g128 else d.k.ngen_swiglu;
        try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, .{ a.h, w.w, w.s, w.bg, a.p, M, @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast((T + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) });
        try d.launch(d.k.rowsums_ff, .{ .x = T }, .{ a.p, a.ahi, a.alo, gs, q.rot_on });
        return a.p;
    }
    try gemmRows(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T, wide);
    try d.launch(d.k.swiglu, .{ .x = T, .y = (sh.ff + 1023) / 1024 }, .{ a.p, a.h, a.ahi, a.alo, gs, @as(i32, @intFromBool(w.ilv)), q.rot_on });
    return a.h;
}

pub fn embedRows(d: *Dev, m: *const Model, ids: *anyopaque, x: *anyopaque, rows: u32) !void {
    if (m.embed_s) |es|
        try d.launch(d.k.embed, .{ .x = rows }, .{ ids, m.embed_w, es, m.embed_b.?, x, @as(i32, D) })
    else
        try d.launch(d.k.embed_dense, .{ .x = rows }, .{ ids, m.embed_w, x, @as(i32, D) });
}

pub inline fn gStride(g: usize) usize {
    return @max(@as(usize, 32), (g + 7) & ~@as(usize, 7));
}

pub fn gemm(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    return gemmRows(d, pscr, h, ahi, alo, w, out, m, m > 16);
}

pub fn gemmRows(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32, wide: bool) !void {
    const M: i32 = @intCast(m);
    const N: i32 = @intCast(w.n);
    const K: i32 = @intCast(w.k);
    const G: i32 = @intCast(w.k / w.gs);
    const g128 = w.gs == 128;
    const is_wide = w.wide or ((w.k / w.gs) % 4 != 0);
    const k = d.k;
    const dec_wide = if (g128) k.dec_wide_g128 else k.dec_wide;
    const dec_wide64 = if (g128) k.dec_wide64_g128 else k.dec_wide64;
    const dec_narrow = if (g128) k.dec_narrow_g128 else k.dec_narrow;
    const ngen_wide = if (g128) k.ngen_wide_g128 else k.ngen_wide;
    const ngen_narrow = if (g128) k.ngen_narrow_g128 else k.ngen_narrow;
    const mp: i32 = @intCast((m + 7) / 8 * 8);
    if (!wide) {
        if (w.wide)
            if (w.n >= 30000)
                try d.launch(dec_wide64, .{ .x = 1, .y = w.n / 256 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try d.launch(dec_wide, .{ .x = 1, .y = w.n / 128 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(dec_narrow, .{ .x = 1, .y = w.n / 32 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    } else {
        const tiles = (m + 63) / 64;
        const groups: ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
        if (is_wide)
            try d.launch(ngen_wide, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(ngen_narrow, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G, pscr });
    }
    if (gemm_hook) |f| try f(d, h, w, out, m);
}

pub var gemm_hook: ?*const fn (*Dev, *anyopaque, Proj, *anyopaque, u32) anyerror!void = null;

extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

pub fn dump(d: *Dev, a: *Acts, T: u32, name: []const u8, with_r: bool) !void {
    const dir = q.dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, T * D);
    defer alloc.free(x);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, x.ptr, a.x, T * D * 4, null, 0, null));
    errdefer d.settle();
    if (with_r) {
        const r = try alloc.alloc(f16, T * D);
        defer alloc.free(r);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.ptr, a.r, T * D * 2, null, 0, null));
        try d.sync();
        for (x, r) |*xv, rv| xv.* += @floatCast(rv);
    } else try d.sync();
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_x_{s}.f32", .{ dir, name }, 0);
    const fd = creat(path.ptr, 0o644);
    _ = write(fd, @ptrCast(x.ptr), x.len * 4);
    _ = close(fd);
}

pub fn dumpBuf(d: *Dev, name: []const u8, p: *anyopaque, n: usize, kind: enum { f16, bf16, f32 }) !void {
    const dir = q.dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const raw = try alloc.alloc(u16, n * 2);
    defer alloc.free(raw);
    const bytes: usize = if (kind == .f32) n * 4 else n * 2;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, raw.ptr, p, bytes, null, 0, null));
    try d.sync();
    for (x, 0..) |*v, i| v.* = switch (kind) {
        .f16 => @floatCast(@as(f16, @bitCast(raw[i]))),
        .bf16 => @bitCast(@as(u32, raw[i]) << 16),
        .f32 => @bitCast(@as(u32, raw[2 * i]) | (@as(u32, raw[2 * i + 1]) << 16)),
    };
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_{s}.f32", .{ dir, name }, 0);
    const fd = creat(path.ptr, 0o644);
    _ = write(fd, @ptrCast(x.ptr), x.len * 4);
    _ = close(fd);
}

pub const Want = tail.Want;

comptime {
    if (LAYERS != tail.layers) @compileError("xpu/prompt_tail.zig's layer count does not match qwen27.zig");
    for (TAP_LAYERS) |tl| if (tl == LAYERS - 1) @compileError("the XPU prompt tail assumes the last layer is not tapped");
}

fn atRows(a: *const Acts, base: usize, sh: q.TpShape) Acts {
    var t = a.*;
    t.x = off(a.x, base * D * 4);
    t.r = off(a.r, base * D * 2);
    t.q = off(a.q, base * @as(usize, sh.aq) * 256 * 2);
    t.y = off(a.y, base * @as(usize, sh.od) * 2);
    return t;
}

pub fn kvBytes() usize {
    const sh = q.tpShape();
    return @as(usize, q.MAX_CTX) * @as(usize, sh.akv) * @as(usize, if (q.kv_i8) 256 + 4 else 256 * 2);
}

pub fn kvCap() i32 {
    return if (q.kv_i8) @intCast(q.MAX_CTX) else 0;
}

pub const no_q8: ?*anyopaque = null;

pub fn pgemm(d: *Dev, a: *Acts, x: *anyopaque, w: Proj, out: *anyopaque, T: u32, pre: bool, qrot: i32, int4: bool, wide: bool) !void {
    if (!q.prefill_int8 or !wide or w.gs != 128) return gemmRows(d, a.pscr, x, a.ahi, a.alo, w, out, T, wide);
    if (int4 and q.prefill_int4) {
        try d.launch(d.k.quant_a4, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(w.k)), @as(i32, @intCast(q.maxRows() + 32)), qrot });
        return d.launch(d.k.a4, a8Groups(w, T), a8Args(a, w, out, T));
    }
    if (!pre) try quantA8(d, a, x, w.k, T, qrot);
    try d.launch(d.k.a8, a8Groups(w, T), a8Args(a, w, out, T));
}

pub fn quantA8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, rot: i32) !void {
    try d.launch(d.k.quant_a8, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(q.maxRows() + 32)), rot });
}

pub fn a8Groups(w: Proj, T: u32) ze.GroupCount {
    const tiles = (T + 63) / 64;
    return .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
}

pub fn a8Args(a: *Acts, w: Proj, out: *anyopaque, T: u32) struct { *anyopaque, *anyopaque, *anyopaque, *anyopaque, *anyopaque, i32, i32, i32, *anyopaque, *anyopaque, i32, i32, *anyopaque } {
    return .{ a.a8, w.w, w.s, w.bg, out, @intCast(T), @intCast(w.n), @intCast(w.k), a.ahi, a.alo, @intCast(q.maxRows() + 32), @intCast(w.k / 128), a.sa };
}

pub fn a8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, site: u32) !void {
    if (q.a8_group == 0 or T <= 16 or q.a8_sites & site == 0) return;
    const qg = if (q.a8_group == std.math.maxInt(u32)) K else q.a8_group;
    try d.launch(d.k.fakequant_rows, .{ .x = T }, .{ x, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(qg)), @as(i32, @intCast(q.GS)), q.a8_rot, if (q.a8_sites4 & site != 0) q.a8_levels else @as(i32, 127) });
}

pub var gemm_probe: ?*const fn (*Dev, *Acts, Proj, *anyopaque, u32, usize, u32) anyerror!void = null;
pub var gdn_probe: ?*const fn (*Dev, *Acts, *anyopaque, u32, usize) anyerror!void = null;

pub const Mode = struct { prompt: bool = false, want: Want = .last, stash: bool = false, taps: bool = false, tree: bool = false, multi_max_pos: ?u32 = null, rows_out: ?*[16]i32 = null, cand: ?cand.Mode = null };

/// Multi-GPU TP={1,2,4} forward over `ids` (< q.maxRows() rows) at p0..; returns last row's greedy token.
pub fn forwardRunner(r: *q.Runner, ids: []const i32, p0: u32, mode: Mode) !?i32 {
    const sh = q.tpShape();
    const tp: usize = r.tp;
    const want = mode.want != .none;
    if (ids.len > q.maxRows()) return error.TooManyRows;
    for (ids) |id| if (id < 0 or id >= VOCAB) return error.InvalidQwenToken;
    const T: u32 = @intCast(ids.len);
    const wide = T > 16 or mode.prompt;
    const t32: i32 = @intCast(T);
    const tp_rows = (T + 63) / 64 * 64;
    const nul: ?*anyopaque = null;
    const gs: i32 = @intCast(q.GS);
    const q8on = q.prefill_int8 and wide and q.GS == 128 and q.a8_group == 0;
    const q8ms: i32 = @intCast(q.maxRows() + 32);
    const q8out = q8on and !(q.prefill_int4 and q.int4_sites & 2 != 0);

    var gpus: [4]ze.Gpu = undefined;
    var lists: [4]ze.H = undefined;
    for (0..tp) |rk| {
        const d = r.devAt(rk);
        gpus[rk] = d.g;
        lists[rk] = d.list;
    }

    errdefer r.settleAll();
    for (0..tp) |rk| {
        const d = r.devAt(rk);
        const a = r.actsAt(rk);
        const m = r.modelAt(rk);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
        try embedRows(d, m, a.ids, a.x, T);
    }
    var buff: [16]u8 = undefined;
    if (q.dump_dir != null and p0 == 0) try dump(&r.d, &r.acts, T, "embed", false);
    _ = &buff;

    const tail_hooks = q.dump_h != null or q.dump_dir != null or gemm_probe != null or gdn_probe != null or gemm_hook != null or q.a8_group != 0 or !q.prompt_tail_on;
    var tail_used = false;
    var tail_at: ?usize = null;
    var tail_acts: [4]Acts = undefined;

    for (0..LAYERS) |i| {
        const is_attn = q.isAttention(i);
        const tapped = mode.taps and std.mem.indexOfScalar(usize, &TAP_LAYERS, i) != null;
        const plan = if (mode.prompt and is_attn) tail.tailPlan(T, mode.want, i, tail_hooks or tapped) else tail.Plan.all;
        const kv_only = plan == .kv_only;

        var wr: u32 = T;
        var ap0: u32 = p0;
        switch (plan) {
            .tail => |t0| {
                tail_used = true;
                tail_at = t0;
                for (0..tp) |rk| tail_acts[rk] = atRows(r.actsAt(rk), t0, sh);
                wr = @intCast(T - t0);
                ap0 = p0 + @as(u32, @intCast(t0));
            },
            else => {},
        }

        var kv_broke = false;
        var wa_r_ptrs: [4]*anyopaque = undefined;

        for (0..tp) |rk| {
            const d = r.devAt(rk);
            const m = r.modelAt(rk);
            const a = r.actsAt(rk);
            const l = &m.layers[i];
            const wa: *Acts = if (tail_at != null and plan == .tail) &tail_acts[rk] else a;
            const tree: ?*anyopaque = if (mode.tree) a.tree else null;
            const table = if (mode.tree) a.tree else a.chain;
            const mpos: ?*anyopaque = if (mode.multi_max_pos != null) a.mpos else null;
            const q8a: ?*anyopaque = if (q8on) a.a8 else null;
            const q8s: ?*anyopaque = if (q8on) a.sa else null;
            const q8oa: ?*anyopaque = if (q8out) a.a8 else null;
            const q8os: ?*anyopaque = if (q8out) a.sa else null;

            try d.launch(if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, if (i == 0) 0 else 1), gs, q8a, q8s, q8ms, q.rot_on });

            const min_in = if (q.a8_group != 0 and wide and q.a8_sites & 1 != 0) a.r else a.h;
            if (min_in != a.h) try copy(d, a.r, a.h, T * D * 2);
            try a8(d, a, min_in, D, T, 1);
            try pgemm(d, a, min_in, l.mixer_in, a.p, T, q8on, 0, false, wide);
            if (rk == 0) if (gemm_probe) |f| try f(d, a, l.mixer_in, min_in, T, i, 0);
            if (l.ab) |ab| try d.launch(d.k.dense_rows, .{ .x = (T + 15) / 16, .y = (2 * sh.gv + 15) / 16 }, .{ a.h, ab, a.p, t32, @as(i32, @intCast(sh.gdn_in)), @as(i32, @intCast(sh.ga)) });

            if (l.attn) {
                const ctx = if (mode.multi_max_pos) |mp_val| mp_val + 1 else p0 + T;
                const mt: ?*anyopaque = if (mode.multi_max_pos != null) off(a.mt, i * 16 * 8 * 8) else null;
                const ns = (ctx + SPLIT - 1) / SPLIT;
                try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, @as(i32, @intCast(p0)), EPS, tree, mt, mpos, q.kv8, kvCap() });
                if (kv_only) {
                    kv_broke = true;
                    continue;
                }
                if (wide and q.kv_i8) {
                    try d.launch(d.k.attn_prompt8, .{ .x = (wr + 7) / 8, .y = sh.akv }, .{ wa.q, l.kc.?, l.vc.?, wa.y, @as(i32, @intCast(ap0)), @as(i32, @intCast(wr)), @as(f32, 1.0 / 16.0), kvCap() });
                } else if (wide) {
                    try d.launch(d.k.attn_prompt, .{ .x = (wr + 7) / 8, .y = sh.akv }, .{ wa.q, l.kc.?, l.vc.?, wa.y, @as(i32, @intCast(ap0)), @as(i32, @intCast(wr)), @as(f32, 1.0 / 16.0) });
                } else {
                    try d.launch(d.k.attn_decode, .{ .x = ns, .y = sh.akv, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), @as(f32, 1.0 / 16.0), if (mode.multi_max_pos != null) tree else table, mt, mpos, kvCap() });
                    try d.launch(d.k.attn_merge, .{ .x = sh.aq, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), table, mpos });
                }
                if (q.kv8 & 1 != 0) try d.launch(d.k.rot_heads, .{ .x = wr, .y = sh.aq }, .{wa.y});
                const p_attn: *anyopaque = if (tail_at) |t0| off(a.p, t0 * @as(usize, sh.att_in) * 2) else a.p;
                try d.launch(d.k.attn_gate, .{ .x = wr }, .{ wa.y, p_attn, wa.h, wa.ahi, wa.alo, gs, q.rot_on, q8oa, q8os, q8ms });
            } else {
                const cin = l.conv[l.conv_cur].?;
                const cout = l.conv[1 - l.conv_cur].?;
                const gmt: ?*anyopaque = if (mode.multi_max_pos != null) off(a.mt, i * 16 * 8 * 8) else null;
                if (mode.multi_max_pos == null) l.conv_cur = 1 - l.conv_cur;
                if (wide)
                    try d.launch(d.k.gdn_pre_rows, .{ .x = (T + 7) / 8, .y = (sh.nconv + 2047) / 2048 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32 })
                else
                    try d.launch(d.k.gdn_pre, .{ .x = T, .y = sh.nconv / 256 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32, tree, gmt });
                if (wide and tp_rows > T) {
                    const pad = tp_rows - T;
                    const cp = struct {
                        fn f(dd: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
                            try ze.check(dd.g.api.zeCommandListAppendMemoryCopy(dd.list, dst, src, n, null, 0, null));
                        }
                    }.f;
                    try cp(d, off(a.k, T * @as(usize, sh.gk) * 128 * 2), a.zeros, pad * @as(usize, sh.gk) * 128 * 2);
                    try cp(d, off(a.v, T * @as(usize, sh.gv) * 128 * 2), a.zeros, pad * @as(usize, sh.gv) * 128 * 2);
                    try cp(d, off(a.beta, T * @as(usize, sh.gv) * 4), a.zeros, pad * @as(usize, sh.gv) * 4);
                    try cp(d, off(a.g, T * @as(usize, sh.gv) * 4), a.ones, pad * @as(usize, sh.gv) * 4);
                }
                if (!wide) {
                    if (mode.stash) {
                        const st = l.stash.?;
                        try copy(d, off(st, sh.st_q), a.q, T * @as(usize, sh.gk) * 128 * 2);
                        try copy(d, off(st, sh.st_k), a.k, T * @as(usize, sh.gk) * 128 * 2);
                        try copy(d, off(st, sh.st_v), a.v, T * @as(usize, sh.gv) * 128 * 2);
                        try copy(d, off(st, sh.st_g), a.g, T * @as(usize, sh.gv) * 4);
                        try copy(d, off(st, sh.st_b), a.beta, T * @as(usize, sh.gv) * 4);
                        try copy(d, off(st, sh.st_p), a.p, T * @as(usize, sh.gdn_in) * 2);
                    }
                    try d.launch(d.k.gdn_step, .{ .x = 16, .y = sh.gv, .z = if (mode.multi_max_pos != null) T else 1 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, if (mode.stash) l.rec_save.? else l.rec.?, a.y, t32, nul, tree, a.sl, gmt });
                } else {
                    const nc: i32 = @intCast(tp_rows / 64);
                    const ncu: u32 = tp_rows / 64;
                    const hk: i32 = @intCast(sh.gk);
                    const hv: i32 = @intCast(sh.gv);
                    try d.launch(d.k.gdn_prep, .{ .x = ncu, .y = sh.gv }, .{ a.k, a.v, a.g, a.beta, a.la, a.wb, a.u0, hk, hv, nc, @as(i32, 0) });
                    try d.launch(d.k.gdn_state, .{ .x = sh.gv }, .{ a.k, a.wb, a.u0, a.la, l.rec.?, l.rec.?, a.hs, a.vn, hk, hv, nc });
                    try d.launch(d.k.gdn_out, .{ .x = ncu, .y = sh.gv }, .{ a.q, a.k, a.la, a.hs, a.vn, a.y, hk, hv, nc });
                    if (rk == 0) if (gdn_probe) |f| if (p0 == 0) try f(d, a, l.rec.?, T, i);
                }
                try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs, q.rot_on, q8oa, q8os, q8ms });
            }

            try a8(d, wa, wa.h, sh.od, wr, 2);
            const out_target = if (tp > 1) r.comm.currentSendBuf(rk) else wa.r;
            try pgemm(d, wa, wa.h, l.mixer_out, out_target, wr, q8out, 0, q.int4_sites & 2 != 0, wide);
            if (rk == 0) if (gemm_probe) |f| try f(d, wa, l.mixer_out, wa.h, wr, i, 1);
            wa_r_ptrs[rk] = wa.r;
        }
        if (kv_broke) break;

        if (tp > 1) try r.comm.allReduceF16(gpus[0..tp], lists[0..tp], null, wa_r_ptrs[0..tp], @as(usize, wr) * D);

        for (0..tp) |rk| {
            const d = r.devAt(rk);
            const m = r.modelAt(rk);
            const a = r.actsAt(rk);
            const l = &m.layers[i];
            const wa: *Acts = if (tail_at != null and plan == .tail) &tail_acts[rk] else a;
            const q8a: ?*anyopaque = if (q8on) a.a8 else null;
            const q8s: ?*anyopaque = if (q8on) a.sa else null;

            try d.launch(if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = wr }, .{ wa.x, wa.r, l.post_norm, wa.h, wa.ahi, wa.alo, @as(i32, D), EPS, @as(i32, 1), gs, q8a, q8s, q8ms, q.rot_on });
            try a8(d, wa, wa.h, D, wr, 4);
            if (rk == 0) if (gemm_probe) |f| try f(d, wa, l.gu, wa.h, wr, i, 2);
            const act = try mlpIn(d, wa, l.gu, wr, gs, q8on, wide);
            try a8(d, wa, act, sh.ff, wr, 8);
            const down_target = if (tp > 1) r.comm.currentSendBuf(rk) else wa.r;
            try pgemm(d, wa, act, l.down, down_target, wr, false, if (act == wa.p) q.rot_on else 0, q.int4_sites & 8 != 0, wide);
            if (rk == 0) if (gemm_probe) |f| try f(d, wa, l.down, act, wr, i, 3);
            wa_r_ptrs[rk] = wa.r;
        }

        if (tp > 1) try r.comm.allReduceF16(gpus[0..tp], lists[0..tp], null, wa_r_ptrs[0..tp], @as(usize, wr) * D);
        if (q.dump_dir != null and p0 == 0 and (i < 4 or i == 31 or i == 63)) {
            try r.syncAll();
            try dump(&r.d, &r.acts, T, try std.fmt.bufPrint(&buff, "{d}", .{i}), true);
        }

        if (mode.taps) for (TAP_LAYERS, 0..) |tl, slot| if (tl == i) {
            try r.d.launch(r.d.k.tap, .{ .x = D / 256, .y = T }, .{ r.acts.x, r.acts.r, r.acts.taps, @as(i32, D), @as(i32, @intCast(slot)) });
        };
    }

    if (!want and q.dump_h == null) {
        try r.syncAll();
        return null;
    }

    const hgs: i32 = @intCast(r.model.head.gs);
    const fr: u32 = if (tail_at) |t0| @intCast(T - t0) else T;
    for (0..tp) |rk| {
        const d = r.devAt(rk);
        const m = r.modelAt(rk);
        const fa: *Acts = if (tail_used) &tail_acts[rk] else r.actsAt(rk);
        try d.launch(if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = fr }, .{ fa.x, fa.r, m.norm, fa.h, fa.ahi, fa.alo, @as(i32, D), EPS, @as(i32, 1), hgs, no_q8, no_q8, @as(i32, 0), @as(i32, 0) });
    }

    if (q.dump_h) |fd| {
        try r.syncAll();
        const host = try alloc.alloc(u16, T * D);
        defer alloc.free(host);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, host.ptr, r.acts.h, T * D * 2, null, 0, null));
        try r.d.sync();
        _ = write(fd, @ptrCast(host.ptr), T * D * 2);
        if (!want) return null;
    }

    const first = if (mode.want == .rows) 0 else fr - 1;
    const rows = fr - first;

    if (tp == 1) {
        const fa: *Acts = if (tail_used) &tail_acts[0] else &r.acts;
        const hgp = gStride(D / r.model.head.gs);
        if (gemm_probe) |f| try f(&r.d, fa, r.model.head, fa.h, fr, LAYERS, 4);
        try gemm(&r.d, r.acts.pscr, off(fa.h, first * D * 2), off(fa.ahi, first * hgp * 2), off(fa.alo, first * hgp * 2), r.model.head, r.acts.logits, rows);
    } else {
        for (0..tp) |rk| {
            const d = r.devAt(rk);
            const m = r.modelAt(rk);
            const a = r.actsAt(rk);
            const fa: *Acts = if (tail_used) &tail_acts[rk] else a;
            const hgp = gStride(D / m.head.gs);
            try gemm(d, a.pscr, off(fa.h, first * D * 2), off(fa.ahi, first * hgp * 2), off(fa.alo, first * hgp * 2), m.head, a.shard_logits, rows);
            try ze.check(d.g.api.zeCommandListAppendSignalEvent(d.list, r.logits_ev[rk]));
        }
        try ze.check(r.d.g.api.zeCommandListAppendWaitOnEvents(r.d.list, @intCast(tp), &r.logits_ev));
        for (0..tp) |rk| {
            const a_rk = r.actsAt(rk);
            const voff: usize = r.vocab_off[rk];
            const vlen: usize = r.vocab_len[rk];
            for (0..rows) |row| {
                const dst_p = off(r.acts.logits, (row * VOCAB + voff) * 2);
                const src_p = off(a_rk.shard_logits, row * vlen * 2);
                try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, dst_p, src_p, vlen * 2, null, 0, null));
            }
        }
    }

    try r.d.launch(r.d.k.argmax_part, .{ .x = 256, .y = rows }, .{ r.acts.logits, @as(i32, VOCAB), r.acts.pv, r.acts.pi });
    try r.d.launch(r.d.k.argmax, .{ .x = rows }, .{ r.acts.pv, r.acts.pi, r.acts.tok });
    if (mode.cand) |c| {
        const d = &r.d;
        const a = &r.acts;
        try d.launch(d.k.topk_part, .{ .x = cand.TOPK_PARTS, .y = rows }, .{ a.logits, @as(i32, VOCAB), @as(i32, @intCast(c.k)), a.tkv, a.tki, a.tkf });
        try d.launch(d.k.topk_merge, .{ .x = rows }, .{ a.tkv, a.tki, a.tkf, @as(i32, @intCast(c.k)), @as(i32, cand.TOPK_PARTS), a.cand });
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, c.out, a.cand, rows * cand.CAND_WORDS * 4, null, 0, null));
    }
    var toks: [16]i32 = undefined;
    try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, rows * 4, null, 0, null));
    try r.syncAll();
    if (mode.cand) |c| c.rows.* = rows; // the candidates are the host's once the copy has landed
    if (mode.rows_out) |o| @memcpy(o[0..rows], toks[0..rows]);
    return toks[rows - 1];
}
