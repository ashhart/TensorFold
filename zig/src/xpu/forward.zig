//! One Qwen3.8-27B forward on the XPU: activations, prompt GEMMs (fp16, W4A8, W4A4) and the layer loop.
const std = @import("std");
const ze = @import("ze.zig");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const loader = @import("load.zig");
const D = q.D;
const Dev = dev.Dev;
const EPS = q.EPS;
const FF = q.FF;
const GDN_IN = q.GDN_IN;
const LAYERS = q.LAYERS;
const MASK_WORDS = q.MASK_WORDS;
const Model = loader.Model;
const Proj = loader.Proj;
const SPLIT = q.SPLIT;
const ST_B = q.ST_B;
const ST_G = q.ST_G;
const ST_K = q.ST_K;
const ST_P = q.ST_P;
const ST_Q = q.ST_Q;
const ST_V = q.ST_V;
const TAP_LAYERS = q.TAP_LAYERS;
const VOCAB = q.VOCAB;
const alloc = q.alloc;
extern "c" fn close(fd: c_int) c_int;
const copy = dev.copy;
const off = dev.off;

// ---------------------------------------------------------------- activations and the forward

pub const Acts = struct {
    x: *anyopaque, // fp32 [q.CHUNK, D]
    h: *anyopaque, // fp16 [q.CHUNK, FF]
    ahi: *anyopaque, // fp16 [q.CHUNK, FF / 64]
    alo: *anyopaque,
    r: *anyopaque, // fp16 [q.CHUNK, D]
    p: *anyopaque, // fp16 [q.CHUNK, 2 FF]
    q: *anyopaque, // bf16 [q.CHUNK, 16, 128] (GDN) / [q.CHUNK, 24, 256] (attention)
    k: *anyopaque, // bf16 [q.CHUNK, 16, 128]
    v: *anyopaque, // bf16 [q.CHUNK, 48, 128]
    g: *anyopaque, // fp32 [q.CHUNK, 48]
    beta: *anyopaque,
    y: *anyopaque, // bf16 [q.CHUNK, 6144] (GDN out, attention out)
    la: *anyopaque,
    wb: *anyopaque,
    u0: *anyopaque,
    hs: *anyopaque,
    vn: *anyopaque,
    pm: *anyopaque,
    pl: *anyopaque,
    po: *anyopaque,
    pscr: *anyopaque, // fp32 [q.CHUNK, D] (slice totals of the narrow prompt GEMM)
    logits: *anyopaque, // fp16 [16, VOCAB]
    taps: *anyopaque, // bf16 [q.CHUNK, 5 D]: hidden states after TAP_LAYERS, when Mode.taps
    tok: *anyopaque, // i32
    pv: *anyopaque, // argmax partials: f32 [256], i32 [256]
    pi: *anyopaque,
    mrows: *anyopaque, // i32 [16]: argmaxMasked's rows
    mbits: *anyopaque, // u32 [16, MASK_WORDS]: and their grammar bits
    mtok: *anyopaque, // i32 [16]: and their tokens
    ids: *anyopaque, // i32 [q.CHUNK]
    zeros: *anyopaque, // padding rows of the GDN chunks: k = v = 0, β = 0 …
    ones: *anyopaque, // … and g = 1
    chain: *anyopaque, // i32 [16, 17]: the tree table of a chain (row t at depth t), for the decode attention
    tree: *anyopaque, // i32 [16, 17]: the verified block's tree table (decode.cl, DEPTH and ANC)
    path: *anyopaque, // i32 [16]: the rows of the kept path
    mpath: *anyopaque, // i32 [16, 2, 16]: keepPaths' per-window paths, rows of the block then rows of the window
    sl: *anyopaque, // fp32 [16, 48, 128, 128]: GDN state after each row of a verified tree
    mt: *anyopaque, // u64 [LAYERS, 16, 8]: stepMulti's per-row state addresses
    mpos: *anyopaque, // i32 [16]: and positions
    a8: *anyopaque, // W4A8 prompt: activations int8 [q.CHUNK, FF] (permuted by blocks of 8) …
    sa: *anyopaque, // … and their scales fp32 [FF / 128, q.CHUNK + 32]

    pub fn init(d: *Dev) !Acts {
        const nc = q.CHUNK / 64;
        const ns = q.MAX_CTX / SPLIT;
        return .{
            .x = try d.alloc_(q.CHUNK * D * 4),
            .h = try d.alloc_(q.CHUNK * FF * 2),
            .ahi = try d.alloc_(q.CHUNK * (FF / 64) * 2),
            .alo = try d.alloc_(q.CHUNK * (FF / 64) * 2),
            .r = try d.alloc_(q.CHUNK * D * 2),
            .p = try d.alloc_(q.CHUNK * 2 * FF * 2),
            .q = try d.alloc_(q.CHUNK * 24 * 256 * 2),
            .k = try d.alloc_(q.CHUNK * 16 * 128 * 2),
            .v = try d.alloc_(q.CHUNK * 48 * 128 * 2),
            .g = try d.alloc_(q.CHUNK * 48 * 4),
            .beta = try d.alloc_(q.CHUNK * 48 * 4),
            .y = try d.alloc_(q.CHUNK * 6144 * 2),
            .la = try d.alloc_(48 * nc * 64 * 4),
            .wb = try d.alloc_(nc * 48 * 64 * 128 * 2),
            .u0 = try d.alloc_(nc * 48 * 64 * 128 * 4),
            .hs = try d.alloc_(nc * 48 * 128 * 128 * 2),
            .vn = try d.alloc_(nc * 48 * 128 * 64 * 2),
            // decode attention partials: at most 16 rows (the prompt kernel needs none)
            .pm = try d.alloc_(16 * 24 * ns * 4),
            .pl = try d.alloc_(16 * 24 * ns * 4),
            .po = try d.alloc_(16 * 24 * ns * 256 * 4),
            .pscr = try d.alloc_(q.CHUNK * D * 4),
            .logits = try d.alloc_(16 * VOCAB * 2),
            .tok = try d.alloc_(16 * 4),
            .pv = try d.alloc_(16 * 1024),
            .pi = try d.alloc_(16 * 1024),
            .mrows = try d.alloc_(16 * 4),
            .mbits = try d.alloc_(16 * MASK_WORDS * 4),
            .mtok = try d.alloc_(16 * 4),
            .taps = try d.alloc_(q.CHUNK * 5 * D * 2),
            .ids = try d.alloc_(q.CHUNK * 4),
            .zeros = try d.alloc_(q.CHUNK * 48 * 128 * 2),
            .ones = try d.alloc_(q.CHUNK * 48 * 4),
            .chain = try d.upload(std.mem.sliceAsBytes(&chainTable())),
            .tree = try d.alloc_(16 * 17 * 4),
            .path = try d.alloc_(16 * 4),
            .mpath = try d.alloc_(16 * 2 * 16 * 4),
            .sl = try d.alloc_(16 * 48 * 128 * 128 * 4),
            .mt = try d.alloc_(LAYERS * 16 * 8 * 8),
            .mpos = try d.alloc_(16 * 4),
            .a8 = try d.alloc_(q.CHUNK * FF),
            .sa = try d.alloc_((FF / 128) * (q.CHUNK + 32) * 4),
        };
    }
};

/// Row t at depth t, its ancestors rows 0 .. t: a chain's tree table.
pub fn chainTable() [16 * 17]i32 {
    var t: [16 * 17]i32 = @splat(0);
    for (0..16) |r| {
        t[r * 17] = @intCast(r);
        for (0..r + 1) |dd| t[r * 17 + 1 + dd] = @intCast(dd);
    }
    return t;
}

/// gate|up and SwiGLU: the activation and its row sums; prompt rows via the GEMM epilogue, decode rows via `swiglu`.
pub fn mlpIn(d: *Dev, a: *Acts, w: Proj, T: u32, gs: i32, pre: bool) !*anyopaque {
    if (q.prefill_int8 and T > 16 and w.ilv and w.gs == 128) { // W4A8: SwiGLU epilogue; down quantizes its own input
        if (!pre) try quantA8(d, a, a.h, w.k, T, 0);
        try d.launch(d.k.a8_swiglu, a8Groups(w, T), a8Args(a, w, a.p, T));
        return a.p;
    }
    if (T > 16 and w.ilv) {
        const M: i32 = @intCast(T);
        const tiles = (T + 63) / 64;
        const k = if (w.gs == 128) d.k.ngen_swiglu_g128 else d.k.ngen_swiglu;
        try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, .{ a.h, w.w, w.s, w.bg, a.p, M, @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast((T + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) });
        try d.launch(d.k.rowsums_ff, .{ .x = T }, .{ a.p, a.ahi, a.alo, gs, q.rot_on });
        return a.p;
    }
    try gemm(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T);
    try d.launch(d.k.swiglu, .{ .x = T, .y = 17 }, .{ a.p, a.h, a.ahi, a.alo, gs, @as(i32, @intFromBool(w.ilv)), q.rot_on });
    return a.h;
}

/// The rows' embeddings: x fp32 [T, D] (bf16 values) for the token ids at `ids` on the device.
pub fn embedRows(d: *Dev, m: *const Model, ids: *anyopaque, x: *anyopaque, rows: u32) !void {
    if (m.embed_s) |es|
        try d.launch(d.k.embed, .{ .x = rows }, .{ ids, m.embed_w, es, m.embed_b.?, x, @as(i32, D) })
    else
        try d.launch(d.k.embed_dense, .{ .x = rows }, .{ ids, m.embed_w, x, @as(i32, D) });
}

/// out[M, N] fp16 = h[M, K] · Wᵀ : decode kernels up to 16 rows, the nGEN prompt GEMMs above.
pub fn gemm(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    const M: i32 = @intCast(m);
    const N: i32 = @intCast(w.n);
    const K: i32 = @intCast(w.k);
    const G: i32 = @intCast(w.k / w.gs);
    const g128 = w.gs == 128;
    const k = d.k;
    const dec_wide = if (g128) k.dec_wide_g128 else k.dec_wide;
    const dec_wide64 = if (g128) k.dec_wide64_g128 else k.dec_wide64;
    const dec_narrow = if (g128) k.dec_narrow_g128 else k.dec_narrow;
    const ngen_wide = if (g128) k.ngen_wide_g128 else k.ngen_wide;
    const ngen_narrow = if (g128) k.ngen_narrow_g128 else k.ngen_narrow;
    const mp: i32 = @intCast((m + 7) / 8 * 8);
    if (m <= 16) {
        if (w.wide)
            // 64 columns per subgroup for the widest shapes (gate|up, head), 32 below (measured end to end)
            if (w.n >= 30000)
                try d.launch(dec_wide64, .{ .x = 1, .y = w.n / 256 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try d.launch(dec_wide, .{ .x = 1, .y = w.n / 128 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(dec_narrow, .{ .x = 1, .y = w.n / 32 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    } else {
        const tiles = (m + 63) / 64;
        const groups: ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
        if (w.wide)
            try d.launch(ngen_wide, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(ngen_narrow, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G, pscr });
    }
}

extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

pub fn dump(d: *Dev, a: *Acts, T: u32, name: []const u8, with_r: bool) !void {
    const dir = q.dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, T * D);
    defer alloc.free(x);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, x.ptr, a.x, T * D * 4, null, 0, null));
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

/// A device buffer of `n` values (2 bytes: fp16 or bf16, 4: fp32) as fp32 in DIR/xpu_{name}.f32.
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

/// What a forward leaves: nothing, the last row's logits and greedy token, or every row's (verification, ≤ 16 rows).
pub const Want = enum { none, last, rows };
pub fn kvBytes() usize {
    return @as(usize, q.MAX_CTX) * 4 * @as(usize, if (q.kv_i8) 256 + 4 else 256 * 2);
}
pub fn kvCap() i32 {
    return if (q.kv_i8) @intCast(q.MAX_CTX) else 0;
}
/// add_rmsnorm without the fused W4A8 quantization (its A8 and SA arguments).
pub const no_q8: ?*anyopaque = null;

/// Prompt GEMM: W4A8 when on and the rows are prompt rows, else fp16; `pre`: x already in acts.a8; `qrot`: see quantA8.
pub fn pgemm(d: *Dev, a: *Acts, x: *anyopaque, w: Proj, out: *anyopaque, T: u32, pre: bool, qrot: i32, int4: bool) !void {
    if (!q.prefill_int8 or T <= 16 or w.gs != 128) return gemm(d, a.pscr, x, a.ahi, a.alo, w, out, T);
    if (int4 and q.prefill_int4) {
        try d.launch(d.k.quant_a4, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(w.k)), @as(i32, @intCast(q.CHUNK + 32)), qrot });
        return d.launch(d.k.a4, a8Groups(w, T), a8Args(a, w, out, T));
    }
    if (!pre) try quantA8(d, a, x, w.k, T, qrot);
    try d.launch(d.k.a8, a8Groups(w, T), a8Args(a, w, out, T));
}
pub fn quantA8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, rot: i32) !void {
    try d.launch(d.k.quant_a8, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(q.CHUNK + 32)), rot });
}
pub fn a8Groups(w: Proj, T: u32) ze.GroupCount {
    const tiles = (T + 63) / 64;
    return .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
}
pub fn a8Args(a: *Acts, w: Proj, out: *anyopaque, T: u32) struct { *anyopaque, *anyopaque, *anyopaque, *anyopaque, *anyopaque, i32, i32, i32, *anyopaque, *anyopaque, i32, i32, *anyopaque } {
    return .{ a.a8, w.w, w.s, w.bg, out, @intCast(T), @intCast(w.n), @intCast(w.k), a.ahi, a.alo, @intCast(q.CHUNK + 32), @intCast(w.k / 128), a.sa };
}

pub fn a8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, site: u32) !void {
    if (q.a8_group == 0 or T <= 16 or q.a8_sites & site == 0) return;
    const qg = if (q.a8_group == std.math.maxInt(u32)) K else q.a8_group;
    try d.launch(d.k.fakequant_rows, .{ .x = T }, .{ x, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(qg)), @as(i32, @intCast(q.GS)), q.a8_rot, if (q.a8_sites4 & site != 0) q.a8_levels else @as(i32, 127) });
}
/// tree: rows form a tree (acts.tree), each seeing its ancestors; multi: each row a different slot's next token.
pub const Mode = struct { want: Want = .last, stash: bool = false, taps: bool = false, tree: bool = false, multi: ?struct { mt: *anyopaque, pos: *anyopaque, max_pos: u32 } = null };

/// One forward over `ids` (≤ q.CHUNK rows) at p0..; returns the last row's greedy token; ≤ 16 rows run GDN serially.
pub fn forward(d: *Dev, m: *Model, a: *Acts, ids: []const i32, p0: u32, mode: Mode) !?i32 {
    const want = mode.want != .none;
    const T: u32 = @intCast(ids.len);
    const t32: i32 = @intCast(T);
    const tp = (T + 63) / 64 * 64; // GDN chunk rows (padded)
    const nul: ?*anyopaque = null;
    const gs: i32 = @intCast(q.GS);
    const tree: ?*anyopaque = if (mode.tree) a.tree else null;
    // W4A8 prompt: the norms before mixer in and gate|up quantize h themselves (not with the study mode)
    const q8on = q.prefill_int8 and T > 16 and q.GS == 128 and q.a8_group == 0;
    const q8a: ?*anyopaque = if (q8on) a.a8 else null;
    const q8s: ?*anyopaque = if (q8on) a.sa else null;
    const q8ms: i32 = @intCast(q.CHUNK + 32);
    // the mixers' out projection in W4A8: its input quantized by attn_gate / gated_norm (not when it is in int4)
    const q8out = q8on and !(q.prefill_int4 and q.int4_sites & 2 != 0);
    const q8oa: ?*anyopaque = if (q8out) a.a8 else null;
    const q8os: ?*anyopaque = if (q8out) a.sa else null;
    const table = if (mode.tree) a.tree else a.chain;
    const mpos: ?*anyopaque = if (mode.multi) |mm| mm.pos else null;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
    try embedRows(d, m, a.ids, a.x, T);
    if (p0 == 0) try dump(d, a, T, "embed", false);
    for (&m.layers, 0..) |*l, i| {
        try d.launch(if (T > 16) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, if (i == 0) 0 else 1), gs, q8a, q8s, q8ms, q.rot_on });
        const probe = p0 == 0 and i == 0;
        if (probe) try dumpBuf(d, "h0", a.h, T * D, .f16);
        // W4A8 study: the GEMM reads a quantized copy (in r, free until mixer_out); the dense a/b gates read h as is
        const min_in = if (q.a8_group != 0 and T > 16 and q.a8_sites & 1 != 0) a.r else a.h;
        if (min_in != a.h) try copy(d, a.r, a.h, T * D * 2);
        try a8(d, a, min_in, D, T, 1);
        try pgemm(d, a, min_in, l.mixer_in, a.p, T, q8on, 0, false);
        if (l.ab) |ab| try d.launch(d.k.dense_rows, .{ .x = (T + 15) / 16, .y = 96 / 16 }, .{ a.h, ab, a.p, t32, @as(i32, GDN_IN), @as(i32, 16384) });
        if (probe) try dumpBuf(d, "p0", a.p, T * GDN_IN, .f16);
        if (l.attn) {
            const ctx = if (mode.multi) |mm| mm.max_pos + 1 else p0 + T; // tree rows see fewer slices
            const mt: ?*anyopaque = if (mode.multi) |mm| off(mm.mt, i * 16 * 8 * 8) else null;
            const ns = (ctx + SPLIT - 1) / SPLIT;
            try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, @as(i32, @intCast(p0)), EPS, tree, mt, mpos, q.kv8, kvCap() });
            if (T > 16 and q.kv_i8) {
                try d.launch(d.k.attn_prompt8, .{ .x = (T + 7) / 8, .y = 4 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(f32, 1.0 / 16.0), kvCap() });
            } else if (T > 16 and q.attn_own) {
                try d.launch(d.k.attn_prompt, .{ .x = (T + 7) / 8, .y = 4 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(f32, 1.0 / 16.0) });
            } else if (T > 16) { // the port's Triton prompt kernel: 256 rows a program
                try d.launch(d.k.prompt_attn, .{ .x = (T + 255) / 256, .y = 24 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(i32, @intCast(ctx)), nul, nul });
            } else {
                try d.launch(d.k.attn_decode, .{ .x = ns, .y = 4, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), @as(f32, 1.0 / 16.0), if (mode.multi != null) tree else table, mt, mpos, kvCap() });
                try d.launch(d.k.attn_merge, .{ .x = 24, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), table, mpos });
            }
            if (q.kv8 & 1 != 0) try d.launch(d.k.rot_heads, .{ .x = T, .y = 24 }, .{a.y});
            try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, q.rot_on, q8oa, q8os, q8ms });
        } else {
            const cin = l.conv[l.conv_cur].?;
            const cout = l.conv[1 - l.conv_cur].?;
            const gmt: ?*anyopaque = if (mode.multi) |mm| off(mm.mt, i * 16 * 8 * 8) else null;
            if (mode.multi == null) l.conv_cur = 1 - l.conv_cur; // multi: each slot's flips after the forward
            if (T > 16)
                try d.launch(d.k.gdn_pre_rows, .{ .x = (T + 7) / 8, .y = 5 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32 })
            else
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = 40 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32, tree, gmt });
            if (T > 16 and tp > T) { // neutral padding rows: the state does not move
                const pad = tp - T;
                const cp = struct {
                    fn f(dd: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
                        try ze.check(dd.g.api.zeCommandListAppendMemoryCopy(dd.list, dst, src, n, null, 0, null));
                    }
                }.f;
                try cp(d, off(a.k, T * 16 * 128 * 2), a.zeros, pad * 16 * 128 * 2);
                try cp(d, off(a.v, T * 48 * 128 * 2), a.zeros, pad * 48 * 128 * 2);
                try cp(d, off(a.beta, T * 48 * 4), a.zeros, pad * 48 * 4);
                try cp(d, off(a.g, T * 48 * 4), a.ones, pad * 48 * 4);
            }
            if (T <= 16) {
                if (mode.stash) { // inputs, for Runner.commit's replay
                    const st = l.stash.?;
                    try copy(d, off(st, ST_Q), a.q, T * 16 * 128 * 2);
                    try copy(d, off(st, ST_K), a.k, T * 16 * 128 * 2);
                    try copy(d, off(st, ST_V), a.v, T * 48 * 128 * 2);
                    try copy(d, off(st, ST_G), a.g, T * 48 * 4);
                    try copy(d, off(st, ST_B), a.beta, T * 48 * 4);
                    try copy(d, off(st, ST_P), a.p, T * GDN_IN * 2);
                }
                // verification: post-block state to rec_save, rec keeps the entry; tree: states to acts.sl
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = 48, .z = if (mode.multi != null) T else 1 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, if (mode.stash) l.rec_save.? else l.rec.?, a.y, t32, nul, tree, a.sl, gmt });
            } else {
                const nc: i32 = @intCast(tp / 64);
                const ncu: u32 = tp / 64;
                const hk: i32 = 16;
                const hv: i32 = 48;
                try d.launch(d.k.gdn_prep, .{ .x = ncu, .y = 48 }, .{ a.k, a.v, a.g, a.beta, a.la, a.wb, a.u0, hk, hv, nc, @as(i32, 0) });
                try d.launch(d.k.gdn_state, .{ .x = 48 }, .{ a.k, a.wb, a.u0, a.la, l.rec.?, l.rec.?, a.hs, a.vn, hk, hv, nc });
                try d.launch(d.k.gdn_out, .{ .x = ncu, .y = 48 }, .{ a.q, a.k, a.la, a.hs, a.vn, a.y, hk, hv, nc });
            }
            if (probe) {
                try dumpBuf(d, "q0", a.q, T * 2048, .bf16);
                try dumpBuf(d, "k0", a.k, T * 2048, .bf16);
                try dumpBuf(d, "v0", a.v, T * 6144, .bf16);
                try dumpBuf(d, "g0", a.g, T * 48, .f32);
                try dumpBuf(d, "beta0", a.beta, T * 48, .f32);
                try dumpBuf(d, "y0", a.y, T * 6144, .bf16);
            }
            try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs, q.rot_on, q8oa, q8os, q8ms });
            if (probe) try dumpBuf(d, "o0", a.h, T * 6144, .f16);
        }
        try a8(d, a, a.h, 6144, T, 2);
        try pgemm(d, a, a.h, l.mixer_out, a.r, T, q8out, 0, q.int4_sites & 2 != 0);
        if (probe) try dumpBuf(d, "r0", a.r, T * D, .f16);
        try d.launch(if (T > 16) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), gs, q8a, q8s, q8ms, q.rot_on });
        try a8(d, a, a.h, D, T, 4);
        const act = try mlpIn(d, a, l.gu, T, gs, q8on);
        try a8(d, a, act, 17408, T, 8);
        try pgemm(d, a, act, l.down, a.r, T, false, if (act == a.p) q.rot_on else 0, q.int4_sites & 8 != 0);
        if (p0 == 0 and i < 4) try dump(d, a, T, try std.fmt.allocPrint(alloc, "{d}", .{i}), true);
        if (mode.taps) for (TAP_LAYERS, 0..) |tl, slot| if (tl == i)
            try d.launch(d.k.tap, .{ .x = D / 256, .y = T }, .{ a.x, a.r, a.taps, @as(i32, D), @as(i32, @intCast(slot)) });
    }
    if (!want and q.dump_h == null) { // x is per token: nothing carries to the next chunk
        try d.sync();
        return null;
    }
    const hgs: i32 = @intCast(m.head.gs);
    try d.launch(if (T > 16) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), hgs, no_q8, no_q8, @as(i32, 0), @as(i32, 0) });
    if (q.dump_h) |fd| { // XPU_DUMP_H: every row's final normed hidden state, appended
        const host = try alloc.alloc(u16, T * D);
        defer alloc.free(host);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, host.ptr, a.h, T * D * 2, null, 0, null));
        try d.sync();
        _ = write(fd, @ptrCast(host.ptr), T * D * 2);
        if (!want) return null;
    }
    const first = if (mode.want == .rows) 0 else T - 1;
    const rows = T - first;
    try gemm(d, a.pscr, off(a.h, first * D * 2), off(a.ahi, first * (D / m.head.gs) * 2), off(a.alo, first * (D / m.head.gs) * 2), m.head, a.logits, rows);
    try d.launch(d.k.argmax_part, .{ .x = 256, .y = rows }, .{ a.logits, @as(i32, VOCAB), a.pv, a.pi });
    try d.launch(d.k.argmax, .{ .x = rows }, .{ a.pv, a.pi, a.tok });
    var toks: [16]i32 = undefined;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, a.tok, rows * 4, null, 0, null));
    try d.sync();
    return toks[rows - 1];
}
