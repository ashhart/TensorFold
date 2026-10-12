//! MiMo-V2's prompt path, launch for launch as the Python engine's prompt chunks (forward(prompt=True)).

const std = @import("std");
const cuda = @import("cuda");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const exl3 = cuda.exl3;
const aot = cuda.aot;
const Stream = cuda.Stream;
const Model = fwd.Model;

pub const EXPERTS_MIN = 256; // moe.PROMPT_EXPERTS_MIN
pub const EXPERTS_FROM = 3072; // chunk rows from which experts run per expert on the prompt GEMM, else grouped
pub const GROUPED_ROWS = 1024; // the grouped experts kernel's window under EXPERTS_FROM
pub const TOKENS_PER_UPDATE = 1024; // dflash.TOKENS_PER_UPDATE
pub const HAD_SCALE: f32 = 0.08838834764831845; // prefill.HAD_SCALE, as the fp32 constexpr
const BM_GEMM = 128; // prefill.tiles
const PROMPT_BM = 8; // attention.BM

const p = aot.ptr;
fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}
fn ii(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}
fn u(x: usize) u32 {
    return @intCast(x);
}
fn pow2(x: usize) usize {
    return std.math.ceilPowerOfTwo(usize, x) catch unreachable;
}

pub const Prompt = struct {
    m: *Model,
    gpa: std.mem.Allocator,
    cap: usize, // rows a chunk
    ar: load_.Arena,
    f_count: cuda.Function,
    f_order: cuda.Function,
    f_gather: cuda.Function,
    f_combine: cuda.Function,
    // rows [cap]
    ids: u64,
    pos: u64,
    slot: u64,
    res: u64,
    x: u64,
    q: u64,
    k: u64,
    v: u64,
    qo: u64,
    att: u64,
    po: u64,
    pm: u64,
    pl: u64,
    xh: u64,
    part: u64,
    recv: u64,
    summed: u64,
    logits: u64,
    pick: u64,
    wts: u64,
    g: u64,
    up: u64,
    h: u64,
    wq: u64, // fp16 [max K * N]: W_q decoded a call
    had: u64, // bf16 [128, 128]
    // per-expert prompt MoE: copies [cap * k]
    counts: u64,
    starts: u64,
    order: u64,
    inv: u64,
    xs: u64,
    ws: u64,
    eg: u64,
    eu: u64,
    eh: u64,
    ed: u64,
    kd: u64 = 0, // kv8: a full layer's fp16 keys [context, HK, 192] / values for the prompt kernel
    vd: u64 = 0,
    sc: exl3.Scratch, // the grouped kernel's, for chunks under EXPERTS_FROM rows (in windows of sc.rows)
    taps: u64, // drafter: the chunk's last TOKENS_PER_UPDATE rows' taps
    h_ids: []i64,
    h_pos: []i32,
    h_slot: []i32,
    h_counts: []i32,
    h_starts: []i32,

    pub fn init(m: *Model, gpa: std.mem.Allocator, cap: usize, s: Stream) !*Prompt {
        const w = m.w;
        const cfg = w.cfg;
        const pr = try gpa.create(Prompt);
        errdefer gpa.destroy(pr);
        pr.* = undefined;
        pr.m = m;
        pr.gpa = gpa;
        pr.cap = cap;
        pr.ar = .{ .d = m.d, .gpa = gpa };
        errdefer pr.ar.deinit();
        pr.f_count = try m.glue.function("mimo_moe_count");
        pr.f_order = try m.glue.function("mimo_moe_order");
        pr.f_gather = try m.glue.function("mimo_moe_gather");
        pr.f_combine = try m.glue.function("mimo_moe_combine");
        const ar = &pr.ar;
        const R = cap;
        const D = cfg.hidden;
        const H = cfg.heads;
        const hk = @max(cfg.kv_full, cfg.kv_swa);
        const nd = cfg.dense_width / w.world;
        const nch = (cfg.window + fwd.CK - 1) / fwd.CK + 1;
        var max_k: usize = @max(D, H * 128);
        max_k = @max(max_k, nd);
        var max_kn: usize = 0;
        var ex: ?exl3.Experts = null;
        for (w.layers) |L| {
            inline for (.{ "q", "k", "v", "o" }) |f| max_kn = @max(max_kn, @field(L, f).k * @field(L, f).n);
            inline for (.{ "gate", "up", "down" }) |f| if (@field(L, f)) |l| {
                max_kn = @max(max_kn, l.k * l.n);
            };
            if (L.experts) |e| {
                max_kn = @max(max_kn, e.dims * e.width);
                if (ex == null) ex = e;
            }
        }
        const e0 = ex orelse return error.NoMoE;
        const I = e0.width;
        const K8 = cfg.top_k;
        pr.ids = try ar.zeros(R * 8);
        pr.pos = try ar.zeros(R * 4);
        pr.slot = try ar.zeros(R * 4);
        pr.res = try ar.zeros(R * D * 4);
        pr.x = try ar.zeros(R * D * 2);
        pr.q = try ar.zeros(R * H * 192 * 2);
        pr.k = try ar.zeros(R * hk * 192 * 2);
        pr.v = try ar.zeros(R * hk * 192 * 2);
        pr.qo = try ar.zeros(R * H * 192 * 2);
        pr.att = try ar.zeros(R * H * 128 * 2);
        pr.po = try ar.zeros(R * nch * H * 128 * 4);
        pr.pm = try ar.zeros(R * nch * H * 4);
        pr.pl = try ar.zeros(R * nch * H * 4);
        pr.xh = try ar.zeros(R * max_k * 2);
        pr.part = try ar.zeros(R * D * 4);
        pr.recv = try ar.zeros(2 * R * D * 4);
        pr.summed = try ar.zeros(R * D * 4);
        pr.logits = try ar.zeros(R * cfg.experts * 4);
        pr.pick = try ar.zeros(R * K8 * 4);
        pr.wts = try ar.zeros(R * K8 * 4);
        pr.g = try ar.zeros(R * nd * 2);
        pr.up = try ar.zeros(R * nd * 2);
        pr.h = try ar.zeros(R * nd * 2);
        pr.wq = try ar.zeros(max_kn * 2);
        pr.counts = try ar.zeros(cfg.experts * 4);
        pr.starts = try ar.zeros(cfg.experts * 4);
        pr.order = try ar.zeros(R * K8 * 4);
        pr.inv = try ar.zeros(R * K8 * 4);
        pr.xs = try ar.zeros(R * K8 * D * 2);
        pr.ws = try ar.zeros(R * K8 * 4);
        pr.eg = try ar.zeros(R * K8 * I * 2);
        pr.eu = try ar.zeros(R * K8 * I * 2);
        pr.eh = try ar.zeros(R * K8 * I * 2);
        pr.ed = try ar.zeros(R * K8 * D * 4);
        // prefill.Workspace.hadamard: +-1 by the parity of popcount(i & j), bf16
        var hb: [128 * 128]u16 = undefined;
        for (0..128) |i| for (0..128) |j| {
            hb[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xBF80 else 0x3F80;
        };
        pr.had = try ar.upload(std.mem.sliceAsBytes(&hb));
        const rows_g = @min(GROUPED_ROWS, R);
        const cfg_gu = try exl3.defaultConfig(e0.dims, e0.width, true);
        const cfg_d = try exl3.defaultConfig(e0.width, e0.dims, false);
        const sz = exl3.Scratch.bytes(e0, rows_g, K8, cfg_gu, cfg_d);
        pr.sc = .{ .xg = try ar.zeros(sz.xg), .xu = try ar.zeros(sz.xg), .xd = try ar.zeros(sz.xd), .z = try ar.zeros(sz.z), .y = try ar.zeros(sz.y), .ids = try ar.zeros(sz.ids), .count = try ar.zeros(4), .members = try ar.alloc(sz.members), .rows = rows_g, .slots = K8, .cfg_gu = cfg_gu, .cfg_d = cfg_d };
        try m.d.check(m.d.api.cuMemsetD32_v2(pr.sc.members, 0xFFFF_FFFF, sz.members / 4), "members -1");
        if (m.kv8 != null) {
            pr.kd = try ar.zeros(m.context * cfg.kv_full * 192 * 2);
            pr.vd = try ar.zeros(m.context * cfg.kv_full * 128 * 2);
        }
        pr.taps = if (m.dr) |dr| try ar.zeros(@min(R, TOKENS_PER_UPDATE) * dr.taps.len * D * 2) else 0; // fp16, as the tap buffer
        pr.h_ids = try gpa.alloc(i64, R);
        pr.h_pos = try gpa.alloc(i32, R);
        pr.h_slot = try gpa.alloc(i32, R);
        pr.h_counts = try gpa.alloc(i32, cfg.experts);
        pr.h_starts = try gpa.alloc(i32, cfg.experts);
        try s.synchronize();
        return pr;
    }

    pub fn deinit(pr: *Prompt) void {
        pr.gpa.free(pr.h_ids);
        pr.gpa.free(pr.h_pos);
        pr.gpa.free(pr.h_slot);
        pr.gpa.free(pr.h_counts);
        pr.gpa.free(pr.h_starts);
        pr.ar.deinit();
        pr.gpa.destroy(pr);
    }

    pub fn bytes(pr: *const Prompt) usize {
        return pr.ar.total;
    }

    /// prefill.matmul: out [n, N] (row stride N) = x [n, K] fp16 @ W + bias, the prompt path's arithmetic.
    fn gemm(pr: *Prompt, s: Stream, l: exl3.Linear, stored: bool, x: u64, out: u64, out_f32: bool, n: usize) !void {
        const m = pr.m;
        try m.k3.linearRotIn(s, x, .f16, l.suh, pr.xh, n, l.k);
        const st = if (stored) exl3.Kernels.storedStrides(l) else l.strides();
        try m.k3.unpackStrided(s, l, pr.wq, st);
        try m.set.run(s, "_gemm", .{ u(((n + BM_GEMM - 1) / BM_GEMM) * (l.n / 128)), 1, 1 }, &.{
            p("X", "*fp16", pr.xh),   p("W", "*fp16", pr.wq),                                 p("H", "*bf16", pr.had),
            p("SVH", "*fp16", l.svh), p("BIAS", "*fp16", if (l.bias != 0) l.bias else l.svh), p("OUT", if (out_f32) "*fp32" else "*fp16", out),
            ii("M", n),               ii("o_stride", l.n),
        }, &.{ ci("K", l.k), ci("N", l.n), ci("BM", BM_GEMM), ci("BK", 32), ci("GROUP", 8), ci("HAS_BIAS", @intFromBool(l.bias != 0)), aot.cf("SCALE", HAD_SCALE) });
    }

    fn rms(pr: *Prompt, s: Stream, x: u64, w: u64, out: u64, n: usize) !void {
        const D = pr.m.w.cfg.hidden;
        try pr.m.set.run(s, "_rms", .{ u(n), 1, 1 }, &.{ p("X", "*fp32", x), p("W", "*fp32", w), p("OUT", "*fp16", out) }, &.{ ci("N", D), aot.cf("EPS", @floatCast(pr.m.w.cfg.eps)), ci("BLOCK", pow2(D)) });
    }

    fn addRms(pr: *Prompt, s: Stream, x: u64, w: u64, n: usize) !void {
        const D = pr.m.w.cfg.hidden;
        try pr.m.set.run(s, "_add_rms", .{ u(n), 1, 1 }, &.{ p("X", "*fp32", x), p("RES", "*fp32", pr.res), p("W", "*fp32", w), p("OUT", "*fp16", pr.x) }, &.{ ci("N", D), aot.cf("EPS", @floatCast(pr.m.w.cfg.eps)), ci("BLOCK", pow2(D)) });
    }

    /// forward._sum over the chunk's rows.
    fn sum(pr: *Prompt, s: Stream, part: u64, n: usize) !u64 {
        const m = pr.m;
        const c = m.comm orelse return part;
        const D = m.w.cfg.hidden;
        try c.allGather(part, pr.recv, n * D, .f32, s.handle);
        try m.add2(s, pr.recv, pr.recv + n * D * 4, pr.summed, n * D);
        return pr.summed;
    }

    fn launch(f: cuda.Function, s: Stream, gx: usize, threads: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(gx), .y = 1, .z = 1 }, .block = .{ .x = u(threads), .y = 1, .z = 1 } }, s, a);
    }

    /// moe.MoE._prompt: per expert on the prompt GEMM; the weighted sum of a row's 8 into pr.part (fp32).
    fn experts(pr: *Prompt, s: Stream, e: exl3.Experts, scale: f32, n: usize) !void {
        const m = pr.m;
        const d = m.d;
        const E = e.count;
        const D = e.dims;
        const I = e.width;
        const K8 = m.w.cfg.top_k;
        const nk = n * K8;
        try d.check(d.api.cuMemsetD32Async(pr.counts, 0, E, s.handle), "counts 0");
        {
            var a: cuda.Args = .{};
            a.add(pr.pick);
            a.add(@as(c_int, @intCast(nk)));
            a.add(pr.counts);
            try launch(pr.f_count, s, @min((nk + 255) / 256, 1024), 256, &a);
        }
        try s.synchronize(); // torch.bincount(...).tolist(): one host sync a layer
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(pr.h_counts.ptr), pr.counts, E * 4), "counts");
        var at: i32 = 0;
        for (pr.h_counts, pr.h_starts) |c, *st| {
            st.* = at;
            at += c;
        }
        try d.check(d.api.cuMemcpyHtoD_v2(pr.starts, @ptrCast(pr.h_starts.ptr), E * 4), "starts");
        {
            var a: cuda.Args = .{};
            a.add(pr.pick);
            a.add(@as(c_int, @intCast(nk)));
            a.add(pr.starts);
            a.add(pr.order);
            try launch(pr.f_order, s, E, 1024, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(pr.x);
            a.add(pr.order);
            a.add(pr.wts);
            a.add(@as(f32, 1.0) / scale); // torch: x / scalar as x * (1 / scalar) in fp32
            a.add(@as(c_int, @intCast(K8)));
            a.add(@as(c_int, @intCast(D)));
            a.add(pr.xs);
            a.add(pr.ws);
            a.add(pr.inv);
            try launch(pr.f_gather, s, nk, 256, &a);
        }
        for (0..E) |ei| {
            const c: usize = @intCast(pr.h_counts[ei]);
            if (c == 0) continue;
            const o: usize = @intCast(pr.h_starts[ei]);
            try pr.gemm(s, e.linear(ei, 0), true, pr.xs + o * D * 2, pr.eg + o * I * 2, false, c);
            try pr.gemm(s, e.linear(ei, 1), true, pr.xs + o * D * 2, pr.eu + o * I * 2, false, c);
            try m.set.run(s, "_silu_mul_scale", .{ u(c), 1, 1 }, &.{
                p("G", "*fp16", pr.eg + o * I * 2), p("U", "*fp16", pr.eu + o * I * 2), p("W", "*fp32", pr.ws + o * 4), p("OUT", "*fp16", pr.eh + o * I * 2),
            }, &.{ ci("N", I), ci("BLOCK", pow2(I)) });
            try pr.gemm(s, e.linear(ei, 2), true, pr.eh + o * I * 2, pr.ed + o * D * 4, true, c);
        }
        {
            var a: cuda.Args = .{};
            a.add(pr.ed);
            a.add(pr.inv);
            a.add(@as(c_int, @intCast(D)));
            a.add(scale);
            a.add(@as(c_int, @intFromBool(scale != 1.0)));
            a.add(pr.part);
            try launch(pr.f_combine, s, n, 256, &a);
        }
    }

    /// One prompt chunk at [p0, p0 + n) of `slot`: caches written, its last row's logits in the decode buffers.
    pub fn chunk(pr: *Prompt, s: Stream, slot: u32, p0: usize, tokens: []const i64, cap: ?Model.Capture) !void {
        const m = pr.m;
        const cfg = m.w.cfg;
        const D = cfg.hidden;
        const H = cfg.heads;
        const n = tokens.len;
        if (n == 0 or n > pr.cap) return error.ChunkTooWide;
        if (n > m.ring - cfg.window) return error.RingOverrun;
        if (p0 + n > m.context) return error.PastContext;
        if (slot >= m.slots) return error.BadSlot;
        for (tokens, 0..) |t, j| {
            pr.h_ids[j] = t;
            pr.h_pos[j] = @intCast(p0 + j);
            pr.h_slot[j] = @intCast(slot);
        }
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.ids, @ptrCast(pr.h_ids.ptr), n * 8), "ids");
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.pos, @ptrCast(pr.h_pos.ptr), n * 4), "pos");
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.slot, @ptrCast(pr.h_slot.ptr), n * 4), "slot");
        const dr = if (cap == null) m.dr else null;
        const keep = @min(n, TOKENS_PER_UPDATE);
        {
            var a: cuda.Args = .{};
            a.add(m.w.embed);
            a.add(pr.ids);
            a.add(pr.res);
            a.add(@as(c_int, @intCast(D)));
            try launch(m.f_embed, s, n, 256, &a);
        }
        try pr.rms(s, pr.res, m.w.layers[0].input_norm, pr.x, n);
        for (m.w.layers, 0..) |L, i| {
            const hk = cfg.kvHeads(i);
            try pr.gemm(s, L.q, false, pr.x, pr.q, false, n);
            try pr.gemm(s, L.k, false, pr.x, pr.k, false, n);
            try pr.gemm(s, L.v, false, pr.x, pr.v, false, n);
            const full = cfg.full[i];
            const len = if (full) m.context else m.ring;
            if (full and m.kv8 != null) try m.kv8.?.qkv(m, s, i, pr.q, pr.k, pr.v, pr.pos, pr.slot, pr.qo, n) else try m.set.run(s, "_qkv", .{ u(n), u(H + 2 * hk), 1 }, &.{
                p("Q", "*fp16", pr.q),                                  p("K", "*fp16", pr.k),                                  p("V", "*fp16", pr.v),
                p("COS", "*fp32", if (full) m.cos_full else m.cos_swa), p("SIN", "*fp32", if (full) m.sin_full else m.sin_swa), p("POS", "*i32", pr.pos),
                p("SLOT", "*i32", pr.slot),                             p("QO", "*fp16", pr.qo),                                p("KC", "*fp16", m.kc[i]),
                p("VC", "*fp16", m.vc[i]),                              ii("LEN", len),
            }, &.{ ci("H", H), ci("HK", hk), ci("RD", cfg.rope_dim) });
            const g = H / hk;
            if (full) {
                // attention.prompt over the slot's whole cache (8-bit: its keys so far back to fp16 first)
                if (m.kv8) |*k8| try k8.dequant(m, s, i, slot, p0 + n, pr.kd, pr.vd);
                const kc = if (m.kv8 != null) pr.kd else m.kc[i] + @as(u64, slot) * m.context * hk * 192 * 2;
                const vc = if (m.kv8 != null) pr.vd else m.vc[i] + @as(u64, slot) * m.context * hk * 128 * 2;
                try m.set.run(s, "_prompt", .{ u((n + PROMPT_BM - 1) / PROMPT_BM), u(hk), 1 }, &.{
                    p("Q", "*fp16", pr.qo), p("KC", "*fp16", kc), p("VC", "*fp16", vc), p("OUT", "*fp16", pr.att),
                    ii("P0", p0),           ii("W", n),
                }, &.{ ci("H", H), ci("HK", hk), ci("G", g), aot.cf("SCALE", m.scale), ci("BM", PROMPT_BM), ci("BN", fwd.BN) });
            } else {
                const nch: usize = (cfg.window + fwd.CK - 1) / fwd.CK + 1;
                try m.set.run(s, "_rows", .{ u(n), u(hk), u(nch) }, &.{
                    p("Q", "*fp16", pr.qo),   p("KC", "*fp16", m.kc[i]),  p("VC", "*fp16", m.vc[i]),
                    p("POS", "*i32", pr.pos), p("SLOT", "*i32", pr.slot), p("PO", "*fp32", pr.po),
                    p("PM", "*fp32", pr.pm),  p("PL", "*fp32", pr.pl),    ii("LEN", len),
                    ii("NCH", nch),
                }, &.{ ci("H", H), ci("HK", hk), ci("G", g), ci("WIN", cfg.window), aot.cf("SCALE", m.scale), ci("CK", fwd.CK), ci("BN", fwd.BN), ci("GP", fwd.GP) });
                const sinks = L.sinks;
                try m.set.run(s, "_merge", .{ u(n), u(H), 1 }, &.{
                    p("PO", "*fp32", pr.po),                               p("PM", "*fp32", pr.pm),   p("PL", "*fp32", pr.pl),
                    p("SINKS", "*fp32", if (sinks != 0) sinks else pr.pm), p("OUT", "*fp16", pr.att), ii("NCH", nch),
                }, &.{ ci("H", H), ci("HAS_SINK", @intFromBool(sinks != 0)) });
            }
            if (cap) |c| if (c.hook) |hf| try hf(c.ctx, .att, i, pr.att, n * H * 128 * 2);
            try pr.gemm(s, L.o, false, pr.att, pr.part, true, n);
            try pr.addRms(s, try pr.sum(s, pr.part, n), L.post_norm, n);
            if (cap) |c| if (c.hook) |hf| try hf(c.ctx, .attn, i, pr.res, n * D * 4);
            if (L.experts) |e| {
                const bm = fwd.routerTile(n);
                const be: usize, const bk: usize = if (bm == 16) .{ 32, 256 } else .{ 64, 64 };
                try m.set.run(s, "_router", .{ u((n + bm - 1) / bm), u((cfg.experts + be - 1) / be), 1 }, &.{
                    p("X", "*fp16", pr.x), p("W", "*fp16", L.router), p("OUT", "*fp32", pr.logits), ii("M", n), ii("x_stride", D),
                }, &.{ ci("D", D), ci("NE", cfg.experts), ci("BM", bm), ci("BLOCK_E", be), ci("BK", bk) });
                try m.set.run(s, "_topk", .{ u(n), 1, 1 }, &.{
                    p("L", "*fp32", pr.logits), p("BIAS", "*fp32", L.bias), p("PICK", "*i32", pr.pick), p("WTS", "*fp32", pr.wts), aot.float("SCALE", L.route_scale),
                }, &.{ ci("NE", cfg.experts), ci("TOPK", cfg.top_k), ci("BLOCK", pow2(cfg.experts)), ci("KP", pow2(cfg.top_k)) });
                if (n >= EXPERTS_FROM) {
                    try pr.experts(s, e, L.route_scale, n);
                } else {
                    // the grouped kernel reads the trellis directly; rows are independent, so windows change no bits
                    var w0: usize = 0;
                    while (w0 < n) : (w0 += pr.sc.rows) {
                        const k = @min(pr.sc.rows, n - w0);
                        try m.k3.routed(s, pr.x + w0 * D * 2, false, D, pr.pick + w0 * cfg.top_k * 4, pr.wts + w0 * cfg.top_k * 4, e, pr.sc, pr.part + w0 * D * 4, k, std.math.inf(f32), 1, true);
                    }
                }
            } else {
                try pr.gemm(s, L.gate.?, false, pr.x, pr.g, false, n);
                try pr.gemm(s, L.up.?, false, pr.x, pr.up, false, n);
                const nd = L.gate.?.n;
                try m.set.run(s, "_silu_mul", .{ u(n), u(nd / 1024), 1 }, &.{ p("G", "*fp16", pr.g), p("U", "*fp16", pr.up), p("OUT", "*fp16", pr.h) }, &.{ ci("N", nd), ci("BLOCK", 1024) });
                try pr.gemm(s, L.down.?, false, pr.h, pr.part, true, n);
            }
            const after = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].input_norm else m.w.norm;
            try pr.addRms(s, try pr.sum(s, pr.part, n), after, n);
            if (cap) |c| if (c.hook) |hf| try hf(c.ctx, .mlp, i, pr.res, n * D * 4);
            if (dr) |dd| try dd.tapInto(s, i, pr.res + (n - keep) * D * 4, keep, pr.taps);
        }
        if (dr) |dd| {
            const row = dd.taps.len * D * 2;
            var w0: usize = 0;
            while (w0 < keep) : (w0 += fwd.DECODE_ROWS) {
                const k = @min(fwd.DECODE_ROWS, keep - w0);
                const at = n - keep + w0;
                try dd.updateFrom(s, pr.taps + w0 * row, pr.pos + at * 4, pr.slot + at * 4, k);
            }
        }
        // _head on the last row: the decode GEMV, gathered into the model's decode buffers
        try m.k3.linear(s, m.w.head, pr.x + (n - 1) * D * 2, .f16, m.b.xh, m.b.head, .f32, if (m.w.head.split[0] > 1) m.b.z else 0, 1);
        if (m.comm) |c| try c.allGather(m.b.head, m.b.recv, m.w.head.n, .f32, s.handle);
    }
};

/// Python's lone-stream chunks cut to at most `cap` rows, parts a multiple of 16 rows and none under EXPERTS_MIN.
pub fn plan(gpa: std.mem.Allocator, start: usize, len: usize, cap: usize) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    var a = start;
    while (a < len) {
        const b = @min(len, a + fwd.PROMPT_CHUNK);
        const c = b - a;
        if (c <= cap) {
            try out.append(gpa, c);
        } else {
            const parts = (c + cap - 1) / cap;
            const part = (c + parts - 1) / parts;
            const p16 = (part + 15) / 16 * 16;
            var left = c;
            for (0..parts - 1) |_| {
                try out.append(gpa, p16);
                left -= p16;
            }
            if (left < EXPERTS_MIN) return error.ChunkPlan;
            try out.append(gpa, left);
        }
        a = b;
    }
    return out.toOwnedSlice(gpa);
}

test "prompt chunk plan" {
    const g = std.testing.allocator;
    const a = try plan(g, 0, 40, 4096);
    defer g.free(a);
    try std.testing.expectEqualSlices(usize, &.{40}, a);
    const b = try plan(g, 0, 9000, 4096);
    defer g.free(b);
    try std.testing.expectEqualSlices(usize, &.{ 3008, 3008, 2984 }, b);
    const c = try plan(g, 0, 16384 + 100, 4096);
    defer g.free(c);
    try std.testing.expectEqualSlices(usize, &.{ 4096, 4096, 4096, 4096, 100 }, c);
}
