//! Gemma 4 E4B forward on Intel Xe2 over Level Zero: windows of up to 16 rows (decode, verification, prompt).
const std = @import("std");
const ze = @import("ze");
const kb = @import("xpu_gemma4_kernels");
const dev = @import("xpu_dev.zig");
const cfg = @import("xpu_config.zig");
const load_mod = @import("xpu_load.zig");
const Dev = dev.Dev;
const Proj = load_mod.Proj;
const alloc = dev.alloc;
const off = dev.off;

pub const ROWS = 16; // rows of a decode or verification window
pub const MAXT = 256; // rows of a prompt forward at most
pub const KEYS = 128;
pub const MAXK = 64; // candidates the device keeps of a row of logits (top_k above it: the host draws from all)
const EPS: f32 = cfg.eps;
const nul: ?*anyopaque = null;

pub const Kernels = struct {
    repack: ze.H,
    embed: ze.H,
    x2h: ze.H,
    ple_tok: ze.H,
    ple_combine: ze.H,
    resid: ze.H,
    geglu: ze.H,
    ple_mid: ze.H,
    attn_prep: [2]ze.H, // [sliding, global]
    attn_part: [2]ze.H,
    attn_merge: [2]ze.H,
    argmax_part: ze.H,
    argmax: ze.H,
    dec_wide: ze.H,
    dec_wide64: ze.H,
    dec_narrow: ze.H,
    gemv256: ze.H,
    dec_wide16: ze.H,
    attn_rows: [2]ze.H,
    ngen_wide: ze.H,
    ngen_narrow: ze.H,
    topk_part: ze.H,
    topk_merge: ze.H,
};

pub const Acts = struct {
    ids: *anyopaque,
    x: *anyopaque, // fp32 [ROWS, 2560], bf16 values
    h: *anyopaque, // fp16 GEMM input [ROWS, K <= 10240]
    ahi: *anyopaque, // fp16 group sums [ROWS, K / 64]
    alo: *anyopaque,
    p: *anyopaque, // fp16 GEMM output [ROWS, N <= 20480]
    r: *anyopaque, // fp16 [ROWS, 2560]
    q: *anyopaque, // bf16 [ROWS, 8, hd]
    pm: *anyopaque, // attention tile partials: m, l [ROWS, 8, ntl], o [ROWS, 8, ntl, hd]
    pl: *anyopaque,
    po: *anyopaque,
    tok: *anyopaque, // bf16 [ROWS, 10752]: the per-layer table's rows
    pli: *anyopaque, // bf16 [ROWS, 42, 256]
    logits: *anyopaque, // fp16 [ROWS, 262144]
    pv: *anyopaque,
    pi: *anyopaque,
    out: *anyopaque, // i32 [ROWS]
    pscr: *anyopaque, // fp32 [MAXT, 2560]: slice totals of the narrow prompt GEMM
    tkv: *anyopaque, // top-k stage 1: f32 [ROWS, 256, 64], i32
    tki: *anyopaque,
    tov: *anyopaque, // top-k result: f32 [ROWS, 64], i32
    toi: *anyopaque,
};

pub const Want = enum { none, last, rows };

pub const Runner = struct {
    d: *Dev,
    k: Kernels,
    m: load_mod.Model,
    a: Acts,
    kc: [24]*anyopaque,
    vc: [24]*anyopaque,
    ckpt: dev.Checkpoint,
    context: u32,
    ntl_max: u32,
    pos: u32 = 0,
    last: u32 = 0,
    host_logits: []f32,
    dump_dir: ?[]const u8 = null,
    tag: []const u8 = "",
    window: u32 = MAXT, // rows per forward of `feed` (tests cut a prompt differently)

    pub fn init(dir: []const u8, context: u32) !*Runner {
        const cfg_path = try std.fmt.allocPrint(alloc, "{s}/config.json", .{dir});
        defer alloc.free(cfg_path);
        try cfg.admit(alloc, try dev.mapFile(cfg_path));
        const r = try alloc.create(Runner);
        errdefer alloc.destroy(r);
        r.* = undefined;
        r.ckpt = .{};
        errdefer r.ckpt.deinit();
        try r.ckpt.open(dir);
        r.d = try Dev.open(0xe223);
        errdefer r.d.deinit();
        const d = r.d;
        r.context = context;
        r.pos = 0;
        r.last = 0;
        r.window = MAXT;
        r.dump_dir = if (std.c.getenv("XPU_DUMP")) |v| std.mem.span(v) else null;
        const mg = try d.g.module(&kb.glue);
        const mgg = try d.g.module(&kb.glue_g);
        const mw = try d.g.module(&kb.ngen_wide);
        const mn = try d.g.module(&kb.ngen_narrow);
        const mm = try d.g.module(&kb.gemm);
        r.k = .{
            .repack = try d.kernel(mg, "repack", 16),
            .embed = try d.kernel(mg, "g4_embed", 256),
            .x2h = try d.kernel(mg, "g4_x2h", 256),
            .ple_tok = try d.kernel(mg, "g4_ple_tok", 256),
            .ple_combine = try d.kernel(mg, "g4_ple_combine", 16),
            .resid = try d.kernel(mg, "g4_resid", 512),
            .geglu = try d.kernel(mg, "g4_geglu", 256),
            .ple_mid = try d.kernel(mg, "g4_ple_mid", 64),
            .attn_prep = .{ try d.kernel(mg, "g4_attn_prep", 256), try d.kernel(mgg, "g4_attn_prep", 256) },
            .attn_part = .{ try d.kernel(mg, "g4_attn_part", 128), try d.kernel(mgg, "g4_attn_part", 128) },
            .attn_merge = .{ try d.kernel(mg, "g4_attn_merge", 128), try d.kernel(mgg, "g4_attn_merge", 128) },
            .argmax_part = try d.kernel(mg, "argmax_part", 256),
            .argmax = try d.kernel(mg, "argmax", 256),
            .dec_wide = try d.kernel(mm, "w4k1_decode", 64),
            .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64),
            .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
            .gemv256 = try d.kernel(mg, "g4_gemv256", 64),
            .dec_wide16 = try d.kernel(mm, "w4k1_d16", 64),
            .attn_rows = .{ try d.kernel(mg, "g4_attn_rows", 128), try d.kernel(mgg, "g4_attn_rows", 128) },
            .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
            .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256),
            .topk_part = try d.kernel(mg, "g4_topk_part", 256),
            .topk_merge = try d.kernel(mg, "g4_topk_merge", 256),
        };
        if (std.c.getenv("XPU_PROFILE") != null) try d.profOn();
        const prof = d.prof_on;
        d.prof_on = false;
        r.m = try load_mod.load(d, r.k.repack, &r.ckpt);
        d.prof_on = prof;
        r.ntl_max = (context + KEYS - 1) / KEYS + 1;
        const nt: usize = r.ntl_max;
        r.a = .{
            .ids = try d.alloc_(MAXT * 4),
            .x = try d.alloc_(MAXT * cfg.hidden * 4),
            .h = try d.alloc_(MAXT * cfg.ff * 2),
            .ahi = try d.alloc_(MAXT * (cfg.ff / 64) * 2),
            .alo = try d.alloc_(MAXT * (cfg.ff / 64) * 2),
            .p = try d.alloc_(MAXT * 2 * cfg.ff * 2),
            .r = try d.alloc_(MAXT * cfg.hidden * 2),
            .q = try d.alloc_(MAXT * cfg.heads * cfg.dim_g * 2),
            .pm = try d.alloc_(ROWS * cfg.heads * nt * 4),
            .pl = try d.alloc_(ROWS * cfg.heads * nt * 4),
            .po = try d.alloc_(ROWS * cfg.heads * nt * cfg.dim_g * 4),
            .tok = try d.alloc_(MAXT * cfg.ple_all * 2),
            .pli = try d.alloc_(MAXT * cfg.ple_all * 2),
            .pscr = try d.alloc_(MAXT * cfg.hidden * 4),
            .logits = try d.alloc_(ROWS * cfg.vocab * 2),
            .pv = try d.alloc_(ROWS * 256 * 4),
            .pi = try d.alloc_(ROWS * 256 * 4),
            .out = try d.alloc_(ROWS * 4),
            .tkv = try d.alloc_(ROWS * 256 * MAXK * 4),
            .tki = try d.alloc_(ROWS * 256 * MAXK * 4),
            .tov = try d.alloc_(ROWS * MAXK * 4),
            .toi = try d.alloc_(ROWS * MAXK * 4),
        };
        for (0..24) |i| {
            const bytes: usize = if (cfg.isGlobal(i)) @as(usize, context) * cfg.kv_heads * cfg.dim_g * 2 else cfg.ring * cfg.kv_heads * cfg.dim_s * 2;
            r.kc[i] = try d.alloc_(bytes);
            r.vc[i] = try d.alloc_(bytes);
        }
        r.host_logits = try alloc.alloc(f32, cfg.vocab);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.d.deinit();
        r.ckpt.deinit();
        alloc.free(r.host_logits);
        alloc.destroy(r);
    }

    pub fn reset(r: *Runner) void {
        r.pos = 0;
    }

    /// Feeds tokens in windows of 16 rows; `want`: the last row's logits and greedy token are kept.
    pub fn feed(r: *Runner, ids: []const u32, want: bool) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + r.window);
            if (r.pos + (end - at) > r.context) return error.ContextFull;
            var win: [MAXT]i32 = undefined;
            for (ids[at..end], 0..) |t, j| win[j] = @intCast(t);
            const w: Want = if (want and end == ids.len) .last else .none;
            if (try r.forward(win[0 .. end - at], r.pos, w)) |t| r.last = @intCast(t);
            r.pos += @intCast(end - at);
            at = end;
        }
    }

    /// The `k` (<= 64) largest logits of the last forward's row `row`, softcapped, descending (ties: lower id).
    pub fn topK(r: *Runner, row: u32, k: u32, vals: []f32, ids: []u32) !void {
        if (k == 0 or k > MAXK) return error.BadTopK;
        const d = r.d;
        const a = &r.a;
        const Kk: i32 = @intCast(k);
        // the row's logits start at row * vocab
        try d.launch(r.k.topk_part, .{ .x = 256, .y = 1 }, .{ off(a.logits, @as(usize, row) * cfg.vocab * 2), @as(i32, cfg.vocab), Kk, a.tkv, a.tki });
        try d.launch(r.k.topk_merge, .{ .x = 1 }, .{ a.tkv, a.tki, Kk, a.tov, a.toi });
        var hv: [MAXK]f32 = undefined;
        var hi: [MAXK]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &hv, a.tov, k * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &hi, a.toi, k * 4, null, 0, null));
        try d.sync();
        for (0..k) |i| {
            vals[i] = cfg.softcap * std.math.tanh(hv[i] / cfg.softcap);
            ids[i] = @intCast(hi[i]);
        }
    }

    /// The last forward's row `row` of logits after the softcap, fp32 on the host.
    pub fn logits(r: *Runner, row: u32) ![]const f32 {
        const raw = try alloc.alloc(f16, cfg.vocab);
        defer alloc.free(raw);
        try r.d.read(std.mem.sliceAsBytes(raw), off(r.a.logits, @as(usize, row) * cfg.vocab * 2));
        for (r.host_logits, raw) |*o, v| o.* = cfg.softcap * std.math.tanh(@as(f32, v) / cfg.softcap);
        return r.host_logits;
    }

    pub fn gemm(r: *Runner, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
        if (r.d.prof_on) r.d.tag = r.tag;
        defer r.d.tag = "";
        const M: i32 = @intCast(m);
        const N: i32 = @intCast(w.n);
        const K: i32 = @intCast(w.k);
        const G: i32 = @intCast(w.k / 64);
        const mp: i32 = @intCast((m + 7) / 8 * 8);
        const k = r.k;
        if (m > ROWS and w.k != 256) { // prompt rows: the nGEN GEMMs, the decode kernels' arithmetic
            const tiles = (m + 63) / 64;
            const groups: ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
            if (w.wide)
                try r.d.launch(k.ngen_wide, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try r.d.launch(k.ngen_narrow, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G, r.a.pscr });
            return;
        }
        if (w.k == 256) { // K = 256: the GEMM families' group-sum planes are too narrow for a 2D block (per-layer projection)
            try r.d.launch(k.gemv256, .{ .x = w.n / 64, .y = m }, .{ h, w.w, w.s, w.bg, out, M, N });
            return;
        }
        if (w.wide) {
            if (w.n <= 6144 and std.c.getenv("XPU_D16") != null)
                try r.d.launch(k.dec_wide16, .{ .x = 1, .y = w.n / 64 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else if (w.n >= 30000)
                try r.d.launch(k.dec_wide64, .{ .x = 1, .y = w.n / 256 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try r.d.launch(k.dec_wide, .{ .x = 1, .y = w.n / 128 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
        } else try r.d.launch(k.dec_narrow, .{ .x = 1, .y = w.n / 32 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    }

    fn dump(r: *Runner, name: []const u8, p0: u32, rows: u32) !void {
        const dir = r.dump_dir orelse return;
        if (p0 != 0) return;
        const buf = try alloc.alloc(u8, rows * cfg.hidden * 4);
        defer alloc.free(buf);
        try r.d.read(buf, r.a.x);
        const path = try std.fmt.allocPrint(alloc, "{s}/xpu_x_{s}.f32", .{ dir, name });
        defer alloc.free(path);
        try dev.writeFile(path, &.{buf});
    }

    fn dumpBuf(r: *Runner, name: []const u8, p: *anyopaque, bytes: usize) !void {
        const dir = r.dump_dir orelse return;
        const buf = try alloc.alloc(u8, bytes);
        defer alloc.free(buf);
        try r.d.read(buf, p);
        const path = try std.fmt.allocPrint(alloc, "{s}/xpu_{s}.bin", .{ dir, name });
        defer alloc.free(path);
        try dev.writeFile(path, &.{buf});
    }

    /// One forward over `ids` (at most 16 rows) at positions p0 ..; with `want`, the last row's greedy token.
    pub fn forward(r: *Runner, ids: []const i32, p0: u32, want: Want) !?i32 {
        const d = r.d;
        const a = &r.a;
        const m = &r.m;
        const k = r.k;
        const T: u32 = @intCast(ids.len);
        if (T == 0 or T > MAXT or (T > ROWS and want == .rows)) return error.BadWindow;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
        errdefer d.settle(); // the queued copy reads the caller's ids
        // the embeddings, and the per-layer inputs they give
        try d.launch(k.embed, .{ .x = T }, .{ a.ids, m.embed_w, m.embed_s, m.embed_b, a.x });
        try r.dump("embed", p0, T);
        try d.launch(k.x2h, .{ .x = T }, .{ a.x, a.h, a.ahi, a.alo });
        r.tag = "ple_model_proj";
            try r.gemm(a.h, a.ahi, a.alo, m.ple_proj, a.p, T);
        try d.launch(k.ple_tok, .{ .x = T }, .{ a.ids, m.ple_w, m.ple_s, m.ple_b, a.tok });
        try d.launch(k.ple_combine, .{ .x = T, .y = cfg.layers }, .{ a.p, a.tok, m.ple_norm, a.pli, EPS });
        try d.launch(k.resid, .{ .x = T }, .{ a.x, nul, nul, nul, m.layers[0].in_norm, a.h, a.ahi, a.alo, EPS, @as(i32, 0) });
        if (r.dump_dir != null and p0 == 0) try r.dumpBuf("pli", a.pli, T * cfg.ple_all * 2);
        for (&m.layers, 0..) |*l, i| {
            const glob = cfg.isGlobal(i);
            const hd: u32 = if (glob) cfg.dim_g else cfg.dim_s;
            const src = cfg.kvSource(i);
            const own = i < cfg.shared_from;
            r.tag = "qkv";
            try r.gemm(a.h, a.ahi, a.alo, l.qkv, a.p, T);
            try d.launch(k.attn_prep[@intFromBool(glob)], .{ .x = T }, .{ a.p, @as(i32, @intCast(l.qkv.n)), l.qn, l.kn orelse l.qn, if (glob) m.invf_g else m.invf_s, a.q, r.kc[src], r.vc[src], @as(i32, @intCast(p0)), EPS, @as(i32, if (glob) 0 else cfg.ring), @as(i32, @intFromBool(own)) });
            const ntl: u32 = if (glob) (p0 + T - 1) / KEYS + 1 else 5;
            const window: i32 = if (glob) 0 else cfg.window;
            const ring: i32 = if (glob) 0 else cfg.ring;
            if (T > ROWS) {
                try d.launch(k.attn_rows[@intFromBool(glob)], .{ .x = 8, .y = T }, .{ a.q, r.kc[src], r.vc[src], @as(i32, @intCast(p0)), window, ring, a.h, a.ahi, a.alo });
            } else {
            try d.launch(k.attn_part[@intFromBool(glob)], .{ .x = ntl, .y = 8, .z = T }, .{ a.q, r.kc[src], r.vc[src], a.pm, a.pl, a.po, @as(i32, @intCast(p0)), window, ring, @as(i32, @intCast(ntl)) });
            try d.launch(k.attn_merge[@intFromBool(glob)], .{ .x = 8, .y = T }, .{ a.pm, a.pl, a.po, @as(i32, @intCast(p0)), window, @as(i32, @intCast(ntl)), a.h, a.ahi, a.alo });
            }
            if (r.dump_dir != null and p0 == 0 and i == 0) {
                try r.dumpBuf("q0", a.q, T * 8 * hd * 2);
                try r.dumpBuf("k0", r.kc[0], T * 2 * hd * 2);
                try r.dumpBuf("v0", r.vc[0], T * 2 * hd * 2);
                try r.dumpBuf("attn0", a.h, T * 8 * hd * 2);
            }
            r.tag = "o";
            try r.gemm(a.h, a.ahi, a.alo, l.o, a.r, T);
            try d.launch(k.resid, .{ .x = T }, .{ a.x, a.r, l.post_attn_norm, nul, l.pre_ffn_norm, a.h, a.ahi, a.alo, EPS, @as(i32, 0) });
            if (r.dump_dir != null and p0 == 0 and i == 0) try r.dump("0_attn", p0, T);
            r.tag = "gate_up";
            try r.gemm(a.h, a.ahi, a.alo, l.gu, a.p, T);
            try d.launch(k.geglu, .{ .x = T, .y = cfg.ff / 1024 }, .{ a.p, a.h, a.ahi, a.alo });
            r.tag = "down";
            try r.gemm(a.h, a.ahi, a.alo, l.down, a.r, T);
            try d.launch(k.resid, .{ .x = T }, .{ a.x, a.r, l.post_ffn_norm, nul, nul, a.h, a.ahi, a.alo, EPS, @as(i32, 1) });
            if (r.dump_dir != null and p0 == 0 and i == 0) try r.dump("0_ffn", p0, T);
            r.tag = "ple_gate";
            try r.gemm(a.h, a.ahi, a.alo, l.ple_gate, a.p, T);
            try d.launch(k.ple_mid, .{ .x = T }, .{ a.p, a.pli, @as(i32, @intCast(i)), a.h, a.ahi, a.alo });
            r.tag = "ple_proj";
            try r.gemm(a.h, a.ahi, a.alo, l.ple_proj, a.r, T);
            const next = if (i + 1 < cfg.layers) m.layers[i + 1].in_norm else m.norm;
            try d.launch(k.resid, .{ .x = T }, .{ a.x, a.r, l.post_ple_norm, l.scalar, next, a.h, a.ahi, a.alo, EPS, @as(i32, 0) });
            if (r.dump_dir != null and i < 4) {
                const name = try std.fmt.allocPrint(alloc, "{d}", .{i});
                defer alloc.free(name);
                try r.dump(name, p0, T);
            }
        }
        if (want == .none) {
            try d.sync();
            return null;
        }
        const first: u32 = if (want == .rows) 0 else T - 1;
        const rows = T - first;
        const G = cfg.hidden / 64;
        r.tag = "head";
        try r.gemm(off(a.h, first * cfg.hidden * 2), off(a.ahi, first * G * 2), off(a.alo, first * G * 2), m.head, a.logits, rows);
        try d.launch(k.argmax_part, .{ .x = 256, .y = rows }, .{ a.logits, @as(i32, cfg.vocab), a.pv, a.pi });
        try d.launch(k.argmax, .{ .x = rows }, .{ a.pv, a.pi, a.out });
        var toks: [ROWS]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, a.out, rows * 4, null, 0, null));
        try d.sync();
        return toks[rows - 1];
    }
};
