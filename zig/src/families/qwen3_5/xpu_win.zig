//! Multi-row window forward of Qwen3.8: R <= 16 tokens per pass, per-row arithmetic equal to single-row decode.

const std = @import("std");
const rt = @import("xpu").rt;
const ld = @import("xpu").loader;
const cq = @import("xpu_config.zig");
const exl3 = @import("xpu").exl3;
const qb = @import("xpu_blocks.zig");
const f16p = @import("xpu_f16pf.zig");
const al = @import("xpu_attn_long.zig");
const mx = @import("xpu_mlx4.zig");
const stop = @import("xpu").stop;

const spv_basic = @import("xpu").kernels.qwen_basic;
const spv_rows = @import("xpu").kernels.qwen_rows;
const spv_pf = @import("xpu").kernels.qwen_attn_pf;

const Buf = qb.Buf;
pub const max_rows: u32 = 16;
/// Row capacity of the next Win created (windows above max_rows are in-place prompt chunks using the prefill GEMMs).
pub var default_rows: u32 = 16;
/// default_rows = 0 chooses the prompt window from the memory left (autoRows); the window built is chosen_rows.
pub const auto_rows: u32 = 0;
pub var chosen_rows: u32 = 16;

/// Scratch a prompt window costs per row (0.5 MB budgeted, above every format's need so the choice errs small).
const window_row_bytes: u64 = 500_000;

/// Largest of 2048 / 1024 / 512 / 256 rows whose scratch fits in free device memory less a 1.5 GB reserve.
pub fn autoRows() u32 {
    const free = rt.vramFree() -| 1_500_000_000;
    inline for (.{ 2048, 1024, 512 }) |rows| {
        if (free >= @as(u64, rows) * window_row_bytes) return rows;
    }
    return 256;
}
/// head_rows value of a big window that keeps the normalised hidden rows for headBatch (e.g. KL scoring).
pub const keep_hidden: u32 = 0xffff;
/// Attention row block of a big window (bounds the split-K partial scratch).
const attn_block: u32 = 64;
const hidden = qb.hidden;
const inter = qb.inter;

pub const Mixer = union(enum) { gdn: qb.GdnW, attn: qb.AttnW };
pub const Layer = struct { mixer: Mixer, mlp: qb.MlpW };

/// Per-GDN-layer window state: the state after the window (before commit) and its projection outputs (replay inputs).
const GdnX = struct { sstate2: Buf, wqkv: Buf, wb: Buf, wa: Buf };

const kern = @import("xpu_win_kernels.zig");
const Kernels = kern.Kernels;

/// Per-kernel and per-projection timing of windows (prof_on): every launch is synced, so times include the round trip.
pub var prof_on: bool = false;
/// Env ATTN_PROF=1: synced wall time of the whole attention step of every window (q/k/v/o projections, K/V append).
pub var attn_prof: bool = false;
pub var attn_ns: u64 = 0;
const KProf = struct { h: ?*const anyopaque = null, ns: u64 = 0, calls: u64 = 0 };
var kprof: [48]KProf = @splat(.{});
const PProf = struct { rows: u32 = 0, in: u32 = 0, fmt: qb.Format = .mlx_affine4_g64, ns: u64 = 0, calls: u64 = 0 };
var pprof: [32]PProf = @splat(.{});

/// Env ARC_PF picks the prompt-window attention (unset = pfs, `old` = pf, else per-row); true if unset or == `want`.
fn pfEnv(want: ?[]const u8) bool {
    const v = std.c.getenv("ARC_PF") orelse return want == null;
    return if (want) |w| std.mem.eql(u8, std.mem.span(v), w) else false;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    if (!prof_on) return runRaw(k, groups, a);
    try k.rt.sync();
    const t0 = nowNs();
    try runRaw(k, groups, a);
    try k.rt.sync();
    const dt = nowNs() - t0;
    for (&kprof) |*e| {
        if (e.h == null) e.h = @ptrCast(k.handle);
        if (e.h == @as(?*const anyopaque, @ptrCast(k.handle))) {
            e.ns += dt;
            e.calls += 1;
            return;
        }
    }
}

fn runRaw(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    qb.touch();
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else if (T == u64) {
            var x = v;
            try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, i, 8, @ptrCast(&x)), "setU64");
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

fn at(b: Buf, off: usize) Buf {
    return exl3.at(b, off);
}

pub const Win = struct {
    o: *qb.Ops,
    r: *rt.Runtime,
    vocab: u32,
    k: Kernels,
    gx: []GdnX,
    tok: Buf,
    x: Buf,
    xn: Buf,
    delta: Buf,
    z: Buf,
    qn: Buf,
    kn: Buf,
    y: Buf,
    vconv: Buf,
    gbeta: Buf,
    ggate: Buf,
    yg: Buf,
    qg: Buf,
    kraw: Buf,
    /// New K / V rows of a quantized KV cache before they are appended (Ops.kvAppend).
    ktmp: Buf,
    vtmp: Buf,
    q: Buf,
    att: Buf,
    gate: Buf,
    up: Buf,
    act: Buf,
    logits: Buf,
    po: Buf,
    pm: Buf,
    pl: Buf,
    rope: Buf,
    am_scratch: Buf,
    am_out: Buf,
    pending: bool = false,
    /// MLX matvec family: .exact is bit-identical to decode (verify windows), .fast is systolic (prompt chunks).
    fam: mx.Family = .exact,
    /// Rows of the last window and whether its GDN state was committed in place; start position of that window.
    rows: u32 = 0,
    inplace: bool = true,
    pos0: u32 = 0,
    head_rows: u32 = 0,
    rcap: u32 = max_rows,
    ab: u32 = max_rows,
    bigx: GdnX = undefined,
    xp: ?exl3.Prefill = null,
    f16pf: ?*f16p.Pf = null, // plain fp16 projections (GDN in_proj_a/b) of a prompt chunk on the 2D-load GEMM, created on first use
    /// Host staging ring of 4 for token ids and rope table: uploads read late, so a slot is rewritten after a sync.
    host_tok: [4][]u32 = undefined,
    rope_host: [4][]f32 = undefined,
    ring: u32 = 0,
    /// Matrix-engine attention for prompt windows (qwen_attn_pf.cl), bf16 KV only; null = per-row kernels (ARC_PF=0).
    pf: ?rt.Kernel = null,
    pfs: [3]?al.Pfs = .{ null, null, null }, // matrix-engine prompt-window attention by KV cache mode (bf16, q8, q4), created on first use

    pub fn init(gpa: std.mem.Allocator, o: *qb.Ops, l: *ld.Loader, cfg: cq.Config) !Win {
        const r = o.r;
        const rcap = @max(if (default_rows == auto_rows) autoRows() else default_rows, max_rows);
        chosen_rows = rcap;
        var bm = try r.module(spv_basic);
        var rm = try r.module(spv_rows);
        const k: Kernels = .{
            .embed4 = try bm.kernel("embed4", .{ 64, 1, 1 }),
            .embed16 = try bm.kernel("embed_bf16", .{ 64, 1, 1 }),
            .swiglu = try bm.kernel("swiglu", .{ 64, 1, 1 }),
            .add = try bm.kernel("add_bf16", .{ 64, 1, 1 }),
            .round = try bm.kernel("round_bf16_f32", .{ 64, 1, 1 }),
            .am_part = try bm.kernel("argmax_partial", .{ 256, 1, 1 }),
            .am_fin = try bm.kernel("argmax_final", .{ 256, 1, 1 }),
            .add_rms = try rm.kernel("add_rmsnorm_r", .{ 512, 1, 1 }),
            .add_rms32 = try rm.kernel("add_rmsnorm_r_f32w", .{ 512, 1, 1 }),
            .rms = try rm.kernel("rmsnorm_r", .{ 512, 1, 1 }),
            .rms32 = try rm.kernel("rmsnorm_r_f32w", .{ 512, 1, 1 }),
            .g_prep = try rm.kernel("gdn_conv_prep_r", .{ 16, 1, 1 }),
            .g_step = try rm.kernel(kern.gdnStepName(), .{ 64, 1, 1 }),
            .g_vconv = try rm.kernel("gdn_vconv_r", .{ 64, 1, 1 }),
            .g_gates = try rm.kernel("gdn_gates_r", .{ 48, 1, 1 }),
            .g_norm = try rm.kernel("gdn_gate_norm_r", .{ 16, 1, 1 }),
            .g_commit = try rm.kernel("gdn_conv_commit", .{ 64, 1, 1 }),
            .mv16r = try rm.kernel("mv_f16_rows", .{ 64, 1, 1 }),
            .a_prep = try rm.kernel("attn_prep_r", .{ 256, 1, 1 }),
            .a_prep32 = try rm.kernel("attn_prep_r_f32w", .{ 256, 1, 1 }),
            .a_part = try rm.kernel("attn_partial_r", .{ 128, 1, 1 }),
            .a_merge = try rm.kernel("attn_merge_r", .{ 256, 1, 1 }),
        };
        const v: usize = cfg.text_config.vocab_size;
        const n: usize = rcap;
        const nh: usize = max_rows; // rows of the head scratch
        const ab: usize = @min(rcap, attn_block);
        const nl = cfg.text_config.num_hidden_layers;
        const gx = try gpa.alloc(GdnX, nl);
        for (gx, 0..) |*e, i| {
            if (cfg.isAttention(i)) {
                e.* = undefined;
                continue;
            }
            e.* = .{
                .sstate2 = try l.empty(@as(usize, qb.gdn_heads) * 128 * 128 * 4),
                .wqkv = try l.empty(nh * qb.gdn_qkv * 2), // windows above max_rows use bigx (sized for the whole chunk): these hold at most max_rows rows
                .wb = try l.empty(nh * qb.gdn_heads * 2),
                .wa = try l.empty(nh * qb.gdn_heads * 2),
            };
        }
        var pfk: ?rt.Kernel = null;
        if (rcap > max_rows and al.kvMode() == .bf16 and pfEnv("old")) {
            var pfm = try r.moduleWith(spv_pf, "-cl-intel-256-GRF-per-thread");
            pfk = try pfm.kernel("attn_prefill", .{ 64, 1, 1 });
        }
        const gate_buf = try l.empty(n * inter * 2);
        return .{
            .o = o,
            .r = r,
            .vocab = cfg.text_config.vocab_size,
            .k = k,
            .gx = gx,
            .rcap = rcap,
            .ab = @intCast(ab),
            .host_tok = .{ try gpa.alloc(u32, n), try gpa.alloc(u32, n), try gpa.alloc(u32, n), try gpa.alloc(u32, n) },
            .rope_host = .{ try gpa.alloc(f32, n * 64), try gpa.alloc(f32, n * 64), try gpa.alloc(f32, n * 64), try gpa.alloc(f32, n * 64) },
            .bigx = if (rcap > max_rows) .{ .sstate2 = undefined, .wqkv = try l.empty(n * qb.gdn_qkv * 2), .wb = try l.empty(n * qb.gdn_heads * 2), .wa = try l.empty(n * qb.gdn_heads * 2) } else undefined,
            .pf = pfk,
            .xp = if (rcap > max_rows and o.ex != null) try exl3.Prefill.init(r, rcap, 17408, 17408, 544) else null,
            .tok = try l.empty(n * 4),
            .x = try l.empty(n * hidden * 2),
            .xn = try l.empty(n * hidden * 2),
            .delta = try l.empty(n * hidden * 2),
            .z = try l.empty(n * qb.gdn_v * 2),
            .qn = try l.empty(n * 2048 * 4),
            .kn = try l.empty(n * 2048 * 4),
            .y = try l.empty(n * qb.gdn_v * 2),
            .vconv = try l.empty(n * qb.gdn_v * 2),
            .gbeta = try l.empty(n * qb.gdn_heads * 4),
            .ggate = try l.empty(n * qb.gdn_heads * 4),
            .yg = try l.empty(n * qb.gdn_v * 2),
            .qg = try l.empty(n * 2 * qb.q_dim * 2),
            .kraw = try l.empty(n * qb.kv_dim * 2),
            .ktmp = try l.empty(n * qb.kv_dim * 2),
            .vtmp = try l.empty(n * qb.kv_dim * 2),
            .q = try l.empty(n * qb.q_dim * 2),
            .att = try l.empty(n * qb.q_dim * 2),
            .gate = gate_buf,
            .up = try l.empty(n * inter * 2),
            .act = gate_buf, // SwiGLU is elementwise (each item reads gate[i], up[i] before it writes act[i]), so act lives in the gate buffer
            .logits = try l.empty(nh * v * 4),
            .po = try l.empty(ab * qb.max_chunks * qb.heads * qb.head_dim * 4),
            .pm = try l.empty(ab * qb.max_chunks * qb.heads * 4),
            .pl = try l.empty(ab * qb.max_chunks * qb.heads * 4),
            .rope = try l.empty(n * 64 * 4),
            .am_scratch = try l.empty(nh * qb.argmax_parts * 8),
            .am_out = try l.empty(nh * 4),
        };
    }

    fn setRope(w: *Win, pos0: u32, m: u32) !void {
        const half = 32;
        for (0..m) |row| for (0..half) |i| {
            const inv: f32 = @floatCast(std.math.pow(f64, w.o.rope_theta, -@as(f64, @floatFromInt(i)) / @as(f64, half)));
            const ang: f32 = @as(f32, @floatFromInt(pos0 + row)) * inv;
            w.rope_host[w.ring][row * 64 + i] = @cos(ang);
            w.rope_host[w.ring][row * 64 + half + i] = @sin(ang);
        };
        try w.r.upload(w.rope, std.mem.sliceAsBytes(w.rope_host[w.ring][0 .. m * 64]));
    }

    fn embed(w: *Win, t: qb.Table, tokens: []const u32) !void {
        const m: u32 = @intCast(tokens.len);
        @memcpy(w.host_tok[w.ring][0..tokens.len], tokens);
        try w.r.upload(w.tok, std.mem.sliceAsBytes(w.host_tok[w.ring][0..tokens.len]));
        switch (t.format) {
            .mlx_affine4_g64 => try run(&w.k.embed4, .{ m, 1, 1 }, .{ t.w, t.s, t.b, w.tok, w.x, hidden }),
            .bf16 => try run(&w.k.embed16, .{ m, 1, 1 }, .{ t.w, w.tok, w.x, hidden }),
            .f16, .exl3 => return error.Invalid,
            else => try w.o.gg.?.embedRows(qb.ggType(t.format).?, t.w, w.tok, w.x, hidden, m),
        }
        w.pending = false;
    }

    /// The norm in front of a block for all rows, folding the previous block's pending residual add into one kernel.
    fn normIn(w: *Win, wt: Buf, m: u32) !void {
        const f32n = w.o.f32_norm;
        if (!w.pending) return run(if (f32n) &w.k.rms32 else &w.k.rms, .{ m, 1, 1 }, .{ w.x, wt, w.xn, hidden, w.o.eps });
        w.pending = false;
        try run(if (f32n) &w.k.add_rms32 else &w.k.add_rms, .{ m, 1, 1 }, .{ w.x, w.delta, wt, w.xn, hidden, w.o.eps });
    }

    /// y[row][..] = bf16(W x[row]) for m rows (y_off elements into y).
    pub fn proj(w: *Win, t: qb.Linear, x: Buf, m: u32, y: Buf, y_off: u32) !void {
        if (!prof_on) return w.projInner(t, x, m, y, y_off);
        try w.r.sync();
        const t0 = nowNs();
        try w.projInner(t, x, m, y, y_off);
        try w.r.sync();
        const dt = nowNs() - t0;
        for (&pprof) |*e| {
            if (e.calls == 0) e.* = .{ .rows = t.rows, .in = t.in, .fmt = t.format };
            if (e.rows == t.rows and e.in == t.in and e.fmt == t.format) {
                e.ns += dt;
                e.calls += 1;
                return;
            }
        }
    }

    /// Prints the window profile (kernels and projections), in ms for the whole run and per `tokens`.
    pub fn printProfile(w: *Win, tokens: usize) void {
        var proj_ns: u64 = 0;
        for (pprof) |e| if (e.calls > 0) {
            proj_ns += e.ns;
            std.debug.print("  proj {s:<10} [{d:>6} x {d:>5}] {d:>5} calls {d:>9.1} ms\n", .{ @tagName(e.fmt), e.rows, e.in, e.calls, @as(f64, @floatFromInt(e.ns)) / 1e6 });
        };
        var k_ns: u64 = 0;
        inline for (@typeInfo(Kernels).@"struct".field_names) |fname| {
            for (kprof) |e| if (e.h != null and e.h == @as(?*const anyopaque, @ptrCast(@field(w.k, fname).handle))) {
                k_ns += e.ns;
                std.debug.print("  kernel {s:<9} {d:>6} calls {d:>9.1} ms\n", .{ fname, e.calls, @as(f64, @floatFromInt(e.ns)) / 1e6 });
            };
        }
        std.debug.print("  projections {d:.1} ms, kernels {d:.1} ms over {d} tokens\n", .{ @as(f64, @floatFromInt(proj_ns)) / 1e6, @as(f64, @floatFromInt(k_ns)) / 1e6, tokens });
    }

    fn projInner(w: *Win, t: qb.Linear, x: Buf, m: u32, y: Buf, y_off: u32) !void {
        if (m > max_rows) return w.projBig(t, x, m, y, y_off);
        switch (t.format) {
            .f16 => for (0..m) |i| try w.o.matvecRaw(t, at(x, i * t.in * 2), at(y, i * t.rows * 2), y_off, .bf16),
            else => try w.o.matvecRows(t, x, m, y, y_off, w.fam),
        }
    }

    /// Prompt-chunk projections (m > 16 rows): GGUF and EXL3 use their prefill GEMMs, MLX goes through 16-row windows.
    fn projBig(w: *Win, t: qb.Linear, x: Buf, m: u32, y: Buf, y_off: u32) !void {
        const o = w.o;
        switch (t.format) {
            // the exact decode kernel row by row below the prefill-GEMM threshold (keeps prompts equal to decode)
            .f16 => if (m >= 256 and f16p.Pf.ok(t.rows, t.in) and std.c.getenv("F16_NOPF") == null) {
                if (w.f16pf == null) {
                    const p = try std.heap.page_allocator.create(f16p.Pf);
                    p.* = try f16p.Pf.init(o.r);
                    w.f16pf = p;
                }
                try w.f16pf.?.run(o.r, t.w, x, m, y, t.in, y_off, t.rows);
            } else if (m >= 256) try run(&w.k.mv16r, .{ (t.rows + 3) / 4, m, 1 }, .{ t.w, x, y, t.in, y_off, t.rows }) else for (0..m) |i| try o.matvecRaw(t, at(x, i * t.in * 2), at(y, i * t.rows * 2), y_off, .bf16),
            .bf16 => try o.matvecRows(t, x, m, y, y_off, w.fam),
            .exl3 => if (m >= 256 and w.xp != null) {
                try t.ex.?.forwardPrefill(&o.ex.?.eng, &w.xp.?, x, .bf16, m, at(y, @as(usize, y_off) * 2), .bf16);
            } else try w.chunks16(t, x, m, y, y_off),
            .mlx_affine4_g64 => if (m >= 64 and t.rows % 16 == 0 and t.in % 512 == 0 and std.c.getenv("MLX4_NOPF") == null) try o.prefillRows(t, x, m, y, y_off) else try w.chunks16(t, x, m, y, y_off),
            else => if (m >= 32) try o.prefillRows(t, x, m, y, y_off) else try o.matvecRows(t, x, m, y, y_off, w.fam),
        }
    }

    fn chunks16(w: *Win, t: qb.Linear, x: Buf, m: u32, y: Buf, y_off: u32) !void {
        var d: u32 = 0;
        while (d < m) {
            const n = @min(max_rows, m - d);
            try w.o.matvecRows(t, at(x, @as(usize, d) * t.in * 2), n, at(y, @as(usize, d) * t.rows * 2), y_off, w.fam);
            d += n;
        }
    }

    /// GGUF projections that have a fused multi-row kernel (not IQ1_M, Q6_K).
    fn ggFuse(w: *Win, f: qb.Format) bool {
        return w.o.gg != null and !prof_on and qb.ggType(f) != null and f != .iq1_m and f != .q6_k and f != .q5_k and f != .q8_0 and f != .iq4_nl;
    }

    /// qkv + z of a window: on a GGUF one fused launch for up to 8 rows (same bits as decode), else two projections.
    fn gdnProj(w: *Win, g: qb.GdnW, gx: GdnX, m: u32) !void {
        if (m <= max_rows and w.ggFuse(g.qkv.format) and w.ggFuse(g.z.format)) {
            var d: u32 = 0;
            while (d < m) : (d += 8) {
                const n = @min(8, m - d);
                try w.o.gg.?.matvecMultiRows(&.{
                    .{ .t = qb.ggType(g.qkv.format).?, .w = g.qkv.w, .y = gx.wqkv, .rows = g.qkv.rows, .y_off = d * g.qkv.rows },
                    .{ .t = qb.ggType(g.z.format).?, .w = g.z.w, .y = w.z, .rows = g.z.rows, .y_off = d * g.z.rows },
                }, at(w.xn, @as(usize, d) * hidden * 2), hidden, n);
            }
        } else {
            try w.proj(g.qkv, w.xn, m, gx.wqkv, 0);
            try w.proj(g.z, w.xn, m, w.z, 0);
        }
    }

    /// q, k, v of a window (v into the KV cache at pos0): fused for GGUF as in gdnProj.
    fn attnProj(w: *Win, a: qb.AttnW, m: u32, pos0: u32) !void {
        const quant = al.kvMode() != .bf16; // a quantized cache takes the new rows through w.vtmp
        const vdst = if (quant) w.vtmp else a.vc;
        const voff: u32 = if (quant) 0 else pos0 * qb.kv_dim;
        if (m <= max_rows and w.ggFuse(a.q.format) and w.ggFuse(a.k.format) and w.ggFuse(a.v.format)) {
            var d: u32 = 0;
            while (d < m) : (d += 8) {
                const n = @min(8, m - d);
                try w.o.gg.?.matvecMultiRows(&.{
                    .{ .t = qb.ggType(a.q.format).?, .w = a.q.w, .y = w.qg, .rows = a.q.rows, .y_off = d * a.q.rows },
                    .{ .t = qb.ggType(a.k.format).?, .w = a.k.w, .y = w.kraw, .rows = a.k.rows, .y_off = d * a.k.rows },
                    .{ .t = qb.ggType(a.v.format).?, .w = a.v.w, .y = vdst, .rows = a.v.rows, .y_off = voff + d * qb.kv_dim },
                }, at(w.xn, @as(usize, d) * hidden * 2), hidden, n);
            }
        } else {
            try w.proj(a.q, w.xn, m, w.qg, 0);
            try w.proj(a.k, w.xn, m, w.kraw, 0);
            try w.proj(a.v, w.xn, m, vdst, voff);
        }
    }

    fn gdn(w: *Win, g: qb.GdnW, gx: GdnX, m: u32, inplace: bool) !void {
        try w.normIn(g.norm, m);
        try w.gdnProj(g, gx, m);
        try w.proj(g.b, w.xn, m, gx.wb, 0);
        try w.proj(g.a, w.xn, m, gx.wa, 0);
        try w.gdnCore(g, gx, m, inplace, if (inplace) g.sstate else gx.sstate2);
        try run(&w.k.g_norm, .{ qb.gdn_heads, m, 1 }, .{ w.y, w.z, g.gnorm, w.yg, w.o.eps });
        try w.proj(g.out, w.yg, m, w.delta, 0);
        w.pending = true;
    }

    /// conv + q/k prep and the delta rule over m rows from g.sstate into s_out.
    fn gdnCore(w: *Win, g: qb.GdnW, gx: GdnX, m: u32, inplace: bool, s_out: Buf) !void {
        // row-parallel recurrence inputs (read the old conv state), then the sequential delta rule, then the conv shift
        try run(&w.k.g_prep, .{ 16, m, 1 }, .{ gx.wqkv, g.cstate, g.conv, w.qn, w.kn, qb.gdn_qkv });
        try run(&w.k.g_vconv, .{ m, 96, 1 }, .{ gx.wqkv, g.cstate, g.conv, w.vconv, qb.gdn_qkv });
        try run(&w.k.g_gates, .{ m, 1, 1 }, .{ gx.wb, gx.wa, g.a_log, g.dt_bias, w.gbeta, w.ggate });
        try run(&w.k.g_step, .{ qb.gdn_heads, 32 / kern.gdnRows(), 1 }, .{ w.qn, w.kn, w.vconv, w.gbeta, w.ggate, g.sstate, s_out, w.y, @as(u32, @intFromBool(w.o.tiled)), m });
        if (inplace) try run(&w.k.g_commit, .{ (qb.gdn_qkv + 63) / 64, 1, 1 }, .{ gx.wqkv, g.cstate, qb.gdn_qkv, m, @as(u32, 0), qb.gdn_qkv });
    }

    fn attention(w: *Win, a: qb.AttnW, m: u32, pos0: u32) !void {
        if (!attn_prof) return w.attentionInner(a, m, pos0);
        try w.r.sync();
        const t0 = nowNs();
        try w.attentionInner(a, m, pos0);
        try w.r.sync();
        attn_ns += nowNs() - t0;
    }

    fn attentionInner(w: *Win, a: qb.AttnW, m: u32, pos0: u32) !void {
        if (pos0 + m > w.o.cap) return error.ContextFull;
        try w.normIn(a.norm, m);
        try w.attnProj(a, m, pos0);
        if (al.kvMode() != .bf16) {
            try run(if (w.o.f32_norm) &w.k.a_prep32 else &w.k.a_prep, .{ qb.heads + qb.kv_heads, m, 1 }, .{ w.qg, w.kraw, a.qn, a.kn, w.rope, w.q, w.ktmp, @as(u64, 0), w.o.eps });
            try w.o.kvAppend(a.kc, a.vc, w.ktmp, w.vtmp, pos0, m);
        } else try run(if (w.o.f32_norm) &w.k.a_prep32 else &w.k.a_prep, .{ qb.heads + qb.kv_heads, m, 1 }, .{ w.qg, w.kraw, a.qn, a.kn, w.rope, w.q, a.kc, @as(u64, pos0) * qb.kv_dim, w.o.eps });
        if (m > max_rows and w.rcap > max_rows and pfEnv(null)) {
            // prompt window on the matrix engine (its own K / V rows are already cached): 8 rows x 6 heads a work-group
            const mi: usize = @intFromEnum(al.kvMode());
            if (w.pfs[mi] == null) w.pfs[mi] = try al.Pfs.init(w.r, al.kvMode(), w.rcap);
            qb.touch();
            try w.pfs[mi].?.run(w.q, a.kc, a.vc, w.qg, w.att, pos0, m);
            try w.proj(a.o, w.att, m, w.delta, 0);
            w.pending = true;
            return;
        }
        if (m > max_rows and w.pf != null) {
            // prompt window: causal flash attention on the matrix engine, 16 rows a work-group, 24 heads
            try run(&w.pf.?, .{ (m + 15) / 16, qb.heads, 1 }, .{ w.q, a.kc, a.vc, w.qg, w.att, pos0, m, @as(f32, 0.0625) });
            try w.proj(a.o, w.att, m, w.delta, 0);
            w.pending = true;
            return;
        }
        var off: u32 = 0;
        while (off < m) { // row blocks bound the split-K partial scratch; rows are independent, so blocking does not change any bit
            const rb = @min(w.ab, m - off);
            const qoff = @as(usize, off) * qb.q_dim * 2;
            const nold = al.nOld(al.threshold(), pos0 + off + 1, rb); // rows with at most `threshold` keys use the original kernels, the others the long-context ones
            if (nold > 0) {
                const nch = (pos0 + off + nold + qb.attn_chunk - 1) / qb.attn_chunk;
                try run(&w.k.a_part, .{ qb.kv_heads, nch, nold }, .{ at(w.q, qoff), a.kc, a.vc, w.po, w.pm, w.pl, pos0 + off, @as(f32, 0.0625), qb.max_chunks });
                try run(&w.k.a_merge, .{ qb.heads, nold, 1 }, .{ w.po, w.pm, w.pl, at(w.qg, @as(usize, off) * 2 * qb.q_dim * 2), at(w.att, qoff), pos0 + off, qb.max_chunks });
            }
            if (nold < rb) try w.o.longAttn(a.kc, a.vc, at(w.q, qoff + @as(usize, nold) * qb.q_dim * 2), w.po, w.pm, w.pl, at(w.qg, @as(usize, off + nold) * 2 * qb.q_dim * 2), at(w.att, qoff + @as(usize, nold) * qb.q_dim * 2), pos0 + off + nold + 1, rb - nold);
            off += rb;
        }
        try w.proj(a.o, w.att, m, w.delta, 0);
        w.pending = true;
    }

    fn mlp(w: *Win, p: qb.MlpW, m: u32) !void {
        const o = w.o;
        try w.normIn(p.norm, m);
        if (m <= max_rows and p.gate.block and try o.gateUpBlock(p.gate, p.up, w.xn, w.act, m)) {
            // block layout: gate and up partials, one merge + SwiGLU launch
        } else if (m <= max_rows and w.fam == .exact and p.gate.format == .mlx_affine4_g64 and p.up.format == .mlx_affine4_g64 and !p.gate.block) {
            try o.ensureRows();
            try o.mr.?.gateUp(p.gate.w, p.gate.s, p.gate.b, p.up.w, p.up.s, p.up.b, w.xn, w.act, hidden, inter, m);
        } else if (m <= max_rows and w.ggFuse(p.gate.format) and w.ggFuse(p.up.format)) {
            var d: u32 = 0; // gate + up + SwiGLU in one launch an 8-row chunk (the single-row decode's kernel per row)
            while (d < m) : (d += 8) try o.gg.?.gateUpRows(qb.ggType(p.gate.format).?, p.gate.w, qb.ggType(p.up.format).?, p.up.w, at(w.xn, @as(usize, d) * hidden * 2), at(w.act, @as(usize, d) * inter * 2), hidden, inter, @min(8, m - d));
        } else {
            try w.proj(p.gate, w.xn, m, w.gate, 0);
            try w.proj(p.up, w.xn, m, w.up, 0);
            try run(&w.k.swiglu, .{ m * inter / 64, 1, 1 }, .{ w.gate, w.up, w.act, m * inter });
        }
        try w.proj(p.down, w.act, m, w.delta, 0);
        w.pending = true;
    }

    /// Runs `tokens` (1..16) at pos0.. through all layers; last `head_rows` rows get logits; else commit(c).
    pub fn forward(w: *Win, layers: []Layer, emb: qb.Table, norm_f: Buf, head: qb.Table, tokens: []const u32, pos0: u32, inplace: bool, head_rows: u32, bf16_logits: bool) !void {
        const m: u32 = @intCast(tokens.len);
        if (m == 0 or m > w.rcap or (head_rows > m and head_rows != keep_hidden)) return error.Invalid;
        if (m > max_rows) {
            // prompt chunk: in place, logits for at most the last row (the head scratch is 16 rows)
            if (!inplace or (head_rows > 1 and head_rows != keep_hidden)) return error.Invalid;
            if (w.o.ex == null and w.o.gg == null and std.c.getenv("MLX4_NOPF") != null) { // MLX without its prefill GEMM (MLX4_NOPF): 16-row windows
                if (head_rows == keep_hidden) return error.Invalid;
                var off: usize = 0;
                while (off < m) {
                    const sub = tokens[off..@min(off + max_rows, m)];
                    try w.forwardOnce(layers, emb, norm_f, head, sub, pos0 + @as(u32, @intCast(off)), true, if (off + sub.len == m) head_rows else 0, bf16_logits);
                    off += sub.len;
                }
                return;
            }
        }
        try w.forwardOnce(layers, emb, norm_f, head, tokens, pos0, inplace, head_rows, bf16_logits);
    }

    fn forwardOnce(w: *Win, layers: []Layer, emb: qb.Table, norm_f: Buf, head: qb.Table, tokens: []const u32, pos0: u32, inplace: bool, head_rows: u32, bf16_logits: bool) !void {
        if (stop.requested()) {
            w.r.sync() catch {};
            return error.Interrupted;
        }
        const m: u32 = @intCast(tokens.len);
        // token ids and rope table upload from reused host memory (ring of 4 slots, a sync before every 4th window)
        w.ring = (w.ring + 1) % 4;
        if (w.ring == 0) try w.r.sync();
        try w.embed(emb, tokens);
        try w.setRope(pos0, m);
        for (layers, 0..) |ly, i| {
            switch (ly.mixer) {
                .gdn => |g| try w.gdn(g, if (m > max_rows) w.bigx else w.gx[i], m, inplace),
                .attn => |a| try w.attention(a, m, pos0),
            }
            try w.mlp(ly.mlp, m);
        }
        w.rows = m;
        w.inplace = inplace;
        w.pos0 = pos0;
        w.head_rows = if (head_rows == keep_hidden) 0 else head_rows;
        if (head_rows == 0) return;
        try w.normIn(norm_f, m);
        if (head_rows == keep_hidden) return;
        try w.headBatch(head, m - head_rows, head_rows, bf16_logits);
    }

    /// lm_head over n <= 16 normalised hidden rows from row0 of the last window: logits and argmax in w.
    pub fn headBatch(w: *Win, head: qb.Table, row0: u32, n: u32, bf16_logits: bool) !void {
        if (n == 0 or n > max_rows) return error.Invalid;
        w.head_rows = n;
        const v = w.vocab;
        const head_rows = n;
        const off = row0 * hidden * 2;
        switch (head.format) {
            .mlx_affine4_g64 => if (head.block) try w.o.blockMv(head, at(w.xn, off), head_rows, w.logits, 0, true) else {
                try w.o.ensureRows();
                try w.o.mr.?.matvec(.exact, head.w, head.s, head.b, at(w.xn, off), w.logits, hidden, v, head_rows, 0, true);
            },
            .exl3 => {
                const e = w.o.ex.?;
                try head.ex.?.forward(&e.eng, &e.scratch, at(w.xn, off), .bf16, head_rows, w.logits, .f32);
            },
            .q4_k, .iq4_xs => if (w.o.gg != null) {
                var d: u32 = 0; // Q4_K head (GGUF): fp32 rows kernel for up to 8 rows a launch
                while (d < head_rows) : (d += 8) try w.o.gg.?.headRows(qb.ggType(head.format).?, head.w, at(w.xn, off + d * hidden * 2), @min(8, head_rows - d), at(w.logits, @as(usize, d) * v * 4), hidden, v);
            } else for (0..head_rows) |i| try w.o.matvecRaw(head, at(w.xn, off + i * hidden * 2), at(w.logits, i * v * 4), 0, .f32),
            else => for (0..head_rows) |i| try w.o.matvecRaw(head, at(w.xn, off + i * hidden * 2), at(w.logits, i * v * 4), 0, .f32),
        }
        if (bf16_logits) try run(&w.k.round, .{ (head_rows * v + 63) / 64, 1, 1 }, .{ w.logits, head_rows * v });
        const per = (v + qb.argmax_parts - 1) / qb.argmax_parts;
        try run(&w.k.am_part, .{ qb.argmax_parts, head_rows, 1 }, .{ w.logits, w.am_scratch, v, per });
        try run(&w.k.am_fin, .{ head_rows, 1, 1 }, .{ w.am_scratch, w.am_out, qb.argmax_parts });
    }

    /// After a non-inplace window keep the first c rows: all kept swaps in the GDN state, fewer replays them.
    pub fn commit(w: *Win, layers: []Layer, c: u32) !void {
        if (w.inplace or c > w.rows) return error.Invalid;
        if (c == 0) return;
        for (layers, 0..) |*ly, i| switch (ly.mixer) {
            .attn => {},
            .gdn => |*g| {
                const gx = &w.gx[i];
                if (c == w.rows) {
                    std.mem.swap(Buf, &g.sstate, &gx.sstate2);
                    try run(&w.k.g_commit, .{ (qb.gdn_qkv + 63) / 64, 1, 1 }, .{ gx.wqkv, g.cstate, qb.gdn_qkv, c, @as(u32, 0), qb.gdn_qkv });
                } else try w.gdnCore(g.*, gx.*, c, true, g.sstate);
            },
        };
    }
};
