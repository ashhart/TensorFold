//! Qwen3.8 dense decode blocks on the Intel GPU: weight tables, scratch, GDN / attention / SwiGLU blocks, embed, head.

const std = @import("std");
const rt = @import("xpu").rt;
const ld = @import("xpu").loader;
const cq = @import("xpu_config.zig");
const exl3 = @import("xpu").exl3;
const gg = @import("xpu").ggml;
const mx = @import("xpu_mlx4.zig");
const mxb = @import("xpu_mlx4b.zig");
const al = @import("xpu_attn_long.zig");
const mpf = @import("xpu_mlx4_pf.zig");

const spv_basic = @import("xpu").kernels.qwen_basic;
const spv_gdn = @import("xpu").kernels.qwen_gdn;
const spv_attn = @import("xpu").kernels.qwen_attn;
const spv_mlx4 = @import("xpu").kernels.qwen_mlx4; // tuned MLX 4-bit matvec (one sub-group a row)
const spv_small = @import("xpu").kernels.qwen_small; // fp16 small projections (in_proj_a/b) on 4 sub-groups a row

pub const Buf = rt.Buffer;
/// EXL3 engine state shared by all layers (set by qwen_load.attach when the checkpoint is EXL3).
pub const Exl = struct { eng: exl3.Engine, scratch: exl3.Scratch };
pub const attn_chunk: u32 = 64; // keys per split-K chunk (qwen_attn.cl CH)
pub const max_chunks: u32 = 128;
pub const argmax_parts: u32 = 64;
pub const hidden: u32 = 5120;
pub const inter: u32 = 17408;
pub const heads: u32 = 24;
pub const kv_heads: u32 = 4;
pub const head_dim: u32 = 256;
pub const q_dim: u32 = heads * head_dim;
pub const kv_dim: u32 = kv_heads * head_dim;
pub const gdn_qkv: u32 = 10240; // 2 * 16 * 128 + 48 * 128
pub const gdn_v: u32 = 6144; // 48 * 128
pub const gdn_heads: u32 = 48;
const rope_half: u32 = 32;

const lin = @import("xpu_linear.zig");
pub const Format = lin.Format;
pub const ggType = lin.ggType;
pub const ggFormat = lin.ggFormat;
pub const Linear = lin.Linear;
pub const Table = Linear;
pub const GdnW = struct { norm: Buf, qkv: Table, z: Table, b: Table, a: Table, out: Table, conv: Buf, a_log: Buf, dt_bias: Buf, gnorm: Buf, cstate: Buf, sstate: Buf };
pub const AttnW = struct { norm: Buf, q: Table, k: Table, v: Table, o: Table, qn: Buf, kn: Buf, kc: Buf, vc: Buf };
pub const MlpW = struct { norm: Buf, gate: Table, up: Table, down: Table };

const Kernels = @import("xpu_block_kernels.zig").Kernels;

/// Device scratch shared by all layers (one token at a time).
pub const Scratch = struct {
    tok: Buf,
    x: Buf,
    xn: Buf,
    delta: Buf,
    qkv: Buf,
    z: Buf,
    b: Buf,
    a: Buf,
    conv: Buf,
    qn: Buf,
    kn: Buf,
    y: Buf,
    yg: Buf,
    qg: Buf,
    kraw: Buf,
    /// New K / V row of a quantized KV cache before it is appended (al.Kvq).
    ktmp: Buf,
    vtmp: Buf,
    q: Buf,
    att: Buf,
    po: Buf,
    pm: Buf,
    pl: Buf,
    rope: Buf,
    gate: Buf,
    up: Buf,
    act: Buf,
    logits: Buf,
    am_scratch: Buf,
    am_out: Buf,
    /// int8 blocks + scales of the last activation a ggml matvec consumed (ggml.Set.quantize layout, max 17408 inputs).
    xq: Buf,
};

/// Set by every launch through `run`: the int8 activation copy (Ops.xq_src) may be stale; ggml matvecs leave it.
var xq_dirty: bool = true;
/// The block-layout activation prepass (mb.xt) holds (xt_src, xt_in, xt_m); any launch through `run` may invalidate it.
var xt_fresh: bool = false;
var xt_src: ?*anyopaque = null;
var xt_in: u32 = 0;
var xt_m: u32 = 0;

/// Per-kernel timing (Ops.profile): every `run` launch is synced and timed, so each entry includes the sync round trip.
pub var kprof_on: bool = false;
const KProf = struct { h: ?*const anyopaque = null, ns: u64 = 0, calls: u64 = 0 };
var kprof: [40]KProf = @splat(.{});

fn kprofAccount(k: *rt.Kernel, t0: u64) void {
    k.rt.sync() catch return;
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

/// Marks the cached activation copies (int8 for ggml, bf16 prepass) stale: call after any launch writing an activation.
pub fn touch() void {
    xq_dirty = true;
    xt_fresh = false;
}

/// Sets every argument (Buffer, f32 or integer) in order and queues the kernel.
fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    xq_dirty = true;
    xt_fresh = false;
    const t0 = if (kprof_on) blk: {
        try k.rt.sync();
        break :blk nowNs();
    } else 0;
    defer if (kprof_on) kprofAccount(k, t0);
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub const ProfEntry = struct { rows: u32 = 0, in: u32 = 0, fmt: Format = .mlx_affine4_g64, ns: u64 = 0, calls: u64 = 0 };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const Ops = struct {
    r: *rt.Runtime,
    eps: f32,
    rope_theta: f64,
    cap: u32,
    k: Kernels,
    s: Scratch,
    ex: ?*Exl = null,
    /// Multi-row MLX 4-bit kernels (created on the first matvecRows).
    mr: ?*mx.Rows = null,
    /// Systolic kernels for block-layout tensors (created on first use).
    mb: ?*mxb.Block = null,
    /// Long-context attention (rows with more than al.threshold keys; created on first use).
    long: [3]?*al.Long = .{ null, null, null }, // by KV mode (the MTP block's cache may differ from the target's)
    /// MLX 4-bit prefill GEMM (created on first use).
    mpf: ?*mpf.Pf = null,
    /// Appends rows to a quantized KV cache (created on first use).
    kvq: ?*al.Kvq = null,
    /// ggml kernels (set by qwen_load.attach when the checkpoint is a GGUF file).
    gg: ?*gg.Set = null,
    /// Source buffer and length of the activation currently quantized in s.xq.
    xq_src: ?*anyopaque = null,
    xq_in: u32 = 0,
    /// Value heads are in llama.cpp's tiled order (GGUF) instead of the checkpoint's grouped order.
    tiled: bool = false,
    /// RMSNorm weights are fp32 buffers (GGUF) and go through the f32-weight norm kernels.
    f32_norm: bool = false,
    /// s.delta holds a block output not yet added to s.x; the next norm (add_rmsnorm) or flush() adds it.
    pending: bool = false,
    /// When set, matvec syncs around each launch and accumulates its time per shape in `prof`.
    profile: bool = false,
    prof: [96]ProfEntry = @splat(.{}),
    // Host staging of the position table and token: a slot is reused only after a sync (ring of 4).
    rope_host: [4][64]f32 = @splat(@splat(0)),
    host_tok: [4]u32 = @splat(0),
    rope_n: u32 = 0,
    tok_n: u32 = 0,

    pub fn init(r: *rt.Runtime, l: *ld.Loader, cfg: cq.Config, cap: u32) !Ops {
        try cfg.validate();
        if (cap > al.max_ctx) return error.ContextTooLarge;
        var bm = try r.module(spv_basic);
        var gm = try r.module(spv_gdn);
        var am = try r.module(spv_attn);
        var xm = try r.module(spv_mlx4);
        var sm = try r.module(spv_small);
        const k: Kernels = .{
            .embed = try bm.kernel("embed4", .{ 64, 1, 1 }),
            .embed16 = try bm.kernel("embed_bf16", .{ 64, 1, 1 }),
            .mv16 = try sm.kernel("mv16x", .{ 64, 1, 1 }),
            .rms = try bm.kernel("rmsnorm", .{ 64, 1, 1 }),
            .rms512 = try bm.kernel("rmsnorm512", .{ 512, 1, 1 }),
            .add_rms = try bm.kernel("add_rmsnorm", .{ 512, 1, 1 }),
            .add_rms32 = try bm.kernel("add_rmsnorm_f32w", .{ 512, 1, 1 }),
            .add = try bm.kernel("add_bf16", .{ 64, 1, 1 }),
            .round = try bm.kernel("round_bf16_f32", .{ 64, 1, 1 }),
            .qmv = try xm.kernel("qmv4x", .{ 16, 1, 1 }),
            .gateup = try xm.kernel("qmv4x_gateup", .{ 16, 1, 1 }),
            .head = try xm.kernel("qmv4x", .{ 16, 1, 1 }),
            .swiglu = try bm.kernel("swiglu", .{ 64, 1, 1 }),
            .am_part = try bm.kernel("argmax_partial", .{ 256, 1, 1 }),
            .am_fin = try bm.kernel("argmax_final", .{ 256, 1, 1 }),
            .g_prep = try gm.kernel("gdn_conv_prep", .{ 16, 1, 1 }),
            .g_step = try gm.kernel("gdn_step", .{ 64, 1, 1 }),
            .g_norm = try gm.kernel("gdn_gate_norm", .{ 16, 1, 1 }),
            .a_prep = try am.kernel("attn_prep", .{ 256, 1, 1 }),
            .a_part = try am.kernel("attn_partial", .{ 128, 1, 1 }),
            .a_merge = try am.kernel("attn_merge", .{ 256, 1, 1 }),
        };
        const v: usize = cfg.text_config.vocab_size;
        const s: Scratch = .{
            .tok = try l.empty(4),
            .x = try l.empty(hidden * 2),
            .xn = try l.empty(hidden * 2),
            .delta = try l.empty(hidden * 2),
            .qkv = try l.empty(gdn_qkv * 2),
            .z = try l.empty(gdn_v * 2),
            .b = try l.empty(gdn_heads * 2),
            .a = try l.empty(gdn_heads * 2),
            .conv = try l.empty(gdn_qkv * 2),
            .qn = try l.empty(2048 * 4),
            .kn = try l.empty(2048 * 4),
            .y = try l.empty(gdn_v * 2),
            .yg = try l.empty(gdn_v * 2),
            .qg = try l.empty(2 * q_dim * 2),
            .kraw = try l.empty(kv_dim * 2),
            .ktmp = try l.empty(kv_dim * 2),
            .vtmp = try l.empty(kv_dim * 2),
            .q = try l.empty(q_dim * 2),
            .att = try l.empty(q_dim * 2),
            .po = try l.empty(@as(usize, max_chunks) * heads * head_dim * 4),
            .pm = try l.empty(max_chunks * heads * 4),
            .pl = try l.empty(max_chunks * heads * 4),
            .rope = try l.empty(64 * 4),
            .gate = try l.empty(inter * 2),
            .up = try l.empty(inter * 2),
            .act = try l.empty(inter * 2),
            .logits = try l.empty(v * 4),
            .am_scratch = try l.empty(argmax_parts * 8),
            .am_out = try l.empty(4),
            .xq = try l.empty(17408 + 17408 / 8),
        };
        return .{ .r = r, .eps = cfg.text_config.rms_norm_eps, .rope_theta = cfg.text_config.rope_parameters.rope_theta, .cap = cap, .k = k, .s = s };
    }

    /// Uploads the RoPE cos/sin table (fp32 angle = pos * inv_freq, as the reference) for the next attention layers.
    pub fn setPosition(o: *Ops, pos: u32) !void {
        const slot = o.rope_n % 4;
        if (slot == 0 and o.rope_n != 0) try o.r.sync(); // all earlier uploads have run: their slots are free again
        o.rope_n +%= 1;
        const tab = &o.rope_host[slot];
        for (0..rope_half) |i| {
            const inv: f32 = @floatCast(std.math.pow(f64, o.rope_theta, -@as(f64, @floatFromInt(i)) / @as(f64, rope_half)));
            const ang: f32 = @as(f32, @floatFromInt(pos)) * inv;
            tab[i] = @cos(ang);
            tab[rope_half + i] = @sin(ang);
        }
        try o.r.upload(o.s.rope, std.mem.sliceAsBytes(tab));
    }

    pub fn embed(o: *Ops, t: Table, token: u32) !void {
        const slot = o.tok_n % 4;
        if (slot == 0 and o.tok_n != 0) try o.r.sync();
        o.tok_n +%= 1;
        o.host_tok[slot] = token;
        o.pending = false;
        try o.r.upload(o.s.tok, std.mem.asBytes(&o.host_tok[slot]));
        switch (t.format) {
            .mlx_affine4_g64 => try run(&o.k.embed, .{ 1, 1, 1 }, .{ t.w, t.s, t.b, o.s.tok, o.s.x, hidden }),
            .f16, .exl3 => return error.Invalid,
            .bf16 => try run(&o.k.embed16, .{ 1, 1, 1 }, .{ t.w, o.s.tok, o.s.x, hidden }),
            else => try o.gg.?.embedRow(ggType(t.format).?, t.w, o.s.tok, o.s.x, hidden),
        }
    }

    fn rms(o: *Ops, w: Buf) !void {
        if (!o.f32_norm) return run(&o.k.rms512, .{ 1, 1, 1 }, .{ o.s.x, w, o.s.xn, hidden, o.eps });
        try run(&o.gg.?.rms32, .{ 1, 1, 1 }, .{ o.s.x, w, o.s.xn, hidden, o.eps });
    }

    /// The norm in front of a block: folds the previous block's pending residual add into one kernel.
    fn normIn(o: *Ops, w: Buf) !void {
        if (!o.pending) return o.rms(w);
        o.pending = false;
        try run(if (o.f32_norm) &o.k.add_rms32 else &o.k.add_rms, .{ 1, 1, 1 }, .{ o.s.x, o.s.delta, w, o.s.xn, hidden, o.eps });
    }

    /// Applies a pending residual add so s.x is the residual stream after the last block.
    pub fn flush(o: *Ops) !void {
        if (!o.pending) return;
        o.pending = false;
        try o.addResidual();
    }

    /// Creates the multi-row MLX kernels on first use (matvecRows: m = 1..16 rows, row invariant).
    pub fn ensureRows(o: *Ops) !void {
        if (o.mr != null) return;
        const p = try std.heap.page_allocator.create(mx.Rows);
        p.* = try mx.Rows.init(o.r, 17408, 17408);
        o.mr = p;
    }

    /// y (bf16, or fp32 when f32out) = block-layout tensor t times m activation rows; one kernel for every m.
    pub fn blockMv(o: *Ops, t: Linear, x: Buf, m: u32, y: Buf, y_off: u32, f32out: bool) !void {
        if (o.mb == null) {
            const p = try std.heap.page_allocator.create(mxb.Block);
            p.* = try mxb.Block.init(o.r, 17408, 17408);
            o.mb = p;
        }
        xq_dirty = true;
        const same = xt_fresh and xt_src == x.ptr and xt_in == t.in and xt_m == m;
        try o.mb.?.matvecP(t.w, t.s, t.b, x, y, t.in, t.rows, m, y_off, f32out, !same);
        xt_fresh = true;
        xt_src = x.ptr;
        xt_in = t.in;
        xt_m = m;
    }

    /// act [m][inter] = SwiGLU(gate x, up x) for block-layout tensors; false when the split count is 1.
    pub fn gateUpBlock(o: *Ops, gate: Linear, up: Linear, x: Buf, act: Buf, m: u32) !bool {
        if (o.mb == null) {
            const p = try std.heap.page_allocator.create(mxb.Block);
            p.* = try mxb.Block.init(o.r, 17408, 17408);
            o.mb = p;
        }
        xq_dirty = true;
        const same = xt_fresh and xt_src == x.ptr and xt_in == gate.in and xt_m == m;
        const ok = try o.mb.?.gateUp(gate.w, gate.s, gate.b, up.w, up.s, up.b, x, act, gate.in, gate.rows, m, !same);
        if (ok) {
            xt_fresh = true;
            xt_src = x.ptr;
            xt_in = gate.in;
            xt_m = m;
        }
        return ok;
    }

    /// Attention of `rows` rows over len0 + z keys with the long-context kernels (rows of a window past the threshold).
    pub fn longAttn(o: *Ops, kc: Buf, vc: Buf, q: Buf, po: Buf, pm: Buf, pl: Buf, qg: Buf, out: Buf, len0: u32, rows: u32) !void {
        const mi = @intFromEnum(al.kvMode());
        if (o.long[mi] == null) {
            const p = try std.heap.page_allocator.create(al.Long);
            p.* = try al.Long.init(o.r, al.kvMode());
            o.long[mi] = p;
        }
        try o.long[mi].?.run(q, kc, vc, po, pm, pl, qg, out, len0, rows);
    }

    /// Quantizes `rows` new K / V rows (bf16 [rows][kv_dim]) into the caches at positions pos0 ... (KV mode q8 / q4).
    pub fn kvAppend(o: *Ops, kc: Buf, vc: Buf, ksrc: Buf, vsrc: Buf, pos0: u32, rows: u32) !void {
        if (o.kvq == null) {
            const p = try std.heap.page_allocator.create(al.Kvq);
            p.* = try al.Kvq.init(o.r);
            o.kvq = p;
        }
        try o.kvq.?.append(al.kvMode(), ksrc, kc, pos0, rows);
        try o.kvq.?.append(al.kvMode(), vsrc, vc, pos0, rows);
    }

    pub fn matvecRows(o: *Ops, t: Linear, x: Buf, m: u32, y: Buf, y_off: u32, fam: mx.Family) !void {
        switch (t.format) {
            .mlx_affine4_g64 => {
                if (t.block) return o.blockMv(t, x, m, y, y_off, false);
                if (o.mr == null) {
                    const p = try std.heap.page_allocator.create(mx.Rows);
                    p.* = try mx.Rows.init(o.r, 17408, 17408);
                    o.mr = p;
                }
                xq_dirty = true;
                try o.mr.?.matvec(fam, t.w, t.s, t.b, x, y, t.in, t.rows, m, y_off, false);
            },
            .exl3 => {
                const e = o.ex.?;
                try t.ex.?.forward(&e.eng, &e.scratch, x, .bf16, m, exl3.at(y, @as(usize, y_off) * 2), .bf16);
            },
            .bf16 => try o.gg.?.matvecBf16Rows(t.w, x, y, t.in, y_off, t.rows, m),
            else => {
                // GGUF block types: row invariant 8-row windows (the weights are read once a window)
                const gt = ggType(t.format) orelse return error.Invalid;
                var done: u32 = 0;
                while (done < m) {
                    const n = @min(8, m - done);
                    try o.gg.?.matvecRows(gt, t.w, gg.sub(x, @as(usize, done) * t.in * 2), n, gg.sub(y, @as(usize, done) * t.rows * 2), t.in, y_off, t.rows);
                    done += n;
                }
            },
        }
    }

    /// R rows x [R][t.in] (bf16) against one projection on the matrix engine into y [R][t.rows] (R >= 16).
    pub fn prefillRows(o: *Ops, t: Linear, x: Buf, R: u32, y: Buf, y_off: u32) !void {
        if (t.format == .mlx_affine4_g64 and t.rows % 16 == 0 and t.in % 512 == 0) { // MLX 4-bit: fp16 DPAS GEMM, chunk invariant, fp16-rounded weights and activations
            if (o.mpf == null) {
                const p = try std.heap.page_allocator.create(mpf.Pf);
                p.* = try mpf.Pf.init(o.r);
                o.mpf = p;
            }
            return o.mpf.?.run(t.block, t.w, t.s, t.b, x, R, y, t.in, y_off, t.rows);
        }
        if (ggType(t.format)) |gt| {
            if (gt != .iq1_m and t.rows % 16 == 0) return o.gg.?.prefillRows(gt, t.w, x, R, y, t.in, y_off, t.rows);
        }
        try o.matvecRows(t, x, R, y, y_off, .exact);
    }

    pub fn matvec(o: *Ops, t: Linear, x: Buf, y: Buf, y_off: u32) !void {
        const t0 = if (o.profile) blk: {
            try o.r.sync();
            break :blk nowNs();
        } else 0;
        try o.matvecRaw(t, x, y, y_off, .bf16);
        if (o.profile) {
            try o.r.sync();
            o.account(t, nowNs() - t0);
        }
    }

    /// Prints the per-kernel totals of `run` launches per token (see kprof_on).
    pub fn printKernelProfile(o: *Ops, tokens: usize) void {
        inline for (@typeInfo(Kernels).@"struct".field_names) |fname| {
            for (kprof) |e| if (e.h != null and e.h == @as(?*const anyopaque, @ptrCast(@field(o.k, fname).handle))) std.debug.print("  kernel {s:<8} {d:>5} launches/token {d:>8.1} us avg  {d:>7.3} ms/token\n", .{ fname, e.calls / tokens, @as(f64, @floatFromInt(e.ns)) / 1e3 / @as(f64, @floatFromInt(e.calls)), @as(f64, @floatFromInt(e.ns)) / 1e6 / @as(f64, @floatFromInt(tokens)) });
        }
    }

    fn account(o: *Ops, t: Linear, ns: u64) void {
        for (&o.prof) |*e| {
            if (e.calls == 0) e.* = .{ .rows = t.rows, .in = t.in, .fmt = t.format };
            if (e.rows == t.rows and e.in == t.in and e.fmt == t.format) {
                e.ns += ns;
                e.calls += 1;
                return;
            }
        }
    }

    pub fn matvecRaw(o: *Ops, t: Linear, x: Buf, y: Buf, y_off: u32, ydt: exl3.Dtype) !void {
        switch (t.format) {
            .mlx_affine4_g64 => if (t.block) try o.blockMv(t, x, 1, y, y_off, ydt == .f32) else try run(&o.k.qmv, .{ t.rows, 1, 1 }, .{ t.w, t.s, t.b, x, y, t.in, @as(u32, 0), y_off, t.rows, @as(u32, 0) }),
            .bf16 => try o.gg.?.matvecBf16(t.w, x, y, t.in, y_off, t.rows),
            .f16 => try run(&o.k.mv16, .{ t.rows, 1, 1 }, .{ t.w, x, y, t.in, y_off, t.rows }),
            .exl3 => {
                const e = o.ex.?;
                const ydst = exl3.at(y, @as(usize, y_off) * if (ydt == .f32) @as(usize, 4) else 2);
                try t.ex.?.forward(&e.eng, &e.scratch, x, .bf16, 1, ydst, ydt);
            },
            else => {
                // the ggml kernels quantize the bf16 activation themselves
                try o.gg.?.matvec(ggType(t.format).?, t.w, x, y, t.in, y_off, t.rows, ydt == .f32);
            },
        }
    }

    fn addResidual(o: *Ops) !void {
        try run(&o.k.add, .{ hidden / 64, 1, 1 }, .{ o.s.x, o.s.delta, hidden });
    }

    fn ggFusable(f: Format) bool {
        return ggType(f) != null and f != .iq1_m and f != .q6_k and f != .q5_k and f != .q8_0 and f != .iq4_nl;
    }

    /// The GDN input projections qkv, z, b, a: a GGUF fuses qkv + z and the gates (else four matvecs).
    fn gdnProj(o: *Ops, w: GdnW) !void {
        const s = o.s;
        if (o.gg != null and !o.profile and ggFusable(w.qkv.format) and ggFusable(w.z.format)) {
            try o.gg.?.matvecMulti(&.{
                .{ .t = ggType(w.qkv.format).?, .w = w.qkv.w, .y = s.qkv, .rows = w.qkv.rows, .y_off = 0 },
                .{ .t = ggType(w.z.format).?, .w = w.z.w, .y = s.z, .rows = w.z.rows, .y_off = 0 },
            }, s.xn, hidden);
        } else {
            try o.matvec(w.qkv, s.xn, s.qkv, 0);
            try o.matvec(w.z, s.xn, s.z, 0);
        }
        if (o.gg != null and !o.profile and w.b.format == .bf16 and w.a.format == .bf16) {
            try o.gg.?.matvecBf16Pair(w.b.w, w.a.w, s.xn, s.b, s.a, hidden, gdn_heads);
        } else {
            try o.matvec(w.b, s.xn, s.b, 0);
            try o.matvec(w.a, s.xn, s.a, 0);
        }
    }

    /// The attention q, k, v projections (v into the cache at `pos`): one fused launch on a GGUF, else three matvecs.
    fn attnProj(o: *Ops, w: AttnW, pos: u32) !void {
        const s = o.s;
        const quant = al.kvMode() != .bf16; // a quantized cache takes the new row through s.vtmp
        const vdst = if (quant) s.vtmp else w.vc;
        const voff: u32 = if (quant) 0 else pos * kv_dim;
        if (o.gg != null and !o.profile and ggFusable(w.q.format) and ggFusable(w.k.format) and ggFusable(w.v.format)) {
            try o.gg.?.matvecMulti(&.{
                .{ .t = ggType(w.q.format).?, .w = w.q.w, .y = s.qg, .rows = w.q.rows, .y_off = 0 },
                .{ .t = ggType(w.k.format).?, .w = w.k.w, .y = s.kraw, .rows = w.k.rows, .y_off = 0 },
                .{ .t = ggType(w.v.format).?, .w = w.v.w, .y = vdst, .rows = w.v.rows, .y_off = voff },
            }, s.xn, hidden);
        } else {
            try o.matvec(w.q, s.xn, s.qg, 0);
            try o.matvec(w.k, s.xn, s.kraw, 0);
            try o.matvec(w.v, s.xn, vdst, voff);
        }
    }

    /// The Gated DeltaNet block: x += out_proj(gated_norm(delta_rule(conv(in_proj(rmsnorm(x)))))).
    pub fn gdn(o: *Ops, w: GdnW) !void {
        const s = o.s;
        try o.normIn(w.norm);
        try o.gdnProj(w);
        try run(&o.k.g_prep, .{ 16, 1, 1 }, .{ s.qkv, w.cstate, w.conv, s.qn, s.kn, gdn_qkv });
        try run(&o.k.g_step, .{ gdn_heads, 32, 1 }, .{ s.qn, s.kn, s.qkv, w.cstate, w.conv, s.b, s.a, w.a_log, w.dt_bias, w.sstate, s.y, @as(u32, @intFromBool(o.tiled)), gdn_qkv });
        try run(&o.k.g_norm, .{ gdn_heads, 1, 1 }, .{ s.y, s.z, w.gnorm, s.yg, o.eps });
        try o.matvec(w.out, s.yg, s.delta, 0);
        o.pending = true;
    }

    /// The full-attention block at position `pos` (setPosition(pos) first): gated q, q/k norm, RoPE, KV cache, o_proj.
    pub fn attention(o: *Ops, w: AttnW, pos: u32) !void {
        if (pos >= o.cap) return error.ContextFull;
        const s = o.s;
        const len = pos + 1;
        try o.normIn(w.norm);
        try o.attnProj(w, pos);
        if (al.kvMode() != .bf16) {
            try run(if (o.f32_norm) &o.gg.?.aprep32 else &o.k.a_prep, .{ heads + kv_heads, 1, 1 }, .{ s.qg, s.kraw, w.qn, w.kn, s.rope, s.q, s.ktmp, @as(u32, 0), o.eps });
            try o.kvAppend(w.kc, w.vc, s.ktmp, s.vtmp, pos, 1);
        } else try run(if (o.f32_norm) &o.gg.?.aprep32 else &o.k.a_prep, .{ heads + kv_heads, 1, 1 }, .{ s.qg, s.kraw, w.qn, w.kn, s.rope, s.q, w.kc, pos * kv_dim, o.eps });
        if (len > al.threshold()) {
            try o.longAttn(w.kc, w.vc, s.q, s.po, s.pm, s.pl, s.qg, s.att, len, 1);
        } else {
            const nch = (len + attn_chunk - 1) / attn_chunk;
            try run(&o.k.a_part, .{ kv_heads, nch, 1 }, .{ s.q, w.kc, w.vc, s.po, s.pm, s.pl, len, @as(f32, 0.0625) });
            try run(&o.k.a_merge, .{ heads, 1, 1 }, .{ s.po, s.pm, s.pl, s.qg, s.att, len });
        }
        try o.matvec(w.o, s.att, s.delta, 0);
        o.pending = true;
    }

    /// The SwiGLU block: x += down(silu(gate(rmsnorm(x))) * up(rmsnorm(x))).
    pub fn mlp(o: *Ops, w: MlpW) !void {
        const s = o.s;
        try o.normIn(w.norm);
        if (o.gg != null and !o.profile and ggFusable(w.gate.format) and ggFusable(w.up.format)) {
            // GGUF: gate, up and SwiGLU in one launch (bf16-rounded gate and up, as the three kernels)
            try o.gg.?.gateUp(ggType(w.gate.format).?, w.gate.w, ggType(w.up.format).?, w.up.w, s.xn, s.act, hidden, inter);
        } else if (w.gate.block and !o.profile and try o.gateUpBlock(w.gate, w.up, s.xn, s.act, 1)) {
            // block layout: gate, up partials and one merge + SwiGLU launch
        } else if (w.gate.format == .mlx_affine4_g64 and w.up.format == .mlx_affine4_g64 and !o.profile and !w.gate.block) {
            // gate, up and SwiGLU in one launch (same rounding as the three kernels)
            try run(&o.k.gateup, .{ inter, 1, 1 }, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, s.xn, s.act, hidden, inter });
        } else {
            try o.matvec(w.gate, s.xn, s.gate, 0);
            try o.matvec(w.up, s.xn, s.up, 0);
            try run(&o.k.swiglu, .{ inter / 64, 1, 1 }, .{ s.gate, s.up, s.act, inter });
        }
        try o.matvec(w.down, s.act, s.delta, 0);
        o.pending = true;
    }

    /// Final norm, lm_head (fp32 logits, bf16-rounded when asked) and greedy argmax into s.am_out.
    pub fn head(o: *Ops, norm: Buf, t: Table, vocab: u32, bf16_logits: bool) !void {
        const s = o.s;
        try o.normIn(norm);
        switch (t.format) {
            .mlx_affine4_g64 => if (t.block) try o.blockMv(t, s.xn, 1, s.logits, 0, true) else try run(&o.k.head, .{ vocab, 1, 1 }, .{ t.w, t.s, t.b, s.xn, s.logits, hidden, @as(u32, 0), @as(u32, 0), vocab, @as(u32, 1) }),
            .f16, .bf16 => return error.Invalid,
            else => try o.matvecRaw(t, s.xn, s.logits, 0, .f32),
        }
        if (bf16_logits) try run(&o.k.round, .{ (vocab + 63) / 64, 1, 1 }, .{ s.logits, vocab });
        const per = (vocab + argmax_parts - 1) / argmax_parts;
        try run(&o.k.am_part, .{ argmax_parts, 1, 1 }, .{ s.logits, s.am_scratch, vocab, per });
        try run(&o.k.am_fin, .{ 1, 1, 1 }, .{ s.am_scratch, s.am_out, argmax_parts });
    }

    /// The greedy token of the last head() (syncs).
    pub fn argmax(o: *Ops) !u32 {
        var out: [1]i32 = undefined;
        try o.r.download(std.mem.sliceAsBytes(&out), o.s.am_out);
        try o.r.sync();
        return @intCast(out[0]);
    }
};
