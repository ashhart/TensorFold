//! The Qwen3 LM's XPU binding: lfm3_gq2's norm, SwiGLU and attention, glue's add, qwen3_asr.cl's GEMM and QK RoPE.
const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("../config.zig");
const shared = @import("../lm.zig");
const store = @import("store.zig");
const tower = @import("tower.zig");
const rt = xpu.rt;
const run = tower.run;

/// lfm3_attn_partial's chunk: keys a partial covers.
const chunk = 128;

pub const Lm = struct {
    t: cfg.Text,
    r: *rt.Runtime,
    asr: rt.Module,
    lf: rt.Module,
    glue: rt.Module,
    k: struct { rms: rt.Kernel, add: rt.Kernel, gemm: rt.Kernel, qk: rt.Kernel, part: rt.Kernel, merge: rt.Kernel, swiglu: rt.Kernel },
    cap: usize, // positions the caches hold
    maxc: usize, // chunks a row's keys span at most
    pos0: usize = 0, // the pass's first position
    b: struct { q: u64, k: u64, v: u64, logits: u64, cos: u64, sin: u64, kc: u64, vc: u64, po: u64, pm: u64, pl: u64, gate: u64 },
    ids: []u32,
    rows: usize = 0,
    logits: []u16,
    last_margin: f32 = 0, // the last pick's lead over the runner-up
    sink: ?tower.Dump = null,

    /// Passes of up to `max_rows` rows over up to `span` positions.
    pub fn init(gpa: std.mem.Allocator, r: *rt.Runtime, t: cfg.Text, max_rows: usize, span: usize, st: *store.Store) !Lm {
        if (t.head_dim != 128 or t.heads != 2 * t.kv_heads) return error.UnsupportedQwen3AsrGeometry; // GQ = 2
        if (span > t.context) return error.ContextTooLong;
        if (!xpu.kernels.available) return error.BuiltWithoutKernels;
        var asr = try r.module(xpu.kernels.qwen3_asr);
        errdefer asr.deinit();
        var lf = try r.module(xpu.kernels.lfm3_gq2);
        errdefer lf.deinit();
        var glue = try r.module(xpu.kernels.glue);
        errdefer glue.deinit();
        const maxc = (span + chunk - 1) / chunk;
        var lm: Lm = .{
            .t = t,
            .r = r,
            .asr = asr,
            .lf = lf,
            .glue = glue,
            .k = .{
                .rms = try lf.kernel("lfm3_rmsnorm", .{ 256, 1, 1 }),
                .add = try glue.kernel("add_bf16", .{ 64, 1, 1 }),
                .gemm = try asr.kernel("asr_gemm", .{ 16, 16, 1 }),
                .qk = try asr.kernel("asr_qk_rope", .{ 16, 1, 1 }),
                .part = try lf.kernel("lfm3_attn_partial", .{ 32, 1, 1 }),
                .merge = try lf.kernel("lfm3_attn_merge_gate", .{ 16, 1, 1 }),
                .swiglu = try lf.kernel("lfm3_swiglu", .{ 64, 1, 1 }),
            },
            .cap = span,
            .maxc = maxc,
            .b = undefined,
            .ids = &.{},
            .logits = &.{},
        };
        const qd = t.heads * t.head_dim;
        const kvd = t.kv_heads * t.head_dim;
        const kv = t.layers * span * kvd * 2;
        lm.b = .{
            .q = try st.alloc(2 * max_rows * qd),
            .k = try st.alloc(2 * max_rows * kvd),
            .v = try st.alloc(2 * max_rows * kvd),
            .logits = try st.alloc(2 * t.vocab),
            .cos = try st.alloc(4 * span * t.head_dim),
            .sin = try st.alloc(4 * span * t.head_dim),
            .kc = try st.alloc(kv),
            .vc = try st.alloc(kv),
            .po = try st.alloc(4 * max_rows * t.heads * maxc * t.head_dim),
            .pm = try st.alloc(4 * max_rows * t.heads * maxc),
            .pl = try st.alloc(4 * max_rows * t.heads * maxc),
            .gate = try st.alloc(2 * max_rows * t.heads),
        };
        const cs = try gpa.alloc(f32, span * t.head_dim);
        defer gpa.free(cs);
        const sn = try gpa.alloc(f32, span * t.head_dim);
        defer gpa.free(sn);
        shared.ropeTables(t, cs, sn, span);
        try r.upload(store.view(r, lm.b.cos, cs.len * 4), std.mem.sliceAsBytes(cs));
        try r.upload(store.view(r, lm.b.sin, sn.len * 4), std.mem.sliceAsBytes(sn));
        // no gate on Qwen3: lfm3_attn_merge_gate's bf16(sigmoid(+inf)) is 1, so its rows are bf16(O / L) unscaled
        const gate = try gpa.alloc(u16, max_rows * t.heads);
        defer gpa.free(gate);
        @memset(gate, 0x7f80);
        try r.upload(store.view(r, lm.b.gate, gate.len * 2), std.mem.sliceAsBytes(gate));
        try r.sync();
        lm.ids = try gpa.alloc(u32, max_rows);
        errdefer gpa.free(lm.ids);
        lm.logits = try gpa.alloc(u16, t.vocab);
        return lm;
    }

    pub fn deinit(lm: *Lm, gpa: std.mem.Allocator) void {
        gpa.free(lm.logits);
        gpa.free(lm.ids);
        inline for (.{ &lm.k.rms, &lm.k.add, &lm.k.gemm, &lm.k.qk, &lm.k.part, &lm.k.merge, &lm.k.swiglu }) |k| k.deinit();
        lm.glue.deinit();
        lm.lf.deinit();
        lm.asr.deinit();
    }

    /// The pass's ids (the embedding copies rows by them on the host's side) and its first position.
    pub fn positions(lm: *Lm, ids: []const u32, pos: usize) !void {
        if (ids.len > lm.ids.len) return error.PassRows;
        if (pos + ids.len > lm.cap) return error.ContextFull;
        @memcpy(lm.ids[0..ids.len], ids);
        lm.rows = ids.len;
        lm.pos0 = pos;
    }

    pub fn dump(lm: *Lm, stage: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        if (lm.sink) |dm| try dm.put(dm.ctx, stage, ptr, rows, cols, null);
    }

    /// The table's rows by the pass's ids, device to device.
    pub fn embed(lm: *Lm, table: u64, out: u64, rows: usize, hidden: usize) !void {
        const row = hidden * 2;
        for (lm.ids[0..rows], 0..) |id, i| try lm.copy(out + i * row, table + id * row, row);
    }

    pub fn copy(lm: *Lm, dst: u64, src: u64, n: usize) !void {
        try lm.r.copy(store.view(lm.r, dst, n), store.view(lm.r, src, n), n);
    }

    /// h = bf16(x + res) (x when res is null, h unwritten), y = bf16(w bf16(h rsqrt(mean h^2 + eps))).
    pub fn addNorm(lm: *Lm, x: u64, res: ?u64, w: u64, h: u64, y: u64, rows: usize) !void {
        const n = lm.t.hidden;
        const eps: f32 = @floatCast(lm.t.eps);
        const src = if (res) |rs| blk: {
            try lm.copy(h, rs, rows * n * 2);
            try run(lm.r, &lm.k.add, .{ (rows * n + 63) / 64, 1, 1 }, .{ h, x, rows * n });
            break :blk h;
        } else x;
        try run(lm.r, &lm.k.rms, .{ rows, 1, 1 }, .{ src, w, y, n, eps });
    }

    pub fn gemm(lm: *Lm, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const z = lm.b.gate; // unread: no bias, no residual
        try run(lm.r, &lm.k.gemm, .{ (n + 63) / 64, (rows + 63) / 64, 1 }, .{ x, w, z, @as(u32, 0), z, @as(u32, 0), y, rows, n, k, n, @as(u32, 0) });
    }

    fn kvAt(lm: *const Lm, base: u64, layer: usize) u64 {
        return base + layer * lm.cap * lm.t.kv_heads * lm.t.head_dim * 2;
    }

    /// q and k RMS-normed per head and rotated, k and v into layer's caches at the pass's positions.
    pub fn qkRope(lm: *Lm, qkv: u64, q_norm: u64, k_norm: u64, layer: usize, rows: usize) !void {
        const t = lm.t;
        const b = lm.b;
        try run(lm.r, &lm.k.qk, .{ t.heads + 2 * t.kv_heads, rows, 1 }, .{ qkv, q_norm, k_norm, b.cos, b.sin, b.q, b.k, b.v, lm.kvAt(b.kc, layer), lm.kvAt(b.vc, layer), t.heads, t.kv_heads, lm.pos0, @as(f32, @floatCast(t.eps)) });
    }

    /// Causal GQA attention of the pass's q rows over layer's caches into out [rows, heads x dim]: lfm3's split-K pair.
    pub fn attention(lm: *Lm, layer: usize, out: u64, rows: usize) !void {
        const t = lm.t;
        const b = lm.b;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(t.head_dim))));
        const nch = (lm.pos0 + rows + chunk - 1) / chunk;
        try run(lm.r, &lm.k.part, .{ t.kv_heads, nch, rows }, .{ b.q, lm.kvAt(b.kc, layer), lm.kvAt(b.vc, layer), b.po, b.pm, b.pl, lm.pos0, @as(u32, chunk), lm.cap, @as(u32, 0), t.kv_heads, lm.maxc, scale });
        try run(lm.r, &lm.k.merge, .{ t.heads, rows, 1 }, .{ b.po, b.pm, b.pl, b.gate, out, lm.pos0, @as(u32, chunk), @as(u32, 0), t.heads, lm.maxc });
    }

    /// out [rows, i] = bf16(bf16(silu(gate)) up) of gu's [gate | up] rows: lfm3_swiglu a row at a time.
    pub fn swiglu(lm: *Lm, gu: u64, out: u64, rows: usize, i: usize) !void {
        for (0..rows) |r| try run(lm.r, &lm.k.swiglu, .{ (i + 63) / 64, 1, 1 }, .{ gu + r * 4 * i, gu + r * 4 * i + 2 * i, out + r * 2 * i, i });
    }

    /// The tied head over one normed row, downloaded: its greedy pick.
    pub fn greedy(lm: *Lm, row: u64, embed_table: u64) !u32 {
        try lm.gemm(row, embed_table, lm.b.logits, 1, lm.t.vocab, lm.t.hidden);
        try lm.r.download(std.mem.sliceAsBytes(lm.logits), store.view(lm.r, lm.b.logits, lm.logits.len * 2));
        try lm.r.sync();
        const best = shared.argmax(lm.logits);
        lm.last_margin = shared.margin(lm.logits, best);
        return best;
    }
};
