//! The Qwen3 language model's binding on CUDA's shared rows: RMS norms, QK norm with RoPE, Nemotron's GQA attention.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const shared = @import("../lm.zig");
const store = @import("store.zig");

/// nemotron_attention.cu's chunk: keys a partial covers, and the most chunk blocks a row launches.
const chunk_keys = 512;
const chunk_blocks = 64;

const Attn = struct {
    mod: cuda.Module,
    chunk: cuda.Function,
    chunk_pf: cuda.Function,
    merge: cuda.Function,

    fn load(d: *const cuda.Driver) !Attn {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.nemotron_attention);
        errdefer mod.unload();
        const f = .{ try mod.function("tf_nemo_attn_chunk"), try mod.function("tf_nemo_attn_chunk_pf"), try mod.function("tf_nemo_attn_merge") };
        return .{ .mod = mod, .chunk = f[0], .chunk_pf = f[1], .merge = f[2] };
    }
};

pub const Lm = struct {
    t: cfg.Text,
    d: *const cuda.Driver,
    s: cuda.Stream,
    g: cuda.vit.Gemms,
    nm: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    act: cuda.act_rows.ActRows,
    emb: cuda.embed_rows.EmbedRows,
    attn: Attn,
    cap: usize, // positions the caches hold
    b: Buffers,
    host_ids: []i32,
    host_pos: []i32,
    logits: []u16,
    last_margin: f32 = 0, // the last pick's lead over the runner-up
    sink: ?@import("tower.zig").Dump = null,

    const Buffers = struct { ids: u64, pos: u64, meta: u64, q: u64, k: u64, v: u64, po: u64, pm: u64, pl: u64, xs: u64, logits: u64, cos: u64, sin: u64, kc: u64, vc: u64 };

    /// Passes of up to `max_rows` rows over up to `span` positions.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, t: cfg.Text, max_rows: usize, span: usize, st: *store.Store) !Lm {
        if (t.head_dim != 128 or t.heads % t.kv_heads != 0 or t.heads / t.kv_heads > 16) return error.UnsupportedQwen3AsrGeometry;
        const cap = std.mem.alignForward(usize, span, chunk_keys);
        if (cap > t.context) return error.ContextTooLong;
        var lm: Lm = undefined;
        lm.t = t;
        lm.d = d;
        lm.s = s;
        lm.cap = cap;
        lm.g = try cuda.vit.Gemms.init(gpa, d);
        errdefer lm.g.deinit();
        lm.nm = try cuda.norms.Norms.load(d);
        errdefer lm.nm.unload();
        lm.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer lm.qk.unload();
        lm.act = try cuda.act_rows.ActRows.load(d);
        errdefer lm.act.unload();
        lm.emb = try cuda.embed_rows.EmbedRows.load(d);
        errdefer lm.emb.unload();
        lm.attn = try Attn.load(d);
        errdefer lm.attn.mod.unload();
        const r = max_rows;
        const qd = t.heads * t.head_dim;
        const kvd = t.kv_heads * t.head_dim;
        const nch = cap / chunk_keys;
        const kv = t.layers * cap * kvd * 2;
        lm.b = .{
            .ids = try st.alloc(4 * r),
            .pos = try st.alloc(4 * r),
            .meta = try st.alloc(4),
            .q = try st.alloc(2 * r * qd),
            .k = try st.alloc(2 * r * kvd),
            .v = try st.alloc(2 * r * kvd),
            .po = try st.alloc(4 * r * nch * qd),
            .pm = try st.alloc(4 * r * nch * t.heads),
            .pl = try st.alloc(4 * r * nch * t.heads),
            .xs = try st.alloc(4 * r * qd / 64),
            .logits = try st.alloc(2 * t.vocab),
            .cos = try st.alloc(4 * cap * t.head_dim),
            .sin = try st.alloc(4 * cap * t.head_dim),
            .kc = try st.alloc(kv),
            .vc = try st.alloc(kv),
        };
        const cs = try gpa.alloc(f32, cap * t.head_dim);
        defer gpa.free(cs);
        const sn = try gpa.alloc(f32, cap * t.head_dim);
        defer gpa.free(sn);
        shared.ropeTables(t, cs, sn, cap);
        try store.view(d, lm.b.cos, cs.len * 4).upload(0, std.mem.sliceAsBytes(cs));
        try store.view(d, lm.b.sin, sn.len * 4).upload(0, std.mem.sliceAsBytes(sn));
        lm.host_ids = try gpa.alloc(i32, r);
        errdefer gpa.free(lm.host_ids);
        lm.host_pos = try gpa.alloc(i32, r);
        errdefer gpa.free(lm.host_pos);
        lm.logits = try gpa.alloc(u16, t.vocab);
        return lm;
    }

    pub fn deinit(lm: *Lm, gpa: std.mem.Allocator) void {
        gpa.free(lm.logits);
        gpa.free(lm.host_pos);
        gpa.free(lm.host_ids);
        lm.attn.mod.unload();
        lm.emb.unload();
        lm.act.unload();
        lm.qk.unload();
        lm.nm.unload();
        lm.g.deinit();
    }

    /// The pass's ids and positions [pos, pos + ids.len) on the device; the first position is the attention's.
    pub fn positions(lm: *Lm, ids: []const u32, pos: usize) !void {
        const rows = ids.len;
        if (rows > lm.host_ids.len) return error.PassRows;
        if (pos + rows > lm.cap) return error.ContextFull;
        for (ids, lm.host_ids[0..rows], lm.host_pos[0..rows], 0..) |id, *x, *p, i| {
            x.* = @intCast(id);
            p.* = @intCast(pos + i);
        }
        try lm.s.synchronize(); // the last pass may still read them
        try store.view(lm.d, lm.b.ids, 4 * rows).upload(0, std.mem.sliceAsBytes(lm.host_ids[0..rows]));
        try store.view(lm.d, lm.b.pos, 4 * rows).upload(0, std.mem.sliceAsBytes(lm.host_pos[0..rows]));
        try store.view(lm.d, lm.b.meta, 4).upload(0, std.mem.asBytes(&lm.host_pos[0]));
    }

    pub fn dump(lm: *Lm, stage: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        if (lm.sink) |dm| try dm.put(dm.ctx, stage, ptr, rows, cols, null);
    }

    pub fn embed(lm: *Lm, table: u64, out: u64, rows: usize, hidden: usize) !void {
        try lm.emb.bf16(lm.s, lm.b.ids, table, out, rows, hidden);
    }

    pub fn copy(lm: *Lm, dst: u64, src: u64, n: usize) !void {
        try store.view(lm.d, dst, n).copyFrom(0, src, n, lm.s.handle);
    }

    /// h = bf16(x + res) (x when res is null), y = bf16(w bf16(h rsqrt(mean h^2 + eps))).
    pub fn addNorm(lm: *Lm, x: u64, res: ?u64, w: u64, h: u64, y: u64, rows: usize) !void {
        try lm.nm.addRmsTail(lm.s, x, res, w, h, y, null, rows, lm.t.hidden, lm.t.eps, .{ .round = true });
    }

    pub fn gemm(lm: *Lm, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        try lm.g.run(lm.s, x, w, y, rows, n, k, .bf16);
    }

    fn kvAt(lm: *const Lm, base: u64, layer: usize) u64 {
        return base + layer * lm.cap * lm.t.kv_heads * lm.t.head_dim * 2;
    }

    /// q and k RMS-normed per head and rotated, k and v into layer's caches at the pass's positions.
    pub fn qkRope(lm: *Lm, qkv: u64, q_norm: u64, k_norm: u64, layer: usize, rows: usize) !void {
        const t = lm.t;
        const b = lm.b;
        const shape: cuda.qk_rows.Shape = .{ .norms = q_norm, .norm_k = k_norm, .norms_bf16 = true, .cos = b.cos, .sin = b.sin, .rot = t.head_dim, .eps = t.eps, .round = true };
        const out: cuda.qk_rows.Out = .{ .q = b.q, .k = b.k, .v = b.v, .kc = lm.kvAt(b.kc, layer), .vc = lm.kvAt(b.vc, layer), .len = lm.cap };
        try lm.qk.rows(lm.s, qkv, .{ .h = t.heads, .hk = t.kv_heads, .dim = t.head_dim }, shape, b.pos, null, out, rows);
    }

    /// Causal GQA attention of the pass's q rows over layer's caches into out [rows, heads x dim].
    pub fn attention(lm: *Lm, layer: usize, out: u64, rows: usize) !void {
        const t = lm.t;
        const b = lm.b;
        const qd = t.heads * t.head_dim;
        const nch = lm.cap / chunk_keys;
        const grp = t.heads / t.kv_heads;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(t.head_dim))));
        var a: cuda.Args = .{};
        for ([_]u64{ b.q, lm.kvAt(b.kc, layer), lm.kvAt(b.vc, layer), b.meta, b.po, b.pm, b.pl }) |x| a.add(x);
        for ([_]usize{ qd, t.heads, t.kv_heads, grp, chunk_keys, nch }) |x| a.add(@as(c_int, @intCast(x)));
        a.add(scale);
        a.add(@as(u64, 0));
        a.add(@as(i64, 0));
        const f = if (rows > 1) lm.attn.chunk_pf else lm.attn.chunk;
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(rows), .y = @intCast(t.kv_heads), .z = @intCast(@min(nch, chunk_blocks)) };
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, lm.s, &a);
        var m: cuda.Args = .{};
        for ([_]u64{ b.po, b.pm, b.pl, b.meta, out, b.xs }) |x| m.add(x);
        for ([_]usize{ t.heads, t.kv_heads, grp, chunk_keys, nch }) |x| m.add(@as(c_int, @intCast(x)));
        try cuda.launch.launch(lm.attn.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(t.kv_heads), .z = 1 }, .block = .{ .x = 256 } }, lm.s, &m);
    }

    /// out [rows, i] = bf16(silu(gate) up) of gu's [gate | up] rows.
    pub fn swiglu(lm: *Lm, gu: u64, out: u64, rows: usize, i: usize) !void {
        try lm.act.swiglu(lm.s, gu, 2 * i, gu + i * 2, 2 * i, out, null, rows, i);
    }

    /// The tied head over one normed row, downloaded: its greedy pick.
    pub fn greedy(lm: *Lm, row: u64, embed_table: u64) !u32 {
        try lm.g.run(lm.s, row, embed_table, lm.b.logits, 1, lm.t.vocab, lm.t.hidden, .bf16);
        try store.view(lm.d, lm.b.logits, lm.logits.len * 2).downloadAsync(0, std.mem.sliceAsBytes(lm.logits), lm.s.handle);
        try lm.s.synchronize();
        const best = shared.argmax(lm.logits);
        lm.last_margin = shared.margin(lm.logits, best);
        return best;
    }
};
