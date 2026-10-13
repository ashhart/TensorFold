//! Qwen3.5-2B on CUDA: the affine-4 projections through cuda.qlinear, the glue in qwen35.cu, one forward for any
//! mix of streams' windows. Each stream owns its recurrences, convolution tails and KV cache (Cache).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = @import("config.zig");

const Affine4 = cuda.qlinear.Affine4;
const DeviceBuffer = cuda.DeviceBuffer;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = 128;
/// Streams' windows in one forward at most.
pub const max_segs = 32;
/// Rows of a verify forward at most: each row's recurrences are recorded for a partial acceptance.
pub const max_verify = 32;
/// A window's rows at most (exact: one-row steps' bits).
pub const window_rows = 16;
/// Partial attention chunks a row at most: the chunk width grows with the window instead.
const max_chunks = 64;

const deltas = c.layers - c.layers / 4;
const attentions = c.layers / 4;
pub const rec_bytes = c.linear_heads * c.linear_dim * c.linear_dim * 4;
pub const conv_bytes = (c.conv_taps - 1) * c.conv_dim * 2;
const state_bytes = deltas * (rec_bytes + conv_bytes);

/// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values.
pub fn streamBytes(cap: usize) usize {
    return state_bytes + kvBytes(cap);
}

fn kvBytes(cap: usize) usize {
    return attentions * 2 * c.kv_heads * cap * c.head_dim * 2;
}

/// qwen35.cu's Seg.
const Seg = extern struct { rec: u64, conv: u64, kv: u64, cap: c_int, len: c_int, row0: c_int, rows: c_int };

/// One stream's state on the device and the logits of its newest row on the host.
pub const Cache = struct {
    state: DeviceBuffer,
    kv: DeviceBuffer,
    cap: usize,
    len: usize = 0,
    logits: []u16,
    /// The newest verify window: where it started, its first row in that forward, its rows.
    last: ?struct { start: usize, row0: usize, rows: usize } = null,

    pub fn init(gpa: std.mem.Allocator, e: *const Engine, cap: usize) !Cache {
        var state = try DeviceBuffer.alloc(e.d, state_bytes);
        errdefer state.free();
        try state.fill8(0, null);
        var kv = try DeviceBuffer.alloc(e.d, kvBytes(cap));
        errdefer kv.free();
        return .{ .state = state, .kv = kv, .cap = cap, .logits = try gpa.alloc(u16, c.vocab) };
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        self.state.free();
        self.kv.free();
        gpa.free(self.logits);
    }

    fn seg(self: *const Cache, row0: usize, rows: usize) Seg {
        return .{ .rec = self.state.ptr, .conv = self.state.ptr + deltas * rec_bytes, .kv = self.kv.ptr, .cap = @intCast(self.cap), .len = @intCast(self.len), .row0 = @intCast(row0), .rows = @intCast(rows) };
    }
};

const Delta = struct { qkv: Affine4, z: Affine4, a: Affine4, b: Affine4, out: Affine4, conv: u64, norm: u64, a_log: u64, dt_bias: u64 };
const Attention = struct { q: Affine4, k: Affine4, v: Affine4, o: Affine4, q_norm: u64, k_norm: u64 };
const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    gate: Affine4,
    up: Affine4,
    down: Affine4,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

const Fns = struct {
    embed: cuda.Function,
    norm: cuda.Function,
    sums: cuda.Function,
    gdn_pre: cuda.Function,
    conv_commit: cuda.Function,
    gdn_chain: cuda.Function,
    gdn_post: cuda.Function,
    head_norm: cuda.Function,
    rope: cuda.Function,
    attn_partial: cuda.Function,
    attn_merge: cuda.Function,
    attn_gate: cuda.Function,
    mlp_act: cuda.Function,
    argmax: cuda.Function,
};

/// A window of one stream in a forward.
pub const Part = struct { cache: *Cache, rows: usize };

/// What a forward keeps. A prompt chunk (`last`, one window) puts its last row's logits in its cache's `logits`;
/// verify windows keep every row, drawn on the device into `tokens` (`draws`, one a row) or their logits for
/// `rowLogits` (`logits`), and record each row's state for `rewind`.
pub const Keep = union(enum) { last, draws: []const Draw, logits };

/// A kept row's draw on the device: sample.cu's keyed rule at `position`, or greedy without one.
pub const Draw = struct { rule: ?cuda.sample.Rule, position: u64 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    mods: [6]cuda.Module,
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    fns: Fns,
    held: std.ArrayList(DeviceBuffer) = .empty, // weights and scratch, freed together
    // weights
    embed_w: u64,
    embed_s: u64,
    embed_b: u64,
    head: Affine4,
    final_norm: u64,
    layers: [c.layers]Layer,
    // scratch, max_rows rows each
    h: u64,
    x: u64,
    r: u64,
    xs: u64,
    qkv: u64,
    z: u64,
    a: u64,
    b: u64,
    q: u64,
    k: u64,
    v: u64,
    g: u64,
    beta: u64,
    y: u64,
    mix: u64,
    qraw: u64,
    kraw: u64,
    vraw: u64,
    qn: u64,
    kn: u64,
    gate: u64,
    up: u64,
    act: u64,
    pm: u64,
    pl: u64,
    po: u64,
    logits: u64,
    // the per-forward tables: ids, segments, each row's segment and the draws' rules, staged in pinned memory
    tables: DeviceBuffer,
    staged: cuda.HostBuffer,
    host_logits: cuda.HostBuffer,
    /// The kept rows' device draws, when forward was given them.
    tokens: [max_verify]u32 = undefined,
    rec_snap: u64, // recorded recurrences: max_verify rows a linear layer
    conv_snap: u64, // recorded convolution tails, the same

    const ids_at = 0;
    const segs_at = max_rows * 4;
    const rowseg_at = segs_at + max_segs * @sizeOf(Seg);
    const rules_at = rowseg_at + max_rows * 4;
    const meta_at = rules_at + max_verify * @sizeOf(cuda.sample.Rule);
    const tokens_at = meta_at + max_verify * 4;
    const tables_len = tokens_at + max_verify * 4;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const config = try c.Config.read(gpa, io, dir);
        const d = ctx.d;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.d = d;
        e.held = .empty;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qwen35, kk.sample };
        var loaded: usize = 0;
        errdefer for (e.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            e.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        const gb10 = try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1;
        e.affine = try cuda.qlinear.Affine4Kernels.resolve(d, e.mods[0], e.mods[1], e.mods[2], e.mods[3], sms, gb10);
        errdefer e.affine.deinit();
        e.draws = try cuda.sample.Draws.resolve(e.mods[5]);
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(e.fns, name) = try e.mods[4].function("tf_q35_" ++ name);
        errdefer e.freeHeld();
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        try e.loadWeights(&ck, config.tied);
        try e.allocScratch();
        e.tables = try DeviceBuffer.alloc(d, tables_len);
        errdefer e.tables.free();
        e.staged = try cuda.HostBuffer.alloc(d, tables_len);
        errdefer e.staged.free();
        e.host_logits = try cuda.HostBuffer.alloc(d, max_verify * c.vocab * 2);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.host_logits.free();
        e.staged.free();
        e.tables.free();
        e.freeHeld();
        e.affine.deinit();
        for (&e.mods) |*m| m.unload();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    fn freeHeld(e: *Engine) void {
        for (e.held.items) |*b| b.free();
        e.held.deinit(e.gpa);
    }

    fn own(e: *Engine, b: DeviceBuffer) !u64 {
        try e.held.append(e.gpa, b);
        return b.ptr;
    }

    fn alloc(e: *Engine, bytes: usize) !u64 {
        var b = try DeviceBuffer.alloc(e.d, bytes);
        errdefer b.free();
        return e.own(b);
    }

    fn put(e: *Engine, t: core.checkpoint.Tensor) !u64 {
        var b = try DeviceBuffer.fromHost(e.d, t.bytes);
        errdefer b.free();
        return e.own(b);
    }

    fn tensor(e: *Engine, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        var buf: [160]u8 = undefined;
        return e.put(try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), dtype, shape));
    }

    /// An MLX affine-4 projection packed into qlinear's tiles; the checkpoint's own layout leaves the device after.
    fn linear(e: *Engine, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize) !Affine4 {
        var buf: [160]u8 = undefined;
        var raw: [3]DeviceBuffer = undefined;
        var got: usize = 0;
        defer for (raw[0..got]) |*b| b.free();
        const parts = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k / 8 }, .{ "scales", .bf16, k / c.group }, .{ "biases", .bf16, k / c.group } };
        for (parts) |p| {
            const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] });
            raw[got] = try DeviceBuffer.fromHost(e.d, t.bytes);
            got += 1;
        }
        const l = Affine4.layout(n, k);
        const q: Affine4 = .{ .w = try e.alloc(l.words), .s = try e.alloc(l.scales), .b = try e.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
        try e.affine.pack(e.stream, raw[0].ptr, raw[1].ptr, raw[2].ptr, q);
        try e.stream.synchronize();
        return q;
    }

    fn loadWeights(e: *Engine, ck: *core.Checkpoint, tied: bool) !void {
        const m = "language_model.model.";
        e.embed_w = try e.tensor(ck, "{s}embed_tokens.weight", .{m}, .u32, &.{ c.vocab, c.hidden / 8 });
        e.embed_s = try e.tensor(ck, "{s}embed_tokens.scales", .{m}, .bf16, &.{ c.vocab, c.hidden / c.group });
        e.embed_b = try e.tensor(ck, "{s}embed_tokens.biases", .{m}, .bf16, &.{ c.vocab, c.hidden / c.group });
        e.head = if (tied) try e.linear(ck, m, "embed_tokens", c.vocab, c.hidden) else try e.linear(ck, "language_model.", "lm_head", c.vocab, c.hidden);
        e.final_norm = try e.tensor(ck, "{s}norm.weight", .{m}, .bf16, &.{c.hidden});
        for (&e.layers, 0..) |*l, i| {
            var pbuf: [96]u8 = undefined;
            const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
            l.* = .{
                .input_norm = try e.tensor(ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{c.hidden}),
                .post_norm = try e.tensor(ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{c.hidden}),
                .gate = try e.linear(ck, p, "mlp.gate_proj", c.intermediate, c.hidden),
                .up = try e.linear(ck, p, "mlp.up_proj", c.intermediate, c.hidden),
                .down = try e.linear(ck, p, "mlp.down_proj", c.hidden, c.intermediate),
                .mixer = undefined,
            };
            if (c.linear(i)) {
                l.mixer = .{ .delta = .{
                    .qkv = try e.linear(ck, p, "linear_attn.in_proj_qkv", c.conv_dim, c.hidden),
                    .z = try e.linear(ck, p, "linear_attn.in_proj_z", c.hidden, c.hidden),
                    .a = try e.linear(ck, p, "linear_attn.in_proj_a", c.linear_heads, c.hidden),
                    .b = try e.linear(ck, p, "linear_attn.in_proj_b", c.linear_heads, c.hidden),
                    .out = try e.linear(ck, p, "linear_attn.out_proj", c.hidden, c.hidden),
                    .conv = try e.tensor(ck, "{s}linear_attn.conv1d.weight", .{p}, .bf16, &.{ c.conv_dim, c.conv_taps, 1 }),
                    .norm = try e.tensor(ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{c.linear_dim}),
                    .a_log = try e.tensor(ck, "{s}linear_attn.A_log", .{p}, .f32, &.{c.linear_heads}),
                    .dt_bias = try e.tensor(ck, "{s}linear_attn.dt_bias", .{p}, .bf16, &.{c.linear_heads}),
                } };
            } else {
                l.mixer = .{ .attention = .{
                    .q = try e.linear(ck, p, "self_attn.q_proj", 2 * c.hidden, c.hidden),
                    .k = try e.linear(ck, p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden),
                    .v = try e.linear(ck, p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden),
                    .o = try e.linear(ck, p, "self_attn.o_proj", c.hidden, c.hidden),
                    .q_norm = try e.tensor(ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{c.head_dim}),
                    .k_norm = try e.tensor(ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{c.head_dim}),
                } };
            }
        }
    }

    fn allocScratch(e: *Engine) !void {
        const R = max_rows;
        inline for (.{ "h", "x", "r", "z", "q", "k", "v", "y", "mix", "qn" }) |name| @field(e, name) = try e.alloc(R * c.hidden * 2);
        inline for (.{ "qkv", "gate", "up", "act" }) |name| @field(e, name) = try e.alloc(R * c.intermediate * 2);
        e.xs = try e.alloc(R * (c.intermediate / 64) * 4);
        inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try e.alloc(R * c.linear_heads * 2);
        e.g = try e.alloc(R * c.linear_heads * 4);
        e.qraw = try e.alloc(R * 2 * c.hidden * 2);
        inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try e.alloc(R * c.kv_heads * c.head_dim * 2);
        e.pm = try e.alloc(R * c.query_heads * max_chunks * 4);
        e.pl = try e.alloc(R * c.query_heads * max_chunks * 4);
        e.po = try e.alloc(R * c.query_heads * max_chunks * c.head_dim * 4);
        e.logits = try e.alloc(max_verify * c.vocab * 2);
        e.rec_snap = try e.alloc(deltas * max_verify * rec_bytes);
        e.conv_snap = try e.alloc(deltas * max_verify * conv_bytes);
    }

    fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// out (rows, n) = x (rows, k) through `l`: decode tiles 16 rows at a time (each row's bits whatever the batch),
    /// or the prompt GEMM for a prompt chunk wider than one tile.
    fn proj(e: *Engine, l: Affine4, x: u64, out: u64, rows: usize, prompt: bool) !void {
        if (prompt and rows > 16) return e.affine.prompt(e.stream, x, l, out, rows);
        try e.run(e.fns.sums, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ x, e.xs, @as(c_int, @intCast(l.k)) });
        var at: usize = 0;
        while (at < rows) : (at += 16) {
            const n = @min(16, rows - at);
            try e.affine.decode(e.stream, x + at * l.k * 2, e.xs + at * (l.k / 64) * 4, l, out + at * l.n * 2, n);
        }
    }

    fn norm(e: *Engine, residual: bool, w: u64, rows: usize) !void {
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, @as(u64, 0), c.eps });
    }

    /// One forward over `parts`' windows, the rows of `ids` in order. `keep` names the rows it keeps and how.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        var total: usize = 0;
        var widest: usize = 0;
        for (parts) |p| {
            if (p.rows == 0 or p.cache.len + p.rows > p.cache.cap) return error.QwenContextFull;
            total += p.rows;
            widest = @max(widest, p.cache.len + p.rows);
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        const record = keep != .last;
        if (record and total > max_verify) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= c.vocab) return error.InvalidQwenToken;
        const heads: usize = if (record) total else 1;
        // the prompt GEMM only for prompt chunks: a window's rows keep the bits of one-row steps
        const prompt = !record;
        // the tables
        const st = e.staged.bytes;
        @memcpy(std.mem.bytesAsSlice(u32, st[ids_at..][0 .. total * 4]), ids);
        const segs = std.mem.bytesAsSlice(Seg, st[segs_at..][0 .. parts.len * @sizeOf(Seg)]);
        const rowseg = std.mem.bytesAsSlice(c_int, st[rowseg_at..][0 .. total * 4]);
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            segs[i] = p.cache.seg(row0, p.rows);
            for (row0..row0 + p.rows) |r| rowseg[r] = @intCast(i);
            p.cache.last = if (record) .{ .start = p.cache.len, .row0 = row0, .rows = p.rows } else null;
            row0 += p.rows;
        }
        if (keep == .draws) {
            const rules = std.mem.bytesAsSlice(cuda.sample.Rule, st[rules_at..][0 .. total * @sizeOf(cuda.sample.Rule)]);
            const meta = std.mem.bytesAsSlice(c_int, st[meta_at..][0 .. total * 4]);
            for (keep.draws, rules, meta) |p, *r, *m| {
                if (p.rule) |x| r.* = x;
                m.* = @intCast(p.position -% 1); // sample.cu keys row r at meta + r + 1
            }
        }
        try e.tables.uploadAsync(0, st, e.stream.handle);
        const t_ids = e.tables.ptr + ids_at;
        const t_segs = e.tables.ptr + segs_at;
        const t_rows = e.tables.ptr + rowseg_at;
        const nseg = parts.len;
        // attention chunks: at most max_chunks a row, each a multiple of 128 keys
        const ck = @max(128, (widest + max_chunks * 128 - 1) / (max_chunks * 128) * 128);
        const nch = (widest + ck - 1) / ck;

        try e.run(e.fns.embed, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ t_ids, e.embed_w, e.embed_s, e.embed_b, e.h });
        var di: usize = 0;
        var ai: c_int = 0;
        for (e.layers, 0..) |l, i| {
            try e.norm(i != 0, l.input_norm, total);
            switch (l.mixer) {
                .delta => |m| {
                    const layer: c_int = @intCast(di);
                    const rec_snap: u64 = if (record) e.rec_snap + di * max_verify * rec_bytes else 0;
                    const conv_snap: u64 = if (record) e.conv_snap + di * max_verify * conv_bytes else 0;
                    try e.proj(m.qkv, e.x, e.qkv, total, prompt);
                    try e.proj(m.z, e.x, e.z, total, prompt);
                    try e.proj(m.a, e.x, e.a, total, prompt);
                    try e.proj(m.b, e.x, e.b, total, prompt);
                    try e.run(e.fns.gdn_pre, .{ 1, 3 * c.linear_heads, total }, .{ 32, 1, 1 }, .{ e.qkv, t_segs, t_rows, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g, e.beta });
                    try e.run(e.fns.conv_commit, .{ (c.conv_dim + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ e.qkv, t_segs, layer, conv_snap });
                    try e.run(e.fns.gdn_chain, .{ 1, c.linear_dim / 4, c.linear_heads * nseg }, .{ 32, 4, 1 }, .{ e.q, e.k, e.v, e.g, e.beta, t_segs, layer, e.y, rec_snap });
                    try e.run(e.fns.gdn_post, .{ 1, c.linear_heads, total }, .{ 32, 1, 1 }, .{ e.y, e.z, m.norm, c.eps, e.mix });
                    try e.proj(m.out, e.mix, e.r, total, prompt);
                    di += 1;
                },
                .attention => |m| {
                    try e.proj(m.q, e.x, e.qraw, total, prompt);
                    try e.proj(m.k, e.x, e.kraw, total, prompt);
                    try e.proj(m.v, e.x, e.vraw, total, prompt);
                    try e.run(e.fns.head_norm, .{ total * c.query_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, c.query_heads), @as(c_int, 2 * c.hidden), @as(c_int, 2 * c.head_dim), c.eps });
                    try e.run(e.fns.head_norm, .{ total * c.kv_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, c.kv_heads), @as(c_int, c.kv_heads * c.head_dim), @as(c_int, c.head_dim), c.eps });
                    try e.run(e.fns.rope, .{ total, 1, 1 }, .{ c.head_dim, 1, 1 }, .{ e.qn, e.kn, e.vraw, t_segs, t_rows, ai });
                    try e.run(e.fns.attn_partial, .{ nch, c.kv_heads, total }, .{ 128, 1, 1 }, .{ e.qn, t_segs, t_rows, ai, @as(c_int, @intCast(nch)), @as(c_int, @intCast(ck)), @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
                    try e.run(e.fns.attn_merge, .{ c.query_heads, total, 1 }, .{ 32, 1, 1 }, .{ e.pm, e.pl, e.po, @as(c_int, @intCast(nch)), e.y });
                    try e.run(e.fns.attn_gate, .{ (total * c.hidden + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.y, e.qraw, e.mix, @as(c_int, @intCast(total)) });
                    try e.proj(m.o, e.mix, e.r, total, prompt);
                    ai += 1;
                },
            }
            try e.norm(true, l.post_norm, total);
            try e.proj(l.gate, e.x, e.gate, total, prompt);
            try e.proj(l.up, e.x, e.up, total, prompt);
            const n: c_longlong = @intCast(total * c.intermediate);
            try e.run(e.fns.mlp_act, .{ (total * c.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n });
            try e.proj(l.down, e.act, e.r, total, prompt);
        }
        try e.norm(true, e.final_norm, total);
        // the head over the kept rows
        try e.proj(e.head, e.x + (total - heads) * c.hidden * 2, e.logits, heads, false);
        for (parts) |p| p.cache.len += p.rows;
        switch (keep) {
            .draws => |pk| {
                const toks = e.tables.ptr + tokens_at;
                for (pk, 0..) |p, i| {
                    const row = e.logits + i * c.vocab * 2;
                    if (p.rule != null) {
                        try e.draws.draw(e.stream, row, c.vocab, e.tables.ptr + rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + meta_at + i * 4, 0, toks + i * 4, 1, null, null);
                    } else try e.run(e.fns.argmax, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, .{ row, @as(c_int, c.vocab), toks + i * 4 });
                }
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.host_logits.bytes.ptr, toks, total * 4, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                @memcpy(e.tokens[0..total], e.host_logits.slice(u32)[0..total]);
            },
            .logits, .last => {
                const out = e.host_logits.bytes[0 .. heads * c.vocab * 2];
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, e.logits, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                if (keep == .last) @memcpy(parts[0].cache.logits, e.rowLogits(0));
            },
        }
    }

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) []const u16 {
        return e.host_logits.slice(u16)[i * c.vocab ..][0..c.vocab];
    }

    /// Keeps the first `rows` of the cache's newest window: its recurrences and convolution tails as they were after
    /// that row (the forward recorded them), its length at that row. The window's later keys stay as unread slots.
    pub fn rewind(e: *Engine, cache: *Cache, rows: usize) !void {
        const last = cache.last orelse return error.NothingToKeep;
        if (rows == 0 or rows > last.rows) return error.NothingToKeep;
        cache.last = null;
        cache.len = last.start + rows;
        if (rows == last.rows) return;
        const row = last.row0 + rows - 1;
        for (0..deltas) |di| {
            try cache.state.copyFrom(di * rec_bytes, e.rec_snap + (di * max_verify + row) * rec_bytes, rec_bytes, e.stream.handle);
            try cache.state.copyFrom(deltas * rec_bytes + di * conv_bytes, e.conv_snap + (di * max_verify + row) * conv_bytes, conv_bytes, e.stream.handle);
        }
        try e.stream.synchronize();
    }
};
