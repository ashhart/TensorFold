//! Qwen3.5's hybrid text models on CUDA (the 2B, Qwen3.8-27B): the affine-4 projections through cuda.qlinear, the
//! glue in qwen35.cu, one forward for any mix of streams' windows. Each stream owns its recurrences, convolution tails
//! and KV cache (Cache); the widths come from config.json (cuda_geometry.zig), the weights from cuda_weights.zig.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");

pub const Geometry = geometry.Geometry;
const Affine4 = cuda.qlinear.Affine4;
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = 128;
/// Streams' windows in one forward at most.
pub const max_segs = 32;
/// Rows of a verify forward at most: each row's recurrences are recorded for a partial acceptance.
const most_verify = 32;
/// A window's rows at most (exact: one-row steps' bits).
const most_window = 16;
/// Device bytes the recorded rows may take: a wide model records fewer rows a forward.
const snapshot_budget = 1280 << 20;
/// Partial attention chunks a row at most: the chunk width grows with the window instead.
const max_chunks = 64;

/// Device allocations are 256-byte aligned, as the driver's are.
pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// qwen35.cu's Geo and Seg.
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int };
const Seg = extern struct { rec: u64, conv: u64, kv: u64, cap: c_int, len: c_int, row0: c_int, rows: c_int };

/// One stream's state on the device and the logits of its newest prompt chunk on the host.
pub const Cache = struct {
    state: DeviceBuffer,
    kv: DeviceBuffer,
    cap: usize,
    len: usize = 0,
    logits: []u16,
    /// The newest verify window: where it started, its first row in that forward, its rows.
    last: ?struct { start: usize, row0: usize, rows: usize } = null,

    pub fn init(gpa: std.mem.Allocator, e: *const Engine, cap: usize) !Cache {
        var state = try DeviceBuffer.alloc(e.d, e.g.stateBytes());
        errdefer state.free();
        try state.fill8(0, null);
        var kv = try DeviceBuffer.alloc(e.d, e.g.kvBytes(cap));
        errdefer kv.free();
        try kv.fill8(0, null); // touched now, not as the first request's keys land (the driver commits on first touch)
        return .{ .state = state, .kv = kv, .cap = cap, .logits = try gpa.alloc(u16, e.g.vocab) };
    }

    /// Ready for another stream: no rows, the recurrences and tails zero (on the engine's stream, before its forwards).
    pub fn reset(self: *Cache, e: *const Engine) !void {
        self.len = 0;
        self.last = null;
        try self.state.fill8(0, e.stream.handle);
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        self.state.free();
        self.kv.free();
        gpa.free(self.logits);
    }

    fn seg(self: *const Cache, g: Geometry, row0: usize, rows: usize) Seg {
        return .{ .rec = self.state.ptr, .conv = self.state.ptr + g.deltas() * g.recBytes(), .kv = self.kv.ptr, .cap = @intCast(self.cap), .len = @intCast(self.len), .row0 = @intCast(row0), .rows = @intCast(rows) };
    }
};

pub const Delta = struct { qkv: Affine4, z: Affine4, a: Affine4, b: Affine4, out: Affine4, conv: u64, norm: u64, a_log: u64, dt_bias: u64 };
pub const Attention = struct { q: Affine4, k: Affine4, v: Affine4, o: Affine4, q_norm: u64, k_norm: u64 };
pub const Layer = struct {
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
    g: Geometry,
    geo: Geo,
    stream: cuda.Stream,
    mods: [6]cuda.Module,
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    fns: Fns,
    /// Every weight and scratch buffer in one allocation: the WDDM driver's launch cost grows with a process's
    /// allocation count, and a 64-layer model would have thousands.
    arena: DeviceBuffer,
    arena_at: usize = 0,
    stage: DeviceBuffer, // the load's staging of each projection as stored, packed from there
    /// Rows a verify forward records (snapshot_budget over a row's states), and a window's rows at most.
    verify_rows: usize,
    window_rows: usize,
    // weights
    embed_w: u64,
    embed_s: u64,
    embed_b: u64,
    head: Affine4,
    final_norm: u64,
    layers: []Layer,
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
    g_: u64,
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
    rec_snap: u64, // recorded recurrences: verify_rows rows a linear layer
    conv_snap: u64, // recorded convolution tails, the same
    // the per-forward tables: ids, segments, each row's segment and the draws' rules, staged in pinned memory
    tables: DeviceBuffer,
    staged: cuda.HostBuffer,
    host_logits: cuda.HostBuffer,
    /// The kept rows' device draws, when forward was given them.
    tokens: [most_verify]u32 = undefined,

    const ids_at = 0;
    const segs_at = max_rows * 4;
    const rowseg_at = segs_at + max_segs * @sizeOf(Seg);
    const rules_at = rowseg_at + max_rows * 4;
    const meta_at = rules_at + most_verify * @sizeOf(cuda.sample.Rule);
    const tokens_at = meta_at + most_verify * 4;
    const tables_len = tokens_at + most_verify * 4;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const g = try Geometry.read(gpa, io, dir);
        const d = ctx.d;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.d = d;
        e.g = g;
        e.geo = .{ .hid = @intCast(g.hidden), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .qh = @intCast(g.q_heads), .kvh = @intCast(g.kv_heads) };
        e.verify_rows = std.math.clamp(snapshot_budget / (g.deltas() * (g.recBytes() + g.convBytes())), 1, most_verify);
        e.window_rows = @min(most_window, e.verify_rows);
        e.arena_at = 0;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.arena = try DeviceBuffer.alloc(d, weights.total(g) + try e.scratch(.measure));
        errdefer e.arena.free();
        try e.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        e.stage = try DeviceBuffer.alloc(d, weights.stageBytes(g));
        errdefer e.stage.free();
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
        e.layers = try gpa.alloc(Layer, g.layers);
        errdefer gpa.free(e.layers);
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        try weights.load(e, &ck);
        _ = try e.scratch(.allocate);
        e.tables = try DeviceBuffer.alloc(d, tables_len);
        errdefer e.tables.free();
        e.staged = try cuda.HostBuffer.alloc(d, tables_len);
        errdefer e.staged.free();
        e.host_logits = try cuda.HostBuffer.alloc(d, most_verify * g.vocab * 2);
        errdefer e.host_logits.free();
        // every path a request takes, once now: the WDDM driver stalls a process's first forwards for seconds
        // (a 16 GiB model: a 20 s launch, then a 3 s one), which no request should pay
        var warm = try Cache.init(gpa, e, 64);
        defer warm.deinit(gpa);
        const ids: [32]u32 = @splat(0);
        try e.forward(&.{.{ .cache = &warm, .rows = ids.len }}, &ids, .last); // a prompt chunk: the prompt GEMMs
        try e.forward(&.{.{ .cache = &warm, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = null, .position = 33 }} });
        try e.forward(&.{.{ .cache = &warm, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = cuda.sample.rule(1, 0.7, 0.95, -std.math.inf(f64), 20), .position = 34 }} });
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.host_logits.free();
        e.staged.free();
        e.tables.free();
        e.stage.free();
        e.arena.free();
        e.gpa.free(e.layers);
        e.affine.deinit();
        for (&e.mods) |*m| m.unload();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values.
    pub fn streamBytes(e: *const Engine, cap: usize) usize {
        return e.g.stateBytes() + e.g.kvBytes(cap);
    }

    /// The next `bytes` of the arena, 256-byte aligned.
    pub fn alloc(e: *Engine, bytes: usize) !u64 {
        const at = aligned(e.arena_at);
        if (at + bytes > e.arena.len) return error.ArenaExhausted;
        e.arena_at = at + bytes;
        return e.arena.ptr + at;
    }

    /// The scratch rows' bytes (`measure`: summed, nothing taken), or the rows themselves from the arena.
    fn scratch(e: *Engine, mode: enum { measure, allocate }) !usize {
        var total: usize = 0;
        const Take = struct {
            e: *Engine,
            mode: @TypeOf(mode),
            total: *usize,
            fn alloc(t: @This(), bytes: usize) !u64 {
                t.total.* += aligned(bytes);
                return if (t.mode == .measure) 0 else try t.e.alloc(bytes);
            }
        };
        const t = Take{ .e = e, .mode = mode, .total = &total };
        const g = e.g;
        const R = max_rows;
        const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden); // y and mix: either mixer's rows
        inline for (.{ "h", "x", "r" }) |name| @field(e, name) = try t.alloc(R * g.hidden * 2);
        inline for (.{ "y", "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(R * wide * 2);
        inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(R * g.k_heads * geometry.dk * 2);
        e.qkv = try t.alloc(R * g.convWidth() * 2);
        inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(R * g.intermediate * 2);
        e.xs = try t.alloc(R * (@max(g.intermediate, wide) / 64) * 4);
        inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(R * g.v_heads * 2);
        e.g_ = try t.alloc(R * g.v_heads * 4);
        e.qraw = try t.alloc(R * 2 * g.attnWidth() * 2);
        e.qn = try t.alloc(R * g.attnWidth() * 2);
        inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(R * g.kv_heads * head_dim * 2);
        e.pm = try t.alloc(R * g.q_heads * max_chunks * 4);
        e.pl = try t.alloc(R * g.q_heads * max_chunks * 4);
        e.po = try t.alloc(R * g.q_heads * max_chunks * head_dim * 4);
        e.logits = try t.alloc(most_verify * g.vocab * 2);
        e.rec_snap = try t.alloc(g.deltas() * e.verify_rows * g.recBytes());
        e.conv_snap = try t.alloc(g.deltas() * e.verify_rows * g.convBytes());
        return total;
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
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, e.g.eps, @as(c_int, @intCast(e.g.hidden)) });
    }

    /// One forward over `parts`' windows, the rows of `ids` in order. `keep` names the rows it keeps and how.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        const g = e.g;
        var total: usize = 0;
        var widest: usize = 0;
        for (parts) |p| {
            if (p.rows == 0 or p.cache.len + p.rows > p.cache.cap) return error.QwenContextFull;
            total += p.rows;
            widest = @max(widest, p.cache.len + p.rows);
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        const record = keep != .last;
        if (record and total > e.verify_rows) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= g.vocab) return error.InvalidQwenToken;
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
            segs[i] = p.cache.seg(g, row0, p.rows);
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
        const geo = e.geo;
        // attention chunks: at most max_chunks a row, each a multiple of 128 keys
        const ck = @max(128, (widest + max_chunks * 128 - 1) / (max_chunks * 128) * 128);
        const nch = (widest + ck - 1) / ck;
        const group = g.q_heads / g.kv_heads;

        try e.run(e.fns.embed, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ t_ids, e.embed_w, e.embed_s, e.embed_b, e.h, @as(c_int, @intCast(g.hidden)) });
        var di: usize = 0;
        var ai: c_int = 0;
        for (e.layers, 0..) |l, i| {
            try e.norm(i != 0, l.input_norm, total);
            switch (l.mixer) {
                .delta => |m| {
                    const layer: c_int = @intCast(di);
                    const rec_snap: u64 = if (record) e.rec_snap + di * e.verify_rows * g.recBytes() else 0;
                    const conv_snap: u64 = if (record) e.conv_snap + di * e.verify_rows * g.convBytes() else 0;
                    try e.proj(m.qkv, e.x, e.qkv, total, prompt);
                    try e.proj(m.z, e.x, e.z, total, prompt);
                    try e.proj(m.a, e.x, e.a, total, prompt);
                    try e.proj(m.b, e.x, e.b, total, prompt);
                    try e.run(e.fns.gdn_pre, .{ 1, 2 * g.k_heads + g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.qkv, t_segs, t_rows, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta });
                    try e.run(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, layer, conv_snap });
                    try e.run(e.fns.gdn_chain, .{ 1, geometry.dk / 4, g.v_heads * nseg }, .{ 32, 4, 1 }, .{ geo, e.q, e.k, e.v, e.g_, e.beta, t_segs, layer, e.y, rec_snap });
                    try e.run(e.fns.gdn_post, .{ 1, g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix });
                    try e.proj(m.out, e.mix, e.r, total, prompt);
                    di += 1;
                },
                .attention => |m| {
                    const aw = g.attnWidth();
                    try e.proj(m.q, e.x, e.qraw, total, prompt);
                    try e.proj(m.k, e.x, e.kraw, total, prompt);
                    try e.proj(m.v, e.x, e.vraw, total, prompt);
                    try e.run(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps });
                    try e.run(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps });
                    try e.run(e.fns.rope, .{ total, 1, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, ai });
                    try e.run(e.fns.attn_partial, .{ nch, g.kv_heads, total }, .{ 32 * group, 1, 1 }, .{ geo, e.qn, t_segs, t_rows, ai, @as(c_int, @intCast(nch)), @as(c_int, @intCast(ck)), @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
                    try e.run(e.fns.attn_merge, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, @as(c_int, @intCast(nch)), e.y });
                    try e.run(e.fns.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)) });
                    try e.proj(m.o, e.mix, e.r, total, prompt);
                    ai += 1;
                },
            }
            try e.norm(true, l.post_norm, total);
            try e.proj(l.gate, e.x, e.gate, total, prompt);
            try e.proj(l.up, e.x, e.up, total, prompt);
            const n: c_longlong = @intCast(total * g.intermediate);
            try e.run(e.fns.mlp_act, .{ (total * g.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n });
            try e.proj(l.down, e.act, e.r, total, prompt);
        }
        try e.norm(true, e.final_norm, total);
        // the head over the kept rows
        try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
        for (parts) |p| p.cache.len += p.rows;
        switch (keep) {
            .draws => |pk| {
                const toks = e.tables.ptr + tokens_at;
                for (pk, 0..) |p, i| {
                    const row = e.logits + i * g.vocab * 2;
                    if (p.rule != null) {
                        try e.draws.draw(e.stream, row, g.vocab, e.tables.ptr + rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + meta_at + i * 4, 0, toks + i * 4, 1, null, null);
                    } else try e.run(e.fns.argmax, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, .{ row, @as(c_int, @intCast(g.vocab)), toks + i * 4 });
                }
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.host_logits.bytes.ptr, toks, total * 4, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                @memcpy(e.tokens[0..total], e.host_logits.slice(u32)[0..total]);
            },
            .logits, .last => {
                const out = e.host_logits.bytes[0 .. heads * g.vocab * 2];
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, e.logits, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                if (keep == .last) @memcpy(parts[0].cache.logits, e.rowLogits(0));
            },
        }
    }

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) []const u16 {
        return e.host_logits.slice(u16)[i * e.g.vocab ..][0..e.g.vocab];
    }

    /// Keeps the first `rows` of the cache's newest window: its recurrences and convolution tails as they were after
    /// that row (the forward recorded them), its length at that row. The window's later keys stay as unread slots.
    pub fn rewind(e: *Engine, cache: *Cache, rows: usize) !void {
        const g = e.g;
        const last = cache.last orelse return error.NothingToKeep;
        if (rows == 0 or rows > last.rows) return error.NothingToKeep;
        cache.last = null;
        cache.len = last.start + rows;
        if (rows == last.rows) return;
        const row = last.row0 + rows - 1;
        const rb = g.recBytes();
        const cb = g.convBytes();
        for (0..g.deltas()) |di| {
            try cache.state.copyFrom(di * rb, e.rec_snap + (di * e.verify_rows + row) * rb, rb, e.stream.handle);
            try cache.state.copyFrom(g.deltas() * rb + di * cb, e.conv_snap + (di * e.verify_rows + row) * cb, cb, e.stream.handle);
        }
        try e.stream.synchronize();
    }
};
