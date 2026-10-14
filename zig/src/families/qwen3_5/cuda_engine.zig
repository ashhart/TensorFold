//! Qwen3.5's hybrids on CUDA: qlinear projections, qwen35.cu glue, one captured forward over every stream's window.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const caches = @import("cuda_cache.zig");

pub const Geometry = geometry.Geometry;
pub const Cache = caches.Cache;
pub const tap_rows = caches.tap_rows;
const Seg = caches.Seg;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = 128;
/// Streams' windows in one forward at most.
pub const max_segs = 32;
/// Rows of a verify forward at most: each row's logits are kept and drawn.
pub const most_verify = 32;
/// A window's rows at most (exact: one-row steps' bits; qwen35.cuh's MAXW, the rows a stream's log holds).
pub const most_window = 16;
/// Partial attention chunks a row at most (qwen35.cuh's MAXCH): each row's chunk width grows with its own keys.
const max_chunks = 64;

/// Device allocations are 256-byte aligned, as the driver's are.
pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// qwen35.cuh's Geo.
pub const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };

/// A projection on the tiles of the width it is served at: Affine4's for 4-bit codes, AffineN's for the rest.
pub const Proj = union(enum) {
    a4: Affine4,
    an: AffineN,

    pub fn outputs(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.n,
        };
    }

    pub fn inputs(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.k,
        };
    }
};

pub const Delta = struct { qkv: Proj, z: Proj, a: Proj, b: Proj, out: Proj, conv: u64, norm: u64, a_log: u64, dt_bias: u64 };
pub const Attention = struct { q: Proj, k: Proj, v: Proj, o: Proj, q_norm: u64, k_norm: u64 };
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    gate: Proj,
    up: Proj,
    down: Proj,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

const Fns = struct {
    embed: cuda.Function,
    embed8: cuda.Function,
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
    tap: cuda.Function,
};

/// qwen35_keep.cu's kernels (Cache.keep).
pub const Keeps = struct { apply: cuda.Function, conv: cuda.Function, compact: cuda.Function };

/// A window of one stream in a forward: `parents`, for a tree, names each row's parent row (-1: the first row).
pub const Part = struct { cache: *Cache, rows: usize, parents: ?[]const i32 = null };

/// What a forward keeps: a prompt chunk its last row's logits, a verify window every row's draw or logits and log.
pub const Keep = union(enum) { last, draws: []const Draw, logits };

/// A kept row's draw on the device: sample.cu's keyed rule at `position`, or greedy without one.
pub const Draw = struct { rule: ?cuda.sample.Rule, position: u64 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: Geometry,
    geo: Geo,
    stream: cuda.Stream,
    mods: [8]cuda.Module,
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    wide: cuda.qlinear.AffineNKernels, // the tiles of every width but 4, and their packs
    fns: Fns,
    keeps: Keeps,
    /// Every weight and scratch buffer in one allocation: WDDM's launch cost grows with the allocation count.
    arena: DeviceBuffer,
    arena_at: usize = 0,
    stage: DeviceBuffer, // the load's staging of each projection as stored, packed from there
    /// Rows a verify forward takes at most, and a window's rows at most.
    verify_rows: usize,
    window_rows: usize,
    // weights
    embed_w: u64,
    embed_s: u64,
    embed_b: u64,
    embed_bits: u8, // the embedding's rows' width: 4 or 8
    head: Proj,
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
    // per-forward tables (ids, segments, rows' segment, parent and ancestors, the draws' rules) staged in pinned memory
    tables: DeviceBuffer,
    staged: cuda.HostBuffer,
    host_logits: cuda.HostBuffer,
    /// The kept rows' device draws, when forward was given them.
    tokens: [most_verify]u32 = undefined,
    /// The captured forwards by shape (`Shape.key`): the launches from the embedding to the head.
    graphs: std.AutoHashMapUnmanaged(u32, cuda.graph.Exec) = .empty,
    /// A drafter's tapped layers (`setTaps`): each row's residual after them lands in its stream's tap ring.
    taps: [8]u32 = undefined,
    n_taps: usize = 0,

    const ids_at = 0;
    const segs_at = max_rows * 4;
    const rowseg_at = segs_at + max_segs * @sizeOf(Seg);
    const rowpar_at = rowseg_at + max_rows * 4;
    const rowanc_at = rowpar_at + max_rows * 4;
    const rules_at = rowanc_at + max_rows * 4;
    const meta_at = rules_at + most_verify * @sizeOf(cuda.sample.Rule);
    const tokens_at = meta_at + most_verify * 4;
    const params_at = tokens_at + most_verify * 4; // two unused words, the rows, the windows, record
    const tables_len = params_at + 8 * 4;

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
        e.geo = .{ .hid = @intCast(g.hidden), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .qh = @intCast(g.q_heads), .kvh = @intCast(g.kv_heads), .nl = @intCast(g.deltas()) };
        e.verify_rows = most_verify;
        e.window_rows = most_window;
        e.arena_at = 0;
        e.graphs = .empty;
        e.n_taps = 0;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.arena = try DeviceBuffer.alloc(d, weights.total(g) + try e.scratch(.measure));
        errdefer e.arena.free();
        try e.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        e.stage = try DeviceBuffer.alloc(d, weights.stageBytes(g));
        errdefer e.stage.free();
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qwen35, kk.sample, kk.qwen35_keep, kk.affine8 };
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
        e.wide = try cuda.qlinear.AffineNKernels.resolve(d, e.mods[7], e.mods[3], sms, gb10);
        errdefer e.wide.deinit();
        e.draws = try cuda.sample.Draws.resolve(e.mods[5]);
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(e.fns, name) = try e.mods[4].function("tf_q35_" ++ name);
        inline for (@typeInfo(Keeps).@"struct".field_names) |name| @field(e.keeps, name) = try e.mods[6].function("tf_q35k_" ++ name);
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
        try e.warm(gpa);
        return e;
    }

    /// Every path a request takes, run once now: WDDM stalls a process's first forwards for seconds (a 20 s launch).
    fn warm(e: *Engine, gpa: std.mem.Allocator) !void {
        var warm_cache = try Cache.init(gpa, e, 64);
        defer warm_cache.deinit(gpa);
        const ids: [32]u32 = @splat(0);
        const greedy: Draw = .{ .rule = null, .position = 33 };
        try e.forward(&.{.{ .cache = &warm_cache, .rows = ids.len }}, &ids, .last); // a prompt chunk: the prompt GEMMs
        try e.forward(&.{.{ .cache = &warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{greedy} });
        try e.forward(&.{.{ .cache = &warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = .{ .seed = 1, .inv_t = 1.0 / 0.7, .top_p = 0.95, .near = 20.0, .min_log = -std.math.inf(f32), .top_k = 20 }, .position = 34 }} });
        // a tree window and a branch of it kept: the keep kernels, and the forward's branch path
        try e.forward(&.{.{ .cache = &warm_cache, .rows = 3, .parents = &.{ -1, 0, 0 } }}, ids[0..3], .{ .draws = &.{ greedy, greedy, greedy } });
        try warm_cache.keep(e, &.{ 0, 2 });
        try e.stream.synchronize();
    }

    pub fn deinit(e: *Engine) void {
        var it = e.graphs.valueIterator();
        while (it.next()) |x| x.deinit();
        e.graphs.deinit(e.gpa);
        e.host_logits.free();
        e.staged.free();
        e.tables.free();
        e.stage.free();
        e.arena.free();
        e.gpa.free(e.layers);
        e.affine.deinit();
        e.wide.deinit();
        for (&e.mods) |*m| m.unload();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values, taps, log.
    pub fn streamBytes(e: *const Engine, cap: usize) usize {
        return e.g.stateBytes() + e.g.kvBytes(cap) + e.tapBytes() + e.g.logBytes(most_window);
    }

    /// A stream's tap ring: tap_rows rows of the tapped layers' residuals.
    pub fn tapBytes(e: *const Engine) usize {
        return tap_rows * e.n_taps * e.g.hidden * 2;
    }

    /// The layers a drafter taps, told before the streams' caches exist; each forward then fills their rings.
    pub fn setTaps(e: *Engine, taps: []const u32) !void {
        if (taps.len > e.taps.len) return error.TooManyTaps;
        for (taps) |t| if (t >= e.g.layers) return error.BadTap;
        @memcpy(e.taps[0..taps.len], taps);
        e.n_taps = taps.len;
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
        return total;
    }

    pub fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// out (rows, n) = x (rows, k) through `l`: 16-row decode tiles (a row's bits batch-independent) or a prompt GEMM.
    pub fn proj(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
        return e.projSums(l, x, e.xs, out, rows, prompt, e.tables.ptr + params_at);
    }

    /// `proj` with the rows' group sums in `xs` and the device counts at `params` (0: every row given is real).
    /// A prompt's rows past 16 take the format's prompt GEMM; a window's rows the decode tiles, 16 at a time.
    pub fn projSums(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64) !void {
        const n_out = l.outputs();
        const k = l.inputs();
        if (prompt and rows > 16) return switch (l) {
            .a4 => |q| e.affine.prompt(e.stream, x, q, out, rows),
            .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
        };
        try e.run(e.fns.sums, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
        var at: usize = 0;
        while (at < rows) : (at += 16) {
            const n = @min(16, rows - at);
            const xa = x + at * k * 2;
            const sa = xs + at * (k / 64) * 4;
            const oa = out + at * n_out * 2;
            switch (l) {
                .a4 => |q| try e.affine.decode(e.stream, xa, sa, q, oa, n),
                .an => |q| try e.wide.decode(e.stream, xa, sa, q, oa, n),
            }
        }
    }

    /// Token rows of the embedding into `out` (rows, hidden) bf16, at the width its words are served.
    pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize, params: u64) !void {
        const f = if (e.embed_bits == 8) e.fns.embed8 else e.fns.embed;
        try e.run(f, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ids, e.embed_w, e.embed_s, e.embed_b, out, @as(c_int, @intCast(e.g.hidden)), params });
    }

    fn norm(e: *Engine, residual: bool, w: u64, rows: usize) !void {
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, e.g.eps, @as(c_int, @intCast(e.g.hidden)), e.tables.ptr + params_at });
    }

    /// A chain row's rowanc: its index in its window, as a prompt chunk's rows outnumber the bits.
    const chain_row: u32 = 1 << 31;

    /// The row tables of `parts`: each row's segment, parent row (-1: a window's first) and ancestors-and-self bits.
    fn rowTables(st: []u8, parts: []const Part) !void {
        const rowseg = std.mem.bytesAsSlice(c_int, st[rowseg_at..][0 .. max_rows * 4]);
        const rowpar = std.mem.bytesAsSlice(c_int, st[rowpar_at..][0 .. max_rows * 4]);
        const rowanc = std.mem.bytesAsSlice(u32, st[rowanc_at..][0 .. max_rows * 4]);
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.parents) |ps| if (ps.len != p.rows or p.rows > most_window or ps[0] != -1) return error.InvalidQwenParents;
            for (0..p.rows) |r| {
                const parent: i32 = if (p.parents) |ps| ps[r] else if (r == 0) -1 else @intCast(r - 1);
                if (parent >= @as(i32, @intCast(r)) or parent < -1) return error.InvalidQwenParents;
                rowseg[row0 + r] = @intCast(i);
                rowpar[row0 + r] = if (parent < 0) -1 else @intCast(row0 + @as(usize, @intCast(parent)));
                rowanc[row0 + r] = if (p.parents == null) chain_row | @as(u32, @intCast(r)) else (@as(u32, 1) << @intCast(r)) | (if (parent < 0) 0 else rowanc[row0 + @as(usize, @intCast(parent))]);
            }
            row0 += p.rows;
        }
    }

    /// One forward over `parts`' windows, `ids` in order; `keep` names the kept rows; an unkept verify is kept first.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        const g = e.g;
        var total: usize = 0;
        for (parts) |p| {
            try p.cache.settle(e);
            if (p.rows == 0 or p.cache.len + p.rows > p.cache.cap) return error.QwenContextFull;
            total += p.rows;
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        const record = keep != .last;
        if (record and total > e.verify_rows) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= g.vocab) return error.InvalidQwenToken;
        for (parts) |p| if (p.parents != null and !record) return error.InvalidQwenParents;
        const heads: usize = if (record) total else 1;
        // the prompt GEMM only for prompt chunks: a window's rows keep the bits of one-row steps
        const prompt = !record;
        // the tables
        const st = e.staged.bytes;
        @memcpy(std.mem.bytesAsSlice(u32, st[ids_at..][0 .. total * 4]), ids);
        const segs = std.mem.bytesAsSlice(Seg, st[segs_at..][0 .. parts.len * @sizeOf(Seg)]);
        try rowTables(st, parts);
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            segs[i] = p.cache.seg(e, p.cache.len, row0, p.rows);
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
        // words 0 and 1 are reserved (each row's attention chunks follow its own keys: qwen35.cu's chunk_keys)
        const params = std.mem.bytesAsSlice(c_int, st[params_at..][0..32]);
        params[0] = 0;
        params[1] = 0;
        params[2] = @intCast(total);
        params[3] = @intCast(parts.len);
        params[4] = @intFromBool(record);
        try e.tables.uploadAsync(0, st, e.stream.handle);
        // a captured forward covers a row bucket and every window; kernels read the true counts; a lone step is exact
        const lone = total == 1 and parts.len == 1;
        const bucket: usize = if (lone) 1 else if (total <= 16) 16 else most_verify;
        const windows: usize = if (lone) 1 else max_segs;
        const shape: Shape = .{ .total = @intCast(if (prompt) total else bucket), .nseg = @intCast(if (prompt) parts.len else windows), .record = record, .prompt = prompt, .taps = e.n_taps > 0 };
        if (prompt) {
            // uncaptured: a prompt chunk's GEMMs dwarf its launches, and each length's graph would pay the stall
            try e.body(shape);
        } else {
            const gop = try e.graphs.getOrPut(e.gpa, shape.key());
            if (!gop.found_existing) {
                errdefer _ = e.graphs.remove(shape.key());
                // captured, nothing runs: the replay below is this shape's first run
                try cuda.graph.beginCapture(e.stream, .thread_local);
                e.body(shape) catch |err| {
                    var broken = cuda.graph.endCapture(e.stream) catch return err;
                    broken.deinit();
                    return err;
                };
                var captured = try cuda.graph.endCapture(e.stream);
                defer captured.deinit();
                gop.value_ptr.* = try captured.instantiate();
            }
            try gop.value_ptr.launchOn(e.stream);
        }
        for (parts) |p| {
            p.cache.last = if (record and p.rows > 1) .{ .start = p.cache.len, .rows = p.rows } else null;
            p.cache.len += p.rows;
            p.cache.written = p.cache.len;
        }
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

    /// A forward's shape, which fixes its launches: its tables and the caches' lengths are read on the device.
    const Shape = struct {
        total: u8,
        nseg: u8,
        record: bool,
        prompt: bool,
        taps: bool,

        fn key(s: Shape) u32 {
            return @as(u32, s.total) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18;
        }
    };

    /// The launches of a forward of `shape`, from the embedding to the kept rows' logits (captured once a shape).
    fn body(e: *Engine, shape: Shape) !void {
        const g = e.g;
        const total: usize = shape.total;
        const nseg: usize = shape.nseg;
        const record = shape.record;
        const prompt = shape.prompt;
        const heads: usize = if (record) total else 1;
        const t_ids = e.tables.ptr + ids_at;
        const t_segs = e.tables.ptr + segs_at;
        const t_rows = e.tables.ptr + rowseg_at;
        const t_par = e.tables.ptr + rowpar_at;
        const t_anc = e.tables.ptr + rowanc_at;
        const t_params = e.tables.ptr + params_at;
        const geo = e.geo;
        const group = g.q_heads / g.kv_heads;

        try e.embed(t_ids, e.h, total, t_params);
        var di: usize = 0;
        var ai: c_int = 0;
        for (e.layers, 0..) |l, i| {
            try e.norm(i != 0, l.input_norm, total);
            switch (l.mixer) {
                .delta => |m| {
                    const layer: c_int = @intCast(di);
                    try e.proj(m.qkv, e.x, e.qkv, total, prompt);
                    try e.proj(m.z, e.x, e.z, total, prompt);
                    try e.proj(m.a, e.x, e.a, total, prompt);
                    try e.proj(m.b, e.x, e.b, total, prompt);
                    try e.run(e.fns.gdn_pre, .{ 1, 2 * g.k_heads + g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.qkv, t_segs, t_rows, t_par, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta, t_params });
                    try e.run(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                    try e.run(e.fns.gdn_chain, .{ 1, geometry.dk / 4, g.v_heads * nseg }, .{ 32, 4, 1 }, .{ geo, e.q, e.k, e.v, e.g_, e.beta, t_segs, t_par, layer, e.y, t_params });
                    try e.run(e.fns.gdn_post, .{ 1, g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params });
                    try e.proj(m.out, e.mix, e.r, total, prompt);
                    di += 1;
                },
                .attention => |m| {
                    const aw = g.attnWidth();
                    try e.proj(m.q, e.x, e.qraw, total, prompt);
                    try e.proj(m.k, e.x, e.kraw, total, prompt);
                    try e.proj(m.v, e.x, e.vraw, total, prompt);
                    try e.run(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps, t_params });
                    try e.run(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps, t_params });
                    try e.run(e.fns.rope, .{ total, 1, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, t_anc, ai, t_params });
                    try e.run(e.fns.attn_partial, .{ max_chunks, g.kv_heads, total }, .{ 32 * group, 1, 1 }, .{ geo, e.qn, t_segs, t_rows, t_anc, ai, t_params, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
                    try e.run(e.fns.attn_merge, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.y });
                    try e.run(e.fns.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params });
                    try e.proj(m.o, e.mix, e.r, total, prompt);
                    ai += 1;
                },
            }
            try e.norm(true, l.post_norm, total);
            try e.proj(l.gate, e.x, e.gate, total, prompt);
            try e.proj(l.up, e.x, e.up, total, prompt);
            const n: c_longlong = @intCast(total * g.intermediate);
            try e.run(e.fns.mlp_act, .{ (total * g.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n, t_params, @as(c_int, @intCast(g.intermediate)) });
            try e.proj(l.down, e.act, e.r, total, prompt);
            if (shape.taps) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == i) {
                try e.run(e.fns.tap, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.r, t_segs, t_rows, @as(c_int, @intCast(ti)), @as(c_int, @intCast(g.hidden)), t_params });
            };
        }
        try e.norm(true, e.final_norm, total);
        // the head over the kept rows
        try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
    }

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) []const u16 {
        return e.host_logits.slice(u16)[i * e.g.vocab ..][0..e.g.vocab];
    }
};
