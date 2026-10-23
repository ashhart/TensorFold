//! Qwen3.5's hybrids on CUDA: qlinear projections, qwen35.cu glue, one captured forward over every stream's window.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Round = @import("cuda_round.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const caches = @import("cuda_cache.zig");
const routed = @import("cuda_routed.zig");
const hooks = @import("cuda_learn_hooks.zig");
const pocket = @import("cuda_pocket.zig");
const profile = @import("cuda_startup_profile.zig");
pub const recurrent = @import("cuda_recurrent.zig");

pub const Geometry = geometry.Geometry;
pub const Cache = caches.Cache;
pub const tap_rows = caches.tap_rows;
const Seg = caches.Seg;
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = 128;
/// Streams' windows in one forward at most.
pub const max_segs = Round.max_segs;
/// Rows of a verify forward at most: each row's logits are kept and drawn.
pub const most_verify = Round.most_verify;
/// tf_q35_argmax_slices' blocks a row (ARGMAX_SLICE entries each).
fn argmaxSlices(vocab: usize) usize {
    return (vocab + 4095) / 4096;
}
/// A window's rows at most (exact: one-row steps' bits; qwen35.cuh's MAXW, the rows a stream's log holds).
pub const most_window = Round.most_window;
/// Partial attention chunks a row at most (qwen35.cuh's MAXCH): each row's chunk width grows with its own keys.
const max_chunks = 64;

/// Device allocations are 256-byte aligned, as the driver's are.
pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// qwen35.cuh's Geo.
pub const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };

const layers = @import("cuda_layers.zig");
pub const Proj = layers.Proj;
pub const Delta = layers.Delta;
pub const Attention = layers.Attention;
pub const Layer = layers.Layer;
pub const Mlp = layers.Mlp;
const Fns = layers.Fns;
pub const Keeps = layers.Keeps;
const Kv8 = layers.Kv8;

/// A window of one stream in a forward: `parents`, for a tree, names each row's parent row (-1: the first row).
pub const Part = struct { cache: *Cache, rows: usize, parents: ?[]const i32 = null, images: ?*const Images = null };

/// An image prompt's rows for its prompt chunks: device feature rows, each prompt row's feature row, its positions.
pub const Images = struct {
    features: u64, // bf16 [image rows, hidden]
    src: []const i32, // [prompt]: a feature row, or -1 for text
    pos: []const i32, // [3][prompt]: t, h and w rotary positions (get_rope_index)
};

/// What a forward keeps: a prompt chunk its last row's logits, a verify window every row's draw or logits and log.
pub const Keep = union(enum) { last, draws: []const Draw, logits, hidden };

/// A kept row's draw on the device: sample.cu's keyed rule at `position`, or greedy without one.
pub const Draw = struct { rule: ?cuda.sample.Rule, position: u64 };

pub const Engine = struct {
    rec_scratch: recurrent.Pool,
    startup: profile.Times = .{},
    gpa: std.mem.Allocator,
    dir: []const u8,
    pockets: pocket.Pockets,
    learning: ?*const hooks.Keep = null,
    mlp: ?hooks.MlpAdd = null,
    taught: u64 = 0,
    d: *const cuda.Driver,
    g: Geometry,
    geo: Geo,
    stream: cuda.Stream,
    mods: [16]cuda.Module,
    kv8: bool, // attention caches as e4m3 rows with a scale each
    fp8: Kv8,
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    wide: cuda.qlinear.AffineNKernels, // the tiles of every width but 4, and their packs
    experts: cuda.experts.Affine4Experts,
    experts3: cuda.experts3.Affine3Experts,
    plan: cuda.grouped.Router,
    moe: cuda.moe.Fns,
    fns: Fns,
    keeps: Keeps,
    /// Every weight and scratch buffer in one allocation: WDDM's launch cost grows with the allocation count.
    arena: DeviceBuffer,
    arena_at: usize = 0,
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
    routed: routed.Scratch = .{}, // qwen3_5_moe's routed rows
    amax: u64, // the argmax slices' bests: most_verify rows of argmaxSlices(vocab) (f32, i32)
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
    /// During an image prompt chunk's forward: its feature rows (0: none); the chunk's src and positions are staged.
    features: u64 = 0,

    pub const ids_at = Round.ids_at;
    pub const segs_at = Round.segs_at;
    pub const rowseg_at = Round.rowseg_at;
    pub const rowpar_at = Round.rowpar_at;
    pub const rowanc_at = Round.rowanc_at;
    const rules_at = Round.rules_at;
    const meta_at = Round.meta_at;
    const tokens_at = Round.tokens_at;
    pub const params_at = Round.params_at;
    pub const src_at = Round.src_at;
    pub const mpos_at = Round.mpos_at;
    const tables_len = Round.bytes;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kv8: bool) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const started = profile.now(io);
        const g = try Geometry.read(gpa, io, dir);
        const d = ctx.d;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.dir = try gpa.dupe(u8, dir);
        errdefer gpa.free(e.dir);
        e.pockets = try pocket.Pockets.init(gpa, d, g.hidden, g.layers);
        errdefer e.pockets.deinit();
        e.learning = null;
        e.mlp = null;
        e.taught = 0;
        e.d = d;
        e.g = g;
        e.geo = .{ .hid = @intCast(g.hidden), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .qh = @intCast(g.q_heads), .kvh = @intCast(g.kv_heads), .nl = @intCast(g.deltas()) };
        e.verify_rows = most_verify;
        e.kv8 = kv8;
        e.window_rows = most_window;
        e.arena_at = 0;
        e.graphs = .empty;
        e.rec_scratch = recurrent.Pool.init(gpa, @intFromPtr(e));
        e.n_taps = 0;
        e.features = 0;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.arena = try DeviceBuffer.alloc(d, weights.total(g) + try e.scratch(.measure));
        errdefer e.arena.free();
        try e.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qwen35, kk.sample, kk.qwen35_keep, kk.affine8, kk.affine_prefill, kk.qwen35_kv8, kk.experts, kk.experts_prefill, kk.experts_pack, kk.moe, kk.experts3, kk.experts3_prefill };
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
        e.wide = try cuda.qlinear.AffineNKernels.resolve(d, e.mods[7], e.mods[8], e.mods[3], sms, gb10);
        errdefer e.wide.deinit();
        e.draws = try cuda.sample.Draws.resolve(e.mods[5]);
        e.experts = try cuda.experts.Affine4Experts.resolve(e.mods[10], e.mods[11], e.mods[12], sms);
        e.experts3 = try cuda.experts3.Affine3Experts.resolve(e.mods[14], e.mods[15], sms);
        e.plan = try cuda.grouped.Router.resolve(e.mods[10]);
        e.moe = try cuda.moe.Fns.resolve(e.mods[13]);
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(e.fns, name) = try e.mods[4].function("tf_q35_" ++ name);
        inline for (@typeInfo(Keeps).@"struct".field_names) |name| @field(e.keeps, name) = try e.mods[6].function("tf_q35k_" ++ name);
        e.fp8 = .{ .rope = try e.mods[9].function("tf_q35_rope8"), .attn_partial = try e.mods[9].function("tf_q35_attn_partial8"), .compact = try e.mods[9].function("tf_q35k_compact8") };
        e.layers = try gpa.alloc(Layer, g.layers);
        errdefer gpa.free(e.layers);
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        {
            // each projection as stored, packed from here; freed before the server counts what streams can use
            var stage = try DeviceBuffer.alloc(d, weights.stageBytes(g));
            defer stage.free();
            try weights.load(e, &ck, &stage);
        }
        _ = try e.scratch(.allocate);
        e.tables = try DeviceBuffer.alloc(d, tables_len);
        errdefer e.tables.free();
        e.staged = try cuda.HostBuffer.alloc(d, tables_len);
        errdefer e.staged.free();
        e.host_logits = try cuda.HostBuffer.alloc(d, most_verify * g.vocab * 2);
        errdefer e.host_logits.free();
        try e.stream.synchronize();
        const warm_start = profile.now(io);
        e.startup = .{ .load = warm_start - started };
        errdefer recurrent.shutdown(e);
        e.startup.add(try profile.withCache(gpa, io, e, Cache));
        return e;
    }

    /// Every path a request takes, run once now: WDDM stalls a process's first forwards for seconds (a 20 s launch).
    pub fn warmCache(e: *Engine, warm_cache: *Cache) !void {
        const ids: [32]u32 = @splat(0);
        const greedy: Draw = .{ .rule = null, .position = 33 };
        try e.forward(&.{.{ .cache = warm_cache, .rows = ids.len }}, &ids, .last); // a prompt chunk: the prompt GEMMs
        try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{greedy} });
        try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = .{ .seed = 1, .inv_t = 1.0 / 0.7, .top_p = 0.95, .near = 20.0, .min_log = -std.math.inf(f32), .top_k = 20 }, .position = 34 }} });
        // a tree window and a branch of it kept: the keep kernels, and the forward's branch path
        try e.forward(&.{.{ .cache = warm_cache, .rows = 3, .parents = &.{ -1, 0, 0 } }}, ids[0..3], .{ .draws = &.{ greedy, greedy, greedy } });
        try warm_cache.keep(e, &.{ 0, 2 });
        try e.stream.synchronize();
        recurrent.completed(e);
    }

    pub fn deinit(e: *Engine) void {
        recurrent.shutdown(e);
        e.pockets.deinit();
        var it = e.graphs.valueIterator();
        while (it.next()) |x| x.deinit();
        e.graphs.deinit(e.gpa);
        e.host_logits.free();
        e.staged.free();
        e.tables.free();
        e.arena.free();
        e.gpa.free(e.layers);
        e.gpa.free(e.dir);
        e.affine.deinit();
        e.wide.deinit();
        for (&e.mods) |*m| m.unload();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values, taps, log.
    pub fn streamBytes(e: *const Engine, cap: usize) usize {
        return 2 * e.g.stateBytes() + e.g.kvBytes(cap, e.kv8) + e.tapBytes() + e.g.logBytes(most_window);
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
            pub fn alloc(t: @This(), bytes: usize) !u64 {
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
        if (g.moe()) e.routed = try routed.Scratch.carve(t, g, R);
        e.amax = try t.alloc(most_verify * argmaxSlices(g.vocab) * 8);
        return total;
    }

    pub fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// out (rows, n) = x (rows, k) through `l`: row tiles with batch-independent bits or a prompt GEMM.
    pub fn proj(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
        return e.projSums(l, x, e.xs, out, rows, prompt, e.tables.ptr + params_at);
    }

    /// `proj` with group sums `xs` and device counts `params` (0: all real); prompts past 16 rows take a prompt GEMM.
    pub fn projSums(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64) !void {
        return e.project(l, x, xs, out, rows, prompt, params, true);
    }

    pub fn projAgain(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
        return e.project(l, x, e.xs, out, rows, prompt, e.tables.ptr + params_at, false);
    }

    /// `projAgain` of each of `ls` into `outs` over one x; on the split form's GPUs one segmented launch a row tile.
    pub fn projMany(e: *Engine, ls: []const Proj, x: u64, outs: []const u64, rows: usize, prompt: bool) !void {
        return e.projectMany(ls, x, e.xs, outs, rows, prompt, e.tables.ptr + params_at, false);
    }

    /// `projMany` with shared input sums, preserving the caller's prompt or native projection role.
    pub fn projManySums(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64) !void {
        return e.projectMany(ls, x, xs, outs, rows, prompt, params, true);
    }

    pub const projectMany = @import("cuda_projection_many.zig").projectMany;

    pub fn project(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
        const n_out = l.outputs();
        const k = l.inputs();
        if (prompt and rows > 16) return switch (l) {
            .a4 => |q| e.affine.prompt(e.stream, x, q, out, rows),
            .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
        };
        if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
        const tile: usize = switch (l) {
            .a4 => e.affine.verifyRows(),
            .an => 16,
        };
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            const n = @min(tile, rows - at);
            const xa = x + at * k * 2;
            const sa = xs + at * (k / 64) * 4;
            const oa = out + at * n_out * 2;
            switch (l) {
                .a4 => |q| try e.affine.verify(e.stream, xa, sa, q, oa, n),
                .an => |q| try e.wide.decode(e.stream, xa, sa, q, oa, n),
            }
        }
    }

    /// Token rows of the embedding into `out` (rows, hidden) bf16, at the width its words are served.
    pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize, params: u64) !void {
        const f = if (e.embed_bits == 8) e.fns.embed8 else e.fns.embed;
        try e.run(f, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ids, e.embed_w, e.embed_s, e.embed_b, out, @as(c_int, @intCast(e.g.hidden)), params });
    }

    /// Logits rows first..first + n's greedy tokens into toks: the highest logit, ties to the lower id.
    fn argmax(e: *Engine, first: usize, n: usize, toks: u64) !void {
        if (n == 0) return;
        const slices = argmaxSlices(e.g.vocab);
        const pv = e.amax;
        const pi = e.amax + n * slices * 4;
        try e.run(e.fns.argmax_slices, .{ slices, n, 1 }, .{ 256, 1, 1 }, .{ e.logits + first * e.g.vocab * 2, @as(c_int, @intCast(e.g.vocab)), pv, pi });
        try e.run(e.fns.argmax_merge, .{ n, 1, 1 }, .{ 32, 1, 1 }, .{ pv, pi, @as(c_int, @intCast(slices)), toks + first * 4 });
    }

    pub fn norm(e: *Engine, residual: bool, w: u64, rows: usize) !void {
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, e.g.eps, @as(c_int, @intCast(e.g.hidden)), e.tables.ptr + params_at, e.xs });
    }

    /// Refuse the whole frame before settling any prior window or launching work.
    pub fn admit(e: *const Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        const g = e.g;
        var total: usize = 0;
        if (parts.len == 0 or parts.len > max_segs) return error.InvalidQwenSegments;
        for (parts, 0..) |p, index| {
            for (parts[0..index]) |prior| if (prior.cache == p.cache) return error.InvalidQwenSegments;
            if (p.rows == 0 or p.cache.len > p.cache.cap or p.rows > p.cache.cap - p.cache.len) return error.QwenContextFull;
            if (p.rows > max_rows - total) return error.InvalidQwenSegments;
            total += p.rows;
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        if (e.learning != null and total != 1) return error.BadLearningRows;
        if (keep == .hidden and e.learning == null) return error.BadLearningRows;
        const record = keep != .last;
        if (record and total > e.verify_rows) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= g.vocab) return error.InvalidQwenToken;
        for (parts) |p| if (p.parents != null and !record) return error.InvalidQwenParents;
        for (parts) |p| if (p.images != null and record) return error.InvalidQwenImages;
    }

    /// One forward over `parts`' windows, with every input admitted before an unkept verify is settled.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        try e.admit(parts, ids, keep);
        const g = e.g;
        const total = ids.len;
        const record = keep != .last;
        const heads: usize = if (record) total else 1;
        // the prompt GEMM only for prompt chunks: a window's rows keep the bits of one-row steps
        const prompt = !record;
        // the tables
        const st = e.staged.bytes;
        @memcpy(std.mem.bytesAsSlice(u32, st[ids_at..][0 .. total * 4]), ids);
        const segs = std.mem.bytesAsSlice(Seg, st[segs_at..][0 .. parts.len * @sizeOf(Seg)]);
        const slot_peak = try Round.build(st, parts, record);
        for (parts) |p| {
            try p.cache.settle(e);
            try recurrent.prepare(e, p.cache, p.rows, record);
        }
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
        defer e.taught = 0;
        try @import("cuda_forward_extra.zig").stageScales(e, parts, total);
        // an image prompt's chunk (one part, its rows at prompt positions len..): feature rows and t, h, w positions
        const images: ?*const Images = if (record) null else parts[0].images;
        if (images) |im| {
            const p0 = parts[0].cache.len;
            if (p0 + total > im.src.len or im.pos.len != 3 * im.src.len) return error.InvalidQwenImages;
            const n = im.src.len;
            @memcpy(std.mem.bytesAsSlice(i32, st[src_at..][0 .. total * 4]), im.src[p0..][0..total]);
            const mpos = std.mem.bytesAsSlice(i32, st[mpos_at..][0 .. 3 * max_rows * 4]);
            for (0..3) |a| @memcpy(mpos[a * max_rows ..][0..total], im.pos[a * n + p0 ..][0..total]);
        }
        try e.tables.uploadAsync(0, st[0..if (images != null) tables_len else src_at], e.stream.handle); // text: as before
        var shape = Shape.init(total, parts.len, record, e.n_taps > 0, @intCast(Round.variant(slot_peak)));
        if (record) for (parts) |p| {
            try recurrent.mark(p.cache, false);
            if (p.rows == 1) try recurrent.mark(p.cache, true);
        };
        shape.hidden = keep == .hidden;
        if (e.learning != null or e.pockets.any()) {
            var exact = shape;
            exact.total = @intCast(total);
            exact.nseg = @intCast(parts.len);
            e.features = if (images) |im| im.features else 0;
            defer e.features = 0;
            try e.body(exact);
        } else if (prompt) {
            // uncaptured: a prompt chunk's GEMMs dwarf its launches, and each length's graph would pay the stall
            e.features = if (images) |im| im.features else 0;
            defer e.features = 0;
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
        for (parts) |p| p.cache.advanced(p.rows, record);
        switch (keep) {
            .hidden => try e.stream.synchronize(), // mapped layer inputs are ready before the learner reads them
            .draws => |pk| {
                const toks = e.tables.ptr + tokens_at;
                var greedy: usize = 0; // the run of greedy rows ending at i, argmaxed together
                for (pk, 0..) |p, i| {
                    if (p.rule == null) {
                        greedy += 1;
                        continue;
                    }
                    try e.argmax(i - greedy, greedy, toks);
                    greedy = 0;
                    try e.draws.draw(e.stream, e.logits + i * g.vocab * 2, g.vocab, e.tables.ptr + rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + meta_at + i * 4, 0, toks + i * 4, 1, null, null);
                }
                try e.argmax(pk.len - greedy, greedy, toks);
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
        recurrent.completed(e);
    }

    pub const Shape = @import("cuda_shapes.zig").Shape;

    /// The same scan selected by body, exposed for host launch-contract fixtures.
    pub fn deltaScan(e: *Engine, record: bool, slot_count: u8, segments: usize, layer: c_int) !void {
        const scan = if (!record) e.fns.gdn_chain else switch (slot_count) {
            1 => e.fns.gdn_slots_1,
            2 => e.fns.gdn_slots_2,
            4 => e.fns.gdn_slots_4,
            8 => e.fns.gdn_slots_8,
            16 => e.fns.gdn_slots_16,
            else => return error.InvalidGdnSlots,
        };
        const input = e.tables.ptr + @as(usize, if (record) Round.steps_at else rowpar_at);
        try e.run(scan, .{ 1, geometry.dk / 4, e.g.v_heads * segments }, .{ 32, 4, 1 }, .{ e.geo, e.q, e.k, e.v, e.g_, e.beta, e.tables.ptr + segs_at, input, layer, e.y, e.tables.ptr + params_at });
    }

    /// The launches of a forward of `shape`, from the embedding to the kept rows' logits (captured once a shape).
    pub const body = @import("cuda_forward.zig").body;

    pub const routedRows = routed.rows;

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) []const u16 {
        return e.host_logits.slice(u16)[i * e.g.vocab ..][0..e.g.vocab];
    }
};

test "dense read cost: hidden and slot variants have distinct graph keys" {
    var seen: [32]u32 = undefined;
    var n: usize = 0;
    for ([_]u8{ 1, 2, 4, 8, 16 }) |slots| {
        for ([_]bool{ false, true }) |hidden| {
            const shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = true, .prompt = false, .taps = true, .slots = slots, .hidden = hidden };
            for (seen[0..n]) |key| try std.testing.expect(key != shape.key());
            seen[n] = shape.key();
            n += 1;
        }
    }
}
