//! LFM2.5-24B-A2B (MLX affine 4-bit) behind the native CUDA registry: plain and copy-drafted decoding, no prompt cache.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const mdl = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const kv = @import("core").kv_pool;

pub const model_type = "lfm2_moe";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
pub const snap_lookahead: u32 = 0;
/// A request's last verify window: its pages hold that many rows past its prompt and reply.
pub const window_rows: u32 = be.max_window;
pub const explains_open = true;

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// No `precision` field: the server refuses --precision for this family by name.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Lfm,
    /// The attention caches as pages of the core pool (null: every stream owns a window, made at open).
    layout: ?kv.Layout = null,
    /// With a layout: the first stream's rows are pages too, a window of them set aside by the plan.
    engine_paged: bool = true,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: kern.Kernels, s: cuda.Stream, m: mdl.Model, lb: be.Lfm };

/// Streams to allocate in `room`: as asked, else (auto) the most that fit; a fixed count past them or a round refuses.
fn fit(room: u64, asked: u32, fixed: bool, plan: anytype) !usize {
    if (fixed and asked > be.round_rows) return error.ParallelPastRound;
    var n = std.math.clamp(@as(usize, asked), 1, be.round_rows);
    while (try plan.bytes(n) > room) : (n -= 1) {
        if (fixed) return error.ParallelPastMemory;
        if (n == 1) return error.NoStreamFits;
    }
    return n;
}

/// TENSORFOLD_LFM2_KV_PAGES=0: every stream owns a window of pages, made at open; else they lease the pool's.
fn pagesOn() bool {
    const v = std.c.getenv("TENSORFOLD_LFM2_KV_PAGES") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The device bytes `n` streams take (paged: the first's states and a round's slot buffers, no pages).
const Plan = struct {
    own: *const Owned,
    d: *const cuda.Driver,
    context: usize,
    paged: bool,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        return try mdl.Model.openBytes(o.gpa, p.d, o.s, &o.w, p.context, n, p.paged) + be.Lfm.openBytes(if (p.paged) be.round_rows else n);
    }
};

/// The pool: `streams` more slots beside the first and an arena of `n` pages.
pub fn attach(p: *anyopaque, streams: u32, n: usize) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.lb.attach(streams, n);
}

/// Whether a request's caches could start now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(p));
    return own.lb.admits(prompt, max_tokens);
}

/// The attention layers the kernels' page arena holds: the admitted 10 of 40 (conv the rest).
fn checkLayers(config: *const cfg.Config) !void {
    var n: usize = 0;
    for (config.attention) |a| n += @intFromBool(a);
    if (n != mdl.attention_layers) return error.UnsupportedLfmGeometry;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .w = undefined, .k = undefined, .s = undefined, .m = undefined, .lb = undefined };
    own.k = try kern.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    own.w = try wts.load(gpa, io, ctx.d, own.s, &own.k, dir, null);
    errdefer own.w.deinit();
    try checkLayers(&own.w.config);
    const context = @min(mdl.Model.positions(o.context), own.w.config.context);
    const paged = pagesOn();
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context, .paged = paged };
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const n = if (paged) try fitFirst(room, plan) else fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, gib(room), most, if (most == 1) "" else "s" });
        }
        return e;
    };
    own.m = try mdl.Model.init(gpa, ctx.d, own.s, &own.w, context, n, paged);
    errdefer own.m.deinit();
    own.lb = try be.Lfm.init(gpa, ctx.d, &own.m, &own.k, own.s);
    errdefer own.lb.deinit();
    return .{
        .backend = own.lb.backend(),
        .facts = own.lb.facts(),
        .rows = be.max_window,
        .stream_bytes = mdl.Model.streamBytes(context),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.lb,
        .layout = if (paged) own.m.layout() else null,
    };
}

/// Paged: the first stream's states and the scratch must fit; the host plans the pool after.
fn fitFirst(room: u64, plan: anytype) !usize {
    if (try plan.bytes(1) > room) return error.NoStreamFits;
    return 1;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "LFM2.5-24B-A2B on CUDA has no draft head: it drafts from the prompt's own text",
        error.LfmTextOnly => "LFM2.5-24B-A2B is a text model: this request has images",
        error.StructuresUnsupported => "structured output is not built for LFM2.5-24B-A2B on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "LFM2.5-24B-A2B runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedLfmQuantization => "LFM2.5-24B-A2B on CUDA reads MLX affine 4-bit checkpoints in groups of 64 (the embedding at 4, 6 or 8 bits): the 6- and 8-bit exports are served on Metal",
        error.UnsupportedLfmGeometry, error.UnsupportedLfmConfig, error.UnsupportedLfmRotary => "this lfm2_moe checkpoint's geometry is not LFM2.5-24B-A2B's, which this engine serves",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: LFM2.5-24B-A2B on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer LFM2.5-24B-A2B streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no LFM2.5-24B-A2B stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lb.deinit();
    own.m.deinit();
    own.w.deinit();
    own.s.deinit();
    own.k.deinit();
    own.gpa.destroy(own);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.LfmTextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.UnsupportedLfmQuantization, error.UnsupportedLfmGeometry, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

test "the admitted layer pattern has the arena's ten attention layers" {
    var config = try cfg.parse(std.testing.allocator, cfg.fixture);
    try checkLayers(&config);
    config.attention[0] = true;
    try std.testing.expectError(error.UnsupportedLfmGeometry, checkLayers(&config));
}
