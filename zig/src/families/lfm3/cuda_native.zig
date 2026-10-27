//! LFM3-1.7B (ModelOpt NVFP4, weight-only) behind the native CUDA registry: plain and drafted decoding, prompt reuse.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const mdl = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");
const kv = @import("core").kv_pool;

pub const model_type = "lfm3";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
/// A kept state is the caches through its position and nothing past it (no draft head reads a token on).
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
    target: *be.Lfm3,
    /// The attention caches as pages of the core pool (null: every stream owns a window, made at open).
    layout: ?kv.Layout = null,
    /// With a layout: the first stream's rows are pages too, a window of them set aside by the plan.
    engine_paged: bool = true,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: kern.Kernels, s: cuda.Stream, m: mdl.Model, lb: be.Lfm3 };

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

/// TENSORFOLD_LFM3_KV_PAGES=0: every stream owns a window of pages, made at open; else they lease the pool's.
fn pagesOn() bool {
    const v = std.c.getenv("TENSORFOLD_LFM3_KV_PAGES") orelse return true;
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
        return try mdl.Model.openBytes(o.gpa, p.d, o.s, &o.w, &o.k, p.context, n, p.paged) + be.Lfm3.openBytes(if (p.paged) be.round_rows else n);
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
    own.m = try mdl.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, n, paged);
    errdefer own.m.deinit();
    own.lb = try be.Lfm3.init(gpa, ctx.d, &own.m, &own.k, own.s);
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

fn lfm(ptr: *anyopaque) *be.Lfm3 {
    return @ptrCast(@alignCast(ptr));
}

/// Bytes a state at `at` prompt tokens takes.
pub fn snap_bytes(_: *anyopaque, at: u32) u64 {
    return snap.bytes(at);
}

/// The prompt pass stands at `at` (a chunk end or a mark): copy the stream's caches there.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try lfm(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return lfm(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    lfm(ptr).dropState(@ptrCast(@alignCast(saved)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "LFM3 on CUDA has no draft head: it drafts from the prompt's own text",
        error.Lfm3TextOnly => "LFM3-1.7B is a text model, and LFM3-VL's image tower is not built on CUDA yet: this request has images",
        error.StructuresUnsupported => "structured output is not built for LFM3 on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "LFM3 runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedLfm3Quantization => "LFM3 on CUDA reads the ModelOpt NVFP4 text export (LiquidAI/LFM3-1.7B-NVFP4) for now: its MLX affine and FP8 exports are not loaded yet",
        error.UnsupportedLfm3Config, error.UnsupportedLfm3Geometry, error.UnsupportedLfm3LayerSchedule, error.UnsupportedLfm3Rotary, error.UnsupportedLfm3Context, error.BadLfm3Config => "this lfm3 checkpoint's config is not LFM3-1.7B's, which this engine serves",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: LFM3 on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer LFM3 streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no LFM3 stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
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
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.Lfm3TextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.UnsupportedLfm3Quantization, error.UnsupportedLfm3Geometry, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

test "the prompt cache's hooks compile against the store's vtable" {
    const pc: struct { bytes: *const fn (*anyopaque, u32) u64, save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque, restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void, drop: *const fn (*anyopaque, *anyopaque) void } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
    try std.testing.expectEqual(snap.bytes(100), snap_bytes(undefined, 100));
}
