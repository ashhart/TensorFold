//! Dory behind the native CUDA registry (native/cuda.zig): open at a loop count, the prompt cache's states, refusals.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("cuda_weights.zig");
const fwd = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");
const config = @import("config.zig");
const pages = @import("cuda_pages.zig");
const kv = @import("core").kv_pool;

pub const model_type = "dory";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
/// A kept state is the caches through its position and nothing past it (no draft head reads a token on).
pub const snap_lookahead: u32 = 0;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Dory on CUDA loads no drafter; drafts come from the prompt's own text";
pub const explains_open = true;

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// --loops: passes over the recurrent block (null: the checkpoint's n_recurrent_loops); no vision, no precision.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null, loops: ?u32 = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Dory, // the prompt cache keeps this backend's states (snap_*)
    /// The full-attention caches as pages of the core pool (null: every stream allocated at open).
    layout: ?kv.Layout = null,
    /// With a layout: the first stream's rows are pages too, a window of them set aside by the plan.
    engine_paged: bool = true,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: fwd.Kernels, s: cuda.Stream, m: fwd.Model, db: be.Dory };

/// Streams to allocate in `room`: as asked, else (auto) the most that fit; a fixed count past them or a round refuses.
fn fit(room: u64, asked: u32, fixed: bool, plan: anytype) !usize {
    const most = be.round_rows / be.max_window;
    if (fixed and asked > most) return error.ParallelPastRound;
    var n = std.math.clamp(@as(usize, asked), 1, most);
    while (try plan.bytes(n) > room) : (n -= 1) {
        if (fixed) return error.ParallelPastMemory;
        if (n == 1) return error.NoStreamFits;
    }
    return n;
}

/// TENSORFOLD_DORY_KV_PAGES=0 keeps every stream's caches contiguous and allocated at open.
fn pagesOn() bool {
    return !envOff("TENSORFOLD_DORY_KV_PAGES");
}

/// The device bytes `n` streams take: their caches by loop count and context, the scratch, the backend's buffers.
const Plan = struct {
    own: *const Owned,
    d: *const cuda.Driver,
    context: usize,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        if (pagesOn()) return try fwd.Model.openBytesPaged(o.gpa, p.d, o.s, &o.w, &o.k, p.context) + be.Dory.openBytes(o.w.config.vocab, pages.max_slots);
        return try fwd.Model.openBytes(o.gpa, p.d, o.s, &o.w, &o.k, p.context, n) + be.Dory.openBytes(o.w.config.vocab, n);
    }
};

/// Paged: the first stream's rings and scratch must fit; the pool is planned by the host after.
fn fitFirst(room: u64, plan: anytype) !usize {
    if (try plan.bytes(1) > room) return error.NoStreamFits;
    return 1;
}

/// The pool: `streams` more slots beside the first (rings made now) and an arena of `n` pages.
pub fn attach(p: *anyopaque, streams: u32, n: usize) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    try own.db.attach(streams, n);
}

/// Whether a request's caches could start now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(p));
    return own.db.admits(prompt, max_tokens);
}

/// The checkpoint at its loop count, the shared kernels and every stream's caches.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows / be.max_window) return error.ParallelPastRound;
    if (o.loops) |l| if (l == 0 or l > config.max_loops) return error.BadLoops;
    const context = try be.capacity(o.context);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.w = try wts.load(gpa, io, ctx.d, dir, if (o.loops) |l| @as(usize, l) else null);
    errdefer own.w.deinit();
    if (context > own.w.config.positions + 1024) return error.PastContext;
    own.k = try fwd.Kernels.load(ctx);
    errdefer own.k.deinit();
    if (!own.k.served) return error.BuiltWithoutKernels;
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context };
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const n = if (pagesOn()) try fitFirst(room, plan) else fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window and {d} loops, and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, own.w.config.loops, gib(room), most, if (most == 1) "" else "s" });
        }
        return e;
    };
    own.m = if (pagesOn()) try fwd.Model.initPaged(gpa, ctx.d, own.s, &own.w, &own.k, context) else try fwd.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, n);
    errdefer own.m.deinit();
    own.m.graphed = !envOff("TENSORFOLD_DORY_GRAPHS"); // =0: every decode forward eager
    own.db = try be.Dory.init(gpa, ctx.d, &own.m, &own.k, own.s);
    if (own.m.paged) {
        std.log.info("Dory at {d} loops (per_loop KV): {d} caches a stream, the full ones ({d}) in pages of {d} positions, CUDA graphs {s}", .{ own.w.config.loops, own.m.sched.caches, own.m.sched.cacheCount(.full), pages.tokens, if (own.m.graphed) "on" else "off" });
    } else std.log.info("Dory at {d} loops (per_loop KV): {d} caches a stream, {d} streams, CUDA graphs {s}", .{ own.w.config.loops, own.m.sched.caches, n, if (own.m.graphed) "on" else "off" });
    return .{
        .backend = own.db.backend(),
        .facts = own.db.facts(),
        .rows = be.max_window,
        .stream_bytes = fwd.Model.streamBytes(own.w.config, context),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.db,
        .layout = if (own.m.paged) own.m.layout() else null,
    };
}

fn envOff(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

fn dory(ptr: *anyopaque) *be.Dory {
    return @ptrCast(@alignCast(ptr));
}

/// Bytes a state at `at` prompt tokens takes.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return snap.bytes(dory(ptr).m, at);
}

/// The prompt pass stands at `at` (a chunk start or a mark): copy the stream's caches there.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try dory(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return dory(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    dory(ptr).dropState(@ptrCast(@alignCast(saved)));
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Dory on CUDA has no draft head: drafts come from the prompt's own text only",
        error.StructuresUnsupported => "structured output is not built for Dory on CUDA yet",
        error.CapturedKernels => "Dory runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BuiltWithoutKernels => "this build carries no CUDA kernel images: build with -Dnvcc",
        error.UnsupportedDoryCudaQuantization => "Dory on CUDA requires ModelOpt W4A16 NVFP4 weights; BF16 and MXFP8 maps are host-only",
        error.BadLoops => std.fmt.comptimePrint("--loops: passes over Dory's recurrent block, 1 to {d}", .{config.max_loops}),
        error.UnsupportedKvMode => "this checkpoint asks for last_loop KV, whose output depends on chunk boundaries: Dory serves per_loop only",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Dory on CUDA holds at most {d} streams (a round's rows over 16-row windows): serve --parallel {d} or fewer, or auto", .{ be.round_rows / be.max_window, be.round_rows / be.max_window }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Dory streams than asked (each stream's caches, 8 + 20 per loop + 8, are allocated at load): lower --parallel, --context or --loops, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no Dory stream after the weights: lower --context or --loops, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.db.deinit();
    own.m.deinit();
    own.s.deinit();
    own.k.deinit();
    own.w.deinit();
    own.gpa.destroy(own);
}

test "the prompt cache's hooks compile against the store's vtable" {
    const pc: struct { bytes: *const fn (*anyopaque, u32) u64, save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque, restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void, drop: *const fn (*anyopaque, *anyopaque) void } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.StructuresUnsupported, error.CapturedKernels, error.UnsupportedDoryCudaQuantization, error.BadLoops, error.UnsupportedKvMode, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

/// A synthetic plan: `stream` bytes a stream plus `fixed` bytes of scratch.
const Fake = struct {
    stream: u64,
    fixed: u64,
    fn bytes(f: Fake, n: usize) !u64 {
        return f.fixed + n * f.stream;
    }
};

test "the streams allocated at open fit the room: auto takes what fits, a fixed count refuses, a round holds 8" {
    const g: u64 = 1 << 30;
    const one: Fake = .{ .stream = g, .fixed = 0 };
    try std.testing.expectEqual(@as(usize, 2), try fit(2 * g, 8, false, one));
    try std.testing.expectError(error.ParallelPastMemory, fit(2 * g, 8, true, one));
    try std.testing.expectEqual(@as(usize, 1), try fit(5 * g / 2, 8, false, Fake{ .stream = g, .fixed = g }));
    try std.testing.expectError(error.NoStreamFits, fit(g - 1, 8, false, one));
    try std.testing.expectEqual(@as(usize, 8), try fit(std.math.maxInt(u64), 8, false, one));
    try std.testing.expectEqual(@as(usize, 8), try fit(std.math.maxInt(u64), 64, false, one));
    try std.testing.expectError(error.ParallelPastRound, fit(1024 * g, 9, true, one));
}

test "KV a stream by loops and context: the 1,024-key window kept as a 3,072-key ring beside each prompt chunk" {
    const text = try config.pinnedText(std.testing.allocator);
    defer std.testing.allocator.free(text);
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, text, &why);
    const ctx = try be.capacity(32768);
    for ([_][3]usize{ .{ 1, 9, 27 }, .{ 2, 14, 42 }, .{ 4, 24, 72 } }) |row| {
        const cl = try c.withLoops(row[0]);
        try std.testing.expectEqual((row[1] * ctx + row[2] * 3072) * 4096, fwd.Model.streamBytes(cl, ctx));
    }
}
