//! Kolibri-1 behind the native CUDA registry (native/cuda.zig): open, the prompt cache's states, refusals in words.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const weights = @import("weights.zig");
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const be = @import("backend.zig");
const snap = @import("snap.zig");

pub const model_type = "kolibri1";
pub const formats: []const []const u8 = &.{"fp8-block"};
pub const default_context: i64 = 65536;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
/// A kept state is the caches through its position and nothing past it (no draft head reads a token on).
pub const snap_lookahead: u32 = 0;
/// No drafter yet: --drafter is refused before the load.
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Kolibri-1 on CUDA loads no drafter yet; serve it without --drafter";

/// `explain` words this family's open refusals too.
pub const explains_open = true;

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// No `vision` or `precision` field: the server refuses --vision and --precision for this family by name.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// What the native server drives; every stream's caches are allocated at open (`free_streams` of them).
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    learn_bytes: usize = 0,
    target: *be.Kolibri, // the prompt cache keeps this backend's states (snap_*)
};

const Owned = struct { gpa: std.mem.Allocator, w: weights.Weights, k: kern.Kernels, s: cuda.Stream, m: fwd.Model, kb: be.Kolibri };

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

/// The device bytes `n` streams take: their caches, the forward's scratch and the backend's buffers.
const Plan = struct {
    own: *const Owned,
    d: *const cuda.Driver,
    context: usize,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        return try fwd.Model.openBytes(o.gpa, p.d, o.s, &o.w, &o.k, p.context, n, be.prompt_rows) + be.Kolibri.openBytes(o.w.config.vocab, n);
    }
};

/// Positions a stream's caches hold: the window, whole 1,024s.
fn positions(context: usize) usize {
    return (context + 1023) / 1024 * 1024;
}

/// The checkpoint, the shared kernels and every stream's caches.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound; // before the weights load
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.w = try weights.load(gpa, io, ctx.d, dir, null);
    errdefer own.w.deinit();
    own.k = try kern.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    const context = positions(o.context);
    // every stream's caches are allocated here and none later: plan them inside the room the weights leave
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context };
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const n = fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, gib(room), most, if (most == 1) "" else "s" });
        }
        return e;
    };
    own.m = try fwd.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, n, be.prompt_rows);
    errdefer own.m.deinit();
    own.kb = try be.Kolibri.init(gpa, ctx.d, &own.m, &own.k, own.s);
    return .{
        .backend = own.kb.backend(),
        .facts = own.kb.facts(),
        .rows = be.max_window,
        .stream_bytes = fwd.Model.streamBytes(own.w.config, context),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.kb,
    };
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn kolibri(ptr: *anyopaque) *be.Kolibri {
    return @ptrCast(@alignCast(ptr));
}

/// Bytes a state at `at` prompt tokens takes.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return snap.bytes(kolibri(ptr).m, at);
}

/// The prompt pass stands at `at` (a chunk start): copy the stream's caches there.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try kolibri(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return kolibri(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    kolibri(ptr).dropState(@ptrCast(@alignCast(saved)));
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Kolibri-1 on CUDA has no draft head yet: drafts come from the prompt's own text only",
        error.StructuresUnsupported => "structured output is not built for Kolibri-1 on CUDA yet",
        error.CapturedKernels => "Kolibri-1 runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Kolibri-1 on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Kolibri-1 streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no Kolibri-1 stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.kb.deinit();
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
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.StructuresUnsupported, error.CapturedKernels, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
    try std.testing.expectEqual(@as(usize, 65536), positions(65000));
}

/// A synthetic plan: `stream` bytes a stream plus `fixed` bytes of scratch.
const Fake = struct {
    stream: u64,
    fixed: u64,
    fn bytes(f: Fake, n: usize) !u64 {
        return f.fixed + n * f.stream;
    }
};

test "the streams allocated at open fit the room the weights leave: auto takes what fits, a fixed count refuses" {
    const g: u64 = 1 << 30;
    const one: Fake = .{ .stream = g, .fixed = 0 };
    // 70 GiB of weights under a 72 GiB cap leave 2 GiB: eight 1 GiB streams would hold 78 GiB
    try std.testing.expectEqual(@as(usize, 2), try fit(72 * g - 70 * g, 8, false, one));
    try std.testing.expectError(error.ParallelPastMemory, fit(72 * g - 70 * g, 8, true, one));
    // the reserve instead of a cap: 10 GiB free less an 8 GiB reserve
    try std.testing.expectEqual(@as(usize, 2), try fit(10 * g - 8 * g, 8, false, one));
    // scratch counts: 2.5 GiB of room, 1 GiB of it scratch, holds one stream
    try std.testing.expectEqual(@as(usize, 1), try fit(5 * g / 2, 8, false, Fake{ .stream = g, .fixed = g }));
    try std.testing.expectError(error.NoStreamFits, fit(g - 1, 8, false, one));
    try std.testing.expectEqual(@as(usize, 8), try fit(8 * g, 8, true, one));
    try std.testing.expectEqual(@as(usize, 8), try fit(std.math.maxInt(u64), 8, false, one)); // no room hook: as asked
}

test "a fixed --parallel past a round's rows refuses, whatever the room" {
    const g: u64 = 1 << 30;
    const one: Fake = .{ .stream = g, .fixed = 0 };
    try std.testing.expectError(error.ParallelPastRound, fit(1024 * g, be.round_rows + 1, true, one));
    try std.testing.expectEqual(@as(usize, be.round_rows), try fit(1024 * g, be.round_rows, true, one));
}
