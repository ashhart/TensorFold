//! Mini (inkling2_loop) in the native CUDA registry: one rank, text only; images, drafters, --loops refused by name.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("mini_weights.zig");
const mm = @import("mini_model.zig");
const be = @import("mini_backend.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Inkling 2 Mini on CUDA serves plain decode so far; its DSpark drafter is the next slice (C5)";
pub const explains_open = true;

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// `loops` is here to refuse it by name: the pass count is the checkpoint's loop_config.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 4, lanes_fixed: bool = false, room: ?Room = null, loops: ?u32 = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Mini,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: mm.Kernels, x: be.Draws, s: cuda.Stream, m: mm.Model, b: be.Mini };

/// Streams to allocate in `room`: as asked, else (auto) the most that fit; a fixed count past them refuses.
fn fit(room: u64, asked: u32, fixed: bool, plan: anytype) !usize {
    const most = be.round_rows;
    if (fixed and asked > most) return error.ParallelPastRound;
    var n = std.math.clamp(@as(usize, asked), 1, most);
    while (try plan.bytes(n) > room) : (n -= 1) {
        if (fixed) return error.ParallelPastMemory;
        if (n == 1) return error.NoStreamFits;
    }
    return n;
}

const Plan = struct {
    own: *const Owned,
    d: *const cuda.Driver,
    context: usize,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        return try mm.Model.openBytes(o.gpa, p.d, o.s, &o.w, &o.k, p.context, n) + be.Mini.openBytes(o.w.config.text.unpadded_vocab, n);
    }
};

/// Refused before any weight is read: --loops and captured kernels; `drafts` (the server default) drafts nothing.
fn refuse(kernels: ?[]const u8, o: Options) !void {
    if (o.loops != null) return error.Inkling2MiniLoopsFixed;
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound;
}

/// The checkpoint, the kernels and every stream's caches.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    try refuse(kernels, o);
    const context = try be.capacity(o.context);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.w = try wts.load(gpa, io, ctx.d, dir);
    errdefer own.w.deinit();
    if (context > own.w.config.text.max_context) return error.PastContext;
    own.k = try mm.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.x = try be.Draws.load(ctx.d);
    errdefer own.x.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context };
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const n = fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window, and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, gib(room), most, if (most == 1) "" else "s" });
        }
        return e;
    };
    own.m = try mm.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, n);
    errdefer own.m.deinit();
    own.b = try be.Mini.init(gpa, ctx.d, &own.m, &own.k, &own.x, own.s, own.w.config.image_token);
    std.log.info("Inkling 2 Mini: {d} visits a token ({d} sliding, {d} full, {d} conv slots a stream), {d} streams of {d} positions, {d:.2} GiB of weights", .{ own.w.plan.visits.len, own.w.plan.visits.count(.sliding), own.w.plan.visits.count(.full), own.w.plan.visits.count(.conv), n, context, gib(own.w.bytes) });
    return .{
        .backend = own.b.backend(),
        .facts = own.b.facts(),
        .rows = 1,
        .stream_bytes = own.m.streamBytes(),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.b,
    };
}

/// Whether a request's caches could start now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(p));
    return own.b.admits(prompt, max_tokens);
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Inkling2MiniLoopsFixed => "--loops: Inkling 2 Mini runs the passes its loop_config names (3 over layers 6 to 17); serve without --loops",
        error.Inkling2MiniImages => "images: Inkling 2 Mini on CUDA reads text only until its vision slice (C6) lands; send text parts only",
        error.Inkling2MiniNoRollback => "Inkling 2 Mini on CUDA keeps every verified row (no drafts are offered to roll back)",
        error.Inkling2NotLooped => "Inkling 2 Small (model_type inkling2) has no CUDA engine yet: this one serves inkling2_loop",
        error.Inkling2WeightMap => "the checkpoint's tensors differ from what its config.json implies (the log names each one)",
        error.Inkling2Nvfp4Denominator => "an NVFP4 expert's outer_scale_denominator is not 2688, the one the checkpoint's factors are checked against",
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Inkling 2 Mini on CUDA has no drafter yet (C5): serve without --drafter or drafts",
        error.StructuresUnsupported => "structured output is not built for Inkling 2 Mini on CUDA yet",
        error.CapturedKernels => "Inkling 2 Mini runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BuiltWithoutKernels => "this build carries no CUDA kernel images: build with -Dnvcc",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Inkling 2 Mini on CUDA holds at most {d} streams (a round's rows)", .{be.round_rows}),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Inkling 2 Mini streams than asked (each stream's 48 visit slots are allocated at load): lower --parallel or --context, or serve --parallel auto",
        error.NoStreamFits => "the CUDA memory budget fits no Inkling 2 Mini stream after the weights: lower --context or free device memory",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.b.deinit();
    own.m.deinit();
    own.s.deinit();
    own.x.deinit();
    own.k.deinit();
    own.w.deinit();
    own.gpa.destroy(own);
}

test "Inkling 2 Mini open: --loops and captured kernels are refused by name before any load" {
    const base: Options = .{ .context = 4096, .drafts = false };
    try refuse(null, base);
    var o = base;
    o.loops = 3;
    try std.testing.expectError(error.Inkling2MiniLoopsFixed, refuse(null, o));
    o = base;
    o.drafts = true; // the server's default: nothing to draft with, not a refusal
    try refuse(null, o);
    try std.testing.expectError(error.CapturedKernels, refuse("dir", base));
    // open refuses before it allocates: a failing allocator sees no request
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var d: cuda.Driver = undefined;
    const ctx: cuda.Context = .{ .d = &d, .handle = undefined, .device = 0 };
    o = base;
    o.loops = 2;
    try std.testing.expectError(error.Inkling2MiniLoopsFixed, open(failing.allocator(), std.testing.io, &ctx, "/nonexistent", null, o));
    try std.testing.expectEqual(@as(usize, 0), failing.allocations);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.Inkling2MiniLoopsFixed, error.Inkling2MiniImages, error.Inkling2MiniNoRollback, error.Inkling2NotLooped, error.Inkling2WeightMap, error.Inkling2Nvfp4Denominator, error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.StructuresUnsupported, error.CapturedKernels, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
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

test "the streams allocated at open fit the room: auto takes what fits, a fixed count refuses" {
    const g: u64 = 1 << 30;
    const one: Fake = .{ .stream = g, .fixed = 0 };
    try std.testing.expectEqual(@as(usize, 2), try fit(2 * g, 8, false, one));
    try std.testing.expectError(error.ParallelPastMemory, fit(2 * g, 8, true, one));
    try std.testing.expectError(error.NoStreamFits, fit(g - 1, 8, false, one));
    try std.testing.expectError(error.ParallelPastRound, fit(1024 * g, 65, true, one));
}
