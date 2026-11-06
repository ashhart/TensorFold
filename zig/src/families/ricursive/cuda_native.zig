//! Ripple behind the native CUDA registry: several streams, an arena slot each; the prompt cache's states; refusals.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("cuda_weights.zig");
const fwd = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const cfg = @import("config.zig");

const snap = be.snap;

pub const model_type = "ricursive";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
/// A kept state is the caches through its position and nothing past it (no draft head reads a token on).
pub const snap_lookahead: u32 = 0;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Ripple on CUDA loads no drafter yet; serve it without --drafter";
/// `explain` words this family's open refusals too.
pub const explains_open = true;

/// GB10's memory (121.7 GiB), the budget when the server gives no reading of its room.
pub const gb10_bytes: u64 = @intFromFloat(121.7 * (1 << 30));

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// --parallel: the streams made at open (auto: the most that fit, at most a round's rows over 16-row windows).
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };

/// What the native server drives: every stream's caches, allocated at open.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Ripple, // the prompt cache keeps this backend's states (snap_*)
};

/// The weights, kernels, stream, forward and backend one open owns.
pub const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: fwd.Kernels, s: cuda.Stream, m: fwd.Model, rb: be.Ripple };

/// Device bytes past the weights for the CLI's one stream at `context` positions: scratch, caches, the backend's.
pub fn openBytes(w: *const wts.Weights, k: *const fwd.Kernels, context: usize) u64 {
    return fwd.Model.openBytes(w, k, context, 1, fwd.decode_rows) + be.Ripple.openBytes(w.config.vocab, 1);
}

/// `need` bytes in `room`, or the refusal: the caches of the asked context do not fit.
fn admit(need: u64, room: u64, context: usize) !void {
    if (need <= room) return;
    std.log.err("ripple: a {d}-token context needs {d:.2} GiB of caches and scratch past the weights; the CUDA memory budget leaves {d:.2} GiB", .{ context, gib(need), gib(room) });
    return error.NoStreamFits;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// The request's context option, refused past the model's or empty.
fn contextOf(context: usize) !usize {
    if (context == 0 or context > cfg.max_context) return error.PastContext;
    return context;
}

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

/// The device bytes `n` streams take at `positions`: the forward's scratch for a round, their slots, the backend's.
const Plan = struct {
    w: *const wts.Weights,
    k: *const fwd.Kernels,
    positions: usize,

    fn bytes(p: Plan, n: usize) !u64 {
        return fwd.Model.openBytes(p.w, p.k, p.positions, n, be.round_rows) + be.Ripple.openBytes(p.w.config.vocab, n);
    }
};

/// The checkpoint through the CUDA gate, the kernels and a stream; the forward is made by the caller.
fn begin(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Owned {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.w = try wts.load(gpa, io, ctx.d, dir);
    errdefer own.w.deinit();
    own.k = try fwd.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    return own;
}

/// begin's parts, freed when an open stops before its forward is made.
fn abandon(own: *Owned) void {
    own.s.deinit();
    own.k.deinit();
    own.w.deinit();
    own.gpa.destroy(own);
}

fn roomOf(room_reading: ?Room) u64 {
    // the server's budget; a CLI run has none and lets the allocation refuse, as Dory's does
    return if (room_reading) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
}

/// The CLI's open: one slot of `context` positions and one-row rounds, if they fit the room (null: unchecked).
pub fn openOwned(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, context_asked: usize, room_reading: ?Room) !*Owned {
    const context = try contextOf(context_asked);
    const own = try begin(gpa, io, ctx, dir);
    errdefer abandon(own);
    try admit(openBytes(&own.w, &own.k, context), roomOf(room_reading), context);
    own.m = try fwd.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, 1, fwd.decode_rows);
    errdefer own.m.deinit();
    own.rb = try be.Ripple.init(gpa, ctx.d, &own.m);
    return own;
}

pub fn closeOwned(own: *Owned) void {
    own.rb.deinit();
    own.m.deinit();
    abandon(own);
}

/// The server's open: as many streams as asked and fit (auto: the most), each a slot of context and verify reserve.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows / be.max_window) return error.ParallelPastRound; // before the weights load
    if (o.segments > 1) return error.SegmentsUnsupported;
    const positions = be.physical(try contextOf(o.context));
    const own = try begin(gpa, io, ctx, dir);
    errdefer abandon(own);
    const plan: Plan = .{ .w = &own.w, .k = &own.k, .positions = positions };
    const room = roomOf(o.room);
    const n = fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token context, and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), o.context, gib(room), most, if (most == 1) "" else "s" });
        }
        if (e == error.NoStreamFits) std.log.err("ripple: one stream at a {d}-token context needs {d:.2} GiB of caches and scratch past the weights; the CUDA memory budget leaves {d:.2} GiB", .{ o.context, gib(try plan.bytes(1)), gib(room) });
        return e;
    };
    own.m = try fwd.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, positions, n, be.round_rows);
    errdefer own.m.deinit();
    own.rb = try be.Ripple.init(gpa, ctx.d, &own.m);
    std.log.info("Ripple on CUDA: {d} stream{s} of {d} positions (a {d}-token context and a {d}-row verify reserve), rounds of {d} rows, prompt chunks of {d}", .{ n, if (n == 1) "" else "s", positions, o.context, be.reserve, be.round_rows, fwd.prompt_rows });
    return .{
        .backend = own.rb.backend(),
        .facts = own.rb.facts(),
        .rows = be.max_window,
        .stream_bytes = fwd.Arena.bytes(own.w.config, positions, 1),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.rb,
    };
}

fn release(p: *anyopaque) void {
    closeOwned(@ptrCast(@alignCast(p)));
}

/// Whether a request's caches could start now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(p));
    return own.rb.admits(prompt, max_tokens);
}

fn ripple(ptr: *anyopaque) *be.Ripple {
    return @ptrCast(@alignCast(ptr));
}

/// Bytes a state at `at` prompt tokens takes.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return snap.bytes(ripple(ptr).m.c, at);
}

/// The prompt pass stands at `at` (a chunk start or a mark): copy the stream's caches there.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try ripple(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return ripple(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    ripple(ptr).dropState(@ptrCast(@alignCast(saved)));
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it, lower max_tokens or raise --context (at most 262,144)",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoStreamFits => "the CUDA memory budget fits no Ripple stream at this context after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Ripple on CUDA holds at most {d} streams (a round's rows over 16-row windows): serve --parallel {d} or fewer, or auto", .{ be.round_rows / be.max_window, be.round_rows / be.max_window }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Ripple streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.SegmentsUnsupported => "--segments: Ripple on CUDA runs its prompt chunks one after another: serve without --segments",
        error.NoDraftHead => "Ripple on CUDA has no draft head: drafts come from the prompt's own text only",
        error.StructuresUnsupported => "structured output is not built for Ripple on CUDA yet",
        error.CapturedKernels => "Ripple runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BuiltWithoutKernels => "this build carries no CUDA kernel images: build with -Dnvcc",
        error.RicursiveCudaNeedsNvfp4 => "Ripple on CUDA serves the NVFP4 release (ModelOpt W4A16 MLPs, bf16 attention): the BF16 checkpoint's 130 GB do not fit GB10",
        error.UnsupportedRicursiveCudaShape => "this Ripple checkpoint's shape is not one the CUDA kernels take (group of 2 to 16 query heads in pairs, hidden size at most 8,192, widths in whole 128s)",
        error.Unimplemented => "Ripple's CUDA global-layer decode is not in this build yet",
        else => null,
    };
}

test "ricursive cuda native: every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoStreamFits, error.ParallelPastRound, error.ParallelPastMemory, error.SegmentsUnsupported, error.NoDraftHead, error.StructuresUnsupported, error.CapturedKernels, error.BuiltWithoutKernels, error.RicursiveCudaNeedsNvfp4, error.UnsupportedRicursiveCudaShape, error.Unimplemented }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    for ([_]anyerror{ error.OutOfMemory, error.SamplingUnsupported, error.LogprobsUnsupported, error.WindowTooWide }) |e| {
        try std.testing.expect(explain(null, e) == null);
    }
}

test "ricursive cuda native: the release at 32,768 and 262,144 positions on GB10; a context past the room refused" {
    var c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    c.fp4_mlp = true;
    const weights: u64 = wts.deviceBytes(c);
    const scratch: u64 = 1_200_000_000; // the forward's 2,048 rows (cuda_model's test bounds it)
    for ([_]usize{ 32768, 262144 }) |context| {
        const need = fwd.Arena.bytes(c, be.physical(context), 1) + scratch;
        try admit(need, gb10_bytes - weights, context);
    }
    try std.testing.expectEqual(@as(usize, 32768), try contextOf(32768));
    try std.testing.expectError(error.PastContext, contextOf(cfg.max_context + 1));
    try std.testing.expectError(error.PastContext, contextOf(0));
}

/// A synthetic plan: `stream` bytes a stream plus `fixed` bytes of scratch.
const Fake = struct {
    stream: u64,
    fixed: u64,
    fn bytes(f: Fake, n: usize) !u64 {
        return f.fixed + n * f.stream;
    }
};

test "ricursive cuda native: the streams made at open fit the room; auto takes what fits, a fixed count refuses" {
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

test "ricursive cuda native: streams at a context on GB10 by the release's slot bytes" {
    var c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    c.fp4_mlp = true;
    const room = gb10_bytes - wts.deviceBytes(c);
    const scratch: u64 = 1_400_000_000; // a round's and a chunk's scratch, about (cuda_model's test bounds the chunk's)
    // a slot: 64 rings of 3,072 rows (805 MB) and 16 global layers' pages for the positions
    const Slots = struct {
        c: cfg.Config,
        positions: usize,
        fixed: u64,
        fn bytes(p: @This(), n: usize) !u64 {
            return p.fixed + fwd.Arena.bytes(p.c, p.positions, n);
        }
    };
    try std.testing.expectEqual(@as(usize, 8), try fit(room, 8, false, Slots{ .c = c, .positions = be.physical(32768), .fixed = scratch }));
    try std.testing.expect(try fit(room, 8, false, Slots{ .c = c, .positions = be.physical(262144), .fixed = scratch }) < 8);
}

test "the prompt cache's hooks compile against the store's vtable" {
    const pc: struct { bytes: *const fn (*anyopaque, u32) u64, save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque, restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void, drop: *const fn (*anyopaque, *anyopaque) void } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
}

test "ricursive cuda native: the open path type-checks against the registry's shape" {
    std.testing.refAllDecls(@This());
}
