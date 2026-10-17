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

/// No `vision` or `precision` field: the server refuses --vision and --precision for this family by name.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8 };

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

/// Streams whose caches the open allocates: the server's --parallel, at most a round's rows.
fn streams(lanes_asked: u32) usize {
    return std.math.clamp(@as(usize, lanes_asked), 1, be.round_rows);
}

/// Positions a stream's caches hold: the window, whole 1,024s.
fn positions(context: usize) usize {
    return (context + 1023) / 1024 * 1024;
}

/// The checkpoint, the shared kernels and every stream's caches.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
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
    const n = streams(o.lanes);
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
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.StructuresUnsupported, error.CapturedKernels }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
    try std.testing.expectEqual(@as(usize, 65536), positions(65000));
    try std.testing.expectEqual(@as(usize, 1), streams(0));
    try std.testing.expectEqual(@as(usize, be.round_rows), streams(1000));
}
