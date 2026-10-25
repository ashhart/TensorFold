//! Spark2.5 (MLX affine text) behind the native CUDA registry: plain and copy-drafted decoding, prompt reuse.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const be = @import("backend.zig");
const snap = @import("snap.zig");

pub const model_type = "spark2_5";
pub const formats: []const []const u8 = &.{ "mlx-q2g64", "mlx-q3g64", "mlx-q4g64", "mlx-q5g64", "mlx-q6g64", "mlx-q8g64" };
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
pub const snap_lookahead: u32 = 0;
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
    target: *be.Spark,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: kern.Kernels, s: cuda.Stream, m: fwd.Model, sb: be.Spark };

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

const Plan = struct {
    own: *const Owned,
    d: *const cuda.Driver,
    context: usize,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        return try fwd.Model.openBytes(o.gpa, p.d, o.s, &o.w, p.context, n) + be.Spark.openBytes(n);
    }
};

/// Positions a stream's caches hold: whole 1,024s.
fn positions(context: usize) usize {
    return (context + 1023) / 1024 * 1024;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .w = undefined, .k = undefined, .s = undefined, .m = undefined, .sb = undefined };
    own.k = try kern.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    own.w = try wts.load(gpa, io, ctx.d, own.s, &own.k, dir, null);
    errdefer own.w.deinit();
    const context = @min(positions(o.context), own.w.config.context);
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context };
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const n = fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, gib(room), most, if (most == 1) "" else "s" });
        }
        return e;
    };
    own.m = try fwd.Model.init(gpa, ctx.d, own.s, &own.w, context, n);
    errdefer own.m.deinit();
    own.m.graphed = !envOff("TENSORFOLD_SPARK_GRAPHS"); // =0: every decode forward eager
    own.sb = try be.Spark.init(gpa, ctx.d, &own.m, &own.k, own.s);
    errdefer own.sb.deinit();
    return .{
        .backend = own.sb.backend(),
        .facts = own.sb.facts(),
        .rows = be.max_window,
        .stream_bytes = fwd.Model.streamBytes(own.w.config.geo, own.w.layers.len, context),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.sb,
    };
}

fn envOff(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn spark(ptr: *anyopaque) *be.Spark {
    return @ptrCast(@alignCast(ptr));
}

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return snap.bytes(spark(ptr).m, at);
}

pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try spark(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return spark(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    spark(ptr).dropState(@ptrCast(@alignCast(saved)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Spark2.5 on CUDA has no draft head: it drafts from the prompt's own text",
        error.SparkTextOnly => "Spark2.5 is a text model: this request has images",
        error.StructuresUnsupported => "structured output is not built for Spark2.5 on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "Spark2.5 runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedSparkQuantization => "Spark2.5 on CUDA reads MLX affine checkpoints, one width (2-6 or 8 bits) in groups of 64: convert the bf16 release with tools/zig/spark25_convert.py",
        error.UnsupportedSparkGeometry, error.UnsupportedSparkRotary => "this spark2_5 checkpoint's geometry is not the 4B's this engine serves",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Spark2.5 on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Spark2.5 streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no Spark2.5 stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.sb.deinit();
    own.m.deinit();
    own.w.deinit();
    own.s.deinit();
    own.k.deinit();
    own.gpa.destroy(own);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.SparkTextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.UnsupportedSparkQuantization, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}
