//! Gemma 4 (affine-4/64 text) behind the native CUDA registry; images, widths, drafters (steps 9-12) refused.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const fwd = @import("cuda_forward.zig");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");
const asst = @import("cuda_assistant.zig");
const cv = @import("cuda_vision.zig");
const tw = @import("cuda_tower.zig");
const dfl = @import("cuda_dflash.zig");

/// The registry's entry for one model_type: gemma4_unified (12B) or gemma4 (31B).
pub fn Family(comptime name: []const u8) type {
    return struct {
        pub const model_type = name;
        pub const formats: []const []const u8 = &.{ "mlx-q2g64", "mlx-q3g64", "mlx-q4g64", "mlx-q5g64", "mlx-q6g64", "mlx-q8g64" };
        pub const default_context: i64 = 32768;
        pub const max_segments: u32 = 1;
        pub const prompt_rows: u32 = be.prompt_rows;
        pub const snap_lookahead: u32 = 0;
        pub const takes_drafter = true;
        pub const openDrafter = native.openDrafter;
        pub const explains_open = true;
        /// An image's most tokens (the processor's largest budget) unless --image-tokens says otherwise.
        pub const default_image_tokens: u32 = 280;
        pub const vision = cv;
        pub const Room = native.Room;
        pub const Options = native.Options;
        pub const Loaded = native.Loaded;
        pub const open = native.open;
        pub const snap_bytes = native.snap_bytes;
        pub const snap_save = native.snap_save;
        pub const snap_restore = native.snap_restore;
        pub const snap_drop = native.snap_drop;
        pub const explain = native.explain;
    };
}

const native = @This();

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// No `precision` field: the server refuses --precision for this family by name. --vision: the 12B's image embedder.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null, vision: ?cv.Limits = null, drafter: ?[]const u8 = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Gemma,
    /// --vision: what the server's image requests reach (native/cuda_vision_offer.zig).
    vision: ?*cv.Vision = null,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: kern.Kernels, s: cuda.Stream, m: fwd.Model, gb: be.Gemma, v: ?*cv.Vision = null, room: ?Room = null };

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
    /// What open allocates beside the streams: the vision workspace and the drafter's weights.
    beside: u64,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        return try fwd.Model.openBytes(o.gpa, p.d, o.s, &o.w, p.context, n) + be.Gemma.openBytes(n) + p.beside;
    }
};

/// The device bytes of the image path's workspace (the 31B's tower or the 12B's embedder), 0 without --vision.
fn visionBytes(vision: bool, tower: bool) u64 {
    if (!vision) return 0;
    return if (tower) tw.Tower.workspaceBytes() else cv.Embedder.workspaceBytes();
}

/// The drafter's weights on the device: the checkpoint's files, a DFlash drafter's bf16 linears made 4-bit (5/16).
fn drafterBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (std.mem.endsWith(u8, e.name, ".safetensors")) total += (try d.statFile(io, e.name, .{})).size;
    }
    return if (try dfl.isDFlash(gpa, io, dir)) total / 16 * 5 else total;
}

/// Positions a stream's caches hold: whole 1,024s.
fn positions(context: usize) usize {
    return (context + 1023) / 1024 * 1024;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .w = undefined, .k = undefined, .s = undefined, .m = undefined, .gb = undefined };
    own.k = try kern.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    own.w = try wts.load(gpa, io, ctx.d, own.s, &own.k, dir, null, o.vision != null);
    errdefer own.w.deinit();
    const context = @min(positions(o.context), own.w.config.context);
    const beside = visionBytes(o.vision != null, own.w.tower != null) + if (o.drafter) |dd| (if (o.drafts) try drafterBytes(gpa, io, dd) else 0) else 0;
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context, .beside = beside };
    own.room = o.room;
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
    own.m.graphed = !envOff("TENSORFOLD_GEMMA4_GRAPHS"); // =0: every decode forward eager
    own.gb = try be.Gemma.init(gpa, ctx.d, &own.m, &own.k, own.s);
    errdefer own.gb.deinit();
    if (o.vision) |lim| {
        const v = try gpa.create(cv.Vision);
        errdefer gpa.destroy(v);
        v.* = .{ .gpa = gpa, .io = io, .limits = lim, .e = if (own.w.tower) |*t| .{ .tower = try tw.Tower.init(gpa, ctx.d, t, &own.k) } else .{ .embedder = try cv.Embedder.init(gpa, ctx.d, &own.w.vision.?, &own.k) } };
        own.v = v;
        own.gb.vision = v;
    }
    return .{
        .backend = own.gb.backend(),
        .facts = own.gb.facts(),
        .rows = be.max_window,
        .stream_bytes = fwd.Model.streamBytes(own.w.config.geo, own.w.layers.len, context),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.gb,
        .vision = own.v,
    };
}

fn envOff(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn gemma(ptr: *anyopaque) *be.Gemma {
    return @ptrCast(@alignCast(ptr));
}

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const g = gemma(ptr);
    return snap.bytes(g.m, at) + (if (g.keeper) |kp| kp.bytes(kp.ctx, at) else 0);
}

pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try gemma(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return gemma(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    gemma(ptr).dropState(@ptrCast(@alignCast(saved)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Gemma 4 on CUDA drafts with an assistant loaded by --drafter, or from the prompt's own text",
        error.GemmaDrafterNotServed => "--drafter: Gemma 4 on CUDA loads the 12B's MTP assistant (gemma4_unified_assistant) or a DFlash drafter (DFlashDraftModel)",
        error.DraftForAnotherTarget => "--drafter: this DFlash drafter was trained for another target (its width or tapped layers do not fit)",
        error.NotDFlashV1, error.UnsupportedDraftMask, error.UnsupportedDraftShape => "--drafter: this DFlash drafter's layout is not the stock one this engine serves",
        error.AssistantForAnotherTarget => "--drafter: this assistant was trained for another Gemma 4 target (its backbone width or key heads differ)",
        error.UnsupportedAssistantGeometry, error.UnsupportedAssistantType, error.UnsupportedAssistantRotary, error.UnsupportedAssistantQuantization, error.AssistantOrderedEmbeddingsNotServed => "--drafter: this Gemma 4 assistant's layout is not the stock one this engine serves",
        error.DrafterBitsUnsupported => "--drafter-bits: the Gemma 4 assistant runs at 4 bits on CUDA",
        error.NoGemmaVision => "--vision: this Gemma 4 checkpoint has no image embedder (vision_embedder. and embed_vision. tensors)",
        error.GemmaVisionDenseQuantized => "--vision: this checkpoint's patch_dense is quantized, and the embedder reads it as bf16 (as the reference does): use a conversion that keeps it bf16",
        error.GemmaServedWithoutVision => "this request has images, and the server was started without --vision",
        error.StructuresUnsupported => "structured output is not built for Gemma 4 on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "Gemma 4 runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Gemma 4 on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Gemma 4 streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.DrafterPastMemory => "--drafter: the CUDA memory budget leaves less room than the drafter's weights need after the streams: lower --parallel or --context, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no Gemma 4 stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

/// --drafter: the 12B's MTP assistant over the target's caches, or z-lab's DFlash drafter on the 31B's taps.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void } {
    if (bits != 4) return error.DrafterBitsUnsupported;
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.gb.lanes.count() != 0) return error.TapsAfterStreams;
    if (own.room) |r| if (r.read(r.ctx)) |left| if (left < try drafterBytes(gpa, io, dir)) return error.DrafterPastMemory;
    if (try dfl.isDFlash(gpa, io, dir)) {
        const d = try dfl.Drafter.init(gpa, io, &own.gb, dir);
        errdefer d.deinit();
        own.gb.keeper = .{ .ctx = d, .bytes = keptBytes, .keep = keepLane };
        return .{ .drafter = d.drafter(), .ctx = d, .close = closeDFlash, .reserve = reserveNone };
    }
    if (!try asst.isAssistant(gpa, io, dir)) return error.GemmaDrafterNotServed;
    const d = try asst.Drafter.init(gpa, io, &own.gb, dir);
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveNone };
}

fn closeDrafter(p: *anyopaque) void {
    const d: *asst.Drafter = @ptrCast(@alignCast(p));
    d.deinit();
}

fn closeDFlash(p: *anyopaque) void {
    const d: *dfl.Drafter = @ptrCast(@alignCast(p));
    d.gb.keeper = null;
    d.deinit();
}

fn keptBytes(p: *anyopaque, at: usize) u64 {
    const d: *dfl.Drafter = @ptrCast(@alignCast(p));
    return d.keptBytes(at);
}

fn keepLane(p: *anyopaque, s: *const lanes.Stream, k: *snap.Saved) anyerror!void {
    const d: *dfl.Drafter = @ptrCast(@alignCast(p));
    return d.keep(s, k);
}

fn reserveNone(_: *anyopaque, _: u32) anyerror!void {}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.gb.deinit();
    if (own.v) |v| {
        v.deinit();
        own.gpa.destroy(v);
    }
    own.m.deinit();
    own.w.deinit();
    own.s.deinit();
    own.k.deinit();
    own.gpa.destroy(own);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits, error.DrafterPastMemory, error.GemmaDrafterNotServed, error.AssistantForAnotherTarget, error.DrafterBitsUnsupported, error.NoGemmaVision, error.GemmaVisionDenseQuantized }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
    try std.testing.expectEqualStrings("gemma4_unified", Family("gemma4_unified").model_type);
}

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
    try std.testing.expectEqual(@as(usize, 2), try fit(2 * g, 8, false, one));
    try std.testing.expectError(error.ParallelPastMemory, fit(2 * g, 8, true, one));
    try std.testing.expectError(error.NoStreamFits, fit(g - 1, 8, false, one));
    try std.testing.expectError(error.ParallelPastRound, fit(1024 * g, be.round_rows + 1, true, one));
}

test "admission counts the vision workspace: more bytes per open, fewer streams under a tight room" {
    const g: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 0), visionBytes(false, true));
    try std.testing.expect(visionBytes(true, true) > 0 and visionBytes(true, false) > 0);
    const plain: Fake = .{ .stream = g, .fixed = visionBytes(false, true) };
    const with: Fake = .{ .stream = g, .fixed = visionBytes(true, true) };
    try std.testing.expect(try with.bytes(2) > try plain.bytes(2));
    const room = 4 * g + visionBytes(true, true) / 2;
    try std.testing.expectEqual(@as(usize, 4), try fit(room, 8, false, plain));
    try std.testing.expectEqual(@as(usize, 3), try fit(room, 8, false, with));
}
