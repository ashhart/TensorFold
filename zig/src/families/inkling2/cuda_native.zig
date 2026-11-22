//! Inkling 2 Mini's complete one-rank, text-only CUDA engine and checkpoint/snapshot admission.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const loop = @import("loop.zig");
const engine = @import("loop_engine.zig");
const backend = @import("loop_backend.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{"modelopt-mixed"};
pub const default_context: i64 = 1_048_576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
pub const window_rows: u32 = engine.window_rows;
pub const explains_open = true;
pub const takes_drafter = true;

pub const VisionOptions = struct { image_tokens: u32, max_images: u32 };
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    lanes: u32 = 1,
    lanes_fixed: bool = false,
    slide: bool = false,
    vision: ?VisionOptions = null,
    precision: ?enum { full } = null,
    room: ?Room = null,
    tp: usize = 1,
    rank: usize = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29551,
};

pub const Owned = struct {
    gpa: std.mem.Allocator,
    admission: loop.Admission,
    kernels: engine.Kernels,
    stream: cuda.Stream,
    weights: @import("loop_weights.zig").Weights,
    model: engine.Model,
    backend: backend.Backend,
};

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *Owned,
};

fn own(ptr: *anyopaque) *Owned {
    return @ptrCast(@alignCast(ptr));
}

fn release(p: *anyopaque) void {
    const o = own(p);
    o.stream.synchronize() catch {};
    o.backend.deinit();
    o.stream.deinit();
    o.model.deinit();
    o.kernels.deinit();
    o.weights.deinit();
    const gpa = o.gpa;
    gpa.destroy(o);
}

pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const o: *const Owned = @ptrCast(@alignCast(p));
    return o.admission.fits(prompt, max_tokens) and o.backend.admits(prompt, max_tokens);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.Inkling2CapturedKernels;
    if (o.segments != 1 or o.tp != 1 or o.rank != 0) return error.Inkling2LoopOneRankOnly;
    if (o.vision != null) return error.Inkling2LoopTextOnly;
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    var why: config.Why = .{};
    const c = config.parse(gpa, bytes, &why) catch |err| {
        std.log.err("Inkling 2 Mini config refused: {s}", .{why.text()});
        return err;
    };
    try loop.validateLoad(&c, 1, false);
    if (o.context == 0 or o.context > c.text.max_context or o.context > std.math.maxInt(u32)) return error.BadLoopCache;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try ck.skip("model.visual."); // The checkpoint carries Small's tower; text-only Mini never loads or executes it.
    var precisions: [config.max_layers]plan_.Precision = undefined;
    @memset(&precisions, .nvfp4);
    var header_arena = std.heap.ArenaAllocator.init(gpa);
    defer header_arena.deinit();
    var merged: core.safetensors.Header = .empty;
    for (ck.files.items) |file| try @import("names.zig").merge(header_arena.allocator(), &merged, file.names);
    try @import("names.zig").precisions(merged, c.text.layers, &precisions);
    const p = try plan_.build(&c.text, precisions[0..c.text.layers]);
    var w = try @import("loop_weights.zig").load(gpa, ctx.d, &ck, &c, p);
    errdefer w.deinit();
    if (ck.unused() != 0) return error.UnusedCheckpointTensor;
    ck.release();
    var k = try engine.Kernels.load(ctx);
    errdefer k.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    errdefer stream.deinit();
    const kv_int8 = loop.kvInt8Enabled();
    var slots: u32 = @max(1, @min(o.lanes, engine.max_segments));
    if (o.room) |rm| if (rm.read(rm.ctx)) |avail| {
        const per_slot = (try loop.CacheLayout.initMode(&w.c.text, &w.p, @intCast(o.context), kv_int8)).bytes;
        const fit: u32 = @intCast(@min(@as(u64, engine.max_segments), avail / @max(per_slot, 1)));
        if (o.lanes_fixed and fit < slots) return error.Inkling2LoopSlotsDoNotFit;
        if (fit == 0) return error.Inkling2LoopSlotsDoNotFit;
        if (fit < slots) slots = fit;
    };
    var m = try engine.Model.initMode(gpa, ctx.d, stream, &w, o.context, slots, kv_int8);
    errdefer m.deinit();
    if (kv_int8) std.log.info("inkling2: global KV cache int8 enabled (per_slot={d}B)", .{m.layout.bytes});
    var be = try backend.Backend.init(gpa, ctx.d, &m, &k, stream);
    errdefer be.deinit();
    const out = try gpa.create(Owned);
    errdefer gpa.destroy(out);
    out.* = .{ .gpa = gpa, .admission = .{ .context = @intCast(o.context) }, .kernels = k, .stream = stream, .weights = w, .model = m, .backend = be };
    // The value copies above own pointers to their old stack homes; retarget the live model/backend now.
    out.model.w = &out.weights;
    out.backend.m = &out.model;
    out.backend.k = &out.kernels;
    return .{ .backend = out.backend.backend(), .facts = out.backend.facts(), .rows = window_rows, .stream_bytes = out.model.layout.bytes, .free_streams = slots, .ctx = out, .deinit = release, .target = out };
}

pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, _: u8) !struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void } {
    const o = own(p);
    const d = try @import("dspark_cuda.zig").Drafter.init(gpa, io, o, dir);
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveDrafter };
}

fn closeDrafter(p: *anyopaque) void {
    const d: *@import("dspark_cuda.zig").Drafter = @ptrCast(@alignCast(p));
    d.deinit();
}

fn reserveDrafter(p: *anyopaque, streams: u32) anyerror!void {
    const d: *@import("dspark_cuda.zig").Drafter = @ptrCast(@alignCast(p));
    // Commit the admitted lane caches up front; fixed weights + scratch were committed at init.
    try d.reserve(streams);
}

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return own(ptr).model.snapshotBytes(at);
}

pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try own(ptr).backend.saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return own(ptr).backend.restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    own(ptr).backend.dropState(@ptrCast(@alignCast(saved)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Inkling2LoopOneRankOnly => "Inkling 2 Mini runs at one CUDA rank with one segment; TP and multi-segment modes are refused",
        error.Inkling2LoopSlotsDoNotFit => "Inkling 2 Mini cannot fit the requested --parallel streams at this --context; lower --parallel or --context",
        error.Inkling2LoopTextOnly => "Inkling 2 Mini is text-only on CUDA: images and --vision are refused",
        error.Inkling2DSparkPending => "the Mini DSpark drafter has names and taps but no native draft engine in this build",
        error.Inkling2CapturedKernels => "Inkling 2 Mini runs the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.StructuresUnsupported, error.ChainsOnly => "Inkling 2 Mini serves only unconstrained, non-branching text drafts",
        else => null,
    };
}

test "Mini native registration and refusal wording are explicit" {
    try std.testing.expectEqualStrings("inkling2_loop", model_type);
    try std.testing.expect(max_segments == 1 and prompt_rows == 2048 and window_rows == 16);
    for ([_]anyerror{ error.Inkling2LoopOneRankOnly, error.Inkling2LoopSlotsDoNotFit, error.Inkling2LoopTextOnly, error.Inkling2DSparkPending, error.Inkling2CapturedKernels }) |err| {
        try std.testing.expect(explain(null, err) != null);
    }
}

test "the complete prompt snapshot hooks match the store vtable" {
    const pc: struct {
        bytes: *const fn (*anyopaque, u32) u64,
        save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
        restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
        drop: *const fn (*anyopaque, *anyopaque) void,
    } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
}
