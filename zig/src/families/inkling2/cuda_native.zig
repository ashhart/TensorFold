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
pub const formats: []const []const u8 = &.{ "modelopt-mixed", "tfq3" };
pub const default_context: i64 = 1_048_576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
pub const window_rows: u32 = engine.window_rows;
pub const explains_open = true;

pub const VisionOptions = struct { image_tokens: u32, max_images: u32 };
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    vision: ?VisionOptions = null,
    precision: ?enum { full } = null,
    room: ?Room = null,
    drafter: ?[]const u8 = null,
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
    reserve: *const fn (*anyopaque, u32) anyerror!void = reserve,
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

/// The server's stream count: a cache slot each, past the one allocated at load.
fn reserve(p: *anyopaque, streams: u32) anyerror!void {
    return own(p).backend.reserve(streams);
}

pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const o: *const Owned = @ptrCast(@alignCast(p));
    return o.admission.fits(prompt, max_tokens) and o.backend.admits(prompt, max_tokens);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.Inkling2CapturedKernels;
    if (o.segments != 1 or o.tp != 1 or o.rank != 0) return error.Inkling2LoopOneRankOnly;
    if (o.vision != null) return error.Inkling2LoopTextOnly;
    if (o.drafter != null) return error.Inkling2DSparkPending;
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
    try @import("names.zig").precisionsWithDir(gpa, io, merged, c.text.layers, dir, &precisions);
    const p = try plan_.build(&c.text, precisions[0..c.text.layers]);
    var w = try @import("loop_weights.zig").loadWith(gpa, io, ctx.d, &ck, &c, p, dir);
    errdefer w.deinit();
    if (ck.unused() != 0) return error.UnusedCheckpointTensor;
    ck.release();
    var k = try engine.Kernels.loadWith(ctx, w.hasTfq3(), c.text.hidden, c.text.topk, c.text.expert_inter);
    errdefer k.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    errdefer stream.deinit();
    var m = try engine.Model.init(gpa, ctx.d, stream, &w, o.context);
    errdefer m.deinit();
    var be = try backend.Backend.init(gpa, ctx.d, &m, &k, stream);
    errdefer be.deinit();
    const out = try gpa.create(Owned);
    errdefer gpa.destroy(out);
    out.* = .{ .gpa = gpa, .admission = .{ .context = @intCast(o.context) }, .kernels = k, .stream = stream, .weights = w, .model = m, .backend = be };
    // The value copies above own pointers to their old stack homes; retarget the live model/backend now.
    out.model.w = &out.weights;
    out.backend.m = &out.model;
    out.backend.k = &out.kernels;
    return .{ .backend = out.backend.backend(), .facts = out.backend.facts(), .rows = window_rows, .stream_bytes = backend.streamBytes(&out.model), .ctx = out, .deinit = release, .target = out };
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
        error.Inkling2LoopTextOnly => "Inkling 2 Mini is text-only on CUDA: images and --vision are refused",
        error.Inkling2DSparkPending => "the Mini DSpark drafter has names and taps but no native draft engine in this build",
        error.Inkling2CapturedKernels => "Inkling 2 Mini runs the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.Inkling2TooManyStreams => "Inkling 2 Mini on CUDA holds at most 16 streams: lower --parallel",
        error.StructuresUnsupported, error.ChainsOnly => "Inkling 2 Mini serves only unconstrained, non-branching text drafts",
        else => null,
    };
}

test "Mini native registration and refusal wording are explicit" {
    try std.testing.expectEqualStrings("inkling2_loop", model_type);
    try std.testing.expect(max_segments == 1 and prompt_rows == 2048 and window_rows == 16);
    try std.testing.expectEqual(16, engine.max_slots); // the stream cap explain() names
    for ([_]anyerror{ error.Inkling2LoopOneRankOnly, error.Inkling2LoopTextOnly, error.Inkling2DSparkPending, error.Inkling2CapturedKernels, error.Inkling2TooManyStreams }) |err| {
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

test "stream_bytes is what an extra stream's slot and lane buffers allocate on Mini" {
    const Record = struct { // a DeviceBuffer hook that hands out addresses and sums what is asked
        bytes: usize = 0,
        next: u64 = 1 << 32,
        fn alloc(ctx: *anyopaque, len: usize) ?u64 {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.bytes += len;
            r.next += std.mem.alignForward(u64, len, 512);
            return r.next;
        }
        fn free(_: *anyopaque, _: u64) bool {
            return true;
        }
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    var w: @import("loop_weights.zig").Weights = undefined;
    w.c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    var precisions: [24]plan_.Precision = @splat(.nvfp4);
    w.p = try plan_.build(&w.c.text, &precisions);
    var m: engine.Model = undefined;
    m.d = @ptrFromInt(0x1000); // never called: the hook answers every allocation
    m.w = &w;
    m.layout = try loop.CacheLayout.init(&w.c.text, &w.p, 8192);
    var b: backend.Backend = undefined;
    b.d = m.d;
    b.m = &m;
    b.held = 1;
    var rec: Record = .{};
    const before = cuda.memory.currentHook();
    cuda.memory.setHook(.{ .ctx = &rec, .alloc = Record.alloc, .free = Record.free });
    defer cuda.memory.setHook(before);
    var slot = try m.newSlot();
    defer for ([4]*cuda.DeviceBuffer{ &slot.cache, &slot.sites, &slot.backup, &slot.hist }) |x| x.free();
    try b.slotBuffers(2); // the second stream's lane buffers; the first already holds its own
    defer for (&b.slot[1]) |*x| x.free();
    try std.testing.expectEqual(@as(u64, rec.bytes), backend.streamBytes(&m));
    // past the cache: rewind sites, the window's backup (past conv 0) and history, the last-logits row and the rule
    const kv: u64 = @as(u64, w.c.text.local.kv) * w.c.text.local.dim;
    const backup: u64 = m.layout.bytes - m.layout.at(0).conv_offset;
    const hist: u64 = 48 * (2 * kv + 2 * w.c.text.hidden) * engine.window_rows * 4;
    const past = backend.streamBytes(&m) - m.layout.bytes;
    try std.testing.expectEqual(48 * 4 * 32 + backup + hist + 200_058 * 4 + 32, past);
}

test "the forward refuses a token id at the vocabulary before any launch" {
    var why: config.Why = .{};
    var w: @import("loop_weights.zig").Weights = undefined;
    w.c = try config.parse(std.testing.allocator, @import("config_test.zig").mini, &why);
    // No driver, kernels or caches: reaching a launch from this call would fault.
    var m: engine.Model = .{ .gpa = std.testing.allocator, .d = undefined, .w = &w, .context = 64, .layout = undefined };
    const bad = [_]u32{ 1, w.c.text.vocab };
    try std.testing.expectError(error.InvalidToken, m.forward(undefined, undefined, &bad, 0, 0, 1, false));
}
