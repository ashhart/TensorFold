//! Ling-3.1-flash on the native CUDA server at TP=1: the engine, its lane backend, the prompt cache's copies.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const ling_lanes = @import("lanes.zig");

pub const model_type = "bailing_hybrid";
pub const formats: []const []const u8 = &.{ "bf16", "fp8-block", "modelopt-nvfp4" };
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *engine.Engine, // the prompt cache keeps this engine's states (snap_*)
};

/// Positions a sequence holds: `context` plus a verify window, a multiple of 64.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + engine.window_rows, 64);
}

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, lanes: ling_lanes.Lanes };

/// TF_LING_PROMPT_STEPS=0 passes prompts whole, TF_LING_SHARED=0 runs windows alone.
fn off(name: [*:0]const u8) bool {
    return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // our own kernels: a captured set is not used
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, .{ .capacity = capacity(o.context) });
    errdefer own.e.deinit();
    own.lanes = .{ .e = own.e, .whole = off("TF_LING_PROMPT_STEPS"), .alone = off("TF_LING_SHARED") };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = ling_lanes.max_rows,
        .stream_bytes = own.e.seqBytes(),
        .ctx = own,
        .deinit = close,
        .target = own.e,
    };
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.lanes.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}

/// Bytes a state takes at any position: the sequence's arena.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    _ = at;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.stateBytes();
}

/// The prompt pass stands at `at` (a mark): copy the state there under a new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pos() != at) return error.NotAtMark;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try e.keepState(), .at = at };
    return k;
}

/// The lane backend restores inside its own prompt pass (Lanes.opening), never from the store's `begin`.
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.BackendRestores;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    e.dropState(k.id);
    e.gpa.destroy(k);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Quant => "Ling-3.1 on CUDA reads bf16, block-FP8 and ModelOpt NVFP4 checkpoints (the log names what config.json has)",
        error.QuantizedRouter => "Ling-3.1's routing kernels read a bf16 router (mlp.gate): this checkpoint quantized it",
        error.PromptTooLong => "the prompt and its reply exceed the window: raise --context",
        else => null,
    };
}

test "capacity keeps a verify window past the context and a multiple of 64" {
    try std.testing.expectEqual(@as(usize, 8256), capacity(8192));
    try std.testing.expect(capacity(1000) % 64 == 0 and capacity(1000) >= 1016);
}
