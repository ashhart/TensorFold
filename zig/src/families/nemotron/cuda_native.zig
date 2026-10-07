//! Nemotron's CUDA engine, MTP head and lane backend behind the native family interface.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const Head = @import("cuda_mtp.zig").Head;
const Lanes = @import("cuda_lanes.zig").Cuda;
const lone = @import("cuda_lone.zig");

pub const model_type = "nemotron_h";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = engine.default_context;
pub const max_segments: u32 = @import("cuda_segments.zig").MAX;
pub const prompt_rows: u32 = state.prefill_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence (caches, state, head caches).
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null, // called with `ctx`; null: every stream in the lane core
};

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, head: ?*Head, lanes: Lanes };

/// Own-sequence graphs support both draw modes; drafting also captures the head.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = o.drafts, .graphs = true, .sampling = null, .segments = o.segments });
    errdefer e.deinit();
    const head: ?*Head = if (o.drafts) try Head.init(e) else null;
    errdefer if (head) |h| h.deinit();
    if (head) |h| try h.capture();
    if (!o.drafts) try e.captureWindows(); // the one-row window: Engine.init captures windows only when drafting
    // the sampled mode's sets: a placeholder rule now, each stream's own uploaded at its prefill
    try e.setSampling(.{ .seed = 0, .temperature = 1.0 });
    try e.captureWindows();
    if (head) |h| try h.capture();
    try e.setSampling(null);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = head, .lanes = try Lanes.init(gpa, e, head) };
    errdefer own.lanes.deinit();
    try own.lanes.measure(io, dir);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = if (head != null) state.max_rows else 1,
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .lone = if (head != null) loneRun else null,
    };
}

fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    return lone.run(own.gpa, &own.lanes, s, .{ .ctx = hooks, .committed = committed, .yield = yield });
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.deinit();
    if (own.head) |h| h.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}
