//! Nemotron for the native server on CUDA: the engine, its MTP head and the lane backend, loaded as `tensorfold lanes`
//! loads them. native/cuda.zig drives what `open` returns; nothing here knows the server.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const Head = @import("cuda_mtp.zig").Head;
const Lanes = @import("cuda_lanes.zig").Cuda;

pub const model_type = "nemotron_h";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = engine.default_context;
pub const prefill_step: u32 = state.prefill_rows;

pub const Options = struct { context: usize, drafts: bool };

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
};

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, head: ?*Head, lanes: Lanes };

/// The engine without graphs (the lane rounds' windows vary) and the head when drafting, as the CLI's `lanes` runs it.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = o.drafts, .graphs = false, .sampling = null, .segments = 1 });
    errdefer e.deinit();
    const head: ?*Head = if (o.drafts) try Head.init(e) else null;
    errdefer if (head) |h| h.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = head, .lanes = try Lanes.init(gpa, e, head) };
    errdefer own.lanes.deinit();
    try own.lanes.measure(io, dir);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = if (head != null) state.max_rows else 1,
        .ctx = own,
        .deinit = release,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.deinit();
    if (own.head) |h| h.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}
