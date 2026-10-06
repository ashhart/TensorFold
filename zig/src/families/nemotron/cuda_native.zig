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
pub const max_segments: u32 = @import("cuda_segments.zig").MAX;
pub const prompt_rows: u32 = state.prefill_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence (caches, state, head caches).
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
};

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, head: ?*Head, lanes: Lanes, sampling: []u8 = &.{} };

/// The engine without graphs (the lane rounds' windows vary) and the head when drafting, as the CLI's `lanes` runs it.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = o.drafts, .graphs = false, .sampling = null, .segments = o.segments });
    errdefer e.deinit();
    const head: ?*Head = if (o.drafts) try Head.init(e) else null;
    errdefer if (head) |h| h.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = head, .lanes = try Lanes.init(gpa, e, head) };
    errdefer own.lanes.deinit();
    try own.lanes.measure(io, dir);
    own.sampling = try samplingNote(gpa, e);
    errdefer gpa.free(own.sampling);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = if (head != null) state.max_rows else 1,
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
    };
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(ctx: ?*anyopaque, err: anyerror) ?[]const u8 {
    const own: *Owned = @ptrCast(@alignCast(ctx.?));
    return switch (err) {
        error.SamplingNotCaptured, error.NucleusUnsupported, error.TooManyCandidates => own.sampling,
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

/// The sampled rules the captured kernel set draws, as a refusal names them.
fn samplingNote(gpa: std.mem.Allocator, e: *const engine.Engine) ![]u8 {
    var rules: [16]@import("cuda_triton.zig").Keyed = undefined;
    const n = e.k.sampledRules(&rules);
    if (n == 0) return gpa.dupe(u8, "this CUDA kernel set draws greedily only: send temperature 0");
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try out.writer.writeAll("this CUDA kernel set samples only with ");
    for (rules[0..n], 0..) |r, i| try out.writer.print("{s}top_k {d}, top_p {s}, {s}", .{ if (i > 0) "; or " else "", r.k, if (r.cut) "below 1" else "1", if (r.minp) "a min_p" else "no min_p" });
    try out.writer.writeAll(" (its capture recorded no other rule): send one of those, or temperature 0");
    return out.toOwnedSlice();
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.gpa.free(own.sampling);
    own.lanes.deinit();
    if (own.head) |h| h.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}
