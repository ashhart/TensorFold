//! Qwen3.5-family dense models on CUDA behind the native family interface (native/cuda.zig's registry).

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const c = @import("shape.zig");
const st = @import("state.zig");
const engine = @import("engine.zig");
const Lanes = @import("lanes.zig").Cuda;
const dr = @import("draft.zig");

pub const model_type = "qwen3_5";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, drafter: ?[]const u8 = null };

/// A lone drafted stream's own driver (none here: one stream decodes in the lane core too, as under --parallel).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own state and caches.
    stream_bytes: usize,
    /// Streams that allocate nothing (none here: every stream's state is its own).
    free_streams: u32 = 0,
    /// What a --slide trainer keeps out of the streams' room (none: the 27B does not learn on CUDA yet).
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, lanes: Lanes, draft: ?*dr.DFlash2 = null };

/// The drafter the Python engine pairs with the 27B once pulled (z-lab/Qwen3.8-27B-DFlash2).
const drafter_repo = "models--z-lab--Qwen3.8-27B-DFlash2";

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// --drafter, else the pulled drafter's snapshot in the Hugging Face cache; null when neither holds one.
fn drafterDir(a: std.mem.Allocator, io: std.Io, given: ?[]const u8) !?[]const u8 {
    if (given) |d| return try a.dupe(u8, d);
    const hub = if (getenv("HF_HUB_CACHE")) |h| try a.dupe(u8, h) else if (getenv("HF_HOME")) |h| try std.fs.path.join(a, &.{ h, "hub" }) else blk: {
        const home = getenv("HOME") orelse return null;
        break :blk try std.fs.path.join(a, &.{ home, ".cache", "huggingface", "hub" });
    };
    defer a.free(hub);
    const snaps = try std.fs.path.join(a, &.{ hub, drafter_repo, "snapshots" });
    defer a.free(snaps);
    var dir = std.Io.Dir.cwd().openDir(io, snaps, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| if (entry.kind == .directory) return try std.fs.path.join(a, &.{ snaps, entry.name });
    return null;
}

/// The streams the lane core may admit at most; the server's memory admission picks how many fit.
const most_streams = 32;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    // the 27B's glue runs the Python engine's Triton kernels, captured on this chip (no own glue kernels yet)
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels orelse return error.QwenNeedsCapturedKernels, .{ .context = o.context });
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .lanes = try Lanes.init(gpa, e, most_streams) };
    errdefer own.lanes.deinit();
    if (o.drafts) if (try drafterDir(gpa, io, o.drafter)) |draft_dir| {
        defer gpa.free(draft_dir);
        const f = e.forward();
        own.draft = try dr.DFlash2.init(gpa, io, f.ops, f.t, &e.w, draft_dir, dir);
        own.lanes.drafter = own.draft;
        std.log.info("DFlash2 drafts every stream: {s}", .{draft_dir});
        // the forward's ms by rows: a shared round's tree widths follow the curve
        try own.lanes.calibrate(io, most_streams);
    };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = st.max_rows,
        .stream_bytes = st.Seq.bytes(e.w.g, o.context) + if (own.draft != null) dr.Context.bytes() else 0,
        .ctx = own,
        .deinit = release,
    };
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.QwenCheckpointHasNoDraftHead => "this checkpoint has no draft head on CUDA yet: serve with --no-drafts",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.deinit();
    if (own.draft) |d| d.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}
