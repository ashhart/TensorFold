//! Ling-3.1-flash on the native CUDA server, one GPU or TP=2 over hosts (rank 0 serves, rank 1 follows).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const ling_lanes = @import("lanes.zig");
const tp = @import("tp.zig");

pub const model_type = "bailing_hybrid";
pub const formats: []const []const u8 = &.{ "bf16", "fp8-block", "modelopt-nvfp4", "exl3" };
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
/// Kept states on disk (--learn-conversations, --spill-gib): every rank's share in rank 0's store.
pub const kept_disk = @import("disk.zig");

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

/// What a rank owns: the engine, rank 0's lanes, and at TP=2 the group and the steps' link.
const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, lanes: ling_lanes.Lanes = undefined, group: ?cuda.tp_group.Group = null, link: ?tp.Link = null };

/// `explain` words this family's open refusals too (where to set the rank, the world and the id).
pub const explains_open = true;

/// Bytes this rank's weights take at most, for the server's admission: half at TP=2 (heads, widths, vocabulary).
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    _ = .{ a, io, dir };
    const p = tp.place(tp.env) catch return total;
    return total / p.world;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // our own kernels: a captured set is not used
    const p = try tp.place(tp.env);
    if (p.world == 1) return openRank(gpa, io, ctx, dir, o, p);
    if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    try cuda.tp_watch.start(io, p.rank, p.world); // each rank's TCP line to rank 0 at NCCL_COMM_ID's port + 1
    return openRank(gpa, io, ctx, dir, o, p) catch |err| {
        cuda.tp_watch.fail(@errorName(err)); // the other ranks stop with this one
        return err;
    };
}

/// GLM's two-rank start: NCCL from rank 0's id, this rank's share loaded; rank 1 follows and exits, rank 0 serves.
fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: tp.Place) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = undefined };
    if (p.world > 1) own.group = try tp.group(io, p);
    errdefer if (own.group) |*g| g.close();
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, .{ .capacity = capacity(o.context), .mtp = o.drafts, .context = o.context, .group = if (own.group) |*g| g else null });
    errdefer own.e.deinit();
    if (own.group) |*g| {
        try tp.roce(g, gpa, io, ctx.d, own.e.stream);
        own.link = try tp.Link.init(gpa, ctx.d, &g.lib, g.comm, own.e.stream, capacity(o.context));
        own.e.link = &own.link.?;
        own.e.lead = p.rank == 0;
        try cuda.tp_watch.ready(io, try digest(gpa, io, dir, o)); // the ranks agree on the checkpoint and settings
        if (p.rank != 0) { // a follower never serves: rank 0's steps, then it leaves the process
            tp.follow(own.e, io) catch |err| {
                std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
                cuda.tp_watch.fail(@errorName(err));
                std.c._exit(1); // the stream may hold kernels waiting on a gone rank: a normal exit would join them
            };
            std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
            release(own);
            std.process.exit(0);
        }
        own.e.asleep = cuda.tp_watch.open(); // the other ranks begin asleep: the first step wakes them
    }
    own.lanes = .{ .e = own.e };
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

/// What the ranks must agree on: config.json's bytes, the context and drafts.
fn digest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) ![32]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(text);
    h.update(std.mem.asBytes(&[_]u64{ o.context, @intFromBool(o.drafts) }));
    return h.finalResult();
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    cuda.tp_watch.ended(); // the other ranks leave on the end step, so their lines close on purpose from here
    own.e.end() catch {};
    own.lanes.deinit();
    release(own);
}

/// What a rank owns, freed by rank 0's close and by a follower before it leaves.
fn release(own: *Owned) void {
    own.e.deinit();
    if (own.link) |*l| l.deinit();
    if (own.group) |*g| g.close();
    own.gpa.destroy(own);
}

/// Bytes a state kept at `at` takes: its MLA rows below `at`, the rest of the sequence's arena whole.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.stateBytes(at);
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
        error.NoRank => "Ling at TP over hosts: set TF_LING_RANK and TF_LING_WORLD (1 or 2) on every host, or neither for one GPU",
        error.TpWorld => "Ling on CUDA runs on one GPU or two ranks: set TF_LING_WORLD to 1 or 2",
        error.NoCommId => "Ling's ranks meet over NCCL: with TF_LING_WORLD 2, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.TpFormat => "TP=2 reads EXL3 packs (their 128-wide blocks are the cuts): serve another format on one GPU",
        error.TpShape => "this checkpoint's heads or widths do not split into two shares of whole 128-wide blocks (the log names which)",
        error.RopeScaling => "config.json's rope_scaling is not one the engine runs: yarn only (the log names what it read)",
        else => null,
    };
}

test "capacity keeps a verify window past the context and a multiple of 64" {
    try std.testing.expectEqual(@as(usize, 8256), capacity(8192));
    try std.testing.expect(capacity(1000) % 64 == 0 and capacity(1000) >= 1016);
}
