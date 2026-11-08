//! Ripple (ricursive) at two ranks behind the native CUDA registry: rank 0 serves, rank 1 follows its plan words.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("tp2_forward.zig");
const rl = @import("tp2_lanes.zig");
const kept = @import("tp2_kept.zig");
const family = @import("tp2.zig");

pub const model_type = "ricursive";
pub const formats: []const []const u8 = &.{"exl3"};
/// The publisher's served window (config.json allows 2M): one stream at 1M by default.
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = rl.prompt_rows;
/// A kept state holds the caches through its position and nothing past it (the drafter's context is not kept).
pub const snap_lookahead: u32 = 0;
pub const explains_open = true;
/// A request's verify reserve past its prompt and reply.
pub const window_rows: u32 = rl.round_rows;
pub const placementWeightBytes = family.placementWeightBytes;
/// The ranks this family serves at (--tp 2); the registry passes other counts to another ricursive family.
pub const ranks: u32 = 2;
pub const ranks_refusal = "Ripple's CUDA path here runs on two GB10s: serve it with --tp 2 (rank 0 serves, rank 1 follows)";
/// The server's think block rules: Ripple opens `<think>` itself or answers bare, and its markers take no budget.
pub fn thinking(info: anytype) void {
    info.reasoning_optional = true;
    info.thinking_budget = false;
}

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 1, lanes_fixed: bool = false, room: ?Room = null, tp: u32 = 1, rank: u32 = 0, master: ?[]const u8 = null, master_port: u16 = 29551 };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *rl.Ripple, // the prompt cache keeps this backend's states (snap_*)
};

const Owned = struct { gpa: std.mem.Allocator, m: *fwd.Model, r: rl.Ripple, dir: []const u8 = "" };

/// Slots to allocate in `room` (`per` bytes a slot past `fixed`): as asked, else (not fixed) the most that fit.
fn fit(room: u64, asked: u32, fixed: bool, per: u64, fixed_bytes: u64) !usize {
    if (asked > fwd.max_streams) return error.ParallelPastRound;
    var n: usize = @max(asked, 1);
    while (fixed_bytes + n * per > room) : (n -= 1) {
        if (fixed) return error.ParallelPastMemory;
        if (n == 1) return error.NoStreamFits;
    }
    return n;
}

/// Rank 0's model and lanes (--tp 2 --rank 0, NCCL_COMM_ID); rank 1 (--rank 1) follows rank 0's calls, then leaves.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    const p: fwd.Place = .{ .rank = o.rank, .world = o.tp };
    try fwd.admit(try ctx.capability(), p.world, p.rank);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.dir = dir;
    // every slot's caches are allocated at load: plan them inside the room the weights leave (a rank's share)
    const c = try readConfig(gpa, io, dir);
    const hk = c.kv_heads / p.world;
    const per = fwd.Model.cacheBytes(c, hk, o.context, 1);
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const fixed = try family.placementWeightBytes(gpa, io, dir, p.world, p.rank, o.drafts) + (3 << 30); // weights, scratch
    const slots = fit(room, o.lanes, o.lanes_fixed, per, fixed) catch |e| {
        std.log.err("ricursive: {d} slot(s) at {d} tokens need {d:.2} GiB a rank with the weights and the CUDA memory budget leaves {d:.2} GiB", .{ o.lanes, o.context, gib(per * @max(o.lanes, 1) + fixed), gib(room) });
        return e;
    };
    own.m = try fwd.Model.load(gpa, io, ctx, ctx.d, dir, .{ .context = o.context, .slots = slots, .place = p, .roce = true, .graphs = true });
    errdefer own.m.deinit();
    if (p.rank != 0) follow(own);
    own.r = try rl.Ripple.init(gpa, own.m);
    own.r.priority_copies = try cuda.priority_copies.read("TF_RICURSIVE_COPIES", o.drafts);
    std.log.info("ricursive: rank 0 of {d}, {d} slot(s) x {d} tokens, {d:.2} GiB of caches a rank", .{ p.world, slots, o.context, gib(per * slots) });
    return .{
        .backend = own.r.backend(),
        .facts = own.r.facts(),
        .rows = rl.round_rows,
        .stream_bytes = per,
        .free_streams = @intCast(slots),
        .ctx = own,
        .deinit = release,
        .target = &own.r,
    };
}

/// A follower never serves: it runs rank 0's calls until rank 0 stops, frees what it owns, then leaves the process.
fn follow(own: *Owned) noreturn {
    own.m.follow() catch |err| {
        std.log.err("ricursive rank {d}: following rank 0 stopped ({s})", .{ own.m.rank(), @errorName(err) });
        std.c._exit(1); // the stream may hold kernels waiting on a gone rank: a normal exit would join them
    };
    std.log.info("ricursive rank {d}: rank 0 ended; leaving", .{own.m.rank()});
    own.m.deinit();
    std.process.exit(0);
}

fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !family.config.Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return family.config.parse(gpa, bytes);
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn ripple(ptr: *anyopaque) *rl.Ripple {
    return @ptrCast(@alignCast(ptr));
}

/// Bytes a state takes on each rank: one slot's sliding rings (the full layers' rows stay in the slot).
pub fn snap_bytes(ptr: *anyopaque, _: u32) u64 {
    return kept.bytes(ripple(ptr).m);
}

/// The prompt pass stands at `at` (a chunk end at a mark): every rank copies the slot's rings under one new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try ripple(ptr).saveState(owner, at));
}

/// States resume inside a prompt pass (Ripple.opening), where every rank restores the same slot.
pub fn snap_restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    ripple(ptr).dropState(@ptrCast(@alignCast(saved)));
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Ripple on CUDA drafts from the prompt's own text only",
        error.StructuresUnsupported => "structured output is not built for Ripple on CUDA yet",
        error.CapturedKernels => "Ripple runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.RicursiveRequiresSm121 => "Ripple's two-rank CUDA path runs on GB10 (sm_121) only so far, and this GPU is another",
        error.RicursiveNeedsTwoRanks => "Ripple's two-rank CUDA path serves at --tp 2 only (rank 0 serves, rank 1 follows)",
        error.RicursiveRankPastWorld => "--rank: Ripple's two ranks are 0 (serves) and 1 (follows)",
        error.NoCommId => "Ripple's two ranks meet over NCCL: set NCCL_COMM_ID to rank 0's address:port on both hosts",
        error.ParallelPastRound => "--parallel: Ripple on CUDA holds at most 16 streams",
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Ripple streams than asked at this --context: lower --parallel or --context, or adjust TENSORFOLD_MEMORY_RESERVE_GIB",
        error.NoStreamFits => "the CUDA memory budget fits no Ripple stream after the weights: lower --context, or adjust TENSORFOLD_MEMORY_RESERVE_GIB",
        error.RicursiveRanksDisagree => "Ripple's ranks serve different slots or contexts: start both ranks with the same --parallel and --context",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.m.stop() catch {};
    own.r.deinit();
    own.m.deinit();
    own.gpa.destroy(own);
}

test "the slots allocated at open fit the room: auto takes what fits, a fixed count refuses" {
    const g: u64 = 1 << 30;
    try std.testing.expectEqual(@as(usize, 2), try fit(10 * g, 4, false, 4 * g, g));
    try std.testing.expectError(error.ParallelPastMemory, fit(10 * g, 4, true, 4 * g, g));
    try std.testing.expectError(error.NoStreamFits, fit(g, 1, false, 4 * g, g));
    try std.testing.expectError(error.ParallelPastRound, fit(1000 * g, 17, false, g, 0));
    try std.testing.expect(explain(null, error.PastContext) != null and explain(null, error.OutOfMemory) == null);
}
