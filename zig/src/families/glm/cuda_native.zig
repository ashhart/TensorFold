//! GLM-5.3-Flash NVFP4 native CUDA at TP over processes (TF_GLM_RANK, NCCL_COMM_ID): rank 0 serves, others follow.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const images_mod = @import("cuda_images.zig");
const glm_lanes = @import("cuda_lanes.zig");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const dflash = @import("dflash.zig");
const dflash_load = @import("dflash_load.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
// cc-defer: a fixed window; the Python family sizes it from free memory (capacity.admit), port that when serving needs
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fw.prefill_rows;
/// Prompt kernels change arithmetic with a chunk's rows: states keep and resume only on chunk starts.
pub const snap_grid: u32 = prompt_rows;
/// `explain` words this family's open refusals too (where to set the rank and the id).
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// `slide` stays false: the server refuses --slide for a family without a learner (no `slide` here).
pub const Options = struct {
    context: usize,
    kv8: bool = false,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    vision: ?images_mod.Limits = null, // --vision: rank 0's tower, every rank's image rows
    drafter: ?[]const u8 = null, // --drafter: DFlash2's folder (the same on every rank; needs drafts on)
    drafter_bits: u8 = 4, // --drafter-bits: 4 (default) or 0 (bf16)
};
/// --drafter: DFlash2 (incoai/GLM-5.3-Flash-DFlash2, non-commercial weights) drafts instead of the MTP head.
pub const takes_drafter = true;
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *engine.Engine, // the prompt cache keeps this engine's states (snap_*)
    peer_room: ?Peer = null, // rank 0: the server admits streams and kept states by this when it is smaller
    vision: ?*images_mod.Images = null, // rank 0's tower for the server's image offer
};

/// This rank's place: TF_GLM_RANK of TF_GLM_WORLD (both, or neither for a refusal that names them).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = std.c.getenv("TF_GLM_RANK") orelse return error.NoRank;
    const w = std.c.getenv("TF_GLM_WORLD") orelse return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, std.mem.span(r), 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, std.mem.span(w), 10) catch return error.NoRank };
    return checked(p);
}

/// A place the family serves: a rank inside its world, and at most `ranks.max_world` ranks.
fn checked(p: Place) !Place {
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > ranks.max_world) return error.TooManyRanks;
    return p;
}

/// Bytes this rank's weights take at most, for the server's admission: the checkpoint times its largest axis share.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    const p = place() catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    return @intCast(@as(u128, total) * f[0] / f[1]);
}

/// Slots for `context` plus a verify window, a multiple of 64 so the sparse pool keeps the AOT set.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + fw.window_rows, 64);
}

const snapshots = @import("cuda_snapshots.zig");

const Owned = struct {
    gpa: std.mem.Allocator,
    group: cuda.tp_group.Group, // the ranks' communicator; two ranks add RoCE writes for the main stream's small gathers
    e: *engine.Engine,
    link: follow.Link,
    lanes: glm_lanes.Lanes,
};

const env = @import("sys").env;

/// Every peer's send/recv connected now, before the weights fill memory (connected on first use, ibv_reg_mr can fail).
fn warmPeers(d: *const cuda.Driver, g: *const cuda.tp_group.Group) !void {
    const bytes = 4 << 20; // a prompt piece's exchange: every channel a large send/recv takes
    var buf = try cuda.DeviceBuffer.alloc(d, 2 * bytes * g.world);
    defer buf.free();
    var s = try cuda.Stream.init(d, false);
    defer s.deinit();
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        for (0..g.world) |peer| if (peer != g.rank) {
            g.send(s, buf.ptr + peer * bytes, bytes, peer) catch |err| break :post err;
            g.recv(s, buf.ptr + (g.world + peer) * bytes, bytes, peer) catch |err| break :post err;
        };
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    try s.synchronize();
}

/// TF_ROCE_MAX_KB in KiB, bounded by the RDMA scatter/gather length before any allocation.
pub fn roceBytes(text: ?[]const u8) !usize {
    const v = text orelse return 4096 << 10;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadRoceCap;
    if (n == 0 or n > std.math.maxInt(u32) / 1024) return error.BadRoceCap;
    return std.math.mul(usize, n, 1024) catch return error.BadRoceCap;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // our own kernels: a captured set is not used
    const p = try place();
    const roce_bytes = try roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null);
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    var problem: []const u8 = "";
    var set = ranks.Settings.read(&ranks.environment, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    set.kv8 = o.kv8;
    // the ranks' lines before NCCL (cuda/tp_watch.zig): a rank that fails, refuses or dies stops every rank
    if (p.world > 1) try cuda.tp_watch.start(io, p.rank, p.world);
    return openRank(gpa, io, ctx, dir, o, p, set, roce_bytes) catch |err| {
        cuda.tp_watch.fail(@errorName(err)); // the other ranks stop with this rank's error
        return err;
    };
}

/// `open` once the ranks' lines are up: NCCL, the engine and the link, then a follower's loop or rank 0's lanes.
fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: Place, set: ranks.Settings, roce_bytes: usize) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    // every connection registered at init, while memory is free (lazily, a full GB10's ibv_reg_mr can refuse it)
    if (std.c.getenv("NCCL_RUNTIME_CONNECT") == null) _ = env.setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    // NCCL's cuMem host buffers wore out GB10's driver pool under load (NV_ERR_NO_MEMORY; next ranks failed to start)
    if (std.c.getenv("NCCL_CUMEM_HOST_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_HOST_ENABLE", "0", 0);
    if (std.c.getenv("NCCL_CUMEM_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_ENABLE", "0", 0);
    var lib = try cuda.nccl.Library.open();
    errdefer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // NCCL_COMM_ID: the same id on every rank
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    own.group = .{ .lib = lib, .comm = comm, .rank = p.rank, .world = p.world };
    errdefer _ = own.group.lib.api.ncclCommDestroy(own.group.comm);
    if (p.world > 1) try warmPeers(ctx.d, &own.group);
    const cap = capacity(o.context);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.group.lib, own.group.comm, p.rank, p.world, .{ .capacity = cap, .mtp = o.drafts, .settings = set, .vision = o.vision, .drafter = o.drafter != null });
    errdefer own.e.deinit();
    if (o.vision) |lim| { // before the rooms: the tower's bytes count against rank 0's
        var problem: []const u8 = "";
        const ql: cuda.qlinear.Linear = .{ .lane = &own.e.m.lane, .nvfp4_prompt = &own.e.m.prompt }; // the tower's MXFP8, NVFP4
        own.e.images = (try images_mod.Images.init(gpa, io, ctx.d, own.e.stream, dir, lim, fw.prefill_rows + 1, p.rank == 0, ql, &problem)) orelse {
            std.log.err("{s}", .{problem});
            return error.NoVisionTower;
        };
    }
    if (o.drafter) |dd| { // before the rooms: its weights and scratch count against every rank's
        if (!o.drafts) return error.DrafterNeedsDrafts;
        own.e.drafter = try dflash_load.load(gpa, io, own.e, dd, p.rank, p.world, o.drafter_bits);
    }
    try own.group.openRoce(gpa, io, ctx.d, own.e.stream, roce_bytes); // both ranks bring it up or neither does
    errdefer if (own.group.fast) |f| f.deinit();
    own.e.group = &own.group;
    own.link = try follow.Link.init(gpa, ctx.d, &own.group.lib, own.group.comm, own.e.stream, try linkCapacity(cap, o.vision));
    errdefer own.link.deinit();
    own.e.snap_setup = .{ .ptr = own, .run = readySnapshots };
    // Rank 0 admits by the smallest room of the ranks; its sizes are the largest (the first ranks take the remainder).
    const rooms = try gpa.alloc(u64, p.world);
    defer gpa.free(rooms);
    const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    try ranks.gather(u64, ctx.d, &own.group.lib, own.group.comm, own.e.stream, p.rank, mine, rooms);
    var peer: ?Peer = null;
    for (rooms[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
        peer = .{ .bytes = r, .rank = q };
    };
    // every rank loaded and agreed (cuda_ranks) before the first step; the digest names what they agreed on
    if (p.world > 1) try cuda.tp_watch.ready(io, own.e.checkpoint_digest);
    if (p.rank != 0) { // a follower never serves: it runs rank 0's steps, then leaves the process
        follow.follow(own.e, &own.link) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            cuda.tp_watch.fail(@errorName(err)); // rank 0 and the others stop with it
            std.c._exit(1); // the stream may hold kernels waiting on a gone rank: a normal exit would join them
        };
        cuda.tp_watch.ended();
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        std.process.exit(0);
    }
    // Rank 0 alone reads these: TF_GLM_PROMPT_STEPS=0 passes prompts whole, TF_GLM_SHARED=0 runs windows alone.
    const off = struct {
        fn get(name: [*:0]const u8) bool {
            return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        }
    }.get;
    own.link.asleep = cuda.tp_watch.open(); // the other ranks begin asleep (follow): the first step wakes them
    own.lanes = .{ .x = .{ .e = own.e, .link = &own.link }, .whole = off("TF_GLM_PROMPT_STEPS"), .alone = off("TF_GLM_SHARED") };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = glm_lanes.max_rows,
        .stream_bytes = own.e.seqBytes() + (if (o.drafter != null) dflash.ring_bytes else 0) + if (own.e.images) |v| v.streamBytes() else @as(usize, 0), // each stream past the first makes its own sequence on every rank
        .free_streams = 1, // the first is the sequence the engine made
        .ctx = own,
        .deinit = close,
        .target = own.e,
        .peer_room = peer,
        .vision = own.e.images,
    };
}

// Each rank keeps a copy under rank 0's id, in the room rank 0 admitted by (the smallest over the ranks).

/// The rounded pages for a committed prefix, including the head's possible verify window.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.stateBytes(at);
}

/// The first legal chunk mark, with the same page rounding and possible head window used by cache admission.
pub fn snap_min_bytes(ptr: *anyopaque) u64 {
    return snap_bytes(ptr, snap_grid);
}

/// Reserve every spare page inside the admitted cache budget before requests can enter either rank.
pub fn snap_ready(ptr: *engine.Engine, budget: u64) !u64 {
    if (ptr.snap_setup) |s| try s.run(s.ptr, budget) else try ptr.prepareSnapshots(budget);
    return 0; // the store accounts both held and spare pages through its vtable
}

fn readySnapshots(ptr: *anyopaque, budget: u64) !void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const x: follow.Leader = .{ .e = own.e, .link = &own.link }; // its steps wake the ranks asleep on their lines
    try x.snapshots(budget);
    try x.idle(); // no stream yet: they sleep again until the first
}

pub fn snap_charged(ptr: *anyopaque, saved: *anyopaque) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    return e.snaps.get(k.id).?.lease.ids.items.len * snapshots.pages.page;
}

pub fn snap_spare(ptr: *anyopaque) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return if (e.snap_pages) |*p| p.spare() else 0;
}

pub fn snap_reuses(ptr: *anyopaque, at: u32) bool {
    return snap_spare(ptr) >= snap_bytes(ptr, at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): copy the state there under a new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pos() != at) return error.NotAtMark;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try e.keepState(), .at = at };
    return k;
}

/// A restore outside a prompt pass (the store's `begin`) would leave the other ranks behind, so it refuses.
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    e.dropState(k.id);
    e.gpa.destroy(k);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    cuda.tp_watch.ended(); // the other ranks leave on the end step, so their lines close on purpose from here
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    own.link.deinit();
    if (own.e.images) |v| v.deinit();
    own.e.deinit();
    own.group.close();
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadRoceCap => "TF_ROCE_MAX_KB: give an integer from 1 through 4194303 KiB, or unset it for 4096 KiB",
        error.NoRank => "GLM-5.3-Flash runs at TP over hosts: set TF_GLM_RANK and TF_GLM_WORLD on every host (WORLD 1: one GPU, with TF_GLM_LAYERS for a thin model)",
        error.NoCommId => "GLM-5.3-Flash's ranks meet over NCCL: with TF_GLM_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.RankMismatch => "the ranks' settings differ (each rank's log names the one): start them alike",
        error.TooManyRanks => std.fmt.comptimePrint(
            "GLM-5.3-Flash on CUDA runs on at most {d} ranks: set TF_GLM_WORLD to {d} or fewer",
            .{ ranks.max_world, ranks.max_world },
        ),
        error.NotModelOpt, error.UnsupportedQuant => "GLM-5.3-Flash on CUDA reads ModelOpt NVFP4 checkpoints (the log names what config.json has)",
        error.UnreadableFormat => "a projection is stored in a format GLM-5.3-Flash on CUDA does not read (the log names its module, recipe and tensors)",
        else => null,
    };
}

test "a place inside its world of at most 16 ranks serves; a larger world is refused by name" {
    try std.testing.expectEqual(Place{ .rank = 15, .world = 16 }, try checked(.{ .rank = 15, .world = 16 }));
    try std.testing.expectError(error.TooManyRanks, checked(.{ .rank = 0, .world = 17 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expect(explain(null, error.TooManyRanks) != null);
}

test "a checkpoint refused for its format, at the entry or by the loader, is explained" {
    for ([_]anyerror{ error.NotModelOpt, error.UnsupportedQuant, error.UnreadableFormat }) |e| try std.testing.expect(explain(null, e) != null);
}

test "capacity keeps a verify window past the context and a multiple of 64" {
    try std.testing.expectEqual(@as(usize, 8256), capacity(8192));
    try std.testing.expectEqual(@as(usize, 330560), capacity(330529));
    try std.testing.expect(capacity(330529) % 64 == 0);
}

// Enough words for every offered frame pair and image, even with a small text context.
pub fn linkCapacity(cap: usize, media: ?images_mod.Limits) !usize {
    const lim = media orelse return cap;
    const frames: usize = (@as(usize, lim.max_frames) + 1) / 2;
    const pairs = std.math.mul(usize, frames, lim.max_videos) catch return error.TooManyImages;
    const spans = std.math.add(usize, pairs, lim.max_images) catch return error.TooManyImages;
    const words = std.math.mul(usize, spans, 3) catch return error.TooManyImages;
    return @max(cap, words);
}
test "a small text context still sizes the link for every offered media span" {
    try std.testing.expectEqual(@as(usize, 792), try linkCapacity(64, .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 }));
}
