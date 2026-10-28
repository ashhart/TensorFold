//! MiniMax-M3.1-Flash native CUDA at TP (TF_MINIMAX_RANK, TF_MINIMAX_WORLD, NCCL_COMM_ID): rank 0 serves.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const mm_lanes = @import("cuda_lanes.zig");
const vision_mod = @import("cuda_vision.zig");

/// config.json's name in the release and its published packs (the family is minimax_m3).
pub const model_type = "minimax_m3_vl";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
/// Kept prompt states are taken where a prompt chunk starts (the rows a pass leaves the caches at).
pub const snap_grid: u32 = prompt_rows;
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    vision: ?vision_mod.Limits = null, // --vision: the last rank's tower, every rank's image rows
};
/// An image takes at most 576 tokens; a request's images 4096 in all unless --vision-image-tokens says otherwise.
pub const default_image_tokens: u32 = 4096;
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    peer_room: ?Peer = null,
    target: *anyopaque, // the prompt cache keeps this rank group's states (snap_*)
    vision: ?*vision_mod.Images = null, // the server's image offer (rank 0's processor and limits)
};

/// The tower's bytes on this rank: the last rank holds it, sized from its tensors (every rank's staging is small).
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const p = place() catch return 0;
    return if (p.rank + 1 == p.world) vision_mod.Tower.bytesFor(a, io, dir) else 0;
}

/// This rank's place: TF_MINIMAX_RANK of TF_MINIMAX_WORLD (both, or neither for one rank).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = std.c.getenv("TF_MINIMAX_RANK");
    const w = std.c.getenv("TF_MINIMAX_WORLD");
    if (r == null and w == null) return .{ .rank = 0, .world = 1 };
    if (r == null or w == null) return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, std.mem.span(r.?), 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, std.mem.span(w.?), 10) catch return error.NoRank };
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    return p;
}

/// MiniMax-M3.1-Flash on CUDA is served only as it was run: on GB10 (sm_121), at two TP ranks.
pub fn admit(sm: u32, world: usize) error{ MiniMaxM3RequiresSm121, MiniMaxM3RequiresTwoRanks }!void {
    if (sm != 121) return error.MiniMaxM3RequiresSm121;
    if (world != 2) return error.MiniMaxM3RequiresTwoRanks;
}

/// The weights' bytes: the text shards (the tower's counted by visionBytes) and dspark/'s (drafts are on by default).
pub fn tensorWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    _ = vision;
    return try shardBytes(a, io, dir, true) + (draftBytes(a, io, dir) catch 0);
}

fn shardBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, skip_tower: bool) !u64 {
    const files = try core.checkpoint.shardFiles(a, io, dir);
    defer core.checkpoint.freeShardFiles(a, files);
    var total: u64 = 0;
    for (files) |f| {
        if (skip_tower and std.mem.indexOf(u8, std.fs.path.basename(f), "vision") != null) continue;
        total += (try std.Io.Dir.cwd().statFile(io, f, .{})).size;
    }
    return total;
}

fn draftBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const sub = try std.fs.path.join(a, &.{ dir, "dspark" });
    defer a.free(sub);
    return shardBytes(a, io, sub, false);
}

/// Bytes this rank's weights take: its share of the split axes; the embeddings, the routers and dspark/ counted whole.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    const p = place() catch return total;
    const c = cfg.Config.read(a, io, dir) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    var moe: u64 = 0;
    for (c.layer[0..c.layers]) |l| moe += @intFromBool(l.moe);
    const whole = @as(u64, c.vocab) * c.hidden * 2 + moe * (@as(u64, c.experts) * c.hidden + c.experts) * 4 + (draftBytes(a, io, dir) catch 0);
    if (total < whole) return total;
    return @intCast(@as(u128, total - whole) * f[0] / f[1] + whole);
}

/// Slots for `context` plus a verify window, whole index blocks.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + engine.window_rows, 128);
}

pub const window_rows: u32 = engine.window_rows;

const Owned = struct {
    gpa: std.mem.Allocator,
    lib: ?cuda.nccl.Library = null,
    comm: cuda.nccl.Comm = null,
    group: ?cuda.tp_group.Group = null, // lib and comm (owned above) with RoCE writes for two ranks' small gathers
    e: *engine.Engine,
    link: ?follow.Link = null,
    lanes: mm_lanes.Lanes,
    place: Place,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels;
    const p = try place();
    try admit(@intCast(try ctx.attribute(.compute_capability_major) * 10 + try ctx.attribute(.compute_capability_minor)), p.world);
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    if (p.world > 1) try cuda.tp_watch.start(io, p.rank, p.world);
    return openRank(gpa, io, ctx, dir, o, p) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
}

fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: Place) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = undefined, .lanes = undefined, .place = p };
    if (p.world > 1) {
        var lib = try cuda.nccl.Library.open();
        errdefer lib.close();
        var id: cuda.nccl.UniqueId = undefined;
        if (p.rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            cuda.tp_watch.sendId(&id.internal);
        } else id.internal = try cuda.tp_watch.takeId(io);
        var comm: cuda.nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
        own.lib = lib;
        own.comm = comm;
    }
    const cap = capacity(o.context);
    {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        try @import("cuda_exl3.zig").admit(gpa, text);
    }
    // the release's DSpark head drafts when drafts are on and dspark/ is there
    const has = blk: {
        const path = try std.fs.path.join(gpa, &.{ dir, "dspark", "config.json" });
        defer gpa.free(path);
        std.Io.Dir.cwd().access(io, path, .{}) catch break :blk false;
        break :blk true;
    };
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, p.rank, p.world, .{ .capacity = cap, .drafts = o.drafts, .drafter = o.drafts and has, .vision = o.vision });
    errdefer own.e.deinit();
    if (p.world > 1) {
        own.link = try follow.Link.init(gpa, ctx.d, &own.lib.?, own.comm, own.e.stream, @max(cap, engine.prompt_rows));
    }
    errdefer if (own.link) |*l| l.deinit();
    if (p.world > 1) {
        // small gathers by RoCE writes on two ranks, both or neither (TF_ROCE=0: NCCL; TF_ROCE_MAX_KB: the cap)
        own.group = .{ .lib = own.lib.?, .comm = own.comm, .rank = p.rank, .world = p.world };
        try own.group.?.openRoce(gpa, io, ctx.d, own.e.stream, try roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null));
        own.e.exchange = .{ .nccl = .{ .lib = &own.lib.?, .comm = own.comm, .group = &own.group.? } };
    }
    errdefer if (own.group) |*g| if (g.fast) |f| f.deinit();
    // rank 0 admits by the smallest room of the ranks
    var peer: ?Peer = null;
    if (p.world > 1) {
        const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
        var rooms = try cuda.DeviceBuffer.alloc(ctx.d, 8 * p.world);
        defer rooms.free();
        const at = rooms.ptr + p.rank * 8;
        try rooms.upload(p.rank * 8, std.mem.asBytes(&mine));
        try own.lib.?.check(own.lib.?.api.ncclAllGather(at, rooms.ptr, 8, .u8, own.comm, own.e.stream.handle), "ncclAllGather");
        try own.e.stream.synchronize();
        const all = try gpa.alloc(u64, p.world);
        defer gpa.free(all);
        try rooms.download(0, std.mem.sliceAsBytes(all));
        for (all[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
            peer = .{ .bytes = r, .rank = q };
        };
        try cuda.tp_watch.ready(io, own.e.checkpoint_digest);
    }
    if (p.rank != 0) {
        follow.follow(own.e, &own.link.?) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            cuda.tp_watch.fail(@errorName(err));
            std.c._exit(1);
        };
        cuda.tp_watch.ended();
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        release(own);
        std.process.exit(0);
    }
    if (own.link) |*l| l.asleep = cuda.tp_watch.open();
    own.lanes = .{ .x = .{ .e = own.e, .link = if (own.link) |*l| l else null } };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = mm_lanes.max_rows,
        .stream_bytes = own.e.seqBytes(),
        .free_streams = 1,
        .ctx = own,
        .deinit = close,
        .peer_room = peer,
        .target = own,
        .vision = own.e.images,
    };
}

/// A state kept at `at` (rounded as every rank's copy): its bytes on this rank, the largest's.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return own.e.stateBytes(at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): every rank copies the state there under one new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const own: *Owned = @ptrCast(@alignCast(ptr));
    if (own.e.pos() != at) return error.NotAtMark;
    const k = try own.gpa.create(engine.Kept);
    errdefer own.gpa.destroy(k);
    k.* = try own.lanes.x.save(at);
    return k;
}

/// A restore outside a prompt pass would leave the other ranks behind: states resume inside it (Lanes.resumeFrom).
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    own.lanes.x.drop(k.id);
    own.gpa.destroy(k);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    cuda.tp_watch.ended();
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    release(own);
}

/// TF_ROCE_MAX_KB in KiB (default 4 MiB, as GLM's): a rank's largest gather by RoCE writes.
fn roceBytes(text: ?[]const u8) !usize {
    const v = text orelse return 4096 << 10;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadRoceCap;
    if (n == 0 or n > std.math.maxInt(u32) / 1024) return error.BadRoceCap;
    return n * 1024;
}

fn release(own: *Owned) void {
    if (own.link) |*l| l.deinit();
    own.e.deinit();
    if (own.group) |*g| if (g.fast) |f| f.deinit();
    if (own.lib) |*lib| {
        _ = lib.api.ncclCommDestroy(own.comm);
        lib.close();
    }
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "MiniMax-M3.1-Flash runs at TP over two hosts: set TF_MINIMAX_RANK (0 or 1) and TF_MINIMAX_WORLD=2 on both",
        error.NoCommId => "MiniMax-M3.1-Flash's ranks meet over NCCL: set NCCL_COMM_ID to rank 0's address:port on both hosts",
        error.MiniMaxM3RequiresSm121 => "MiniMax-M3.1-Flash on CUDA is served on GB10 (sm_121) only so far, and this GPU is another",
        error.MiniMaxM3RequiresTwoRanks => "MiniMax-M3.1-Flash on CUDA is served at two TP ranks only so far: TF_MINIMAX_WORLD=2 on both hosts",
        error.NotExl3 => "MiniMax-M3.1-Flash on CUDA serves a published EXL3 pack (quantization_config quant_method \"exl3\"), e.g. Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw",
        error.UnsupportedExl3 => "this EXL3 pack is not one the CUDA kernels read: exllamav3 1.x on the mul1 codebook, every projection at a width the dense kernels build",
        error.UnexpectedMiniMaxTensor => "a tensor is not stored as MiniMax-M3.1-Flash's EXL3 packs store it (the log names it)",
        error.NoVision => "images need --vision on every rank (the last rank runs the tower)",
        else => null,
    };
}

test "refusals explain themselves; a capacity holds the window plus a verify window in whole blocks" {
    try std.testing.expect(explain(null, error.MiniMaxM3RequiresSm121) != null);
    try std.testing.expect(explain(null, error.MiniMaxM3RequiresTwoRanks) != null);
    try std.testing.expect(explain(null, error.NotExl3) != null);
    try std.testing.expectEqual(@as(usize, 32768 + 128), capacity(32768));
}

test "MiniMax-M3.1-Flash is admitted on GB10 (sm_121) at two TP ranks only" {
    try admit(121, 2);
    try std.testing.expectError(error.MiniMaxM3RequiresSm121, admit(120, 2));
    try std.testing.expectError(error.MiniMaxM3RequiresSm121, admit(90, 2));
    try std.testing.expectError(error.MiniMaxM3RequiresTwoRanks, admit(121, 1));
    try std.testing.expectError(error.MiniMaxM3RequiresTwoRanks, admit(121, 4));
}

test "the tower's scratch is load's buffers at the largest image" {
    // load's eleven scratch buffers at 2,304 patches (576 tokens, 2 x 2 patches each), summed by hand
    try std.testing.expectEqual(@as(u64, 183_785_472), vision_mod.Tower.scratch_bytes);
}
