//! Inkling 2 native CUDA at TP=2 (TF_INKLING2_RANK, TF_INKLING2_WORLD, NCCL_COMM_ID): rank 0 serves, rank 1 follows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const cfg = @import("shape.zig");
const admission = @import("admit.zig");
const shard = @import("shard.zig");
const engine = @import("engine.zig");
const follow = @import("follow.zig");
const ink_lanes = @import("lanes.zig");
const pictures = @import("pictures.zig");
const tower = @import("vision.zig");

pub const model_type = "inkling2";
pub const formats: []const []const u8 = &.{"exl3"};
/// The model's native window: no rope, so 1M is served as trained.
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
/// Kept prompt states are taken where a prompt chunk starts.
pub const snap_grid: u32 = prompt_rows;
pub const window_rows: u32 = engine.window_rows;
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    vision: ?pictures.Limits = null, // --vision on every rank: rank 0's tower, every rank's image rows
};
/// A request's visual tokens in all unless --vision-image-tokens says otherwise.
pub const default_image_tokens: u32 = pictures.default_image_tokens;
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 0, // every slot's caches are made by `reserve`, after admission
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    peer_room: ?Peer = null,
    target: *anyopaque, // the prompt cache keeps this rank group's states (snap_*)
    vision: ?*pictures.Offer = null, // the server's image offer (rank 0's processor and limits)
};

/// The tower's bytes on this rank: rank 0 holds it, its tensors plus scratch for a picture at the cap.
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, _: u32, _: u32) !u64 {
    const p = admission.place() catch return 0;
    if (p.rank != 0) return 0;
    const c = try cfg.Config.read(a, io, dir);
    return tower.Tower.bytesFor(a, io, dir, c.vision, pictures.picture_tokens);
}

/// The text shards' bytes (vision.safetensors is counted beside them; a drafter's mtp.safetensors is not loaded).
pub fn tensorWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    _ = vision;
    const files = try core.checkpoint.shardFiles(a, io, dir);
    defer core.checkpoint.freeShardFiles(a, files);
    var total: u64 = 0;
    for (files) |f| {
        const name = std.fs.path.basename(f);
        if (std.mem.indexOf(u8, name, "vision") != null or std.mem.indexOf(u8, name, "mtp") != null) continue;
        total += (try std.Io.Dir.cwd().statFile(io, f, .{})).size;
    }
    return total;
}

/// Bytes this rank's weights take: half the split tensors, the embedding and the routers whole.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64, drafts: bool, _: u32, _: u32) u64 {
    _ = drafts;
    const p = admission.place() catch return total;
    if (p.world == 1) return total;
    const c = cfg.Config.read(a, io, dir) catch return total;
    const whole = @as(u64, c.vocab) * c.hidden * 2 + @as(u64, c.layers) * (@as(u64, c.experts) * c.hidden * 2 + c.experts * 4);
    if (total < whole) return total;
    return (total - whole) / p.world + whole;
}

/// Positions a stream holds: `context` plus a window, in whole 64-key tiles.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + engine.window_rows, 64);
}

const Owned = struct {
    gpa: std.mem.Allocator,
    lib: ?cuda.nccl.Library = null,
    comm: cuda.nccl.Comm = null,
    group: ?cuda.tp_group.Group = null, // lib and comm (owned above) with RoCE writes for two ranks' small gathers
    e: *engine.Engine,
    link: ?follow.Link = null,
    lanes: ink_lanes.Lanes,
    offer: ?*pictures.Offer = null,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels;
    try @import("sys").environment.checkScope(.inkling2, @import("sys").environment.Process{});
    const p = try admission.place();
    try admission.check(ctx, p.world);
    if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    try cuda.tp_watch.start(io, p.rank, p.world);
    return openRank(gpa, io, ctx, dir, o, p) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
}

fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: admission.Place) !Loaded {
    var load_cache = core.checkpoint.cache.Load.begin(io, true);
    defer load_cache.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = undefined, .lanes = undefined };
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
    errdefer _ = own.lib.?.api.ncclCommDestroy(own.comm);
    const cap = capacity(o.context);
    const lim: ?pictures.Limits = if (o.vision) |v| .{ .image_tokens = v.image_tokens, .max_images = @min(v.max_images, 64) } else null;
    // every shard byte, hashed beside the weights' load; the ranks compare the digests at ready
    var content: ?*core.checkpoint_identity.Content = try startIdentity(gpa, io, dir);
    errdefer if (content) |x| x.abandon();
    own.e = try engine.Engine.init(gpa, io, ctx.d, dir, p.rank, p.world, .{ .capacity = cap, .vision = lim });
    errdefer own.e.deinit();
    own.link = try follow.Link.init(gpa, ctx.d, &own.lib.?, own.comm, own.e.stream, @max(cap, engine.prompt_rows));
    errdefer own.link.?.deinit();
    // gathers up to a prompt call's partial by RoCE writes, both ranks or neither (TF_ROCE=0: NCCL); plan word on NCCL
    own.group = .{ .lib = own.lib.?, .comm = own.comm, .rank = p.rank, .world = p.world };
    try own.group.?.openRoce(gpa, io, ctx.d, own.e.stream, own.e.roceBytes());
    errdefer if (own.group.?.fast) |f| f.deinit();
    own.e.exchange = .{ .nccl = .{ .lib = &own.lib.?, .comm = own.comm, .group = &own.group.? } };
    const hashing = content.?;
    content = null;
    own.e.checkpoint_digest = readyDigest(try hashing.finish(), &own.e.c, cap, lim);
    try load_cache.finishAfter(.{ io, own.e.checkpoint_digest }, cuda.tp_watch.ready);
    // rank 0 admits by the smallest room of the ranks
    var peer: ?Peer = null;
    {
        const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
        var rooms = try cuda.DeviceBuffer.alloc(ctx.d, 8 * p.world);
        defer rooms.free();
        try rooms.upload(p.rank * 8, std.mem.asBytes(&mine));
        try own.lib.?.check(own.lib.?.api.ncclAllGather(rooms.ptr + p.rank * 8, rooms.ptr, 8, .u8, own.comm, own.e.stream.handle), "ncclAllGather");
        try own.e.stream.synchronize();
        var all: [2]u64 = undefined;
        try rooms.download(0, std.mem.sliceAsBytes(all[0..p.world]));
        for (all[1..p.world], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
            peer = .{ .bytes = r, .rank = q };
        };
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
    own.link.?.asleep = cuda.tp_watch.open();
    own.lanes = .{ .x = .{ .e = own.e, .link = &own.link.? } };
    if (lim) |l| own.offer = try pictures.Offer.init(gpa, io, l);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = ink_lanes.max_rows,
        .stream_bytes = own.e.seqBytes(),
        .ctx = own,
        .deinit = close,
        .reserve = reserveFn,
        .peer_room = peer,
        .target = own,
        .vision = own.offer,
    };
}

/// The host's admitted stream count: every rank's slots made at once.
fn reserveFn(ptr: *anyopaque, streams: u32) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    try own.lanes.x.reserve(streams);
}

/// A state kept at `at`: its bytes on this rank (rings and conv states; the global rows stay in their slot).
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    _ = at;
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return own.e.stateBytes();
}

/// The prompt pass stands at `at` (a mark at a chunk start): every rank copies the state there under one new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const own: *Owned = @ptrCast(@alignCast(ptr));
    if (own.e.pos() != at) return error.NotAtMark;
    const k = try own.gpa.create(follow.Kept);
    errdefer own.gpa.destroy(k);
    k.* = try own.lanes.x.save(at);
    return k;
}

/// A restore outside a prompt pass would leave the other rank behind: states resume inside it (Lanes.resumeFrom).
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const k: *follow.Kept = @ptrCast(@alignCast(saved));
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

fn release(own: *Owned) void {
    if (own.offer) |v| v.deinit();
    if (own.link) |*l| l.deinit();
    own.e.deinit();
    if (own.group) |*g| if (g.fast) |f| f.deinit();
    if (own.lib) |*lib| {
        _ = lib.api.ncclCommDestroy(own.comm);
        lib.close();
    }
    own.gpa.destroy(own);
}

/// Every shard byte of the pack, hashed on threads: both ranks read whole files, so their digests must match.
pub fn startIdentity(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*core.checkpoint_identity.Content {
    return core.checkpoint_identity.Content.start(gpa, io, dir, false);
}

/// What the ranks compare at ready: every shard byte, the config the engine runs and the stream shape.
pub fn readyDigest(shards: [32]u8, c: *const cfg.Config, cap: usize, lim: ?pictures.Limits) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(&shards);
    c.hash(&h);
    const shape = [3]u64{ cap, if (lim) |l| l.image_tokens else 0, if (lim) |l| l.max_images else 0 };
    h.update(std.mem.sliceAsBytes(&shape));
    return h.finalResult();
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return admission.explain(err) orelse switch (err) {
        error.NoCommId => "Inkling 2's ranks meet over NCCL: set NCCL_COMM_ID to rank 0's address:port on both hosts",
        error.UnexpectedInkling2Tensor => "a tensor is not stored as the Inkling 2 EXL3 pack stores it (the log names it)",
        error.NotExl3 => "Inkling 2 on CUDA reads EXL3 packs: config.json needs a quantization_config with quant_method exl3",
        error.UnsupportedExl3 => "this EXL3 pack is not from exllamav3 1.x on the mul1 codebook, the only form validated here",
        error.CheckpointBytesDiffer => "the two ranks' packs or settings differ: copy the same pack to both hosts and start both alike",
        error.NoVision => "images need --vision on both ranks (rank 0 runs the tower from the pack's vision.safetensors)",
        error.ImagePlaceholders => "the prompt's image placeholders do not match its pictures",
        error.ImageTooLarge => pictures.Offer.explain(error.ImageTooLarge).?,
        else => null,
    };
}

test "native: refusals explain themselves; a stream's capacity holds the context and a window in whole tiles" {
    try std.testing.expect(explain(null, error.Inkling2RequiresSm121) != null);
    try std.testing.expect(explain(null, error.Inkling2RequiresTwoRanks) != null);
    try std.testing.expect(explain(null, error.NoCommId) != null);
    try std.testing.expectEqual(@as(usize, 1048576 + 64), capacity(1048576));
    try std.testing.expectEqual(@as(usize, 131072 + 64), capacity(131072));
}

test "native: the ranks compare every shard byte, so one changed byte in a pack changes the digest" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const head = "{\"t\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}";
    var bytes: [8 + head.len + 64]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], head.len, .little);
    @memcpy(bytes[8..][0..head.len], head);
    for (bytes[8 + head.len ..], 0..) |*b, i| b.* = @truncate(i);
    const subs = [_][]const u8{ "a", "b", "c" };
    var got: [3][32]u8 = undefined;
    for (subs, &got, 0..) |sub, *g, k| {
        if (k == 2) bytes[bytes.len - 1] ^= 1; // the last tensor byte of the third copy
        try tmp.dir.createDirPath(io, sub);
        var d = try tmp.dir.openDir(io, sub, .{});
        defer d.close(io);
        try d.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
        try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
        const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, sub });
        defer gpa.free(path);
        g.* = try (try startIdentity(gpa, io, path)).finish();
    }
    try std.testing.expectEqual(got[0], got[1]);
    try std.testing.expect(!std.mem.eql(u8, &got[0], &got[2]));
    try std.testing.expect(!std.mem.eql(u8, &got[0], &@as([32]u8, @splat(0))));
    try std.testing.expect(explain(null, error.CheckpointBytesDiffer) != null);
}

test "native: the ready digest binds the config the engine runs, so a config-only change differs" {
    const gpa = std.testing.allocator;
    const text = @embedFile("fixtures/config.json");
    const eps = "\"rms_norm_eps\": 1e-06";
    const at = std.mem.indexOf(u8, text, eps).?; // the text model's, the first in the file
    const other = try std.mem.concat(gpa, u8, &.{ text[0..at], "\"rms_norm_eps\": 1e-05", text[at + eps.len ..] });
    defer gpa.free(other);
    const a = try cfg.parse(gpa, text);
    const b = try cfg.parse(gpa, other);
    try std.testing.expect(a.eps != b.eps);
    const shards: [32]u8 = @splat(7);
    const base = readyDigest(shards, &a, capacity(131072), null);
    try std.testing.expectEqual(base, readyDigest(shards, &try cfg.parse(gpa, text), capacity(131072), null));
    try std.testing.expect(!std.mem.eql(u8, &base, &readyDigest(shards, &b, capacity(131072), null)));
    try std.testing.expect(!std.mem.eql(u8, &base, &readyDigest(shards, &a, capacity(1048576), null)));
    try std.testing.expect(!std.mem.eql(u8, &base, &readyDigest(shards, &a, capacity(131072), .{ .image_tokens = 1024, .max_images = 8 })));
    try std.testing.expect(!std.mem.eql(u8, &base, &readyDigest(@splat(8), &a, capacity(131072), null)));
}
