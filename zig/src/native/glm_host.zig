//! GLM-5.3-Flash behind the native server on the lane core: concurrent requests share rounds; rank 1 replays rank 0's.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const glm = tf.glm;
const ge = glm.engine;
const Allocator = std.mem.Allocator;

/// The warm-up reply's draft depth: every rank of a pair pays a process's first-round costs at the same step.
const DEPTH = 2;

pub const Host = struct {
    gpa: Allocator,
    eng: *ge.Engine,
    slots: glm.slots.Slots,
    back: glm.backend.Backend,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // rank 0's kept prompt states (rank 1 holds copies by id)
    learned: ?api.prompt_imprint.Imprint = null, // --learn: shared states on disk (rank 1 keeps its halves there)
    warm: mtl.keepalive.Target,
    follower: ?std.Thread = null, // speed-up mode's rank 1: the thread replaying rank 0's slot commands

    pub fn engine(h: *Host) api.Engine {
        return h.host.engine();
    }
};

/// The words for a request the engine refuses.
fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.GreedyOnly => "the native GLM-5.3-Flash engine decodes greedily only: send temperature 0",
        error.FollowsPeer => "speed-up mode: this Mac runs rank 0's requests; send requests to rank 0",
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        else => null,
    };
}

/// Streams whose caches fit beside the weights under the load limit, at most `want` (`fixed`: all of them or none).
fn fit(eng: *const ge.Engine, want: u32, fixed: bool) !u32 {
    const per = glm.state.stateBytes(&eng.c, eng.s.cap, true) + 64 * 1024;
    const used = eng.w.bytes + eng.arena.bytes;
    const limit = eng.limit.bytes;
    const room: u64 = if (limit > used) (limit - used) / per else 0;
    const n: u32 = @intCast(@min(@as(u64, @max(want, 1)), 1 + room));
    if (n < want and fixed) {
        std.log.err("glm: {d} streams of {d}-token caches pass this Mac's {d:.1} GB load limit ({s}); {d} fit (lower --parallel or --context)", .{ want, eng.s.cap, @as(f64, @floatFromInt(limit)) / 1e9, eng.limit.source(), n });
        return error.OverMemoryLimit;
    }
    return n;
}

/// Marks sit on planned chunk starts that every pass cuts anyway, so a prompt under 4,096 tokens keeps them too.
const cache_rules: api.prompt_cache.Rules = .{ .lookahead = 1, .planned = true, .min_prompt = 0 };

/// Kept prompt states' room: --prompt-cache-gib, else 16 GiB, inside what the load limit read at load leaves.
fn cacheBudget(eng: *const ge.Engine, gib: ?f64) u64 {
    const want: u64 = @intFromFloat((gib orelse 16) * (1 << 30));
    const used = eng.w.bytes + eng.arena.bytes;
    const limit = eng.limit.bytes;
    return if (limit > used) @min(want, limit - used) else 0;
}

/// Learned states under `root` by checkpoint, split, chip, OS build, kernel sources and a probe's prompt-pass bits.
fn learnedStates(gpa: Allocator, eng: *const ge.Engine, sl: *glm.slots.Slots, root: []const u8, cap: u64) !api.prompt_imprint.Imprint {
    var h = std.hash.Wyhash.init(0x62);
    const probe = try sl.probe(); // both Macs of a pair run it in step: its prompt pass exchanges with the peer
    for ([_]u64{ eng.model_hash, eng.c.tp, eng.chunk_rows, @intFromBool(eng.ep != null), eng.k.source_hash, probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(eng.device.name()));
    const m = try api.prompt_imprint.Imprint.open(gpa, root, h.final(), cap);
    std.log.info("glm: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ m.dir, m.metas.items.len, m.total() >> 20, cap >> 20 });
    return m;
}

/// A GLM-5.3-Flash checkpoint served: `window` tokens of cache a stream, warmed first (the pair: `speed_up`).
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: u32, speed_up: ?[]const u8, streams: u32, fixed: bool, cache_gib: ?f64, learn: ?[]const u8, learn_cap: u64, learn_floor: u64) !*Host {
    const eng = try ge.Engine.loadWith(gpa, dir, window + 64, speed_up, learn != null);
    errdefer eng.deinit();
    var toks: [96]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
    var dummy: u8 = 0;
    _ = try eng.generate(&toks, 16, &.{}, DEPTH, .{ .ctx = &dummy, .prefilled = ge.Quiet.prefilled, .tokens = ge.Quiet.tokens, .cancelled = ge.Quiet.cancelled });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.eng = eng;
    h.follower = null;
    h.learned = null;
    h.slots = try glm.slots.Slots.init(gpa, eng, try fit(eng, streams, fixed));
    errdefer h.slots.deinit(gpa);
    const n: u32 = @intCast(h.slots.slots.len);
    h.back = .{ .gpa = gpa, .sl = &h.slots };
    errdefer h.back.deinit();
    if (eng.ep != null) h.back.costs = try glm.timing.measure(&h.slots, gpa, io); // both Macs in step, as the warm-up
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), glm.state.max_rows, glm.state.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.wall = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "glm-zig", .lanes = n, .context_window = window, .prefill_step = eng.chunk_rows, .greedy_only = true });
    h.cache = null;
    const budget = cacheBudget(eng, cache_gib);
    if (!eng.followsPeer() and budget > 0) {
        const B = glm.backend.Backend;
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = B.snapBytes, .save = B.snapSave, .restore = B.snapRestore, .drop = B.snapDrop, .write = B.snapWrite, .read = B.snapRead, .forget = B.snapForget, .forget_checked = B.snapForgetChecked, .reclaim = B.snapReclaim, .peer_other_used = B.peerOtherUsed, .peer_other_next = B.peerOtherNext, .peer_other_bytes = B.peerOtherBytes, .peer_other_remove = B.peerOtherRemove, .peer_need = B.peerNeed, .peer_reclaim = B.peerReclaim, .peer_reserve = B.peerReserve, .peer_finish = B.peerFinish } }, cache_rules, budget);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    if (learn) |root| {
        h.learned = try learnedStates(gpa, eng, &h.slots, root, learn_cap);
        h.learned.?.admission.floor = learn_floor;
        h.slots.disk.admission.floor = learn_floor;
        h.slots.disk.cap = learn_cap;
        h.slots.learned = h.learned.?.dir;
        if (h.cache) |*store| store.imprint = &h.learned.?;
    }
    errdefer if (h.learned) |*m| m.deinit();
    h.host.explain = .{ .text = words };
    h.warm = eng.keepalive_target;
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start(); // the host thread first: once the follower runs, nothing after it can fail
    errdefer h.host.stop();
    if (eng.followsPeer()) h.follower = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, follow, .{h});
    const role = if (eng.ep == null) "" else if (eng.followsPeer()) ", speed-up rank 1" else ", speed-up rank 0";
    std.log.info("GLM-5.3-Flash loaded in {d:.1} s ({d:.1} GB of weights{s}), context {d} tokens, {d} streams", .{ eng.load_seconds, @as(f64, @floatFromInt(eng.w.bytes)) / 1e9, role, window, n });
    return h;
}

fn follow(h: *Host) void {
    glm.mirror.follow(h.eng, &h.slots) catch |err| std.log.err("speed-up mode: following rank 0 ended: {s}", .{@errorName(err)});
}

pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    if (h.cache) |*store| store.deinit();
    if (h.follower) |th| { // rank 1: its wait for rank 0's next command ends, then the thread
        h.eng.stopFollowing();
        th.join();
    }
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    h.slots.deinit(h.gpa);
    if (h.learned) |*m| m.deinit();
    h.eng.deinit();
    h.gpa.destroy(h);
}

test "a 3.5k-token prompt keeps its shared system cut, a planned chunk start" {
    const Stub = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 0;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
    };
    var unused: u8 = 0;
    var store = api.prompt_cache.Store.init(std.testing.allocator, .{ .ptr = &unused, .vtable = &.{ .bytes = Stub.bytes, .save = Stub.save, .restore = Stub.restore, .drop = Stub.drop } }, cache_rules, 1 << 30);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const prompt = try arena.allocator().alloc(u32, 3547);
    for (prompt, 0..) |*t, i| t.* = @intCast(i);
    // a 3,523-token system prompt: the server cuts at 3,522 and names 1,475, 3,011 and 3,523 as shared cuts
    const marks = try store.marks(arena.allocator(), prompt, 0, 3540, &.{ 1475, 3011, 3523 }, &.{3522}, &.{});
    try std.testing.expectEqualSlices(u32, &.{3522}, marks);
}
