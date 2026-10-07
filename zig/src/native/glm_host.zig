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

/// Streams whose caches fit beside the weights under the 70% load limit, at most `want` (`fixed`: all of them or none).
fn fit(eng: *const ge.Engine, want: u32, fixed: bool) !u32 {
    const per = glm.state.stateBytes(&eng.c, eng.s.cap, true) + 64 * 1024;
    const used = eng.w.bytes + eng.arena.bytes;
    const limit = ge.Engine.loadLimit();
    const room: u64 = if (limit > used) (limit - used) / per else 0;
    const n: u32 = @intCast(@min(@as(u64, @max(want, 1)), 1 + room));
    if (n < want and fixed) {
        std.log.err("glm: {d} streams of {d}-token caches pass this Mac's load limit; {d} fit (lower --parallel or --context)", .{ want, eng.s.cap, n });
        return error.OverMemoryLimit;
    }
    return n;
}

/// A GLM-5.3-Flash checkpoint served: `window` tokens of cache a stream, warmed first (the pair: `speed_up`).
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: u32, speed_up: ?[]const u8, streams: u32, fixed: bool) !*Host {
    const eng = try ge.Engine.loadWith(gpa, dir, window + 64, speed_up);
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
    h.slots = try glm.slots.Slots.init(gpa, eng, try fit(eng, streams, fixed));
    errdefer h.slots.deinit(gpa);
    const n: u32 = @intCast(h.slots.slots.len);
    h.back = .{ .gpa = gpa, .sl = &h.slots };
    errdefer h.back.deinit();
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), glm.state.max_rows, glm.state.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.wall = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "glm-zig", .lanes = n, .context_window = window });
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
    if (h.follower) |th| { // rank 1: its wait for rank 0's next command ends, then the thread
        h.eng.stopFollowing();
        th.join();
    }
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    h.slots.deinit(h.gpa);
    h.eng.deinit();
    h.gpa.destroy(h);
}
