//! GLM-5.3 (glm_moe_dsa, EXLR8) behind the native server on a pair: rank 0 serves, rank 1 runs its commands.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const g = tf.glm_dsa;
const Allocator = std.mem.Allocator;

pub const Host = struct {
    gpa: Allocator,
    e: *g.engine.Engine,
    back: g.backend.Backend,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // rank 0's kept prompt states (rank 1 holds the same ids)
    warm: mtl.keepalive.Target,
    follower: ?std.Thread = null,

    pub fn engine(h: *Host) api.Engine {
        return h.host.engine();
    }
};

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.GreedyOnly => "the native GLM-5.3 engine decodes greedily only: send temperature 0",
        error.FollowsPeer => "this Mac is rank 1 of the pair: send requests to rank 0",
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        error.NoFreeSlot => "every stream slot is busy",
        else => null,
    };
}

/// Any chunk end may be a mark: every prompt row's arithmetic is the same whatever window it runs in.
const cache_rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = false, .min_prompt = 0 };

/// GLM-5.3 on the pair named by `speed_up` (rank 0 or 1's settings), `window` tokens a stream, `streams` streams.
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: u32, speed_up: []const u8, streams: u32, limit: u64, cache_gib: ?f64) !*Host {
    const e = try g.engine.Engine.load(gpa, io, dir, speed_up, null, true, window + g.engine.max_depth + 2 * g.gpu.max_rows, streams, limit);
    errdefer e.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.e = e;
    h.follower = null;
    h.back = try g.backend.Backend.init(gpa, e);
    errdefer h.back.deinit();
    if (e.rank() == 0) try h.back.time(io);
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), g.gpu.max_rows, g.gpu.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.wall = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "glm-dsa-zig", .lanes = streams, .context_window = window, .prefill_step = g.gpu.prompt_rows, .greedy_only = true });
    h.host.explain = .{ .text = words };
    h.cache = null;
    const used = e.device.allocated();
    const budget: u64 = @min(@as(u64, @intFromFloat((cache_gib orelse 8) * (1 << 30))), limit -| used);
    if (e.rank() == 0 and budget > 0) {
        const B = g.backend.Backend;
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = B.snapBytes, .save = B.snapSave, .restore = B.snapRestore, .drop = B.snapDrop } }, cache_rules, budget);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.warm = .{ .queue = e.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    errdefer h.host.stop();
    if (e.rank() == 1) h.follower = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, follow, .{h});
    std.log.info("GLM-5.3 (EXLR8) loaded in {d:.1} s, {d:.1} GB on this Mac, pair rank {d}, context {d} tokens, {d} streams", .{ e.load_seconds, @as(f64, @floatFromInt(e.m.bytes)) / 1e9, e.rank(), window, streams });
    if (h.back.timed) for (h.back.costs) |c| std.log.info("glm_dsa: a window of {d} rows {d:.1} ms", .{ c.width, c.ms });
    return h;
}

fn follow(h: *Host) void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    g.follow.follow(h.e) catch |err| std.log.err("glm_dsa rank 1: following rank 0 ended: {s}", .{@errorName(err)});
}

pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    if (h.cache) |*store| store.deinit();
    if (h.follower) |th| {
        h.e.pair.stopFollowing();
        th.join();
    }
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    h.e.deinit();
    h.gpa.destroy(h);
}
