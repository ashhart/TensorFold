//! LFM2-MoE served by the existing lane host and scheduler.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");

const Host = struct {
    gpa: std.mem.Allocator,
    model: *q.Model,
    metal: *q.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: a zero budget

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

const cannot = "the native LFM2 engine cannot load this checkpoint ({s})";

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const config = q.config.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    const most = @min(config.context, q.state.max_context);
    const window = q.state.window(config.context, o.context) orelse {
        const outside = "--context {d} is outside LFM2.5-24B-A2B's 1 to {d} tokens";
        problem.* = try std.fmt.allocPrint(a, outside, .{ o.context.?, most });
        return null;
    };
    if (o.context == null and config.context > most) std.log.info(
        "lfm2: serving {d} of the checkpoint's {d} positions, all its attention holds",
        .{ most, config.context },
    );
    const model = q.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    const chunk = 2048;
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = window, .chunk = chunk, .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = chunk, .structures = true });
    h.cache = null;
    var why: []const u8 = "";
    const dev = model.device;
    const plan = cache_fit.budget(
        o.prompt_cache_gib,
        o.prompt_cache_over_cap,
        dev.maxWorkingSet() -| dev.allocated() -| (8 << 30),
        a,
        &why,
        "",
    ) catch |e| {
        if (e == error.CacheOverCap) problem.* = why;
        return e;
    };
    h.host.info_.prompt_cache_plan = plan;
    if (plan.budget_bytes > 0) {
        const S = q.backend.Snaps;
        h.cache = api.prompt_cache.Store.init(gpa, .{
            .ptr = h.metal,
            .vtable = &.{ .bytes = S.bytes, .save = S.save, .restore = S.restore, .drop = S.drop },
        }, .{ .chunk = chunk }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.warm = .{ .queue = h.metal.model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
