//! Step-5 served by the lane host: one-row rounds, contexts the dense full-attention path covers (4,096 tokens).
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const s5 = tf.step5;
const lanes = tf.lanes;

const Host = struct {
    gpa: std.mem.Allocator,
    model: *s5.Model,
    metal: *s5.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = s5.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Step-5 engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const limit: i64 = model.config.dense_limit;
    const window: i64 = o.context orelse limit;
    if (window < 1 or window > limit) {
        model.deinit();
        problem.* = try std.fmt.allocPrint(a, "--context must be 1 to {d}: past that Step-5's sparse indexer picks keys, which this engine does not run yet", .{limit});
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.metal = try s5.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = @max(o.lanes, 1) });
    errdefer h.metal.deinit();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), 1, 0);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = 16, .structures = true, .logprobs = true });
    h.warm = .{ .queue = h.metal.model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
