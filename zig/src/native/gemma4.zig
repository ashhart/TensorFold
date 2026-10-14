//! Gemma 4 dense (the 31B) served by the lane host: windows of up to 16 rows, several streams a forward, copy drafts.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const g = tf.gemma4;
const lanes = tf.lanes;

/// The context a stream gets without --context: attention's per-call scratch grows with it (pass --context for more).
const default_context = 32768;

const Host = struct {
    gpa: std.mem.Allocator,
    model: *g.Model,
    metal: *g.backend.Metal,
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
    const model = g.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Gemma 4 engine cannot load this checkpoint ({s}): it serves the dense 31B in 4-bit groups of 64", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const window: i64 = o.context orelse @min(default_context, @as(i64, @intCast(model.config.context)));
    if (window < 1 or window > model.config.context) {
        model.deinit();
        problem.* = "--context must fit the Gemma 4 checkpoint's positive context window";
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.metal = try g.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), g.state.max_rows, g.state.max_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = g.state.chunk_rows, .structures = true });
    h.warm = .{ .queue = h.metal.model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    std.log.info("gemma4: the dense 31B, context {d} tokens, {d} streams", .{ window, @max(o.lanes, 2) });
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
