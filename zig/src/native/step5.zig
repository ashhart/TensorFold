//! Step-5 served by the lane host: one-row rounds, contexts to 1M tokens as far as this Mac's cache budget reaches.
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
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    const text: ?[]const u8 = if (std.c.getenv("STEP5_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const scratch: u64 = 64 << 20; // a 16-row window's buffers, and the scores at 1M positions
    const budget = s5.budget.caches(ram, model.device.maxWorkingSet(), text, model.checkpoint.residentBytes() + scratch) catch {
        model.deinit();
        problem.* = "STEP5_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 75% of RAM";
        return null;
    };
    const fits = s5.budget.longest(&model.config, budget, model.config.context);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        model.deinit();
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's Step-5 caches fit {d} tokens under its load limit (75% of RAM, or STEP5_LOAD_LIMIT_GB)", .{ window, fits });
        return null;
    }
    std.log.info("step5: {d:.1} GiB for caches, context up to {d} tokens", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), window });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.metal = try s5.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = @max(o.lanes, 1), .budget = budget });
    errdefer h.metal.deinit();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), 1, 0);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = s5.state.prompt_rows, .structures = true, .logprobs = true });
    h.warm = .{ .queue = h.metal.model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
