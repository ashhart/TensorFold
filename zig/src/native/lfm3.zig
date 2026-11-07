//! LFM3 served by the existing lane host and scheduler.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm3;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");
const metal_admit = @import("metal_admit.zig");

const Host = struct {
    resident: ?mtl.resident.Resident = null,
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
        mtl.resident.streams.drop(h);
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

const cannot = "the native LFM3 engine cannot load this checkpoint ({s})";
const prompt_chunk = 128;

fn cacheRules() api.prompt_cache.Rules {
    return .{ .chunk = prompt_chunk, .repeat_min = 2 * prompt_chunk };
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var config = q.config.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    defer config.deinit();
    if (config.kind != .text) {
        problem.* = "LFM3-VL inference requires the stacked vision contribution";
        return null;
    }
    if (o.vision != null and config.kind == .text) {
        problem.* = "LFM3-1.7B is text-only; --vision requires an LFM3-VL checkpoint";
        return null;
    }
    const most = @min(config.context, q.state.max_context);
    const window = q.state.window(config.context, o.context orelse @as(i64, @intCast(@min(most, 8192)))) orelse {
        const outside = "--context {d} is outside LFM3's 1 to {d} tokens";
        problem.* = try std.fmt.allocPrint(a, outside, .{ o.context.?, most });
        return null;
    };
    if (o.context == null and config.context > most) std.log.info(
        "lfm3: serving {d} of the checkpoint's {d} positions, all its attention holds",
        .{ most, config.context },
    );
    const model = q.Model.loadWithResidency(gpa, io, o.dir, false) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.resident = null;
    const chunk = prompt_chunk;
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = window, .chunk = chunk, .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    const per_stream = q.state.cacheBytes(&model.config.layer_types, window);
    const served = try metal_admit.streams(a, "lfm3", per_stream, metal_admit.room(model.device), o, window, problem) orelse {
        h.metal.deinit();
        model.deinit();
        gpa.destroy(h);
        return null;
    };
    h.metal.options.streams = @max(served, 2);
    h.config = try lanes.Config.init(gpa, h.metal.facts(), q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "lfm3", .lanes = served, .context_window = @intCast(window), .prefill_step = chunk, .structures = true, .vision = null });
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
        }, cacheRules(), plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    try mtl.resident.streams.hold(h, io, "lfm3");
    errdefer mtl.resident.streams.drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

test "LFM3 repeats keep a resumable raw prefix without caching an unrelated prompt" {
    const a = std.testing.allocator;
    var store = api.prompt_cache.Store.init(a, undefined, cacheRules(), 102 << 20);
    defer store.deinit();
    var tokens: [910]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(i + 1);
    const first = try store.marks(a, &tokens, 0, 0, &.{}, &.{}, &.{});
    defer a.free(first);
    try std.testing.expectEqual(@as(usize, 0), first.len);
    store.repeat_log.note(a, store.rules.repeat_min, &tokens);
    const repeated = try store.marks(a, &tokens, 0, 0, &.{}, &.{}, &.{});
    defer a.free(repeated);
    try std.testing.expectEqualSlices(u32, &.{896}, repeated);
    tokens[0] += 1;
    const changed = try store.marks(a, &tokens, 0, 0, &.{}, &.{}, &.{});
    defer a.free(changed);
    try std.testing.expectEqual(@as(usize, 0), changed.len);
}
