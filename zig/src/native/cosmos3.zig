//! Cosmos3-Edge's reasoner served by the lane host: text, and images (--vision) through its SigLIP2 tower.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.cosmos3;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");

/// Image rows one request holds when --vision-image-tokens is not given: a 1280 x 960 picture's.
const default_image_tokens = 1200;

const Host = struct {
    resident: ?mtl.resident.Resident = null,
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *q.Model,
    metal: *q.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: a zero budget
    markers: [3]u32 = .{ 20, 19, 21 }, // vision_start, image_pad, vision_end

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

const cannot = "the native Cosmos3-Edge engine cannot load this checkpoint ({s})";

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const config = q.config.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    const window = q.state.window(config.context, o.context) orelse {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside Cosmos3-Edge's 1 to {d} tokens", .{ o.context.?, @min(config.context, q.state.max_context) });
        return null;
    };
    const model = q.Model.loadWithResidency(gpa, io, o.dir, true) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .model = model, .metal = undefined, .config = undefined, .clock = .{ .io = io }, .core = undefined, .host = undefined };
    const image_tokens: usize = if (o.vision) |v| (if (v.image_tokens_set) v.image_tokens else default_image_tokens) else 0;
    const chunk = q.state.prompt_rows;
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = window, .chunk = chunk, .streams = @max(o.lanes, 1), .image_tokens = image_tokens });
    errdefer h.metal.deinit();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    var vis: ?api.Vision = null;
    if (o.vision) |v| {
        h.markers = .{ config.vision_start, config.image_token, config.vision_end };
        vis = .{ .ctx = h, .prepare = prepareImage, .image_token = config.image_token, .markers = &h.markers, .part_type = "image", .image_tokens = @intCast(image_tokens), .max_images = v.max_images, .explain = q.image.decoderProblem };
    }
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = chunk, .structures = true, .logprobs = true, .vision = vis });
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
    try mtl.resident.streams.hold(h, io, "cosmos3");
    errdefer mtl.resident.streams.drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    std.log.info("cosmos3: Cosmos3-Edge reasoner loaded ({d}-bit), context {d}, image rows a request {d}", .{ config.bits, window, image_tokens });
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Info.vision's prepare: an image's bytes to the tower's patches, on the HTTP thread.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try q.image.prepare(h.io, a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
