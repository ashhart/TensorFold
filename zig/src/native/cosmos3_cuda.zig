//! Cosmos3-Edge's reasoner on CUDA served by the lane host: text, and images (--vision) through its SigLIP2 tower.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const q = @import("cosmos3_family");

/// Image rows one request holds when --vision-image-tokens is not given: a 1280 x 960 picture's.
const default_image_tokens = 1200;

pub const family: api.Family = .{ .model_type = "cosmos3_edge", .formats = &.{"mlx-q8g64"} };

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *q.Model,
    cuda: *q.backend.Cuda,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    markers: [3]u32 = .{ 20, 19, 21 }, // vision_start, image_pad, vision_end

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.cuda.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

const cannot = "the native Cosmos3-Edge engine cannot load this checkpoint ({s})";

/// The served window: --context, else the checkpoint's, capped at the engine's; null when --context is outside it.
fn window(checkpoint: usize, requested: ?i64) ?usize {
    const most = @min(checkpoint, q.model.max_context);
    const w = requested orelse return most;
    return if (w >= 1 and w <= most) @as(usize, @intCast(w)) else null;
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const config = q.config.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    const win = window(config.context, o.context) orelse {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside Cosmos3-Edge's 1 to {d} tokens", .{ o.context.?, @min(config.context, q.model.max_context) });
        return null;
    };
    const image_tokens: usize = if (o.vision) |v| (if (v.image_tokens_set) v.image_tokens else default_image_tokens) else 0;
    const chunk = q.model.prompt_rows;
    const model = q.Model.load(gpa, io, o.dir, &config, chunk, image_tokens) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .model = model, .cuda = undefined, .config = undefined, .clock = .{ .io = io }, .core = undefined, .host = undefined };
    h.cuda = try q.backend.Cuda.init(gpa, model, .{ .capacity = win, .chunk = chunk, .streams = @max(o.lanes, 1), .image_tokens = image_tokens });
    errdefer h.cuda.deinit();
    h.config = try lanes.Config.init(gpa, h.cuda.facts(), q.model.window_rows, q.model.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.config, h.cuda.backend(), h.clock.clock());
    errdefer h.core.deinit();
    var vis: ?api.Vision = null;
    if (o.vision) |v| {
        h.markers = .{ config.vision_start, config.image_token, config.vision_end };
        vis = .{ .ctx = h, .prepare = prepareImage, .image_token = config.image_token, .markers = &h.markers, .part_type = "image", .image_tokens = @intCast(image_tokens), .max_images = v.max_images, .explain = q.image.decoderProblem };
    }
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(win), .prefill_step = chunk, .structures = true, .logprobs = true, .vision = vis });
    try h.host.start();
    std.log.info("cosmos3: Cosmos3-Edge reasoner loaded on CUDA (8-bit), context {d}, image rows a request {d}", .{ win, image_tokens });
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Info.vision's prepare: an image's bytes to the tower's patches, on the HTTP thread.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try q.image.prepare(h.io, a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
