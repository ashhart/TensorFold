//! Unlimited-OCR-2 on Metal served by the lane host: page images in, the greedy reply out, one request at a time.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fl = tf.uocr;
const lanes = tf.lanes;

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *fl.serve.Model,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    markers: [2]u32 = .{ fl.config.page_token, fl.config.image_token },

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    }
    if (o.lanes_fixed and o.lanes > 1) {
        problem.* = "Unlimited-OCR-2 serves one stream: serve it without --parallel";
        return null;
    }
    const vis_open = o.vision orelse {
        problem.* = "Unlimited-OCR-2 reads page images: serve it with --vision";
        return null;
    };
    const context: i64 = o.context orelse fl.config.max_context;
    if (context < 1 or context > fl.config.max_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: Unlimited-OCR-2 runs to {d} tokens", .{ context, fl.config.max_context });
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = fl.serve.Model.load(gpa, io, o.dir, @intCast(context)) catch |err| {
        problem.* = if (err == error.NoTensorUnits) "Unlimited-OCR-2 needs an M5-class GPU (tensor units)" else try std.fmt.allocPrint(a, "the native Unlimited-OCR-2 engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    std.log.info("unlimited-ocr: context {d} tokens, one request at a time", .{context});
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .model = model, .config = undefined, .clock = .{ .io = io }, .core = undefined, .host = undefined };
    h.config = try lanes.Config.init(gpa, model.facts(), 1, 0);
    errdefer h.config.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.config, model.backend(), h.clock.clock());
    errdefer h.core.deinit();
    // a page at most the widest gundam grid; without --vision-image-tokens, every allowed page may take it
    const image_tokens = if (vis_open.image_tokens_set) vis_open.image_tokens else vis_open.max_images * fl.serve.maxPageTokens();
    const vis: api.Vision = .{ .ctx = h, .prepare = prepareImage, .image_token = fl.config.image_token, .markers = &h.markers, .image_tokens = image_tokens, .max_images = vis_open.max_images };
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = 1, .context_window = @intCast(context), .vision = vis, .greedy_only = true, .reasoning = false });
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Info.vision's prepare (on the HTTP thread): a page's views, gundam tiles when they fit `max_tokens`.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try h.model.prepare(a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

test "Unlimited-OCR-2 refuses a named --parallel above one before reading the checkpoint" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const asked: api.Open = .{ .dir = "missing", .model_type = "unlimited-ocr", .lanes = 4, .lanes_fixed = true };
    try std.testing.expectEqual(null, try open(arena.allocator(), std.testing.allocator, std.testing.io, asked, &problem));
    try std.testing.expectEqualStrings("Unlimited-OCR-2 serves one stream: serve it without --parallel", problem);
    const one: api.Open = .{ .dir = "missing", .model_type = "unlimited-ocr", .lanes = 1, .lanes_fixed = true };
    try std.testing.expectEqual(null, try open(arena.allocator(), std.testing.allocator, std.testing.io, one, &problem));
    try std.testing.expectEqualStrings("Unlimited-OCR-2 reads page images: serve it with --vision", problem);
}
