//! Unlimited-OCR-2 on CUDA through the lane host: page images in, the greedy reply out, one request at a time.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const fl = @import("uocr_cuda");
const refusals = @import("cuda_refusal.zig");
const device = @import("cuda_device.zig");

pub const family: api.Family = .{ .model_type = "unlimited-ocr", .formats = &.{"bf16"} };

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    model: *fl.serve.Model,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    markers: [2]u32 = .{ fl.config.page_token, fl.config.image_token },

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.ctx.makeCurrent() catch {};
        h.model.deinit();
        h.ctx.deinit();
        h.driver.close();
        h.gpa.destroy(h);
    }

    /// The lane thread binds the context before its first round.
    fn bind(p: *anyopaque) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(p));
        try h.ctx.makeCurrent();
    }

    fn memory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }
};

/// The context a 16 GiB card holds beside the weights and scratch (its caches take 63 KiB a token).
pub const default_context = 65536;

/// The family as the CUDA server's per-family refusals read it: page images, no drafter, no kept prompt states.
const Admission = struct {
    pub const Options = struct { vision: ?api.VisionOpen };
    pub const drafter_refusal = "--drafter: Unlimited-OCR-2 has no drafter; serve it without --drafter";
    pub const keeps_no_prompts = true;
};

/// A flag the family has no use for, refused by name rather than ignored; null when `o` may open.
fn flagRefusal(o: api.Open) ?[]const u8 {
    if (refusals.family(Admission, o)) |why| return why;
    if (o.precision != null) return "--precision: Unlimited-OCR-2 reads its bf16 release only; serve it without --precision";
    if (o.record != null) return "--record: Unlimited-OCR-2 gives no rows to record on CUDA; serve it without --record";
    if (o.recall) return "--recall: Unlimited-OCR-2 keeps no draft record; serve it without --recall";
    if (o.segments != null) return "--segments: Unlimited-OCR-2 on CUDA does not split its prompt into segments; serve it without --segments";
    if (o.decode_share != null) return "--decode-share: Unlimited-OCR-2 serves one request at a time; serve it without --decode-share";
    if (o.keep != null) return "--checkpoint-slots: Unlimited-OCR-2 keeps no prompt states; serve it without --checkpoint-slots";
    return null;
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try o.slideRefusal(a, "CUDA")) |why| {
        problem.* = why;
        return null;
    }
    if (flagRefusal(o)) |why| {
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
    const context: i64 = o.context orelse default_context;
    if (context < 1 or context > fl.config.max_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: Unlimited-OCR-2 runs to {d} tokens", .{ context, fl.config.max_context });
        return null;
    }
    const ordinal = try device.setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const h = start(a, gpa, io, o, vis_open, context, ordinal, problem) catch |err| switch (err) {
        error.Refused => return null,
        else => return err,
    };
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The host, or error.Refused once `problem` says why; either error path releases what was made before it.
fn start(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, vis_open: api.VisionOpen, context: i64, ordinal: u32, problem: *[]const u8) !*Host {
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .driver = undefined, .ctx = undefined, .model = undefined, .config = undefined, .clock = .{ .io = io }, .core = undefined, .host = undefined };
    h.driver = cuda.Driver.open() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(err)});
        return error.Refused;
    };
    errdefer h.driver.close();
    const count: u32 = @intCast(try h.driver.deviceCount());
    if (ordinal >= count) {
        const named = if (o.device != null) "--device " else "TF_CUDA_DEVICE=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: this machine has {d} CUDA device{s} (0 to {d})", .{ named, ordinal, count, if (count == 1) "" else "s", @max(count, 1) - 1 });
        return error.Refused;
    }
    h.ctx = try cuda.Context.init(&h.driver, @intCast(ordinal));
    errdefer h.ctx.deinit();
    try h.ctx.makeCurrent();
    h.model = fl.serve.Model.load(gpa, io, &h.driver, o.dir, @intCast(context)) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Unlimited-OCR-2 engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return error.Refused;
    };
    errdefer h.model.deinit();
    std.log.info("unlimited-ocr: context {d} tokens, one request at a time, {d:.2} GiB of weights", .{ context, @as(f64, @floatFromInt(h.model.weights.bytes)) / (1 << 30) });
    h.config = try lanes.Config.init(gpa, h.model.facts(), 1, 0);
    errdefer h.config.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.config, h.model.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = 1, .context_window = @intCast(context), .vision = offer(h, vis_open), .greedy_only = true, .reasoning = false });
    h.host.on_thread = .{ .ctx = h, .run = Host.bind };
    h.host.memory = .{ .read = Host.memory };
    try h.host.start();
    return h;
}

/// The host's vision offer: --vision's image tokens (else its images' widest pages) and --vision-image-share's value.
fn offer(h: *Host, vis_open: api.VisionOpen) api.Vision {
    const image_tokens = if (vis_open.image_tokens_set) vis_open.image_tokens else vis_open.max_images * fl.serve.maxPageTokens();
    return .{ .ctx = h, .prepare = prepareImage, .image_token = fl.config.image_token, .markers = &h.markers, .image_tokens = image_tokens, .max_images = vis_open.max_images, .image_share = vis_open.image_share };
}

/// Info.vision's prepare (on the HTTP thread, host work only): a page's views, gundam tiles when they fit.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try h.model.prepare(a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

test "Unlimited-OCR-2 on CUDA refuses a named --parallel above one before reading the checkpoint" {
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

test "Unlimited-OCR-2 on CUDA refuses a --context past its window or below one, naming both, before reading the checkpoint" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    for ([_]i64{ fl.config.max_context + 1, 1 << 20, 0, -1 }) |context| {
        const asked: api.Open = .{ .dir = "missing", .model_type = "unlimited-ocr", .vision = .{}, .context = context };
        try std.testing.expectEqual(null, try open(a, std.testing.allocator, std.testing.io, asked, &problem));
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "--context {d}: Unlimited-OCR-2 runs to 131072 tokens", .{context}), problem);
    }
    // the window and one below pass the check and reach the load: no checkpoint here, or no CUDA driver
    for ([_]i64{ fl.config.max_context, fl.config.max_context - 1, 1 }) |context| {
        const asked: api.Open = .{ .dir = "missing", .model_type = "unlimited-ocr", .vision = .{}, .context = context };
        try std.testing.expectEqual(null, try open(a, std.testing.allocator, std.testing.io, asked, &problem));
        try std.testing.expect(std.mem.startsWith(u8, problem, "the native Unlimited-OCR-2 engine cannot load this checkpoint (") or
            std.mem.startsWith(u8, problem, "no CUDA driver ("));
    }
}

test "Unlimited-OCR-2 on CUDA offers --vision-image-share's value, as the Metal host does" {
    var h: Host = undefined;
    for ([_]api.ImageShare{ .request, .fixed }) |share| {
        const v = offer(&h, .{ .max_images = 3, .image_share = share });
        try std.testing.expectEqual(share, v.image_share);
        try std.testing.expectEqual(@as(u32, 3), v.max_images);
        try std.testing.expectEqual(3 * fl.serve.maxPageTokens(), v.image_tokens);
    }
    try std.testing.expectEqual(@as(u32, 700), offer(&h, .{ .image_tokens = 700, .image_tokens_set = true }).image_tokens);
}

test "Unlimited-OCR-2 on CUDA refuses each flag it has no use for by name, before reading the checkpoint or the device" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const base: api.Open = .{ .dir = "missing", .model_type = "unlimited-ocr", .vision = .{} };
    const Case = struct { flag: []const u8, o: api.Open };
    var cases: [14]Case = undefined;
    cases[0] = .{ .flag = "--drafter", .o = base };
    cases[0].o.drafter = "drafter";
    cases[1] = .{ .flag = "--kv-cache fp8", .o = base };
    cases[1].o.kv8 = true;
    cases[2] = .{ .flag = "--prompt-precision int8", .o = base };
    cases[2].o.prompt_int8 = true;
    cases[3] = .{ .flag = "--loops", .o = base };
    cases[3].o.loops = 2;
    cases[4] = .{ .flag = "--prompt-cache-gib", .o = base };
    cases[4].o.prompt_cache_gib = 4;
    cases[5] = .{ .flag = "--learn-conversations", .o = base };
    cases[5].o.learn = "states";
    cases[5].o.learn_conversations = true;
    cases[6] = .{ .flag = "--spill-gib", .o = base };
    cases[6].o.learn = "states";
    cases[6].o.learn_conversations = true;
    cases[6].o.spill = .evicted;
    cases[7] = .{ .flag = "--precision", .o = base };
    cases[7].o.precision = .checkpoint;
    cases[8] = .{ .flag = "--record", .o = base };
    cases[8].o.record = "rows";
    cases[9] = .{ .flag = "--recall", .o = base };
    cases[9].o.recall = true;
    cases[10] = .{ .flag = "--segments", .o = base };
    cases[10].o.segments = 2;
    cases[11] = .{ .flag = "--decode-share", .o = base };
    cases[11].o.decode_share = 0.5;
    cases[12] = .{ .flag = "--checkpoint-slots", .o = base };
    cases[12].o.keep = 3;
    cases[13] = .{ .flag = "this CUDA family has no tensor-parallel ranks", .o = base };
    cases[13].o.tp = 2;
    for (cases) |c| {
        var problem: []const u8 = "";
        try std.testing.expectEqual(null, try open(arena.allocator(), std.testing.allocator, std.testing.io, c.o, &problem));
        std.testing.expectStringStartsWith(problem, c.flag) catch |err| {
            std.debug.print("asked {s}, refused: {s}\n", .{ c.flag, problem });
            return err;
        };
    }
    var problem: []const u8 = "";
    try std.testing.expectEqual(null, try open(arena.allocator(), std.testing.allocator, std.testing.io, cases[0].o, &problem));
    try std.testing.expectEqualStrings(Admission.drafter_refusal, problem);
}

test "Unlimited-OCR-2 on CUDA reads --device and refuses a device the machine lacks before loading" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const asked: api.Open = .{ .dir = "missing", .model_type = "unlimited-ocr", .vision = .{}, .device = 4096 };
    try std.testing.expectEqual(null, try open(arena.allocator(), std.testing.allocator, std.testing.io, asked, &problem));
    try std.testing.expect(std.mem.startsWith(u8, problem, "--device 4096: this machine has ") or std.mem.startsWith(u8, problem, "no CUDA driver ("));
}
