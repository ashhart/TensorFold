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
    cache: ?api.prompt_cache.Store = null, // kept prompt states, in what the streams' caches leave of the budget
    cache_cap: u64 = 0, // --prompt-cache-gib's bytes (the budget when unset)
    markers: [6]u32 = .{ 128000, 128001, 128002, 128003, 128004, 128005 }, // the image wrapper's tokens (--vision)

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
    const c = &model.config; // the decode window's and the prompt chunk's buffers, scores sized for the longest context
    const scratch: u64 = s5.state.scratchBytes(c, c.context, s5.state.window_rows) + s5.state.scratchBytes(c, c.context, s5.state.prompt_rows);
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
    try h.metal.warm();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), s5.state.window_rows, s5.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    var vis: ?api.Vision = null;
    if (o.vision) |v| {
        if (model.vision == null) {
            problem.* = "--vision: this Step-5 checkpoint has no image tower (tools/step5_fp8_to_mlx.py --only-vision adds it)";
            return null;
        }
        h.markers = .{ 128000, 128001, 128002, 128003, 128004, 128005 };
        vis = .{ .ctx = h, .prepare = prepareImage, .image_token = h.markers[1], .markers = &h.markers, .part_type = "image", .image_tokens = v.image_tokens, .max_images = v.max_images };
    }
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = s5.state.prompt_rows, .structures = true, .logprobs = true, .vision = vis });
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else budget;
    if (h.cache_cap > 0) {
        const M = s5.backend.Metal;
        // the MTP head keys row at-1 with token at; prompt chunks change arithmetic with their rows
        const rules: api.prompt_cache.Rules = .{ .lookahead = @intFromBool(model.config.mtp > 0), .planned = true, .grid = s5.state.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop } }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    h.warm = .{ .queue = h.metal.model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Metal.room's fit: the kept states' budget is what the streams leave, at most the cap; the least valuable go first.
fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = &(h.cache orelse return);
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

/// Info.vision's prepare: an image's bytes to the tower's pieces (families/step5/image.zig), on the HTTP thread.
fn prepareImage(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const p = try s5.image.prepare(a, bytes, .{}, max_tokens);
    const pieces = try a.alloc(api.Piece, p.pieces.len);
    for (p.pieces, pieces) |x, *y| y.* = .{ .pixels = x.pixels, .gh = x.gh, .gw = x.gw, .tokens = x.tokens, .before = x.before, .after = x.after };
    const last = p.pieces[p.pieces.len - 1];
    return .{ .pixels = last.pixels, .gh = last.gh, .gw = last.gw, .tokens = p.tokens, .hash = p.hash, .pieces = pieces };
}
