//! Gemma 4 dense (31B, 12B) on the lane host: windows of up to 16 rows, several streams a forward, external drafts.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const g = tf.gemma4;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");
const Resident = @import("resident.zig").Resident;

/// The context a stream gets without --context: attention's per-call scratch grows with it (pass --context for more).
const default_context = 32768;

/// --drafter loads z-lab's DFlash drafter (31B) or Google's MTP assistant (12B) in front of the engine.
pub const takes_drafter = true;

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io, // the image decoder's, on the HTTP thread
    model: *g.Model,
    metal: *g.backend.Metal,
    draft: ?*g.dflash.Drafter = null,
    assist: ?*g.assistant.Drafter = null,
    wrap: ?lanes.drafted.Drafted = null,
    warm: mtl.keepalive.Target = undefined, // the residency set's queue, for the lane host's idle ticker
    resident: ?Resident = null, // weights, scratches, the drafter's and the tower's buffers, wired between requests
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (--prompt-cache-gib 0: none)
    /// The image wrapper's tokens (--vision): begin, image, end, and each image's rows between the first and last.
    markers: [3]u32 = .{ 255999, 258880, 258882 },

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        if (h.wrap) |*w| w.deinit();
        if (h.draft) |d| d.deinit();
        if (h.assist) |d| d.deinit();
        h.config.deinit(h.gpa);
        if (h.resident) |*r| r.deinit();
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const h = start(a, gpa, io, o, problem) catch |err| switch (err) {
        error.Refused => return null,
        else => return err,
    };
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The host, or error.Refused once `problem` says why; either error path releases what was made before it.
fn start(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !*Host {
    const model = g.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Gemma 4 engine cannot load this checkpoint ({s}): it serves Gemma 4 31B and 12B in 4-bit groups of 64", .{@errorName(err)});
        return error.Refused;
    };
    errdefer model.deinit();
    const window: i64 = o.context orelse @min(default_context, @as(i64, @intCast(model.config.context)));
    if (window < 1 or window > model.config.context) {
        problem.* = "--context must fit the Gemma 4 checkpoint's positive context window";
        return error.Refused;
    }
    if (o.vision != null) model.loadVision(io, o.dir) catch |err| switch (err) {
        error.NoGemmaVision => {
            const names = if (model.config.geo.unified()) "vision_embedder. and embed_vision. (tools/zig/gemma4_12b_convert.py keeps them)" else "vision_tower. and embed_vision.";
            problem.* = try std.fmt.allocPrint(a, "--vision: this Gemma 4 {s} checkpoint has no image input ({s} tensors)", .{ model.config.geo.name, names });
            return error.Refused;
        },
        else => return err,
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.markers = .{ 255999, 258880, 258882 };
    h.metal = try g.backend.Metal.init(gpa, io, model, .{ .capacity = @intCast(window), .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    try h.metal.measure(); // the depth rule prices drafts against the window forward's cost by rows
    h.draft = null;
    h.assist = null;
    h.wrap = null;
    h.resident = null;
    const draft_dir = if (o.drafts) o.drafter else null;
    const assistant = if (draft_dir) |dir| g.assistant_weights.isAssistant(a, io, dir) catch false else false;
    if (draft_dir) |dir| if (assistant) {
        h.assist = g.assistant.Drafter.init(gpa, io, h.metal, dir) catch |err| {
            problem.* = if (err == error.AssistantForAnotherTarget)
                try std.fmt.allocPrint(a, "the Gemma 4 MTP assistant at {s} was trained on another target: its backbone width and key heads do not fit Gemma 4 {s} (google/gemma-4-12B-it-assistant drafts for the 12B only)", .{ dir, model.config.geo.name })
            else
                try std.fmt.allocPrint(a, "the native Gemma 4 engine cannot load the MTP assistant at {s} ({s}): it takes google/gemma-4-12B-it-assistant as released (bf16, no ordered embeddings)", .{ dir, @errorName(err) });
            return error.Refused;
        };
    } else {
        h.draft = g.dflash.Drafter.init(gpa, io, model, dir, .{ .capacity = @intCast(window) }) catch |err| {
            problem.* = if (err == error.DraftForAnotherTarget)
                try std.fmt.allocPrint(a, "the native Gemma 4 drafter cannot load {s}: its width or taps do not fit Gemma 4 {s} (DFlash drafts take z-lab/gemma-4-31B-it-DFlash on the 31B only)", .{ dir, model.config.geo.name })
            else
                try std.fmt.allocPrint(a, "the native Gemma 4 drafter cannot load {s} ({s}): it takes z-lab/gemma-4-31B-it-DFlash", .{ dir, @errorName(err) });
            return error.Refused;
        };
    };
    errdefer if (h.draft) |d| d.deinit();
    errdefer if (h.assist) |d| d.deinit();
    const drafter: ?lanes.drafter.Drafter = if (h.draft) |d| d.drafter() else if (h.assist) |d| d.drafter() else null;
    if (drafter) |d| h.wrap = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), d);
    errdefer if (h.wrap) |*w| w.deinit();
    const facts = if (h.wrap) |*w| w.facts(h.metal.facts()) else h.metal.facts();
    h.config = try lanes.Config.init(gpa, facts, g.state.max_rows, g.state.max_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.wrap) |*w| w.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const vis: ?api.Vision = if (o.vision) |v| .{ .ctx = h, .prepare = prepareImage, .image_token = h.markers[1], .markers = &h.markers, .wrap = .{ h.markers[0], h.markers[2] }, .part_type = "image", .image_tokens = v.image_tokens, .max_images = v.max_images, .explain = g.image.decoderProblem } else null;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = g.state.chunk_rows, .structures = true, .vision = vis });
    h.cache = null;
    try enableCache(h, a, o, @intCast(window), problem);
    errdefer if (h.cache) |*store| store.deinit();
    h.warm = .{ .queue = model.queue };
    holdResident(h) catch |err| std.log.warn("gemma4: no residency set ({s}); the first request after an idle second re-wires the weights", .{@errorName(err)});
    errdefer if (h.resident) |*r| r.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start(); // the host's lists are empty until its thread runs, so a failed start leaves nothing to free
    std.log.info("gemma4: Gemma 4 {s}, context {d} tokens, {d} streams{s}{s}", .{ model.config.geo.name, window, @max(o.lanes, 2), if (h.draft != null) ", DFlash drafts" else if (h.assist != null) ", MTP assistant drafts" else "", if (vis != null) ", images" else "" });
    return h;
}

/// The buffers that outlive a request, in one set the idle keepalive holds wired (a stream's caches come and go).
fn holdResident(h: *Host) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer); // the weights, the tower's among them
    try list.appendSlice(h.gpa, m.weights.owned.items);
    try list.appendSlice(h.gpa, &.{ m.inv_sliding, m.inv_global });
    try buffersOf(h.gpa, &list, &h.metal.scratch);
    try buffersOf(h.gpa, &list, &h.metal.prompt_scratch);
    if (h.draft) |d| {
        try list.appendSlice(h.gpa, d.w.owned.items);
        try buffersOf(h.gpa, &list, d);
    }
    if (h.assist) |d| {
        try list.appendSlice(h.gpa, d.w.owned.items);
        try buffersOf(h.gpa, &list, d);
    }
    if (m.vision) |v| try list.appendSlice(h.gpa, v.buffers());
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "gemma4") orelse return;
    h.warm = (&h.resident.?).target();
}

/// `x`'s buffer fields, plain, optional or in arrays: a scratch's or the drafter's working buffers.
fn buffersOf(gpa: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer), x: anytype) !void {
    const info = @typeInfo(@TypeOf(x.*)).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        const v = @field(x.*, name);
        switch (@typeInfo(T)) {
            .array => |a| if (a.child == mtl.Buffer) try list.appendSlice(gpa, &v),
            .optional => |o| if (o.child == mtl.Buffer) if (v) |b| try list.append(gpa, b),
            else => if (T == mtl.Buffer) try list.append(gpa, v),
        }
    }
}

/// The prompt cache's copies of a stream: the target's caches at a mark (backend.zig), and the drafter's context there.
const Snaps = struct {
    const M = g.backend.Metal;
    fn host(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h = host(ptr);
        return M.snapBytes(h.metal, at) + if (h.draft) |d| d.keptBytes(at) else 0;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const h = host(ptr);
        const saved = try M.snapSave(h.metal, owner, at);
        if (h.draft) |d| d.keepLane(@ptrCast(@alignCast(owner.?)), @ptrCast(@alignCast(saved))) catch |err| {
            M.snapDrop(h.metal, saved);
            return err;
        };
        return saved;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }
    fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        M.snapDrop(host(ptr).metal, saved);
    }
};

/// Prompt reuse in what 70% of RAM leaves beside the streams' caches (16 GiB by default), at planned chunk starts.
fn enableCache(h: *Host, a: std.mem.Allocator, o: api.Open, window: usize, problem: *[]const u8) !void {
    const later = @as(u64, @max(o.lanes, 2)) * g.state.cacheBytes(h.model.config.geo, window);
    const room = h.model.device.maxWorkingSet() -| h.model.device.allocated() -| (8 << 30) -| later;
    var why: []const u8 = "";
    const plan = cache_fit.budgetBeside(o.prompt_cache_gib, o.prompt_cache_over_cap, room, later, a, &why, "") catch |err| switch (err) {
        error.CacheOverCap => {
            problem.* = why;
            return error.Refused;
        },
        else => return err,
    };
    if (plan.budget_bytes == 0) return;
    // a prompt chunk's rows are its own arithmetic: keep and resume only where the request's chunks start
    const rules: api.prompt_cache.Rules = .{ .planned = true, .grid = g.state.chunk_rows, .min_prompt = 0 };
    h.cache = api.prompt_cache.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, rules, plan.budget_bytes);
    h.host.cache = &h.cache.?;
    h.host.info_.prompt_cache = true;
    h.host.info_.prompt_cache_plan = plan;
}

/// Info.vision's prepare (on the HTTP thread); the template writes <|image|> alone: the server adds boi and eoi.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try g.image.prepare(h.io, a, bytes, max_tokens, h.model.vision.?.layout());
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
