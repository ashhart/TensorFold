//! Inkling 2 Mini (inkling2_loop) on one Metal device: resident weights, shared lanes, and the bundled DSpark head.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fl = tf.inkling2_metal;
const lanes = tf.lanes;
const target = @import("inkling2_backend.zig");
const admit = @import("metal_admit.zig");
const cache_fit = @import("cache_fit.zig");
const Resident = @import("resident.zig").Resident;
const Allocator = std.mem.Allocator;

/// config.json's model_type of the Mini release (the loop model); Small's `inkling2` is not served on Metal.
pub const model_type = "inkling2_loop";

const Host = struct {
    gpa: Allocator,
    io: std.Io,
    model: *fl.Model,
    target: *target.Target,
    vision: ?*fl.vision.Tower = null,
    draft: ?*fl.draft.Adapter = null,
    wrapped: ?lanes.drafted.Drafted = null,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: none
    resident: ?Resident = null,
    warm: mtl.keepalive.Target = undefined,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    markers: [1]u32,

    fn dropResidency(h: *Host) void {
        h.model.setResident(null);
        h.target.setResident(null);
        if (h.draft) |d| d.setResident(null);
        if (h.vision) |v| v.setResident(null);
        if (h.resident) |*r| r.deinit();
        h.resident = null;
    }

    fn close(ptr: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        if (h.wrapped) |*w| w.deinit();
        h.dropResidency();
        h.target.deinit();
        if (h.draft) |d| d.deinit();
        if (h.vision) |v| v.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }

    fn encode(ptr: *anyopaque, image: lanes.stream.Image) !mtl.Buffer {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const tower = h.vision orelse return error.InklingVisionDisabled;
        if (image.tokens != @as(u64, image.gh) * image.gw) return error.BadInklingImageGrid;
        return tower.encodeInput(image.pixels, image.gh, image.gw);
    }

    fn prepare(ptr: *anyopaque, a: Allocator, bytes: []const u8, cap: u32) !api.PreparedImage {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const prepared = try fl.vision.prepare(h.io, a, bytes, cap);
        return .{ .pixels = prepared.pixels, .gh = prepared.gh, .gw = prepared.gw, .tokens = prepared.tokens, .hash = prepared.hash };
    }
};

fn refusal(a: Allocator, o: api.Open) !?[]const u8 {
    if (!std.mem.eql(u8, o.model_type, model_type))
        return try std.fmt.allocPrint(a, "the Inkling 2 Metal engine serves Inkling 2 Mini (model_type {s}) only, not {s}", .{ model_type, o.model_type });
    if (try o.slideRefusal(a, "Macs")) |why| return why;
    if (o.ssd_experts_gib != null) return "Inkling 2 Mini keeps every expert resident; --ssd-experts is unsupported";
    if (o.speed_up != null) return "Inkling 2 Mini serves one Metal device; --speed-up is unsupported";
    if (o.tp != 1 or o.rank != 0) return "Inkling 2's Metal engine serves one device; tensor parallelism is unsupported";
    if (o.precision != null or o.device != null or o.segments != null) return "--precision, --device and --segments are CUDA options, not Inkling 2 Metal options";
    if (o.mtp_confidence != null) return "Inkling 2's DSpark head does not implement --mtp-confidence";
    if (o.mtp_drafts) |n| if (n == 0) return "use --no-drafts to disable Inkling 2's DSpark head, or a positive --mtp-drafts depth";
    if (o.drafter != null) return "Inkling 2 loads its bundled mtp.safetensors DSpark head; --drafter is unsupported";
    if (o.keep) |n| if (n != 0) return "Inkling 2 keeps prompt states with --prompt-cache-gib; --checkpoint-slots must be 0";
    if (o.policy.len != 0 or o.loops != null) return "--policy and --loops are unsupported for Inkling 2 on Metal";
    if (o.kv8) return "Inkling 2's Metal attention stores BF16 KV; --kv-cache fp8 is unsupported";
    if (o.record != null or o.recall or o.learn != null or o.learn_conversations or o.spill != .off)
        return "Inkling 2's Metal engine keeps prompt states in memory only; --record, --recall, --learn, --learn-conversations and --spill-gib are unsupported";
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0) return "--prompt-cache-gib must be a finite GiB count of at least 0";
    if (o.context) |n| if (n < 1 or n >= std.math.maxInt(u32)) return "--context must be a positive Inkling 2 window below 2^32 - 1";
    if (o.lanes == 0 or o.lanes > 64) return "Inkling 2 serves 1 to 64 parallel Metal streams";
    return null;
}

/// The bytes the loader maps: core/checkpoint.zig's shard list (index, else model*.safetensors) plus mtp when drafted.
fn checkpointFileBytes(a: Allocator, io: std.Io, dir: []const u8, drafted: bool) u64 {
    const files = tf.checkpoint_host.shardFiles(a, io, dir) catch return 0;
    defer tf.checkpoint_host.freeShardFiles(a, files);
    var total: u64 = 0;
    for (files) |path| total +|= (std.Io.Dir.cwd().statFile(io, path, .{}) catch continue).size;
    if (!drafted) return total;
    const head = std.fs.path.join(a, &.{ dir, "mtp.safetensors" }) catch return total;
    defer a.free(head);
    return total +| (std.Io.Dir.cwd().statFile(io, head, .{}) catch return total).size;
}

/// Kept-state bytes: --prompt-cache-gib, else half the room.
fn cacheBudget(a: Allocator, o: api.Open, left: u64, problem: *[]const u8) !?u64 {
    const gib = o.prompt_cache_gib orelse return left / 2;
    if (gib == 0) return 0;
    // The conversion must be safe even when an explicit over-cap budget is allowed.
    if (gib >= 0x1p34) {
        problem.* = "--prompt-cache-gib is too large to represent in bytes";
        return null;
    }
    const bytes: u64 = @intFromFloat(gib * (1 << 30));
    if (bytes > left and !o.prompt_cache_over_cap) {
        problem.* = try std.fmt.allocPrint(a, "--prompt-cache-gib {d:.2} is more than the {d:.2} GiB left under the load limit after the weights and streams", .{ gib, @as(f64, @floatFromInt(left)) / (1 << 30) });
        return null;
    }
    return bytes;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try refusal(a, o)) |why| {
        problem.* = why;
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const probe = try mtl.Device.init();
    const room = admit.room(probe);
    probe.deinit();
    const weights = checkpointFileBytes(a, io, o.dir, o.drafts);
    if (weights >= room) {
        problem.* = try std.fmt.allocPrint(a, "Inkling 2 Mini's {d:.2} GiB checkpoint leaves no room under the {d:.2} GiB load limit; free memory", .{ @as(f64, @floatFromInt(weights)) / (1 << 30), @as(f64, @floatFromInt(room)) / (1 << 30) });
        return null;
    }
    const requested: u32 = @intCast(o.context orelse 8192);
    const drafted = o.drafts;
    const model = fl.Model.load(gpa, io, o.dir, .{ .capacity = requested, .streams = o.lanes, .rows = target.max_rows, .drafts = drafted, .vision = o.vision != null, .metal4 = try @import("sys").environment.read(bool, "TF_METAL4", true) }) catch |err| {
        problem.* = switch (err) {
            error.InvalidOptions => try std.fmt.allocPrint(a, "--context {d} is outside this Inkling 2 Mini checkpoint's window", .{requested}),
            error.NoDraftHead => "this Inkling 2 Mini checkpoint has no DSpark head; serve with --no-drafts",
            else => try std.fmt.allocPrint(a, "Inkling 2 cannot load this Mini checkpoint ({s}); it needs the original thinkingmachines/Inkling-2-Mini release", .{@errorName(err)}),
        };
        return null;
    };
    var transfer = false;
    defer if (!transfer) model.deinit();
    const window = requested;
    if (window > model.config.text.max_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this Inkling 2 checkpoint's {d}-token window", .{ window, model.config.text.max_context });
        return null;
    }
    const h = try gpa.create(Host);
    var started = false;
    defer if (!started) gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.vision = null;
    h.draft = null;
    h.wrapped = null;
    h.cache = null;
    h.resident = null;
    h.markers = .{model.config.image_token};
    defer if (!started) {
        if (h.draft) |d| d.deinit();
        if (h.vision) |v| v.deinit();
    };
    if (o.vision) |v| {
        h.vision = fl.vision.Tower.loadFromCheckpoint(gpa, model.device, model.queue, &model.config, .{ .max_tokens = v.image_tokens }, &model.checkpoint) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "Inkling 2 cannot load its native image tower ({s})", .{@errorName(err)});
            return null;
        };
    }
    if (drafted) {
        h.draft = fl.draft.Adapter.init(gpa, io, o.dir, model.device, model.queue, &model.config, .{ .embed = model.weights.embed, .embed_norm = model.weights.embed_norm, .unembed = model.weights.unembed }, .{ .streams = o.lanes, .capacity = window, .depth = o.mtp_drafts orelse 0 }) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "Inkling 2 cannot load its bundled DSpark head ({s}); check mtp.safetensors", .{@errorName(err)});
            return null;
        };
    }
    const image_bytes: u64 = if (o.vision) |v| @as(u64, v.max_images) * v.image_tokens * (@as(u64, model.config.text.hidden) * 2 + 1200 * @sizeOf(f32)) else 0;
    const stream_bytes = target.Target.laneBytes(model, drafted) + (if (h.draft) |d| d.laneBytes() else 0) + image_bytes;
    const sampling_bytes = target.Target.samplingBytes(model);
    // Some shared buffers are backed by mapped storage: do not depend solely on device or footprint sampling.
    const tower_bytes = if (h.vision) |v| v.bytes() else 0;
    const draft_bytes = if (h.draft) |d| d.bytes() else 0;
    const fixed_bytes = model.fixedBytes() + tower_bytes + draft_bytes;
    const demand: fl.admit.Demand = .{ .fixed = fixed_bytes, .sampling = sampling_bytes, .stream = stream_bytes };
    const left = fl.admit.left(room, admit.room(model.device), demand);
    const count = switch (fl.admit.verdict(left, demand, o.lanes, o.lanes_fixed)) {
        .serve => |n| n,
        else => |v| {
            problem.* = try fl.admit.refusal(a, v, demand, left, o.lanes, window);
            return null;
        },
    };
    if (count < o.lanes) std.log.info("inkling2 mini: {d} streams at a {d}-token window fit the {d:.2} GiB left under the load limit ({d:.2} GiB a stream), not {d}", .{ count, window, @as(f64, @floatFromInt(left)) / (1 << 30), @as(f64, @floatFromInt(stream_bytes)) / (1 << 30), o.lanes });
    // The adapter's lanes are lazy; cap them to the count that actually fits.
    if (h.draft) |d| d.opts.streams = count;
    const cache_left = left -| (stream_bytes * count) -| cache_fit.MARGIN;
    const budget = try cacheBudget(a, o, cache_left, problem) orelse return null;
    h.target = try target.Target.init(gpa, model, .{ .capacity = window, .streams = count, .rows = target.max_rows, .drafts = drafted, .images = if (h.vision != null) .{ .ctx = h, .encode = Host.encode } else null });
    h.target.head = h.draft;
    errdefer h.target.deinit();
    if (drafted) try h.target.measureCosts(io);
    var backend = h.target.backend();
    var facts = h.target.facts();
    if (h.draft) |d| {
        h.wrapped = try lanes.drafted.Drafted.init(gpa, backend, d.drafter());
        h.wrapped.?.first_rows = true;
        backend = h.wrapped.?.backend();
        facts = h.wrapped.?.facts(facts);
    }
    errdefer if (h.wrapped) |*w| w.deinit();
    h.config = try lanes.Config.init(gpa, facts, target.window_rows, target.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, backend, h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "inkling2-mini-metal", .lanes = count, .context_window = window, .prefill_step = target.max_rows, .structures = true, .logprobs = true, .plain_only = h.draft == null, .decode_share = o.decode_share });
    if (h.host.pressure) |*g| g.streamsFit(); // the streams admitted above fit the load limit: only critical holds them to one
    if (h.draft != null) h.host.info_.draft_sources = .{ .drafter = .{ .name = "DSpark", .kind = "DSpark" } };
    if (o.vision) |v| h.host.info_.vision = .{ .ctx = h, .prepare = Host.prepare, .image_token = model.config.image_token, .markers = &h.markers, .part_type = "image", .image_tokens = v.image_tokens, .max_images = v.max_images };
    if (budget > 0) {
        // prompt rows are cut on the chunk grid: a kept state is resumed only where a pass of this width stood
        const rules: api.prompt_cache.Rules = .{ .planned = true, .grid = target.max_rows, .tail = true };
        h.cache = api.prompt_cache.Store.init(gpa, h.target.snapshots(), rules, budget);
        h.host.cache = &h.cache.?;
        h.host.advertiseCache(&h.cache.?);
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.resident = try Resident.hold(gpa, io, model.device, model.queue, &.{}, "Inkling 2");
    errdefer h.dropResidency();
    if (h.resident) |*r| {
        model.setResident(r.follow());
        h.target.setResident(r.follow());
        if (h.vision) |v| v.setResident(r.follow());
        if (h.draft) |d| d.setResident(r.follow());
        r.addNow(&.{});
        h.warm = r.target();
    } else h.warm = .{ .queue = model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    std.log.info("inkling2 mini: {d} streams, {d}-token context, {d} logical layers over {d} physical; weights {d:.2} GiB resident, dense+scratch {d:.2} GiB; DSpark {s}, images {s}, prompt cache {d:.2} GiB; demand {d:.2} GiB ({d:.2} GiB fixed, {d} x {d:.2} GiB a stream)", .{ count, window, model.config.text.executionLayers(), model.config.text.layers, @as(f64, @floatFromInt(model.weightBytes())) / (1 << 30), @as(f64, @floatFromInt(model.fixedBytes())) / (1 << 30), if (h.draft != null) "on" else "off", if (h.vision != null) "on" else "off", @as(f64, @floatFromInt(budget)) / (1 << 30), @as(f64, @floatFromInt(demand.of(count))) / (1 << 30), @as(f64, @floatFromInt(fixed_bytes + sampling_bytes)) / (1 << 30), count, @as(f64, @floatFromInt(stream_bytes)) / (1 << 30) });
    try h.host.start();
    started = true;
    transfer = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

fn scratchDir(tmp: anytype, a: Allocator) ![]const u8 {
    return tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
}

test "inkling2 admission serves Mini only and never Small or EXL3" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    for ([_][]const u8{ "inkling2", "qwen3_5", "inkling2_text" }) |other| {
        problem = "";
        try std.testing.expect((try open(a, a, std.testing.io, .{ .dir = "missing", .model_type = other }, &problem)) == null);
        try std.testing.expect(std.mem.indexOf(u8, problem, "Inkling 2 Mini (model_type inkling2_loop) only") != null);
    }
    try std.testing.expect((try refusal(a, .{ .dir = "d", .model_type = model_type })) == null);
}

test "inkling2 Mini refuses options it cannot honour before any load" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base: api.Open = .{ .dir = "d", .model_type = model_type };
    var o = base;
    o.ssd_experts_gib = 4;
    try std.testing.expect(std.mem.indexOf(u8, (try refusal(a, o)).?, "every expert resident") != null);
    o = base;
    o.mtp_drafts = 0;
    try std.testing.expect((try refusal(a, o)) != null);
    o = base;
    o.kv8 = true;
    try std.testing.expect((try refusal(a, o)) != null);
    o = base;
    o.spill = .evicted;
    try std.testing.expect((try refusal(a, o)) != null);
    o = base;
    o.slide = true;
    try std.testing.expectEqualStrings("--slide: inkling2_loop checkpoints do not learn on Macs yet", (try refusal(a, o)).?);
    o = base;
    o.prompt_cache_gib = -1;
    try std.testing.expect((try refusal(a, o)) != null);
    o = base;
    o.loops = 2;
    try std.testing.expectEqualStrings("--policy and --loops are unsupported for Inkling 2 on Metal", (try refusal(a, o)).?);
    o = base;
    o.policy = "x=1";
    try std.testing.expectEqualStrings("--policy and --loops are unsupported for Inkling 2 on Metal", (try refusal(a, o)).?);
    o = base;
    o.prompt_cache_gib = 2;
    o.keep = 0;
    try std.testing.expect((try refusal(a, o)) == null);
}

test "inkling2 DSpark is the checkpoint's bundled head: --drafter is refused, not substituted" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(std.mem.indexOf(u8, (try refusal(a, .{ .dir = "d", .model_type = model_type, .drafter = "other" })).?, "bundled mtp.safetensors") != null);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = "12345" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "mtp.safetensors", .data = "678" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "other.safetensors", .data = "ignored" });
    const dir = try scratchDir(tmp, a);
    try std.testing.expectEqual(@as(u64, 5), checkpointFileBytes(a, std.testing.io, dir, false));
    try std.testing.expectEqual(@as(u64, 8), checkpointFileBytes(a, std.testing.io, dir, true));
    var shards = std.testing.tmpDir(.{});
    defer shards.cleanup();
    try shards.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = "1234" });
    try shards.dir.writeFile(std.testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = "567" });
    try shards.dir.writeFile(std.testing.io, .{ .sub_path = "mtp.safetensors", .data = "89" });
    const shard_dir = try scratchDir(shards, a);
    try std.testing.expectEqual(@as(u64, 7), checkpointFileBytes(a, std.testing.io, shard_dir, false));
    try std.testing.expectEqual(@as(u64, 9), checkpointFileBytes(a, std.testing.io, shard_dir, true));
    var indexed = std.testing.tmpDir(.{});
    defer indexed.cleanup();
    try indexed.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = "1234" });
    try indexed.dir.writeFile(std.testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = "567" });
    try indexed.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00001-of-00002.safetensors"}}
    });
    // The index names one shard: the other file is not what the loader maps.
    try std.testing.expectEqual(@as(u64, 4), checkpointFileBytes(a, std.testing.io, try scratchDir(indexed, a), false));
    try std.testing.expectEqual(@as(u64, 0), checkpointFileBytes(a, std.testing.io, "/nonexistent-inkling2-dir", true));
}

test "inkling2 prompt cache budget: half the room, explicit sizes checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(@as(?u64, 4 * gib), try cacheBudget(a, .{ .dir = "d", .model_type = model_type }, 8 * gib, &problem));
    try std.testing.expectEqual(@as(?u64, 0), try cacheBudget(a, .{ .dir = "d", .model_type = model_type, .prompt_cache_gib = 0 }, 8 * gib, &problem));
    try std.testing.expectEqual(@as(?u64, 2 * gib), try cacheBudget(a, .{ .dir = "d", .model_type = model_type, .prompt_cache_gib = 2 }, 8 * gib, &problem));
    try std.testing.expectEqual(@as(?u64, null), try cacheBudget(a, .{ .dir = "d", .model_type = model_type, .prompt_cache_gib = 9 }, 8 * gib, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "left under the load limit") != null);
    try std.testing.expectEqual(@as(?u64, 9 * gib), try cacheBudget(a, .{ .dir = "d", .model_type = model_type, .prompt_cache_gib = 9, .prompt_cache_over_cap = true }, 8 * gib, &problem));
}
