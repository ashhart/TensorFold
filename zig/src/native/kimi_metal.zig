//! K3 on one Metal device through the existing lane host, with a resident-weight budget and exact prompt snapshots.
const std = @import("std");
const mtl = @import("metal");
const family = @import("kimi_k3");
const api = @import("engine_api");
const lanes = @import("tensorfold").lanes;
const Gpu = @import("kimi_gpu.zig").Gpu;
const M = family.backend.Metal;
const A = std.mem.Allocator;
const GiB: u64 = 1 << 30;

const Host = struct {
    a: A,
    gpu: ?Gpu = null,
    shards: ?family.shards.Shards = null,
    graph: ?family.graph.Graph = null,
    metal: ?*M = null,
    cache: ?api.prompt_cache.Store = null,
    config: ?lanes.Config = null,
    clock: lanes.backend.WallClock = undefined,
    core: ?lanes.Engine = null,
    host: ?api.LaneHost = null,
    started: bool = false,

    fn close(ptr: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        if (h.started) h.host.?.stop();
        if (h.cache) |*store| store.deinit();
        if (h.core) |*core| core.deinit();
        if (h.config) |*config| config.deinit(h.a);
        if (h.metal) |m| m.deinit();
        if (h.graph) |*graph| graph.deinit();
        if (h.shards) |*shards| shards.deinit();
        if (h.gpu) |*gpu| gpu.deinit();
        h.a.destroy(h);
    }
    fn healthy(ptr: *anyopaque) bool {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.metal.?.healthy.load(.acquire);
    }
    fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return family.snapshot.bound(&h.graph.?.c, at, family.backend.window);
    }
    fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !api.prompt_cache.Saved {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.KimiSnapshotOwner));
        const m = h.metal.?;
        const slot = try m.find(stream);
        if (m.state.streams[slot].pos != at or at > stream.prompt_len) return error.KimiSnapshotPosition;
        const saved = try family.snapshot.save(m, slot, stream.prompt()[0..at]);
        return @as(api.prompt_cache.Saved, @ptrCast(saved));
    }
    fn snapRestore(ptr: *anyopaque, owner: ?*anyopaque, saved: api.prompt_cache.Saved) !void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.KimiSnapshotOwner));
        const snap: *const family.snapshot.Snap = @ptrCast(@alignCast(saved));
        const m = h.metal.?;
        const slot = m.find(stream) catch try m.acquire(stream);
        try family.snapshot.restore(m, slot, snap, stream.prompt(), @intCast(stream.reuse.at));
    }
    fn snapDrop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const snap: *family.snapshot.Snap = @ptrCast(@alignCast(saved));
        snap.deinit(h.a);
    }
    fn snapCharge(_: *anyopaque, saved: api.prompt_cache.Saved) u64 {
        const snap: *const family.snapshot.Snap = @ptrCast(@alignCast(saved));
        return snap.charge();
    }
};

pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.tp != 1 or o.rank != 0) return "Kimi: this native adapter serves one Metal device, --tp 1 and --rank 0";
    if (o.device) |id| if (id != 0) return "Kimi: this native adapter serves the default Metal device only";
    if (o.vision != null) return "Kimi: image input is not supported by this text adapter";
    if (o.kv8 or o.prompt_int8) return "Kimi: this adapter retains bf16 activations and the exact latent cache";
    if (o.precision) |p| if (p != .full) return "Kimi: --precision checkpoint is unsupported; use full";
    if (o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null) return "Kimi: external draft heads are not implemented; exact context-copy drafts remain available";
    if (o.slide or o.learn != null or o.learn_conversations or o.recall or o.record != null or o.spill != .off) return "Kimi: learned states and on-disk prompt states are not implemented";
    if (o.loops != null) return "Kimi: --loops is for a looped model, not this fixed-layer graph";
    if (o.speed_up != null or o.policy.len != 0) return "Kimi: --speed-up and --policy are unsupported";
    if (o.decode_share != null) return "Kimi: --decode-share is not qualified for this bring-up adapter";
    if (o.segments) |n| if (n != 1) return "Kimi: --segments must be 1; independent streams already share exact rounds";
    if (o.lanes == 0 or o.lanes > 8) return "Kimi: --parallel must be 1 through 8";
    if (o.keep) |n| if (n < 0) return "Kimi: --checkpoint-slots must be nonnegative";
    if (o.prompt_cache_gib) |n| if (!std.math.isFinite(n) or n < 0 or n > 1024) return "Kimi: --prompt-cache-gib must be a finite budget from 0 through 1024";
    return null;
}

pub fn open(a: A, gpa: A, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusal(o)) |why| {
        problem.* = why;
        return null;
    }
    return inner(a, gpa, io, o, problem) catch |err| {
        if (err == error.OutOfMemory) return err;
        if (problem.*.len == 0) problem.* = words(null, err) orelse
            try std.fmt.allocPrint(a, "Kimi: this native Metal adapter cannot open the checkpoint ({s})", .{@errorName(err)});
        return null;
    };
}

fn inner(a: A, gpa: A, io: std.Io, o: api.Open, problem: *[]const u8) !api.Opened {
    const config_path = try std.fs.path.join(a, &.{ o.dir, "config.json" });
    defer a.free(config_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(16 << 20));
    defer a.free(bytes);
    const c = try family.config.parse(gpa, bytes);
    try family.graph.nativeShape(c);
    const capacity: i64 = o.context orelse if (c.dsa != null) 2047 else 4096;
    if (capacity < 1 or capacity > 65536 or (c.dsa != null and capacity > 2047)) {
        problem.* = if (c.dsa != null) family.config.dsa_config.limit_message else "Kimi: --context must be 1 through 65536";
        return error.KimiContextLimit;
    }
    const streams = o.lanes;
    const cache: u64 = if (o.prompt_cache_gib) |g| @intFromFloat(g * GiB) else GiB;
    const weight = try family.graph.weightBytes(gpa, c);
    const ram = mtl.ResidencySet.totalRam() orelse return error.KimiUnknownMemoryBudget;
    const limit = ram / 100 * mtl.ResidencySet.share_percent;
    // This conservative scratch allowance is checked against the actual allocations before the server starts.
    const fixed = 4 * GiB + family.snapshot.bound(&c, @intCast(capacity), family.backend.window) * streams;
    try budget(a, weight.source + weight.prepared + fixed + cache, limit, problem);
    const h = try gpa.create(Host);
    h.* = .{ .a = gpa };
    errdefer Host.close(h);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    h.gpu = try Gpu.init(c.dsa != null);
    const gpu = &h.gpu.?;
    h.shards = try family.shards.Shards.open(gpa, io, gpu.device, o.dir);
    var physical: u64 = 0;
    for (h.shards.?.files.items) |file| physical += (try std.Io.Dir.cwd().statFile(io, file.path, .{})).size;
    const working_limit = @min(limit, gpu.device.maxWorkingSet());
    try budget(a, @max(weight.source, physical) + weight.prepared + fixed + cache, working_limit, problem);
    h.graph = try family.graph.Graph.load(gpa, gpu.device, gpu.queue, &gpu.k, c, h.shards.?.source());
    var buffers: std.ArrayList(mtl.Buffer) = .empty;
    defer buffers.deinit(gpa);
    try h.shards.?.buffers(&buffers, gpa);
    h.metal = try M.init(gpa, &h.graph.?, gpu.device, gpu.queue, &gpu.k, streams, @intCast(capacity), buffers.items);
    const m = h.metal.?;
    var actual: u64 = m.last.length();
    inline for (@typeInfo(family.round.Scratch).@"struct".field_names, @typeInfo(family.round.Scratch).@"struct".field_types) |f, T| {
        if (T == mtl.Buffer) actual += @field(m.scratch, f).length();
    }
    buffers.clearRetainingCapacity();
    if (m.index) |*ix| try ix.buffers(&buffers, gpa);
    for (buffers.items) |b| actual += b.length();
    buffers.clearRetainingCapacity();
    try m.state.buffers(&buffers, gpa);
    for (buffers.items) |b| actual += b.length();
    if (actual > fixed) return error.KimiScratchBound;
    var facts = m.facts();
    facts.copy_drafts = o.drafts;
    h.config = try lanes.Config.init(gpa, facts, family.backend.window, family.backend.window - 1);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config.?, m.backend(), h.clock.clock());
    h.host = nativeHost(gpa, io, &h.core.?, .{ .name = if (c.dsa != null) "kimi_k31" else "kimi_k3", .lanes = streams, .context_window = @intCast(capacity), .prefill_step = 256, .structures = false, .thinking_budget = false, .reasoning = true, .prompt_cache = cache > 0, .warm_turns = cache > 0, .plain_only = !o.drafts, .policy = "Kimi resident Metal bring-up; synthetic qualification only, no real-checkpoint reference" });
    h.host.?.health = .{ .ctx = h, .read = Host.healthy };
    if (cache > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h, .vtable = &.{ .bytes = Host.snapBytes, .save = Host.snapSave, .restore = Host.snapRestore, .drop = Host.snapDrop, .charged = Host.snapCharge } }, .{ .chunk = 256, .min_prompt = 0, .warm = .on, .repeat_min = 1 }, cache);
        if (o.keep) |n| h.cache.?.most = @intCast(n);
        h.host.?.cache = &h.cache.?;
        h.host.?.info_.prompt_cache_capacity = .{ .budget_bytes = cache, .minimum_state_bytes = Host.snapBytes(h, 1) };
    }
    try h.host.?.start();
    h.started = true;
    return .{ .engine = h.host.?.engine(), .close = Host.close, .ctx = h };
}

fn nativeHost(a: A, io: std.Io, core: *lanes.Engine, info: api.Info) api.LaneHost {
    var h = api.LaneHost.init(a, io, core, info);
    // XTML has a multi-token channel closer; do not inherit the generic lane host's answer-budget claims.
    h.info_.answer_limit = false;
    h.info_.loop_guard = false;
    h.explain = .{ .text = words };
    return h;
}

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.KimiScratchBound => "Kimi: stream state and working buffers exceed this adapter's reserved memory budget; the checkpoint cannot be opened safely",
        error.K31ContextLimit => family.config.dsa_config.limit_message,
        error.KimiContextLimit => "Kimi: prompt plus reply, including the reused prefix, exceeds configured --context",
        error.KimiSnapshotCapacity => "Kimi: cached prompt state exceeds this engine's configured --context",
        error.KimiSnapshotPrefix => "Kimi: cached prompt state does not match this prompt or engine",
        else => null,
    };
}

fn budget(a: A, need: u64, limit: u64, problem: *[]const u8) !void {
    if (need <= limit and mtl.ResidencySet.margin <= limit - need) return;
    problem.* = try std.fmt.allocPrint(a, "Kimi: resident checkpoint, prepared projections, streams and prompt states need {d:.2} GiB plus reserve; the load budget is {d:.2} GiB. The complete model does not fit on one Mac; this adapter has no SSD residency path", .{ @as(f64, @floatFromInt(need)) / GiB, @as(f64, @floatFromInt(limit)) / GiB });
    return error.KimiDoesNotFit;
}

test "Kimi full release refuses by budget before opening shaders, shards or weights" {
    const a = std.testing.allocator;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = family.graph.k31_fixture });
    const path = try temp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(path);
    var why: []const u8 = "";
    const result = try open(a, a, std.testing.io, pathOpen(path), &why);
    defer a.free(why);
    try std.testing.expect(result == null);
    try std.testing.expect(std.mem.indexOf(u8, why, "complete model does not fit") != null);
}

fn pathOpen(path: []const u8) api.Open {
    return .{ .dir = path, .model_type = "kimi_k31", .lanes = 1 };
}

test "Kimi native capabilities remain limited after the shared lane host constructor" {
    const a = std.testing.allocator;
    var config = try lanes.Config.init(a, .{}, 1, 0);
    defer config.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &config, target.backend(), clock.clock());
    defer core.deinit();
    var h = nativeHost(a, std.testing.io, &core, .{ .thinking_budget = false, .reasoning = true });
    const actual = h.engine().info();
    try std.testing.expect(actual.reasoning);
    try std.testing.expect(!actual.thinking_budget and !actual.answer_limit and !actual.loop_guard);
    try std.testing.expectEqualStrings(family.config.dsa_config.limit_message, h.words(error.K31ContextLimit));
    try std.testing.expect(std.mem.indexOf(u8, h.words(error.KimiSnapshotCapacity), "configured --context") != null);
}

test "Kimi scratch budget refusal names working memory rather than an internal error" {
    const want = "Kimi: stream state and working buffers exceed this adapter's reserved memory budget; the checkpoint cannot be opened safely";
    try std.testing.expectEqualStrings(want, words(null, error.KimiScratchBound) orelse @errorName(error.KimiScratchBound));
}
