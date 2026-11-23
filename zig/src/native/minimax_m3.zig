//! MiniMax-M3.1-Flash served by the lane host: one-row rounds and copy-drafted windows within the memory budget.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fl = tf.minimax_m3;
const lanes = tf.lanes;
const Resident = @import("resident.zig").Resident;
const metal_admit = @import("metal_admit.zig");

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *fl.Model,
    metal: *fl.backend.Metal,
    draft: ?*fl.dspark.Drafter = null, // the checkpoint's DSpark head (MINIMAX_DRAFTER=dspark), when drafts are on
    draft2: ?*fl.dflash2.Drafter = null, // incoai's DFlash2, when the checkpoint has it (the default)
    wrap: ?lanes.drafted.Drafted = null,
    warm: mtl.keepalive.Target = undefined, // the residency set's, or the bare queue's, for the lane host's idle ticker
    resident: ?Resident = null, // weights and scratch kept wired between requests, when they fit the 70% limit
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states, in what the streams' caches leave of the budget
    cache_cap: u64 = 0, // --prompt-cache-gib's bytes (the budget when unset)
    markers: [3]u32 = undefined, // an image's start, its token and end (--vision): text that spells one stays text

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        if (h.wrap) |*w| w.deinit();
        if (h.draft) |d| d.deinit();
        if (h.draft2) |d| d.deinit();
        h.config.deinit(h.gpa);
        fl.residency.drop(h);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

fn admittedStreams(a: std.mem.Allocator, stream_bytes: u64, left: u64, o: api.Open, window: u64, problem: *[]const u8) !?u32 {
    return metal_admit.streams(a, "minimax_m3", stream_bytes, left, o, window, problem);
}
const DraftCost = struct {
    fixed: u64 = 0,
    lane: u64 = 0,
    tap: u64 = 0,

    fn stream(d: DraftCost) u64 {
        return d.lane + d.tap;
    }
};

const Admission = struct { lanes: u32, ledger: u64 };

fn longestWindow(c: *const fl.config.Config, left: u64, most: usize, draft: DraftCost) usize {
    return fl.budget.longest(c, (left -| draft.fixed) -| draft.stream(), most);
}

fn fitAdmission(a: std.mem.Allocator, c: *const fl.config.Config, left: u64, o: api.Open, window: u64, draft: DraftCost, problem: *[]const u8) !?Admission {
    const target = fl.state.cacheBytes(c, @intCast(window + 1));
    const available = left -| draft.fixed;
    const reserved = draft.stream();
    const opened = try admittedStreams(a, target + reserved, available, o, window, problem) orelse return null;
    return .{ .lanes = opened, .ledger = available - opened * reserved };
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const kind = @import("tensorfold").minimax_m3.drafterKind() orelse {
        problem.* = "MINIMAX_DRAFTER names no drafter: give dflash2 or dspark, or unset it for dflash2";
        return null;
    };
    const model = fl.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native MiniMax-M3 engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    if (o.vision != null) model.loadVision(io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "--vision: this MiniMax-M3 checkpoint's image tower cannot load ({s}; {s} holds it)", .{ @errorName(err), fl.vision.shard });
        model.deinit();
        return null;
    };
    const df2 = if (o.drafts and kind == .dflash2) fl.dflash2.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the checkpoint's DFlash2 config cannot be read ({s})", .{@errorName(err)});
        model.deinit();
        return null;
    } else null;
    const dspark = if (o.drafts and df2 == null) fl.dspark.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the checkpoint's DSpark config cannot be read ({s})", .{@errorName(err)});
        model.deinit();
        return null;
    } else null;
    const draft2 = if (df2) |dc| try fl.dflash2.Drafter.init(gpa, model, dc) else null;
    errdefer if (draft2) |d| d.deinit();
    const draft = if (dspark) |dc| try fl.dspark.Drafter.init(gpa, model, dc) else null;
    errdefer if (draft) |d| d.deinit();
    const tap_count: u64 = if (df2) |dc| dc.taps.len else if (dspark) |dc| dc.taps.len else 0;
    const tap = tap_count * model.config.hidden * 2;
    const fixed: u64 = if (draft2) |d| d.fixedBytes() else if (draft) |d| d.fixedBytes() else 0;
    const lane: u64 = if (draft2) |d| d.laneBytes() else if (draft) |d| d.laneBytes() else 0;
    const cost: DraftCost = .{ .fixed = fixed + (fl.state.batch_rows + fl.state.prompt_rows) * tap, .lane = lane, .tap = tap };
    // refusals that return null free what they hold themselves: errdefer runs only for errors
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    const text: ?[]const u8 = if (std.c.getenv("MINIMAX_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const c = &model.config; // the decode window's and the prompt chunk's buffers
    const tower: u64 = if (model.vision) |v| v.bytes() else 0;
    const scratch: u64 = fl.state.scratchBytes(c, fl.state.batch_rows, c.context) + fl.state.scratchBytes(c, fl.state.prompt_rows, c.context) + tower;
    const budget = fl.budget.caches(ram, model.device.maxWorkingSet(), text, model.checkpoint.residentBytes() + scratch) catch {
        if (draft) |d| d.deinit();
        if (draft2) |d| d.deinit();
        model.deinit();
        problem.* = "MINIMAX_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 75% of RAM";
        return null;
    };
    const fits = longestWindow(c, budget, model.config.context, cost);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's MiniMax-M3 caches fit {d} tokens under its load limit (75% of RAM, or MINIMAX_LOAD_LIMIT_GB)", .{ window, fits });
        if (draft) |d| d.deinit();
        if (draft2) |d| d.deinit();
        model.deinit(); // after the message, so a failed allocation leaves the model to the errdefer
        return null;
    }
    const admission = try fitAdmission(a, c, budget, o, @intCast(window), cost, problem) orelse {
        if (draft) |d| d.deinit();
        if (draft2) |d| d.deinit();
        model.deinit();
        return null;
    };
    const served = admission.lanes;
    const ledger = admission.ledger;
    std.log.info("minimax_m3: {d:.1} GiB for caches, context up to {d} tokens", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), window });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.draft = draft;
    h.draft2 = draft2;
    h.metal = try fl.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = served, .budget = ledger, .drafts = dspark != null or df2 != null });
    errdefer h.metal.deinit();
    try h.metal.warm();
    h.wrap = null;
    if (df2 != null) {
        try h.metal.measure();
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), draft2.?.drafter());
    } else if (dspark != null) {
        try h.metal.measure(); // the depth rule prices drafts against the window forward's cost by rows
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), draft.?.drafter());
    }
    errdefer if (h.wrap) |*w| w.deinit();
    const facts = if (h.wrap) |*w| w.facts(h.metal.facts()) else h.metal.facts();
    h.config = try lanes.Config.init(gpa, facts, fl.state.window_rows, fl.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.wrap) |*w| w.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.markers = .{ fl.vision.image_start, model.config.image_token, fl.vision.image_end };
    // the template writes the image token alone: the server adds start and end around each image's rows
    const vis: ?api.Vision = if (o.vision) |v| .{ .ctx = h, .prepare = prepareImage, .image_token = h.markers[1], .markers = &h.markers, .wrap = .{ h.markers[0], h.markers[2] }, .part_type = "image", .image_tokens = v.image_tokens, .max_images = v.max_images, .image_share = v.image_share, .explain = tf.qwen_vl.geometry.problem } else null;
    h.host = api.LaneHost.init(gpa, io, &h.core, servedInfo(h, served, @intCast(window), vis));
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else ledger;
    if (h.cache_cap > 0) {
        const M = fl.backend.Metal;
        // prompt chunks change arithmetic with their rows
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = fl.state.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop } }, rules, @min(h.cache_cap, ledger));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    try fl.residency.hold(h);
    errdefer fl.residency.drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Info.vision's prepare (on the HTTP thread): at most the tower's 576 tokens an image.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try fl.vision.prepare(h.io, a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

/// Metal.room's fit: the kept states' budget is what the streams leave, at most the cap; the least valuable go first.
fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = if (h.cache) |*owned| owned else return;
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

const CacheFixture = struct {
    live: usize = 0,
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !api.prompt_cache.Saved {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const held = try std.testing.allocator.create(u32);
        held.* = at;
        self.live += 1;
        return held;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) !void {}
    fn drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        std.testing.allocator.destroy(@as(*u32, @ptrCast(@alignCast(saved))));
        self.live -= 1;
    }
    fn snapshots(self: *@This()) api.prompt_cache.Snapshots {
        return .{ .ptr = self, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

test "minimax_m3 host: cache callback updates the owned store rather than a temporary value" {
    var f: CacheFixture = .{};
    var h: Host = undefined;
    h.cache_cap = 500;
    h.cache = api.prompt_cache.Store.init(std.testing.allocator, f.snapshots(), .{ .min_prompt = 0 }, 500);
    defer h.cache.?.deinit();
    fitCache(&h, 200, null, &.{});
    try std.testing.expectEqual(@as(u64, 200), h.cache.?.budget);
    fitCache(&h, 800, null, &.{});
    try std.testing.expectEqual(@as(u64, 500), h.cache.?.budget);
}

test "minimax_m3 host: cache callback evicts from the real store and preserves the restored entry" {
    var f: CacheFixture = .{};
    var h: Host = undefined;
    h.cache_cap = 1000;
    h.cache = api.prompt_cache.Store.init(std.testing.allocator, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer h.cache.?.deinit();
    const store = &h.cache.?;
    try std.testing.expect(store.keep(&.{ 1, 1, 1, 1, 9 }, 4, null, &.{}, &.{}));
    try std.testing.expect(store.keep(&.{ 2, 2, 2, 2, 2, 2, 9 }, 6, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 210), store.held);
    const kept = store.entries.items[0].saved;
    fitCache(&h, 110, kept, &.{ 1, 1, 1, 1, 9 });
    try std.testing.expectEqual(@as(u64, 110), store.budget);
    try std.testing.expectEqual(@as(u64, 104), store.held);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expect(store.entries.items[0].saved == kept);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    fitCache(&h, 110, kept, &.{ 1, 1, 1, 1, 9 });
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    fitCache(&h, 0, null, &.{});
    try std.testing.expectEqual(@as(u64, 0), store.held);
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(u64, 2), store.counts.evicted);
}

test "minimax_m3 host: cache callback with no configured cache leaves no state to mutate" {
    var h: Host = undefined;
    h.cache = null;
    fitCache(&h, 0, null, &.{});
    try std.testing.expect(h.cache == null);
}

fn servedInfo(h: *const Host, streams: u32, window: u32, vision: ?api.Vision) api.Info {
    return .{ .lanes = streams, .context_window = window, .prefill_step = fl.state.prompt_rows, .structures = true, .logprobs = h.wrap == null, .vision = vision };
}

test "MiniMax capabilities: logprobs match the actual plain or external-draft backend" {
    var metal: fl.backend.Metal = undefined;
    var h: Host = undefined;
    h.wrap = null;
    const plain = servedInfo(&h, 2, 1024, null);
    try std.testing.expect(metal.backend().vtable.first_row != null);
    try std.testing.expect(plain.logprobs and plain.structures);
    h.wrap = .{ .gpa = std.testing.allocator, .target = metal.backend(), .drafter = .{ .ptr = &metal, .vtable = undefined } };
    const wrapped = h.wrap.?.backend();
    const drafted = servedInfo(&h, 2, 1024, null);
    try std.testing.expect(wrapped.vtable.first_row == null);
    try std.testing.expectEqual(wrapped.vtable.first_row != null, drafted.logprobs);
    try std.testing.expect(drafted.structures and wrapped.vtable.first_masked != null);
    try std.testing.expectEqual(plain.context_window, drafted.context_window);
    try std.testing.expectEqual(plain.prefill_step, drafted.prefill_step);
}

test "Metal capabilities: MiniMax admission fits automatic 64 and refuses explicit over-capacity" {
    const a = std.testing.allocator;
    const auto: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 64 };
    const exact: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 3, .lanes_fixed = true };
    const over: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 64, .lanes_fixed = true };
    var why: []const u8 = "";
    try std.testing.expectEqual(@as(?u32, 3), try admittedStreams(a, 3, 11, auto, 4096, &why));
    try std.testing.expectEqual(@as(?u32, 3), try admittedStreams(a, 3, 11, exact, 4096, &why));
    try std.testing.expectEqual(@as(?u32, null), try admittedStreams(a, 3, 11, over, 4096, &why));
    defer a.free(why);
    try std.testing.expect(std.mem.indexOf(u8, why, "--parallel 64") != null);
}

test "MiniMax admission: window allocation includes the extra position and default stops one token earlier" {
    const a = std.testing.allocator;
    const c: fl.config.Config = .{};
    const short = fl.state.cacheBytes(&c, 128);
    var why: []const u8 = "";
    const auto: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 3 };
    const exact: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 3, .lanes_fixed = true };
    const three = (try fitAdmission(a, &c, 3 * short, auto, 128, .{}, &why)).?;
    try std.testing.expectEqual(@as(u32, 2), three.lanes);
    try std.testing.expect((try fitAdmission(a, &c, 3 * short, exact, 128, .{}, &why)) == null);
    defer a.free(why);
    try std.testing.expectEqual(@as(usize, 127), longestWindow(&c, short, 128, .{}));
}

fn expectDraftedAdmission(comptime D: type) !void {
    const a = std.testing.allocator;
    const c: fl.config.Config = .{};
    var drafter: D.Drafter = undefined;
    drafter.config = .{};
    const fixed = 257;
    const tap = drafter.config.taps.len * c.hidden * 2;
    const draft: DraftCost = .{ .fixed = fixed, .lane = drafter.laneBytes(), .tap = tap };
    const target = fl.state.cacheBytes(&c, 129);
    const reserved = draft.lane + draft.tap;
    const left = fixed + 3 * target + 2 * reserved;
    const auto: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 3 };
    const exact: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 3, .lanes_fixed = true };
    const two: api.Open = .{ .dir = "", .model_type = "minimax_m3_vl", .lanes = 2, .lanes_fixed = true };
    var why: []const u8 = "";
    const fitted = (try fitAdmission(a, &c, left, auto, 128, draft, &why)).?;
    try std.testing.expectEqual(@as(u32, 2), fitted.lanes);
    try std.testing.expectEqual(2 * target, fitted.ledger);
    try std.testing.expect((try fitAdmission(a, &c, left, exact, 128, draft, &why)) == null);
    defer a.free(why);
    const pair = (try fitAdmission(a, &c, left, two, 128, draft, &why)).?;
    try std.testing.expectEqual(2 * target, pair.ledger);
    try std.testing.expectEqual(@as(usize, 127), longestWindow(&c, fixed + fl.state.cacheBytes(&c, 128) + reserved, 128, draft));
}

test "MiniMax admission: DFlash2 fixed, lane and tap reservations share the target and snapshot ledger" {
    try expectDraftedAdmission(fl.dflash2);
}

test "MiniMax admission: DSpark fixed, lane and tap reservations share the target and snapshot ledger" {
    try expectDraftedAdmission(fl.dspark);
}
