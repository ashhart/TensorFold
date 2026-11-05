//! Ripple on Metal served by the lane host: one-row rounds, windows of copied drafts, contexts budgeted.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fr = tf.ricursive;
const lanes = tf.lanes;
const Resident = @import("resident.zig").Resident;

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *fr.Model,
    metal: *fr.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the residency set's, or the bare queue's, for the lane host's idle ticker
    resident: ?Resident = null, // weights and scratch kept wired between requests, when they fit the 70% limit
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states, in what the streams' caches leave of the budget
    cache_cap: u64 = 0, // --prompt-cache-gib's bytes (the budget when unset)

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        drop(h);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = fr.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = switch (err) {
            error.RicursiveNeedsTensorUnits => "the native Ripple engine needs Metal 4 tensor units (an M5 or later) for its prompt GEMM and attention: this Mac's GPU (M1 to M4) has none",
            error.UnqualifiedRicursiveCheckpoint => "the native Ripple engine serves only the checkpoints checked against the release (Ripple-65B-Preview): this ricursive checkpoint's geometry is not one of them",
            error.UnsupportedRicursivePrecision => "the native Ripple engine serves the publisher's releases as published (ricursiveAI/Ripple-65B-Preview in bf16, or its -NVFP4 MLPs); this checkpoint is quantized otherwise",
            else => try std.fmt.allocPrint(a, "the native Ripple engine cannot load this checkpoint ({s})", .{@errorName(err)}),
        };
        return null;
    };
    errdefer model.deinit();
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    // the decode window's and the prompt chunk's buffers
    const scratch: u64 = fr.state.scratchBytes(&model.config, fr.state.batch_rows) + fr.state.scratchBytes(&model.config, fr.state.prompt_rows);
    const budget = try fr.budget.caches(ram, model.device.maxWorkingSet(), null, model.checkpoint.residentBytes() + scratch);
    const most: usize = @min(model.config.context, fr.state.max_capacity);
    const fits = fr.budget.longest(&model.config, budget, most);
    const window: i64 = o.context orelse @intCast(@min(fits, 262144));
    if (window < 1 or window > fits) {
        problem.* = if (o.context) |asked|
            try std.fmt.allocPrint(a, "--context {d}: this Mac's Ripple caches fit {d} tokens beside the weights under its load limit (75% of RAM)", .{ asked, fits })
        else
            try a.dupe(u8, "this Mac's load limit (75% of RAM) leaves no room for a Ripple stream's caches beside the weights");
        model.deinit(); // after the message, so a failed allocation leaves the model to the errdefer
        return null;
    }
    std.log.info("ricursive: {d:.1} GiB for caches, context up to {d} tokens ({d:.2} GiB a full stream)", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), window, @as(f64, @floatFromInt(fr.state.cacheBytes(&model.config, @intCast(window + 1)))) / (1 << 30) });
    const streams: u32 = @max(o.lanes, 1);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.metal = try fr.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = streams, .budget = budget });
    errdefer h.metal.deinit();
    try h.metal.warm();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), fr.state.window_rows, fr.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = fr.state.prompt_rows, .structures = true, .logprobs = true });
    h.host.info_.reasoning_optional = true; // Ripple writes `<think>` itself when it thinks, and answers bare otherwise
    h.host.info_.thinking_budget = false; // its think markers are spelled, not one token each: no close can be forced
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else budget;
    if (h.cache_cap > 0) {
        const M = fr.backend.Metal;
        // prompt chunks change arithmetic with their rows
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = fr.state.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop } }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    try hold(h);
    errdefer drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Weights, tables and scratches in one residency set for the keepalive, under the 70% limit.
fn hold(h: *Host) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    try list.append(h.gpa, m.inv);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.prompt);
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "ricursive");
    if (h.resident) |*r| h.metal.setResident(r.follow());
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = m.queue };
}

fn drop(h: *Host) void {
    h.metal.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
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
