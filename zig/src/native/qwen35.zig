//! Qwen3.5-2B served by the existing lane host and scheduler.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen35;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");
const metal_admit = @import("metal_admit.zig");

/// The host's memory counts: `cache_fit.hostMemory` from the process footprint (no configured load limit).
fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    return cache_fit.hostMemory(0, reset_peak);
}

const Host = struct {
    resident: ?mtl.resident.Resident = null,
    gpa: std.mem.Allocator,
    model: *q.Model,
    metal: *q.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        if (h.metal.pages) |p| {
            const n = p.pool.census();
            std.log.info("qwen3.5: KV pages at close: {d} of {d} free, {d} leased, {d} promised ({s})", .{ n.free, n.total, n.leased, n.promised, if (p.idle()) "idle" else "NOT IDLE" });
        }
        mtl.resident.streams.drop(h);
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

/// TF_QWEN35_METAL_KV_PAGES=1: the streams' keys and values in pages of the core KV pool.
fn pagesOn() bool {
    const v = std.c.getenv("TF_QWEN35_METAL_KV_PAGES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The pool beside the model under the load limit: kv_pool's plan of the streams and pages, made now; why not, or null.
fn planPages(a: std.mem.Allocator, metal: *q.backend.Metal, device: mtl.Device, o: api.Open, window: u64) !?[]const u8 {
    const kv = q.state.kv;
    const l = q.state.pagedLayout();
    const limit = (cache_fit.ceiling(cache_fit.ram() orelse 0, device.maxWorkingSet(), null) catch unreachable).bytes;
    const used = @max(cache_fit.footprint() orelse 0, device.allocated());
    const room = limit -| used;
    const lanes_n: u32 = @max(o.lanes, 2);
    const p = kv.plan(l, .{ .room = room, .engine = l.fixed_bytes, .lanes = lanes_n, .fixed = o.lanes_fixed, .window = window, .engine_paged = true }) catch |e| return switch (e) {
        error.NoStream => try std.fmt.allocPrint(a, "TF_QWEN35_METAL_KV_PAGES: one stream's {d}-token window of KV pages ({d:.2} GiB) does not fit the {d:.2} GiB left under the load limit; lower --context", .{ window, gib(l.pages(window) * l.pageBytes()), gib(room) }),
        error.TooMany => try std.fmt.allocPrint(a, "TF_QWEN35_METAL_KV_PAGES: --parallel {d} needs a fixed part and a page for each stream beside the first, more than the {d:.2} GiB left under the load limit", .{ lanes_n, gib(room) }),
    };
    try metal.attachPages(l, p.lanes, p.pages, window);
    std.log.info("qwen3.5: KV in pages of the core pool: {d} streams over {d} pages of 256 rows ({d:.2} GiB, {d:.2} MiB a page), {d:.2} GiB left under the load limit", .{ p.lanes, p.pages, gib(p.pages * l.pageBytes()), @as(f64, @floatFromInt(l.pageBytes())) / (1 << 20), gib(room) });
    return null;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = q.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Qwen engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const window: i64 = o.context orelse @intCast(model.config.context);
    if (window < 1 or window > model.config.context) {
        model.deinit();
        problem.* = "--context must fit the Qwen checkpoint's positive context window";
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.resident = null;
    const chunk = 128;
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .chunk = chunk, .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    // TF_QWEN35_METAL_KV_PAGES: the page plan sets the streams; otherwise each stream's whole cache is admitted at load
    const paged = pagesOn();
    const served: u32 = if (paged) o.lanes else try metal_admit.streams(a, "qwen3.5", q.state.cacheBytes(@intCast(window)), metal_admit.room(model.device), o, @intCast(window), problem) orelse {
        h.metal.deinit();
        model.deinit();
        gpa.destroy(h);
        return null;
    };
    if (paged) if (try planPages(a, h.metal, model.device, o, @intCast(window))) |fail| {
        problem.* = fail;
        h.metal.deinit();
        model.deinit();
        gpa.destroy(h);
        return null;
    };
    if (!paged) h.metal.options.streams = @max(served, 2);
    h.config = try lanes.Config.init(gpa, h.metal.facts(), q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = served, .context_window = @intCast(window), .prefill_step = chunk, .structures = true });
    h.host.memory = .{ .ctx = h, .read = readMemory };
    h.cache = null;
    if (o.prompt_cache_gib) |budget_gib| {
        var why: []const u8 = "";
        const later = @as(u64, h.metal.options.streams) * if (paged)
            q.state.pagedLayout().fixed_bytes
        else
            q.state.cacheBytes(@intCast(window));
        const plan = cache_fit.budgetBeside(budget_gib, o.prompt_cache_over_cap, 0, later, a, &why, "") catch |err| {
            if (err == error.CacheOverCap) problem.* = why;
            return err;
        };
        h.host.info_.prompt_cache_plan = plan;
        if (plan.budget_bytes > 0) {
            const S = q.backend.Snaps;
            h.cache = api.prompt_cache.Store.init(gpa, .{
                .ptr = h.metal,
                .vtable = &.{ .bytes = S.bytes, .save = S.save, .restore = S.restore, .drop = S.drop },
            }, .{ .planned = true, .grid = chunk, .chunk = chunk }, plan.budget_bytes);
            h.host.cache = &h.cache.?;
            h.host.info_.prompt_cache = true;
        }
    }
    errdefer if (h.cache) |*store| store.deinit();
    try mtl.resident.streams.hold(h, io, "qwen3.5");
    errdefer mtl.resident.streams.drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
