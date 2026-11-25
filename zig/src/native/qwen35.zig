//! Qwen3.5-2B served by the existing lane host and scheduler.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen35;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");
const metal_admit = @import("metal_admit.zig");
const metal_witness = @import("metal_witness.zig");
const mem_ctrl = @import("memory_controller.zig");

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
    witness: ?*metal_witness.Witness = null, // --execution-witness on its path, closed before the model
    folded_mem: ?*mem_ctrl.Holder = null,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.witness) |w| w.close();
        if (h.folded_mem) |fm| fm.deinit();
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
fn pagesOn() !bool {
    return @import("sys").environment.read(bool, "TF_QWEN35_METAL_KV_PAGES", false);
}

fn chunkRows(_: std.mem.Allocator, window: usize, problem: *[]const u8) !usize {
    const settings = @import("sys").environment;
    comptime settings.registered("TF_QWEN35_CHUNK");
    const chunk = settings.read(usize, "TF_QWEN35_CHUNK", 128) catch |err| {
        if (err == error.InvalidEnvironment) {
            problem.* = "TF_QWEN35_CHUNK: a whole number 1 to the model's context window; unset for the shipped 128";
            return error.InvalidChunk;
        }
        return err;
    };
    if (chunk == 0 or chunk > window) {
        problem.* = "TF_QWEN35_CHUNK: a whole number 1 to the model's context window; unset for the shipped 128";
        return error.InvalidChunk;
    }
    return chunk;
}

test "TF_QWEN35_CHUNK is parsed through platform/environment.zig" {
    const settings = @import("sys").environment;
    try std.testing.expectEqual(@as(?u64, null), try settings.parse("TF_QWEN35_CHUNK", null));
    try std.testing.expectEqual(@as(?u64, 512), try settings.parse("TF_QWEN35_CHUNK", "512"));
    try std.testing.expectError(error.InvalidEnvironment, settings.parse("TF_QWEN35_CHUNK", "0"));
    try std.testing.expectError(error.InvalidEnvironment, settings.parse("TF_QWEN35_CHUNK", "4k"));
}

test "qwen35 startup: TF_QWEN35_CHUNK refuses before the checkpoint loader runs" {
    const env = @import("sys").env;
    const saved = std.c.getenv("TF_QWEN35_CHUNK");
    defer if (saved) |v| {
        _ = env.setenv("TF_QWEN35_CHUNK", std.mem.span(v), 1);
    } else {
        _ = env.unsetenv("TF_QWEN35_CHUNK");
    };
    try std.testing.expectEqual(@as(c_int, 0), env.setenv("TF_QWEN35_CHUNK", "4k", 1));
    var problem: []const u8 = "";
    try std.testing.expectError(error.InvalidChunk, chunkRows(std.testing.allocator, 8192, &problem));
    try std.testing.expectEqualStrings(
        "TF_QWEN35_CHUNK: a whole number 1 to the model's context window; unset for the shipped 128",
        problem,
    );
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
    const settings = @import("sys").environment;
    if (try settings.refusal(a, settings.Process{}, .{ .metal = true })) |why| {
        problem.* = why;
        return null;
    }
    if (o.slide and mem_ctrl.useLegacyPocket()) if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    };
    const cfg = q.config.Config.read(a, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Qwen engine cannot read this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    const window: i64 = o.context orelse @intCast(cfg.context);
    if (window < 1 or window > cfg.context) {
        problem.* = "--context must fit the Qwen checkpoint's positive context window";
        return null;
    }
    const chunk = chunkRows(a, @intCast(@max(window, 1)), problem) catch return null;
    const chunk_u32: u32 = @intCast(chunk);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const witness = if (metal_witness.wanted(o)) try metal_witness.openOrFailed(gpa, io, o.dir) else null; // before the load
    defer metal_witness.release(witness);
    const model = q.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Qwen engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.resident = null;
    h.witness = null;
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .chunk = chunk_u32, .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    // TF_QWEN35_METAL_KV_PAGES: the page plan sets the streams; otherwise each stream's whole cache is admitted at load
    const paged = try pagesOn();
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
    const bound = metal_witness.bound(witness, model.device, .{ @intCast(window), served, chunk, h.metal.width, @intFromBool(paged) });
    h.config = try lanes.Config.init(gpa, h.metal.facts(), q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = served, .context_window = @intCast(window), .prefill_step = chunk_u32, .decode_share = o.decode_share, .structures = true, .witness = metal_witness.source(bound) });
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
            }, .{ .planned = true, .grid = chunk_u32, .chunk = chunk_u32, .warm_usable = true }, plan.budget_bytes);
            h.host.cache = &h.cache.?;
        }
    }
    h.host.advertiseCache(if (h.cache) |*store| store else null);
    errdefer if (h.cache) |*store| store.deinit();
    try mtl.resident.streams.hold(h, io, "qwen3.5");
    errdefer mtl.resident.streams.drop(h);
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    h.folded_mem = null;
    if (o.slide) { // a saved bank alone never attaches it: --slide asks for learning
        const adapter = try q.memory_adapter.Adapter.init(gpa, h.metal);
        errdefer adapter.deinit(gpa);
        const holder = try mem_ctrl.Holder.init(gpa, io, o.dir, adapter.backend(), adapter, q.memory_adapter.Adapter.deinitOpaque);
        h.folded_mem = holder;
        h.host.folded_memory = holder.controller();
        h.host.info_.folded_memory = true;
    }
    errdefer if (h.folded_mem) |fm| fm.deinit();
    try h.host.start();
    h.witness = bound;
    if (bound) |w| w.adopted = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
