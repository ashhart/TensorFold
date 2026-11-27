//! Qwen3.5-2B on the CPU, served by the existing lane host and scheduler (qwen35.zig's twin without a device).
const std = @import("std");
const api = @import("engine_api");
const q = @import("qwen35_cpu");
const lanes = @import("lanes");
const admit = @import("cpu_admit.zig");
const cache_budget = @import("cache_budget.zig");

/// The host's memory counts for /health: the process footprint and its peak.
fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    return admit.memory(reset_peak);
}

const Host = struct {
    gpa: std.mem.Allocator,
    model: *q.Model,
    cpu: *q.backend.Cpu,
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
        h.config.deinit(h.gpa);
        h.cpu.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

/// TF_QWEN35_CPU_THREADS=N: worker threads beside the engine thread (default: the logical CPUs less one).
fn threads() !?usize {
    return @import("sys").environment.read(?usize, "TF_QWEN35_CPU_THREADS", null);
}

/// TF_QWEN35_CPU_WINDOWS=1: verify windows of up to 16 rows (copy drafts) once the local exactness check passes.
fn wideWindows() !bool {
    return @import("sys").environment.read(bool, "TF_QWEN35_CPU_WINDOWS", false);
}

/// TF_QWEN35_CPU_TRACE=1: each one-token draw's top two logits and their margin on stderr, read once at open.
fn tracing() !bool {
    return @import("sys").environment.read(bool, "TF_QWEN35_CPU_TRACE", false);
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const started = std.Io.Timestamp.now(io, .awake);
    const model = q.Model.load(gpa, io, o.dir, try threads()) catch |err| {
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
    const chunk = 128;
    const served = try admit.streams(a, "qwen3.5", q.state.cacheBytes(@intCast(window)), o, @intCast(window), problem) orelse {
        model.deinit();
        return null;
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.cpu = try q.backend.Cpu.init(gpa, model, .{ .capacity = @intCast(window), .chunk = chunk, .streams = @max(served, 2), .trace = try tracing() }, try wideWindows());
    errdefer h.cpu.deinit();
    h.config = try lanes.Config.init(gpa, h.cpu.facts(), q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.cpu.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = served, .context_window = @intCast(window), .prefill_step = chunk, .decode_share = o.decode_share, .structures = true });
    h.host.memory = .{ .ctx = h, .read = readMemory };
    h.cache = null;
    if (o.prompt_cache_gib) |budget_gib| {
        const later = @as(u64, h.cpu.options.streams) * q.state.cacheBytes(@intCast(window));
        const plan = try cacheBudget(a, budget_gib, o.prompt_cache_over_cap, later, problem);
        h.host.info_.prompt_cache_plan = plan;
        if (plan.budget_bytes > 0) {
            const S = q.backend.Snaps;
            h.cache = api.prompt_cache.Store.init(gpa, .{
                .ptr = h.cpu,
                .vtable = &.{ .bytes = S.bytes, .save = S.save, .restore = S.restore, .drop = S.drop },
            }, .{ .planned = true, .grid = chunk, .chunk = chunk }, plan.budget_bytes);
            h.host.cache = &h.cache.?;
            h.host.info_.prompt_cache = true;
        }
    }
    errdefer if (h.cache) |*store| store.deinit();
    const loaded = started.durationTo(std.Io.Timestamp.now(io, .awake));
    std.log.info("qwen3.5 on the CPU: {d:.2} GiB of weights, {d} worker threads, {d} streams at a {d}-token window, windows of {d} rows, loaded in {d:.1} s", .{ @as(f64, @floatFromInt(model.bytes)) / (1 << 30), model.pool.threads.len, served, window, h.cpu.width, @as(f64, @floatFromInt(loaded.nanoseconds)) / 1e9 });
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The --prompt-cache-gib plan under the share of this host's RAM (cache_fit.zig's rule); no readings: nothing kept.
fn cacheBudget(a: std.mem.Allocator, gib: f64, over: bool, later: u64, problem: *[]const u8) !api.PromptCachePlan {
    const r = admit.readings() orelse {
        std.log.warn("prompt cache: this host's RAM or this process's footprint could not be read; keeping no prompt states", .{});
        return .{ .source = .explicit, .budget_bytes = 0, .explicit_budget = true };
    };
    const total = r.total;
    const loaded = r.used;
    const ready = loaded + later;
    const cap = total / 100 * admit.SHARE_PERCENT;
    const policy: cache_budget.Policy = .{ .margin = admit.MARGIN, .ceiling = 16 * cache_budget.GiB };
    const f = cache_budget.fitUnder(cap, ready, gib, over, policy) catch |e| {
        const left = (cache_budget.fitUnder(cap, ready, null, false, policy) catch unreachable).room;
        problem.* = try std.fmt.allocPrint(a, "--prompt-cache-gib {d} would take this server past {d}% of this host's RAM: it holds {d:.1} GiB once loaded and {d:.1} GiB more for its streams, so {d:.1} GiB is left for kept prompt states. Pass a smaller one, serve a smaller --parallel, or add --prompt-cache-over-cap to keep {d} GiB anyway.", .{ gib, admit.SHARE_PERCENT, @as(f64, @floatFromInt(loaded)) / (1 << 30), @as(f64, @floatFromInt(later)) / (1 << 30), @as(f64, @floatFromInt(left)) / (1 << 30), gib });
        return e;
    };
    std.log.info("prompt cache: {d:.1} GiB from {d:.1} GiB free under the {d}% cap", .{ @as(f64, @floatFromInt(f.budget)) / (1 << 30), @as(f64, @floatFromInt(f.room)) / (1 << 30), admit.SHARE_PERCENT });
    return .{ .source = .physical_footprint, .budget_bytes = f.budget, .explicit_budget = true, .over_cap = f.budget > f.room, .ram_bytes = total, .ready_footprint_bytes = loaded, .cap_bytes = f.cap, .room_bytes = f.room, .margin_bytes = admit.MARGIN };
}
