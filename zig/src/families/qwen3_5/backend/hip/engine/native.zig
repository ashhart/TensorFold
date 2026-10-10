//! Qwen3.5 / Qwen3.6 for the native server on HIP: the engine sized to GPU memory and the lane backend over it.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const Engine = engine.Engine;
const memory = @import("memory.zig");
const mtp = @import("mtp.zig");
const prefix = @import("prefix.zig");
const pages = @import("../forward/pages.zig");
const state = @import("../forward/state.zig");
const worker = @import("worker.zig");
const hip_lanes = @import("hip_lanes.zig");

/// The MLX affine widths the family loads.
const mlx_formats: []const []const u8 = &.{ "mlx-q2", "mlx-q3", "mlx-q4", "mlx-q5", "mlx-q6", "mlx-q8" };

/// The dense and the sparse checkpoints are two registry entries over one engine.
pub fn Native(comptime type_name: []const u8) type {
    return struct {
        pub const model_type = type_name;
        pub const formats = mlx_formats;
        /// Positions a stream holds when neither --context nor the checkpoint's config names a window.
        pub const default_context: i64 = 32768;
        /// The engine cuts prompts itself.
        pub const prefill_step: u32 = 0;
        pub const open = openEngine;
        pub const follow = followRank;
    };
}

/// What the serve flags ask of the engine, as the host read them.
pub const Options = struct {
    /// Prompt plus reply tokens a request may use: --context, or the family default within the model's window.
    window: usize,
    /// Streams asked for at once; `fixed` when --parallel named the number, which then fits or is refused.
    streams: usize,
    fixed: bool = false,
    /// This process is `rank` of `world`.
    rank: u32 = 0,
    world: u32 = 1,
    /// Bytes kept prompt states may hold (null: the policy's, else what the streams leave) and their snapshot slots.
    cache_gib: ?f64 = null,
    /// A --prompt-cache-gib past what the streams leave is kept, not refused.
    cache_over_cap: bool = false,
};

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// The streams admitted at once, and the server's startup line (the memory plan).
    streams: usize,
    startup: []const u8,
};

/// A flag or a resource the engine cannot take: `problem` says why.
pub const Refused = error{Refused};

fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// One rank's loaded and sized engine: the streams and prompt cache every rank agreed on.
const Prepared = struct {
    e: *Engine,
    streams: usize,
    /// What the prompt cache holds: pages of the pool and linear snapshots (none when it is off).
    cache: memory.Cache,
    startup: []const u8,
};

/// Device bytes of `streams` lanes at `capacity` beside the weights; the prompt cache takes what is left.
fn served(e: *Engine, streams: usize, capacity: usize) !usize {
    const rows = streams * hip_lanes.Hip.max_window;
    return e.planBytes(streams, rows, capacity, streams * pages.pagesFor(capacity) + pages.pagesFor(rows), 0);
}

/// Loads this rank's share of the model, admits the streams its memory holds (with the other ranks) and sizes them.
fn prepare(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) (Refused || std.mem.Allocator.Error)!Prepared {
    const rows = hip_lanes.Hip.max_window;
    const held = hip.usage(false).device;
    var pf: engine.Preflight = .{};
    const e = Engine.load(gpa, io, dir, .{
        .preflight = &pf,
        .slack = rows + 1,
        .device = dev.index,
        .policy = dev.policy,
        .rank = o.rank,
        .world = o.world,
        .id = if (dev.group) |g| g.id else null,
    }) catch |err| return switch (err) {
        error.WeightsDoNotFit => refuse(a, problem, "the checkpoint's {d:.1} GiB of weights do not fit the {d:.1} GiB the HIP memory budget grants ({d:.1} GiB free less a {d:.1} GiB reserve{s}); free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_HIP_MEMORY_LIMIT_GB", .{ gibs(pf.weights), gibs(pf.pool.room(pf.held)), gibs(pf.pool.free), gibs(pf.pool.reserve), if (pf.pool.limit != null) ", under TENSORFOLD_HIP_MEMORY_LIMIT_GB" else "" }),
        error.BadReserve => refuse(a, problem, "TENSORFOLD_MEMORY_RESERVE_GIB: a number of GiB from 2 to the memory's size", .{}),
        error.BadLimit => refuse(a, problem, "TENSORFOLD_HIP_MEMORY_LIMIT_GB: a positive number of GiB whose byte count fits in a 64-bit size", .{}),
        error.HostMemoryUnavailable => refuse(a, problem, "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing HIP unified-memory admission", .{}),
        else => refuse(a, problem, "the native HIP engine cannot load {s} ({s})", .{ dir, @errorName(err) }),
    };
    errdefer e.deinit();
    const model = hip.usage(false).device - held;
    const after = e.budget(io) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    const room = after.room(model);
    const reserve = after.reserve;
    const capacity = o.window + rows + 1;
    // the most streams up to the ones asked whose bytes fit, then the least of that over the ranks
    var streams = o.streams;
    while (streams > 0) : (streams -= 1) {
        if ((served(e, streams, capacity) catch |err| return refuse(a, problem, "the native HIP engine cannot count its scratch ({s})", .{@errorName(err)})) <= room) break;
    }
    if (o.world > 1) streams = (e.least(.{ streams, 0 }) catch |err| return refuse(a, problem, "the ranks could not agree on the streams ({s})", .{@errorName(err)}))[0];
    if (streams == 0) {
        const one = served(e, 1, capacity) catch 0;
        return refuse(a, problem, "the HIP memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; lower --context, add ranks (--tp), or free device memory", .{ o.window, gibs(one), gibs(room), gibs(model), gibs(reserve) });
    }
    if (streams < o.streams and o.fixed) {
        const asked = served(e, o.streams, capacity) catch 0;
        return refuse(a, problem, "--parallel {d} needs {d:.2} GiB at a {d}-token window, and the HIP memory budget leaves {d:.2} GiB: serve --parallel {d}, or lower --context", .{ o.streams, gibs(asked), o.window, gibs(room), streams });
    }
    const need = served(e, streams, capacity) catch |err| return refuse(a, problem, "the native HIP engine cannot count its scratch ({s})", .{@errorName(err)});
    // the prompt cache: --prompt-cache-gib, else what the streams leave
    const left = room - need;
    const policy = dev.policy;
    const keep: usize = policy.slots;
    var budget: usize = if (o.cache_gib) |g| @intFromFloat(g * (1 << 30)) else left;
    if (budget > left and !o.cache_over_cap) {
        return refuse(a, problem, "a {d:.2} GiB prompt cache does not fit the {d:.2} GiB the streams leave; lower --prompt-cache-gib, or --prompt-cache-over-cap to keep it", .{ gibs(budget), gibs(left) });
    }
    if (keep == 0) budget = 0;
    var cache = memory.cache(e.weights.spec, e.act.size(), budget, keep);
    if (o.world > 1) {
        const fit = e.least(.{ cache.pages, cache.snaps }) catch |err| return refuse(a, problem, "the ranks could not agree on a prompt cache ({s})", .{@errorName(err)});
        cache = .{ .pages = fit[0], .snaps = fit[1] };
    }
    e.o.batch_rows = streams * rows;
    e.o.streams = streams;
    e.size(capacity, streams * pages.pagesFor(capacity) + pages.pagesFor(e.o.batch_rows) + cache.pages) catch |err|
        return refuse(a, problem, "the native HIP engine cannot allocate its scratch and pages ({s})", .{@errorName(err)});
    const lane = gibs(state.Caches.deviceBytes(e.model(), capacity) + pages.pagesFor(capacity) * memory.pageBytes(e.weights.spec, e.act.size()));
    const startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}, rank {d} of {d}: model {d:.2} GiB; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.1} GiB left after a {d:.1} GiB reserve; prompt cache {d:.2} GiB ({d} pages, {d} snapshots); prompts in {d}-row chunks", .{
        hip.kernels.arch, dev.index, o.rank, o.world, gibs(model), streams, if (streams == 1) "" else "s", lane, o.window, gibs(room), gibs(reserve), gibs(budget), cache.pages, cache.snaps, prefix.chunk,
    });
    return .{ .e = e, .streams = streams, .cache = cache, .startup = startup };
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    backend: *hip_lanes.Hip,
    startup: []const u8,
};

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    // the other ranks hear that the rounds are over, then every rank frees its share
    own.backend.deinit();
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.destroy(own);
}

/// The engine for `dir` on `dev`; under tensor parallelism this is rank 0, which serves.
fn openEngine(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!Loaded {
    const p = try prepare(a, gpa, io, dev, dir, o, problem);
    errdefer p.e.deinit();
    errdefer gpa.free(p.startup);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const before = hip.usage(false).device;
    own.* = .{ .gpa = gpa, .e = p.e, .backend = try hip_lanes.Hip.init(gpa, p.e), .startup = p.startup };
    errdefer own.backend.deinit();
    // the head is the one allocation of the plan made here: it holds what the plan counted for it
    const head = mtp.Head.deviceBytes(gpa, &p.e.driver, &p.e.lib, &p.e.weights, p.e.model(), @min(p.e.o.batch_rows, mtp.max_chains)) catch 0;
    if (hip.usage(false).device - before != head) return refuse(a, problem, "the draft head holds {d} device bytes, the plan says {d}", .{ hip.usage(false).device - before, head });
    if (p.cache.pages > 0) own.backend.keepPages(p.cache.pages, p.cache.snaps);
    // a lone rank times its forwards for the depth rule; the ranks of a group draft without costs
    if (dev.group) |g| own.backend.withLink(&g.link) else own.backend.measure();
    return .{
        .backend = own.backend.backend(),
        .facts = own.backend.facts(),
        .rows = hip_lanes.Hip.max_window,
        .ctx = own,
        .deinit = release,
        .streams = p.streams,
        .startup = p.startup,
    };
}

/// A rank above 0: holds its share of the model and runs rank 0's steps until rank 0 stops.
fn followRank(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!void {
    const p = try prepare(a, gpa, io, dev, dir, o, problem);
    defer p.e.deinit();
    defer gpa.free(p.startup);
    var w = try worker.Worker.init(gpa, p.e);
    defer w.deinit();
    try w.follow(&dev.group.?.link);
}
