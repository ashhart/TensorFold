//! Nemotron on the XPU server: its lane host (one family behind the lane core, one stream, host sampling).
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const lanes = @import("lanes");
const lane = @import("xpu_lane.zig");
const Allocator = std.mem.Allocator;

/// The families this host serves, each a file with its table row, loader and prompt window.
const specs = .{
    @import("xpu_nemotron.zig"),
};

/// Prompt plus reply tokens when --context names none (and the model's window is larger).
const default_context: i64 = 32768;

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

var stopping = std.atomic.Value(bool).init(false);
var prev_int: std.posix.Sigaction = undefined;
var prev_term: std.posix.Sigaction = undefined;

fn onStopSignal(sig: std.posix.SIG) callconv(.c) void {
    stopping.store(true, .release);
    const prev = if (sig == .INT) prev_int else prev_term;
    const f = prev.handler.handler orelse return;
    if (f != std.posix.SIG.DFL and f != std.posix.SIG.IGN) f(sig);
}

/// Called on the first prompt, after the server's handlers exist: ours flags the backend, then calls the server's.
fn chainStopHandlers() void {
    const act: std.posix.Sigaction = .{ .handler = .{ .handler = onStopSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &act, &prev_int);
    std.posix.sigaction(.TERM, &act, &prev_term);
}

fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const m: api.Memory = .{ .active = xpu.rt.alloc_now, .cache = 0, .peak = xpu.rt.alloc_peak };
    if (reset_peak) xpu.rt.alloc_peak = xpu.rt.alloc_now;
    return m;
}

/// Device memory left: Sysman's free bytes (ze.Gpu.memory), at most the card's size less what this engine allocated.
fn freeMemory(ctx: *xpu.Context) u64 {
    const mem = ctx.gpu.memory() catch return 0;
    return @min(mem.free, mem.size -| xpu.rt.alloc_now);
}

/// One loaded model behind the lane host: the device, the engine, the lane backend and the round loop.
fn Host(comptime S: type) type {
    return struct {
        const Self = @This();
        gpa: Allocator,
        driver: xpu.Driver,
        ctx: xpu.Context,
        engine: *S.Engine,
        backend: lane.Xpu(S),
        cfg: lanes.Config,
        clock: lanes.backend.WallClock,
        core: lanes.Engine,
        host: api.LaneHost,
        startup: []u8 = &.{},
        cache: ?api.prompt_cache.Store = null,

        fn close(p: *anyopaque) void {
            const h: *Self = @ptrCast(@alignCast(p));
            h.host.stop();
            if (h.cache) |*store| store.deinit();
            h.core.deinit();
            h.cfg.deinit(h.gpa);
            h.backend.deinit();
            h.engine.deinit();
            h.ctx.deinit();
            h.driver.close();
            h.gpa.free(h.startup);
            h.gpa.destroy(h);
        }
    };
}

/// The model types this host reads.
const served = blk: {
    var s: []const u8 = "";
    for (specs, 0..) |S, i| s = s ++ (if (i > 0) ", " else "") ++ S.model_type;
    break :blk s;
};

/// The engine for `o.dir`, or null with `problem` set when no Intel GPU engine reads the checkpoint.
pub fn serves(model_type: []const u8) bool {
    inline for (specs) |S| if (std.mem.eql(u8, model_type, S.model_type)) return true;
    return false;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    inline for (specs) |S| if (std.mem.eql(u8, o.model_type, S.model_type)) return openFamily(S, a, gpa, io, o, problem);
    problem.* = try std.fmt.allocPrint(a, "the native Intel GPU engine has no backend for {s} checkpoints yet (it reads {s})", .{ o.model_type, served });
    return null;
}

fn openFamily(comptime S: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse (if (native > 0) @min(native, default_context) else default_context);
    if (window <= 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside this model's {d}-token window", .{ window, native });
        return null;
    }
    const h = try gpa.create(Host(S));
    h.gpa = gpa;
    h.driver = xpu.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "cannot open the Level Zero driver ({s})", .{@errorName(e)});
        gpa.destroy(h);
        return null;
    };
    h.ctx = xpu.Context.init(&h.driver, o.device) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no Intel GPU device ({s})", .{@errorName(e)});
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    h.engine = (try S.init(a, io, &h.ctx, o, @intCast(window), problem)) orelse {
        h.ctx.deinit();
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    h.backend = try lane.Xpu(S).init(gpa, h.engine);
    h.backend.halt = &stopping;
    h.backend.started = chainStopHandlers;
    try h.backend.measure();
    if (h.backend.facts().exact_width > 1) h.backend.streams = @max(1, o.lanes); // drafted windows: up to --parallel streams a round
    const facts = h.backend.facts();
    h.cfg = try lanes.Config.init(gpa, facts, facts.exact_width, facts.exact_width - 1);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.backend.backend(), h.clock.clock());
    const note = try S.note(a, h.engine);
    h.startup = try std.fmt.allocPrint(gpa, "Intel GPU engine: {s}, {d:.2} GB on the device, context {d}{s}; {d} stream(s) a round on the lane core, sampling on the host", .{ h.ctx.name(), @as(f64, @floatFromInt(xpu.rt.alloc_now)) / 1e9, window, note, h.backend.streams });
    const drafted = facts.exact_width > 1;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = S.name, .lanes = h.backend.streams, .context_window = @intCast(window), .context_fitted = true, .startup = h.startup, .prefill_step = if (drafted) S.promptRows(h.engine) else 0 });
    // kept prompt states (with sequences): --prompt-cache-gib, else 4 GiB, inside the device memory left past 2 GiB
    const want: u64 = @intFromFloat((o.prompt_cache_gib orelse 4) * (1 << 30));
    const free = freeMemory(&h.ctx);
    const budget: u64 = if (free > 2 << 30) @min(want, free - (2 << 30)) else 0;
    h.cache = null;
    if (drafted and budget > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, h.backend.snapshots(), .{ .lookahead = 1, .planned = true }, budget);
        h.host.cache = &h.cache.?;
    }
    h.host.memory = .{ .read = readMemory };
    h.host.explain = .{ .text = lane.explain };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host(S).close, .ctx = h };
}
