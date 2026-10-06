//! The engines a native server opens on Metal: Nemotron 4-bit (MLX affine, groups of 64) on the lane core, and Flash
//! Next 6-bit (groups of 32) on the replay engine, one reply at a time (its kernels and packs in TF_FLASHNEXT_DUMP).
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const nemotron = tf.nemotron;
const Allocator = std.mem.Allocator;
const flashnext = @import("flashnext_host.zig");

pub const backends: []const []const u8 = &.{"metal"};
pub const families: []const api.Family = &.{ .{ .model_type = "nemotron_h", .formats = &.{"mlx-q4g64"} }, .{ .model_type = "qwen4_exp", .formats = &.{"mlx-q6g32"} } };

/// The chip class gate entries name ("apple-m5" for an Apple M5 Max); null without an Apple GPU.
pub fn chip(a: Allocator) ?[]const u8 {
    const device = mtl.Device.init() catch return null;
    defer device.deinit();
    const gen = generation(std.mem.span(device.name())) orelse return null;
    return std.fmt.allocPrint(a, "apple-m{d}", .{gen}) catch null;
}

/// The M generation in a Metal device name ("Apple M5 Max": 5).
fn generation(name: []const u8) ?u32 {
    var words = std.mem.tokenizeScalar(u8, name, ' ');
    if (!std.ascii.eqlIgnoreCase(words.next() orelse return null, "apple")) return null;
    const word = words.next() orelse return null;
    if (word.len < 2 or std.ascii.toLower(word[0]) != 'm') return null;
    const gen = std.fmt.parseInt(u32, word[1..], 10) catch return null;
    return if (gen >= 1) gen else null;
}

/// Python's prompt chunk: 8,192 rows with tensor units (M5 on) where memory allows, else its 2,048.
fn prefillStep(device: mtl.Device) u32 {
    const gen = generation(std.mem.span(device.name())) orelse 0;
    return if (gen >= 5 and device.maxWorkingSet() >= 48 << 30) 8192 else 2048;
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

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    m: *nemotron.Model,
    metal: *nemotron.backend.Metal,
    cfg: lanes.Config,
    clock: nemotron.timing.RoundClock,
    core: lanes.Engine,
    host: api.LaneHost,
    round: nemotron.gpu_round.Options = .{ .depth = 8 }, // a lone greedy stream's GPU-side rounds, as the CLI runs them

    /// A lone greedy stream's rounds on the GPU until it finishes, or a hand-over to the lane core (true).
    fn lone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const h: *Host = @ptrCast(@alignCast(ctx));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const r = try nemotron.gpu_round.run(h.gpa, h.metal, s, h.round, .{ .ctx = hooks.ctx, .committed = hooks.committed, .yield = hooks.yield });
        return r.paused;
    }

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.metal.deinit();
        h.m.deinit();
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when no Metal engine reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (std.mem.eql(u8, o.model_type, "qwen4_exp")) return openFlashNext(a, gpa, io, o, problem);
    if (!std.mem.eql(u8, o.model_type, "nemotron_h")) {
        problem.* = try std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
        return null;
    }
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse native;
    if (window < 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ window, native });
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = nemotron.Model.load(gpa, io, o.dir, o.drafts) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    errdefer m.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.m = m;
    const step = prefillStep(h.m.device);
    const capacity: usize = @intCast(if (window > 0) window else 1 << 18);
    h.metal = try nemotron.backend.Metal.init(gpa, h.m, .{
        .capacity = capacity + nemotron.backend.max_lanes + 1,
        .chunk = step,
        .drafts = o.drafts,
        .streams = @max(o.lanes, 2),
        .batch_rows = 32,
    });
    errdefer h.metal.deinit();
    if (h.metal.head != null) try nemotron.timing.measure(h.metal);
    const rows: u32 = if (h.metal.head != null) nemotron.backend.max_window else 1;
    h.cfg = try lanes.Config.init(gpa, h.metal.facts(), rows, rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .b = h.metal, .wall = .{ .io = io } };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(@max(window, 0)), .prefill_step = step });
    h.round = .{ .depth = 8 };
    if (h.metal.head != null) h.host.lone = .{ .ctx = h, .run = Host.lone };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Flash Next on the replay engine: the recorded kernels and packs from tools/zig/flashnext_dump.py in TF_FLASHNEXT_DUMP.
fn openFlashNext(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const dump = std.mem.span(std.c.getenv("TF_FLASHNEXT_DUMP") orelse {
        problem.* = "the native Flash Next engine needs TF_FLASHNEXT_DUMP: a folder from tools/zig/flashnext_dump.py";
        return null;
    });
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse native;
    if (window < 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ window, native });
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var why: []const u8 = "";
    const h = flashnext.open(gpa, io, o.dir, dump, window, o.speed_up, o.prompt_cache_gib, o.prompt_cache_over_cap, a, &why) catch |e| {
        problem.* = if (e == error.CacheOverCap) why else try std.fmt.allocPrint(a, "the native Flash Next engine cannot load {s} with {s} ({s})", .{ o.dir, dump, @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = flashnext.close, .ctx = h };
}

test {
    _ = flashnext;
    _ = @import("cache_fit.zig");
}

test "chip classes from Metal device names" {
    try std.testing.expectEqual(@as(?u32, 5), generation("Apple M5 Max"));
    try std.testing.expectEqual(@as(?u32, 12), generation("Apple M12"));
    try std.testing.expectEqual(@as(?u32, null), generation("AMD Radeon Pro"));
    try std.testing.expectEqual(@as(?u32, null), generation("Apple Mx"));
}
