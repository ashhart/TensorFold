//! The Metal engines: lane core (Nemotron, Qwen3.5-2B, GLM), serial host (Qwen3.8-27B), Flash Next replay.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const nemotron = tf.nemotron;
const Allocator = std.mem.Allocator;
const qwen27 = @import("qwen27_host.zig");
const flashnext = @import("flashnext_host.zig");
const glm = @import("glm_host.zig");
const nemotron_slide = @import("nemotron_slide.zig");

pub const backends: []const []const u8 = &.{"metal"};
pub const families: []const api.Family = &.{
    .{ .model_type = "nemotron_h", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "qwen4_exp", .formats = &.{"mlx-q6g32"} },
    .{ .model_type = "glm5_next", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "qwen3_5", .formats = &.{"mlx-q4g64"} },
};

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
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    cfg: lanes.Config,
    clock: nemotron.timing.RoundClock,
    core: lanes.Engine,
    host: api.LaneHost,
    round: nemotron.gpu_round.Options = .{ .depth = 8 }, // a lone greedy stream's GPU-side rounds, as the CLI runs them
    slide: ?*nemotron_slide.Adapter = null, // Sliding Weights' learner, with --slide

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
        if (h.slide) |s| {
            s.deinit();
            h.gpa.destroy(s);
        }
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.metal.deinit();
        h.m.deinit();
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when no Metal engine reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const dense = std.mem.eql(u8, o.model_type, "qwen3_5") and !tiedHead(a, io, o.dir);
    if (o.drafter != null and !dense) {
        problem.* = "--drafter is supported for Qwen3.8-27B-class checkpoints (qwen3_5 with its own output head)";
        return null;
    }
    if (dense) return openQwen27(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "qwen4_exp")) return openFlashNext(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "glm5_next")) return openGlm(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "qwen3_5")) return @import("qwen35.zig").open(a, gpa, io, o, problem);
    if (!std.mem.eql(u8, o.model_type, "nemotron_h")) {
        problem.* = try std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
        return null;
    }
    if (o.slide) if (try tf.shard_edit.linked(a, io, o.dir)) |name| {
        problem.* = try std.fmt.allocPrint(a, "--slide rewrites the model's own files, and {s} in {s} is a link to data another file shares (a Hugging Face cache's blob, or a hard link), so learning would change that file too: serve a copy made with `cp -cRL` (an APFS clone that follows links)", .{ name, o.dir });
        return null;
    };
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
    if (h.metal.head != null) try nemotron.timing.measure(h.metal, io);
    const rows: u32 = if (h.metal.head != null) nemotron.backend.max_window else 1;
    h.cfg = try lanes.Config.init(gpa, h.metal.facts(), rows, rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .b = h.metal, .wall = .{ .io = io } };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(@max(window, 0)), .prefill_step = step });
    h.round = .{ .depth = 8 };
    if (h.metal.head != null) h.host.lone = .{ .ctx = h, .run = Host.lone };
    h.warm = .{ .queue = h.m.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    h.slide = null;
    if (o.slide) {
        const s = try gpa.create(nemotron_slide.Adapter);
        errdefer gpa.destroy(s);
        s.* = nemotron_slide.Adapter.init(gpa, io, h.metal);
        h.slide = s;
        h.host.learner = s.hook();
    }
    errdefer if (h.slide) |s| {
        s.deinit();
        gpa.destroy(s);
    };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// qwen3_5 checkpoints tie their head to the embedding (Qwen3.5-2B) or keep their own (Qwen3.8-27B).
fn tiedHead(a: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return false;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return false;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return false;
    if (doc != .object) return false;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    for ([_]std.json.Value{ doc, text }) |o| if (o.object.get("tie_word_embeddings")) |v| if (v == .bool and v.bool) return true;
    return false;
}

/// Qwen3.8-27B on the serial host: DFlash2 drafts from --drafter, prompt reuse, a resident weight set.
fn openQwen27(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.speed_up != null) {
        problem.* = "speed-up mode is not available for Qwen3.8-27B yet";
        return null;
    }
    const context = o.context orelse 32768;
    if (context <= 0 or context > 262144) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside Qwen3.8-27B's 1 to 262,144 tokens", .{context});
        return null;
    }
    const h = qwen27.open(gpa, io, o.dir, @intCast(context)) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Qwen3.8-27B engine cannot load {s} ({s})", .{ o.dir, @errorName(err) });
        return null;
    };
    if (o.drafts) if (o.drafter) |dir| qwen27.enableDraft(h, io, dir, o.drafter_bits) catch |err| {
        qwen27.close(h);
        problem.* = try std.fmt.allocPrint(a, "the native Qwen3.8-27B drafter cannot load {s} ({s})", .{ dir, @errorName(err) });
        return null;
    };
    var why: []const u8 = "";
    qwen27.enableCache(h, o.prompt_cache_gib, o.prompt_cache_over_cap, a, &why) catch |err| {
        qwen27.close(h);
        problem.* = if (err == error.CacheOverCap) why else try std.fmt.allocPrint(a, "the native Qwen3.8-27B prompt cache cannot start ({s})", .{@errorName(err)});
        return null;
    };
    qwen27.holdResident(h) catch |err| std.log.warn("qwen27: no residency set ({s}); the first request after an idle second re-wires the weights", .{@errorName(err)});
    return .{ .engine = h.engine(), .close = qwen27.close, .ctx = h };
}

/// Flash Next uses the dump named by TF_FLASHNEXT_DUMP, or the checked-in kernels when that variable is unset.
fn openFlashNext(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const dump: ?[]const u8 = if (std.c.getenv("TF_FLASHNEXT_DUMP")) |d| std.mem.span(d) else null;
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
        problem.* = tf.flashnext_engine.quantizationProblem(e) orelse if (e == error.CacheOverCap) why else try std.fmt.allocPrint(a, "the native Flash Next engine cannot load {s} with {s} ({s})", .{ o.dir, dump orelse "(none: the checked-in kernels)", @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = flashnext.close, .ctx = h };
}

test {
    _ = flashnext;
    _ = @import("cache_fit.zig");
}

/// GLM-5.3-Flash's caches hold this many tokens unless --context asks otherwise (its window is 1,048,576).
const glm_window: i64 = 131072;

fn openGlm(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse @min(glm_window, if (native > 0) native else glm_window);
    if (window <= 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ window, native });
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const h = glm.open(gpa, io, o.dir, @intCast(window), o.speed_up, o.lanes, o.lanes_fixed, o.prompt_cache_gib, o.learn, @intFromFloat(o.learn_gib * (1 << 30))) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native GLM-5.3-Flash engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = glm.close, .ctx = h };
}

test "chip classes from Metal device names" {
    try std.testing.expectEqual(@as(?u32, 5), generation("Apple M5 Max"));
    try std.testing.expectEqual(@as(?u32, 12), generation("Apple M12"));
    try std.testing.expectEqual(@as(?u32, null), generation("AMD Radeon Pro"));
    try std.testing.expectEqual(@as(?u32, null), generation("Apple Mx"));
}

test "a qwen3_5 checkpoint with its own output head opens the 27B engine; a tied head stays on the 2B's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expect(!tiedHead(a, io, dir)); // no config.json: not tied, so the 27B's loader explains the refusal
    const cases = [_]struct { config: []const u8, tied: bool }{
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true}}", .tied = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"text_config\":{}}", .tied = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{\"tie_word_embeddings\":false}}", .tied = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120}", .tied = false },
    };
    for (cases) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config });
        try std.testing.expectEqual(c.tied, tiedHead(a, io, dir));
    }
}
