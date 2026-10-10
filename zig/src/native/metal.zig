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
const Slide = nemotron_slide.Adapter(nemotron.slide);
const cache_fit = @import("cache_fit.zig");

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
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: a zero budget
    learned: ?api.prompt_imprint.Imprint = null, // --learn: shared states on disk, read back by later servers
    round: nemotron.gpu_round.Options = .{ .depth = 8 }, // a lone greedy stream's GPU-side rounds, as the CLI runs them
    slide: ?*Slide = null, // Sliding Weights' learner, with --slide

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
        if (h.cache) |*store| store.deinit();
        if (h.learned) |*m| m.deinit();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.metal.deinit();
        h.m.deinit();
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when no Metal engine reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const dense = std.mem.eql(u8, o.model_type, "qwen3_5") and qwen27Engine(a, io, o.dir);
    if (o.drafter != null and !dense) {
        problem.* = "--drafter is supported for Qwen3.8-27B-class checkpoints (qwen3_5 with its own output head)";
        return null;
    }
    if (dense) return openQwen27(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "qwen4_exp")) return openFlashNext(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "glm5_next")) return openGlm(a, gpa, io, o, problem);
    if (o.vision != null) {
        problem.* = try std.fmt.allocPrint(a, "--vision: the native engine reads images for GLM-5.3-Flash only, not {s} checkpoints", .{o.model_type});
        return null;
    }
    if (std.mem.eql(u8, o.model_type, "qwen3_5")) return @import("qwen35.zig").open(a, gpa, io, o, problem);
    if (!std.mem.eql(u8, o.model_type, "nemotron_h")) {
        problem.* = try std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{o.model_type});
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
        .batch_rows = @min(128, @max(64, 4 * o.lanes)), // four rows a lane, so 32 and 64 sessions keep their drafts
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
    h.cache = null;
    var why: []const u8 = "";
    const plan = cache_fit.budget(o.prompt_cache_gib, o.prompt_cache_over_cap, @min(h.m.device.maxWorkingSet() -| h.m.device.allocated() -| (8 << 30), 16 << 30), a, &why, "") catch |e| {
        if (e == error.CacheOverCap) problem.* = why;
        return e;
    };
    h.host.info_.prompt_cache_plan = plan;
    if (plan.budget_bytes > 0) {
        const S = nemotron.snapshot.Cached;
        const L = nemotron.learned_prompts;
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = S.snapBytes, .save = S.snapSave, .restore = S.snapRestore, .drop = S.snapDrop, .write = L.snapWrite, .read = L.snapRead, .forget = L.snapForget, .forget_checked = api.prompt_cache.singleForgetChecked, .reclaim = api.prompt_cache.singleReclaim } }, .{ .lookahead = 1, .planned = true }, plan.budget_bytes);
        h.host.cache = &h.cache.?; // the lane host's store: a Sliding Weights step that moves weights clears it
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.learned = null;
    if (o.learn) |root| {
        h.learned = learnedStates(a, io, o.dir, h.metal, root, @intFromFloat(o.learn_gib * (1 << 30))) catch |e| {
            if (e == error.PromptTooLong) problem.* = "--learn keys its states with a 4,143-token prompt at startup: serve with --context 4096 or more, or without --learn";
            return e;
        };
        h.learned.?.admission.floor = @intFromFloat(o.learn_min_free_gib * (1 << 30));
        if (h.cache) |*store| store.imprint = &h.learned.?;
    }
    errdefer if (h.learned) |*im| im.deinit();
    if (h.metal.head != null) h.host.lone = .{ .ctx = h, .run = Host.lone };
    h.warm = .{ .queue = h.m.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    h.slide = null;
    if (o.slide) {
        const s = try gpa.create(Slide);
        errdefer gpa.destroy(s);
        s.* = Slide.init(gpa, io, h.metal);
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

/// Learned states under `root` by checkpoint, shards, prefill step, head, kernels, probe bits, OS build and chip.
fn learnedStates(a: Allocator, io: std.Io, dir: []const u8, b: *nemotron.backend.Metal, root: []const u8, cap: u64) !api.prompt_imprint.Imprint {
    var h = std.hash.Wyhash.init(0x6e65);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        h.update(try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)));
    }
    try shardStamps(&h, a, io, dir);
    if (b.head != null) h.update(try safetensorsHeader(a, dir, "mtp-4bit.safetensors"));
    const probe = try nemotron.learned_prompts.probe(b);
    for ([_]u64{ b.o.chunk, @intFromBool(b.head != null), nemotron.kernels.sourceHash(), probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(b.m.device.name()));
    const m = try api.prompt_imprint.Imprint.open(b.gpa, root, h.final(), cap);
    std.log.info("nemotron: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ m.dir, m.metas.items.len, m.total() >> 20, cap >> 20 });
    return m;
}

/// Each safetensors file's name, size, inode and change times: a weight written in place (--slide) moves the identity.
fn shardStamps(h: *std.hash.Wyhash, a: Allocator, io: std.Io, dir: []const u8) !void {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items) |name| {
        const info = try d.statFile(io, name, .{});
        const stamp = [4]i128{ info.size, info.inode, info.mtime.nanoseconds, info.ctime.nanoseconds };
        h.update(name);
        h.update(std.mem.sliceAsBytes(&stamp));
    }
}

/// A safetensors file's header (its tensors' names, shapes and offsets), in `a`.
fn safetensorsHeader(a: Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var n: u64 = 0;
    if (!api.prompt_imprint.readAt(fd, std.mem.asBytes(&n), 0) or n > 1 << 24) return error.BadHeader;
    const head = try a.alloc(u8, @intCast(n));
    if (!api.prompt_imprint.readAt(fd, head, 8)) return error.BadHeader;
    return head;
}

/// A qwen3_5 checkpoint opens the 27B engine unless it has the pinned 2B's hidden size or a tied head, which stay on the 2B's.
fn qwen27Engine(a: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return true;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return true;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return true;
    if (doc != .object) return true;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    for ([_]std.json.Value{ doc, text }) |o| {
        if (o.object.get("tie_word_embeddings")) |v| if (v == .bool and v.bool) return false;
        if (o.object.get("hidden_size")) |v| if (v == .integer and v.integer == tf.qwen35.config.hidden) return false;
    }
    return true;
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
        problem.* = tf.flashnext_engine.quantizationProblem(e) orelse if (e == error.CacheOverCap) why else if (linkProblem(e)) |link| try std.fmt.allocPrint(a, "the native Flash Next engine cannot load {s}: {s}", .{ o.dir, link }) else try std.fmt.allocPrint(a, "the native Flash Next engine cannot load {s} with {s} ({s})", .{ o.dir, dump orelse "(none: the checked-in kernels)", @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = flashnext.close, .ctx = h };
}

test {
    _ = flashnext;
    _ = glm;
    _ = @import("cache_fit.zig");
}

/// What a two-Mac link that never connected asks of whoever starts the servers; null for any other error.
fn linkProblem(e: anyerror) ?[]const u8 {
    return if (e == error.ConnectFailed) "the other Mac never answered on the link (ConnectFailed). Start both ranks with their own settings file, each link's via naming the other Mac's address on the cable, and run both attached to a session or with Local Network access allowed in System Settings: macOS drops the packets of a server started detached, such as with nohup over SSH" else null;
}

test linkProblem {
    try std.testing.expect(std.mem.indexOf(u8, linkProblem(error.ConnectFailed).?, "Local Network") != null);
    try std.testing.expect(linkProblem(error.OutOfMemory) == null);
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
    if (o.vision != null and o.speed_up != null) {
        problem.* = try std.fmt.allocPrint(a, "--vision is served on one Mac; the --speed-up pair reads text only", .{});
        return null;
    }
    const h = glm.open(gpa, io, o.dir, @intCast(window), o.speed_up, o.lanes, o.lanes_fixed, o.prompt_cache_gib, o.learn, @intFromFloat(o.learn_gib * (1 << 30)), @intFromFloat(o.learn_min_free_gib * (1 << 30)), o.vision) catch |e| {
        if (e == error.BadLoadLimit) problem.* = "GLM_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 70% of RAM" else problem.* = if (linkProblem(e)) |link| try std.fmt.allocPrint(a, "the native GLM-5.3-Flash engine cannot load {s}: {s}", .{ o.dir, link }) else try std.fmt.allocPrint(a, "the native GLM-5.3-Flash engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
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

test "a qwen3_5 checkpoint opens the 27B engine unless it has the 2B's geometry or a tied head" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expect(qwen27Engine(a, io, dir)); // no config.json: the 27B's loader explains the refusal
    const cases = [_]struct { config: []const u8, big: bool }{
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"text_config\":{}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{\"tie_word_embeddings\":false}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":false,\"hidden_size\":2048}}", .big = false },
    };
    for (cases) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config });
        try std.testing.expectEqual(c.big, qwen27Engine(a, io, dir));
    }
}

test "a shard rewritten in place, same size and bytes, gives the learned states a new identity" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = "weights" });
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "not a shard" });
    var first = std.hash.Wyhash.init(0);
    try shardStamps(&first, a, io, dir);
    var again = std.hash.Wyhash.init(0);
    try shardStamps(&again, a, io, dir);
    try std.testing.expectEqual(first.final(), again.final());
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "other text" }); // not a shard: no change
    var notes = std.hash.Wyhash.init(0);
    try shardStamps(&notes, a, io, dir);
    try std.testing.expectEqual(first.final(), notes.final());
    std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model-00001-of-00001.safetensors", .{dir}, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, 7), std.c.pwrite(fd, "weights", 7, 0)); // as Sliding Weights writes a tensor of the same size
    _ = std.c.close(fd);
    var moved = std.hash.Wyhash.init(0);
    try shardStamps(&moved, a, io, dir);
    try std.testing.expect(first.final() != moved.final());
}
