//! The Metal engines: lane core (Nemotron, Qwen3.5-2B, LFM2-MoE, GLM, Qwen3.8-27B), Flash Next replay.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const nemotron = tf.nemotron;
const Allocator = std.mem.Allocator;
const qwen27 = @import("qwen27_served.zig");
const flashnext = @import("flashnext_host.zig");
const glm = @import("glm_host.zig");
const glm_dsa = @import("glm_dsa_host.zig");
const slide_adapter = @import("slide_adapter.zig");
const Slide = slide_adapter.Adapter(nemotron.slide);
const cache_fit = @import("cache_fit.zig");
const nemo_disk = @import("nemotron_disk.zig");

pub const backends: []const []const u8 = &.{"metal"};

/// Every checkpoint-wide MLX affine width the per-module readers take (oQ quants): 2-8 bits but 7, groups 32-128.
const affine_widths = blk: {
    var out: [18][]const u8 = undefined;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }, 0..) |bits, i| for ([_]u16{ 32, 64, 128 }, 0..) |group, j| {
        out[3 * i + j] = std.fmt.comptimePrint("mlx-q{d}g{d}", .{ bits, group });
    };
    const done = out;
    break :blk &done;
};

pub const families: []const api.Family = &.{
    .{ .model_type = "nemotron_h", .formats = affine_widths },
    .{ .model_type = "qwen4_exp", .formats = affine_widths },
    .{ .model_type = "glm5_next", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "qwen3_5", .formats = affine_widths },
    .{ .model_type = "lfm2_moe", .formats = &.{ "mlx-q4g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "glm_moe_dsa", .formats = &.{"exlr8"} },
};

/// The chip class gate entries name ("apple-m5" for an Apple M5 Max); null without an Apple GPU.
pub fn chip(a: Allocator) ?[]const u8 {
    const device = mtl.Device.init() catch return null;
    defer device.deinit();
    const gen = generation(std.mem.span(device.name())) orelse return null;
    return std.fmt.allocPrint(a, "apple-m{d}", .{gen}) catch null;
}

/// The name of the system GPU the Metal engines open ("Apple M5 Max"); null without one.
pub fn deviceName(a: Allocator, _: ?u32) ?[]const u8 {
    const device = mtl.Device.init() catch return null;
    defer device.deinit();
    return a.dupe(u8, std.mem.span(device.name())) catch null;
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

/// The host's memory counts: `cache_fit.hostMemory` from the process footprint (no configured load limit).
fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    return cache_fit.hostMemory(0, reset_peak);
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
        const rows = if (h.cache) |*store| store.disk else null;
        if (h.cache) |*store| store.deinit();
        if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
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
    if (try refusal(a, o, dense)) |why| {
        problem.* = why;
        return null;
    }
    if (o.drafter != null and !dense) {
        problem.* = "--drafter is supported for Qwen3.8-27B-class checkpoints (qwen3_5 with its own output head)";
        return null;
    }
    if (dense) return openQwen27(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "qwen4_exp")) return openFlashNext(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "glm5_next")) return openGlm(a, gpa, io, o, problem);
    if (o.vision != null) {
        problem.* = try std.fmt.allocPrint(a, "--vision: the native engine reads images for GLM-5.3-Flash and Qwen3.8-27B only, not {s} checkpoints", .{o.model_type});
        return null;
    }
    if (std.mem.eql(u8, o.model_type, "glm_moe_dsa")) return openGlmDsa(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "qwen3_5")) return @import("qwen35.zig").open(a, gpa, io, o, problem);
    if (std.mem.eql(u8, o.model_type, "lfm2_moe")) return @import("lfm2_moe.zig").open(a, gpa, io, o, problem);
    if (!std.mem.eql(u8, o.model_type, "nemotron_h")) {
        problem.* = try std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{o.model_type});
        return null;
    }
    if (o.slide) if (try slide_adapter.refusal(tf.shard_edit, a, io, o.dir, "`cp -cRL` (an APFS clone that follows links)")) |why| {
        problem.* = why;
        return null;
    };
    if (o.slide) if (try slideWidths(a, io, o.dir)) |why| {
        problem.* = why;
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
    nemotron.slide.loadLiving(gpa, io, h.metal) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native engine cannot read the saved lessons in {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    if (h.metal.head != null) try nemotron.timing.measure(h.metal, io);
    const rows: u32 = if (h.metal.head != null) nemotron.backend.max_window else 1;
    h.cfg = try lanes.Config.init(gpa, h.metal.facts(), rows, rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .b = h.metal, .wall = .{ .io = io } };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(@max(window, 0)), .prefill_step = step, .structures = true });
    h.host.memory = .{ .ctx = h, .read = readMemory };
    h.round = .{ .depth = 8 };
    h.cache = null;
    var why: []const u8 = "";
    const plan = cache_fit.budget(o.prompt_cache_gib, o.prompt_cache_over_cap, h.m.device.maxWorkingSet() -| h.m.device.allocated() -| (8 << 30), a, &why, "") catch |e| {
        if (e == error.CacheOverCap) problem.* = why;
        return e;
    };
    h.host.info_.prompt_cache_plan = plan;
    if (plan.budget_bytes > 0) {
        const S = nemotron.snapshot.Cached;
        const L = nemotron.learned_prompts;
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = S.snapBytes, .save = S.snapSave, .restore = S.snapRestore, .drop = S.snapDrop, .write = L.snapWrite, .read = L.snapRead, .forget = L.snapForget, .forget_checked = api.prompt_cache.singleForgetChecked, .reclaim = api.prompt_cache.singleReclaim, .rows = &nemo_disk.image } }, .{ .lookahead = 1, .planned = true }, plan.budget_bytes);
        h.host.cache = &h.cache.?; // the lane host's store: a Sliding Weights step that moves weights clears it
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.learned = null;
    if (o.learn) |root| {
        const id = learnedIdentity(a, io, o.dir, h.metal) catch |e| {
            if (e == error.PromptTooLong) problem.* = "--learn keys its states with a 4,143-token prompt at startup: serve with --context 4096 or more, or without --learn";
            return e;
        };
        if (o.learn_conversations) {
            if (h.cache) |*store| {
                const r = try api.prompt_cache.rows.Rows.open(gpa, store.family, .{ .root = root, .identity = id.id, .cap = gibBytes(o.learn_gib), .floor = gibBytes(o.learn_min_free_gib), .daily = gibBytes(o.learn_daily_gib), .lookahead = store.rules.lookahead, .planned = store.rules.planned });
                r.prefill_rate = id.rate;
                store.disk = r;
            }
        } else {
            h.learned = try api.prompt_imprint.Imprint.open(h.metal.gpa, root, id.id, gibBytes(o.learn_gib));
            std.log.info("nemotron: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ h.learned.?.dir, h.learned.?.metas.items.len, h.learned.?.total() >> 20, gibBytes(o.learn_gib) >> 20 });
            h.learned.?.admission.floor = gibBytes(o.learn_min_free_gib);
            if (h.cache) |*store| store.imprint = &h.learned.?;
        }
    }
    errdefer if (h.learned) |*im| im.deinit();
    errdefer if (h.cache) |*store| if (store.disk) |r| r.deinit();
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

/// --slide writes 4-bit codes into the shards: a checkpoint with any other width is refused by its first such matrix.
fn slideWidths(a: Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    const c = nemotron.config.Config.read(a, io, dir) catch return null; // the load names a config.json it cannot read
    var buf: [96]u8 = undefined;
    const m = c.other(&buf) orelse return null;
    return try std.fmt.allocPrint(a, "--slide learns into checkpoints at 4-bit in groups of 64, and {s} in {s} is {d}-bit in groups of {d}: serve it without --slide", .{ m.name, dir, m.width.bits, m.width.group });
}

/// Flash Next's replay engine copies on the GPU from its own history: it reads no Recall record yet.
const recall_refusal = "--recall drafts from a record in the lane engine's rounds; " ++
    "Flash Next's replay engine copies from its own history on the GPU and reads no record yet: " ++
    "serve it without --recall, --recall-dir, --recall-gib or --recall-read-only";

/// Refused before any weights load: --slide but on Nemotron, as CUDA does, and --learn where nothing is kept on disk.
fn refusal(a: Allocator, o: api.Open, dense: bool) !?[]const u8 {
    const t = o.model_type;
    if (o.slide and !std.mem.eql(u8, t, "nemotron_h")) return try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on Macs yet", .{t});
    const one_mac: ?[]const u8 = if (std.mem.eql(u8, t, "nemotron_h")) "Nemotron" else if (std.mem.eql(u8, t, "lfm2_moe")) "LFM2.5-24B-A2B" else if (!std.mem.eql(u8, t, "qwen3_5")) null else if (dense) "Qwen3.8-27B" else "Qwen3.5-2B";
    if (o.speed_up != null) if (one_mac) |name| return try std.fmt.allocPrint(a, "--speed-up joins two Macs for Flash Next, GLM-5.3-Flash and GLM-5.3 only: serve {s} without it", .{name});
    if (o.recall and std.mem.eql(u8, t, "qwen4_exp")) return recall_refusal;
    if (o.learn == null) return null;
    // the 27B writes conversation states
    if (o.learn_conversations and std.mem.eql(u8, t, "qwen3_5") and dense) return null;
    const name: []const u8 = if (std.mem.eql(u8, t, "glm_moe_dsa")) "GLM-5.3 (EXLR8)" else if (std.mem.eql(u8, t, "lfm2_moe")) "LFM2.5-24B-A2B" else if (!std.mem.eql(u8, t, "qwen3_5")) return null else if (dense) "Qwen3.8-27B" else "Qwen3.5-2B";
    if (o.learn_conversations) return try std.fmt.allocPrint(a, "--learn-conversations keeps conversation states on disk for Qwen3.8-27B, Nemotron, Flash Next and GLM-5.3-Flash: serve {s} without it", .{name});
    const or_talks: []const u8 = if (dense) " (Qwen3.8-27B keeps conversation states on disk with --learn-conversations)" else "";
    return try std.fmt.allocPrint(a, "--learn learns prompt prefixes for Nemotron, Flash Next and GLM-5.3-Flash only{s}: serve {s} without --learn, --learn-dir or --learn-gib", .{ or_talks, name });
}

/// What learned and disk states read under: checkpoint, shards, prefill step, head, kernels, a probe, OS build, chip.
fn learnedIdentity(a: Allocator, io: std.Io, dir: []const u8, b: *nemotron.backend.Metal) !struct { id: u64, rate: f64 } {
    var h = std.hash.Wyhash.init(0x6e65);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        h.update(try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)));
    }
    try shardStamps(&h, a, io, dir);
    if (b.head != null) h.update(try safetensorsHeader(a, dir, "mtp-4bit.safetensors"));
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const probe = try nemotron.learned_prompts.probe(b);
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    for ([_]u64{ b.o.chunk, @intFromBool(b.head != null), nemotron.kernels.sourceHash(), probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(b.m.device.name()));
    return .{ .id = h.final(), .rate = 4143.0 / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
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

/// Qwen3.8-27B on the lane host: `--parallel` streams in shared rounds, DFlash2 drafts from --drafter for one of them.
fn openQwen27(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const context = o.context orelse 32768;
    if (context <= 0 or context > 262144) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside Qwen3.8-27B's 1 to 262,144 tokens", .{context});
        return null;
    }
    if (o.drafts and o.drafter != null and o.drafter_bits != 0 and o.drafter_bits != 4) {
        problem.* = "--drafter-bits takes 0 (bf16) or 4";
        return null;
    }
    var why: []const u8 = "";
    const drafter = if (o.drafts) o.drafter else null;
    const learn: ?@import("qwen27_disk.zig").Learn = if (o.learn_conversations) if (o.learn) |root| .{ .root = root, .dir = o.dir, .drafter = drafter, .cap = gibBytes(o.learn_gib), .floor = gibBytes(o.learn_min_free_gib), .daily = gibBytes(o.learn_daily_gib) } else null else null;
    const h = qwen27.open(gpa, io, o.dir, .{ .context = @intCast(context), .streams = o.lanes, .streams_fixed = o.lanes_fixed, .drafter = drafter, .drafter_bits = o.drafter_bits, .cache_gib = o.prompt_cache_gib, .cache_over_cap = o.prompt_cache_over_cap, .learn = learn, .vision = o.vision }, a, &why) catch |err| {
        problem.* = if (why.len > 0) why else try std.fmt.allocPrint(a, "the native Qwen3.8-27B engine cannot load {s} ({s})", .{ o.dir, @errorName(err) });
        return null;
    };
    return .{ .engine = h.engine(), .close = qwen27.close, .ctx = h };
}

/// A GiB count's bytes (the flags allow 0 up to below 2^34).
fn gibBytes(g: f64) u64 {
    return @intFromFloat(g * (1 << 30));
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
    const learn: ?flashnext.Learn = if (o.learn) |root| .{ .root = root, .cap = gibBytes(o.learn_gib), .floor = gibBytes(o.learn_min_free_gib), .daily = gibBytes(o.learn_daily_gib), .conversations = o.learn_conversations } else null;
    const h = flashnext.open(gpa, io, o.dir, dump, window, o.speed_up, o.prompt_cache_gib, o.prompt_cache_over_cap, learn, a, &why) catch |e| {
        problem.* = tf.flashnext_engine.quantizationProblem(e) orelse if (e == error.CacheOverCap or e == error.LearnOnPair) why else if (linkProblem(e)) |link| try std.fmt.allocPrint(a, "the native Flash Next engine cannot load {s}: {s}", .{ o.dir, link }) else try std.fmt.allocPrint(a, "the native Flash Next engine cannot load {s} with {s} ({s})", .{ o.dir, dump orelse "(none: the checked-in kernels)", @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = flashnext.close, .ctx = h };
}

test {
    _ = flashnext;
    _ = glm;
    _ = glm_dsa;
    _ = qwen27;
    _ = @import("cache_fit.zig");
    _ = @import("resident.zig");
    _ = @import("qwen27_landing.zig");
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
    const learn: ?glm.Learn = if (o.learn) |root| .{ .root = root, .cap = gibBytes(o.learn_gib), .floor = gibBytes(o.learn_min_free_gib), .daily = gibBytes(o.learn_daily_gib), .conversations = o.learn_conversations } else null;
    const h = glm.open(gpa, io, o.dir, @intCast(window), o.speed_up, o.lanes, o.lanes_fixed, o.prompt_cache_gib, learn, o.vision) catch |e| {
        if (e == error.LearnConversationsOnPair) problem.* = "--learn-conversations keeps states on one Mac only: serve GLM-5.3-Flash without --speed-up, or without the flag" else if (e == error.BadLoadLimit) problem.* = "GLM_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 70% of RAM" else problem.* = if (linkProblem(e)) |link| try std.fmt.allocPrint(a, "the native GLM-5.3-Flash engine cannot load {s}: {s}", .{ o.dir, link }) else try std.fmt.allocPrint(a, "the native GLM-5.3-Flash engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = glm.close, .ctx = h };
}

/// GLM-5.3 (EXLR8) needs a pair: each Mac holds half of every expert and attention head (--speed-up names its link).
fn openGlmDsa(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const settings = o.speed_up orelse {
        problem.* = "GLM-5.3 (EXLR8) runs on two Macs: start both with --speed-up and each Mac's link settings (rank 0 serves)";
        return null;
    };
    const window: i64 = o.context orelse 16384;
    if (window <= 0 or window > 262144) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside 1 to 262,144 tokens for GLM-5.3 (EXLR8)", .{window});
        return null;
    }
    var ram: u64 = 0;
    var len: usize = 8;
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) return error.NoRamSize;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const h = glm_dsa.open(gpa, io, o.dir, @intCast(window), settings, @max(1, @min(o.lanes, 16)), ram / 100 * 66, o.prompt_cache_gib) catch |e| {
        problem.* = if (linkProblem(e)) |link| try std.fmt.allocPrint(a, "the native GLM-5.3 (EXLR8) engine cannot load {s}: {s}", .{ o.dir, link }) else try std.fmt.allocPrint(a, "the native GLM-5.3 (EXLR8) engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = glm_dsa.close, .ctx = h };
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

test "--learn is refused for the 27B, the 2B and GLM-5.3 (EXLR8), and not for the families that keep prompt states on disk" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const learn: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .learn = "/x" };
    const flags = " without --learn, --learn-dir or --learn-gib";
    try std.testing.expectStringEndsWith((try refusal(a, learn, true)) orelse "", ": serve Qwen3.8-27B" ++ flags);
    var talks = learn;
    talks.learn_conversations = true;
    try std.testing.expect((try refusal(a, talks, true)) == null); // the 27B keeps conversation states on disk
    try std.testing.expectStringEndsWith((try refusal(a, talks, false)) orelse "", ": serve Qwen3.5-2B without it");
    try std.testing.expectStringEndsWith((try refusal(a, learn, false)) orelse "", ": serve Qwen3.5-2B" ++ flags);
    try std.testing.expectStringEndsWith((try refusal(a, .{ .dir = "d", .model_type = "glm_moe_dsa", .learn = "/x" }, false)) orelse "", ": serve GLM-5.3 (EXLR8)" ++ flags);
    for ([_][]const u8{ "qwen3_5", "glm_moe_dsa" }) |t| try std.testing.expect((try refusal(a, .{ .dir = "d", .model_type = t, .learn_gib = 8, .learn_min_free_gib = 2 }, true)) == null);
    for ([_][]const u8{ "nemotron_h", "qwen4_exp", "glm5_next" }) |t| try std.testing.expect((try refusal(a, .{ .dir = "d", .model_type = t, .learn = "/x" }, false)) == null);
}

test "--speed-up is refused by Nemotron, the 27B and the 2B before any load, and not by the two-Mac families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const two: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .speed_up = "pair.json" };
    const tail = " only: serve {s} without it";
    try std.testing.expectStringEndsWith((try refusal(a, two, true)) orelse "", try std.fmt.allocPrint(a, tail, .{"Qwen3.8-27B"}));
    try std.testing.expectStringEndsWith((try refusal(a, two, false)) orelse "", try std.fmt.allocPrint(a, tail, .{"Qwen3.5-2B"}));
    try std.testing.expectStringEndsWith((try refusal(a, .{ .dir = "d", .model_type = "nemotron_h", .speed_up = "pair.json" }, false)) orelse "", try std.fmt.allocPrint(a, tail, .{"Nemotron"}));
    for ([_][]const u8{ "qwen4_exp", "glm5_next", "glm_moe_dsa" }) |t| try std.testing.expect((try refusal(a, .{ .dir = "d", .model_type = t, .speed_up = "pair.json" }, false)) == null);
}

test "--recall is refused by Flash Next before any load, and not by the lane engine's families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const flash_next: api.Open = .{ .dir = "d", .model_type = "qwen4_exp", .recall = true };
    try std.testing.expectEqualStrings(recall_refusal, (try refusal(a, flash_next, false)).?);
    for ([_][]const u8{ "nemotron_h", "qwen3_5", "glm5_next", "glm_moe_dsa" }) |t| {
        try std.testing.expect((try refusal(a, .{ .dir = "d", .model_type = t, .recall = true }, true)) == null);
    }
}

test "--slide is refused by every family but Nemotron, which learns into its weights" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect((try refusal(a, .{ .dir = "d", .model_type = "nemotron_h", .slide = true }, false)) == null);
    for ([_][]const u8{ "glm_moe_dsa", "qwen4_exp", "glm5_next", "qwen3_5" }) |t| {
        const why = (try refusal(a, .{ .dir = "d", .model_type = t, .slide = true }, true)) orelse return error.TestExpectedRefusal;
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on Macs yet", .{t}), why);
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
