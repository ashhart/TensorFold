// A native server opens one Qwen3.5-2B dense model on Vulkan behind the lane host, one stream at a time, greedy only.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const fam = @import("qwen35_vulkan"); // the family's thin vulkan root: backend and engine
const sh = @import("qwen35_shaders"); // the build's compiled SPIR-V images, one decl per shader
const Allocator = std.mem.Allocator;

/// Every fence wait inside the engine is bound by this; the probes serve with the same bound.
const timeout_ns: u64 = 10 * std.time.ns_per_s;

pub const backends: []const []const u8 = &.{"vulkan"};
pub const serves: []const []const u8 = &.{"--backend=vulkan"};
pub const families: []const api.Family = &.{
    .{ .model_type = "qwen3_5", .formats = &.{"mlx-q4g64"} },
};

// GPU family gates name a chip class; Vulkan names no chip; `capabilities` reports none; all eligible devices open.
pub fn chip(a: Allocator) ?[]const u8 {
    _ = a;
    return null;
}

/// A flag or a resource the engine cannot take: `problem` says why.
const Refused = error{Refused};

fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// Flags no family on this engine serves, refused before any device work; null when `o` may open.
pub fn refused(o: api.Open) ?[]const u8 {
    if (o.tp != 1) return "the Vulkan engine has no tensor-parallel ranks: serve without --tp";
    if (o.drafter != null) return "--drafter loads a DFlash2 drafter; the Vulkan engine drafts with no head: serve without --drafter";
    if (o.speed_up != null) return "--speed-up links two Macs: serve without --speed-up on Vulkan";
    if (o.kv8) return "--kv-cache fp8: the Vulkan engine keeps bf16 caches; serve without it";
    if (o.prompt_int8) return "--prompt-precision int8: the Vulkan engine's prompt matmuls take bf16 inputs; serve without it";
    if (o.learn != null) return "--learn keeps learned prompt states for the Mac engines; the Vulkan engine keeps none: serve without --learn";
    if (o.slide) return "--slide learns into the served weights on Metal; the Vulkan engine has no learner: serve without --slide";
    if (o.vision != null) return "--vision: the Vulkan engine reads no images; serve it without --vision";
    if (o.loops != null) return "--loops sets a looped model's recurrent passes (Dory's): the Vulkan engine has none; serve without it";
    if (o.mtp_drafts != null) return "--mtp-drafts: the Vulkan engine drafts with no head; serve without --mtp-drafts";
    if (o.mtp_confidence != null) return "--mtp-confidence tunes a draft head, and the Vulkan engine has none: serve without --mtp-confidence";
    if ((o.prompt_cache_gib orelse 0) > 0) return "--prompt-cache-gib keeps prompt states between requests; the Vulkan engine resets its state for every prompt: serve without it";
    if (o.prompt_cache_over_cap) return "--prompt-cache-over-cap lifts the Mac prompt-cache cap; the Vulkan engine keeps no prompt cache: serve without it";
    if (o.keep) |n| if (n > 0) return "--checkpoint-slots keeps prompt states; the Vulkan engine resets its state for every prompt: serve without --checkpoint-slots";
    if (o.device != null) return "--device picks among several GPUs; the Vulkan engine opens its one eligible device: serve without --device";
    if (o.cfg_enabled or o.cfg_batched) return "--cfg-* guidance belongs to the Drama 3 speech model; the Vulkan engine reads no speech model";
    return null;
}

/// The model's window (config.json's max_position_embeddings), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

// Explicit `--context` must fit model window and SDPA visible keys; by default, use model window capped at arm keys.
fn contextFor(a: Allocator, o: api.Open, native: i64, problem: *[]const u8) error{Refused}!usize {
    if (o.context) |asked| {
        if (asked < 1) return refuse(a, problem, "--context {d}: give a positive token count", .{asked});
        if (native > 0 and asked > native) return refuse(a, problem, "--context {d} exceeds this model's {d}-token window", .{ asked, native });
        if (asked > fam.backend.max_capacity) return refuse(a, problem, "--context {d}: the Vulkan attention arm sees at most {d} visible keys; serve with --context {d} or less", .{ asked, fam.backend.max_capacity, fam.backend.max_capacity });
        return @intCast(asked);
    }
    if (native < 1) return refuse(a, problem, "the checkpoint's config.json names no context window; give --context as a positive token count", .{});
    return @intCast(@min(native, fam.backend.max_capacity));
}

/// The compiled shader images the engine's ops run, one decl per field, `copy_u16` shared by ops and GDN.
fn bundle() fam.engine.ShaderBundle {
    return .{
        .ops = .{
            .qmm_bf16 = &sh.qmm_bf16,
            .dequant_bf16 = &sh.dequant_bf16,
            .gather_u32 = &sh.gather_u32,
            .gather_u16 = &sh.gather_u16,
            .cast_bf16_f32 = &sh.cast_bf16_f32,
            .cast_f32_bf16 = &sh.cast_f32_bf16,
            .copy_u32 = &sh.copy_u32,
            .copy_u16 = &sh.copy_u16,
        },
        .norm = .{
            .norm_bf16 = &sh.norm_bf16,
            .add_bf16 = &sh.add_bf16,
            .add_preserve_bf16 = &sh.add_preserve_bf16,
            .swiglu_bf16 = &sh.swiglu_bf16,
        },
        .attention = .{
            .rope_bf16 = &sh.rope_bf16,
            .sdpa_rows_bf16 = &sh.sdpa_rows_bf16,
            .gate_bf16 = &sh.gate_bf16,
        },
        .gdn = .{
            .conv_dw1d_bf16 = &sh.conv_dw1d_bf16,
            .gdn_conv_decode_bf16 = &sh.gdn_conv_decode_bf16,
            .gdn_split_bf16 = &sh.gdn_split_bf16,
            .prefill_recur_bf16 = &sh.prefill_recur_bf16,
            .gated_delta_decode_bf16 = &sh.gated_delta_decode_bf16,
            .fast_norm_gated_bf16 = &sh.fast_norm_gated_bf16,
            .copy_u16 = &sh.copy_u16,
        },
        .greedy = .{
            .finite_logits = &sh.finite_logits,
            .argreduce_chunk = &sh.argreduce_chunk,
            .argreduce_combine = &sh.argreduce_combine,
        },
    };
}

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    engine: *fam.engine.Engine,
    vk: *fam.backend.Vulkan,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.vk.deinit();
        h.engine.deinit();
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when the checkpoint or the flags are not Vulkan's.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "qwen3_5")) {
        problem.* = try std.fmt.allocPrint(a, "the native Vulkan engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
        return null;
    }
    if (refused(o)) |why| {
        problem.* = why;
        return null;
    }
    if (o.lanes > 1 and o.lanes_fixed) {
        problem.* = try std.fmt.allocPrint(a, "--parallel {d}: the Vulkan engine serves one stream; serve with --parallel 1", .{o.lanes});
        return null;
    }
    const window = contextFor(a, o, modelContext(a, io, o.dir), problem) catch |err| switch (err) {
        error.Refused => return null,
        else => |x| return x,
    };
    const engine = fam.engine.Engine.init(gpa, io, bundle(), o.dir, window, timeout_ns) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.UnsupportedModel => {
            problem.* = "the checkpoint is not the pinned Qwen3.5-2B mlx-q4g64 layout the Vulkan engine reads";
            return null;
        },
        else => {
            problem.* = try std.fmt.allocPrint(a, "the Vulkan engine cannot open a device or its weights ({s})", .{@errorName(err)});
            return null;
        },
    };
    errdefer engine.deinit();
    const vk = try fam.backend.Vulkan.init(gpa, engine);
    errdefer vk.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa,
        .engine = engine,
        .vk = vk,
        .config = try lanes.Config.init(gpa, vk.facts(), 1, 0),
        .clock = undefined,
        .core = undefined,
        .host = undefined,
    };
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.vk.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .lanes = 1,
        .context_window = @intCast(engine.capacity()),
        .prefill_step = fam.backend.max_prompt_rows,
        .greedy_only = true,
    });
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

test "the registry names qwen3_5 over mlx-q4g64 on the vulkan backend" {
    try std.testing.expectEqual(@as(usize, 1), backends.len);
    try std.testing.expectEqualStrings("vulkan", backends[0]);
    try std.testing.expectEqualStrings("qwen3_5", families[0].model_type);
    try std.testing.expectEqual(@as(usize, 1), families[0].formats.len);
    try std.testing.expectEqualStrings("mlx-q4g64", families[0].formats[0]);
    for (serves) |s| try std.testing.expect(!std.mem.eql(u8, s, "--backend=rocm"));
    var has_vulkan = false;
    for (serves) |s| has_vulkan = has_vulkan or std.mem.eql(u8, s, "--backend=vulkan");
    try std.testing.expect(has_vulkan);
}

test "flags nothing on this engine serves are refused, not ignored" {
    const base = api.Open{ .dir = "d", .model_type = "qwen3_5" };
    try std.testing.expect(refused(base) == null);
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .tp = 2 }) orelse "", "the Vulkan engine has no tensor-parallel ranks");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .drafter = "x" }) orelse "", "--drafter");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .speed_up = "r.json" }) orelse "", "--speed-up");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .kv8 = true }) orelse "", "--kv-cache fp8");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .prompt_int8 = true }) orelse "", "--prompt-precision int8");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .learn = "/x" }) orelse "", "--learn");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .slide = true }) orelse "", "--slide");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .vision = .{} }) orelse "", "--vision");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .loops = 2 }) orelse "", "--loops");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .mtp_drafts = 2 }) orelse "", "--mtp-drafts");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .mtp_confidence = 0.5 }) orelse "", "--mtp-confidence");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .prompt_cache_gib = 4 }) orelse "", "--prompt-cache-gib");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .prompt_cache_over_cap = true }) orelse "", "--prompt-cache-over-cap");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .keep = 2 }) orelse "", "--checkpoint-slots");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .device = 1 }) orelse "", "--device");
    try std.testing.expectStringStartsWith(refused(.{ .dir = "d", .model_type = "qwen3_5", .cfg_enabled = true }) orelse "", "--cfg-*");
    try std.testing.expect(refused(.{ .dir = "d", .model_type = "qwen3_5", .prompt_cache_gib = 0 }) == null);
}

test "the context window clamps to the model's own and the arm's visible keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expectEqual(@as(usize, 256), try contextFor(a, .{ .dir = "d", .model_type = "qwen3_5", .context = 256 }, 4096, &problem));
    try std.testing.expectEqual(@as(usize, 4096), try contextFor(a, .{ .dir = "d", .model_type = "qwen3_5" }, 4096, &problem));
    try std.testing.expectEqual(@as(usize, fam.backend.max_capacity), try contextFor(a, .{ .dir = "d", .model_type = "qwen3_5", .context = fam.backend.max_capacity }, 0, &problem));
    try std.testing.expectError(error.Refused, contextFor(a, .{ .dir = "d", .model_type = "qwen3_5", .context = 8000 }, 4096, &problem));
    try std.testing.expectError(error.Refused, contextFor(a, .{ .dir = "d", .model_type = "qwen3_5" }, 0, &problem));
    try std.testing.expectError(error.Refused, contextFor(a, .{ .dir = "d", .model_type = "qwen3_5", .context = fam.backend.max_capacity + 1 }, 0, &problem));
}

test "the pinned checkpoint's wide native window clamps by default, and an explicit over-cap ask is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    // config.json names 262144 (the pinned checkpoint's): a default open serves the arm's 7168.
    try std.testing.expectEqual(@as(usize, fam.backend.max_capacity), try contextFor(a, .{ .dir = "d", .model_type = "qwen3_5" }, 262144, &problem));
    try std.testing.expectError(error.Refused, contextFor(a, .{ .dir = "d", .model_type = "qwen3_5", .context = 262144 }, 262144, &problem));
}

test "the core's production bounds helpers gate prompt and capacity" {
    try std.testing.expect(fam.backend.validPrompt(&.{1}));
    try std.testing.expect(!fam.backend.validPrompt(&.{}));
    try fam.engine.planFor(11, 0, 4, 8); // four prompt rows and seven decodes
    try std.testing.expectError(error.ContextTooLong, fam.engine.planFor(11, 0, 4, 9)); // one past
}
