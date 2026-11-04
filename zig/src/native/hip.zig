//! The engines a native server opens on HIP: the GPU, its policy, the tensor-parallel group, round loop and lane host.

const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const flashnext = @import("flashnext");
const nemotron = @import("nemotron");
const serve = @import("hip_serve");
const glm_ocr = @import("glm_ocr");
const qwen3 = @import("qwen3");
const dots = @import("dots_mocr");
const fam = @import("hip_families.zig");
const Allocator = std.mem.Allocator;
const admission = hip.admission;
const modelContext = admission.modelContext;

/// The HIP families: namespaces with `model_type`, `formats`, `default_context`, `prefill_step`, `open` and `follow`.
const registry = .{ qwen35.native, qwen35.native_moe, flashnext.native, nemotron.native, glm_ocr.native, qwen3.native, dots.native };

pub const backends: []const []const u8 = &.{"hip"};
/// No per-thread alternate signal stack: its thread-local leaves RCCL's and HIP's threads too little stack to start.
pub const std_options: std.Options = .{ .signal_stack_size = null };
/// The serve flags past the common ones this backend takes (server/cli.zig lists and parses them).
pub const serves: []const []const u8 = &.{ "--tp", "--policy", "--mtp-drafts", "--mtp-confidence", "--checkpoint-slots", "--backend=rocm" };
pub const families: []const api.Family = fam.list(registry);

/// The GPU family gate entries name ("rdna2" for gfx103x, "rdna3" for gfx11 / gfx12); null without a usable GPU.
pub fn chip(a: Allocator) ?[]const u8 {
    const caps = hip.Device.capsOf(0) orelse return null;
    return a.dupe(u8, if (caps.generation == .rdna2) "rdna2" else "rdna3") catch null;
}

/// A flag or a resource the engine cannot take: `problem` says why.
const Refused = error{Refused};

fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// Why `matrix=on` cannot run on a GPU of `caps`, or null when its matrix-core tiles exist (gfx11's WMMA).
fn matrixRefused(comptime F: type, policy: hip.Policy, caps: hip.Caps) ?[]const u8 {
    if (policy.matrix != .on) return null;
    return switch (caps.matrix) {
        .wmma11 => null,
        // a family whose products are all BF16 dense ones has gfx12 tiles; the MLX affine tiles have none yet
        .wmma12 => if (@hasDecl(F, "matrix_wmma12") and F.matrix_wmma12) null else "the gfx12 matrix-core tiles (RDNA4's WMMA layout) are not implemented yet for this family",
        .none => "this GPU has no matrix cores",
    };
}

/// The tensor-parallel flags as the Python ROCm server checks them, and the flags this engine cannot take.
fn checkGroup(a: Allocator, o: api.Open, problem: *[]const u8) Refused!void {
    if (o.drafter != null) return refuse(a, problem, "--drafter: the HIP engine runs no DFlash2 drafter yet; serve without --drafter", .{});
    if (o.tp != 1 and o.tp != 2 and o.tp != 4 and o.tp != 8) return refuse(a, problem, "--tp {d} is not a supported ROCm world size; choose 1, 2, 4 or 8", .{o.tp});
    if (o.tp > 1 and o.master.len == 0) return refuse(a, problem, "--tp > 1 needs --master: rank 0's address on the link between the machines", .{});
    // the two-Mac engines take their ranks from --speed-up's settings file; this engine takes them from --tp
    if (o.speed_up != null) return refuse(a, problem, "--speed-up is the two-Mac engines' rank settings; the HIP engine takes its ranks from --tp", .{});
    if (o.tp == 1 and o.rank != 0) return refuse(a, problem, "--rank must be 0 when --tp 1", .{});
    if (o.rank >= o.tp) return refuse(a, problem, "--rank {d} not in [0, --tp {d})", .{ o.rank, o.tp });
    if (o.keep) |n| if (n < 0) return refuse(a, problem, "--checkpoint-slots must be 0 or more", .{});
    if (o.mtp_drafts) |n| if (n > 3) return refuse(a, problem, "--mtp-drafts {d}: the HIP engine drafts at most 3", .{n});
    if (o.mtp_confidence) |c| if (c < 0 or c > 1) return refuse(a, problem, "--mtp-confidence must be from 0 to 1", .{});
    if (o.prompt_cache_gib) |g| if (g < 0) return refuse(a, problem, "--prompt-cache-gib must be 0 or more", .{});
}

/// What every family needs before it loads: the card, the policy, the group, and the window and lanes asked for.
const Ready = struct {
    dev: hip.Device,
    options: Options,
    /// The policy's line, for the server's info.
    policy_line: []const u8,

    const Options = serve.Options;

    fn deinit(r: Ready, gpa: Allocator) void {
        gpa.free(r.policy_line);
        if (r.dev.group) |g| g.close(gpa);
    }
};

/// Why the HIP engines cannot keep fp8 caches yet: every family's caches stay 16-bit.
const kv8_refusal = "--kv-cache fp8 keeps fp8 caches on CUDA; the HIP engines keep 16-bit caches: serve without it";

/// The cache flags this backend cannot honour, refused on every rank before a family loads.
fn cacheRefused(o: api.Open) ?[]const u8 {
    if (o.loops != null) return "--loops sets Dory's recurrent passes, served on CUDA: the HIP engines run no looped model";
    return if (o.kv8) kv8_refusal else null;
}

/// Resolves the policy on rank 0 (defaults, flags, old variables, TF_POLICY), joins the group, sends it to the others.
fn ready(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) (Refused || Allocator.Error)!Ready {
    try checkGroup(a, o, problem);
    if (cacheRefused(o)) |why| return refuse(a, problem, "{s}", .{why});
    const native = modelContext(a, io, o.dir);
    const window = admission.contextWindow(o.context, native, F.default_context) catch |err| return switch (err) {
        error.Negative => refuse(a, problem, "--context {d}: a token count, or 0 for the model's window", .{o.context.?}),
        error.NoNative => refuse(a, problem, "--context 0 asks for the model's window, and its config.json names none: give a token count", .{}),
        error.PastNative => refuse(a, problem, "--context {d} exceeds this model's {d}-token window", .{ o.context.?, native }),
    };
    const index: c_int = if (o.device) |d| @intCast(d) else hip.Device.ordinal(o.rank);
    const caps = hip.Device.capsOf(index) orelse return refuse(a, problem, "HIP device {d} is not a GPU this engine supports", .{index});
    var notes: hip.Policy.Notes = .{};
    var policy = hip.Policy.resolveFrom(hip.Policy.forCaps(caps, o.tp), o.policy, .current, &notes) catch |err| return refuse(a, problem, "the policy \"{s}\" is refused ({s})", .{ o.policy, @errorName(err) });
    // the step came from the GPU's defaults when the plain defaults under the same flags differ: one rank may fall back
    var plain_notes: hip.Policy.Notes = .{};
    const plain = hip.Policy.resolve(o.policy, .current, &plain_notes) catch policy;
    const own_default = o.tp == 1 and plain.prefill_step != policy.prefill_step;
    const fallback_step: ?u32 = if (own_default) plain.prefill_step else null;
    if (fallback_step) |s| notes.add("prefill_step {d} is this GPU's default, {d} when it admits fewer streams; ", .{
        policy.prefill_step,
        s,
    });
    if (o.mtp_drafts) |n| policy.mtp.drafts = @intCast(n);
    if (o.mtp_confidence) |c| policy.mtp.confidence = c;
    if (o.keep) |n| policy.slots = @intCast(@min(n, 1 << 16));
    if (matrixRefused(F, policy, caps)) |why| return refuse(a, problem, "matrix=on: {s}; matrix=auto runs the dot2 tiles", .{why});
    const group: ?hip.collective.Group = if (o.tp > 1) try join(a, gpa, io, o, caps, &policy, problem) else null;
    errdefer if (group) |g| g.close(gpa);
    const policy_line = try std.fmt.allocPrint(gpa, "{f}", .{policy});
    std.debug.print("[tensorfold] HIP rank {d} of {d}: policy {s}{s} {s}\n", .{ o.rank, o.tp, policy_line, if (o.rank > 0) " (rank 0's)" else "", notes.text() });
    return .{
        .dev = .{ .index = index, .caps = caps, .policy = policy, .group = group, .fallback_step = fallback_step },
        .options = .{
            .window = @intCast(window),
            .streams = @max(o.lanes, 1),
            .fixed = o.lanes_fixed,
            .rank = o.rank,
            .world = o.tp,
            .cache_gib = o.prompt_cache_gib,
            .cache_over_cap = o.prompt_cache_over_cap,
            .vision = o.vision,
        },
        .policy_line = policy_line,
    };
}

/// This rank in its group, once every rank runs the same executable, checkpoint, world, GPU and RCCL as rank 0.
fn join(
    a: Allocator,
    gpa: Allocator,
    io: std.Io,
    o: api.Open,
    caps: hip.Caps,
    policy: *hip.Policy,
    problem: *[]const u8,
) (Refused || Allocator.Error)!hip.collective.Group {
    const ident = hip.identity.ofThis(gpa, io, o.dir) catch |err| return refuse(
        a,
        problem,
        "cannot digest this rank's executable or the checkpoint in {s} for the group ({s})",
        .{ o.dir, @errorName(err) },
    );
    var fault: hip.collective.Fault = .{};
    const j: hip.collective.Join = .{
        .rank = o.rank,
        .world = o.tp,
        .master = o.master,
        .port = o.master_port,
        .caps = caps,
        .policy = policy,
        .ident = ident,
        .fault = &fault,
    };
    return hip.joinGroup(gpa, io, j) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.NoTensorParallel => refuse(
            a,
            problem,
            "this build has no tensor parallelism: it comes with the RCCL backend",
            .{},
        ),
        error.NoRccl => refuse(a, problem, "{s}", .{@import("sys").optional_libraries.spec(.rccl).message}),
        error.NoId => refuse(a, problem, "RCCL gave no id", .{}),
        error.RanksDiffer => if (fault.field == .layout)
            refuse(
                a,
                problem,
                "a rank's join handshake differs from rank 0's: run the same TensorFold build on every rank",
                .{},
            )
        else
            refuse(
                a,
                problem,
                "rank {d}'s {s} differs from rank 0's; the group is refused on every rank",
                .{ fault.rank, fault.field.what() },
            ),
        else => refuse(
            a,
            problem,
            "the ranks' link at {s}:{d} failed ({s})",
            .{ o.master, o.master_port, @errorName(err) },
        ),
    };
}

/// The family's Info.reasoning: false when it declares `pub const reasoning = false` (no reasoning block).
fn reasoning(comptime F: type) bool {
    return if (@hasDecl(F, "reasoning")) F.reasoning else true;
}

/// The family's refusals before it loads or joins a group (hip_families.zig).
const flagRefused = fam.refused;

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    group: ?hip.collective.Group,
    policy_line: []const u8,
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.release(h.family);
        if (h.group) |g| g.close(h.gpa);
        h.gpa.free(h.policy_line);
        h.gpa.destroy(h);
    }
};

/// The device bytes this process holds and their peak, as the runtime counts them.
fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = hip.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak };
}

test "HIP optional family offers: image metadata comes from the loaded owner's offer" {
    try std.testing.expect(fam.offerOf(.{}).vision == null);
    try std.testing.expect(fam.offerOf(.{ .offer = serve.Offer{} }).vision == null);
    const Stub = struct {
        fn prepare(_: *anyopaque, _: Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
    };
    var owner: u8 = 0;
    const v: api.Vision = .{ .ctx = &owner, .prepare = Stub.prepare, .image_token = 9, .image_tokens = 6144, .max_images = 2 };
    const offered = fam.offerOf(.{ .offer = serve.Offer{ .vision = v } }).vision.?;
    try std.testing.expectEqual(v.ctx, offered.ctx);
    try std.testing.expectEqual(v.image_token, offered.image_token);
    try std.testing.expectEqual(v.image_tokens, offered.image_tokens);
}

/// The engine for `o.dir`, or null with `problem` when no HIP family reads it; under tensor parallelism, rank 0.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const report = @import("sys").optional_libraries.atBoundary();
    defer report.deinit();
    if (try o.slideRefusal(a, "HIP")) |why| {
        problem.* = why;
        return null;
    }
    if (o.fsrRefusal("hip")) |why| {
        problem.* = why;
        return null;
    }
    inline for (registry) |F| {
        if (fam.reads(F, o.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native HIP engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    return null;
}

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (flagRefused(F, o)) |why| {
        problem.* = why;
        return null;
    }
    const r = ready(F, a, gpa, io, o, problem) catch |err| switch (err) {
        error.Refused => return null,
        else => |x| return x,
    };
    var served = false;
    defer if (!served) r.deinit(gpa);
    const result = if (@hasDecl(F, "openVision")) F.openVision(a, gpa, io, r.dev, o.dir, r.options, o.vision, problem) else F.open(a, gpa, io, r.dev, o.dir, r.options, problem);
    const loaded = result catch |err| switch (err) {
        error.Refused => return null,
        else => |x| return x,
    };
    errdefer loaded.deinit(loaded.ctx);
    const offer = fam.offerOf(loaded);
    const text = fam.generates(offer);
    if (fam.textOnly(o, offer))
        std.debug.print("[tensorfold] --vision: the HIP engine has no image tower for {s}; serving text only, and image requests are refused\n", .{o.model_type});
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.group = r.dev.group;
    h.policy_line = r.policy_line;
    h.family = loaded.ctx;
    h.release = loaded.deinit;
    h.cfg = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, loaded.backend, h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .benchmark_ranks = if (r.dev.group) |g| @intCast(g.world) else 1,
        .lanes = @intCast(loaded.streams),
        .context_window = @intCast(offer.window orelse r.options.window),
        .prefill_step = F.prefill_step,
        .decode_share = o.decode_share,
        .policy = offer.policy orelse h.policy_line,
        .startup = loaded.startup,
        .structures = text and loaded.backend.vtable.first_masked != null,
        .text_generation = text,
        .embeddings = offer.embeddings,
        .decides = offer.decides,
        .rerank = offer.rerank,
        .readouts = offer.readouts,
        .penalties = loaded.backend.penalties,
        .reasoning = reasoning(F),
        .vision = offer.vision,
        .prompt_cache = offer.cache != null,
        .prompt_cache_capacity = offer.cache_capacity,
        .warm_turns = offer.warm_turns,
        .thinking_budget = offer.thinking_budget orelse true,
    });
    h.host.cache = offer.cache;
    h.host.frame_key = offer.frame_key;
    h.host.on_thread = offer.on_thread;
    h.host.memory = .{ .read = readMemory };
    if (@hasDecl(F, "attach")) F.attach(h.family, &h.host, o);
    try h.host.start();
    served = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// A rank above 0: holds its share and runs rank 0's steps until rank 0 stops; false with `problem` if it cannot start.
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !bool {
    const report = @import("sys").optional_libraries.atBoundary();
    defer report.deinit();
    if (try o.slideRefusal(a, "HIP")) |why| {
        problem.* = why;
        return false;
    }
    inline for (registry) |F| {
        if (fam.reads(F, o.model_type)) {
            if (flagRefused(F, o)) |why| {
                problem.* = why;
                return false;
            }
            const r = ready(F, a, gpa, io, o, problem) catch |err| switch (err) {
                error.Refused => return false,
                else => |x| return x,
            };
            defer r.deinit(gpa);
            F.follow(a, gpa, io, r.dev, o.dir, r.options, problem) catch |err| switch (err) {
                error.Refused => return false,
                else => |x| return x,
            };
            return true;
        }
    }
    problem.* = try std.fmt.allocPrint(a, "the native HIP engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    return false;
}

test "a HIP family declares replies without reasoning with `pub const reasoning = false`; others keep the default" {
    try std.testing.expect(reasoning(struct {}));
    try std.testing.expect(!reasoning(struct {
        pub const reasoning = false;
    }));
}

test "the text families carry no offer, so the host gives them the empty one; GLM-OCR and Qwen3 offer through their Loaded" {
    inline for (.{ flashnext.native, nemotron.native }) |F| {
        const Loaded = @typeInfo(@typeInfo(@TypeOf(F.open)).@"fn".return_type.?).error_union.payload;
        try std.testing.expect(!@hasField(Loaded, "offer"));
    }
    try std.testing.expect(@FieldType(glm_ocr.native.Loaded, "offer") == serve.Offer);
    inline for (.{ "vision", "cache", "cache_capacity" }) |old| try std.testing.expect(!@hasField(glm_ocr.native.Loaded, old));
    try std.testing.expect(@FieldType(qwen3.native.Loaded, "offer") == serve.Offer);
    inline for (.{ "text_generation", "embeddings", "window", "decides", "rerank" }) |old| try std.testing.expect(!@hasField(qwen3.native.Loaded, old));
    // Qwen3.5's readouts reach the Info through its offer (both of its model types share one Loaded)
    try std.testing.expect(@FieldType(qwen35.native_engine.Loaded, "offer") == serve.Offer and !@hasField(qwen35.native_engine.Loaded, "readouts"));
}

test {
    _ = fam;
}

test "every registered family is listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len), families.len);
    try std.testing.expectEqualStrings("qwen3_5", families[0].model_type);
    try std.testing.expectEqualStrings("qwen3_5_moe", families[1].model_type);
    try std.testing.expectEqualStrings("qwen4_exp", families[2].model_type);
    try std.testing.expectEqualStrings("nemotron_h", families[3].model_type);
    try std.testing.expectEqualStrings("glm_ocr", families[4].model_type);
    try std.testing.expectEqualStrings("qwen3", families[5].model_type);
    try std.testing.expectEqualStrings("dots_ocr", families[6].model_type);
}

test "GLM-OCR native flags: MTP and tensor ranks refuse before readiness; copy drafts remain" {
    const F = glm_ocr.native;
    try std.testing.expect(!reasoning(F)); // GLM-OCR replies carry no reasoning block
    try std.testing.expectEqualStrings(F.mtp_refusal, flagRefused(F, .{ .dir = "missing", .model_type = F.model_type, .mtp_drafts = 1 }).?);
    try std.testing.expectEqualStrings(F.tp_refusal, flagRefused(F, .{ .dir = "missing", .model_type = F.model_type, .tp = 2 }).?);
    try std.testing.expect(flagRefused(F, .{ .dir = "missing", .model_type = F.model_type, .mtp_drafts = 0 }) == null);
}

test "Nemotron refuses --slide and --mtp-confidence on HIP before it loads, and keeps a cap" {
    const o: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .slide = true };
    try std.testing.expect(std.mem.indexOf(u8, flagRefused(nemotron.native, o).?, "--slide") != null);
    const sure: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .mtp_confidence = 0.5 };
    try std.testing.expect(std.mem.indexOf(u8, flagRefused(nemotron.native, sure).?, "--mtp-confidence") != null);
    const capped: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .mtp_drafts = 1 };
    try std.testing.expect(flagRefused(nemotron.native, capped) == null);
    try std.testing.expect(flagRefused(nemotron.native, .{ .dir = "d", .model_type = "nemotron_h" }) == null);
}

test "a one-GPU family refuses --tp before the group forms; the others are not refused for it" {
    const OneGpu = struct {
        pub fn refusal(o: api.Open) ?[]const u8 {
            return if (o.tp != 1) "--tp: one GPU only" else null;
        }
    };
    try std.testing.expectEqualStrings("--tp: one GPU only", flagRefused(OneGpu, .{ .dir = "d", .model_type = "m", .tp = 2 }).?);
    try std.testing.expect(flagRefused(OneGpu, .{ .dir = "d", .model_type = "m" }) == null);
    try std.testing.expect(flagRefused(nemotron.native, .{ .dir = "d", .model_type = "nemotron_h", .tp = 2 }) == null);
    try std.testing.expectEqualStrings(qwen3.native.tp_refusal, flagRefused(qwen3.native, .{ .dir = "d", .model_type = "qwen3", .tp = 2, .rank = 1 }).?);
    try std.testing.expect(!reasoning(qwen3.native));
}

test "dots refuses unsupported group and head options before opening a device" {
    try std.testing.expect(!reasoning(dots.native)); // dots.mocr replies carry no reasoning block
    const base: api.Open = .{ .dir = "d", .model_type = "dots_ocr" };
    try std.testing.expect(flagRefused(dots.native, base) == null);
    var o = base;
    o.tp = 2;
    try std.testing.expect(std.mem.indexOf(u8, flagRefused(dots.native, o).?, "--tp 1") != null);
    o = base;
    o.mtp_drafts = 1;
    try std.testing.expect(std.mem.indexOf(u8, flagRefused(dots.native, o).?, "no MTP head") != null);
    o.mtp_drafts = 0;
    try std.testing.expect(flagRefused(dots.native, o) == null);
    o = base;
    o.lanes = 129;
    o.lanes_fixed = true;
    try std.testing.expect(flagRefused(dots.native, o) != null);
    try std.testing.expect(dots.native.policyRefusal(.{}) == null);
    try std.testing.expect(dots.native.policyRefusal(.{ .graphs = .off }) == null);
    try std.testing.expect(dots.native.policyRefusal(.{ .attention = .f32 }) != null);
    try std.testing.expect(dots.native.policyRefusal(.{ .graphs = .on }) != null);
    try std.testing.expect(dots.native.policyRefusal(.{ .kernels = .reference }) != null);
    try std.testing.expect(dots.native.policyRefusal(.{ .prefill_step = 64 }) != null);
}

test "a family that names a --slide or --mtp-confidence refusal is refused before it loads; the others are not" {
    const Refuses = struct {
        pub const slide_refusal = "--slide: this family does not learn on HIP";
        pub const confidence_refusal = "--mtp-confidence: this family drafts by its costs on HIP";
    };
    const slid: api.Open = .{ .dir = "d", .model_type = "m", .slide = true };
    const sure: api.Open = .{ .dir = "d", .model_type = "m", .mtp_confidence = 0.5 };
    try std.testing.expectEqualStrings(Refuses.slide_refusal, flagRefused(Refuses, slid).?);
    try std.testing.expectEqualStrings(Refuses.confidence_refusal, flagRefused(Refuses, sure).?);
    try std.testing.expect(flagRefused(Refuses, .{ .dir = "d", .model_type = "m", .mtp_drafts = 2 }) == null);
    try std.testing.expect(flagRefused(Refuses, .{ .dir = "d", .model_type = "m" }) == null);
    inline for (.{ qwen35.native, qwen35.native_moe, flashnext.native }) |F| {
        try std.testing.expect(flagRefused(F, slid) == null);
        try std.testing.expect(flagRefused(F, sure) == null);
    }
}

test "the group flags: one rank source, a supported world, a master past one rank, a rank inside the world" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const base: api.Open = .{ .dir = "d", .model_type = "qwen3_5" };
    try checkGroup(a, base, &problem);
    var two = base;
    two.tp = 2;
    two.master = "node0";
    try checkGroup(a, two, &problem);
    var speed = base;
    speed.speed_up = "settings.json";
    try std.testing.expectError(error.Refused, checkGroup(a, speed, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "--speed-up") != null);
    speed.tp = 2;
    speed.master = "node0";
    try std.testing.expectError(error.Refused, checkGroup(a, speed, &problem));
    var three = two;
    three.tp = 3;
    try std.testing.expectError(error.Refused, checkGroup(a, three, &problem));
    var lone = two;
    lone.master = "";
    try std.testing.expectError(error.Refused, checkGroup(a, lone, &problem));
    var outside = two;
    outside.rank = 2;
    try std.testing.expectError(error.Refused, checkGroup(a, outside, &problem));
}

test "--drafter is refused before any load, on every rank, until the HIP engine runs DFlash2" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const drafted: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter" };
    try std.testing.expectError(error.Refused, checkGroup(a, drafted, &problem));
    try std.testing.expectStringStartsWith(problem, "--drafter: the HIP engine runs no DFlash2 drafter yet");
    var follower = drafted;
    follower.tp = 2;
    follower.rank = 1;
    follower.master = "node0";
    try std.testing.expectError(error.Refused, checkGroup(a, follower, &problem));
}

test "--kv-cache fp8 is refused by name on HIP; bf16 caches are not" {
    const fp8: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .kv8 = true };
    try std.testing.expectEqualStrings(kv8_refusal, cacheRefused(fp8).?);
    try std.testing.expect(cacheRefused(.{ .dir = "d", .model_type = "qwen3_5" }) == null);
}

test "matrix=on runs only where the matrix-core tiles exist; auto runs everywhere" {
    var on: hip.Policy = .{};
    try on.apply("matrix=on");
    try std.testing.expect(matrixRefused(qwen35.native, on, hip.Caps.of("gfx1100").?) == null);
    try std.testing.expect(matrixRefused(qwen35.native, on, hip.Caps.of("gfx1151").?) == null);
    try std.testing.expect(std.mem.indexOf(u8, matrixRefused(qwen35.native, on, hip.Caps.of("gfx1201").?).?, "not implemented") != null);
    // the families on MLX affine tiles keep the gfx12 refusal; qwen3's BF16 dense products have the WMMA tile
    inline for (.{ qwen35.native, qwen35.native_moe, flashnext.native, nemotron.native }) |F| try std.testing.expect(matrixRefused(F, on, hip.Caps.of("gfx1201").?) != null);
    try std.testing.expect(matrixRefused(qwen3.native, on, hip.Caps.of("gfx1201").?) == null);
    try std.testing.expect(matrixRefused(qwen3.native, on, hip.Caps.of("gfx1030").?) != null);
    inline for (.{ qwen35.native, qwen3.native }) |F| try std.testing.expect(matrixRefused(F, on, hip.Caps.of("gfx1100").?) == null);
    try std.testing.expect(matrixRefused(qwen35.native, on, hip.Caps.of("gfx1030").?) != null);
    const auto: hip.Policy = .{};
    for ([_][]const u8{ "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201" }) |name| try std.testing.expect(matrixRefused(qwen35.native, auto, hip.Caps.of(name).?) == null);
}

test {
    _ = @import("draft_sources_hip_test.zig");
}

test "slide admission: hip refuses before checkpoint or device access" {
    try @import("slide_open_test.zig").check(open, "HIP", &.{ "qwen3_5", "qwen3_5_moe", "qwen4_exp", "nemotron_h" });
}

test "slide admission: HIP followers refuse before joining ranks" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (families) |f| {
        var why: []const u8 = "";
        try std.testing.expect(!try follow(a, std.testing.allocator, std.testing.io, .{ .dir = "missing-slide-checkpoint", .model_type = f.model_type, .slide = true, .tp = 2, .rank = 1 }, &why));
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on HIP yet", .{f.model_type}), why);
    }
}

test "a family's offer names what it serves; one without an offer generates text in the asked window" {
    const p = fam.offerOf(.{ .streams = @as(usize, 1) });
    try std.testing.expect(p.text_generation and p.embeddings == null and p.window == null and !p.decides and p.rerank == null);
    const e = fam.offerOf(.{ .offer = serve.Offer{ .text_generation = false, .embeddings = .{ .dimensions = 8, .vocabulary = 4, .max_batch_tokens = 2048 }, .window = 2048 } });
    try std.testing.expect(!e.text_generation and e.embeddings.?.dimensions == 8 and e.window == 2048 and e.rerank == null);
    const r = fam.offerOf(.{ .offer = serve.Offer{ .text_generation = false, .decides = true, .rerank = .{ .yes = 9693, .no = 2152 }, .window = 32768 } });
    try std.testing.expect(!r.text_generation and r.decides and r.embeddings == null and r.window == 32768);
    try std.testing.expectEqual(@as(u32, 9693), r.rerank.?.yes);
    const d = fam.offerOf(.{ .offer = serve.Offer{ .readouts = .{ .max_rows = 16384, .max_questions = 64, .max_options = 1024 } } });
    try std.testing.expect(!fam.generates(d) and d.embeddings == null and d.rerank == null and fam.generates(p));
    try std.testing.expectEqual(@as(u32, 1024), d.readouts.?.max_options);
}
