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
const joint = @import("joint_readout.zig");
const config = @import("../../../weights/config.zig");

/// The MLX affine widths this native text server loads, each with its group size, then ModelOpt's own formats.
const mlx_formats = blk: {
    const widths = @import("core").quant.mlx.bit_widths;
    const extra = [_][]const u8{ "modelopt-nvfp4", "modelopt-fp8" };
    var out: [widths.len * 3 + extra.len][]const u8 = undefined;
    for (widths, 0..) |bits, i| for ([_]u16{ 32, 64, 128 }, 0..) |g, j| {
        out[3 * i + j] = std.fmt.comptimePrint("mlx-q{d}g{d}", .{ bits, g });
    };
    for (extra, 0..) |f, k| out[widths.len * 3 + k] = f;
    const done = out;
    break :blk &done;
};

/// The dense and the sparse checkpoints are two registry entries over one engine.
pub fn Native(comptime type_name: []const u8) type {
    return struct {
        pub const model_type = type_name;
        pub const formats = mlx_formats;
        /// BF16 serves decisions only, with a joint schema head (joint_head_config.json); chat refuses it.
        pub const decision_formats: []const []const u8 = &.{"bf16"};
        /// Positions a stream holds when neither --context nor the checkpoint's config names a window.
        pub const default_context: i64 = 32768;
        /// The engine cuts prompts itself.
        pub const prefill_step: u32 = 0;
        pub const open = openEngine;
        pub const follow = followRank;
    };
}

/// What the serve flags ask of the engine, as the host read them.
const serve = @import("hip_serve");
pub const Options = serve.Options;

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
    /// The readouts the engine answers, if any (then it generates no text); the host's defaults otherwise.
    offer: serve.Offer = .{},
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

/// Why `dir` cannot be served: BF16 Qwen3.5 serves only with its joint head, and the head only on the BF16 weights.
fn decisionOnly(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    // a config that does not read is the loader's to report
    var c = config.Config.read(gpa, io, dir) catch return null;
    defer c.deinit();
    const head = joint.present(io, a, dir);
    if (c.quant != .bf16 and head) {
        const what = "the joint schema head needs the BF16 checkpoint";
        return std.fmt.allocPrint(a, "{s}: {s} is quantized ({t})", .{ what, dir, c.quant }) catch what;
    }
    if (c.quant != .bf16 or c.modelopt or head) return null;
    if (engine.chatBf16(gpa, io, dir)) return null; // an image checkpoint: a tower the engine reads, no joint head
    const what = "BF16 Qwen3.5 serves as a decision backbone only";
    return std.fmt.allocPrint(a, "{s}: {s} has no joint_head_config.json", .{ what, dir }) catch what;
}

/// Why a Clef checkpoint cannot open under tensor parallelism: its head runs on one rank, so no rank opens it.
fn oneRank(a: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) ?[]const u8 {
    if (o.world <= 1 or !joint.present(io, a, dir)) return null;
    return "the joint schema head serves on one rank; drop --tp";
}

/// A bf16 checkpoint's joint schema head on the device (null for every other checkpoint); refused by name.
fn loadJoint(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8, problem: *[]const u8) Refused!?joint.Loaded {
    if (!e.plain() or !joint.present(io, a, dir)) return null;
    var why: @import("core").joint_schema.Problem = .{};
    return joint.Loaded.load(gpa, io, &e.driver, dir, e.model().spec.hidden, &why) catch |err| refuse(
        a,
        problem,
        "the joint schema head cannot open ({s}): {s}",
        .{ @errorName(err), why.text },
    );
}

/// One rank's loaded and sized engine: the streams and prompt cache every rank agreed on.
const Prepared = struct {
    e: *Engine,
    streams: usize,
    /// What the prompt cache holds: pages of the pool and linear snapshots (none when it is off).
    cache: memory.Cache,
    startup: []const u8,
    /// A Clef checkpoint's head weights and the readouts it plans (null for every other checkpoint).
    joint: ?joint.Loaded = null,
    bounds: hip.joint_schema.plan.Bounds = undefined,
};

/// Device bytes of `streams` lanes at `capacity` beside the weights, and their readout states and the head's scratch.
fn served(e: *Engine, streams: usize, capacity: usize, readouts: ?Readouts) !usize {
    const rows = streams * hip_lanes.Hip.max_window;
    const head = if (readouts) |r| hip.joint_schema.plan.bytes(r.cfg, r.bounds, streams) else 0;
    return try e.planBytes(streams, rows, capacity, streams * pages.pagesFor(capacity) + pages.pagesFor(rows), 0) + head;
}

const Readouts = struct { cfg: @import("core").joint_schema.Config, bounds: hip.joint_schema.plan.Bounds };

/// Loads this rank's share of the model, admits the streams its memory holds (with the other ranks) and sizes them.
fn prepare(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: Options,
    steps: *hip.Startup,
    problem: *[]const u8,
) (Refused || std.mem.Allocator.Error)!Prepared {
    const rows = hip_lanes.Hip.max_window;
    const held = hip.usage(false).device;
    var pf: engine.Preflight = .{};
    const e = Engine.load(gpa, io, dir, .{
        .preflight = &pf,
        .vision = o.vision,
        .slack = rows + 1,
        .device = dev.index,
        .policy = dev.policy,
        .rank = o.rank,
        .world = o.world,
        .group = dev.group,
        .startup = steps,
    }) catch |err| return switch (err) {
        error.PeerRefused, error.LinkFailed => steps.stoppedBy(a, problem),
        error.WeightsDoNotFit => steps.refuseAll(
            a,
            problem,
            "the checkpoint's {d:.1} GiB of weights do not fit the {d:.1} GiB the HIP memory budget grants " ++
                "({d:.1} GiB free less a {d:.1} GiB reserve{s}); free device memory or adjust " ++
                "TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_HIP_MEMORY_LIMIT_GB",
            .{
                gibs(pf.weights),
                gibs(pf.pool.room(pf.held)),
                gibs(pf.pool.free),
                gibs(pf.pool.reserve),
                if (pf.pool.limit != null) ", under TENSORFOLD_HIP_MEMORY_LIMIT_GB" else "",
            },
        ),
        error.BadReserve, error.BadLimit => steps.refuseAll(
            a,
            problem,
            "{s}",
            .{try hip.admission.badSetting(a, if (err == error.BadReserve) error.BadReserve else error.BadLimit, .{
                .reserve = @import("sys").getenv("TENSORFOLD_MEMORY_RESERVE_GIB"),
                .limit = @import("sys").getenv("TENSORFOLD_HIP_MEMORY_LIMIT_GB"),
            })},
        ),
        error.HostMemoryUnavailable => steps.refuseAll(
            a,
            problem,
            "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing HIP unified-memory admission",
            .{},
        ),
        // the CUDA engine refuses these alike: --vision asks for images, so a tower it cannot read stops the start
        error.UnsupportedQwenVision => steps.refuseAll(
            a,
            problem,
            "--vision: this checkpoint's image tower is not one the HIP engine reads (64-wide heads, 16-pixel " ++
                "patches, 2x2 merge): serve it without --vision for text",
            .{},
        ),
        error.UnsupportedTorchRope => steps.refuseAll(
            a,
            problem,
            "the HIP engine's image and BF16 text rotary reads BF16 activations, 256-wide heads, 64 rotary " ++
                "dimensions and the interleaved [11, 11, 10] M-RoPE; this checkpoint differs (serve a quantized " ++
                "checkpoint without --vision for text)",
            .{},
        ),
        else => steps.refuseAll(a, problem, "the native HIP engine cannot load {s} ({s})", .{ dir, @errorName(err) }),
    };
    errdefer e.deinit();
    // a Clef head's weights go up beside the model's and count with them
    var head = try loadJoint(a, gpa, io, e, dir, problem);
    errdefer if (head) |*h| h.deinit();
    const model = hip.usage(false).device - held;
    const after = e.budget(io) catch |err| return steps.refuseAll(
        a,
        problem,
        "the native HIP engine cannot read the GPU's memory ({s})",
        .{@errorName(err)},
    );
    const room = after.room(model);
    const reserve = after.reserve;
    // a follower sizes its streams for rank 0's window: the page tables rank 0 sends cover that many positions
    const window: usize = @intCast((try steps.together(a, .{ o.window, 0 }, .first, problem))[0]);
    const capacity = window + rows + 1;
    const readouts: ?Readouts = if (head) |h| .{ .cfg = h.cfg, .bounds = joint.bounds(window, e.o.prompt_rows) } else null;
    // the most streams up to the ones asked whose bytes fit, then the least of that over the ranks
    var streams = o.streams;
    while (streams > 0) : (streams -= 1) {
        if ((served(e, streams, capacity, readouts) catch |err| return steps.refuseAll(
            a,
            problem,
            "the native HIP engine cannot count its scratch ({s})",
            .{@errorName(err)},
        )) <= room) break;
    }
    streams = (try steps.together(a, .{ streams, 0 }, .least, problem))[0];
    // every rank holds the same least, so every rank refuses here alike
    if (streams == 0) {
        const one = served(e, 1, capacity, readouts) catch 0;
        return refuse(a, problem, "the HIP memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; lower --context, add ranks (--tp), or free device memory", .{ window, gibs(one), gibs(room), gibs(model), gibs(reserve) });
    }
    if (streams < o.streams and o.fixed) {
        const asked = served(e, o.streams, capacity, readouts) catch 0;
        return steps.refuseAll(
            a,
            problem,
            "--parallel {d} needs {d:.2} GiB at a {d}-token window, and the HIP memory budget leaves {d:.2} GiB: " ++
                "serve --parallel {d}, or lower --context",
            .{ o.streams, gibs(asked), window, gibs(room), streams },
        );
    }
    const need = served(e, streams, capacity, readouts) catch |err| return steps.refuseAll(
        a,
        problem,
        "the native HIP engine cannot count its scratch ({s})",
        .{@errorName(err)},
    );
    // the prompt cache: --prompt-cache-gib, else what the streams leave
    const left = room - need;
    const policy = dev.policy;
    const keep: usize = policy.slots;
    var budget: usize = if (o.cache_gib) |g| @intFromFloat(g * (1 << 30)) else left;
    if (budget > left and !o.cache_over_cap) {
        return steps.refuseAll(
            a,
            problem,
            "a {d:.2} GiB prompt cache does not fit the {d:.2} GiB the streams leave; " ++
                "lower --prompt-cache-gib, or --prompt-cache-over-cap to keep it",
            .{ gibs(budget), gibs(left) },
        );
    }
    if (keep == 0) budget = 0;
    var cache = memory.cache(e.weights.spec, e.act.size(), budget, keep);
    const fit = try steps.together(a, .{ cache.pages, cache.snaps }, .least, problem);
    cache = .{ .pages = fit[0], .snaps = fit[1] };
    e.o.batch_rows = streams * rows;
    e.o.streams = streams;
    e.size(capacity, streams * pages.pagesFor(capacity) + pages.pagesFor(e.o.batch_rows) + cache.pages) catch |err|
        return steps.refuseAll(
            a,
            problem,
            "the native HIP engine cannot allocate its scratch and pages ({s})",
            .{@errorName(err)},
        );
    const lane = gibs(state.Caches.deviceBytes(e.model(), capacity) + e.model().taps.ringBytes(e.model().spec.hidden) + pages.pagesFor(capacity) * memory.pageBytes(e.weights.spec, e.act.size()));
    const startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}, rank {d} of {d}: model {d:.2} GiB; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.1} GiB left after a {d:.1} GiB reserve; prompt cache {d:.2} GiB ({d} pages, {d} snapshots); prompts in {d}-row chunks", .{
        hip.kernels.arch, dev.index, o.rank, o.world, gibs(model), streams, if (streams == 1) "" else "s", lane, window, gibs(room), gibs(reserve), gibs(budget), cache.pages, cache.snaps, prefix.chunk,
    });
    return .{ .e = e, .streams = streams, .cache = cache, .startup = startup, .joint = head, .bounds = if (readouts) |r| r.bounds else undefined };
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    backend: *hip_lanes.Hip,
    startup: []const u8,
    joint: ?*joint.Joint = null,
};

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    // the other ranks hear that the rounds are over, then every rank frees its share
    own.backend.deinit();
    if (own.joint) |j| {
        j.deinit();
        own.gpa.destroy(j);
    }
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.destroy(own);
}

/// The engine for `dir` on `dev`; under tensor parallelism this is rank 0, which serves.
fn openEngine(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!Loaded {
    if (decisionOnly(a, gpa, io, dir) orelse oneRank(a, io, dir, o)) |why| return refuse(a, problem, "{s}", .{why});
    var steps: hip.Startup = .{ .link = if (dev.group) |g| g.link() else null };
    var p = try prepare(a, gpa, io, dev, dir, o, &steps, problem);
    errdefer p.e.deinit();
    errdefer gpa.free(p.startup);
    var weights_owned = p.joint != null;
    errdefer if (weights_owned) p.joint.?.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const before = hip.usage(false).device;
    const backend = hip_lanes.Hip.init(gpa, p.e) catch |err| return steps.refuseAll(
        a,
        problem,
        "the native HIP engine cannot start its lanes ({s})",
        .{@errorName(err)},
    );
    own.* = .{ .gpa = gpa, .e = p.e, .backend = backend, .startup = p.startup };
    errdefer own.backend.deinit();
    // the head is the one allocation of the plan made here: it holds what the plan counted for it
    const head = mtp.Head.deviceBytes(gpa, &p.e.driver, &p.e.lib, &p.e.weights, p.e.model(), @min(p.e.o.batch_rows, mtp.max_chains)) catch 0;
    if (hip.usage(false).device - before != head) return steps.refuseAll(
        a,
        problem,
        "the draft head holds {d} device bytes, the plan says {d}",
        .{ hip.usage(false).device - before, head },
    );
    if (p.cache.pages > 0) own.backend.keepPages(p.cache.pages, p.cache.snaps);
    errdefer if (own.joint) |j| {
        j.deinit();
        gpa.destroy(j);
    };
    if (p.joint) |loaded| {
        // openJoint owns the weights from here, whatever it returns
        weights_owned = false;
        own.joint = try openJoint(a, gpa, p.e, loaded, p.bounds, &steps, problem);
        try own.backend.useReadout(own.joint.?.readout());
    }
    // rank 0 serves only once every follower is ready behind it
    _ = try steps.together(a, .{ 0, 0 }, .first, problem);
    // a lone rank times its forwards for the depth rule; the ranks of a group draft without costs; bf16 drafts none
    if (dev.group) |g| own.backend.withLink(g.link()) else if (!p.e.plain()) own.backend.measure();
    return .{
        .backend = own.backend.backend(),
        .facts = own.backend.facts(),
        .rows = hip_lanes.Hip.max_window,
        .ctx = own,
        .deinit = release,
        .streams = p.streams,
        .startup = p.startup,
        .offer = .{ .readouts = own.backend.readouts(), .vision = serve.vision.offer(p.e.vision) },
    };
}

/// The device head over the loaded weights (it owns them from here), its scratch exactly what the plan counted.
fn openJoint(a: std.mem.Allocator, gpa: std.mem.Allocator, e: *Engine, loaded: joint.Loaded, bounds: hip.joint_schema.plan.Bounds, steps: *hip.Startup, problem: *[]const u8) !*joint.Joint {
    var w = loaded;
    var owned = true;
    errdefer if (owned) w.deinit();
    const blas = if (e.blas) |*b| b else return steps.refuseAll(a, problem, "the joint schema head needs rocBLAS", .{});
    const head_rows = e.model().head;
    if (head_rows.format() != .bf16) return steps.refuseAll(a, problem, "the joint schema head reads a BF16 lm_head", .{});
    const before = hip.usage(false).device;
    const head = hip.joint_schema.Head.init(gpa, &e.driver, &e.lib, blas, w.weights, bounds, head_rows.handle.bf16.weight, head_rows.n) catch |err|
        return steps.refuseAll(a, problem, "the joint schema head cannot start ({s})", .{@errorName(err)});
    owned = false;
    errdefer head.deinit();
    const want = hip.joint_schema.plan.bytes(head.cfg, bounds, 0);
    if (hip.usage(false).device - before != want) return steps.refuseAll(a, problem, "the joint schema head holds {d} scratch bytes, the plan says {d}", .{ hip.usage(false).device - before, want });
    const j = try gpa.create(joint.Joint);
    j.* = .{ .head = head, .stream = e.stream };
    return j;
}

/// A rank above 0: holds its share of the model and runs rank 0's steps until rank 0 stops.
fn followRank(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!void {
    if (decisionOnly(a, gpa, io, dir) orelse oneRank(a, io, dir, o)) |why| return refuse(a, problem, "{s}", .{why});
    var steps: hip.Startup = .{ .link = dev.group.?.link() };
    const p = try prepare(a, gpa, io, dev, dir, o, &steps, problem);
    defer p.e.deinit();
    defer gpa.free(p.startup);
    var w = worker.Worker.init(gpa, p.e) catch |err| return steps.refuseAll(
        a,
        problem,
        "rank {d} cannot start its worker ({s})",
        .{ o.rank, @errorName(err) },
    );
    defer w.deinit();
    _ = try steps.together(a, .{ 0, 0 }, .first, problem);
    try w.follow(dev.group.?.link());
}

test "a bf16 checkpoint without a joint head is refused as decision-only, an MLX one with a head too; others load" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const text =
        \\{"model_type": "qwen3_5", "tie_word_embeddings": false,
        \\ "text_config": {"hidden_size": 64, "intermediate_size": 32, "num_hidden_layers": 2,
        \\  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "linear_num_key_heads": 1,
        \\  "linear_num_value_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 16,
        \\  "linear_conv_kernel_dim": 4, "vocab_size": 16, "full_attention_interval": 2,
        \\  "layer_types": ["linear_attention", "full_attention"], "eos_token_id": 1}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = text });
    const bare = decisionOnly(a, std.testing.allocator, io, dir).?;
    try std.testing.expect(std.mem.startsWith(u8, bare, "BF16 Qwen3.5 serves as a decision backbone only: "));
    try std.testing.expect(std.mem.endsWith(u8, bare, "has no joint_head_config.json"));
    // with its head's files a bf16 checkpoint opens (the head's own loader checks them), on one rank only
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = "{}" });
    try std.testing.expect(decisionOnly(a, std.testing.allocator, io, dir) == null);
    try std.testing.expect(oneRank(a, io, dir, .{ .window = 64, .streams = 1 }) == null);
    const tp = oneRank(a, io, dir, .{ .window = 64, .streams = 1, .world = 2 }).?;
    try std.testing.expectEqualStrings("the joint schema head serves on one rank; drop --tp", tp);
    inline for (.{ "qwen3_5", "qwen3_5_moe" }) |kind| { // a headed checkpoint loads: BF16 offered for decisions only
        for (Native(kind).formats) |format| try std.testing.expect(!std.mem.eql(u8, format, "bf16"));
        try std.testing.expectEqualStrings("bf16", Native(kind).decision_formats[0]);
    }
    const mlx = try std.mem.replaceOwned(u8, a, text, "\"tie_word_embeddings\": false,", "\"tie_word_embeddings\": false, \"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"},");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = mlx });
    // a quantized checkpoint that ships the head would open as a text model: refused by name instead
    const quantized = decisionOnly(a, std.testing.allocator, io, dir).?;
    try std.testing.expect(std.mem.startsWith(u8, quantized, "the joint schema head needs the BF16 checkpoint: "));
    try tmp.dir.deleteFile(io, "joint_head_config.json");
    try std.testing.expect(decisionOnly(a, std.testing.allocator, io, dir) == null); // without it, a text model
}

test "a bf16 checkpoint without a joint head serves chat when the engine reads its tower" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    // LightOnOCR-3-0.8B's config: bf16 (no quantization) with a vision_config the HIP tower reads
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("../../../fixtures/lighton-ocr-0.8b.json") });
    try std.testing.expect(engine.chatBf16(std.testing.allocator, io, dir));
    try std.testing.expect(decisionOnly(a, std.testing.allocator, io, dir) == null);
    // the same weights with Clef's joint head file are not a chat checkpoint: the joint schema path takes them
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = "{}" });
    try std.testing.expect(!engine.chatBf16(std.testing.allocator, io, dir));
    try std.testing.expect(decisionOnly(a, std.testing.allocator, io, dir) == null);
}

test "capability audit: HIP Qwen format names cover the admitted affine widths" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":8,
        \\ "num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,
        \\ "linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,
        \\ "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"vocab_size":248320,
        \\ "full_attention_interval":4,"eos_token_id":248044}
    ;
    inline for (.{ Native("qwen3_5"), Native("qwen3_5_moe") }) |F| {
        for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u16{ 32, 64, 128 }) |g| {
            var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
            try doc.object.put(a, "model_type", .{ .string = F.model_type });
            var q: std.json.ObjectMap = .empty;
            try q.put(a, "bits", .{ .integer = bits });
            try q.put(a, "group_size", .{ .integer = g });
            try doc.object.put(a, "quantization", .{ .object = q });
            var parsed = try config.Config.parse(a, try std.json.Stringify.valueAlloc(a, doc, .{}), null);
            defer parsed.deinit();
            try std.testing.expectEqual(config.Width{ .bits = bits, .group = g }, try parsed.quant.width(""));
            const matrix: hip.ops.Affine = .{ .words = 0, .scale = 0, .bias = 0, .tables = .bf16, .n = 4096, .k = 4096, .bits = bits, .group = g };
            try matrix.check();
            const name = try std.fmt.allocPrint(a, "mlx-q{d}g{d}", .{ bits, g });
            const found = for (F.formats) |listed| {
                if (std.mem.eql(u8, listed, name)) break true;
            } else false;
            try std.testing.expect(found);
        };
        var affine_count: usize = 0;
        for (F.formats) |name| if (std.mem.startsWith(u8, name, "mlx-")) {
            affine_count += 1;
        };
        try std.testing.expectEqual(@as(usize, 18), affine_count);
    }
}
