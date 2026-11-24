//! Qwen3.5 / Qwen3.6 for the native server on HIP: the engine sized to GPU memory and the lane backend over it.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const Engine = engine.Engine;
const memory = @import("memory.zig");
const mtp = @import("mtp.zig");
const prompt_rows = @import("prompt_rows.zig");
const pages = @import("../forward/pages.zig");
const state = @import("../forward/state.zig");
const worker = @import("worker.zig");
const hip_lanes = @import("hip_lanes.zig");
const joint = @import("joint_readout.zig");
const config = @import("../../../weights/config.zig");
const draft = @import("../draft/draft.zig");
const Draft = draft.Draft;

/// The MLX affine widths this native text server loads, each with its group size.
const mlx_formats = blk: {
    const widths = @import("core").quant.mlx.bit_widths;
    var out: [widths.len * 3][]const u8 = undefined;
    for (widths, 0..) |bits, i| for ([_]u16{ 32, 64, 128 }, 0..) |g, j| {
        out[3 * i + j] = std.fmt.comptimePrint("mlx-q{d}g{d}", .{ bits, g });
    };
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
        /// --drafter loads a DFlash2 draft model over the dense checkpoints (Qwen3.8-27B's); the sparse ones take none.
        pub const takes_drafter = std.mem.eql(u8, type_name, "qwen3_5");
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
    /// The DFlash2 drafter --drafter loaded, its taps already filling the streams' rings (null: none).
    drafter: ?lanes.drafter.Drafter = null,
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
    if (c.quant != .bf16 or head) return null;
    if (engine.chatBf16(gpa, io, dir)) return null; // an image checkpoint: a tower the engine reads, no joint head
    const what = "BF16 Qwen3.5 serves as a decision backbone only";
    return std.fmt.allocPrint(a, "{s}: {s} has no joint_head_config.json", .{ what, dir }) catch what;
}

/// Why a Clef checkpoint cannot open under tensor parallelism: its head runs on one rank, so no rank opens it.
fn oneRank(a: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) ?[]const u8 {
    if (o.world <= 1 or !joint.present(io, a, dir)) return null;
    return "the joint schema head serves on one rank; drop --tp";
}

const bf16_drafter = "--drafter: the HIP engine runs its DFlash2 drafter over quantized checkpoints only; " ++
    "serve this BF16 checkpoint without --drafter";

/// Why --drafter cannot run over `dir`: the drafter's rounds are priced by timed windows, which BF16 targets skip.
fn bf16Drafter(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) ?[]const u8 {
    if (o.drafter == null) return null;
    // a config that does not read is the loader's to report
    var c = config.Config.read(gpa, io, dir) catch return null;
    defer c.deinit();
    return if (c.quant == .bf16) bf16_drafter else null;
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
    draft: ?*Draft = null,
};

/// Device bytes of `streams` lanes at `capacity` beside the weights, and their readout states and the head's scratch.
fn served(e: *Engine, d: ?*Draft, streams: usize, capacity: usize, readouts: ?Readouts) !usize {
    const rows = streams * hip_lanes.Hip.max_window;
    // a drafter's tap rings hold its window past the rows a forward writes: sized for these streams' rows
    if (d != null) {
        e.o.batch_rows = rows;
        try e.setTaps(d.?.c.tapLayers());
    }
    const drafts = if (d) |x| streams * x.streamBytes() else 0;
    const head = if (readouts) |r| hip.joint_schema.plan.bytes(r.cfg, r.bounds, streams) else 0;
    return drafts + try e.planBytes(streams, rows, capacity, streams * pages.pagesFor(capacity) + pages.pagesFor(rows), 0) + head;
}

/// The DFlash2 drafter in `dir` on `e`, its taps set before the streams are sized, frames of up to `streams`.
fn openDraft(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8, streams: usize) !*Draft {
    const c = try draft.config.Config.read(gpa, io, dir, draft.config.Target.of(e.model().spec));
    try e.setTaps(c.tapLayers());
    errdefer e.setTaps(&.{}) catch {};
    return Draft.init(gpa, io, e, dir, @min(@max(streams, 1), draft.max_streams));
}

const Readouts = struct { cfg: @import("core").joint_schema.Config, bounds: hip.joint_schema.plan.Bounds };

/// The admitted streams' plan at some prompt rows, by `served`: the drafter's contexts and tap rings count in it.
const Plans = struct {
    e: *Engine,
    d: ?*Draft,
    streams: usize,
    capacity: usize,
    readouts: ?Readouts,

    pub fn at(p: Plans, rows: usize) !prompt_rows.Plan {
        p.e.o.prompt_rows = rows;
        return .{ .whole = try served(p.e, p.d, p.streams, p.capacity, p.readouts), .prompt = try p.e.promptBytes(p.capacity) };
    }
};

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
            .{try hip.admission.badSetting(a, if (err == error.BadReserve) error.BadReserve else error.BadLimit, hip.admission.envTexts())},
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
    // admission starts from the fallback's rows where a lone rank has them: the policy's must not cost a stream
    const want = prompt_rows.opening(dev.policy, e.plain());
    const small = prompt_rows.fallback(dev.policy, dev.fallback_step, want);
    e.o.prompt_rows = small orelse want;
    // a Clef head's weights go up beside the model's and count with them
    var head = try loadJoint(a, gpa, io, e, dir, problem);
    errdefer if (head) |*h| h.deinit();
    // the drafter's weights and scratch count with the model's; each stream's context with the streams'
    const d: ?*Draft = if (o.drafter) |dir_d| openDraft(gpa, io, e, dir_d, o.streams) catch |err| return steps.refuseAll(
        a,
        problem,
        "the native HIP engine cannot load the drafter {s} ({s})",
        .{ dir_d, @errorName(err) },
    ) else null;
    errdefer if (d) |x| x.deinit();
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
        if ((served(e, d, streams, capacity, readouts) catch |err| return steps.refuseAll(
            a,
            problem,
            "the native HIP engine cannot count its scratch ({s})",
            .{@errorName(err)},
        )) <= room) break;
    }
    streams = (try steps.together(a, .{ streams, 0 }, .least, problem))[0];
    // every rank holds the same least, so every rank refuses here alike
    if (streams == 0) {
        const one = served(e, d, 1, capacity, readouts) catch 0;
        return refuse(a, problem, "the HIP memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; lower --context, add ranks (--tp), or free device memory", .{ window, gibs(one), gibs(room), gibs(model), gibs(reserve) });
    }
    if (streams < o.streams and o.fixed) {
        const asked = served(e, d, o.streams, capacity, readouts) catch 0;
        return steps.refuseAll(
            a,
            problem,
            "--parallel {d} needs {d:.2} GiB at a {d}-token window, and the HIP memory budget leaves {d:.2} GiB: " ++
                "serve --parallel {d}, or lower --context",
            .{ o.streams, gibs(asked), window, gibs(room), streams },
        );
    }
    // a prompt cache a flag asked for claims its bytes beside the streams, unless it may pass the cap
    const flag: ?usize = if (o.cache_gib) |g| @intFromFloat(g * (1 << 30)) else null;
    const claim = prompt_rows.claim(flag, o.cache_over_cap);
    // the policy's rows come back when their larger plan holds that claim and the streams the fallback admitted
    var opened: prompt_rows.Opened = .{ .rows = want };
    const plans: Plans = .{ .e = e, .d = d, .streams = streams, .capacity = capacity, .readouts = readouts };
    if (small) |s| opened = prompt_rows.settled(plans, want, s, room, claim) catch |err| return steps.refuseAll(
        a,
        problem,
        "the native HIP engine cannot count its scratch ({s})",
        .{@errorName(err)},
    );
    e.o.prompt_rows = opened.rows;
    const need = served(e, d, streams, capacity, readouts) catch |err| return steps.refuseAll(
        a,
        problem,
        "the native HIP engine cannot count its scratch ({s})",
        .{@errorName(err)},
    );
    // the prompt cache: --prompt-cache-gib, else what the streams leave
    const left = room - need;
    const policy = dev.policy;
    const keep: usize = policy.slots;
    var budget: usize = flag orelse left;
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
    if (d) |x| x.reserve(streams) catch |err| return steps.refuseAll(
        a,
        problem,
        "the drafter cannot allocate its streams' contexts ({s})",
        .{@errorName(err)},
    );
    const own_bytes = if (d) |x| x.streamBytes() else 0;
    const lane = gibs(state.Caches.deviceBytes(e.model(), capacity) + e.model().taps.ringBytes(e.model().spec.hidden) + pages.pagesFor(capacity) * memory.pageBytes(e.weights.spec, e.act.size()) + own_bytes);
    var why: [96]u8 = undefined;
    const startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}, rank {d} of {d}: model {d:.2} GiB{s}; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.1} GiB left after a {d:.1} GiB reserve; prompt cache {d:.2} GiB ({d} pages, {d} snapshots); prompts in {d}-row chunks{s}", .{
        hip.kernels.arch, dev.index, o.rank, o.world, gibs(model), if (d != null) " with its DFlash2 drafter" else "", streams, if (streams == 1) "" else "s", lane, window, gibs(room), gibs(reserve), gibs(budget), cache.pages, cache.snaps, opened.rows, opened.note(&why),
    });
    return .{ .e = e, .streams = streams, .cache = cache, .startup = startup, .joint = head, .bounds = if (readouts) |r| r.bounds else undefined, .draft = d };
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    backend: *hip_lanes.Hip,
    startup: []const u8,
    joint: ?*joint.Joint = null,
    draft: ?*Draft,
};

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    // the other ranks hear that the rounds are over, then every rank frees its share
    own.backend.deinit();
    if (own.joint) |j| {
        j.deinit();
        own.gpa.destroy(j);
    }
    if (own.draft) |d| d.deinit();
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.destroy(own);
}

/// The engine for `dir` on `dev`; under tensor parallelism this is rank 0, which serves.
fn openEngine(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!Loaded {
    if (decisionOnly(a, gpa, io, dir) orelse oneRank(a, io, dir, o) orelse bf16Drafter(gpa, io, dir, o)) |why| return refuse(a, problem, "{s}", .{why});
    var steps: hip.Startup = .{ .link = if (dev.group) |g| g.link() else null };
    var p = try prepare(a, gpa, io, dev, dir, o, &steps, problem);
    errdefer p.e.deinit();
    errdefer if (p.draft) |d| d.deinit();
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
    own.* = .{ .gpa = gpa, .e = p.e, .backend = backend, .startup = p.startup, .draft = p.draft };
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
    // the drafter prices its rounds by the target's windows, timed with the taps on (cuda_native.zig's timeWindows)
    if (p.draft) |d| for (own.backend.costs.window[0..own.backend.costs.windows]) |c| d.setCost(c.width, c.ms);
    return .{
        .backend = own.backend.backend(),
        .facts = own.backend.facts(),
        .rows = hip_lanes.Hip.max_window,
        .ctx = own,
        .deinit = release,
        .streams = p.streams,
        .startup = p.startup,
        .offer = .{ .readouts = own.backend.readouts(), .vision = serve.vision.offer(p.e.vision) },
        .drafter = if (p.draft) |d| d.drafter() else null,
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

test "--drafter is refused over a BF16 checkpoint before it loads; a quantized one, or no --drafter, is not" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    // a BF16 checkpoint the engine serves for chat: decisionOnly lets it through, --drafter does not
    const bf16 = @embedFile("../../../fixtures/lighton-ocr-0.8b.json");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = bf16 });
    const drafted: Options = .{ .window = 64, .streams = 1, .drafter = "/models/drafter" };
    try std.testing.expect(decisionOnly(a, std.testing.allocator, io, dir) == null);
    try std.testing.expectEqualStrings(bf16_drafter, bf16Drafter(std.testing.allocator, io, dir, drafted) orelse "");
    try std.testing.expect(bf16Drafter(std.testing.allocator, io, dir, .{ .window = 64, .streams = 1 }) == null);
    const affine = "\"image_token_id\": 248056, \"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"},";
    const mlx = try std.mem.replaceOwned(u8, a, bf16, "\"image_token_id\": 248056,", affine);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = mlx });
    var read = try config.Config.read(std.testing.allocator, io, dir); // it reads, so the null below is no read failure
    defer read.deinit();
    try std.testing.expect(read.quant != .bf16 and bf16Drafter(std.testing.allocator, io, dir, drafted) == null);
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
