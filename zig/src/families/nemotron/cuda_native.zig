//! Nemotron's CUDA engine, MTP head and lane backend behind the native family interface.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const Head = @import("cuda_mtp.zig").Head;
const Lanes = @import("cuda_lanes.zig").Cuda;
const lone = @import("cuda_lone.zig");
const reuse = @import("cuda_reuse.zig");
const pages_mod = @import("cuda_pages.zig");
const kv = @import("core").kv_pool;

pub const snap_bytes = reuse.bytes;
pub const snap_save = reuse.save;
pub const snap_restore = reuse.restore;
pub const snap_drop = reuse.drop;
pub const snap_slots = reuse.slots; // the snapshot slots it makes come out of the cache's budget
pub const snap_place = reuse.place; // device or host states, as the native host finds the cache fits on the card
pub const kept_states = reuse; // their shape, a blank one and copies out and in, for the disk (--learn-conversations)
const core = @import("core");

/// Sliding Weights' learner (--slide): lessons change the weights this engine's lanes decode with.
pub const slide = @import("cuda_slide.zig");

pub const model_type = "nemotron_h";
pub const explains_open = true; // explain() words open refusals too (--vision on a checkpoint without a tower)
/// Nemotron 3.5 Bolt's config: the language model under llm_config, beside its vision tower's.
pub const also_model_types: []const []const u8 = &.{"nemotron_h_omni"};
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "modelopt-nvfp4", "modelopt-ggml" };
pub const default_context: i64 = engine.default_context;
pub const max_segments: u32 = @import("cuda_segments.zig").MAX;
pub const prompt_rows: u32 = state.pass_rows;
pub const window_rows: u32 = state.max_rows; // a request's last verify window: its pages hold that many rows past it

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, vision: ?vision.Limits = null, precision: ?@import("config.zig").Precision = null, mtp_drafts: ?u32 = null, mtp_confidence: ?f64 = null };
pub const vision = @import("cuda_vision.zig");

/// Thinking budgets the lone driver takes (cuda_lone.zig): above it, the stream goes to the lane core before its cut.
pub const lone_min_budget = @import("cuda_lone.zig").min_budget;

/// Nanoseconds a reply's background pass waits for the next request first: its first chunk holds the GPU.
pub const warm_quiet: u64 = 150 * std.time.ns_per_ms;

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence (caches, state, head caches).
    stream_bytes: usize,
    /// Streams that allocate nothing: the first takes the engine's own sequence, whose buffers the graphs hold.
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null, // called with `ctx`; null: every stream in the lane core
    target: *reuse.Target, // the prompt cache copies this engine (cuda_reuse.zig)
    learns: ?*engine.Engine = null, // --slide: the engine its learner trains, whose sites it attached at load
    learn_bytes: usize = 0, // --slide: what its trainer allocates at the first lesson, kept out of the streams' room
    vision: ?*vision.Vision = null, // --vision: the tower the server's image requests reach
    taps: []const u32 = &.{}, // the rows a recorder may ask `backend` for (empty: none)
    host_snaps: bool = false, // a discrete card: prompt states go to pinned host memory when the cache does not fit
    /// The other streams' memory as one pool of KV pages (null: each allocates a window at its prefill).
    layout: ?kv.Layout = null,
    /// One kept snapshot's slot (its Mamba states).
    snap_bytes: usize,
};

/// The pool behind the engine's own sequence: `streams` paged sequences and `pages` KV pages, made now.
pub fn attach(p: *anyopaque, streams: u32, pages: usize) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.pages = try pages_mod.Pages.init(own.gpa, own.e, streams, pages);
    own.lanes.pages = own.pages;
    own.lanes.paged_only = paging() == .all and streams > 0; // --parallel 1 has no paged stream: its own serves
}

/// Whether a request's caches fit now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    return own.lanes.fits(prompt, max_tokens);
}

/// The first shard, index or config.json in `dir` that a write in place would reach through a link, or null.
pub fn linked(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    return core.shard_edit.linked(arena, io, dir);
}

/// A shared expert's projection config.json keeps unquantized, as an earlier release's lessons left it, or null.
pub fn plainShared(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    return core.shard_edit.plainShared(arena, io, dir);
}

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, head: ?*Head, lanes: Lanes, target: reuse.Target, carve: ?*cuda.Carveout, living: ?@import("cuda_sites.zig").Sites = null, taps: [1]u32 = undefined, pages: ?*pages_mod.Pages = null };

/// Own-sequence graphs support both draw modes; drafting also captures the head.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (std.c.getenv("TF_NEMOTRON_RANK")) |r| if (!std.mem.eql(u8, std.mem.span(r), "0")) { // expert ranks never serve
        @import("cuda_worker.zig").serve(gpa, io, ctx, dir, o.precision) catch |err| {
            cuda.tp_watch.fail(@errorName(err)); // rank 0 stops with this rank's error
            std.log.err("expert rank stopped: {s}", .{@errorName(err)});
            std.c._exit(1); // a stream may still wait on rank 0: a normal exit would join it
        };
        std.process.exit(0);
    };
    const carve = try openCarveout(gpa, ctx);
    errdefer if (carve) |c| closeCarveout(gpa, c);
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = o.drafts, .graphs = true, .sampling = null, .segments = o.segments, .carveout = carve, .slide = o.slide, .vision = o.vision, .precision = o.precision });
    errdefer e.deinit();
    // Bolt has no MTP weights; copy drafts use the lane core and the target's ordinary verify windows.
    const head_enabled = o.drafts and !e.c.bolt;
    const head: ?*Head = if (head_enabled) try Head.init(e) else null;
    errdefer if (head) |h| h.deinit();
    if (head) |h| try h.capture();
    if (!head_enabled) try e.captureWindows(); // the one-row window: Engine.init captures windows only when drafting
    // the sampled mode's sets: a placeholder rule now, each stream's own uploaded at its prefill
    try e.setSampling(.{ .seed = 0, .temperature = 1.0 });
    try e.captureWindows();
    if (head) |h| try h.capture();
    try e.setSampling(null);
    e.snap_host = false; // states on the device until the native host places them (snap_place)
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = head, .lanes = try Lanes.init(gpa, e, head), .target = .{ .e = e, .head = head }, .carve = carve };
    errdefer own.lanes.deinit();
    own.lanes.copy_drafts = o.drafts;
    own.lanes.mtp_drafts = o.mtp_drafts;
    own.lanes.mtp_confidence = o.mtp_confidence;
    own.living = try slide.loadLiving(gpa, io, e, dir); // before the lanes' costs are measured: they decode through it
    errdefer if (own.living) |*s| s.deinit();
    try own.lanes.measure(io, dir);
    own.taps = .{@import("cuda_lanes.zig").finalTap(e.c)};
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = own.lanes.facts().exact_width,
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .target = &own.target,
        .lone = if (head != null) loneRun else null,
        .learns = if (o.slide) e else null,
        .learn_bytes = if (o.slide) @import("cuda_train.zig").Trainer.memory(e.c) else 0,
        .vision = e.vision,
        .taps = &own.taps,
        .host_snaps = e.k.discrete, // a card with its own memory may keep prompt states in the host's (GB10 shares one)
        .layout = if (o.slide or e.k.triton != null or paging() == .off) null else pages_mod.layout(e), // our glue pages; the learner reads windows
        .snap_bytes = reuse.mambaBytes(e.c),
    };
}

/// What a learner over the loaded engine `p` learns into; `dir` must outlive it.
pub fn learnModel(p: *anyopaque, dir: []const u8) slide.Model {
    const own: *Owned = @ptrCast(@alignCast(p));
    return .{ .e = own.e, .dir = dir, .living = &own.living };
}

fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    return lone.run(own.gpa, &own.lanes, s, .{ .ctx = hooks, .committed = committed, .yield = yield, .resumed = resumed });
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.VisionRefused => "--vision: this checkpoint has no image tower the CUDA engine reads (Nemotron Omni's RADIO); serve it without --vision",
        error.NoVision => "this server reads no images: start it with --vision",
        error.NemotronServedWithoutVision => "image inputs need the server started with --vision",
        error.PrecisionNotBuilt => "--precision checkpoint: NVIDIA's FP4 and FP8 math is built for Nemotron 3.5 Super VL's NVFP4 checkpoint so far; serve this one with --precision full",
        error.CheckpointMathNeedsBlackwell => "--precision checkpoint: NVIDIA's FP4 math runs on Blackwell (compute capability 12.x); serve with --precision full",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.deinit();
    if (own.pages) |x| x.deinit();
    if (own.living) |*s| s.deinit();
    if (own.head) |h| h.deinit();
    own.e.deinit();
    if (own.carve) |c| closeCarveout(own.gpa, c);
    own.gpa.destroy(own);
}

/// TF_CUDA_CARVEOUT=1: the display carveout KV planes go to first (cuda/carveout.zig); null when it is off.
fn openCarveout(gpa: std.mem.Allocator, ctx: *const cuda.Context) !?*cuda.Carveout {
    const bytes = (try cuda.carveout.requested(false, env("TF_CUDA_CARVEOUT"), env("TF_CUDA_CARVEOUT_MIB"))) orelse return null;
    const card: [*:0]const u8 = platform.c.getenv("TF_DRM_CARD") orelse cuda.carveout.default_card;
    const c = try gpa.create(cuda.Carveout);
    errdefer gpa.destroy(c);
    c.* = cuda.Carveout.open(ctx.d, card, bytes) catch |err| {
        std.log.err("display carveout of {d} MiB from {s}: {t} (needs nvidia_drm modeset=1, the card's device node and no display in use)", .{ bytes >> 20, card, err });
        return err;
    };
    std.log.info("display carveout: {d} MiB from {s}; KV planes go there first", .{ bytes >> 20, card });
    return c;
}

fn closeCarveout(gpa: std.mem.Allocator, c: *cuda.Carveout) void {
    c.close();
    gpa.destroy(c);
}

/// TF_KV_PAGES: 0 gives every stream a window at its prefill, as before the pool; all puts every stream on pages.
fn paging() enum { off, on, all } {
    const v = env("TF_KV_PAGES") orelse return .on;
    return if (std.mem.eql(u8, v, "0")) .off else if (std.mem.eql(u8, v, "all")) .all else .on;
}

fn env(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(platform.c.getenv(name) orelse return null);
}

/// The tensors this family loads; the early check uses the same vision switch as open().
pub fn tensorWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision_on: bool) !u64 {
    return @import("core").checkpoint_weights.count(a, io, dir, vision_on);
}
