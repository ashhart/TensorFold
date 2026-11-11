//! One loaded model on one HIP device: the kernel library, a stream, scratch and the pinned buffers rounds go through.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const state = @import("../forward/state.zig");
const pages = @import("../forward/pages.zig");
const fwd = @import("../forward/forward.zig");
const weights = @import("../model/weights.zig");
const bridge = @import("../model/bridge.zig");
const sample = @import("hip_serve").sample;
const memory = @import("memory.zig");
const draw = @import("hip_serve").draw;
const slicing = @import("../../../weights/slicing.zig");
const reduce = hip.reduce;
const round_graphs = @import("round_graphs.zig");
const lane_round = @import("round.zig");
const scratch = @import("scratch.zig");
const mtp = @import("mtp.zig");
const admission = hip.admission;
const prefix = @import("prefix.zig");
const plan = @import("../forward/plan.zig");
const config = @import("../../../weights/config.zig");
const table = @import("../../../weights/table.zig");
const vision = @import("vision.zig");

pub const Options = struct {
    vision: ?@import("engine_api").VisionOpen = null,
    /// Most positions a stream's caches hold (its prompt, its reply and a window's rows).
    capacity: usize = 0,
    /// Pages of the KV pool (0: enough for `default_streams` streams of the capacity and the scratch caches).
    pool_pages: usize = 0,
    /// Rows a shared forward holds at most.
    batch_rows: usize = 32,
    /// Filled before the weights load: the pool, the bytes held and the checkpoint's share (refused when too big).
    preflight: ?*Preflight = null,
    /// Streams the memory plan holds at once (the lanes, or what a check runs together), and linear snapshots beside.
    streams: usize = default_streams,
    snapshots: usize = 0,
    /// Rows one prompt pass takes at most (a longer one is refused): the server's chunk, a check's longest prompt.
    prompt_rows: usize = prefix.chunk,
    /// Test only: every prompt piece ends on a multiple of these rows (0: the chunk rule); `prompt_rows` holds one.
    cut_rows: usize = 0,
    /// The device ordinal among the visible ones.
    device: c_int = 0,
    /// Positions past a reply a verify writes (the window's rows and one more).
    slack: usize = 0,
    /// Tensor parallelism: this rank of `world`, and the group it joined (it opens the communicator).
    rank: usize = 0,
    world: usize = 1,
    group: ?hip.collective.Group = null,
    /// The startup steps the ranks take together over the group's link (null: the engine's own).
    startup: ?*hip.Startup = null,
    /// Replay rounds from captured graphs (the policy's `graphs` turns it off).
    graphs: bool = true,
    /// What the run may use, resolved once by whoever opens the engine.
    policy: hip.Policy = .{},
};

pub const Pick = lane_round.Pick;

/// Streams the pool of an engine opened without a memory plan holds whole.
pub const default_streams = 12;

fn gib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// The checkpoint's device bytes, near enough to refuse before reading: a ModelOpt one's own, or bf16 text.
fn checkpointBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, chat: bool) !u64 {
    var c = try config.Config.read(gpa, io, dir);
    defer c.deinit();
    // a ModelOpt checkpoint keeps its codes as stored and widens its group tables: count what the device holds
    if (c.modelopt) return table.deviceBytes(gpa, io, dir);
    if (c.quant != .bf16) return admission.weightBytes(io, dir);
    const bf16 = try table.textBytes(gpa, io, dir);
    return if (chat) 2 * bf16 else bf16;
}

/// A bf16 checkpoint served for chat (chatBf16): its projections run widened, on the row-invariant dense kernels.
fn chatCheckpoint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    var c = config.Config.read(gpa, io, dir) catch return false;
    defer c.deinit();
    return c.quant == .bf16 and chatBf16(gpa, io, dir);
}

/// Whether a bf16 checkpoint serves chat: an image tower the engine reads and no joint_head_config.json (Clef's).
pub fn chatBf16(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const tower = vision.Vision.config(gpa, io, dir) catch return false;
    @import("qwen_hip_vision").check(tower) catch return false; // a tower of another shape (72-wide heads) is not read
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/joint_head_config.json", .{dir}) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return true;
    return false;
}

/// [p, p, p] for every position below `capacity`: text-only prompt rows under Torch's M-RoPE.
fn linearPositions(gpa: std.mem.Allocator, d: *const hip.Runtime, capacity: usize) !hip.DeviceBuffer {
    var b = try hip.DeviceBuffer.alloc(d, capacity * 12);
    errdefer b.free();
    const rows = try gpa.alloc([3]i32, capacity);
    defer gpa.free(rows);
    for (rows, 0..) |*r, p| r.* = @splat(@intCast(p));
    try b.upload(0, std.mem.sliceAsBytes(rows));
    return b;
}

/// What `load` found before reading the weights.
pub const Preflight = struct { pool: admission.Pool = undefined, held: u64 = 0, weights: u64 = 0 };

/// Rows a drafter reads behind a stream's newest prompt chunk or round (DFlash2's sliding window).
pub const tap_window = 2048;

pub const Engine = struct {
    pub const Rows = lane_round.Rows;

    gpa: std.mem.Allocator,
    /// A ModelOpt checkpoint: its text layers decode in flight, its vision tower runs on rocBLAS.
    modelopt: bool = false,
    driver: hip.Runtime,
    ctx: hip.Context,
    lib: hip.Launcher,
    stream: hip.Stream,
    weights: weights.Model,
    bridge: *bridge.Bridge,
    o: Options,
    vision: ?*vision.Vision = null,
    /// Torch's text M-RoPE (an HF BF16 checkpoint's text, image prompts), owned apart from the tower.
    rope: ?@import("qwen_hip_vision").Kernels = null,
    /// `rope`'s [p, p, p] rows for text-only prompts, sized with the capacity.
    linear: ?hip.DeviceBuffer = null,
    /// The scratch below exists (`size` ran).
    sized: bool,
    /// A rank could not take a round: every rank stops before another collective.
    stopped: bool,
    act: view.Kind,
    dtype: sample.Dtype,
    /// A window's scratch, kept from its verify to its keep (the commit reads the per-row states).
    rounds: hip.Arena,
    /// A prompt's scratch: its rows, the step's temporaries.
    prompts: hip.Arena,
    ids: hip.HostBuffer,
    ids_dev: hip.DeviceBuffer,
    /// The keys and values of every stream and prefix entry.
    pool: pages.Pool,
    drawer: draw.Drawer,
    /// Tensor parallelism: this rank's communicator.
    comm: hip.collective.Comm = undefined,
    graphs: round_graphs.Graphs,
    /// The lane rounds' plan, graph choice and keep.
    round: lane_round.State = undefined,
    /// rocBLAS on `stream` and its planned workspace, for a bf16 checkpoint's projections (opened by `size`).
    blas: ?hip.blas.Blas = null,
    workspace: hip.DeviceBuffer = undefined,

    /// The model on the device, its scratch not yet sized; a rank that fails here leaves the refusal to its caller.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        var own: hip.Startup = .{ .link = if (o.world > 1) (o.group orelse return error.NoGroup).link() else null };
        const steps = o.startup orelse &own;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        // --vision on a checkpoint without a tower serves text, as before; an unreadable tower still refuses at open
        const seen: @FieldType(Options, "vision") = if (o.vision == null) null else if (vision.Vision.config(gpa, io, dir)) |_| o.vision else |err| switch (err) {
            error.NoQwenVision => null,
            else => return err,
        };
        e.o = o;
        e.o.prompt_rows = @max(o.prompt_rows, o.cut_rows); // a test's cut is one piece: the prompts arena holds it
        e.o.vision = seen;
        e.vision = null;
        e.rope = null;
        e.linear = null;
        if (seen != null and o.world > 1) return error.VisionTensorParallelUnsupported;
        e.sized = false;
        e.stopped = false;
        e.blas = null;
        // rounds and the draft head's batches replay graphs unless the policy says `graphs=off`, at any world size
        e.o.graphs = o.graphs and o.policy.graphsOn();
        e.graphs = round_graphs.Graphs.init(gpa);
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const caps = try hip.Device.kernelCaps(&e.driver, o.device);
        e.lib = try hip.Launcher.load(&e.driver, o.policy.choices(), hip.kernels.images);
        errdefer e.lib.unload();
        e.act = if (caps.act == .f16) .f16 else .bf16;
        e.dtype = if (caps.act == .f16) .f16 else .bf16;
        if (seen != null and e.act != .bf16) return error.VisionRequiresBf16;
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        const group: ?slicing.Rank = if (o.world > 1) .{ .rank = o.rank, .world = o.world } else null;
        // RCCL's init waits for every rank: none enters it unless every rank got this far
        _ = try steps.done(.{ 0, 0 }, .first);
        if (group != null) e.comm = try (o.group orelse return error.NoGroup).open();
        errdefer if (group != null) e.comm.close();
        _ = try steps.done(.{ 0, 0 }, .first);
        // a chat checkpoint's drafted and concurrent rounds equal its solo ones: rocBLAS products vary with rows
        const chat = chatCheckpoint(gpa, io, dir);
        if (o.preflight) |pf| {
            // a rank loads its share of the checkpoint, near enough to refuse before reading it
            pf.* = .{ .pool = try e.budget(io), .held = hip.usage(false).device, .weights = try checkpointBytes(gpa, io, dir, chat) / o.world };
            if (seen) |limits| pf.weights += try vision.Vision.bytes(try vision.Vision.config(gpa, io, dir), limits);
            if (pf.weights > pf.pool.room(pf.held)) return error.WeightsDoNotFit;
        }
        _ = try steps.done(.{ 0, 0 }, .first);
        e.weights = try weights.Model.loadRankWith(gpa, io, &e.driver, dir, group, chat);
        errdefer e.weights.deinit();
        var modelopt = false;
        {
            var cfg = config.Config.read(gpa, io, dir) catch null;
            defer if (cfg) |*c| c.deinit();
            modelopt = if (cfg) |c| c.modelopt else false;
            e.modelopt = modelopt;
        }
        if (e.plain() or chat or modelopt) {
            // bf16 products (rocBLAS, or the dense kernels a ModelOpt or chat checkpoint reads): one rank, no graphs
            if (e.act != .bf16) {
                std.log.err("a bf16 checkpoint needs bf16 activations; this GPU computes in fp16", .{});
                return error.UnsupportedQuantization;
            }
            if (o.world > 1) {
                std.log.err("a bf16 checkpoint runs on one rank", .{});
                return error.UnsupportedQuantization;
            }
            e.o.graphs = false;
        }
        e.bridge = try bridge.Bridge.init(gpa, &e.driver, &e.weights, e.act);
        errdefer e.bridge.deinit();
        if (group != null) e.bridge.model.tp = e.comm;
        // the checkpoint, not --vision, picks the text rotary: a bf16 chat checkpoint always takes Torch's
        const torch_text = chat;
        if (torch_text or seen != null) {
            const s = e.weights.spec;
            if (e.act != .bf16 or s.head_dim != 256 or s.rotary_dim != 64 or !s.mrope.interleaved or !std.mem.eql(usize, &s.mrope.section, &.{ 11, 11, 10 })) return error.UnsupportedTorchRope;
            e.rope = try @import("qwen_hip_vision").Kernels.load(&e.driver, hip.kernels.images[@backingInt(hip.kernels.Group.qwen_vision)]);
            e.bridge.model.rope = &e.rope.?;
            e.bridge.model.torch_text = torch_text;
        }
        errdefer if (e.rope) |*k| k.deinit();
        if (seen) |limits| e.vision = try vision.Vision.open(gpa, io, &e.driver, e.stream, &e.lib, dir, e.weights.spec.hidden, limits);
        errdefer if (e.vision) |v| v.deinit();
        _ = try steps.done(.{ 0, 0 }, .first);
        return e;
    }

    /// Allocate the scratch for streams of `capacity` positions and a pool of `pool_pages` pages (zero: the default).
    pub fn size(e: *Engine, capacity: usize, pool_pages: usize) !void {
        const rows = e.o.batch_rows;
        const prompt_rows = @min(e.o.prompt_rows, capacity);
        const count = if (pool_pages > 0) pool_pages else default_streams * pages.pagesFor(capacity) + pages.pagesFor(rows);
        const arenas = try e.arenaBytes(rows, prompt_rows, capacity);
        e.o.capacity = capacity;
        e.o.prompt_rows = prompt_rows;
        const before = hip.usage(false).device;
        e.rounds = try hip.Arena.init(&e.driver, arenas[0]);
        errdefer e.rounds.deinit();
        e.prompts = try hip.Arena.init(&e.driver, arenas[1]);
        errdefer e.prompts.deinit();
        e.ids = try hip.HostBuffer.alloc(&e.driver, prompt_rows * 4);
        errdefer e.ids.free();
        e.ids_dev = try hip.DeviceBuffer.alloc(&e.driver, prompt_rows * 4);
        errdefer e.ids_dev.free();
        e.linear = if (e.bridge.model.torch_text) try linearPositions(e.gpa, &e.driver, capacity) else null;
        errdefer if (e.linear) |*b| b.free();
        e.bridge.model.positions = if (e.linear) |b| b.base() else 0;
        e.drawer = try draw.Drawer.init(e.gpa, &e.driver, e.dtype, e.bridge.model.head.n * e.o.world, rows);
        errdefer e.drawer.deinit();
        e.pool = try pages.Pool.init(e.gpa, &e.driver, &e.bridge.model, count, e.o.rank == 0);
        errdefer e.pool.deinit();
        e.round = try lane_round.State.init(e);
        errdefer e.round.deinit(e);
        // rocBLAS serves an unquantized checkpoint's projections and a ModelOpt checkpoint's vision tower
        if (e.plain() or e.modelopt) {
            e.workspace = try hip.DeviceBuffer.alloc(&e.driver, hip.blas.workspace_bytes);
            e.blas = hip.blas.Blas.open(e.stream) catch |err| blk: {
                // a ModelOpt text checkpoint needs no rocBLAS: only its vision tower would, and it is refused without
                if (e.plain()) {
                    e.workspace.free();
                    return err;
                }
                std.log.warn("rocBLAS is unavailable ({s}): a ModelOpt checkpoint's vision tower is not served", .{@errorName(err)});
                break :blk null;
            };
        }
        errdefer e.closeBlas();
        if (e.blas) |*b| try b.useWorkspace(e.workspace.base(), e.workspace.len);
        const held = hip.usage(false).device - before;
        const planned = e.sizedBytes(arenas, rows, prompt_rows, capacity, count);
        if (held != planned) {
            std.log.err("sizing held {d} device bytes, the plan says {d}", .{ held, planned });
            return error.SizeMismatch;
        }
        e.sized = true;
    }

    /// The rounds and prompts arenas: the forwards counted at their largest shapes for `rows` and `prompt_rows`.
    fn arenaBytes(e: *Engine, rows: usize, prompt_rows: usize, capacity: usize) ![2]usize {
        return .{ try scratch.rounds(e, rows, capacity), try scratch.prompts(e, @min(prompt_rows, capacity), capacity) };
    }

    /// Device bytes `size` allocates beside the weights, with the arenas already counted.
    fn sizedBytes(e: *const Engine, arenas: [2]usize, rows: usize, prompt_rows: usize, capacity: usize, pool_pages: usize) usize {
        const m = e.model();
        const blas: usize = if (e.plain()) hip.blas.workspace_bytes else 0;
        return arenas[0] + arenas[1] + prompt_rows * 4 + draw.Drawer.deviceBytes(rows) + pages.Pool.deviceBytes(m, pool_pages) +
            plan.Buffer.deviceBytes(rows, m.spec.n_layers) + state.Caches.deviceBytes(m, rows) +
            hip.Votes.bytes(e.o.world) + blas + if (m.torch_text) capacity * 12 else 0;
    }

    /// The checkpoint's projections are BF16 as stored (an unquantized HF checkpoint): rocBLAS runs them.
    pub fn plain(e: *const Engine) bool {
        return switch (e.weights.embed) {
            .core => |d| std.meta.activeTag(d) == .bf16,
            .nvfp4, .fp8 => false,
        };
    }

    fn closeBlas(e: *Engine) void {
        if (e.blas) |*b| {
            b.close();
            e.workspace.free();
        }
        e.blas = null;
    }

    /// Device bytes `size(capacity, pool_pages)` would allocate with `rows` rows a forward: what admission weighs.
    pub fn sizeBytes(e: *Engine, rows: usize, capacity: usize, pool_pages: usize) !usize {
        const prompt_rows = @min(e.o.prompt_rows, capacity);
        return e.sizedBytes(try e.arenaBytes(rows, prompt_rows, capacity), rows, prompt_rows, capacity, pool_pages);
    }

    /// Device bytes beside the weights: `size`'s scratch and pool, `streams` lanes, `snapshots`, rank 0's head.
    pub fn planBytes(e: *Engine, streams: usize, rows: usize, capacity: usize, pool_pages: usize, snapshots: usize) !usize {
        const m = e.model();
        const lane = state.Caches.deviceBytes(m, capacity) + m.spec.hidden * m.act.size() + m.taps.ringBytes(m.spec.hidden) + if (e.vision) |v| v.laneBytes(capacity) else @as(usize, 0);
        // under tensor parallelism only rank 0 holds the head
        const head = if (e.o.rank > 0) 0 else mtp.Head.deviceBytes(e.gpa, &e.driver, &e.lib, &e.weights, m, @min(rows, mtp.max_chains)) catch |err| {
            std.log.err("PROBE planBytes: the MTP head fails ({s})", .{@errorName(err)});
            return err;
        };
        const sized = e.sizeBytes(rows, capacity, pool_pages) catch |err| {
            std.log.err("PROBE planBytes: sizeBytes fails ({s})", .{@errorName(err)});
            return err;
        };
        return sized + streams * lane + snapshots * memory.linearBytes(m.spec) + head;
    }

    /// The memory a plan may take, as the native backends count it: free (host's when shared), reserve and cap.
    pub fn budget(e: *const Engine, io: std.Io) !admission.Pool {
        return admission.budget(e.gpa, io, e.ctx);
    }

    /// Load and size in one: `o.capacity` positions, refused when the plan does not fit the memory (on every rank).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        var own: hip.Startup = .{ .link = if (o.world > 1) (o.group orelse return error.NoGroup).link() else null };
        var with = o;
        with.startup = o.startup orelse &own;
        return openSized(gpa, io, dir, with) catch |err| {
            // this rank's own failure reaches the others at the step they wait at
            with.startup.?.failed(err);
            return err;
        };
    }

    fn openSized(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const steps = o.startup.?;
        const held = hip.usage(false).device;
        const e = try load(gpa, io, dir, o);
        errdefer e.deinit();
        const weights_bytes = hip.usage(false).device - held;
        const pool_pages = if (o.pool_pages > 0) o.pool_pages else default_streams * pages.pagesFor(o.capacity) + pages.pagesFor(o.batch_rows);
        const need = try e.planBytes(o.streams, o.batch_rows, o.capacity, pool_pages, o.snapshots);
        const p = try e.budget(io);
        if (need > p.room(weights_bytes)) {
            std.log.err("{d} streams at {d} positions with {d} rows a round need {d:.2} GiB beside the model's {d:.2} GiB, and {d:.2} GiB is left after the {d:.1} GiB reserve", .{ o.streams, o.capacity, o.batch_rows, gib(need), gib(weights_bytes), gib(p.room(weights_bytes)), gib(p.reserve) });
            return error.OutOfDeviceMemory;
        }
        try e.size(o.capacity, pool_pages);
        _ = try steps.done(.{ 0, 0 }, .first);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.graphs.captured > 0) std.log.info("graphs: {d} captured ({d:.0} ms each), {d} of {d} rounds replayed", .{ e.graphs.captured, @as(f64, @floatFromInt(e.graphs.capture_ns)) / @as(f64, @floatFromInt(e.graphs.captured)) / 1e6, e.graphs.replayed, e.graphs.rounds });
        e.graphs.deinit();
        if (e.sized) {
            e.closeBlas();
            e.round.deinit(e);
            e.drawer.deinit();
            if (e.linear) |*b| b.free();
            e.ids_dev.free();
            e.ids.free();
            e.prompts.deinit();
            e.rounds.deinit();
            e.pool.deinit();
        }
        if (e.vision) |v| v.deinit();
        if (e.rope) |*k| k.deinit();
        e.bridge.deinit();
        e.weights.deinit();
        e.stream.deinit();
        e.lib.unload();
        if (e.o.world > 1) e.comm.close();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn model(e: *const Engine) *const view.Model {
        return &e.bridge.model;
    }

    pub fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        const blas: ?*const hip.blas.Blas = if (e.blas) |*b| b else null;
        return .{ .l = &e.lib, .bf16 = e.act == .bf16, .stream = e.stream.handle, .arena = arena, .blas = blas };
    }

    /// Caches for `total` positions (at most the capacity) with every page taken.
    pub fn newCaches(e: *Engine, total: usize) !state.Caches {
        var c = try state.Caches.initFull(e.gpa, &e.pool, e.model(), @min(total, e.o.capacity));
        errdefer c.deinit(e.gpa);
        try e.addTap(&c);
        return c;
    }

    /// Caches for `total` positions (at most the capacity) with no page yet: a stream takes them as it grows.
    pub fn emptyCaches(e: *Engine, total: usize) !state.Caches {
        var c = try state.Caches.init(e.gpa, &e.pool, e.model(), @min(total, e.o.capacity));
        errdefer c.deinit(e.gpa);
        try e.addTap(&c);
        return c;
    }

    fn addTap(e: *Engine, c: *state.Caches) !void {
        const m = e.model();
        if (m.taps.n > 0) try c.addTap(m.taps.ringBytes(m.spec.hidden));
    }

    /// The layers a drafter taps (none: off), filling rings of caches made after; captured rounds are dropped.
    pub fn setTaps(e: *Engine, layers: []const u32) !void {
        const m = &e.bridge.model;
        if (layers.len > m.taps.layers.len) return error.TooManyTaps;
        for (layers) |l| if (l >= m.spec.n_layers) return error.BadTap;
        if (layers.len > 0 and m.act.size() != 2) return error.TapsNeed16Bit;
        try e.stream.synchronize();
        e.graphs.deinit();
        e.graphs = round_graphs.Graphs.init(e.gpa);
        @memcpy(m.taps.layers[0..layers.len], layers);
        m.taps.n = layers.len;
        m.taps.rows = if (layers.len > 0) tap_window + @max(e.o.prompt_rows, e.o.batch_rows) else 0;
    }

    /// Whether the draft head's greedy batches replay graphs.
    pub fn headGraphs(e: *const Engine) bool {
        return e.o.graphs;
    }

    /// Waits for the stream, before caches are freed (a round may still be running on them).
    pub fn drain(e: *Engine) void {
        e.stream.synchronize() catch {};
    }

    /// The tokens of `rows` final rows at `hidden`: one projection, vocabulary slices joined in rank order.
    fn project(e: *Engine, o: hip.ops.Ops, hidden: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32) !void {
        const y = try o.project(hidden, e.model().head, rows, false);
        try e.drawRows(o, y, rows, reqs, out, false);
    }

    /// Each of `rows` logits rows drawn per `reqs`, whole rows joined first across ranks; `argmaxed`: greedy rows done.
    pub fn drawRows(e: *Engine, o: hip.ops.Ops, y: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32, argmaxed: bool) !void {
        try e.drawer.draw(o, e.stream, try e.wholeRows(o, y, rows), reqs[0..rows], out, argmaxed);
    }

    /// `rows` logits rows whole: this rank's when it holds the whole head, else every rank's slices joined.
    pub fn wholeRows(e: *Engine, o: hip.ops.Ops, y: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        return if (e.model().tp) |c| e.joined(o, c, y, rows) else y;
    }

    /// vocab_gather: every rank's slice of `rows` logits rows, joined on the device into whole rows in rank order.
    fn joined(e: *Engine, o: hip.ops.Ops, c: hip.collective.Comm, slice: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        const width = e.model().head.n;
        const n = rows * width;
        const size_of = slice.kind.size();
        const parts = try o.arena.take(c.world * n * size_of);
        try reduce.gather(o, c, slice, parts, n);
        const whole = try o.arena.take(c.world * n * size_of);
        for (0..rows) |row| for (0..c.world) |r| {
            const to = whole + (row * c.world + r) * width * size_of;
            const from = parts + (r * rows + row) * width * size_of;
            if (o.stream != hip.counting) try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(@ptrFromInt(to), @ptrFromInt(from), width * size_of, o.stream));
        };
        return .{ .ptr = whole, .kind = slice.kind };
    }

    /// Asked after each layer of a long prompt (the stream is idle then): true ends the pass with `error.Cancelled`.
    pub const Cancel = @import("lanes").stream.CancelCheck;

    const Poll = struct {
        stream: hip.Stream,
        cancel: Cancel,

        fn layer(ctx: *anyopaque, _: usize, _: hip.ops.Tensor, _: usize) anyerror!void {
            const p: *Poll = @ptrCast(@alignCast(ctx));
            try p.stream.synchronize();
            try @import("hip_serve").cancel.poll(p.cancel);
        }
    };

    /// The pass over `len` rows polling the cancel after each layer; null where it cannot stop (a short prompt, or tp).
    fn poll(e: *Engine, cancel: ?Cancel, len: usize, p: *Poll) ?fwd.Trace {
        const c = cancel orelse return null;
        if (!prefix.pollsLayers(e.o.world, len)) return null;
        p.* = .{ .stream = e.stream, .cancel = c };
        return .{ .ctx = p, .layer = Poll.layer };
    }

    /// Run `prompt[pos0..end]` into `caches`, its logits not read: a cut where the caches are kept.
    pub fn advance(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, end: usize, cancel: ?Cancel) !void {
        _ = try e.spanRows(caches, prompt, pos0, end, cancel);
    }

    /// `advance`, keeping the span's final-norm rows (end - pos0, hidden) in the prompts arena until its next reset.
    pub fn spanRows(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, end: usize, cancel: ?Cancel) !hip.ops.Tensor {
        if (e.stopped) return error.GroupStopped;
        if (end <= pos0 or end > caches.total or end - pos0 > e.o.prompt_rows) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0 .. end - pos0];
        @memcpy(ids, prompt[pos0..end]);
        try e.ids_dev.uploadAsync(0, e.ids.view(0, 4 * ids.len), e.stream);
        var p: Poll = undefined;
        const rows = try fwd.span(e.ops(&e.prompts), e.model(), caches, e.ids_dev.base(), end - pos0, pos0, e.poll(cancel, end - pos0, &p));
        try e.stream.synchronize();
        return rows;
    }

    /// Prefill `prompt[pos0..]` into `caches`; the last row's token, its final row copied to `last` (MTP's input).
    pub fn prefill(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, last: ?hip.DeviceBuffer, req: draw.Request, cancel: ?Cancel) !u32 {
        if (e.stopped) return error.GroupStopped;
        const m = e.model();
        const len = prompt.len - pos0;
        if (len == 0 or prompt.len > caches.total or len > e.o.prompt_rows) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0..len];
        @memcpy(ids, prompt[pos0..]);
        try e.ids_dev.uploadAsync(0, e.ids.view(0, 4 * ids.len), e.stream);
        const o = e.ops(&e.prompts);
        var p: Poll = undefined;
        const hidden = try fwd.span(o, m, caches, e.ids_dev.base(), len, pos0, e.poll(cancel, len, &p));
        const row = fwd.at(hidden, (len - 1) * m.spec.hidden);
        if (last) |b| try hip.raw.copy(b, 0, row.ptr, m.spec.hidden * m.act.size(), e.stream.handle);
        var token: [1]u32 = undefined;
        try e.project(o, row, 1, &.{req}, &token);
        return token[0];
    }

    /// Every window in one forward: `out` the token of every row; the round's snapshots live until its keeps flush.
    pub fn verify(e: *Engine, rows: []const Rows, reqs: []const draw.Request, out: []u32) !lane_round.Verified {
        if (e.stopped) return error.GroupStopped;
        return lane_round.verify(e, rows, reqs, out);
    }

    /// The round's graph choice: from the shape's history, or `forced` (rank 0's pick, which a follower obeys).
    pub fn choose(e: *Engine, rows: []const Rows, forced: ?Pick) !Pick {
        return lane_round.choose(e, rows, forced);
    }

    /// Marks a slot of the last verify to keep its first `rows` rows at the next `flush`.
    pub fn keep(e: *Engine, slot: usize, rows: usize) void {
        lane_round.mark(e, slot, rows);
    }

    /// Keeps every marked slot in one launch.
    pub fn flush(e: *Engine) !void {
        try lane_round.flush(e);
    }
};
