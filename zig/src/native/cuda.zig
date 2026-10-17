//! CUDA family registration, device setup, memory admission and the native lane host.
const openDrama = @import("drama_open.zig").open;
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const flashnext = @import("flashnext");
const family_memory = @import("cuda_family_memory.zig");
const pool = family_memory.pool;
const family_options = @import("cuda_options.zig");
pub const follow = @import("cuda_follow.zig").follow;
const lfm2 = @import("lfm2.zig");
const glm = @import("glm_cuda");
const qwen35 = @import("qwen35");
const Allocator = std.mem.Allocator;
const budget = @import("cuda_memory.zig");
const lane_backend = @import("cuda_bound.zig");
const snaps = @import("cuda_snaps.zig");
const slide_adapter = @import("slide_adapter.zig");
const cuda_disk = @import("cuda_disk.zig");
const vision_offer = @import("cuda_vision_offer.zig");
const drafted = @import("cuda_drafted.zig");
/// CUDA families provide metadata, open their lane backend and explain their refusals.
const registry = .{ nemotron.native, nemotron.omni, glm.native, qwen35.native, qwen35.native_moe, flashnext.native };

/// --record opens here (engines.zig refuses it for a backend without this).
pub const records = true;

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = blk: {
    var out: [registry.len + 2]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    out[registry.len] = lfm2.family;
    out[registry.len + 1] = .{ .model_type = "fish_omni", .formats = &.{"fp8-block128"} };
    const final = out;
    break :blk &final;
};

const gib: f64 = 1 << 30;

const cuda_device = @import("cuda_device.zig");
pub const chip = cuda_device.chip;
pub const deviceName = cuda_device.deviceName;
const getenv = cuda_device.getenv;
const setting = cuda_device.setting;
const windows = @import("cuda_window.zig");
const modelContext = windows.modelContext;
const contextWindow = windows.contextWindow;

/// A lone driver returns false at completion or true when yield hands it to the lane core.
const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// The context a lane thread needs current: the lane host steps rounds on its own thread, CUDA binds per thread.
threadlocal var bound: ?*const cuda.Context = null;

const failed_load = @import("cuda_load_gpu.zig");
const Gpu = failed_load.Gpu;

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    gpu: ?*Gpu, // null in host tests: no context to bind
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    inner: lanes.backend.Backend,
    vtable: lanes.backend.Backend.VTable,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},
    lone: ?LoneRun = null, // the family's driver for a lone drafted stream, called with `family`
    store: ?*api.prompt_cache.Store = null, // kept Nemotron states; null when the budget is 0
    learn: ?api.Learner = null, // the family's Sliding Weights learner (--slide), called with the context current
    slide: ?*anyopaque = null, // that learner's adapter, freed by slide_free before the family
    slide_free: ?*const fn (Allocator, *anyopaque) void = null,
    dir: []u8 = &.{}, // the model folder the learner writes its lessons into
    follow_fn: ?*const fn (*anyopaque) anyerror!void = null,
    health: ?api.HealthSource = null,
    recording: ?*record.Recording = null, // --record: the recorder in front of the family's backend
    served: drafted.Served = undefined, // the family's backend, wrapped with --drafter's draft model when given

    fn followFamily(p: *anyopaque) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(p));
        if (h.gpu) |g| try g.ctx.makeCurrent();
        return h.follow_fn.?(h.family);
    }

    /// A family's readiness reads host state only: no CUDA binding or blocking device work.
    fn readiness(h: *Host, comptime F: type) void {
        if (@hasDecl(F, "healthy")) h.health = .{ .ctx = h.family, .read = F.healthy };
    }

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        if (h.recording) |r| r.free(h.gpa);
        failed_load.closeEntry(if (h.gpu) |g| &g.ctx else null) catch {};
        if (h.store) |s| {
            const rows = s.disk;
            s.deinit();
            h.gpa.destroy(s);
            if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
        }
        if (h.slide) |s| h.slide_free.?(h.gpa, s);
        h.gpa.free(h.dir);
        h.served.close(h.gpa);
        h.release(h.family);
        if (h.gpu) |g| {
            g.ctx.deinit();
            g.driver.close();
            h.gpa.destroy(g);
        }
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }

    pub fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (h.gpu) |g| if (bound != &g.ctx) {
            g.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on the lane thread: {s}", .{@errorName(e)});
            bound = &g.ctx;
        };
        return h;
    }

    /// The round loop and the lane host over the family's backend, its thread started.
    fn serve(h: *Host, io: std.Io, facts: lanes.Model, rows: u32, info: api.Info, explain: ?api.Explain) !void {
        h.cfg = try lanes.Config.init(h.gpa, facts, rows, rows - 1);
        errdefer h.cfg.deinit(h.gpa);
        // a drafter's trees: the planner picks a lone stream's lanes, as deep as it drafts
        if (facts.head_trees) h.cfg.head_lanes = rows - 1;
        if (facts.head_trees) h.cfg.head_depth = @max(1, facts.drafts);
        h.clock = .{ .io = io };
        h.core = lanes.Engine.init(h.gpa, &h.cfg, h.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.host = api.LaneHost.init(h.gpa, io, &h.core, info);
        h.host.memory = if (h.gpu != null) .{ .read = family_memory.read } else null;
        h.host.explain = explain;
        h.host.health = h.health;
        h.host.cache = h.store;
        h.host.warm_quiet = if (info.warm_turns) nemotron.native.warm_quiet else 0;
        if (h.lone != null) h.host.lone = .{ .ctx = h, .run = loneRun, .sampled = true, .logprobs = h.inner.vtable.first_row != null, .structures = h.inner.vtable.first_masked != null, .min_budget = nemotron.native.lone_min_budget, .resumes = true };
        if (h.learn != null) h.host.learner = h.learner();
        try h.host.start();
    }

    /// The family's learner, each call made with the context current on the lane thread.
    fn learner(h: *Host) api.Learner {
        return .{
            .ctx = h,
            .begin = struct {
                fn f(p: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
                    const l = bind(p).learn.?;
                    return l.begin(l.ctx, request, sink);
                }
            }.f,
            .step = struct {
                fn f(p: *anyopaque) api.Learner.Step {
                    const l = bind(p).learn.?;
                    return l.step(l.ctx);
                }
            }.f,
            .abort = struct {
                fn f(p: *anyopaque) void {
                    const l = bind(p).learn.?;
                    l.abort(l.ctx);
                }
            }.f,
        };
    }

    /// The family's lone driver with the context current on the lane thread.
    fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const x = bind(p);
        return x.lone.?(x.family, s, hooks.ctx, hooks.committed, hooks.yield, hooks.resumed);
    }

    /// The family's backend, each call made with the context current on the calling thread.
    fn backend(h: *Host) lanes.backend.Backend {
        h.vtable = lane_backend.vtable(Host, h.inner.vtable.*);
        return .{ .ptr = h, .vtable = &h.vtable };
    }
};

const refusals = @import("cuda_refusal.zig");
const record = @import("cuda_record.zig");

/// The engine for `o.dir`, or null with `problem` set when `refusal` names a flag or no CUDA family reads it.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (@import("save_admission.zig").refused(io, o.dir, problem)) return null;
    if (refusals.backend(o)) |why| {
        problem.* = why;
        return null;
    }
    if (std.mem.eql(u8, o.model_type, "fish_omni")) return openDrama(a, gpa, io, o, problem);
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    if (std.mem.eql(u8, o.model_type, lfm2.family.model_type)) {
        if (refusals.lfm2(o)) |why| {
            problem.* = why;
            return null;
        }
        return lfm2.open(a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native CUDA engine has no backend for {s} checkpoints yet", .{o.model_type});
    return null;
}

/// The admit's refusals: no stream fits, or --parallel asks for more than the budget holds.
const no_stream = "the CUDA memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, " ++
    "and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve{s}; " ++
    "lower --context, or free device memory";
const too_many = "--parallel {d} needs {d:.2} GiB for its streams at a {d}-token window, " ++
    "and the CUDA memory budget leaves {d:.2} GiB{s}: serve --parallel {d}, or lower --context";

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusals.family(F, o)) |why| {
        problem.* = why;
        return null;
    }
    const native = modelContext(a, io, o.dir);
    const window = contextWindow(o.context, native, F.default_context) catch |e| {
        problem.* = switch (e) {
            error.Negative => try std.fmt.allocPrint(a, "--context {d}: a token count, or 0 for the model's window", .{o.context.?}),
            error.NoNative => "--context 0 asks for the model's window, and its config.json names none: give a token count",
            error.PastNative => try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ o.context.?, native }),
        };
        return null;
    };
    if (o.slide) {
        // a family learns once its module declares `slide` (its learner), `learnModel` and `linked`
        if (!@hasDecl(F, "slide")) {
            problem.* = try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on CUDA yet", .{o.model_type});
            return null;
        }
        comptime if (!@hasDecl(F, "learnModel") or !@hasDecl(F, "linked")) @compileError(F.model_type ++ " declares slide without learnModel and linked");
        if (try slide_adapter.refusal(F, a, io, o.dir, "`cp -RL` (a copy that follows links)")) |why| {
            problem.* = why;
            return null;
        }
    }
    const device = try setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const segments = try setting(a, o.segments, "TF_CUDA_SEGMENTS", 1, problem) orelse return null;
    if (segments < 1 or segments > F.max_segments) {
        const named = if (o.segments != null) "--segments " else "TF_CUDA_SEGMENTS=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: whole prompt chunks a call, 1 to {d}", .{ named, segments, F.max_segments });
        return null;
    }
    if (o.slide and segments > 1) {
        problem.* = "--slide runs a prompt one chunk at a time, its change's scratch shared by every layer: serve it without --segments or TF_CUDA_SEGMENTS";
        return null;
    }
    if (failed_load.recover() != 0) {
        problem.* = "a CUDA loader is retained until its context recovers; retry after recovery";
        return null;
    }
    const g = try gpa.create(Gpu);
    g.gpa = gpa;
    g.next = null;
    var opened = false;
    defer if (!opened) gpa.destroy(g);
    g.driver = cuda.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(e)});
        return null;
    };
    defer if (!opened) g.driver.close();
    const count = try g.driver.deviceCount();
    if (device >= count) {
        const named = if (o.device != null) "--device " else "TF_CUDA_DEVICE=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: this machine has {d} CUDA device{s} (0 to {d})", .{ named, device, count, if (count == 1) "" else "s", @max(count, 1) - 1 });
        return null;
    }
    g.ctx = try cuda.Context.init(&g.driver, @intCast(device));
    defer if (!opened) g.ctx.deinit();
    bound = &g.ctx;
    const capability = try g.ctx.capability();
    var name_buf: [256]u8 = undefined;
    const name = g.ctx.name(&name_buf) catch "GPU";
    if (o.slide and capability != 121) {
        problem.* = try std.fmt.allocPrint(a, "--slide learns on a GB10 (sm_121) so far, and this {s} is sm_{d}: serve it without --slide", .{ name, capability });
        return null;
    }
    const kernels: ?[]const u8 = switch (try cuda.aot.pick(a, io, getenv("TENSORFOLD_CUDA_KERNELS"), capability)) {
        .native => null,
        .captured => |dir| dir,
        .missing => |dir| {
            problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_CUDA_KERNELS={s} holds no aot.json: give a captured set's folder, or native for our own kernels", .{dir});
            return null;
        },
    };
    const before = try pool(a, io, &g.ctx, problem) orelse return null;
    // a family split over hosts holds only its rank's share (GLM-5.3-Flash's TP)
    const weights = try family_memory.weights(F, a, io, o, problem) orelse return null;
    const held0 = cuda.usage(false).device;
    // a family over hosts reads every rank's room after its load, by this budget, and admits by the smallest
    const Probe = struct {
        a: Allocator,
        io: std.Io,
        ctx: *const cuda.Context,
        held: u64,
        problem: *[]const u8,
        fn read(p: *anyopaque) ?u64 {
            const self: *@This() = @ptrCast(@alignCast(p));
            const now = (pool(self.a, self.io, self.ctx, self.problem) catch return null) orelse return null;
            return now.room(cuda.usage(false).device - self.held);
        }
    };
    var probe: Probe = .{ .a = a, .io = io, .ctx = &g.ctx, .held = held0, .problem = problem };
    var opts: F.Options = .{ .context = @intCast(window), .drafts = o.drafts, .segments = segments, .slide = o.slide };
    if (!family_options.set(F, o, &opts, problem)) return null;
    if (@hasField(F.Options, "room")) opts.room = .{ .ctx = &probe, .read = Probe.read };
    if (weights > before.room(held0)) {
        problem.* = try std.fmt.allocPrint(a, "the checkpoint's {d:.1} GiB of weights do not fit the {d:.1} GiB the CUDA memory budget grants ({d:.1} GiB free less a {d:.1} GiB reserve{s}); free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB", .{ toGib(weights), toGib(before.room(held0)), toGib(before.free), toGib(before.reserve), if (before.limit != null) ", under TENSORFOLD_CUDA_MEMORY_LIMIT_GB" else "" });
        return null;
    }
    const loaded = F.open(gpa, io, &g.ctx, o.dir, kernels, opts) catch |e| {
        cuda.tp_watch.fail(@errorName(e)); // a family over hosts: its other ranks stop with this error
        if (@hasDecl(F, "explains_open")) if (F.explain(null, e)) |text| {
            problem.* = text;
            return null;
        };
        switch (@as(anyerror, e)) { // every family's open: GLM's error set has no ModeloptNeedsSm90
            error.LoadRetained => {
                opened = true;
                failed_load.keep(g);
                problem.* = "CUDA loading failed without completion; its destinations and GPU owner are retained";
                return null;
            },
            error.ModeloptNeedsSm90 => {
                problem.* = try std.fmt.allocPrint(
                    a,
                    "{s} is sm_{d}: {s} (the 4-bit checkpoints run on it)",
                    .{ name, capability, nemotron.kernels.modeloptRefusal(capability).? },
                );
                return null;
            },
            else => {
                problem.* = try std.fmt.allocPrint(
                    a,
                    "the native CUDA engine cannot load {s} with {s} glue kernels ({s})",
                    .{ o.dir, kernels orelse "its own", @errorName(e) },
                );
                return null;
            },
        }
    };
    defer if (!opened) loaded.deinit(loaded.ctx);
    const taps = record.taps(loaded); // --record is refused before the host is made, so a refusal holds nothing
    if (o.record) |dir| if (try record.refusal(a, io, o.model_type, dir, taps)) |why| {
        problem.* = why;
        return null;
    };
    var served = try drafted.wrap(F, a, gpa, io, loaded, o, problem) orelse return null;
    defer if (!opened) served.close(gpa);
    const model = cuda.usage(false).device - held0;
    const after = try pool(a, io, &g.ctx, problem) orelse return null;
    const L = @TypeOf(loaded); // a family's Loaded may leave out a learner and a lone driver
    const learn_bytes: usize = if (@hasField(L, "learn_bytes")) loaded.learn_bytes else 0;
    var room = try family_memory.room(F, loaded.ctx, after.room(model) -| learn_bytes);
    var low: []const u8 = ""; // names the rank whose room bounds the streams, when it is not this one
    if (@hasField(@TypeOf(loaded), "peer_room")) if (loaded.peer_room) |pr| if (pr.bytes < room) {
        room = pr.bytes;
        low = try std.fmt.allocPrint(a, " on rank {d}, the smallest room of the ranks", .{pr.rank});
    };
    const free = loaded.free_streams;
    const streams = budget.admit(room, served.stream_bytes, free, o.lanes, o.lanes_fixed) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, no_stream, .{
                window,
                toGib(served.stream_bytes),
                toGib(room),
                toGib(model),
                toGib(after.reserve),
                low,
            }),
            error.TooMany => try std.fmt.allocPrint(a, too_many, .{
                o.lanes,
                toGib(served.stream_bytes * (o.lanes -| free)),
                window,
                toGib(room),
                low,
                free + room / served.stream_bytes,
            }),
        };
        return null;
    };
    served.reserve(streams) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot reserve {d} streams' caches at a {d}-token window ({s}): serve fewer with --parallel, or lower --context", .{ streams, window, @errorName(e) });
        return null;
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = g, .family = loaded.ctx, .release = loaded.deinit, .inner = served.backend, .served = served, .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .lone = if (@hasField(L, "lone")) loaded.lone else null };
    h.readiness(F);
    const held = served.stream_bytes * (streams -| free);
    // a family that keeps its states in pinned host memory (snap_host) has a host budget, not the device's room
    const host_cache = @hasDecl(F, "snap_host");
    const device_cache = budget.cacheBytes(o.prompt_cache_gib, room, held);
    const cache = if (!@hasDecl(F, "snap_save")) 0 else if (host_cache) budget.hostCacheBytes(a, io, o.prompt_cache_gib) else device_cache;
    h.startup = try std.fmt.allocPrint(gpa, "CUDA sm_{d} device {d} ({s}{s}): model {d:.2} GiB{s}; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window{s}, of {d:.1} GiB left after a {d:.1} GiB reserve; prompts in passes of up to {d} rows{s}{s}; {s}", .{
        capability,                                                                                                                                                        device,                                                                                                       name, if (after.unified) ", memory shared with the host" else "", toGib(model), if (learn_bytes > 0) try std.fmt.allocPrint(a, " and {d:.2} GiB kept for --slide's trainer", .{toGib(learn_bytes)}) else "", streams, if (streams == 1) "" else "s", toGib(served.stream_bytes), window, if (o.kv8) " (fp8 caches)" else "", toGib(room), toGib(after.reserve), F.prompt_rows, if (segments > 1) try std.fmt.allocPrint(a, ", {d} staggered segments a call", .{segments}) else "",
        if (cache > 0) try std.fmt.allocPrint(a, ", a {d:.1} GiB prompt cache{s}", .{ toGib(cache), if (host_cache) " in host memory" else "" }) else ", no prompt cache", if (kernels) |dir| try std.fmt.allocPrint(a, "glue kernels captured at {s}", .{dir}) else "own glue kernels",
    });
    errdefer gpa.free(h.startup);
    if (@hasDecl(F, "snap_save") and cache > 0) {
        const pc = api.prompt_cache;
        const store = try gpa.create(pc.Store);
        // rules and budget from the family's snap_* declarations (cuda_snaps.zig)
        const image: ?*const pc.rows.Image = if (F == nemotron.native) &cuda_disk.image else null; // its disk rows
        const copies: pc.Snapshots = .{ .ptr = loaded.target, .vtable = &snaps.Snaps(F, image).vtable };
        store.* = pc.Store.init(gpa, copies, snaps.cacheRules(F), try snaps.cacheBudget(F, loaded.target, cache));
        h.store = store;
        if (F == nemotron.native and o.learn_conversations) {
            if (o.learn) |root| try cuda_disk.attach(a, io, &g.ctx, o, root, kernels, loaded, store);
        }
    }
    errdefer if (h.store) |store| {
        store.deinit();
        gpa.destroy(store);
    };
    if (@hasDecl(F, "slide")) if (loaded.learns != null) {
        const Slide = slide_adapter.Adapter(F.slide);
        h.dir = try gpa.dupe(u8, o.dir);
        const s = try gpa.create(Slide);
        s.* = Slide.init(gpa, io, F.learnModel(loaded.ctx, h.dir));
        h.slide = s;
        h.slide_free = struct {
            fn f(al: Allocator, p: *anyopaque) void {
                const x: *Slide = @ptrCast(@alignCast(p));
                x.deinit();
                al.destroy(x);
            }
        }.f;
        h.learn = s.hook();
    };
    errdefer if (h.slide) |s| {
        h.slide_free.?(gpa, s);
        gpa.free(h.dir);
    };
    if (o.record) |dir| {
        const r = try record.open(gpa, io, dir, taps, served.backend);
        h.recording = r;
        h.inner = r.recorded.backend();
        h.lone = null; // every token through the lane rounds, where the recorder sees it
        std.log.info("recording each stream's rows under {s}/{s}", .{ dir, r.recorder.name });
    }
    errdefer if (h.recording) |r| r.free(gpa);
    // the family cuts its own prompt grid from position 0, as `tensorfold run` does: prefill_step 0
    try h.serve(io, served.facts, loaded.rows, .{
        .lanes = streams,
        .context_window = @intCast(window),
        .startup = h.startup,
        .prompt_cache = h.store != null,
        .warm_turns = h.store != null and @hasDecl(F, "warm_quiet"),
        .logprobs = served.backend.vtable.first_row != null,
        .prompt_logprobs = served.backend.vtable.prompt_rows,
        .structures = served.backend.vtable.first_masked != null,
        .vision = if (@hasField(@TypeOf(loaded), "vision")) vision_offer.offer(loaded.vision, o.vision) else null,
    }, .{ .ctx = loaded.ctx, .text = F.explain });
    opened = true;
    if (@hasField(@TypeOf(loaded), "follow")) if (loaded.follow) |f| {
        h.follow_fn = f;
        return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h, .follow = Host.followFamily };
    };
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

const toGib = family_memory.toGib;

test {
    _ = windows;
    _ = budget; // cuda_memory.zig's own tests, which no other test binary reaches
    _ = refusals;
    _ = lane_backend;
    _ = snaps;
    _ = cuda_device; // the device and variable tests
    _ = windows;
}

test "the lane core sees a family's prefill_step through the lane backend, so a served prompt can step" {
    var target: lanes.fake.Fake = .{ .gpa = std.testing.allocator, .prefill_chunks = 3 };
    defer target.deinit();
    var h: Host = undefined;
    h.inner = target.stepped();
    try std.testing.expect(h.backend().vtable.prefill_step != null);
}

test "every registered family and d1's decision engine are listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len + 2), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
    try std.testing.expectEqualStrings("lfm2_vl", families[registry.len].model_type);
}

test "a cancel during a long prompt stops it, and the next request's reply is unchanged (#291)" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 10 };
    defer target.deinit();
    const Nothing = struct {
        fn release(_: *anyopaque) void {}
    };
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = null, .family = &target, .release = Nothing.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    try h.serve(std.testing.io, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 7, .hidden_rows = true }, 8, .{ .lanes = 2 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    const e = h.host.engine();
    try std.testing.expect(e.memory(false) == null); // no device to count
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?api.Reason = null,
        fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (ev.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) api.Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const Breaker = struct {
        engine: api.Engine,
        id: api.Id,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == 2 and c.id != 0) c.engine.cancel(c.id);
        }
    };
    const prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 40 };
    var breaker: Breaker = .{ .engine = e, .id = 1 };
    target.prefill_hook = Breaker.call;
    target.prefill_hook_ctx = &breaker;
    var broken: Box = .{};
    defer broken.tokens.deinit(gpa);
    try e.submit(1, &request, .{ .ctx = &broken, .event = Box.event });
    try std.testing.expectEqual(api.Reason.cancelled, broken.wait());
    try std.testing.expect(target.prefill_count <= 3); // stopped at the next chunk, not after all ten
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released
    try std.testing.expectEqual(@as(usize, 0), broken.tokens.items.len);
    breaker.id = 0;
    var restored: Box = .{};
    defer restored.tokens.deinit(gpa);
    try e.submit(2, &request, .{ .ctx = &restored, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, restored.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (restored.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 40), restored.tokens.items.len);
}

test "family readiness reaches production lane admission without binding CUDA" {
    try @import("cuda_health_test.zig").check(Host, Host.readiness, Host.serve);
}

test "Flash Next refuses tensor-rank draft caching before files or CUDA" {
    try @import("cuda_flashnext_refusal_test.zig").cache(openWith);
}

test "Flash Next takes --vision at family admission" {
    try @import("cuda_flashnext_refusal_test.zig").vision();
}

test {
    _ = @import("cuda_vision_offer.zig");
}

test "nemotron learning safety: actual native close drains Nemotron before any source, trainer or parent free" {
    try @import("nemotron_trainer_test.zig").run(Host, Host.serve, Host.close, false);
}
test "nemotron learning safety: direct Nemotron trainer destruction drains before its first owner free" {
    try @import("nemotron_trainer_test.zig").run(Host, Host.serve, Host.close, true);
}
test "trainer close: real native close drains the accepted trainer before any child or parent is released" {
    try @import("trainer_close_test.zig").run(Host, Host.serve, Host.close);
}
