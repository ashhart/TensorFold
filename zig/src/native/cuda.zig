//! CUDA family registration, device setup, memory admission and the native lane host.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const Allocator = std.mem.Allocator;
const budget = @import("cuda_memory.zig");
const nemotron_slide = @import("nemotron_slide.zig");
const Pool = budget.Pool;

/// CUDA families provide metadata, open their lane backend and explain their refusals.
const registry = .{nemotron.native};

/// Nemotron's prompt-cache copies. A file-level value so the store's pointer stays valid for the process.
const nemotron_snaps: api.prompt_cache.Snapshots.VTable = .{
    .bytes = nemotron.native.snap_bytes,
    .save = nemotron.native.snap_save,
    .restore = nemotron.native.snap_restore,
    .drop = nemotron.native.snap_drop,
};

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = blk: {
    var out: [registry.len]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    const final = out;
    break :blk &final;
};

const gib: f64 = 1 << 30;

/// The chip class gate entries name ("nvidia-sm121" for a GB10) of the device TF_CUDA_DEVICE picks; null without one.
pub fn chip(a: Allocator) ?[]const u8 {
    const ordinal = envNumber(getenv("TF_CUDA_DEVICE")) catch return null;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    if (ordinal orelse 0 >= driver.deviceCount() catch return null) return null;
    return chipClass(a, driver.capability(@intCast(ordinal orelse 0)) catch return null);
}

fn chipClass(a: Allocator, capability: u32) ?[]const u8 {
    return std.fmt.allocPrint(a, "nvidia-sm{d}", .{capability}) catch null;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A variable's whole number: null when unset or empty, an error for anything else that is not one.
fn envNumber(text: ?[]const u8) error{Invalid}!?u32 {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch error.Invalid;
}

/// The flag's value, else the variable's, else `default`; `problem` names a variable that is not a whole number.
fn setting(a: Allocator, flag: ?u32, name: [:0]const u8, default: u32, problem: *[]const u8) !?u32 {
    if (flag) |v| return v;
    const v = envNumber(getenv(name)) catch {
        problem.* = try std.fmt.allocPrint(a, "{s}={s}: a whole number", .{ name, getenv(name).? });
        return null;
    };
    return v orelse default;
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

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// The bytes of the checkpoint's safetensors files: what its weights need on the device, near enough to refuse early.
fn weightBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

/// A /proc file's text, streamed: procfs reports size 0, and a positional read (readFileAlloc) stops there.
fn procText(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch null;
}

/// The pool now, read on the thread whose context is current; `problem` names a bad variable.
fn pool(a: Allocator, io: std.Io, ctx: *const cuda.Context, problem: *[]const u8) !?Pool {
    const unified = (try ctx.attribute(.integrated)) != 0;
    const card = try ctx.memInfo();
    const text = if (unified) procText(a, io, "/proc/meminfo") else null;
    const available = budget.counts(unified, .{ .total = card.total, .available = card.free }, text) catch {
        problem.* = "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing CUDA unified-memory admission";
        return null;
    };
    const free = available.available;
    const total = available.total;
    const reserve = budget.reserveBytes(getenv("TENSORFOLD_MEMORY_RESERVE_GIB"), total, unified) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_MEMORY_RESERVE_GIB={s}: a number of GiB from {d} to the memory's size", .{ getenv("TENSORFOLD_MEMORY_RESERVE_GIB").?, budget.floor(unified).least });
        return null;
    };
    const limit = budget.limitBytes(getenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB")) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_CUDA_MEMORY_LIMIT_GB={s}: a positive number of GiB whose byte count fits in a 64-bit size", .{getenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB").?});
        return null;
    };
    return .{ .free = free, .total = total, .reserve = reserve, .limit = limit, .unified = unified };
}

fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak, .pinned = u.host };
}

/// A lone driver returns false at completion or true when yield hands it to the lane core.
const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// The context a lane thread needs current: the lane host steps rounds on its own thread, CUDA binds per thread.
threadlocal var bound: ?*const cuda.Context = null;

const Gpu = struct { driver: cuda.Driver, ctx: cuda.Context };

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

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        if (h.gpu) |g| g.ctx.makeCurrent() catch {};
        if (h.store) |s| {
            s.deinit();
            h.gpa.destroy(s);
        }
        if (h.slide) |s| h.slide_free.?(h.gpa, s);
        h.release(h.family);
        if (h.gpu) |g| {
            g.ctx.deinit();
            g.driver.close();
            h.gpa.destroy(g);
        }
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }

    fn bind(p: *anyopaque) *Host {
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
        h.clock = .{ .io = io };
        h.core = lanes.Engine.init(h.gpa, &h.cfg, h.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.host = api.LaneHost.init(h.gpa, io, &h.core, info);
        h.host.memory = if (h.gpu != null) .{ .read = readMemory } else null;
        h.host.explain = explain;
        h.host.cache = h.store;
        if (h.lone != null) h.host.lone = .{ .ctx = h, .run = loneRun, .sampled = true };
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
        return x.lone.?(x.family, s, hooks.ctx, hooks.committed, hooks.yield);
    }

    /// The family's backend, each call made with the context current on the calling thread.
    fn backend(h: *Host) lanes.backend.Backend {
        const v = h.inner.vtable;
        h.vtable = .{
            .prefill = struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.prefill(x.inner.ptr, s);
                }
            }.f,
            .first = struct {
                fn f(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.first(x.inner.ptr, s, position);
                }
            }.f,
            .queue = struct {
                fn f(p: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.queue(x.inner.ptr, s, feed, position);
                }
            }.f,
            .read = struct {
                fn f(p: *anyopaque, handle: u64) anyerror!u32 {
                    const x = bind(p);
                    return x.inner.vtable.read(x.inner.ptr, handle);
                }
            }.f,
            .verify = struct {
                fn f(p: *anyopaque, w: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.verify(x.inner.ptr, w, out);
                }
            }.f,
            .keep = struct {
                fn f(p: *anyopaque, w: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.keep(x.inner.ptr, w, paths);
                }
            }.f,
            .draft = struct {
                fn f(p: *anyopaque, r: []const lanes.backend.DraftRequest) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.draft(x.inner.ptr, r);
                }
            }.f,
            .unspeculate = if (v.unspeculate != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.unspeculate.?(x.inner.ptr, s);
                }
            }.f else null,
            .probabilities = if (v.probabilities != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
                    const x = bind(p);
                    return x.inner.vtable.probabilities.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .tree = if (v.tree != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
                    const x = bind(p);
                    return x.inner.vtable.tree.?(x.inner.ptr, s, gpa);
                }
            }.f else null,
            .alternatives = if (v.alternatives != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []lanes.backend.Alternative) anyerror!usize {
                    const x = bind(p);
                    return x.inner.vtable.alternatives.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .first_row = if (v.first_row != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!lanes.LogprobRow {
                    const x = bind(p);
                    return x.inner.vtable.first_row.?(x.inner.ptr, s);
                }
            }.f else null,
            .release = struct {
                fn f(p: *anyopaque, s: *lanes.Stream) void {
                    const x = bind(p);
                    x.inner.vtable.release(x.inner.ptr, s);
                }
            }.f,
        };
        return .{ .ptr = h, .vtable = &h.vtable };
    }
};

/// The engine for `o.dir`, or null with `problem` set when no CUDA family reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native CUDA engine has no backend for {s} checkpoints yet", .{o.model_type});
    return null;
}

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
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
        if (!@hasDecl(F, "slide")) {
            problem.* = try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on CUDA yet", .{o.model_type});
            return null;
        }
        if (try F.linked(a, io, o.dir)) |name| {
            problem.* = try std.fmt.allocPrint(a, "--slide rewrites the model's own files, and {s} in {s} is a link to data another file shares (a Hugging Face cache's blob, or a hard link), so learning would change that file too: serve a copy made with `cp -RL` (a copy that follows links)", .{ name, o.dir });
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
    const g = try gpa.create(Gpu);
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
    const weights = weightBytes(io, o.dir);
    const held0 = cuda.usage(false).device;
    if (weights > before.room(held0)) {
        problem.* = try std.fmt.allocPrint(a, "the checkpoint's {d:.1} GiB of weights do not fit the {d:.1} GiB the CUDA memory budget grants ({d:.1} GiB free less a {d:.1} GiB reserve{s}); free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB", .{ toGib(weights), toGib(before.room(held0)), toGib(before.free), toGib(before.reserve), if (before.limit != null) ", under TENSORFOLD_CUDA_MEMORY_LIMIT_GB" else "" });
        return null;
    }
    const loaded = F.open(gpa, io, &g.ctx, o.dir, kernels, .{ .context = @intCast(window), .drafts = o.drafts, .segments = segments, .slide = o.slide }) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot load {s} with {s} glue kernels ({s})", .{ o.dir, kernels orelse "its own", @errorName(e) });
        return null;
    };
    defer if (!opened) loaded.deinit(loaded.ctx);
    const model = cuda.usage(false).device - held0;
    const after = try pool(a, io, &g.ctx, problem) orelse return null;
    const room = after.room(model) -| loaded.learn_bytes;
    const free = loaded.free_streams;
    const streams = budget.admit(room, loaded.stream_bytes, free, o.lanes, o.lanes_fixed) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, "the CUDA memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; lower --context, or free device memory", .{ window, toGib(loaded.stream_bytes), toGib(room), toGib(model), toGib(after.reserve) }),
            error.TooMany => try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its streams at a {d}-token window, and the CUDA memory budget leaves {d:.2} GiB: serve --parallel {d}, or lower --context", .{ o.lanes, toGib(loaded.stream_bytes * (o.lanes -| free)), window, toGib(room), free + room / loaded.stream_bytes }),
        };
        return null;
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = g, .family = loaded.ctx, .release = loaded.deinit, .inner = loaded.backend, .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .lone = loaded.lone };
    const cache = budget.cacheBytes(o.prompt_cache_gib, room, loaded.stream_bytes * (streams -| free));
    h.startup = try std.fmt.allocPrint(gpa, "CUDA sm_{d} device {d} ({s}{s}): model {d:.2} GiB{s}; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.1} GiB left after a {d:.1} GiB reserve; prompts in {d}-row chunks{s}{s}; {s}", .{
        capability,                                                                                                        device,                                                                                                       name, if (after.unified) ", memory shared with the host" else "", toGib(model), if (loaded.learn_bytes > 0) try std.fmt.allocPrint(a, " and {d:.2} GiB kept for --slide's trainer", .{toGib(loaded.learn_bytes)}) else "", streams, if (streams == 1) "" else "s", toGib(loaded.stream_bytes), window, toGib(room), toGib(after.reserve), F.prompt_rows, if (segments > 1) try std.fmt.allocPrint(a, ", {d} staggered segments a call", .{segments}) else "",
        if (cache > 0) try std.fmt.allocPrint(a, ", a {d:.1} GiB prompt cache", .{toGib(cache)}) else ", no prompt cache", if (kernels) |dir| try std.fmt.allocPrint(a, "glue kernels captured at {s}", .{dir}) else "own glue kernels",
    });
    errdefer gpa.free(h.startup);
    if (cache > 0) {
        const pc = api.prompt_cache;
        const store = try gpa.create(pc.Store);
        // states kept and resumed on the family's grid, where a cache-off pass cuts; the head reads one token on
        store.* = pc.Store.init(gpa, .{ .ptr = loaded.target, .vtable = &nemotron_snaps }, .{ .lookahead = 1, .planned = true, .grid = F.prompt_rows }, cache);
        h.store = store;
    }
    errdefer if (h.store) |store| {
        store.deinit();
        gpa.destroy(store);
    };
    if (loaded.learns) |e| {
        const Slide = nemotron_slide.Adapter(F.slide);
        const s = try gpa.create(Slide);
        s.* = Slide.init(gpa, io, e);
        h.slide = s;
        h.slide_free = struct {
            fn f(al: Allocator, p: *anyopaque) void {
                const x: *Slide = @ptrCast(@alignCast(p));
                x.deinit();
                al.destroy(x);
            }
        }.f;
        h.learn = s.hook();
    }
    errdefer if (h.slide) |s| h.slide_free.?(gpa, s);
    // the family cuts its own prompt grid from position 0, as `tensorfold run` does: prefill_step 0
    try h.serve(io, loaded.facts, loaded.rows, .{ .lanes = streams, .context_window = @intCast(window), .startup = h.startup, .prompt_cache = h.store != null, .logprobs = loaded.backend.vtable.first_row != null }, .{ .ctx = loaded.ctx, .text = F.explain });
    opened = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

test "this host's /proc/meminfo reads whole and parses (Linux)" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const text = procText(a, std.testing.io, "/proc/meminfo") orelse return error.Unreadable;
    defer a.free(text);
    try std.testing.expect(budget.meminfo(text) != null);
}

test "chip classes name the compute capability as gate entries do" {
    const a = std.testing.allocator;
    const name = chipClass(a, 121).?;
    defer a.free(name);
    try std.testing.expectEqualStrings("nvidia-sm121", name);
}

test "every registered family is listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
}

test "variables: unset or empty means the default, anything but a whole number is refused" {
    try std.testing.expectEqual(@as(?u32, null), try envNumber(null));
    try std.testing.expectEqual(@as(?u32, null), try envNumber(""));
    try std.testing.expectEqual(@as(?u32, 3), try envNumber(" 3"));
    try std.testing.expectError(error.Invalid, envNumber("one"));
    try std.testing.expectError(error.Invalid, envNumber("-1"));
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, contextWindow(0, 0, 16384));
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
