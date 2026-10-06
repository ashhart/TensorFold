//! The engines a native server opens on CUDA. Each family in `registry` brings its own lane backend (its `open`); this
//! file only owns the device, the round loop and the lane host, so a family adds itself here without server code.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const Allocator = std.mem.Allocator;

/// The CUDA families: namespaces with `model_type`, `formats`, `default_context`, `prefill_step` and `open`.
const registry = .{nemotron.native};

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = blk: {
    var out: [registry.len]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    const final = out;
    break :blk &final;
};

/// The chip class gate entries name ("nvidia-sm121" for a GB10); null without a CUDA device.
pub fn chip(a: Allocator) ?[]const u8 {
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    var ctx = cuda.Context.init(&driver, deviceOrdinal()) catch return null;
    defer ctx.deinit();
    return chipClass(a, ctx.capability() catch return null);
}

fn chipClass(a: Allocator, capability: u32) ?[]const u8 {
    return std.fmt.allocPrint(a, "nvidia-sm{d}", .{capability}) catch null;
}

/// The GPU ordinal TF_CUDA_DEVICE picks, as `tensorfold run` reads it; unset or empty means 0.
fn deviceOrdinal() c_int {
    const value = std.mem.span(std.c.getenv("TF_CUDA_DEVICE") orelse return 0);
    return std.fmt.parseInt(c_int, value, 10) catch 0;
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

/// The kernel set: TENSORFOLD_CUDA_KERNELS, else share/tensorfold/cuda/sm<capability> beside the binary.
fn kernelDir(a: Allocator, io: std.Io, capability: u32) ![]const u8 {
    if (std.c.getenv("TENSORFOLD_CUDA_KERNELS")) |dir| return a.dupe(u8, std.mem.span(dir));
    const exe = try std.process.executableDirPathAlloc(io, a);
    return std.fs.path.join(a, &.{ exe, "..", "share", "tensorfold", "cuda", try std.fmt.allocPrint(a, "sm{d}", .{capability}) });
}

/// The context a lane thread needs current: the lane host steps rounds on its own thread, CUDA binds per thread.
threadlocal var bound: ?*const cuda.Context = null;

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    inner: lanes.backend.Backend,
    vtable: lanes.backend.Backend.VTable,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.ctx.makeCurrent() catch {};
        h.release(h.family);
        h.ctx.deinit();
        h.driver.close();
        h.gpa.destroy(h);
    }

    fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (bound != &h.ctx) {
            h.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on the lane thread: {s}", .{@errorName(e)});
            bound = &h.ctx;
        }
        return h;
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
    problem.* = try std.fmt.allocPrint(a, "the native CUDA engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    return null;
}

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse @min(F.default_context, if (native > 0) native else F.default_context);
    if (window <= 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ window, native });
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.driver = cuda.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(e)});
        return null;
    };
    errdefer h.driver.close();
    h.ctx = try cuda.Context.init(&h.driver, deviceOrdinal());
    errdefer h.ctx.deinit();
    bound = &h.ctx;
    const kernels = try kernelDir(a, io, try h.ctx.capability());
    const loaded = F.open(gpa, io, &h.ctx, o.dir, kernels, .{ .context = @intCast(window), .drafts = o.drafts }) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot load {s} with kernels {s} ({s})", .{ o.dir, kernels, @errorName(e) });
        return null;
    };
    h.family = loaded.ctx;
    h.release = loaded.deinit;
    errdefer h.release(h.family);
    h.inner = loaded.backend;
    h.cfg = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = F.prefill_step });
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

test "chip classes name the compute capability" {
    const a = std.testing.allocator;
    const name = chipClass(a, 121).?;
    defer a.free(name);
    try std.testing.expectEqualStrings("nvidia-sm121", name);
}

test "every registered family is listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
}
