//! Qwen3 dense checkpoints on the native CUDA server: the head the folder declares, sized to the memory budget.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const engine = @import("engine.zig");
const lane = @import("../lane.zig");
const folder = @import("../head.zig");

const Engine = engine.Engine;
const Prompt = lane.Prompt(Engine, engine.max_inputs);
const Allocator = std.mem.Allocator;

/// The registry entry: `native/cuda.zig` opens `qwen3` checkpoints through it.
pub const model_type = "qwen3";
pub const formats: []const []const u8 = &.{"bf16"};
/// The window when neither --context nor the checkpoint's config names one (the Qwen3 8B serving default).
pub const default_context: i64 = 32768;
/// An embedding engine writes no reply text, so no reasoning block (Info.reasoning).
pub const reasoning = false;
pub const tp_refusal = "--tp: Qwen3 dense checkpoints run on one CUDA GPU; serve with --tp 1";
pub const confidence_refusal = "--mtp-confidence: Qwen3 dense checkpoints have no draft head";
pub const drafter_refusal = "--drafter: Qwen3 dense checkpoints have no drafts to train; serve without --drafter";
pub const vision_refusal = "--vision: this CUDA family reads no images; serve it without --vision";
pub const kv8_refusal = "--kv-cache fp8: this CUDA family keeps bf16 caches; serve without it";
pub const loops_refusal = "--loops sets a looped model's recurrent passes (Dory's): this CUDA family has none; serve without it";
pub const record_refusal = "--record: qwen3 checkpoints give no rows to record on CUDA yet";

/// A refusal: the client's words in `problem`, and an error, so every errdefer of the open runs.
fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{Refused} {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// Opens `o.dir` on CUDA; null with the reason in `problem` when it is refused.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8, comptime budget: type, comptime windows: type) anyerror!?api.Opened {
    return openEngine(a, gpa, io, o, problem, budget, windows) catch |err| switch (err) {
        error.Refused => null,
        else => return err,
    };
}

/// Whatever the server holds: the device handles, the engine, its prompt lane and the host that serves it.
const Owned = struct {
    gpa: Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    e: *Engine,
    p: Prompt,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []const u8,
};

/// The context this thread's CUDA calls go to: open's thread at startup, the lane host's thread at rounds.
threadlocal var bound: ?*const cuda.Context = null;

/// Makes the owner's context current on this thread once.
fn bind(o: *Owned) void {
    if (bound != &o.ctx) {
        o.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on this thread: {s}", .{@errorName(e)});
        bound = &o.ctx;
    }
}

/// The lane thread binds the device context before its first round (on_thread).
fn bindThread(ptr: *anyopaque) anyerror!void {
    bind(@as(*Owned, @ptrCast(@alignCast(ptr))));
}

fn halt(ptr: *anyopaque, reason: []const u8) void {
    const o: *Owned = @ptrCast(@alignCast(ptr));
    o.host.halt(reason);
}

/// The ownership order of `close`: the host joined first, then the lane's and the engine's, the device last.
fn release(ptr: *anyopaque) void {
    const o: *Owned = @ptrCast(@alignCast(ptr));
    o.host.stop(); // the round in flight has ended: the device is this thread's alone
    bind(o);
    o.core.deinit();
    o.cfg.deinit(o.gpa);
    o.p.deinit();
    o.e.deinit();
    o.ctx.deinit();
    o.driver.close();
    o.gpa.free(o.startup);
    o.gpa.destroy(o);
}

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn scratchOf(e: *Engine, window: usize) anyerror!usize {
    return e.scratchBytes(window);
}

/// The driver's ordinal: --device, else TF_CUDA_DEVICE, else 0.
fn deviceOf(a: Allocator, o: api.Open, problem: *[]const u8) !?u32 {
    if (o.device) |d| return d;
    const text = if (std.c.getenv("TF_CUDA_DEVICE")) |v| std.mem.span(v) else "";
    const trimmed = std.mem.trim(u8, text, " ");
    if (trimmed.len == 0) return 0;
    return std.fmt.parseInt(u32, trimmed, 10) catch {
        problem.* = try std.fmt.allocPrint(a, "TF_CUDA_DEVICE={s}: a whole number", .{text});
        return null;
    };
}

/// The engine for `o.dir` on `device`, its head from the folder and its window from the memory budget.
fn openEngine(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8, comptime budget: type, comptime windows: type) anyerror!api.Opened {
    // `budget` and `windows` are the native module's own file imports; the family cannot open them again.
    if (try o.slideRefusal(a, "CUDA")) |why| {
        problem.* = why;
        return error.Refused;
    }
    if (o.tp != 1) {
        problem.* = tp_refusal;
        return error.Refused;
    }
    if (o.mtp_confidence != null) {
        problem.* = confidence_refusal;
        return error.Refused;
    }
    if (o.drafter != null) {
        problem.* = drafter_refusal;
        return error.Refused;
    }
    if (o.vision != null) {
        problem.* = vision_refusal;
        return error.Refused;
    }
    if (o.kv8) {
        problem.* = kv8_refusal;
        return error.Refused;
    }
    if (o.loops != null) {
        problem.* = loops_refusal;
        return error.Refused;
    }
    if (o.record != null) {
        problem.* = record_refusal;
        return error.Refused;
    }
    const device = (try deviceOf(a, o, problem)) orelse return error.Refused;
    const native = windows.modelContext(a, io, o.dir);
    const asked = windows.contextWindow(o.context, native, default_context) catch |e| {
        problem.* = switch (e) {
            error.Negative => try std.fmt.allocPrint(a, "--context {d}: a token count, or 0 for the model's window", .{o.context.?}),
            error.NoNative => "--context 0 asks for the model's window, and its config.json names none: give a token count",
            error.PastNative => try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ o.context.?, native }),
        };
        return error.Refused;
    };
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .driver = undefined, .ctx = undefined, .e = undefined, .p = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .startup = undefined };
    own.driver = cuda.Driver.open() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(err)});
        return error.Refused;
    };
    errdefer own.driver.close();
    const count: u32 = @intCast(own.driver.deviceCount() catch |err|
        return refuse(a, problem, "the CUDA driver cannot count its devices ({s})", .{@errorName(err)}));
    if (device >= count) {
        const named = if (o.device != null) "--device " else "TF_CUDA_DEVICE=";
        return refuse(a, problem, "{s}{d}: this machine has {d} CUDA device{s} (0 to {d})", .{ named, device, count, if (count == 1) "" else "s", count - 1 });
    }
    own.ctx = cuda.Context.init(&own.driver, @intCast(device)) catch |err|
        return refuse(a, problem, "the native CUDA engine cannot open device {d} ({s})", .{ device, @errorName(err) });
    errdefer own.ctx.deinit();
    const cap = try own.ctx.capability();
    const head = folder.headOf(a, io, o.dir, problem) catch |err| switch (err) {
        error.Refused => return error.Refused,
        else => |x| return x,
    };
    const normalize = switch (head) {
        .embedding => |h| h.normalize,
        .rerank => return refuse(a, problem, "{s} is a Qwen3 reranker (1_LogitScore): the CUDA engine serves Qwen3 embedding checkpoints so far", .{o.dir}),
    };
    own.e = Engine.load(gpa, io, o.dir, @intCast(device)) catch |err|
        return refuse(a, problem, "the native CUDA engine cannot load {s} ({s})", .{ o.dir, @errorName(err) });
    errdefer own.e.deinit();
    const e = own.e;
    var why: []const u8 = "";
    const got = budget.readPool(gpa, io, &own.ctx, &why) catch |err|
        return refuse(a, problem, "the native CUDA engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    const pool = got orelse {
        problem.* = why;
        return error.Refused;
    };
    const room = pool.room(e.weights);
    const window = (folder.fitWindow(@intCast(asked), room, engine.least_window, e, scratchOf) catch |err|
        return refuse(a, problem, "the native CUDA engine cannot count its scratch ({s})", .{@errorName(err)})) orelse {
        const least = e.scratchBytes(@min(@as(usize, @intCast(asked)), engine.least_window)) catch 0;
        return refuse(
            a,
            problem,
            "the CUDA memory budget fits no {d}-token pass: it takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; free device memory or lower --context",
            .{ @min(@as(usize, @intCast(asked)), engine.least_window), gibs(least), gibs(room), gibs(e.weights), gibs(pool.reserve) },
        );
    };
    e.size(window) catch |err|
        return refuse(a, problem, "the native CUDA engine cannot allocate its scratch ({s})", .{@errorName(err)});
    own.p = Prompt.init(gpa, e, normalize);
    errdefer own.p.deinit();
    own.cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    errdefer own.cfg.deinit(gpa);
    own.clock = .{ .io = io };
    own.core = lanes.Engine.init(gpa, &own.cfg, own.p.backend(), own.clock.clock());
    errdefer own.core.deinit();
    const shrunk: []const u8 = if (window < asked) " (shrunk from the asked window to fit the budget)" else "";
    own.startup = try std.fmt.allocPrint(gpa, "CUDA sm{d} device {d}: qwen3 embedding, model {d:.2} GiB; {d}-token passes{s}, {d:.2} GiB scratch, of {d:.1} GiB left after a {d:.1} GiB reserve; {s}", .{
        cap, device, gibs(e.weights), window, shrunk, gibs(e.scratchBytes(window) catch 0), gibs(room), gibs(pool.reserve), if (normalize) "vectors unit-normed" else "vectors as pooled",
    });
    errdefer gpa.free(own.startup);
    const spec = e.m.spec;
    own.host = api.LaneHost.init(gpa, io, &own.core, .{
        .lanes = 1,
        .context_window = @intCast(window),
        .text_generation = false,
        .embeddings = .{ .dimensions = @intCast(spec.hidden), .vocabulary = @intCast(spec.vocab), .max_batch_tokens = @intCast(window), .max_inputs = @intCast(engine.max_inputs), .mrl_min = @min(32, @as(u32, @intCast(spec.hidden))) },
        .reasoning = reasoning,
        .startup = own.startup,
    });
    own.host.on_thread = .{ .ctx = own, .run = bindThread };
    try own.host.start();
    return .{ .engine = own.host.engine(), .close = release, .ctx = own, .halt = halt };
}
