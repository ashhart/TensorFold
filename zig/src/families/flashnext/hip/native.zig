//! Flash Next for the native server on HIP: the checkpoint and its rank's bytes admitted before anything loads.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const config = @import("../config.zig");
const index = @import("../index.zig");
const plan = @import("plan.zig");
const Spec = @import("../spec.zig").Spec;
const Engine = @import("engine.zig").Engine;
const cache = @import("cache.zig");
const lanes_mod = @import("lanes.zig");

pub const model_type = "qwen4_exp";
/// The MLX affine checkpoints the plan admits; the HIP products read 2- to 8-bit codes in groups of 32 to 128.
pub const formats: []const []const u8 = &.{ "mlx-q3g32", "mlx-q4g32", "mlx-q6g32" };
/// Positions a stream holds when neither --context nor the checkpoint's config names a window.
pub const default_context: i64 = 32768;
/// The engine cuts prompts itself.
pub const prefill_step: u32 = 0;
/// Device bytes a rank keeps for the forward's scratch until the engine counts its own.
pub const scratch_bytes: u64 = 2 << 30;

/// What the native server drives, as the other HIP families hand it over.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
};

pub const Refused = error{Refused};

fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// The checkpoint's verdict: its format and tensors, and whether a rank's share and one stream fit `room`.
pub const Verdict = struct {
    spec: Spec,
    weights: u64,
    host: u64,
    /// The longest window one stream may hold beside the weights and scratch.
    most: u64,
    fits: bool,
};

/// Admits `c` with the index text `index_json` on rank `rank` of `world` at `window` positions in `room` bytes.
pub fn admit(a: std.mem.Allocator, gpa: std.mem.Allocator, c: *const config.Config, index_json: []const u8, world: u32, rank: u32, window: u64, room: u64, problem: *[]const u8) Refused!Verdict {
    const spec = c.global_affine;
    if (spec.group != 32) return refuse(a, problem, "Flash Next on HIP reads group-32 MLX affine checkpoints; this one has groups of {d}", .{spec.group});
    _ = index.admit(gpa, index_json, c) catch |err| return refuse(a, problem, "the checkpoint's index lacks a Flash Next tensor ({s})", .{@errorName(err)});
    const w = plan.weights(c, spec, world) catch |err| return refuse(a, problem, "--tp {d} does not split Flash Next's heads, experts and vocabulary evenly ({s})", .{ world, @errorName(err) });
    const token = plan.tokenBytes(c, world, 2) catch return refuse(a, problem, "Flash Next's layer types are unreadable", .{});
    const stream = plan.streamBytes(c, world) catch return refuse(a, problem, "Flash Next's DeltaNet heads do not split over --tp {d}", .{world});
    const most = plan.Fit.window(room, w.device(rank), scratch_bytes, token, stream);
    return .{ .spec = spec, .weights = w.device(rank), .host = w.host, .most = most, .fits = most >= window };
}

fn readIndex(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
}

/// The bytes this rank's card grants a plan.
fn cardRoom(gpa: std.mem.Allocator, io: std.Io, dev: hip.Device) !u64 {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, dev.index);
    defer ctx.deinit();
    const pool = try hip.admission.budget(gpa, io, ctx);
    return pool.room(hip.usage(false).device);
}

/// Admission before anything loads: the format, the tensors and whether the rank's share and one stream fit.
fn check(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype, problem: *[]const u8) Refused!void {
    var c = config.Config.read(gpa, io, dir) catch |err| return refuse(a, problem, "{s} is not a Flash Next checkpoint the HIP engine reads ({s})", .{ dir, @errorName(err) });
    defer c.deinit();
    const text = readIndex(a, io, dir) catch |err| return refuse(a, problem, "cannot read {s}/model.safetensors.index.json ({s})", .{ dir, @errorName(err) });
    const bytes = cardRoom(gpa, io, dev) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    if (o.world > 1) return refuse(a, problem, "Flash Next on HIP runs on one rank here: --tp {d} is not served", .{o.world});
    const v = admit(a, gpa, &c, text, o.world, o.rank, o.window, bytes, problem) catch |err| return err;
    if (!v.fits) return refuse(a, problem, "Flash Next {d}-bit on {d} rank(s) holds {d:.2} GiB of weights a rank ({d:.1} GiB of n-gram tables in host memory) and {d:.1} GiB of scratch, of the {d:.2} GiB this card grants: a {d}-token window does not fit (at most {d}); lower --context, add ranks (--tp), or use a lower-bit checkpoint", .{ v.spec.bits, o.world, gibs(v.weights), gibs(v.host), gibs(scratch_bytes), gibs(bytes), o.window, v.most });
}

/// The positions a stream holds: the window and a verify's rows past it, on 64.
fn capacity(window: usize) usize {
    return std.mem.alignForward(usize, window + cache.max_window + 1, 64);
}

fn engineOf(gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype) !*Engine {
    return Engine.open(gpa, io, dir, .{ .device = dev.index, .policy = dev.policy, .capacity = capacity(o.window), .chunk = @max(dev.policy.prefill_step, dev.policy.fill_rows) });
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    fx: *lanes_mod.Fx,
    startup: []const u8,

    fn release(p: *anyopaque) void {
        const own: *Owned = @ptrCast(@alignCast(p));
        own.fx.deinit();
        own.e.deinit();
        own.gpa.free(own.startup);
        own.gpa.destroy(own);
    }
};

/// The engine for `dir` on `dev`; under tensor parallelism this is rank 0, which serves.
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype, problem: *[]const u8) anyerror!Loaded {
    try check(a, gpa, io, dev, dir, o, problem);
    const e = engineOf(gpa, io, dev, dir, o) catch |err| return refuse(a, problem, "the native HIP engine cannot load Flash Next from {s} ({s})", .{ dir, @errorName(err) });
    errdefer e.deinit();
    // the streams the memory left after the weights and the scratch holds at the window
    const room = (hip.admission.budget(gpa, io, e.ctx) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)})).room(hip.usage(false).device);
    const per = streamBytes(e, capacity(o.window));
    const fit = room / per;
    if (fit == 0) return refuse(a, problem, "one stream at a {d}-token window takes {d:.2} GiB and {d:.2} GiB is left after the weights; lower --context or add ranks (--tp)", .{ o.window, gibs(per), gibs(room) });
    if (fit < o.streams and o.fixed) return refuse(a, problem, "--parallel {d} needs {d:.2} GiB at a {d}-token window and {d:.2} GiB is left: serve --parallel {d}, or lower --context", .{ o.streams, gibs(per * o.streams), o.window, gibs(room), fit });
    const streams = @min(fit, @max(o.streams, 1));
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const fx = try lanes_mod.Fx.init(gpa, e, streams);
    errdefer fx.deinit();
    // kept prompt states live in host memory: --prompt-cache-gib, else 16 GiB
    fx.keepPrompts(@intFromFloat((o.cache_gib orelse 16.0) * (1 << 30)));
    const startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}, rank {d} of {d}: Flash Next, {d:.2} GiB on the device, {d} stream{s} of {d} positions, {d:.2} GiB each; prompts in {d}-row chunks", .{ hip.kernels.arch, dev.index, o.rank, o.world, gibs(hip.usage(false).device), streams, if (streams == 1) "" else "s", capacity(o.window), gibs(per), e.o.chunk });
    own.* = .{ .gpa = gpa, .e = e, .fx = fx, .startup = startup };
    return .{ .backend = fx.backend(), .facts = fx.facts(), .rows = cache.max_window, .ctx = own, .deinit = Owned.release, .streams = streams, .startup = startup };
}

/// Device bytes a stream takes at `cap` positions: its caches, window snapshots, the head's caches and its rows.
fn streamBytes(e: *const Engine, cap: usize) usize {
    var buf: [config.max_layers]bool = undefined;
    const kinds = e.kinds(&buf) catch return std.math.maxInt(usize);
    const l = cache.Layout.of(e.weights.model.dims, kinds, cap, e.act);
    var head: usize = 0;
    if (e.weights.model.mtp) |h| {
        const hl = cache.Layout.of(h.dims, &.{false}, cap, e.act);
        head = hl.bytes() + hl.cinBytes() + 4;
    }
    return l.bytes() + l.windowBytes() + l.cinBytes() + head + cache.max_window * e.weights.model.dims.wide() * e.act.size();
}

/// A rank above 0: refused, the engine runs on one rank.
pub fn follow(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype, problem: *[]const u8) anyerror!void {
    try check(a, gpa, io, dev, dir, o, problem);
}

const gib: u64 = 1 << 30;

fn fixture(bits: u8) !config.Config {
    var buf: [16]u8 = undefined;
    const text = try std.mem.replaceOwned(u8, std.testing.allocator, @embedFile("../fixtures/config.json"), "\"bits\": 6", try std.fmt.bufPrint(&buf, "\"bits\": {d}", .{bits}));
    defer std.testing.allocator.free(text);
    return config.parse(std.testing.allocator, text);
}

test "admission: the 4-bit checkpoint on four V620s takes the model's window, on one it is refused with a reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try fixture(4);
    defer c.deinit();
    const idx = @embedFile("../fixtures/index.json");
    var problem: []const u8 = "";
    const four = try admit(a, std.testing.allocator, &c, idx, 4, 0, 262144, 28 * gib, &problem);
    try std.testing.expect(four.fits and four.most >= 262144);
    const one = try admit(a, std.testing.allocator, &c, idx, 1, 0, 4096, 28 * gib, &problem);
    try std.testing.expect(!one.fits and one.most == 0);
    try std.testing.expectError(error.Refused, admit(a, std.testing.allocator, &c, idx, 3, 0, 4096, 28 * gib, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "--tp 3") != null);
    try std.testing.expectError(error.Refused, admit(a, std.testing.allocator, &c, "{\"weight_map\": {\"x\": \"a.safetensors\"}}", 4, 0, 4096, 28 * gib, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "lacks") != null);
}

test "admission: on two W7800s the 3-bit checkpoint takes the model's window and the 4-bit one a shorter window" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const idx = @embedFile("../fixtures/index.json");
    var problem: []const u8 = "";
    var q3 = try fixture(3);
    defer q3.deinit();
    var q4 = try fixture(4);
    defer q4.deinit();
    const three = try admit(a, std.testing.allocator, &q3, idx, 2, 0, 262144, 43 * gib, &problem);
    const four = try admit(a, std.testing.allocator, &q4, idx, 2, 0, 262144, 43 * gib, &problem);
    try std.testing.expect(three.fits);
    try std.testing.expect(!four.fits and four.most >= 65536);
    try std.testing.expect(four.weights > three.weights and four.host > three.host);
}
