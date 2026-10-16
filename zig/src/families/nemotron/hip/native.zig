//! Nemotron-H for the native server on HIP: the checkpoint and its rank's bytes admitted before anything loads.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const core = @import("core");
const living = @import("../living.zig");
const admit = @import("admit.zig");
const tensors = @import("tensors.zig");
const engine = @import("engine.zig");
const plan = @import("plan.zig");
const pages = @import("pages.zig");
const hip_lanes = @import("lanes.zig");
const worker = @import("worker.zig");

pub const model_type = "nemotron_h";
pub const formats = admit.formats;
/// Positions a stream holds when neither --context nor the checkpoint's config names a window.
pub const default_context: i64 = 32768;
/// The engine cuts prompts itself.
pub const prefill_step: u32 = 0;
/// --slide is refused before the family opens: Sliding Weights does not learn on HIP yet.
pub const slide_refusal = "--slide: Sliding Weights does not learn on HIP yet; serve Nemotron without it, or on CUDA";
/// --mtp-confidence is refused too: the head's drafts are sized by measured costs, and --mtp-drafts caps them.
pub const confidence_refusal = "--mtp-confidence: Nemotron on HIP sizes its drafts by measured costs; " ++
    "serve it without --mtp-confidence (--mtp-drafts caps them)";

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

pub const Refused = admit.Refused;

fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

fn readText(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit));
}

fn exists(a: std.mem.Allocator, dir: []const u8, name: []const u8) bool {
    const path = std.fs.path.joinZ(a, &.{ dir, name }) catch return false;
    return std.c.access(path, std.c.F_OK) == 0;
}

/// The MTP head's tensor names from its file's header, or null when the file is not there.
fn mtpNames(a: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const []const u8 {
    if (!exists(a, dir, tensors.mtp_file)) return null;
    var ck: core.checkpoint.Checkpoint = .{ .gpa = a, .io = io };
    defer ck.close();
    try ck.add(dir, tensors.mtp_file);
    var out: std.ArrayList([]const u8) = .empty;
    for (ck.files.items) |*f| for (f.names.keys()) |k| try out.append(a, try a.dupe(u8, k));
    return out.items;
}

/// What admission reads from `dir`.
pub fn factsOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) Refused!admit.Facts {
    var lesson: [64]u8 = undefined;
    return .{
        .config = readText(a, io, dir, "config.json", 1 << 22) catch |err| return refuse(
            a,
            problem,
            "cannot read {s}/config.json ({t})",
            .{ dir, err },
        ),
        .index = readText(a, io, dir, "model.safetensors.index.json", 1 << 26) catch |err| return refuse(
            a,
            problem,
            "cannot read {s}/model.safetensors.index.json ({t})",
            .{ dir, err },
        ),
        .mtp = mtpNames(a, io, dir) catch |err| return refuse(
            a,
            problem,
            "cannot read {s}/{s} ({t})",
            .{ dir, tensors.mtp_file, err },
        ),
        .living = exists(a, dir, living.fileName(&lesson, 0) catch unreachable),
    };
}

/// The bytes the card `index` grants a plan.
pub fn cardRoom(gpa: std.mem.Allocator, io: std.Io, index: c_int) !u64 {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, index);
    defer ctx.deinit();
    const pool = try hip.admission.budget(gpa, io, ctx);
    return pool.room(hip.usage(false).device);
}

/// Admission before anything loads: the format, the tensors, the refusals, and whether the rank's share fits.
pub fn check(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) Refused!admit.Verdict {
    const facts = try factsOf(a, io, dir, problem);
    const room = cardRoom(gpa, io, dev.index) catch |err| return refuse(
        a,
        problem,
        "the native HIP engine cannot read the GPU's memory ({t})",
        .{err},
    );
    const act: u64 = 2;
    const ask: admit.Ask = .{ .world = o.world, .rank = o.rank, .window = o.window, .room = room, .act = act };
    const v = try admit.admit(a, facts, ask, problem);
    if (!v.fits) return admit.tooBig(a, v, ask, problem);
    return v;
}

/// One loaded engine behind the lane host.
const Owned = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    backend: *hip_lanes.Hip,
    startup: []const u8,
};

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.backend.deinit();
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.destroy(own);
}

/// What `budget` bytes of prompt cache hold: at most `slots` snapshots, up to half of it, and pages for the rest.
const Cache = struct { pages: usize, snaps: usize };

fn cacheOf(budget: u64, slots: usize, snap: u64, page: u64) Cache {
    const snaps: usize = @intCast(@min(slots, budget / 2 / @max(snap, 1)));
    return .{ .pages = @intCast((budget - snaps * snap) / @max(page, 1)), .snaps = snaps };
}

/// One rank's loaded and sized engine: the window, streams and prompt cache every rank agreed on.
const Prepared = struct { e: *engine.Engine, streams: usize, cache: Cache, startup: []const u8 };

/// This rank's refusal, its reason already in `problem`: every rank hears it at the step it is on.
fn heard(steps: *hip.Startup, problem: *[]const u8, err: Refused) Refused {
    steps.refuse(problem.*);
    return err;
}

/// Every rank, in steps over the link: rank 0's window, this rank's admission and share, the least streams and cache.
fn prepare(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    steps: *hip.Startup,
    problem: *[]const u8,
) anyerror!Prepared {
    // a follower sizes its streams for rank 0's window: the page tables rank 0 sends cover that many positions
    var mine = o;
    mine.window = @intCast((try steps.together(a, .{ o.window, 0 }, .first, problem))[0]);
    const window: usize = mine.window;
    const v = check(a, gpa, io, dev, dir, mine, problem) catch |err| return heard(steps, problem, err);
    const room = cardRoom(gpa, io, dev.index) catch |err|
        return steps.refuseAll(a, problem, "the native HIP engine cannot read the GPU's memory ({t})", .{err});
    _ = try steps.together(a, .{ 0, 0 }, .first, problem);
    const act: u64 = 2;
    const c = &v.config;
    const e = engine.Engine.load(gpa, io, dir, .{
        .device = dev.index,
        .rank = o.rank,
        .world = o.world,
        .group = dev.group,
        .startup = steps,
        .policy = dev.policy,
        .mtp = dev.policy.mtp.drafts > 0,
        .pool = false,
    }) catch |err| return switch (err) {
        error.PeerRefused, error.LinkFailed => steps.stoppedBy(a, problem),
        else => steps.refuseAll(a, problem, "the native HIP engine cannot load {s} ({t})", .{ dir, err }),
    };
    errdefer e.deinit();
    const capacity: usize = window + hip_lanes.max_window + 1;
    const token = try plan.tokenBytes(c, o.world, act, v.mtp);
    const state = try plan.streamBytes(c, o.world, act);
    const page = token * pages.tokens;
    const snap = state + c.hidden * act;
    // a lane: its pages, its Mamba state, its window's snapshots and the head's rows
    const lane: u64 = pages.pagesFor(capacity) * page + state * (hip_lanes.max_window + 2) +
        hip_lanes.max_window * c.hidden * act;
    const fixed: u64 = v.weights + admit.scratch_bytes + (64 << 20);
    if (fixed >= room) return heard(steps, problem, admit.tooBig(a, v, .{ .window = window, .room = room }, problem));
    const left = room - fixed;
    var streams: usize = @max(o.streams, 1);
    while (streams > 1 and streams * lane > left) streams -= 1;
    if (streams * lane > left) return steps.refuseAll(
        a,
        problem,
        "the HIP memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB " ++
            "is left after the model's {d:.2} GiB and the scratch; lower --context or free device memory",
        .{ window, admit.gibs(lane), admit.gibs(left), admit.gibs(v.weights) },
    );
    if (streams < o.streams and o.fixed) return steps.refuseAll(
        a,
        problem,
        "--parallel {d} needs {d:.2} GiB at a {d}-token window, and the HIP memory budget leaves {d:.2} " ++
            "GiB: serve --parallel {d}, or lower --context",
        .{ o.streams, admit.gibs(o.streams * lane), window, admit.gibs(left), streams },
    );
    streams = @intCast((try steps.together(a, .{ streams, 0 }, .least, problem))[0]);
    const spare = left - streams * lane;
    var budget: u64 = if (o.cache_gib) |g| @intFromFloat(g * (1 << 30)) else spare;
    if (budget > spare and !o.cache_over_cap) return steps.refuseAll(
        a,
        problem,
        "a {d:.2} GiB prompt cache does not fit the {d:.2} GiB the streams leave; lower --prompt-cache-gib, " ++
            "or --prompt-cache-over-cap to keep it",
        .{ admit.gibs(budget), admit.gibs(spare) },
    );
    const slots: usize = dev.policy.slots;
    if (slots == 0) budget = 0;
    var cache = cacheOf(budget, slots, snap, page);
    const fit = try steps.together(a, .{ cache.pages, cache.snaps }, .least, problem);
    cache = .{ .pages = @intCast(fit[0]), .snaps = @intCast(fit[1]) };
    e.sizePool(capacity, streams, cache.pages) catch |err|
        return steps.refuseAll(a, problem, "the native HIP engine cannot allocate its pages ({t})", .{err});
    const startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}, rank {d} of {d}: Nemotron-H model " ++
        "{d:.2} GiB, MTP head {s}; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.2} " ++
        "GiB left after the model and {d:.1} GiB of scratch; prompt cache {d:.2} GiB ({d} pages, {d} " ++
        "snapshots); prompts in {d}-row chunks; windows of {d} rows", .{
        hip.kernels.arch,
        dev.index,
        o.rank,
        o.world,
        admit.gibs(v.weights),
        if (e.model_().mtp != null) "loaded" else "off",
        streams,
        if (streams == 1) "" else "s",
        admit.gibs(lane),
        window,
        admit.gibs(left),
        admit.gibs(admit.scratch_bytes),
        admit.gibs(budget),
        cache.pages,
        cache.snaps,
        e.o.chunk,
        hip_lanes.max_window,
    });
    return .{ .e = e, .streams = streams, .cache = cache, .startup = startup };
}

/// The engine for `dir` on `dev`: admitted, the streams its memory holds sized, the lane backend over it.
pub fn open(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!Loaded {
    var steps: hip.Startup = .{ .link = if (dev.group) |g| g.link() else null };
    const p = try prepare(a, gpa, io, dev, dir, o, &steps, problem);
    errdefer p.e.deinit();
    errdefer gpa.free(p.startup);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const backend = hip_lanes.Hip.init(gpa, p.e) catch |err|
        return steps.refuseAll(a, problem, "the native HIP engine cannot start Nemotron's lanes ({t})", .{err});
    errdefer backend.deinit();
    if (p.cache.pages > 0 and p.cache.snaps > 0) backend.prefix.keepPages(p.cache.pages, p.cache.snaps);
    // rank 0 serves only once every follower is ready behind it
    _ = try steps.together(a, .{ 0, 0 }, .first, problem);
    if (dev.group) |g| backend.withLink(g.link());
    own.* = .{ .gpa = gpa, .e = p.e, .backend = backend, .startup = p.startup };
    return .{
        .backend = backend.backend(),
        .facts = backend.facts(),
        .rows = hip_lanes.max_window,
        .ctx = own,
        .deinit = release,
        .streams = p.streams,
        .startup = p.startup,
    };
}

/// A rank above 0: holds its share of the model and runs rank 0's steps until rank 0 stops.
pub fn follow(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!void {
    const group = dev.group orelse return refuse(a, problem, "rank {d} has no group to follow", .{o.rank});
    var steps: hip.Startup = .{ .link = group.link() };
    const p = try prepare(a, gpa, io, dev, dir, o, &steps, problem);
    defer p.e.deinit();
    defer gpa.free(p.startup);
    std.debug.print("[tensorfold] {s}\n", .{p.startup});
    const backend = hip_lanes.Hip.init(gpa, p.e) catch |err|
        return steps.refuseAll(a, problem, "the native HIP engine cannot start Nemotron's lanes ({t})", .{err});
    defer backend.deinit();
    if (p.cache.pages > 0 and p.cache.snaps > 0) backend.prefix.keepPages(p.cache.pages, p.cache.snaps);
    _ = try steps.together(a, .{ 0, 0 }, .first, problem);
    var w = worker.Worker.init(gpa, backend);
    try w.follow(group.link());
}

test "the registry entry names the family and its format" {
    try std.testing.expectEqualStrings("nemotron_h", model_type);
    try std.testing.expectEqualStrings("mlx-q4g64", formats[0]);
}
