//! Nemotron-H for the native server on HIP: the checkpoint and its rank's bytes admitted before anything loads.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const core = @import("core");
const living = @import("../living.zig");
const admit = @import("admit.zig");
const tensors = @import("tensors.zig");
const mamba = @import("mamba.zig");
const moe = @import("moe.zig");

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

/// The engine for `dir` on `dev`: admitted, then refused until the Mamba mixer, the MoE layer and the lanes exist.
pub fn open(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!Loaded {
    const v = try check(a, gpa, io, dev, dir, o, problem);
    return notYet(a, v, problem);
}

/// A rank above 0: admitted like rank 0, then refused for the same reason.
pub fn follow(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!void {
    const v = try check(a, gpa, io, dev, dir, o, problem);
    return notYet(a, v, problem);
}

fn notYet(a: std.mem.Allocator, v: admit.Verdict, problem: *[]const u8) Refused {
    const missing = if (!mamba.ready and !moe.ready)
        "Mamba-2 mixer and MoE layer"
    else if (!mamba.ready)
        "Mamba-2 mixer"
    else if (!moe.ready)
        "MoE layer"
    else
        "lane host";
    return refuse(
        a,
        problem,
        "Nemotron on HIP admits this checkpoint ({d:.2} GiB a rank, at most {d} positions a stream) but " ++
            "these are not built yet: its {s}; serve with --engine python",
        .{ admit.gibs(v.weights), v.most, missing },
    );
}

test "the registry entry names the family and its format" {
    try std.testing.expectEqualStrings("nemotron_h", model_type);
    try std.testing.expectEqualStrings("mlx-q4g64", formats[0]);
}
