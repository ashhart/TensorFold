//! `tensorfold merge DENSE ADAPTER OUT --quantize SHIPPED`: LoRA into dense weights, quantized fresh, rest copied.
const std = @import("std");
const se = @import("shard_edit.zig");
const lm = @import("lora_merge.zig");
const lp = @import("lora_plan.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const safetensors = @import("safetensors.zig");
pub const math = lm;
pub const plan = lp;
pub const gate = @import("lora_gate.zig");

/// Runs the merge; refusals and failures go to `err` and the exit code is 1.
pub fn run(gpa: Allocator, io: Io, out: *Io.Writer, err: *Io.Writer, dense: []const u8, adapter: []const u8, dest: []const u8, shipped: []const u8) !u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    merge(gpa, arena_state.allocator(), io, out, err, dense, adapter, dest, shipped) catch |e| {
        if (e != error.Refused) err.print("merge: {s}\n", .{@errorName(e)}) catch {};
        return 1;
    };
    return 0;
}

fn merge(gpa: Allocator, arena: Allocator, io: Io, out: *Io.Writer, err: *Io.Writer, dense_dir: []const u8, adapter: []const u8, dest: []const u8, shipped_dir: []const u8) !void {
    var dense = try lp.Shards.init(gpa, arena, io, err, dense_dir);
    defer dense.deinit();
    var shipped = try lp.Shards.init(gpa, arena, io, err, shipped_dir);
    defer shipped.deinit();
    var p = try lp.plan(gpa, arena, io, err, &dense, &shipped, adapter);
    defer p.deinit(io);
    try checkSource(io, err, shipped_dir);
    try freeDest(io, err, dest);
    const stage = try makeStage(arena, io, dest);
    errdefer Io.Dir.cwd().deleteTree(io, stage) catch {};
    try out.print("copying {s} to {s}\n", .{ shipped_dir, stage });
    try copyFolder(io, shipped_dir, stage);
    for (p.jobs) |j| {
        const w = try lp.denseOf(gpa, err, &dense, j);
        defer gpa.free(w);
        const old = try lp.quantOf(gpa, &shipped, j);
        defer lp.free(gpa, old);
        const f = try lp.factors(gpa, err, &p, j);
        defer gpa.free(f.a);
        defer gpa.free(f.b);
        const new = try lm.allocQuant(gpa, j.n, j.k, j.group);
        defer lm.freeQuant(gpa, new);
        var bad: usize = 0;
        lm.mergeDense(gpa, j.group, j.n, j.k, w, .{ .a = f.a, .b = f.b, .r = p.r, .scale = p.scale }, new, &bad) catch |e| switch (e) {
            error.NonFinite => return lp.refuse(err, j.module, try std.fmt.allocPrint(arena, "group {d}: a non-finite value, range or scale", .{bad})),
            else => return e,
        };
        const changed = lm.differingGroups(j.group, lm.viewOf(old), lm.viewOf(new));
        try out.print("{s}: {d} x {d}, group {d}, {d} of {d} groups differ from the shipped tensor\n", .{ j.module, j.n, j.k, j.group, changed, old.scales.len });
        if (changed == 0) continue;
        var names: [3][]const u8 = undefined;
        for (&names, [3][]const u8{ "weight", "scales", "biases" }) |*x, leaf| x.* = try std.fmt.allocPrint(arena, "{s}.{s}", .{ j.shippedName(), leaf });
        try regularShard(arena, io, err, stage, shipped.shardOf(names[0]).?);
        const shapes = [3][2]usize{ .{ j.n, j.k / 8 }, .{ j.n, j.k / j.group }, .{ j.n, j.k / j.group } };
        const edits = [3]se.Replacement{
            .{ .name = names[0], .dtype = j.dtype, .shape = &shapes[0], .bytes = std.mem.sliceAsBytes(new.words) },
            .{ .name = names[1], .dtype = .bf16, .shape = &shapes[1], .bytes = std.mem.sliceAsBytes(new.scales) },
            .{ .name = names[2], .dtype = .bf16, .shape = &shapes[2], .bytes = std.mem.sliceAsBytes(new.biases) },
        };
        try se.overwrite(gpa, io, stage, &edits);
    }
    try publish(io, err, stage, dest);
    try out.print("merged {d} projections, r = {d}, alpha / r = {d}; wrote {s}\n", .{ p.jobs.len, p.r, p.scale, dest });
}

/// Refuses a source folder holding anything but regular files (or symlinks to them): only flat folders are copied.
fn checkSource(io: Io, err: *Io.Writer, base: []const u8) !void {
    var src = try Io.Dir.cwd().openDir(io, base, .{ .iterate = true });
    defer src.close(io);
    var it = src.iterate();
    while (try it.next(io)) |e| {
        if (!lp.isRegular(src, io, e)) return lp.refuse(err, e.name, "not a regular file (a subdirectory, special file or dangling symlink)");
    }
}

/// Moves `stage` to `dest`: rename replaces an empty folder atomically and fails on a non-empty one.
pub fn publish(io: Io, err: *Io.Writer, stage: []const u8, dest: []const u8) !void {
    Io.Dir.cwd().rename(stage, Io.Dir.cwd(), dest, io) catch |e| switch (e) {
        error.DirNotEmpty => return lp.refuse(err, dest, "OUT must be a new or empty folder"),
        else => return e,
    };
}

fn freeDest(io: Io, err: *Io.Writer, dest: []const u8) !void {
    var d = Io.Dir.cwd().openDir(io, dest, .{ .iterate = true }) catch |e| switch (e) {
        error.FileNotFound => return,
        error.NotDir => return lp.refuse(err, dest, "OUT must be a new or empty folder"),
        else => return e,
    };
    defer d.close(io);
    var it = d.iterate();
    if (try it.next(io) != null) return lp.refuse(err, dest, "OUT must be a new or empty folder");
}

/// A sibling folder of `dest`, created exclusively, where the merge is staged until it has all succeeded.
fn makeStage(arena: Allocator, io: Io, dest: []const u8) ![]const u8 {
    const cwd = Io.Dir.cwd();
    const trimmed = std.mem.trimEnd(u8, dest, "/");
    if (std.fs.path.dirname(trimmed)) |parent| try cwd.createDirPath(io, parent);
    const stamp = Io.Clock.awake.now(io).toNanoseconds();
    for (0..100) |attempt| {
        const path = try std.fmt.allocPrint(arena, "{s}.merging-{d}-{d}", .{ trimmed, stamp, attempt });
        cwd.createDir(io, path, .default_dir) catch |e| switch (e) {
            error.PathAlreadyExists => continue,
            else => return e,
        };
        return path;
    }
    return error.PathAlreadyExists;
}

/// `shard` in `dir` must be a regular file, never a symlink a write would pass through.
fn regularShard(arena: Allocator, io: Io, err: *Io.Writer, dir: []const u8, shard: []const u8) !void {
    const st = try Io.Dir.cwd().statFile(io, try std.fs.path.join(arena, &.{ dir, shard }), .{ .follow_symlinks = false });
    if (st.kind != .file) return lp.refuse(err, shard, "a shard that is not a regular file");
}

/// Every file of `base` (symlinks followed) copied byte for byte into `dest`.
fn copyFolder(io: Io, base: []const u8, dest: []const u8) !void {
    const cwd = Io.Dir.cwd();
    var src = try cwd.openDir(io, base, .{ .iterate = true });
    defer src.close(io);
    var dst = try cwd.openDir(io, dest, .{});
    defer dst.close(io);
    var it = src.iterate();
    while (try it.next(io)) |e| {
        if (!lp.isRegular(src, io, e)) return error.NotRegular;
        try Io.Dir.copyFile(src, e.name, dst, e.name, io, .{});
    }
}

test {
    _ = @import("lora_merge_dir_test.zig");
    _ = @import("lora_gate_test.zig");
    _ = @import("lora_quantizer_test.zig");
}
