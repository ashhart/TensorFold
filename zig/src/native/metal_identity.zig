//! The identity a Metal host's learned and disk states read under: checkpoint, shard stamps, MTP header, a probe.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const nemotron = @import("tensorfold").nemotron;
const Allocator = std.mem.Allocator;

/// What learned and disk states read under: checkpoint, shards, prefill step, head, kernels, a probe, OS build, chip.
pub fn learned(a: Allocator, io: std.Io, dir: []const u8, b: *nemotron.backend.Metal) !struct { id: u64, rate: f64 } {
    var h = std.hash.Wyhash.init(0x6e65);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        h.update(try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)));
    }
    try shardStamps(&h, a, io, dir);
    if (b.head != null) h.update(try safetensorsHeader(a, dir, "mtp-4bit.safetensors"));
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const probe = try nemotron.learned_prompts.probe(b);
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    for ([_]u64{ b.o.chunk, @intFromBool(b.head != null), nemotron.kernels.sourceHash(), probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(b.m.device.name()));
    return .{ .id = h.final(), .rate = 4143.0 / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// Each safetensors file's name, size, inode and change times: a weight written in place (--slide) moves the identity.
fn shardStamps(h: *std.hash.Wyhash, a: Allocator, io: std.Io, dir: []const u8) !void {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items) |name| {
        const info = try d.statFile(io, name, .{});
        const stamp = [4]i128{ info.size, info.inode, info.mtime.nanoseconds, info.ctime.nanoseconds };
        h.update(name);
        h.update(std.mem.sliceAsBytes(&stamp));
    }
}

/// A safetensors file's header (its tensors' names, shapes and offsets), in `a`.
fn safetensorsHeader(a: Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = platform.c.close(fd);
    var n: u64 = 0;
    if (!api.prompt_imprint.readAt(fd, std.mem.asBytes(&n), 0) or n > 1 << 24) return error.BadHeader;
    const head = try a.alloc(u8, @intCast(n));
    if (!api.prompt_imprint.readAt(fd, head, 8)) return error.BadHeader;
    return head;
}

test "a shard rewritten in place, same size and bytes, gives the learned states a new identity" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = "weights" });
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "not a shard" });
    var first = std.hash.Wyhash.init(0);
    try shardStamps(&first, a, io, dir);
    var again = std.hash.Wyhash.init(0);
    try shardStamps(&again, a, io, dir);
    try std.testing.expectEqual(first.final(), again.final());
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "other text" }); // not a shard: no change
    var notes = std.hash.Wyhash.init(0);
    try shardStamps(&notes, a, io, dir);
    try std.testing.expectEqual(first.final(), notes.final());
    std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model-00001-of-00001.safetensors", .{dir}, 0);
    const fd = platform.c.open(path, .{ .ACCMODE = .WRONLY }, @as(platform.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, 7), platform.c.pwrite(fd, "weights", 7, 0)); // as Sliding Weights writes a tensor of the same size
    _ = platform.c.close(fd);
    var moved = std.hash.Wyhash.init(0);
    try shardStamps(&moved, a, io, dir);
    try std.testing.expect(first.final() != moved.final());
}
