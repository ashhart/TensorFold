//! The hand-off's checkpoint name binds the shard bytes its load read; it refuses what it cannot bind or check.
const std = @import("std");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Bound = @import("checkpoint_identity.zig").Bound;

/// The name a folder stamped and named at once gives (nothing loads between).
fn checkpointId(a: std.mem.Allocator, io: Io, dir: []const u8) ![]const u8 {
    var b = try Bound.stamp(std.testing.allocator, io, dir);
    defer b.deinit();
    return b.name(a);
}

const head = "{\"m.layers.0.w.weight\":{\"dtype\":\"U8\",\"shape\":[4096],\"data_offsets\":[0,4096]}}";
const config = "{\"hidden_size\": 8}";

/// A one-shard checkpoint's file: the header above, then 4096 payload bytes, `flip` (if any) changed by one bit.
fn shard(buf: *[8 + head.len + 4096]u8, flip: ?usize) []const u8 {
    std.mem.writeInt(u64, buf[0..8], head.len, .little);
    @memcpy(buf[8..][0..head.len], head);
    const body = buf[8 + head.len ..];
    for (body, 0..) |*b, i| b.* = @truncate(i *% 7 + 3);
    if (flip) |at| body[at] ^= 1;
    return buf;
}

fn checkpoint(io: Io, dir: Io.Dir, sub: []const u8, flip: ?usize) !void {
    var buf: [8 + head.len + 4096]u8 = undefined;
    var d = try dir.createDirPathOpen(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = shard(&buf, flip) });
}

fn pathOf(a: std.mem.Allocator, tmp: std.testing.TmpDir, sub: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, sub });
}

test "one payload byte changes the checkpoint name; equal bytes in another folder keep it" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try checkpoint(io, tmp.dir, "a", null);
    try checkpoint(io, tmp.dir, "copy", null);
    try checkpoint(io, tmp.dir, "b", 2049); // same config, header and size: one bit of one weight byte
    const id_a = try checkpointId(a, io, try pathOf(a, tmp, "a"));
    try std.testing.expect(std.mem.startsWith(u8, id_a, "checkpoint-"));
    try std.testing.expectEqualStrings(id_a, try checkpointId(a, io, try pathOf(a, tmp, "copy")));
    const id_b = try checkpointId(a, io, try pathOf(a, tmp, "b"));
    try std.testing.expect(!std.mem.eql(u8, id_a, id_b));
}

/// A temporary folder and an arena for one refusal case.
const Case = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,

    fn init() Case {
        return .{ .arena = .init(std.testing.allocator), .tmp = std.testing.tmpDir(.{}) };
    }

    fn deinit(c: *Case) void {
        c.tmp.cleanup();
        c.arena.deinit();
    }

    fn id(c: *Case, sub: []const u8) ![]const u8 {
        const a = c.arena.allocator();
        return checkpointId(a, std.testing.io, try pathOf(a, c.tmp, sub));
    }
};

test "refused: the index lists a shard the folder lacks" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    var d = try c.tmp.dir.createDirPathOpen(io, "missing", .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\": {\"m.layers.0.w.weight\": \"model-00001-of-00001.safetensors\"}}" });
    try std.testing.expectError(error.FileNotFound, c.id("missing"));
}

test "refused: a shard's header runs past its file" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    var buf: [8 + head.len + 4096]u8 = undefined;
    const bytes = shard(&buf, null);
    std.mem.writeInt(u64, buf[0..8], 1 << 20, .little);
    var d = try c.tmp.dir.createDirPathOpen(io, "header", .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    try std.testing.expectError(error.BadHeader, c.id("header"));
}

test "refused: a Hugging Face shard one byte away from the sha256 its blob is named by" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    var buf: [8 + head.len + 4096]u8 = undefined;
    var sha: [32]u8 = undefined;
    Sha256.hash(shard(&buf, null), &sha, .{});
    const name = std.fmt.bytesToHex(sha, .lower);
    var blobs = try c.tmp.dir.createDirPathOpen(io, "hf/blobs", .{});
    defer blobs.close(io);
    try blobs.writeFile(io, .{ .sub_path = &name, .data = shard(&buf, 100) });
    var snap = try c.tmp.dir.createDirPathOpen(io, "hf/snapshots/r1", .{});
    defer snap.close(io);
    try snap.writeFile(io, .{ .sub_path = "config.json", .data = config });
    var target: [80]u8 = undefined;
    try snap.symLink(io, try std.fmt.bufPrint(&target, "../../blobs/{s}", .{&name}), "model.safetensors", .{});
    try std.testing.expectError(error.ShardBytesDiffer, c.id("hf/snapshots/r1"));
}

/// The load's stand-in: the one shard's bytes, through the descriptor the stamp opened, so a swap at the path is moot.
fn load(a: std.mem.Allocator, io: Io, b: *const Bound) ![]u8 {
    const f = b.shard(0) orelse return error.FileNotFound;
    const buf = try a.alloc(u8, std.math.cast(usize, try f.length(io)) orelse return error.FileTooBig);
    if (try f.readPositionalAll(io, buf, 0) != buf.len) return error.BadHeader;
    return buf;
}

/// Checkpoint B's shard (one weight bit away from A's) renamed over `sub`'s, as a copy or a sync lands a file.
fn replace(io: Io, tmp: std.testing.TmpDir, sub: []const u8) !void {
    var buf: [8 + head.len + 4096]u8 = undefined;
    var d = try tmp.dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "model.safetensors.new", .data = shard(&buf, 2049) });
    try Io.Dir.rename(d, "model.safetensors.new", d, "model.safetensors", io);
}

test "producer order: A loaded, then B put in its place before the name: refused, never B's name" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    const a = c.arena.allocator();
    try checkpoint(io, c.tmp.dir, "a", null);
    var bound = try Bound.stamp(std.testing.allocator, io, try pathOf(a, c.tmp, "a"));
    defer bound.deinit();
    var buf: [8 + head.len + 4096]u8 = undefined;
    try std.testing.expectEqualSlices(u8, shard(&buf, null), try load(a, io, &bound));
    try replace(io, c.tmp, "a");
    try std.testing.expectError(error.CheckpointChanged, bound.name(a));
}

test "consumer order: A stamped, then B put in its place before the load: refused, never A's name over B" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    const a = c.arena.allocator();
    try checkpoint(io, c.tmp.dir, "a", null);
    var bound = try Bound.stamp(std.testing.allocator, io, try pathOf(a, c.tmp, "a"));
    defer bound.deinit();
    try replace(io, c.tmp, "a");
    var buf: [8 + head.len + 4096]u8 = undefined;
    try std.testing.expectEqualSlices(u8, shard(&buf, 2049), try load(a, io, &bound));
    try std.testing.expectError(error.CheckpointChanged, bound.name(a));
}

test "in place: A's shard rewritten with B's bytes after the load (same inode and size): refused" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    const a = c.arena.allocator();
    try checkpoint(io, c.tmp.dir, "a", null);
    var d = try c.tmp.dir.openDir(io, "a", .{});
    defer d.close(io);
    const f = try d.openFile(io, "model.safetensors", .{ .mode = .read_write });
    defer f.close(io);
    // a modify time long past: the rewrite's own time differs from it on every clock granularity
    try f.setTimestamps(io, .{ .modify_timestamp = .{ .new = .fromNanoseconds(1_000_000_000) } });
    var bound = try Bound.stamp(std.testing.allocator, io, try pathOf(a, c.tmp, "a"));
    defer bound.deinit();
    _ = try load(a, io, &bound);
    var buf: [8 + head.len + 4096]u8 = undefined;
    try f.writePositionalAll(io, shard(&buf, 2049), 0);
    try std.testing.expectError(error.CheckpointChanged, bound.name(a));
}

test "a stale digest cache entry is not consulted: the name is the bytes' own" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    const a = c.arena.allocator();
    try checkpoint(io, c.tmp.dir, "a", null);
    const dir = try pathOf(a, c.tmp, "a");
    const truth = try checkpointId(a, io, dir);
    // a cache root holding a record for this very file, with a name nobody's bytes have
    const root = try std.fs.path.join(a, &.{ ".zig-cache/tmp", c.tmp.sub_path, "cache" });
    try Io.Dir.cwd().createDirPath(io, root);
    var d = try Io.Dir.cwd().openDir(io, root, .{});
    defer d.close(io);
    const shard_file = try d.openFile(io, "../../a/model.safetensors", .{});
    defer shard_file.close(io);
    const key = try @import("digest_cache.zig").Key.of(shard_file, io);
    const stale = [_]@import("digest_cache.zig").Record{.{ .key = key, .value = .{ .sha256 = @splat(0) } }};
    const bytes = try @import("digest_cache.zig").encode(a, &stale);
    try d.writeFile(io, .{ .sub_path = "index.bin", .data = bytes });
    var b = try Bound.stampWithCache(std.testing.allocator, io, dir, root);
    defer b.deinit();
    _ = try load(a, io, &b);
    try std.testing.expectEqualStrings(truth, try b.name(a));
}

test "untouched: the name after the load is the folder's own, the one an equal copy elsewhere gives" {
    const io = std.testing.io;
    var c: Case = .init();
    defer c.deinit();
    const a = c.arena.allocator();
    try checkpoint(io, c.tmp.dir, "a", null);
    try checkpoint(io, c.tmp.dir, "copy", null);
    const dir = try pathOf(a, c.tmp, "a");
    var bound = try Bound.stamp(std.testing.allocator, io, dir);
    defer bound.deinit();
    _ = try load(a, io, &bound);
    const named = try bound.name(a);
    try std.testing.expectEqualStrings(try c.id("copy"), named);
    // the same name fullIdentity gives: reading the descriptors must not change what a name means
    const full = try @import("checkpoint_identity.zig").fullWithCache(std.testing.allocator, io, dir, null);
    const short = full[0..8];
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "checkpoint-{x}", .{short}), named);
}
