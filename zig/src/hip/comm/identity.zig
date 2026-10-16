//! What a rank runs, digested for its join: the executable, the checkpoint's config.json, its index and its shards.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Identity = @import("../agree.zig").Identity;

/// Blocks of each shard's tensor bytes the weights' digest reads, spread evenly from its first byte to its last.
const samples = 64;
const block = 4096;

/// This process's executable and the checkpoint in `dir` (Linux, where the HIP engine runs).
pub fn ofThis(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Identity {
    return of(gpa, io, "/proc/self/exe", dir);
}

/// The digests of `program` (this executable) and of the checkpoint in `dir`; paths are not part of them.
pub fn of(gpa: std.mem.Allocator, io: std.Io, program: []const u8, dir: []const u8) !Identity {
    const config = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(config);
    return .{
        .program = try fileDigest(gpa, io, program),
        .config = try fileDigest(gpa, io, config),
        .weights = try weightsDigest(gpa, io, dir),
    };
}

/// sha256 of a whole file, read in pieces.
fn fileDigest(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = try file.length(io);
    const buf = try gpa.alloc(u8, 4 << 20);
    defer gpa.free(buf);
    var h = Sha256.init(.{});
    var at: u64 = 0;
    while (at < size) {
        const n = try file.readPositionalAll(io, buf[0..@intCast(@min(buf.len, size - at))], at);
        if (n == 0) return error.FileShrank;
        h.update(buf[0..n]);
        at += n;
    }
    return h.finalResult();
}

/// sha256 of the weight index (when there is one) and, shard by shard in name order, its size, header and samples.
fn weightsDigest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var h = Sha256.init(.{});
    var names: std.ArrayList([]const u8) = .empty;
    const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26))) |text| {
        h.update(text);
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (root != .object) return error.BadIndex;
        const map = root.object.get("weight_map") orelse return error.BadIndex;
        if (map != .object) return error.BadIndex;
        for (map.object.values()) |v| if (v == .string) try names.append(a, v.string);
    } else |err| switch (err) {
        error.FileNotFound => try names.append(a, "model.safetensors"),
        else => return err,
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    var last: []const u8 = "";
    for (names.items) |name| {
        if (std.mem.eql(u8, name, last)) continue;
        last = name;
        try shardDigest(a, io, try std.fs.path.join(a, &.{ dir, name }), &h);
    }
    return h.finalResult();
}

/// A shard's size, its header, and `samples` blocks of its tensor bytes (all of them when they are fewer).
fn shardDigest(a: std.mem.Allocator, io: std.Io, path: []const u8, h: *Sha256) !void {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = try file.length(io);
    var head: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadHeader;
    const n = std.mem.readInt(u64, &head, .little);
    if (n > 1 << 30 or 8 + n > size) return error.BadHeader;
    h.update(std.mem.asBytes(&size));
    const header = try a.alloc(u8, @intCast(n));
    if (try file.readPositionalAll(io, header, 8) != header.len) return error.BadHeader;
    h.update(header);
    const start = 8 + n;
    const span = size - start;
    var buf: [block]u8 = undefined;
    if (span <= samples * block) {
        var at = start;
        while (at < size) {
            const got = try file.readPositionalAll(io, buf[0..@intCast(@min(block, size - at))], at);
            if (got == 0) return error.FileShrank;
            h.update(buf[0..got]);
            at += got;
        }
        return;
    }
    for (0..samples) |i| {
        const at = start + @as(u64, i) * (span - block) / (samples - 1);
        if (try file.readPositionalAll(io, &buf, at) != block) return error.FileShrank;
        h.update(&buf);
    }
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

/// A one-shard checkpoint in `sub` of `dir`: config.json and a header naming `bytes` of one U8 tensor.
fn writeCheckpoint(io: std.Io, dir: std.Io.Dir, sub: []const u8, config: []const u8, bytes: []const u8) !void {
    var buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(
        &buf,
        "{{\"w\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}",
        .{ bytes.len, bytes.len },
    );
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(std.testing.allocator);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, header.len, .little);
    try file.appendSlice(std.testing.allocator, &len);
    try file.appendSlice(std.testing.allocator, header);
    try file.appendSlice(std.testing.allocator, bytes);
    try dir.createDirPath(io, sub);
    var d = try dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
}

test "a checkpoint's identity follows its contents, not its path: config, header and sampled bytes" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(root);
    const big = try gpa.alloc(u8, 4 * samples * block);
    defer gpa.free(big);
    for (big, 0..) |*b, i| b.* = @truncate(i *% 7);
    try writeCheckpoint(io, tmp.dir, "a", "{\"n\":1}", big);
    try writeCheckpoint(io, tmp.dir, "b/elsewhere", "{\"n\":1}", big);
    big[0] +%= 1;
    try writeCheckpoint(io, tmp.dir, "c", "{\"n\":1}", big);
    big[0] -%= 1;
    try writeCheckpoint(io, tmp.dir, "d", "{\"n\":2}", big);
    const small = [_]u8{ 1, 2, 3, 4, 5 };
    try writeCheckpoint(io, tmp.dir, "e", "{\"n\":1}", &small);
    try writeCheckpoint(io, tmp.dir, "f", "{\"n\":1}", &.{ 1, 2, 3, 4, 6 });
    var ids: [6]Identity = undefined;
    for (&ids, [_][]const u8{ "a", "b/elsewhere", "c", "d", "e", "f" }) |*id, sub| {
        const dir = try std.fs.path.join(gpa, &.{ root, sub });
        defer gpa.free(dir);
        const program = try std.fs.path.join(gpa, &.{ root, "a", "config.json" });
        defer gpa.free(program);
        id.* = try of(gpa, io, program, dir);
    }
    try std.testing.expectEqual(ids[0], ids[1]);
    try std.testing.expect(!std.mem.eql(u8, &ids[0].weights, &ids[2].weights));
    try std.testing.expectEqualSlices(u8, &ids[0].config, &ids[2].config);
    try std.testing.expect(!std.mem.eql(u8, &ids[0].config, &ids[3].config));
    try std.testing.expectEqualSlices(u8, &ids[0].weights, &ids[3].weights);
    try std.testing.expect(!std.mem.eql(u8, &ids[4].weights, &ids[5].weights));
}
