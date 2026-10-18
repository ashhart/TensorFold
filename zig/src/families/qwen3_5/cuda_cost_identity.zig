//! Checkpoint structure and descriptor stamps identify the drafter without hashing its weight payload again.
const std = @import("std");
const core = @import("core");

pub fn checkpoint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    const structural = try core.checkpoint_identity.identity(a, io, dir);
    hash.update(&structural);
    hash.update(dir);
    var names: std.ArrayList([]const u8) = .empty;
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26))) |bytes| {
        const value = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        const map = if (value == .object) value.object.get("weight_map") else null;
        if (map == null or map.? != .object) return error.BadIndex;
        for (map.?.object.values()) |item| {
            if (item != .string) return error.BadIndex;
            try names.append(a, item.string);
        }
    } else |err| switch (err) {
        error.FileNotFound => try names.append(a, "model.safetensors"),
        else => return err,
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    var previous: ?[]const u8 = null;
    for (names.items) |name| {
        if (previous) |p| if (std.mem.eql(u8, p, name)) continue;
        previous = name;
        const file = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, name }), .{});
        defer file.close(io);
        const stamp = try core.digest_cache.Key.of(file, io);
        var text: [256]u8 = undefined;
        hash.update(name);
        hash.update(try std.fmt.bufPrint(&text, "\n{d}/{d}/{d}/{d}/{d}\n", .{ stamp.device, stamp.inode, stamp.size, stamp.mtime, stamp.ctime }));
    }
    return hash.finalResult();
}

test "cost identity: stable reads reuse a drafter, while a changed payload at the same path invalidates it" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"test\"}" });
    const header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}";
    var bytes: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
    const first = try checkpoint(a, io, root);
    try std.testing.expectEqualSlices(u8, &first, &try checkpoint(a, io, root));
    bytes[bytes.len - 1] = 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
    const changed = try checkpoint(a, io, root);
    try std.testing.expect(!std.mem.eql(u8, &first, &changed));
}
