//! Captured CUDA serving identity covers the manifest and the cubins the loader reads.
const std = @import("std");
const aot = @import("aot.zig");

pub fn hash(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    var h = std.hash.Wyhash.init(0x616f74);
    try write(&h, a, io, dir);
    return h.final();
}

/// The manifest's text, then each cubin it names by its SHA-256, into any hasher with `update(bytes)`.
pub fn write(h: anytype, a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, "aot.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    defer a.free(text);
    const parsed = try aot.parseSpecs(a, text);
    defer parsed.deinit();
    h.update(text);
    for (parsed.value.kernels) |k| {
        const cubin = try std.fmt.allocPrint(a, "{s}/cubins/{s}.cubin", .{ dir, k.hash });
        defer a.free(cubin);
        const digest = try @import("sys").file_identity.hash(io, cubin, 1 << 26);
        h.update(&digest);
    }
}

test "persistence identity: captured CUDA cubin replacement moves an unchanged manifest key" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const manifest = "{\"kernels\":[{\"fn\":\"f\",\"hash\":\"h\",\"name\":\"k\",\"num_warps\":4,\"params\":[],\"consts\":{}}]}";
    try tmp.dir.writeFile(io, .{ .sub_path = "aot.json", .data = manifest });
    try tmp.dir.createDir(io, "cubins", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "cubins/h.cubin", .data = "first cubin" });
    const before = try hash(a, io, dir);
    try std.testing.expectEqual(before, try hash(a, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "cubins/h.cubin", .data = "other cubin" });
    try std.testing.expect(before != try hash(a, io, dir));
    try tmp.dir.deleteFile(io, "cubins/h.cubin");
    try std.testing.expectError(error.FileNotFound, hash(a, io, dir));
}
