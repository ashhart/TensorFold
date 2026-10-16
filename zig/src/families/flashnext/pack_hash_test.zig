//! The uniform 6-bit/group-32 packs keep their bytes: one digest over every tensor's name, dtype, shape and payload.
const std = @import("std");
const pack = @import("pack.zig");
const fixture = @import("pack_test.zig");
const st = @import("../../core/safetensors.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

/// The digest of the synthetic 6/32 checkpoint's three packs' tensors: a uniform pack keeps every byte.
const uniform_digest = "21fa29e16708022181ed842263cc03478cf76de2e213426f94feb05b58d8112e";

fn absorb(h: *std.crypto.hash.sha2.Sha256, a: std.mem.Allocator, path: []const u8) !void {
    var file = try st.File.open(a, io, path);
    defer file.close(io);
    const names = try a.dupe([]const u8, file.names.keys());
    std.mem.sort([]const u8, names, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    for (names) |name| {
        const e = file.names.get(name).?;
        h.update(name);
        h.update(&.{ @backingInt(e.dtype), e.rank });
        for (e.shape[0..e.rank]) |d| h.update(std.mem.asBytes(&@as(u64, d)));
        h.update(file.map.memory[file.data + e.begin .. file.data + e.end]);
    }
}

test "the uniform 6-bit group-32 packs hash as they did before mixed widths" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeCheckpoint(tmp, a);
    try tmp.dir.createDirPath(io, "out");
    _ = try pack.build(a, io, try fixture.tmpPath(a, tmp, "."), try fixture.tmpPath(a, tmp, "out"), null);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ "out/pack.safetensors", "out/pack_mlx.safetensors", "out/pack_mtp_mlx.safetensors" }) |name| {
        try absorb(&h, a, try std.fs.path.join(a, &.{ try fixture.tmpPath(a, tmp, "."), name }));
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    try std.testing.expectEqualStrings(uniform_digest, &std.fmt.bytesToHex(digest, .lower));
}
