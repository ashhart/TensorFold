//! Optional vocabulary-column partitioning: each rank retains the checkpoint bits for its own output rows.
const std = @import("std");
const Tensor = @import("core").safetensors.Tensor;

pub fn setting(value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadHeadSharding;
}

pub fn enabled() !bool {
    return setting(if (std.c.getenv("TF_FLASHNEXT_SHARD_HEAD")) |v| std.mem.span(v) else null);
}

pub fn shard(t: Tensor, world: usize, rank: usize) !Tensor {
    if (world != 2 or rank >= world or t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(0) % 256 != 0)
        return error.BadHeadSharding;
    const bytes = try std.math.mul(usize, try std.math.mul(usize, t.dim(0), t.dim(1)), 2);
    if (t.bytes.len != bytes) return error.UnexpectedTensor;
    var out = t;
    out.shape[0] /= world;
    out.bytes = t.bytes[rank * (bytes / world) ..][0 .. bytes / world];
    return out;
}

test "head partitions preserve every BF16 byte in global vocabulary order" {
    var bytes: [256 * 64 * 2]u8 = undefined;
    for (&bytes, 0..) |*v, i| v.* = @truncate(i * 17 + i / 251);
    const t: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 256, 64, 0, 0, 0 }, .bytes = &bytes };
    const a = try shard(t, 2, 0);
    const b = try shard(t, 2, 1);
    try std.testing.expectEqual(@as(usize, 128), a.dim(0));
    try std.testing.expectEqualSlices(u8, bytes[0 .. bytes.len / 2], a.bytes);
    try std.testing.expectEqualSlices(u8, bytes[bytes.len / 2 ..], b.bytes);
    try std.testing.expectError(error.BadHeadSharding, shard(t, 1, 0));
    try std.testing.expectError(error.BadHeadSharding, shard(t, 2, 2));
    var bad = t;
    bad.shape[0] = 255;
    try std.testing.expectError(error.BadHeadSharding, shard(bad, 2, 0));
    bad = t;
    bad.bytes = bytes[1..];
    try std.testing.expectError(error.UnexpectedTensor, shard(bad, 2, 0));
    try std.testing.expect(!try setting(null));
    try std.testing.expect(!try setting("0"));
    try std.testing.expect(try setting("1"));
    try std.testing.expectError(error.BadHeadSharding, setting("2"));
}
