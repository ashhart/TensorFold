//! Rank identity: the cached full digest by default, a given digest when set, and the wait each allows.
const std = @import("std");
const r = @import("cuda_rank_engine.zig");

test "rank identity: the cached full digest by default, a given digest when set, and the wait each allows" {
    const t = std.testing;
    const env = @import("sys").env;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}";
    var shard: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    @memcpy(shard[8 + header.len ..], "abcd");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = &shard });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"hidden_size\":8}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model.safetensors\"}}" });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    _ = env.unsetenv("TENSORFOLD_CHECKPOINT_SHA256");
    try t.expectEqual(@as(u64, 30 * 60_000), r.linkWait());
    const computed = try r.identity(t.allocator, t.io, dir, 4096, 64);
    try t.expectEqual(try @import("core").checkpoint_identity.fullIdentity(t.allocator, t.io, dir), computed.checkpoint);
    _ = env.setenv("TENSORFOLD_CHECKPOINT_SHA256", "abababababababababababababababababababababababababababababababab", 1);
    defer _ = env.unsetenv("TENSORFOLD_CHECKPOINT_SHA256");
    try t.expectEqual(@as(u64, 60_000), r.linkWait());
    try t.expectEqual(@as([32]u8, @splat(0xab)), (try r.identity(t.allocator, t.io, dir, 4096, 64)).checkpoint);
    _ = env.setenv("TENSORFOLD_CHECKPOINT_SHA256", "ab", 1);
    try t.expectError(error.NoCheckpointIdentity, r.identity(t.allocator, t.io, dir, 4096, 64));
}
