//! Indexed subfolders are admitted loader inputs that a top-level scan cannot stamp.
const std = @import("std");
const disk = @import("src/native/disk_identity.zig");
const nemo = @import("src/native/metal_identity.zig");
const fixture = @import("src/native/persistence_fixture.zig");
const t = std.testing;
const Family = enum { nemo, qwen, glm, flash };

fn key(a: std.mem.Allocator, dir: []const u8, family: Family) !u64 {
    var h = std.hash.Wyhash.init(@backingInt(family));
    if (family == .nemo) try nemo.shardStamps(&h, a, t.io, dir) else try disk.checkpoint(&h, a, t.io, dir);
    return h.final();
}
fn nestedMiss(comptime family: Family) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.createDirPath(t.io, "tensors");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"unprobed.weight\":\"tensors/native-QB.safetensors\"}}" });
    const header = "{\"unprobed.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var shard: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tensors/native-QB.safetensors", .data = &shard });
    const before = try key(a, dir, family);
    const same = try key(a, dir, family);
    shard[shard.len - 1] = 1;
    try fixture.rewrite(tmp.dir, "tensors/native-QB.safetensors", &shard);
    try fixture.misses(before, same, &.{try key(a, dir, family)});
    try tmp.dir.deleteFile(t.io, "tensors/native-QB.safetensors");
    try t.expectError(error.IndexedShardMissing, key(a, dir, family));
}

test "persistence identity: Nemotron Metal indexed shard rewrite misses with the same probe" {
    try nestedMiss(.nemo);
}
test "persistence identity: Qwen Metal indexed shard rewrite misses with the same probe" {
    try nestedMiss(.qwen);
}
test "persistence identity: GLM Metal indexed shard rewrite misses with the same probe" {
    try nestedMiss(.glm);
}
test "persistence identity: Flash Next Metal indexed shard rewrite misses with the same probe" {
    try nestedMiss(.flash);
}
test {
    _ = nemo; // its top-level shard rewrite test
}
