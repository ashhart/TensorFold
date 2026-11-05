//! The actual CUDA family facts and shared key reopen real disk states without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const disk = @import("cuda_disk.zig");
const fixture = @import("persistence_fixture.zig");
const t = std.testing;

const Driver = struct {
    fn name(out: [*]u8, _: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        @memcpy(out[0..5], "fake\x00");
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 12;
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        out.* = 13000;
        return 0;
    }
};

fn ropeMiss(comptime F: type) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetName = Driver.name;
    driver.api.cuDeviceGetAttribute = Driver.attribute;
    driver.api.cuDriverGetVersion = Driver.version;
    const gpu: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    var e: qwen.Engine = undefined;
    e.kv8 = false;
    e.n_taps = 0;
    e.int8.on = false;
    e.g.rope = .plain;
    var target: qwen.reuse.Target = .{ .e = &e, .caches = undefined, .seen = undefined };
    const profile: @import("disk_identity.zig").Profile = .{ .family = F.model_type, .backend = "cuda" };
    const before = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    const matching = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    e.g.rope = .{ .yarn = .{ .factor = 4, .trained = 262144, .table = @splat(1) } };
    const scaled = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    e.g.rope.yarn.table[17] = 2;
    const changed_table = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    try fixture.misses(before, matching, &.{ scaled, changed_table });
    try t.expect(scaled != changed_table);
}

test "disk identity: dense Qwen CUDA rotary policy misses with the same probe" {
    try ropeMiss(qwen.native);
}

test "disk identity: MoE Qwen CUDA rotary policy misses with the same probe" {
    try ropeMiss(qwen.native_moe);
}

fn indexedMiss(comptime family: []const u8) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.createDirPath(t.io, "tensors");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"unprobed.weight\":\"tensors/native-QB.bin\"}}" });
    const header = "{\"unprobed.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var shard: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tensors/native-QB.bin", .data = &shard });
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetName = Driver.name;
    driver.api.cuDeviceGetAttribute = Driver.attribute;
    driver.api.cuDriverGetVersion = Driver.version;
    const gpu: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const profile: @import("disk_identity.zig").Profile = .{ .family = family, .backend = "cuda" };
    const before = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    const same = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    try std.Io.sleep(t.io, .fromMilliseconds(5), .awake);
    shard[shard.len - 1] = 1;
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tensors/native-QB.bin", .data = &shard });
    try fixture.misses(before, same, &.{try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile)});
}

test "disk identity: Nemotron CUDA indexed shard misses with the same probe" {
    try indexedMiss("nemotron_h");
}
test "disk identity: dense Qwen CUDA indexed shard misses with the same probe" {
    try indexedMiss("qwen3_5");
}
test "disk identity: MoE Qwen CUDA indexed shard misses with the same probe" {
    try indexedMiss("qwen3_5_moe");
}
test "disk identity: Flash Next CUDA indexed shard misses with the same probe" {
    try indexedMiss("qwen4_exp");
}
