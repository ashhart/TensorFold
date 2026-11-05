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
