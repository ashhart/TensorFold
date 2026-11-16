//! The KV pool's display carveout at load: off unless asked for, refused without a pool, refused by name and cause.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const kvs = @import("cuda_kv.zig");
const t = std.testing;

/// An environment holding only `rows`, read the way platform/environment.zig's Process is.
const Rows = struct {
    rows: []const [2][]const u8,

    pub fn get(e: Rows, name: []const u8) ?[]const u8 {
        for (e.rows) |r| if (std.mem.eql(u8, r[0], name)) return r[1];
        return null;
    }
};

const absent = ".zig-cache/tmp/absent-carveout-daemon/d.sock";

test "the KV pool's carveout is off unless asked for, and asked for without a pool or with a bad row it is refused" {
    var d: cuda.Driver = undefined; // no case below reaches the driver
    _ = &d;
    const none: ?*cuda.Carveout = null;
    try t.expectEqual(none, try kvs.openCarveout(t.allocator, &d, true, Rows{ .rows = &.{} }));
    try t.expectEqual(none, try kvs.openCarveout(t.allocator, &d, false, Rows{ .rows = &.{.{ "TF_CUDA_CARVEOUT", "0" }} }));
    const lend: Rows = .{ .rows = &.{ .{ "TF_CUDA_CARVEOUT", "1" }, .{ "TF_CUDA_CARVEOUT_SOURCE", "daemon" }, .{ "TF_CUDA_CARVEOUT_SOCKET", absent } } };
    try t.expectError(error.CarveoutWithoutPool, kvs.openCarveout(t.allocator, &d, false, lend));
    const magic: Rows = .{ .rows = &.{ .{ "TF_CUDA_CARVEOUT", "1" }, .{ "TF_CUDA_CARVEOUT_SOURCE", "magic" } } };
    try t.expectError(error.InvalidEnvironment, kvs.openCarveout(t.allocator, &d, true, magic));
    const tiny: Rows = .{ .rows = &.{ .{ "TF_CUDA_CARVEOUT", "1" }, .{ "TF_CUDA_CARVEOUT_SOURCE", "daemon" }, .{ "TF_CUDA_CARVEOUT_MIB", "1" } } };
    try t.expectError(error.CarveoutTooSmall, kvs.openCarveout(t.allocator, &d, true, tiny));
}

test "a lending daemon that is not there stops the load, with nothing left allocated" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    var d: cuda.Driver = undefined; // the missing socket stops the claim before any driver call
    _ = &d;
    const lend: Rows = .{ .rows = &.{ .{ "TF_CUDA_CARVEOUT", "1" }, .{ "TF_CUDA_CARVEOUT_SOURCE", "daemon" }, .{ "TF_CUDA_CARVEOUT_SOCKET", absent } } };
    try t.expectError(error.NoDaemon, kvs.openCarveout(t.allocator, &d, true, lend));
}
