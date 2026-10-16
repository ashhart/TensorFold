//! weights.Store on the fake driver: each host or device allocation failing in turn leaves nothing lost or freed twice.
const std = @import("std");
const cuda = @import("cuda");
const proj = @import("weights_proj.zig");
const fake = @import("fake_driver.zig");

const t = std.testing;

/// Twelve buffers (past the lists' first growth), one upload, a lookup; the store freed on every path.
fn fill(gpa: std.mem.Allocator, d: *const cuda.Driver) !void {
    var s: proj.Store = .{ .gpa = gpa, .d = d };
    defer s.deinit();
    for (0..12) |i| {
        var name: [32]u8 = undefined;
        _ = try s.alloc(64 + i, try std.fmt.bufPrint(&name, "buffer.{d}", .{i}));
    }
    _ = try s.keep("bytes", "kept");
    try t.expect(s.find("buffer.3") != null and s.find("kept").?.len == 5);
}

test "Store.alloc: each host allocation failing in turn leaks nothing and frees no buffer twice" {
    const d = fake.driver();
    fake.reset(null, cuda.abi.success);
    try t.checkAllAllocationFailures(t.allocator, fill, .{&d});
    try t.expectEqual(@as(usize, 0), fake.live());
    try t.expectEqual(@as(usize, 0), fake.badReleases());
}

test "Store.alloc: each device call failing in turn (allocations, the upload) leaves nothing held or freed twice" {
    const d = fake.driver();
    fake.reset(null, cuda.abi.success);
    try fill(t.allocator, &d);
    const n = fake.count();
    try t.expect(n >= 13);
    for (1..n + 1) |k| {
        fake.reset(k, cuda.abi.error_not_ready);
        try t.expectError(error.NotReady, fill(t.allocator, &d));
        try t.expect(fake.fired());
        try t.expectEqual(@as(usize, 0), fake.live());
        try t.expectEqual(@as(usize, 0), fake.badReleases());
    }
    fake.reset(null, cuda.abi.success);
}
