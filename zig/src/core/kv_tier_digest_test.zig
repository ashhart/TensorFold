//! The lockstep digest: two ranks holding different unflushed pages can serve different rows, so they must differ.
const std = @import("std");
const kt = @import("kv_tier.zig");
const gpa = std.testing.allocator;

test "ranks holding different dirty pages are told apart by the digest" {
    var host_a: [32]u8 = @splat(0);
    var host_b: [32]u8 = @splat(0);
    var a: kt.Tier = undefined;
    try a.init(gpa, &host_a, 4, 8);
    defer a.deinit();
    var b: kt.Tier = undefined;
    try b.init(gpa, &host_b, 4, 8);
    defer b.deinit();
    var la: kt.Lease = .{};
    defer a.finish(&la);
    var lb: kt.Lease = .{};
    defer b.finish(&lb);
    try a.reserve(&la, 4, 2);
    try a.cover(&la, 4, null);
    try b.reserve(&lb, 4, 2);
    try b.cover(&lb, 4, null);
    const ids = la.pages.ids.items;
    try std.testing.expectEqualSlices(u32, ids, lb.pages.ids.items);
    for ([_]u32{ 1, 2 }) |i| {
        _ = try a.pin(&la, ids[i]);
        _ = try b.pin(&lb, ids[i]);
    }
    try std.testing.expectEqual(a.digest(), b.digest());
    a.markDirty(&la, ids[1]) catch unreachable; // a's rows are not in its store; b's still are
    try std.testing.expect(a.digest() != b.digest());
    b.markDirty(&lb, ids[2]) catch unreachable; // both hold one unflushed page, a different one
    try std.testing.expect(a.digest() != b.digest());
    b.sync(ids[2]);
    a.sync(ids[1]);
    try std.testing.expectEqual(a.digest(), b.digest());
}
