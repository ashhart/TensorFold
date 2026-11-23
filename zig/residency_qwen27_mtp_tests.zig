//! The lane drafter's direct buffers and its nested scratch share the normal residency inventory.
const std = @import("std");
const mtl = @import("metal");
const residency = @import("src/families/qwen27/residency.zig");
const a = std.testing.allocator;

test "residency owner Qwen27 MTP: staging and every scratch plane join and leave the same set" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var buffers: [5]mtl.Buffer = undefined;
        var made: usize = 0;
        defer for (buffers[0..made]) |b| b.deinit();
        for (&buffers, 0..) |*b, i| {
            b.* = try record.device().buffer(64 * (i + 1), mtl.ResourceOptions.shared);
            made += 1;
        }
        var head = .{ .stage = buffers[0], .index = buffers[1], .scratch = .{
            .planes = buffers[2],
            .rows = buffers[3],
            .logits = buffers[4],
        } };
        var list: std.ArrayList(mtl.Buffer) = .empty;
        defer list.deinit(a);
        try residency.appendMtp(&head, a, &list);
        try std.testing.expectEqualSlices(mtl.Buffer, &buffers, list.items);
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), list.items, "qwen27")).?;
        defer owner.deinit();
        residency.mtpBuffers(&head, owner.follow(), true);
        defer residency.mtpBuffers(&head, owner.follow(), false);
        try std.testing.expectEqual(@as(u32, 6), owner.held.members.count());
        try std.testing.expectEqual(@as(u64, 1024), owner.held.bytes);
        try record.check();
    }
    try record.allFreed();
}
