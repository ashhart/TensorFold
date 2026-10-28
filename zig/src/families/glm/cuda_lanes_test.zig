//! The lane backend's slots: chosen before a stream's map entry exists, and none left behind by a failed insertion.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const Lanes = @import("cuda_lanes.zig").Lanes;

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}
