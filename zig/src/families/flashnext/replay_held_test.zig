//! Run.release: a buffer the load frees leaves the held list, and a following residency set, before it is freed (#732).
const std = @import("std");
const mtl = @import("metal");
const Run = @import("replay.zig").Run;

test "a buffer freed during load leaves held, so the residency set never takes it released" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r = Run{ .arena = arena.allocator(), .device = device, .queue = try device.queue() };
    const kept = try r.buffer(64);
    defer kept.deinit();
    const scratch = try r.buffer(64);
    r.release(scratch);
    try std.testing.expectEqual(@as(usize, 1), r.held.items.len);
    try std.testing.expectEqual(kept.id, r.held.items[0].id);
}

test "a buffer made after the set is built joins it, and leaves it when released" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const Seen = struct {
        joined: usize = 0,
        left: usize = 0,
        fn join(ctx: *anyopaque, _: mtl.Buffer) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            s.joined += 1;
        }
        fn leave(ctx: *anyopaque, _: mtl.Buffer) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            s.left += 1;
        }
    };
    var seen: Seen = .{};
    var r = Run{ .arena = arena.allocator(), .device = device, .queue = try device.queue() };
    r.follow = .{ .ctx = &seen, .join = Seen.join, .leave = Seen.leave };
    const scratch = try r.buffer(64);
    r.release(scratch);
    try std.testing.expectEqual(@as(usize, 1), seen.joined);
    try std.testing.expectEqual(@as(usize, 1), seen.left);
    try std.testing.expectEqual(@as(usize, 0), r.held.items.len);
}
