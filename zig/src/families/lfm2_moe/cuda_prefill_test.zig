const std = @import("std");
const lanes = @import("lanes");
const backend = @import("cuda_backend.zig");
const Stream = lanes.Stream;

test "lfm2 moe prefill: cancellation keeps a completed prefix for exact resume" {
    const Kept = struct {
        at: u32 = 0,
        fn interrupted(ctx: *anyopaque, _: *Stream, at: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ctx));
            k.at = at;
        }
        fn cancelled(_: *anyopaque) bool {
            return true;
        }
    };
    var stored: Kept = .{};
    var s = try Stream.init(std.testing.allocator, .{ .id = "stopped", .prompt = &.{ 3, 1, 4, 1, 5, 9 }, .max_new = 1, .reuse = .{ .hook = .{ .ptr = &stored, .at = Kept.interrupted } }, .cancel_check = .{ .ptr = &stored, .check = Kept.cancelled } });
    defer s.deinit(std.testing.allocator);
    try std.testing.expectError(error.Cancelled, backend.Lfm.cancelledPrefix(&s, 2));
    try std.testing.expectEqual(@as(u32, 2), stored.at);
}
