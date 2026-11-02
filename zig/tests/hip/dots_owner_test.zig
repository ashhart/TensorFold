//! Real owner close and late host-allocation failures return all tracked HIP and host bytes.
const std = @import("std");
const hip = @import("hip");
const family = @import("dots_mocr");
pub const std_options: std.Options = .{ .signal_stack_size = null };

test "dots HIP device owner: close and late allocation failures free every owned byte" {
    const path = std.c.getenv("TF_DOTS_DIR") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const dev: hip.Device = .{ .index = 0, .caps = hip.Device.capsOf(0) orelse return error.NoDevice, .policy = .{} };
    var probe = try hip.Runtime.open();
    defer probe.close();
    var ctx = try hip.Context.init(&probe, 0);
    defer ctx.deinit();
    if ((try ctx.memInfo()).free < 10 << 30) return error.DotsMemoryBudget;
    const before = hip.usage(false);
    const options = .{ .window = @as(usize, 512), .streams = @as(usize, 2), .fixed = true, .cache_gib = @as(?f64, 0) };
    var problem: []const u8 = "";
    var counted = std.testing.FailingAllocator.init(a, .{});
    const loaded = try family.native.openVision(arena.allocator(), counted.allocator(), std.testing.io, dev, std.mem.span(path), options, .{ .image_tokens = 64, .image_tokens_set = true }, &problem);
    const allocations = counted.alloc_index;
    try std.testing.expect(hip.usage(false).device > before.device + 6078358528);
    loaded.deinit(loaded.ctx);
    try std.testing.expectEqual(before.device, hip.usage(false).device);
    try std.testing.expectEqual(before.host, hip.usage(false).host);
    try std.testing.expectEqual(counted.allocated_bytes, counted.freed_bytes);
    // The final two host allocations own the adapter and its startup text. Both follow the device bank and drawer.
    for ([_]usize{ allocations - 2, allocations - 1 }) |fail_at| {
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_at });
        try std.testing.expectError(error.OutOfMemory, family.native.openVision(arena.allocator(), failing.allocator(), std.testing.io, dev, std.mem.span(path), options, .{ .image_tokens = 64, .image_tokens_set = true }, &problem));
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        try std.testing.expectEqual(before.device, hip.usage(false).device);
        try std.testing.expectEqual(before.host, hip.usage(false).host);
    }
}
