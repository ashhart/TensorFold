//! The full review's unloading failure reaches the route's Retry-After path.
const std = @import("std");
const idle = @import("idle_unload.zig");
const Harness = @import("idle_unload_test.zig").Harness;
const t = std.testing;

test "idle review: an unloading wake keeps its reason and never reaches the stub submit" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.gate_close.store(true, .release);
    h.expire();
    try h.waitEvent(&h.close_entered);
    try t.expectEqual(idle.State.unloading, h.state());
    var reply: std.heap.ArenaAllocator = .init(t.allocator);
    defer reply.deinit();
    const why = try idle.wakeReload(h.man.?, h.srv.?, reply.allocator());
    try t.expectEqualStrings("the engine is still unloading; send the request again", why);
    try t.expectEqual(@as(u32, 0), h.opens.load(.acquire));
    try t.expectEqual(idle.State.unloading, h.state());
    h.close_release.set(h.io);
    try h.waitUnloaded(2000);
    var problem: []const u8 = "";
    try h.reload(&problem);
    try t.expectEqual(idle.State.resident, h.state());
}
