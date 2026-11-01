//! Untaught Omni registration must leave the serving engine and every borrowed pointer untouched.
const std = @import("std");
const F = @import("nemotron").omni;
const Adapter = @import("slide_adapter").Adapter(F.slide);

test "Omni learner: constructing and freeing the untaught adapter changes no engine byte" {
    const Model = F.slide.Model;
    var e: @typeInfo(@FieldType(Model, "e")).pointer.child = undefined;
    @memset(std.mem.asBytes(&e), 0xa5);
    const before = std.mem.asBytes(&e).*;
    var saved: @typeInfo(@FieldType(Model, "living")).pointer.child = null;
    {
        var adapter = Adapter.init(std.testing.allocator, std.testing.io, @as(Model, .{ .e = &e, .dir = "model", .living = &saved }));
        defer adapter.deinit();
        try std.testing.expect(adapter.learner.trainer == null);
        try std.testing.expect(adapter.learner.b.e == &e);
        _ = adapter.learner.readsConversations();
        try std.testing.expect(adapter.hook().ctx == @as(*anyopaque, &adapter));
    }
    try std.testing.expect(saved == null);
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&e));
}
