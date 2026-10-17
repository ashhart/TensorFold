//! Production constructor coordination with independent handle identities and an injected operation failure.
const std = @import("std");
const load = @import("model_parts.zig").load;
const Fake = struct {
    var step: usize = 0;
    var fail: usize = 0;
    var made: usize = 0;
    var freed: [8]usize = @splat(0);
    const Handle = struct {
        id: usize,
        pub fn deinit(h: @This()) void {
            freed[h.id] += 1;
        }
    };
    const Model = struct { mm: [1]Handle, inv_sliding: Handle, inv_global: Handle };
    fn point() !void {
        step += 1;
        if (step == fail) return error.Injected;
    }
    fn acquire() !Handle {
        try point();
        const id = made;
        made += 1;
        return .{ .id = id };
    }
    pub fn pipelines(_: *Model) ![1]Handle {
        return .{try acquire()};
    }
    pub fn buffer(_: *Model, _: bool) !Handle {
        return acquire();
    }
    pub fn inverse(_: *Model, _: bool, _: Handle) !void {
        try point();
    }
};
fn boundary(fail: usize) !void {
    Fake.step = 0;
    Fake.fail = fail;
    Fake.made = 0;
    Fake.freed = @splat(0);
    var m: Fake.Model = undefined;
    const result = load(&m, Fake);
    var leaks: usize = 0;
    for (Fake.freed[0..Fake.made], 0..) |n, i| {
        if (n == 0) {
            leaks += 1;
            Fake.Handle.deinit(.{ .id = i });
        }
    }
    try std.testing.expectError(error.Injected, result);
    try std.testing.expectEqual(@as(usize, 0), leaks);
    for (Fake.freed[0..Fake.made]) |n| try std.testing.expectEqual(@as(usize, 1), n);
}
test "T59 private pipeline failure has no published handle" {
    try boundary(1);
}
test "T59 sliding inverse allocation releases the private pipeline" {
    try boundary(2);
}
test "T59 global inverse allocation releases the first table and private pipeline" {
    try boundary(3);
}
test "T59 inverse operations unwind both tables and private pipeline" {
    try boundary(4);
    try boundary(5);
}
test "T59 successful late load transfers all three owners" {
    Fake.step = 0;
    Fake.fail = 0;
    Fake.made = 0;
    Fake.freed = @splat(0);
    var m: Fake.Model = undefined;
    try load(&m, Fake);
    try std.testing.expectEqual(@as(usize, 3), Fake.made);
    for (Fake.freed[0..3]) |n| try std.testing.expectEqual(@as(usize, 0), n);
    m.mm[0].deinit();
    m.inv_sliding.deinit();
    m.inv_global.deinit();
    for (Fake.freed[0..3]) |n| try std.testing.expectEqual(@as(usize, 1), n);
}
