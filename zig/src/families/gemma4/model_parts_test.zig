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
    const Model = struct { mm: [3]?[1]Handle = @splat(null), inv_sliding: Handle, inv_global: Handle };
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
    pub fn pipelines(m: *Model) !void {
        for (&m.mm) |*set| {
            const handle = try acquire();
            set.* = .{handle};
        }
    }
    pub fn releasePipelines(m: *Model) void {
        for (&m.mm) |*set| if (set.*) |ps| {
            for (ps) |h| h.deinit();
            set.* = null;
        };
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
    var m: Fake.Model = .{ .inv_sliding = undefined, .inv_global = undefined };
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
test "private pipeline failure has no published handle" {
    try boundary(1);
}
test "sliding inverse allocation releases the private pipeline" {
    try boundary(4);
}
test "global inverse allocation releases the first table and private pipeline" {
    try boundary(5);
}
test "inverse operations unwind both tables and private pipeline" {
    try boundary(6);
    try boundary(7);
}
test "successful late load transfers all owners" {
    Fake.step = 0;
    Fake.fail = 0;
    Fake.made = 0;
    Fake.freed = @splat(0);
    var m: Fake.Model = .{ .inv_sliding = undefined, .inv_global = undefined };
    try load(&m, Fake);
    try std.testing.expectEqual(@as(usize, 5), Fake.made);
    for (Fake.freed[0..5]) |n| try std.testing.expectEqual(@as(usize, 0), n);
    Fake.releasePipelines(&m);
    m.inv_sliding.deinit();
    m.inv_global.deinit();
    for (Fake.freed[0..5]) |n| try std.testing.expectEqual(@as(usize, 1), n);
}

test "later width pipeline failure releases all previously completed widths" {
    try boundary(2);
    try boundary(3);
}
