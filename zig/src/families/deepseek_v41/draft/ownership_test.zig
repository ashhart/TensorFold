const std = @import("std");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
fn initCase(a: std.mem.Allocator, t: *twin.Twin, d: *twin.Drafter) !void {
    var verify: [64]f64 = @splat(20);
    const x = try ln.Lanes.init(a, t.target(), d.pass(), .{ .verify = &verify, .draft = 1 }, .{ .shape = d.shape, .slots = 4, .spec = .{ .on = true } });
    defer x.deinit();
}
test "Draft state: lane adapter initialization unwinds every allocation failure" {
    const a = std.testing.allocator;
    const t = try twin.Twin.init(a, .{}, 4);
    defer t.deinit();
    const d = try twin.Drafter.init(a, t, 4);
    defer d.deinit();
    try std.testing.checkAllAllocationFailures(a, initCase, .{ t, d });
}
