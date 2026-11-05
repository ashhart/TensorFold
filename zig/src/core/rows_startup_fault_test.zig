//! A transient allocation failure at startup must leave every valid stored state available to the next open.
const std = @import("std");
const rows = @import("rows_store.zig");
const fixture = @import("rows_scale_fixture.zig");
const t = std.testing;

test "restore scale: startup allocation failures preserve valid state files" {
    var lost: usize = 0;
    var induced: usize = 0;
    for (0..32) |index| {
        var tag: [40]u8 = undefined;
        var x = try fixture.Fixture.init(try std.fmt.bufPrint(&tag, "fault-{d}", .{index}));
        defer x.close();
        try x.seed(1);
        var fa = t.FailingAllocator.init(t.allocator, .{ .fail_index = index });
        var failed = false;
        if (rows.Rows.open(fa.allocator(), x.f.snapshots(), x.options())) |r| {
            r.deinit();
        } else |err| {
            try t.expectEqual(error.OutOfMemory, err);
            failed = true;
        }
        try t.expectEqual(fa.allocated_bytes, fa.freed_bytes);
        const retained = try x.files();
        if (fa.has_induced_failure) {
            induced += 1;
            try t.expect(failed);
        }
        if (retained != 1) lost += 1;
    }
    try t.expect(induced > 0);
    try t.expectEqual(@as(usize, 0), lost);
}
