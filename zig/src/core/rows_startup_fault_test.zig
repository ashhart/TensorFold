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

/// A state file whose trailer puts it at 2^32 - 1 (one token, lookahead 1), beside one valid state; open keeps one.
fn wrappedPosition() !void {
    var x = try fixture.Fixture.init("wrap");
    defer x.close();
    try x.seed(1);
    {
        const r = try rows.Rows.open(t.allocator, x.f.snapshots(), x.options());
        defer r.deinit();
        const tokens = [_]u32{17};
        const counts: [3]u32 = .{ 1, 0, 0 };
        const name: rows.chain.Name = @splat(0xab);
        const stem = std.fmt.bytesToHex(rows.chain.stem(name), .lower);
        var w = try rows.file.Writer.open(r.dir, &stem, ".st");
        errdefer w.abort();
        try w.append("own");
        const at = std.math.maxInt(u32);
        _ = try w.finish(.{ .kind = @backingInt(rows.file.Kind.state), .payload = 0, .payload_hash = 0, .name = name, .at = at, .block = rows.block_tokens }, &.{ std.mem.asBytes(&counts), std.mem.sliceAsBytes(&tokens) }, false);
    }
    try t.expectEqual(@as(usize, 1), try x.files());
}

// In a child: if readState's u32 `s.at + r.keying.lookahead` is unchecked, a safe build panics and only this test fails
test "restore scale: a state whose position plus its lookahead wraps past 2^32 is dropped at open" {
    try t.expectEqual(@import("sys").isolate.Outcome.passed, try @import("sys").isolate.run(wrappedPosition, .{}));
}
