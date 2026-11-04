//! Startup measurements use real hashed files and checked metadata; payload copies are checked separately afterward.
const std = @import("std");
const rows = @import("rows_store.zig");
const fixture = @import("rows_scale_fixture.zig");
const Peak = @import("rows_peak_allocator.zig").Peak;
const t = std.testing;

test "restore scale: startup 1000 10000 100000 states has measured metadata memory" {
    for ([_]usize{ 1000, 10000, 100000 }) |count| {
        var tag: [40]u8 = undefined;
        var x = try fixture.Fixture.init(try std.fmt.bufPrint(&tag, "startup-{d}", .{count}));
        defer x.close();
        const preparing = rows.write.nowNs();
        try x.seed(count);
        const seeded = rows.write.nowNs();
        var peak: Peak = .{ .child = t.allocator };
        const r = try rows.Rows.open(peak.allocator(), x.f.snapshots(), x.options());
        var closed = false;
        defer if (!closed) r.deinit();
        const elapsed = rows.write.nowNs() - seeded;
        try t.expectEqual(count, r.states.items.len);
        std.debug.print("RESTORE_STARTUP states={d} ns={d} live_bytes={d} peak_bytes={d} fixture_ns={d} tokens_per_state=65 position=64\n", .{ count, elapsed, peak.live, peak.peak, seeded - preparing });
        for ([_]usize{ 0, count / 2, count - 1 }) |i| {
            x.f.prompt = r.states.items[i].tokens;
            const saved = try rows.read.read(r, i);
            defer x.f.snapshots().vtable.drop(&x.f, saved);
            try x.f.snapshots().vtable.restore(&x.f, null, saved);
            try t.expect(x.f.restored_ok);
        }
        r.deinit();
        closed = true;
        try t.expectEqual(@as(usize, 0), peak.live);
        try t.expectEqual(@as(usize, 0), x.f.live);
    }
}
