//! Row-owned GPU taps copy the state bit for bit into their masked columns; rows without a destination stay untouched.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const base = @import("dspark_test.zig");

/// Rows with no destination: their region keeps its guard values.
fn dark(row: usize) bool {
    return row % 5 == 3;
}

test "lfm2 dspark: the tap copies each row's state into the masked columns of its own destination, and a row without one writes none" {
    var rig = try base.Rig.init();
    defer rig.deinit();
    const cols = 3;
    for ([_]usize{ 1, 16, 32, 2048 }) |rows| for ([_]u32{ 0b010, 0b101, 0b111 }) |mask| for ([_]bool{ false, true }) |nonfinite| {
        const count = rows * c.hidden;
        const h = try rig.buffer(u16, count);
        defer h.deinit();
        // odd rows land in another buffer, as a window's streams land in their own stages
        const dst = [2]mtl.Buffer{ try rig.buffer(u16, count * cols), try rig.buffer(u16, count * cols) };
        defer for (dst) |b| b.deinit();
        const table = try rig.buffer(u64, rows);
        defer table.deinit();
        base.noise(30, h.slice(u16, count), 2);
        if (nonfinite) {
            h.slice(u16, count)[0] = 0x7fc0;
            h.slice(u16, count)[1] = 0x7f80;
            h.slice(u16, count)[2] = 0xff80;
        }
        const state = try std.testing.allocator.dupe(u16, h.slice(u16, count));
        defer std.testing.allocator.free(state);
        for (dst) |b| @memset(b.slice(u16, count * cols), 0x7fc0);
        for (table.slice(u64, rows), 0..) |*at, row| at.* = if (dark(row)) 0 else dst[row % 2].gpuAddress() + row * cols * c.hidden * 2;
        const cb = rig.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(rig.k.get("lfm2_tap_rows"));
        e.setBuffer(h, 0, 0);
        for (dst) |b| e.useResource(b, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.setBuffer(table, 0, 1);
        e.setValue(mask, 2);
        e.dispatchThreads(mtl.Size.of(c.hidden / 8, rows, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        defer cb.release();
        if (cb.failure()) |_| return error.GpuFailed;
        try std.testing.expectEqualSlices(u16, state, h.slice(u16, count));
        for (0..rows) |row| for (0..cols) |col| {
            const out = dst[row % 2].slice(u16, count * cols)[(row * cols + col) * c.hidden ..][0..c.hidden];
            const other = dst[(row + 1) % 2].slice(u16, count * cols)[(row * cols + col) * c.hidden ..][0..c.hidden];
            if (!dark(row) and mask >> @intCast(col) & 1 != 0) {
                try std.testing.expectEqualSlices(u16, state[row * c.hidden ..][0..c.hidden], out);
            } else for (out) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
            for (other) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
        };
    };
}
