//! Row-owned GPU taps preserve plain residual and normalized bytes; rows without a destination stay untouched.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const base = @import("dspark_test.zig");

/// Rows with no destination: their region keeps its guard values.
fn dark(row: usize) bool {
    return row % 5 == 3;
}

test "lfm2 dspark: fused normalization taps match the separate copy bit for bit, and a row without a destination writes none" {
    var rig = try base.Rig.init();
    defer rig.deinit();
    for ([_]usize{ 1, 16, 32, 2048 }) |rows| for ([_]bool{ false, true }) |combine| for ([_]bool{ false, true }) |nonfinite| {
        const count = rows * c.hidden;
        const h0 = try rig.buffer(u16, count);
        defer h0.deinit();
        const h1 = try rig.buffer(u16, count);
        defer h1.deinit();
        const residual = try rig.buffer(u16, if (combine) count * c.topk else count);
        defer residual.deinit();
        const router = try rig.buffer(u16, rows * c.topk);
        defer router.deinit();
        const weight = try rig.buffer(u16, c.hidden);
        defer weight.deinit();
        const x0 = try rig.buffer(u16, count);
        defer x0.deinit();
        const x1 = try rig.buffer(u16, count);
        defer x1.deinit();
        const taps = try rig.buffer(u16, count * 3);
        defer taps.deinit();
        const reference_taps = try rig.buffer(u16, count * 3);
        defer reference_taps.deinit();
        const table = try rig.buffer(u64, rows);
        defer table.deinit();
        base.noise(30, h0.slice(u16, count), 2);
        const initial = h0.slice(u16, count);
        if (nonfinite) {
            initial[0] = 0x7fc0;
            initial[1] = 0x7f80;
            initial[2] = 0xff80;
        }
        @memcpy(h1.slice(u16, count), initial);
        base.noise(31, residual.slice(u16, if (combine) count * c.topk else count), 0.2);
        base.noise(32, router.slice(u16, rows * c.topk), 0.2);
        base.noise(33, weight.slice(u16, c.hidden), 1);
        @memset(taps.slice(u16, count * 3), 0x7fc0);
        @memset(reference_taps.slice(u16, count * 3), 0x7fc0);
        for (table.slice(u64, rows), 0..) |*at, row| at.* = if (dark(row)) 0 else taps.gpuAddress() + row * 3 * c.hidden * 2;
        for ([_]bool{ false, true }) |fused| {
            const cb = rig.queue.commandBuffer();
            const e = cb.compute(.serial);
            if (combine) {
                if (fused) e.setPipeline(rig.k.get("lfm2_combine_norm_tap")) else e.setPipeline(rig.k.get("lfm2_combine_norm"));
                e.setBuffer(residual, 0, 0);
                e.setBuffer(router, 0, 1);
                e.setBuffer(if (fused) h1 else h0, 0, 2);
                e.setBuffer(weight, 0, 3);
                e.setValue(c.eps, 4);
            } else {
                if (fused) e.setPipeline(rig.k.get("lfm2_norm_tap")) else e.setPipeline(rig.k.get("lfm2_norm"));
                e.setBuffer(if (fused) h1 else h0, 0, 0);
                e.setBuffer(residual, 0, 1);
                e.setBuffer(weight, 0, 2);
                e.setValue(c.eps, 3);
                e.setValue(@as(u32, 1), 4);
            }
            e.setBuffer(if (fused) x1 else x0, 0, 5);
            if (fused) {
                e.useResource(taps, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
                e.setBuffer(table, 0, 6);
                e.setValue(@as(u32, 0b010), 7);
            }
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(c.hidden / 4, 1, 1));
            if (!fused) {
                e.setPipeline(rig.k.get("lfm2_tap"));
                e.setBuffer(h0, 0, 0);
                e.setBuffer(reference_taps, 0, 1);
                e.setValue([2]u32{ 3 * c.hidden, c.hidden }, 2);
                e.dispatchThreads(mtl.Size.of(c.hidden / 8, rows, 1), mtl.Size.of(256, 1, 1));
            }
            e.end();
            cb.commit();
            cb.wait();
            defer cb.release();
            if (cb.failure()) |_| return error.GpuFailed;
        }
        try std.testing.expectEqualSlices(u16, h0.slice(u16, count), h1.slice(u16, count));
        try std.testing.expectEqualSlices(u16, x0.slice(u16, count), x1.slice(u16, count));
        const out = taps.slice(u16, count * 3);
        const want = reference_taps.slice(u16, count * 3);
        for (0..rows) |row| {
            const at = row * 3 * c.hidden;
            if (dark(row)) {
                for (out[at..][0 .. 3 * c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
                continue;
            }
            try std.testing.expectEqualSlices(u16, want[at + c.hidden ..][0..c.hidden], out[at + c.hidden ..][0..c.hidden]);
            try std.testing.expectEqualSlices(u16, h0.slice(u16, count)[row * c.hidden ..][0..c.hidden], out[at + c.hidden ..][0..c.hidden]);
            for (out[at..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
            for (out[at + 2 * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
        }
    };
}
