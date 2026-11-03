//! hip-dense-tune: both BF16 dense tiles timed at Qwen3-8B's projection shapes, weights cold (rotated past the caches).
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const dense = @import("../launches/dense.zig");
const Event = @import("../stream.zig").Event;
const Shape = dense.Shape;

/// Weight bytes a rotation reads, so launches miss the 64 MiB Infinity Cache and the L2.
const read_target = 512 << 20;
const batches = 7;

const Case = struct { n: usize, k: usize, name: []const u8 };
const shapes = [_]Case{
    .{ .n = 4096, .k = 4096, .name = "q, o" },
    .{ .n = 1024, .k = 4096, .name = "k, v" },
    .{ .n = 12288, .k = 4096, .name = "gate, up" },
    .{ .n = 4096, .k = 12288, .name = "down" },
};
const rows = [_]usize{ 1, 16, 64, 96, 128, 192, 276, 384, 512, 2048 };

test "hip-dense-tune: both tiles at the served shapes, TFLOPS with cold weights" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    const kernels: []const Shape = if (t.on.dense.wmma12) &.{ .dot2, .w12_small, .w12_big } else &.{.dot2};
    var rng: rig.Rng = .{ .state = 0x7E57 };
    var start = try Event.init(&t.r, true);
    defer start.deinit();
    var end = try Event.init(&t.r, true);
    defer end.deinit();
    for (shapes) |sh| {
        const one = sh.n * sh.k * 2;
        const copies = (read_target + one - 1) / one;
        const hw = try gpa.alloc(u16, sh.n * sh.k);
        defer gpa.free(hw);
        for (hw) |*v| v.* = rig.bf16Bits(@floatCast(rng.normal() * 0.03));
        var w = try t.alloc(one * copies);
        defer w.free();
        for (0..copies) |c| try w.upload(c * one, std.mem.sliceAsBytes(hw));
        const most = rows[rows.len - 1];
        const hx = try gpa.alloc(u16, most * sh.k);
        defer gpa.free(hx);
        for (hx) |*v| v.* = rig.bf16Bits(@floatCast(rng.normal()));
        var x = try t.upload(hx);
        defer x.free();
        var y = try t.alloc(most * sh.n * 4);
        defer y.free();
        for (rows) |m| {
            const launch = struct {
                fn go(r: *Rig, kn: Shape, xp: u64, wp: u64, yp: u64, mm: usize, s: Case) !void {
                    const tile: dense.Tile = if (kn == .dot2) .dot2 else .wmma12;
                    try r.on.dense.gemmOn(r.on.r, tile, kn, xp, wp, yp, 0, @intCast(mm), @intCast(s.n), @intCast(s.k), r.stream.handle);
                }
            }.go;
            for (kernels) |kernel| for (0..copies) |c| try launch(t, kernel, at(x), at(w) + c * one, at(y), m, sh); // warm-up
            // batches alternate the kernels' order, so a clock drift between kernels cannot favour one
            var us: [3][batches]f64 = undefined;
            for (0..batches) |b| for (0..kernels.len) |j| {
                const ki = if (b % 2 == 0) j else kernels.len - 1 - j;
                try start.record(t.stream);
                for (0..copies) |c| try launch(t, kernels[ki], at(x), at(w) + c * one, at(y), m, sh);
                try end.record(t.stream);
                try end.synchronize();
                us[ki][b] = @as(f64, try Event.elapsedMs(start, end)) * 1000 / @as(f64, @floatFromInt(copies));
            };
            for (kernels, 0..) |kernel, ki| {
                std.mem.sort(f64, &us[ki], {}, std.sort.asc(f64));
                const mid = us[ki][batches / 2];
                const flops = 2.0 * @as(f64, @floatFromInt(m * sh.n * sh.k));
                std.debug.print("DENSE {s:>8} n{d:>5} k{d:>5} m{d:>5} {t:>9}: {d:>9.2} us {d:>6.1} TFLOPS (noise {d:.1}%)\n", .{
                    sh.name, sh.n, sh.k, m, kernel, mid, flops / mid / 1e6, (us[ki][batches - 1] / us[ki][0] - 1) * 100,
                });
            }
        }
    }
}
