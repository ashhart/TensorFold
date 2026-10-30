//! Replay the original slot seeds through the actual group stress fixture and accumulator simulators.
const std = @import("std");
const core = @import("core");
const act = @import("contract_act.zig");
const c = core.contract;

test "group sums stress: all original 600 slots distinguish both reduced accumulators in every order" {
    const width = 17408;
    const stride = width + 64;
    const x = try std.testing.allocator.alloc(u16, 4 * stride);
    defer std.testing.allocator.free(x);
    var terms: [64]f64 = undefined;
    for (0..600) |slot| {
        var random = std.Random.DefaultPrng.init(0x6b63 +% 0x9e37 *% slot);
        const rng = random.random();
        for (x) |*v| v.* = c.spec.toBf16(rng.float(f32));
        act.stressGroupRows(rng, x, 4, width, stride);
        for ([_]c.gates.Order{ .sequential, .pairwise, .strided }) |order| {
            var worst = [2]f64{ 0, 0 };
            for (0..4) |r| for (0..width / 64) |j| {
                var want: f64 = 0;
                for (&terms, 0..) |*v, i| {
                    v.* = c.spec.fromBf16(x[r * stride + j * 64 + i]);
                    want += v.*;
                }
                try std.testing.expectEqual(want, c.rows.sumIn(.f32, order, &terms));
                for ([_]c.spec.Accum{ .f16, .bf16 }, &worst) |a, *v| v.* = @max(v.*, @abs(c.rows.sumIn(a, order, &terms) - want));
            };
            for (worst) |v| if (!(v >= 1)) {
                std.debug.print("stress failed at slot {d}, order {t}: {any}\n", .{ slot, order, worst });
                return error.StressDidNotSeparateAccumulator;
            };
        }
    }
}
