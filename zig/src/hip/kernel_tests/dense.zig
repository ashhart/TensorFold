//! The BF16 dense GEMM against float64: within the dot2 chain's bound, BF16 out rounded once, rows batch-independent.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const ops = @import("../ops/ops.zig");
const Tensor = ops.Tensor;

const Case = struct { m: usize, n: usize, k: usize };

/// Edges of both tile dimensions, one and several groups of K, and Qwen3's widest K.
const cases = [_]Case{
    .{ .m = 1, .n = 2, .k = 128 },
    .{ .m = 7, .n = 130, .k = 4096 },
    .{ .m = 128, .n = 256, .k = 128 },
    .{ .m = 129, .n = 300, .k = 4096 },
    .{ .m = 300, .n = 130, .k = 12288 },
};

/// The fp32 value as BF16, rounded to nearest even (the tile's epilogue; the rig's bf16Bits truncates).
fn nearest(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf16Of(rng: *rig.Rng) u16 {
    return rig.bf16Bits(@floatCast(rng.normal()));
}

test "dense GEMM: BF16 weights against float64, fp32 within the dot2 chain's bound, BF16 out rounded once" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xDE45E };
    for (cases) |c| {
        const hx = try gpa.alloc(u16, c.m * c.k);
        defer gpa.free(hx);
        const hw = try gpa.alloc(u16, c.n * c.k);
        defer gpa.free(hw);
        for (hx) |*v| v.* = bf16Of(&rng);
        for (hw) |*v| v.* = bf16Of(&rng);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var y32 = try t.alloc(c.m * c.n * 4);
        defer y32.free();
        var y16 = try t.alloc(c.m * c.n * 2);
        defer y16.free();
        const xt: Tensor = .{ .ptr = at(x), .kind = .bf16 };
        try o.denseGemm(xt, at(w), .{ .ptr = at(y32), .kind = .f32 }, c.m, c.n, c.k);
        try o.denseGemm(xt, at(w), .{ .ptr = at(y16), .kind = .bf16 }, c.m, c.n, c.k);
        try t.stream.synchronize();
        const g32 = try rig.download(f32, y32);
        defer gpa.free(g32);
        const g16 = try rig.download(u16, y16);
        defer gpa.free(g16);
        // a group's 64 dot2s (two roundings each at most), then one add a group: the longest rounding chain of a sum
        const depth: f64 = 128 + @as(f64, @floatFromInt(c.k / 128)) + 2;
        var worst: f64 = 0;
        for (0..c.m) |r| for (0..c.n) |j| {
            var want: f64 = 0;
            var mag: f64 = 0;
            for (0..c.k) |i| {
                const p = rig.bf16Value(hx[r * c.k + i]) * rig.bf16Value(hw[j * c.k + i]);
                want += p;
                mag += @abs(p);
            }
            const got = g32[r * c.n + j];
            const err = @abs(@as(f64, got) - want);
            worst = @max(worst, err / @max(mag, 1e-30));
            try rig.expect(err <= depth * std.math.pow(f64, 2, -24) * mag, "dense m {d} n {d} k {d} at ({d}, {d}): {d} vs {d} (mag {d})", .{ c.m, c.n, c.k, r, j, got, want, mag });
            try rig.expect(g16[r * c.n + j] == nearest(got), "dense bf16 out at ({d}, {d}) is {x}, not {x}: the fp32 sum {d} rounded once", .{ r, j, g16[r * c.n + j], nearest(got), got });
        };
        std.log.info("dense m {d} n {d} k {d}: worst error {e:.2} of the magnitude", .{ c.m, c.n, c.k, worst });
        // the last row alone has the bits it has in the batch
        var lone = try t.alloc(c.n * 4);
        defer lone.free();
        const last: Tensor = .{ .ptr = at(x) + (c.m - 1) * c.k * 2, .kind = .bf16 };
        try o.denseGemm(last, at(w), .{ .ptr = at(lone), .kind = .f32 }, 1, c.n, c.k);
        try t.stream.synchronize();
        const g1 = try rig.download(f32, lone);
        defer gpa.free(g1);
        try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(g1), std.mem.sliceAsBytes(g32[(c.m - 1) * c.n ..][0..c.n])), "dense m {d}: the last row alone differs from its batch bits", .{c.m});
    }
}

test "dense GEMM: misaligned x, K off a group and FP16 operands are refused before launch" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    const o = t.ops(&t.on);
    var x = try t.alloc(4 * 256 * 2 + 16);
    defer x.free();
    var w = try t.alloc(4 * 256 * 2);
    defer w.free();
    var y = try t.alloc(4 * 4 * 4);
    defer y.free();
    const out: Tensor = .{ .ptr = at(y), .kind = .f32 };
    try std.testing.expectError(error.BadShape, o.denseGemm(.{ .ptr = at(x) + 2, .kind = .bf16 }, at(w), out, 1, 4, 256));
    try std.testing.expectError(error.BadShape, o.denseGemm(.{ .ptr = at(x), .kind = .bf16 }, at(w), out, 1, 4, 200));
    try std.testing.expectError(error.BadShape, o.denseGemm(.{ .ptr = at(x), .kind = .f16 }, at(w), out, 1, 4, 256));
    try std.testing.expectError(error.BadShape, o.denseGemm(.{ .ptr = at(x), .kind = .bf16 }, at(w), .{ .ptr = at(y), .kind = .f16 }, 1, 4, 256));
}
