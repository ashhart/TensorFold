//! The BF16 dense GEMM tiles against float64, each within its own bound; BF16 out rounded once; rows batch-independent.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const ops = @import("../ops/ops.zig");
const dense = @import("../launches/dense.zig");
const Tensor = ops.Tensor;
const Tile = dense.Tile;

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

/// The tile's bound on |got - exact|: depth u sum|x w|, for WMMA plus k 2^-126 (products below the normal range).
fn bound(tile: Tile, k: usize, mag: f64) f64 {
    const u = std.math.pow(f64, 2, -24);
    return switch (tile) {
        .dot2 => @as(f64, @floatFromInt(dense.Kernels.chain(k))) * u * mag,
        .wmma12 => blk: {
            const ku = @as(f64, @floatFromInt(dense.Kernels.depthWmma12(k))) * u;
            break :blk ku / (1 - ku) * mag + @as(f64, @floatFromInt(k)) * std.math.pow(f64, 2, -126); // gamma(k u)
        },
    };
}

/// x (m, k) times w (n, k) into `y` (fp32 or BF16) on `tile`, on the geometry `on` names or the one m picks.
fn gemm(t: *Rig, tile: Tile, on: ?dense.Shape, x: u64, w: u64, y: Tensor, m: usize, n: usize, k: usize) !void {
    if (on) |shape| {
        const out32: u64 = if (y.kind == .f32) y.ptr else 0;
        const out16: u64 = if (y.kind == .f32) 0 else y.ptr;
        try t.on.dense.gemmOn(t.on.r, tile, shape, x, w, out32, out16, @intCast(m), @intCast(n), @intCast(k), t.stream.handle);
    } else try t.ops(&t.on).denseGemm(tile, .{ .ptr = x, .kind = .bf16 }, w, y, m, n, k);
}

/// x (m, k) times w (n, k) on `tile` (and geometry `on`, if named), fp32 out, synchronized and downloaded.
fn product(t: *Rig, tile: Tile, x: u64, w: u64, m: usize, n: usize, k: usize) ![]f32 {
    return productAt(t, tile, null, x, w, m, n, k);
}

fn productAt(t: *Rig, tile: Tile, on: ?dense.Shape, x: u64, w: u64, m: usize, n: usize, k: usize) ![]f32 {
    var y = try t.alloc(m * n * 4);
    defer y.free();
    try gemm(t, tile, on, x, w, .{ .ptr = at(y), .kind = .f32 }, m, n, k);
    try t.stream.synchronize();
    return rig.download(f32, y);
}

/// The same product on one named WMMA geometry, whatever its row count picks.
fn productOn(t: *Rig, shape: dense.Shape, x: u64, w: u64, m: usize, n: usize, k: usize) ![]f32 {
    var y = try t.alloc(m * n * 4);
    defer y.free();
    try t.on.dense.gemmOn(t.on.r, .wmma12, shape, x, w, at(y), 0, @intCast(m), @intCast(n), @intCast(k), t.stream.handle);
    try t.stream.synchronize();
    return rig.download(f32, y);
}

/// Each output within the bound of the f64 sum, BF16 out rounded once, the last row alone equal; on `on` if named.
fn checkCases(t: *Rig, tile: Tile, on: ?dense.Shape, seed: u64) !void {
    var rng: rig.Rng = .{ .state = seed };
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
        var y16 = try t.alloc(c.m * c.n * 2);
        defer y16.free();
        const g32 = try productAt(t, tile, on, at(x), at(w), c.m, c.n, c.k);
        defer gpa.free(g32);
        try gemm(t, tile, on, at(x), at(w), .{ .ptr = at(y16), .kind = .bf16 }, c.m, c.n, c.k);
        try t.stream.synchronize();
        const g16 = try rig.download(u16, y16);
        defer gpa.free(g16);
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
            try rig.expect(err <= bound(tile, c.k, mag), "dense {t} m {d} n {d} k {d} at ({d}, {d}): {d} vs {d} (mag {d})", .{ tile, c.m, c.n, c.k, r, j, got, want, mag });
            try rig.expect(g16[r * c.n + j] == nearest(got), "dense {t} bf16 out at ({d}, {d}) is {x}, not {x}: the fp32 sum {d} rounded once", .{ tile, r, j, g16[r * c.n + j], nearest(got), got });
        };
        std.log.info("dense {t} m {d} n {d} k {d}: worst error {e:.2} of the magnitude", .{ tile, c.m, c.n, c.k, worst });
        // the last row alone has the bits it has in the batch
        const g1 = try productAt(t, tile, on, at(x) + (c.m - 1) * c.k * 2, at(w), 1, c.n, c.k);
        defer gpa.free(g1);
        try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(g1), std.mem.sliceAsBytes(g32[(c.m - 1) * c.n ..][0..c.n])), "dense {t} m {d}: the last row alone differs from its batch bits", .{ tile, c.m });
    }
}

test "dense GEMM: BF16 weights against float64, fp32 within the dot2 chain's bound, BF16 out rounded once" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    try checkCases(t, .dot2, null, 0xDE45E);
}

test "dense GEMM, gfx12 WMMA tile: within depth k of float64, BF16 out rounded once, a row alone equal to its batch" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.on.dense.wmma12) return error.SkipZigTest;
    try checkCases(t, .wmma12, null, 0xDE45E);
    // each geometry on every case, so the big block meets float64 too (the cases' sizes pick the small one)
    for ([_]dense.Shape{ .w12_small, .w12_big }) |shape| try checkCases(t, .wmma12, shape, 0xDE45E);
}

test "dense GEMM, gfx12 WMMA tile: a row's bytes at every m, on every geometry, and at every place in a block" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.on.dense.wmma12) return error.SkipZigTest;
    const m_all = 300;
    const n = 130;
    const k = 256;
    var rng: rig.Rng = .{ .state = 0x5EED12 };
    const hx = try gpa.alloc(u16, m_all * k);
    defer gpa.free(hx);
    const hw = try gpa.alloc(u16, n * k);
    defer gpa.free(hw);
    for (hx) |*v| v.* = bf16Of(&rng);
    for (hw) |*v| v.* = bf16Of(&rng);
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    const all = try product(t, .wmma12, at(x), at(w), m_all, n, k);
    defer gpa.free(all);
    // m rows from row 0: each row equals its bytes in the 300-row product (every m to 127, then around block edges)
    const edges = [_]usize{ 128, 129, 130, 191, 192, 255, 256, 257, 299, 300 };
    for (0..127 + edges.len) |step| {
        const m = if (step < 127) step + 1 else edges[step - 127];
        const got = try product(t, .wmma12, at(x), at(w), m, n, k);
        defer gpa.free(got);
        try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(all[0 .. m * n])), "dense wmma12: the first {d} rows differ from their bytes in the 300-row product", .{m});
    }
    // every geometry gives the same bytes at any row count, across the launcher's thresholds
    for ([_]usize{ 1, 17, 64, dense.small_rows, dense.small_rows + 1, 200, m_all }) |m| {
        for ([_]dense.Shape{ .w12_small, .w12_big }) |shape| {
            const got = try productOn(t, shape, at(x), at(w), m, n, k);
            defer gpa.free(got);
            try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(all[0 .. m * n])), "dense wmma12: {t} at {d} rows differs from the 300-row product", .{ shape, m });
        }
    }
    // one row placed alone, first, last, and on either side of a block edge: the same bytes everywhere
    const places = [_]usize{ 0, 1, 63, 127, 128, 129, 255, 299 };
    const row = 77;
    const lone = try product(t, .wmma12, at(x) + row * k * 2, at(w), 1, n, k);
    defer gpa.free(lone);
    const moved = try gpa.alloc(u16, m_all * k);
    defer gpa.free(moved);
    for (places) |place| {
        @memcpy(moved, hx);
        @memcpy(moved[place * k ..][0..k], hx[row * k ..][0..k]);
        var mx = try t.upload(moved);
        defer mx.free();
        const got = try product(t, .wmma12, at(mx), at(w), m_all, n, k);
        defer gpa.free(got);
        try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(got[place * n ..][0..n]), std.mem.sliceAsBytes(lone)), "dense wmma12: row {d} placed at {d} differs from the row alone", .{ row, place });
    }
}

test "dense GEMM, gfx12 WMMA tile: exact sums, cancellation and exponent-boundary probes within the bound" {
    const t = try Rig.open(16 << 20);
    defer t.close();
    if (!t.on.dense.wmma12) return error.SkipZigTest;
    const n = 16;
    const k = 256;
    const probes = 3 * 16 + 4;
    const hx = try gpa.alloc(u16, probes * k);
    defer gpa.free(hx);
    const hw = try gpa.alloc(u16, n * k);
    defer gpa.free(hw);
    var rng: rig.Rng = .{ .state = 0xCA2CE1 };
    // column j: small integers for the exact rows, 1 at every k for the probe rows of column 0
    for (0..n) |j| for (0..k) |i| {
        hw[j * k + i] = rig.bf16Bits(if (j == 0) 1 else @floatFromInt(@as(i64, @intCast(rng.next() % 7)) - 3));
    };
    var r: usize = 0;
    // exact rows: integer products whose every partial sum is an integer below 2^24, so fp32 sums them exactly
    while (r < 16) : (r += 1) for (0..k) |i| {
        hx[r * k + i] = rig.bf16Bits(@floatFromInt(@as(i64, @intCast(rng.next() % 61)) - 30));
    };
    // cancellation rows: 2^20 at each of a WMMA step's 16 positions p, -2^20 at p + 16 (1 + r % 3), small terms between
    while (r < 32) : (r += 1) {
        for (0..k) |i| hx[r * k + i] = rig.bf16Bits(@as(f32, @floatCast(rng.normal())) * 0x1p-6);
        const p = r - 16;
        hx[r * k + p] = rig.bf16Bits(0x1p20);
        hx[r * k + p + 16 * (1 + r % 3)] = rig.bf16Bits(-0x1p20);
    }
    // exponent-boundary rows: products of 2^-120 to 2^-133 (normal, subnormal, flushed), then 2^-100 to 2^-130
    while (r < 48) : (r += 1) for (0..k) |i| {
        const e: i32 = -120 - @as(i32, @intCast((r + i) % 14)); // 2^-120 .. 2^-133: normal, subnormal, flushed
        hx[r * k + i] = rig.bf16Bits(std.math.ldexp(@as(f32, if ((r + i) % 2 == 0) 1 else -1.5), e));
    };
    while (r < probes) : (r += 1) for (0..k) |i| {
        hx[r * k + i] = rig.bf16Bits(std.math.ldexp(@as(f32, 1), -100 - @as(i32, @intCast(r - 48)) * 10));
    };
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    const got = try product(t, .wmma12, at(x), at(w), probes, n, k);
    defer gpa.free(got);
    for (0..probes) |row| for (0..n) |j| {
        var want: f64 = 0;
        var mag: f64 = 0;
        for (0..k) |i| {
            const p = rig.bf16Value(hx[row * k + i]) * rig.bf16Value(hw[j * k + i]);
            want += p;
            mag += @abs(p);
        }
        const g: f64 = got[row * n + j];
        if (row < 16) {
            try rig.expect(g == want, "dense wmma12 exact row {d} column {d}: {d}, not the exact {d}: the sums are not fp32", .{ row, j, g, want });
        } else {
            try rig.expect(@abs(g - want) <= bound(.wmma12, k, mag), "dense wmma12 probe row {d} column {d}: {e} vs {e} (mag {e})", .{ row, j, g, want, mag });
        }
    };
}

test "dense GEMM: misaligned x, K off a group, FP16 operands and WMMA off gfx12 are refused before launch" {
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
    for ([_]Tile{ .dot2, .wmma12 }) |tile| {
        try std.testing.expectError(error.BadShape, o.denseGemm(tile, .{ .ptr = at(x) + 2, .kind = .bf16 }, at(w), out, 1, 4, 256));
        try std.testing.expectError(error.BadShape, o.denseGemm(tile, .{ .ptr = at(x), .kind = .bf16 }, at(w), out, 1, 4, 200));
        try std.testing.expectError(error.BadShape, o.denseGemm(tile, .{ .ptr = at(x), .kind = .f16 }, at(w), out, 1, 4, 256));
        try std.testing.expectError(error.BadShape, o.denseGemm(tile, .{ .ptr = at(x), .kind = .bf16 }, at(w), .{ .ptr = at(y), .kind = .f16 }, 1, 4, 256));
    }
    if (!t.on.dense.wmma12) try std.testing.expectError(error.BadShape, o.denseGemm(.wmma12, .{ .ptr = at(x), .kind = .bf16 }, at(w), out, 1, 4, 256));
}
