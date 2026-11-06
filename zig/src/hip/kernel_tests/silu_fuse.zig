//! The packed gate/up WMMA product with silu x up in its epilogue: the plain product, a split and tf_silu_mul's bytes.
const std = @import("std");
const rig = @import("rig.zig");
const dense = @import("../launches/dense.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// Bits of a NaN, an infinity and a large value to plant in x and in the weights.
const plants = [_]u16{ 0x7fc1, 0x7f80, 0x7e8e };

/// Bytes of guard on either side of the fused output, and the pattern they hold.
const guard = 512;
const pattern = 0xA5;

/// The fused product on `shape` against the plain one, split, then tf_silu_mul; `planted` adds NaN, inf and 3e38 rows.
fn compare(t: *Rig, rng: *rig.Rng, shape: dense.PairShape, m: usize, n: usize, k: usize, planted: bool) !void {
    const hx = try gpa.alloc(u16, m * k);
    defer gpa.free(hx);
    const hw = try gpa.alloc(u16, n * k);
    defer gpa.free(hw);
    for (hx) |*v| v.* = rig.bf16Bits(@floatCast(rng.normal()));
    for (hw) |*v| v.* = rig.bf16Bits(@floatCast(rng.normal() * 0.03));
    if (planted) for (plants, 0..) |p, i| {
        if (i < m) hx[i * k + 7 * i] = p; // row i of x: one plant a row, so other rows stay finite
        hw[(i % 16) * k + 3] = p; // a gate row
        if (n > 16) hw[(16 + (i + 5) % 16) * k + 5] = p; // an up row of another feature
    };
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    const half = n / 2;
    // the fused output between two guards, its payload poisoned, so a store past it or a missed store shows
    var fused = try t.alloc(m * half * 2 + 2 * guard);
    defer fused.free();
    const fill = try gpa.alloc(u8, m * half * 2 + 2 * guard);
    defer gpa.free(fill);
    @memset(fill, pattern);
    @memset(fill[guard..][0 .. m * half * 2], 0xCD);
    try fused.upload(0, fill);
    var plain = try t.alloc(m * n * 2);
    defer plain.free();
    var gate = try t.alloc(m * half * 2);
    defer gate.free();
    var up = try t.alloc(m * half * 2);
    defer up.free();
    var want = try t.alloc(m * half * 2);
    defer want.free();
    const s = t.stream.handle;
    const mi: c_int = @intCast(m);
    const ni: c_int = @intCast(n);
    const ki: c_int = @intCast(k);
    try t.on.dense.gemmSiluOn(t.on.r, shape, at(x), at(w), at(fused) + guard, mi, ni, ki, s);
    const geo: dense.Shape = if (shape == .pair) .w12_small else .w12_big;
    try t.on.dense.gemmOn(t.on.r, .wmma12, geo, at(x), at(w), 0, at(plain), mi, ni, ki, s);
    // the plain output's 32-column groups: gate the first 16 columns of each, up the last 16
    const o = t.ops(&t.on);
    const src: @import("../ops/ops.zig").Tensor = .{ .ptr = at(plain), .kind = .bf16 };
    try o.copyCols(src, 32, 0, at(gate), m * n / 32, 16);
    try o.copyCols(src, 32, 16, at(up), m * n / 32, 16);
    try o.siluMul(.{ .ptr = at(gate), .kind = .bf16 }, .{ .ptr = at(up), .kind = .bf16 }, .{ .ptr = at(want), .kind = .bf16 }, m * half);
    try t.stream.synchronize();
    const all = try rig.download(u8, fused);
    defer gpa.free(all);
    const got = all[guard..][0 .. m * half * 2];
    const exp = try rig.download(u8, want);
    defer gpa.free(exp);
    for (all[0..guard], all[guard + got.len ..]) |a, b| try rig.expect(a == pattern and b == pattern, "silu fused {t} m {d} n {d}: a guard changed", .{ shape, m, n });
    const diff = std.mem.indexOfDiff(u8, got, exp);
    try rig.expect(diff == null, "silu fused {t} m {d} n {d} planted {}: differs from the plain product + tf_silu_mul from byte {?d}", .{ shape, m, n, planted, diff });
    // the finite fixture's outputs are all finite: a pairing that read a planted or stale value would show as NaN here
    if (!planted) {
        const vals = std.mem.bytesAsSlice(u16, got);
        for (vals) |v| try rig.expect(v & 0x7f80 != 0x7f80, "silu fused {t} m {d} n {d}: a finite fixture gave a non-finite output", .{ shape, m, n });
    }
}

test "the packed gate/up product writes the plain product then tf_silu_mul's bytes, on both blocks" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.on.dense.wmma12) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x51_1F };
    const k = 4096;
    // Qwen3-8B's 2 x 12,288 packed columns, and widths that end mid-block (96, 160) or are one group (32)
    for ([_]dense.PairShape{ .pair, .big }) |shape| for ([_]usize{ 32, 96, 128, 160, 24576 }) |n| {
        for ([_]usize{ 1, 7, 16, 17, 64, 65, 300, 2048 }) |m| for ([_]bool{ false, true }) |planted| try compare(t, &rng, shape, m, n, k, planted);
    };
}
