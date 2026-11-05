//! A Qwen-VL tower's host tables for one image: each patch's place, its learned-position taps and its rotary angles.
const std = @import("std");

/// Patch `i` of a grid `gw` patches wide, in merge order: its (row, column).
pub fn place(i: usize, gw: u32, merge: u32) [2]u32 {
    const blocks_w = gw / merge;
    const in_col: u32 = @intCast(i % merge);
    const in_row: u32 = @intCast((i / merge) % merge);
    const block_col: u32 = @intCast((i / (merge * merge)) % blocks_w);
    const block_row: u32 = @intCast(i / (merge * merge * blocks_w));
    return .{ block_row * merge + in_row, block_col * merge + in_col };
}

const AxisTaps = struct { i: [2]u32, w: [2]f32 };

/// The four learned positions a patch reads, and their fp32 weights.
pub const Taps = struct { index: [4]u32, weight: [4]f32 };

/// Bilinear align_corners taps on one axis, fp32; `torch` takes Torch's CPU linspace and lerp, else the shared towers'.
fn axisTaps(comptime torch: bool, index: u32, size: u32, side: u32) AxisTaps {
    const src = if (torch) coordinate(index, size, side) else @as(f32, @floatFromInt(index)) * @as(f32, @floatFromInt(side - 1)) / @as(f32, @floatFromInt(@max(size, 2) - 1));
    const fl = @floor(src);
    var out: AxisTaps = undefined;
    for (0..2) |o| {
        const raw = @as(i64, @intFromFloat(fl)) + @as(i64, @intCast(o));
        out.i[o] = @intCast(std.math.clamp(raw, 0, side - 1));
        out.w[o] = if (!torch) @max(1.0 - @abs(src - fl - @as(f32, @floatFromInt(o))), 0.0) else if (o == 0) 1.0 - (src - fl) else src - fl;
    }
    return out;
}

/// PyTorch v2.10.0 CPU linspace, RangeFactoriesKernel.cpp; BSD-3-Clause in LICENSES/PyTorch-BSD-3.txt.
fn coordinate(index: u32, size: u32, side: u32) f32 {
    if (size <= 1) return 0;
    const end: f32 = @floatFromInt(side - 1);
    const step = end / @as(f32, @floatFromInt(size - 1));
    if (index < size / 2) return step * @as(f32, @floatFromInt(index));
    return @mulAdd(f32, -step, @as(f32, @floatFromInt(size - index - 1)), end);
}

fn tapsOf(comptime torch: bool, row: u32, col: u32, gh: u32, gw: u32, side: u32) Taps {
    const h = axisTaps(torch, row, gh, side);
    const w = axisTaps(torch, col, gw, side);
    var out: Taps = undefined;
    for (0..2) |a| for (0..2) |b| {
        out.index[a * 2 + b] = h.i[a] * side + w.i[b];
        out.weight[a * 2 + b] = h.w[a] * w.w[b];
    };
    return out;
}

/// The taps of a patch at (row, col) of a `gh` x `gw` grid into the `side` x `side` table.
pub fn taps(row: u32, col: u32, gh: u32, gw: u32, side: u32) Taps {
    return tapsOf(false, row, col, gh, gw, side);
}

/// `taps` with Torch's CPU interpolation arithmetic, for towers that match it bit for bit (the HIP Qwen tower).
pub fn torchTaps(row: u32, col: u32, gh: u32, gw: u32, side: u32) Taps {
    return tapsOf(true, row, col, gh, gw, side);
}

/// The axial rotary's frequencies as torch makes them in fp32: 1 / theta^(2i / (head_dim / 2)), head_dim / 4 of them.
pub fn invFreq(out: []f32, head_dim: u32, theta: f32) void {
    const half: f32 = @floatFromInt(head_dim / 2);
    for (out, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / half);
}

/// Each patch's cos and sin over a head ([row freqs | column freqs], twice), fp32 angles rounded from fp64 functions.
pub fn rotary(cos: []f32, sin: []f32, gh: u32, gw: u32, merge: u32, inv: []const f32) void {
    const quarter = inv.len;
    const dim = 4 * quarter;
    for (0..@as(usize, gh) * gw) |i| {
        const p = place(i, gw, merge);
        for (0..2 * quarter) |j| {
            const at: f32 = @floatFromInt(if (j < quarter) p[0] else p[1]);
            const angle = at * inv[j % quarter];
            const c: f32 = @floatCast(@cos(@as(f64, angle)));
            const s: f32 = @floatCast(@sin(@as(f64, angle)));
            cos[i * dim + j] = c;
            cos[i * dim + j + 2 * quarter] = c;
            sin[i * dim + j] = s;
            sin[i * dim + j + 2 * quarter] = s;
        }
    }
}

test "patches in merge order: two-by-two blocks, row by row" {
    // a 4 x 6 grid: block (0,0) holds (0,0) (0,1) (1,0) (1,1), then block (0,1) from column 2
    const want = [_][2]u32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 } };
    for (want, 0..) |w, i| try std.testing.expectEqual(w, place(i, 6, 2));
    try std.testing.expectEqual([2]u32{ 2, 0 }, place(12, 6, 2)); // block (1,0)
    try std.testing.expectEqual([2]u32{ 3, 5 }, place(23, 6, 2));
}

test "bilinear taps with align_corners: the ends land on the table's corners" {
    const first = taps(0, 0, 10, 4, 48);
    try std.testing.expectEqual(@as(u32, 0), first.index[0]);
    try std.testing.expectEqual(@as(f32, 1), first.weight[0]);
    const last = taps(9, 3, 10, 4, 48);
    try std.testing.expectEqual(@as(u32, 47 * 48 + 47), last.index[0]);
    try std.testing.expectEqual(@as(f32, 1), last.weight[0]);
    const mid = taps(1, 0, 3, 2, 48); // 1 * 47 / 2 = 23.5: halfway between rows 23 and 24
    try std.testing.expectEqual([4]u32{ 23 * 48, 23 * 48 + 1, 24 * 48, 24 * 48 + 1 }, mid.index);
    try std.testing.expectEqual([4]f32{ 0.5, 0, 0.5, 0 }, mid.weight);
}

fn bf16Bits(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

test "learned-grid linspace uses the CPU processor's FP32 step and fused tail" {
    const fixtures = [_]struct { size: u32, at: u32, bits: u32, fraction: u16 }{
        .{ .size = 144, .at = 61, .bits = 0x41a06441, .fraction = 0x3d49 },
        .{ .size = 180, .at = 107, .bits = 0x41e0c280, .fraction = 0x3dc2 },
        .{ .size = 218, .at = 37, .bits = 0x410038a1, .fraction = 0x3c63 },
        .{ .size = 240, .at = 143, .bits = 0x41e0f880, .fraction = 0x3df8 },
        .{ .size = 256, .at = 196, .bits = 0x42108080, .fraction = 0x3e00 },
        .{ .size = 5820, .at = 3, .bits = 0x3cc68006, .fraction = 0x3cc7 },
    };
    for (fixtures) |f| {
        try std.testing.expectEqual(f.bits, @as(u32, @bitCast(coordinate(f.at, f.size, 48))));
        const t = torchTaps(f.at, 0, f.size, 2, 48);
        try std.testing.expectEqual(f.fraction, bf16Bits(t.weight[2]));
        const swapped = torchTaps(0, f.at, 2, f.size, 48);
        try std.testing.expectEqual(t.weight[2], swapped.weight[1]);
    }
    try std.testing.expectEqual(@as(f32, 0), coordinate(0, 1, 48));
}

test "the shared taps keep their own arithmetic beside the Torch-matching ones" {
    const shared = taps(61, 0, 144, 2, 48);
    const torch = torchTaps(61, 0, 144, 2, 48);
    try std.testing.expectEqual(shared.index, torch.index);
    try std.testing.expectEqual(@as(u16, 0x3d48), bf16Bits(shared.weight[2]));
    try std.testing.expectEqual(@as(u16, 0x3d49), bf16Bits(torch.weight[2]));
}

test "the axial rotary of a 72-wide head: 18 row frequencies, 18 column, repeated" {
    var inv: [18]f32 = undefined;
    invFreq(&inv, 72, 10000);
    try std.testing.expectEqual(@as(f32, 1), inv[0]);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / std.math.pow(f32, 10000, 34.0 / 36.0)), inv[17], 1e-6);
    var cos: [4 * 72]f32 = undefined;
    var sin: [4 * 72]f32 = undefined;
    rotary(&cos, &sin, 2, 2, 2, &inv);
    // patch 3 sits at (1, 1): its row and column angles are both 1 x the frequency
    try std.testing.expectEqual(@as(f32, @floatCast(@cos(@as(f64, 1)))), cos[3 * 72 + 0]);
    try std.testing.expectEqual(cos[3 * 72 + 0], cos[3 * 72 + 18]);
    try std.testing.expectEqual(cos[3 * 72 + 5], cos[3 * 72 + 41]);
    // patch 1 sits at (0, 1): rows turn by 0, columns by 1
    try std.testing.expectEqual(@as(f32, 0), sin[1 * 72 + 0]);
    try std.testing.expectEqual(@as(f32, @floatCast(@sin(@as(f64, 1)))), sin[1 * 72 + 18]);
}
