//! The EXL3 mid-M kernels and linear_wc (17-128 rows): their variants, shared bytes and which one takes a weight.
const std = @import("std");
const exl3 = @import("exl3.zig");
const Weight = exl3.Weight;
const Codebook = exl3.Codebook;

/// The mid-M kernels (17-128 rows, the same bits as the linear's): mul1 at 3 to 6 bits, 2, 4 and 8 warps, P 1 and 2.
pub const Mpg = struct { halves: u32, warps: u32, p: u32 };
pub const mpg_variants = blk: {
    var vs: [24]Mpg = undefined;
    var i: usize = 0;
    for ([_]u32{ 6, 8, 10, 12 }) |h| for ([_]u32{ 1, 2 }) |p| for ([_]u32{ 2, 4, 8 }) |w| {
        vs[i] = .{ .halves = h, .warps = w, .p = p };
        i += 1;
    };
    break :blk vs;
};

/// linear_wc (17-128 rows, 4- and 6-bit mul1): 16 mt rows a program, ks k steps a stage, fold: every split in one.
pub const Wc = struct { halves: u32, mt: u32, ks: u32, fold: bool };
pub const wc_variants = blk: {
    var vs: [16]Wc = undefined;
    var i: usize = 0;
    for ([_]u32{ 8, 12 }) |h| for ([_]bool{ false, true }) |f| for ([_]u32{ 2, 4 }) |mt| for ([_]u32{ 2, 4 }) |ks| {
        vs[i] = .{ .halves = h, .mt = mt, .ks = ks, .fold = f };
        i += 1;
    };
    break :blk vs;
};

pub fn wcShared(v: Wc) u32 {
    const raw = 8 * 4 * v.halves;
    const stage = v.ks * raw * 4 + v.ks * 16 * v.mt * 32;
    return @max(4 * stage, 16 * v.mt * 128 * 4);
}

/// linear.cu dispatch_wc: which linear_wc takes m rows of this weight, or null for the mid-M kernels.
pub fn wcPick(w: Weight, m: usize, sk: u32, wk: u32, min_rows: usize) ?usize {
    if (w.codebook != .mul1 or (w.halves != 8 and w.halves != 12)) return null;
    const per_warp = w.k / 16 / sk / wk;
    if (per_warp % 2 != 0 or m < min_rows or m > 128) return null;
    const nb = w.n / 128;
    const fold = sk > 1 and nb >= 32;
    if (!fold and w.halves == 8 and m > 32 and m <= 40) return null;
    const four = per_warp % 4 == 0 and !(w.halves == 12 and m > 96);
    const mt2 = m <= 32 or (m > 64 and m <= 96) or (fold and m > 96 and nb >= 64);
    const want: Wc = .{ .halves = w.halves, .mt = if (mt2) 2 else 4, .ks = if (four) 4 else 2, .fold = fold };
    for (wc_variants, 0..) |v, i| if (std.meta.eql(v, want)) return i;
    return null;
}

pub fn mpgIndex(halves: u32, codebook: Codebook, warps: u32, p: u32) ?usize {
    if (codebook != .mul1) return null;
    for (mpg_variants, 0..) |v, i| if (v.halves == halves and v.warps == warps and v.p == p) return i;
    return null;
}
