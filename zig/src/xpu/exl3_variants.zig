// SPDX-License-Identifier: Apache-2.0
//! The exl3_matvec builds in the binary: one SPIR-V image per (bits, rows, NS, HS), and the rule that picks one.
//! zig/build/xpu.zig and zig/src/xpu/exl3.zig both import this file, so the images and the picks agree.

const std = @import("std");

pub const Variant = struct { bits: u32, rows: u32, ns: u32, hs: bool };

/// Sage EXL3: 4 bits (language layers and MTP), 6 bits (vision and lm_head).
pub const bits = [_]u32{ 4, 6 };
/// Row buckets: a launch of r rows runs the smallest bucket >= r.
pub const rows = [_]u32{ 1, 2, 4, 8, 16 };
pub const rows_max = rows[rows.len - 1];
const strips_per_group = [_]u32{ 1, 2, 4 };

/// HS (split half) at 2 to 4 rows; it needs two strips a sub-group.
fn hsFor(r: u32, ns: u32) bool {
    return r >= 2 and r <= 4 and ns >= 2;
}

/// Every image: bits x row buckets x NS 1, 2, 4.
pub const all = blk: {
    var out: [bits.len * rows.len * strips_per_group.len]Variant = undefined;
    var i: usize = 0;
    for (bits) |b| for (rows) |r| for (strips_per_group) |ns| {
        out[i] = .{ .bits = b, .rows = r, .ns = ns, .hs = hsFor(r, ns) };
        i += 1;
    };
    break :blk out;
};

/// K splits for a launch of `sub_groups` sub-groups over groups whose K / 16 all divide by `k_tiles_gcd`.
pub fn split(sub_groups: u32, k_tiles_gcd: u32) u32 {
    var ks: u32 = 1;
    while (ks < ks_max and sub_groups * ks < sub_group_target and k_tiles_gcd % (2 * ks) == 0 and k_tiles_gcd / (2 * ks) >= ks_min_tiles) ks *= 2;
    return ks;
}

pub fn name(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("exl3_b{d}_r{d}_n{d}{s}", .{ v.bits, v.rows, v.ns, if (v.hs) "_hs" else "" });
}

pub fn options(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DROWS={d} -DNS={d} -DHS={d}", .{ v.bits, v.rows, v.ns, @intFromBool(v.hs) });
}

/// The fused routed MLP (exl3_moe_*): one image a width. K splits and strips measured on the B70 with one Sage
/// layer (8 picks and the shared expert, 1 token) by tools/xpu/exl3_moe_check.py (2026-10-25).
pub const moe = struct {
    pub const bits = [_]u32{ 4, 6 };
    /// gate|up K splits (a work-group each), down K splits and down strips a sub-group.
    pub const p = 4;
    pub const pd = 2;
    pub const ds = 2;
    /// Largest K of gate|up (hidden) and of down (intermediate): local memory a work-group.
    pub const k_max = 4096;

    /// Prompt rows (P3.4, the same image): exl3.cl GEMM_M rows a sub-group and GEMM_S strips a work-group of
    /// exl3_gemm, GEMM_KC k tiles a local-memory step, TILE_NS strips a sub-group of exl3_moe_tile, and the most rows
    /// of a prompt launch. The kernel defaults are these values.
    pub const gemm_m = 16;
    pub const gemm_s = 4;
    pub const gemm_kc = 8;
    pub const tile_ns = 2;
    pub const tile_rows = 16;
    pub const prompt_rows_max = 1024;

    pub fn name(comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("exl3_moe_b{d}", .{b});
    }

    pub fn options(comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DMOE_P={d} -DMOE_PD={d} -DMOE_DS={d} -DMOE_KMAX={d}", .{ b, p, pd, ds, k_max });
    }
};

/// Sub-groups a launch needs to fill the GPU; fewer strips a sub-group and K splits get it there.
pub const sub_group_target = 4096;
/// Most K splits (the sums scratch holds this many) and fewest 16-row K tiles a split.
pub const ks_max = 16;
const ks_min_tiles = 8;

/// The strip counts (N / 16) of a launch's groups: smallest, common divisor and total.
pub const Strips = struct { min: u32, gcd: u32, total: u32 };

/// The image for `r` rows of `b` bits. Measured on the B70 by tools/xpu/exl3_ab.py (2026-10-24 and -25), the rule
/// of Exl3.shape there: the per-group NS, then fewer strips a sub-group until the launch reaches
/// sub_group_target. Null: no image for these bits or more than rows_max rows.
pub fn pick(b: u32, r: u32, s: Strips) ?usize {
    const bucket = for (rows) |x| {
        if (x >= r) break x;
    } else return null;
    var ns: u32 = if (s.min < 128) // few strips a group (expert gate/up, N 512): keep the sub-group count up
        (if (bucket <= 4) 2 else 1)
    else if (bucket <= 4) 4 else if (bucket <= 8) 2 else 1;
    while (ns > 1 and s.total / ns < sub_group_target) ns /= 2;
    while (s.gcd % ns != 0) ns /= 2;
    for (all, 0..) |v, i| if (v.bits == b and v.rows == bucket and v.ns == ns) return i;
    return null;
}

test "picks follow the measured rule" {
    const at = struct {
        fn v(b: u32, r: u32, smin: u32, sgcd: u32) Variant {
            return all[pick(b, r, .{ .min = smin, .gcd = sgcd, .total = smin * 320 }).?];
        }
    }.v;
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 1, .ns = 4, .hs = false }, at(6, 1, 15520, 15520));
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 4, .ns = 4, .hs = true }, at(6, 3, 15520, 15520));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 8, .ns = 2, .hs = false }, at(4, 5, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 16, .ns = 1, .hs = false }, at(4, 16, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 2, .ns = 2, .hs = true }, at(4, 2, 32, 32));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 1, .ns = 1, .hs = false }, at(4, 1, 33, 33));
    try std.testing.expectEqual(@as(?usize, null), pick(5, 1, .{ .min = 128, .gcd = 128, .total = 1024 }));
    try std.testing.expectEqual(@as(?usize, null), pick(4, 17, .{ .min = 128, .gcd = 128, .total = 1024 }));
    // one layer's top-8: NS 1, then K splits up to the target
    try std.testing.expectEqual(@as(u32, 1), all[pick(4, 1, .{ .min = 32, .gcd = 32, .total = 256 }).?].ns);
    try std.testing.expectEqual(@as(u32, 16), split(256, 128));
    try std.testing.expectEqual(@as(u32, 4), split(1024, 32));
    try std.testing.expectEqual(@as(u32, 2), split(3880, 128));
    try std.testing.expectEqualStrings("exl3_b6_r4_n4_hs", name(all[all.len - 7]));
}
