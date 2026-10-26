// SPDX-License-Identifier: Apache-2.0

//! exl3_matvec SPIR-V images and the host rule that picks (bits, rows, NS).

const std = @import("std");

pub const Variant = struct { bits: u32, rows: u32, ns: u32, hs: bool };

pub const bits = [_]u32{ 4, 6 };
pub const rows = [_]u32{ 1, 2, 4, 8, 16 };
pub const rows_max = rows[rows.len - 1];
const strips_per_group = [_]u32{ 1, 2, 4 };

fn hsFor(_: u32, _: u32) bool {
    return false;
}

pub const all = blk: {
    var out: [bits.len * rows.len * strips_per_group.len]Variant = undefined;
    var i: usize = 0;
    for (bits) |b| for (rows) |r| for (strips_per_group) |ns| {
        out[i] = .{ .bits = b, .rows = r, .ns = ns, .hs = hsFor(r, ns) };
        i += 1;
    };
    break :blk out;
};

pub fn split(k_tiles_gcd: u32) u32 {
    var ks: u32 = 1;
    while (ks < ks_max and k_tiles_gcd % (2 * ks) == 0 and k_tiles_gcd / (2 * ks) >= ks_min_tiles) ks *= 2;
    return ks;
}

pub fn name(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("exl3_b{d}_r{d}_n{d}{s}", .{ v.bits, v.rows, v.ns, if (v.hs) "_hs" else "" });
}

pub fn options(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DROWS={d} -DNS={d} -DHS={d}", .{ v.bits, v.rows, v.ns, @intFromBool(v.hs) });
}

pub const moe = struct {
    pub const bits = [_]u32{ 4, 6 };
    pub const p = 4;
    pub const pd = 2;
    pub const ds = 2;
    pub const k_max = 4096;

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

pub const sub_group_target = 4096;
pub const ks_max = 16;
const ks_min_tiles = 8;

pub const Strips = struct { min: u32, gcd: u32, total: u32 };

pub fn pick(b: u32, r: u32, s: Strips) ?usize {
    const bucket = for (rows) |x| {
        if (x >= r) break x;
    } else return null;
    var ns: u32 = if (s.min >= 128) 4 else 2;
    while (ns > 1 and s.gcd % ns != 0) ns /= 2;
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
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 4, .ns = 4, .hs = false }, at(6, 3, 15520, 15520));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 8, .ns = 4, .hs = false }, at(4, 5, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 16, .ns = 4, .hs = false }, at(4, 16, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 2, .ns = 2, .hs = false }, at(4, 2, 32, 32));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 1, .ns = 1, .hs = false }, at(4, 1, 33, 33));
    try std.testing.expectEqual(@as(?usize, null), pick(5, 1, .{ .min = 128, .gcd = 128, .total = 1024 }));
    try std.testing.expectEqual(@as(?usize, null), pick(4, 17, .{ .min = 128, .gcd = 128, .total = 1024 }));
    // NS follows the smallest strip count, not the launch total, so one group matches alone or beside others.
    try std.testing.expectEqual(@as(u32, 2), all[pick(4, 1, .{ .min = 32, .gcd = 32, .total = 256 }).?].ns);
    try std.testing.expectEqual(@as(u32, 2), all[pick(4, 1, .{ .min = 32, .gcd = 32, .total = 32 }).?].ns);
    try std.testing.expectEqual(@as(u32, 16), split(128));
    try std.testing.expectEqual(@as(u32, 4), split(32));
    try std.testing.expectEqual(@as(u32, 2), split(16));
    try std.testing.expectEqualStrings("exl3_b6_r16_n4", name(all[all.len - 1]));
}
