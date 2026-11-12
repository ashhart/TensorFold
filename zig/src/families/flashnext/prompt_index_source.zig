//! qmmw_nax.metal's expert gathers reading scales and biases through scale_index's index and table.
const std = @import("std");

/// Each read of the two K loops and each gather's row pointers: the scale row is the index row, the bias the table.
const reads = [_][2][]const u8{
    .{ "  T sc = srow[hk / WG], bi = brow[hk / WG];\n", "  const device ushort* ixrow = (const device ushort*)srow;\n  uint si = uint(ixrow[hk / WG]) * 2u;\n  T sc = brow[si], bi = brow[si + 1];\n" },
    .{ "      sc = srow[(hk + k + 64) / WG];\n      bi = brow[(hk + k + 64) / WG];\n", "      si = uint(ixrow[(hk + k + 64) / WG]) * 2u;\n      sc = brow[si];\n      bi = brow[si + 1];\n" },
    .{ "  T gs = sg[hk / WG], gbi = bg[hk / WG], us = su[hk / WG], ubi = bu[hk / WG];\n", "  const device ushort* ixg = (const device ushort*)sg;\n  const device ushort* ixu = (const device ushort*)su;\n  uint gi = uint(ixg[hk / WG]) * 2u, ui = uint(ixu[hk / WG]) * 2u;\n  T gs = bg[gi], gbi = bg[gi + 1], us = bu[ui], ubi = bu[ui + 1];\n" },
    .{ "      gs = sg[g];\n      gbi = bg[g];\n      us = su[g];\n      ubi = bu[g];\n", "      gi = uint(ixg[g]) * 2u;\n      ui = uint(ixu[g]) * 2u;\n      gs = bg[gi];\n      gbi = bg[gi + 1];\n      us = bu[ui];\n      ubi = bu[ui + 1];\n" },
    .{ "W + wrow * WPR + WB * (t % 2), S + wrow * KG, B + wrow * KG, 0, tile, tn, uint(t),", "W + wrow * WPR + WB * (t % 2), S + wrow * KG, B, 0, tile, tn, uint(t)," },
    .{ "WG_ + wrow * WPR + WB * (t % 2), SG + wrow * KG, BG + wrow * KG,", "WG_ + wrow * WPR + WB * (t % 2), SG + wrow * KG, BG," },
    .{ "WU + wrow * WPR + WB * (t % 2), SU + wrow * KG, BU + wrow * KG, tile_g, tile_u, tn,", "WU + wrow * WPR + WB * (t % 2), SU + wrow * KG, BU, tile_g, tile_u, tn," },
    .{ "[[kernel]] void tf_gather_qmmw_nax_##BM(", "[[max_total_threads_per_threadgroup(128)]] [[kernel]] void tf_gather_qmmw_nax_##BM(" },
    .{ "[[kernel]] void tf_gather_guw_nax_##BM(", "[[max_total_threads_per_threadgroup(128)]] [[kernel]] void tf_gather_guw_nax_##BM(" },
};

/// `text` for the gather kernels alone: the dense kernels share the patched loops, so they keep the unpatched text.
pub fn patch(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out = text;
    for (reads) |r| {
        if (std.mem.count(u8, out, r[0]) != 1) return error.KernelPatch;
        out = try std.mem.replaceOwned(u8, a, out, r[0], r[1]);
    }
    return out;
}

test "every scale and bias read of the prompt gathers goes through the index once patched" {
    const ks = @import("kernel_sources");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const out = try patch(arena.allocator(), ks.flashnext_qmmw);
    for ([_][]const u8{ "srow[", "sg[", "su[", "brow[hk", "bg[hk", "bu[hk", "bg[g]", "bu[g]" }) |direct| try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, out, direct));
    try std.testing.expectError(error.KernelPatch, patch(arena.allocator(), out));
}
