//! Gate 5 for the prompt tile over dequantized weights: gfx11's matrix-core tile, which only those chips resolve.
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The matrix-core prompt tile that dequantizes its weights at the stage, `bits` wide, and the chips that have it.
fn dqTile(comptime bits: u32) limits.HipLaunch {
    return .{
        .name = std.fmt.comptimePrint("dequantizing prompt tile {d}-bit", .{bits}),
        .symbol = std.fmt.comptimePrint("_ZN2tf4rocm19affine_wmma_gemm_dqILi{d}EEEvNS0_6AffineE", .{bits}),
        .image = "affine_tiles",
        .threads = 256,
        // gfx1030 has no matrix cores and gfx1201's WMMA has other layouts, so neither resolves this tile
        .chips = .{ false, true, true, false },
    };
}

/// A width's row on either gfx11 chip: `vgprs` a wave, 40 KiB of LDS, and no scratch.
fn row(comptime vgprs: u32) limits.HipUsage {
    return .{ .vgprs = vgprs, .sgprs = 42, .lds = 40960, .private = 0, .bound = 256, .wave = 32 };
}

const none = limits.HipUsage{ .vgprs = 0 };

pub const records = [_]Record{
    .{ .launch = dqTile(2), .usage = .{ none, row(194), row(194), none } },
    .{ .launch = dqTile(3), .usage = .{ none, row(196), row(196), none } },
    .{ .launch = dqTile(4), .usage = .{ none, row(198), row(198), none } },
    .{ .launch = dqTile(5), .usage = .{ none, row(200), row(200), none } },
    .{ .launch = dqTile(6), .usage = .{ none, row(202), row(202), none } },
    .{ .launch = dqTile(8), .usage = .{ none, row(206), row(206), none } },
};
