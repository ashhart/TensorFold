//! The affine tiles' kernels by width, as the tiles module names them: the part of Kernels.load that finds them.

const std = @import("std");
const Module = @import("../module.zig").Module;
const Error = @import("../runtime.zig").Error;
const affine = @import("affine.zig");
const matrix_item = @import("matrix_item.zig");
const matrix_rows = @import("matrix_rows.zig");
const Kernels = affine.Kernels;
const bit_widths = affine.bit_widths;
const kp_tiles = affine.kp_tiles;
const piece_counts = affine.piece_counts;
const row_counts = affine.row_counts;
const stream_rows = affine.stream_rows;
const stream_cbs = affine.stream_cbs;

/// Every width's tiles into `k`: `dot` the dot2 type's mangled name, `shape` the GEMM tile's rows of x a lane keeps.
pub fn resolve(k: *Kernels, comptime dot: []const u8, comptime shape: []const u8, m: Module) Error!void {
    @setEvalBranchQuota(300_000);
    inline for (bit_widths, 0..) |bits, b| {
        k.block[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_blockINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ dot, bits }));
        if (k.matrix_item) {
            k.wmma_gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_gemmILi{d}EEEvNS0_6AffineE", .{bits}));
            k.wmma_skew[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_skewILi{d}EEEvNS0_6AffineE", .{bits}));
            k.item[b] = try matrix_item.load(m, bits);
        } else if (k.matrix) {
            k.wmma_gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_wmma12_gemmILi{d}EEEvNS0_6AffineE", .{bits}));
            k.rows12[b] = try matrix_rows.load(m, bits);
        }
        k.gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_gemm_blockINS0_{s}ELi{d}E{s}EEvNS0_6AffineE", .{ dot, bits, shape }));
        inline for (kp_tiles, 0..) |t, i| {
            k.kp[b][i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm14affine_gemm_kpINS0_{s}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineE", .{ dot, bits, t.cb, t.r, t.waves, t.rb, @intFromBool(t.loop) }));
        }
        inline for (piece_counts, 0..) |pieces, p| {
            k.row[b][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_rowINS0_{s}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot, bits, pieces }));
            inline for (row_counts, 0..) |rows, r| {
                k.lanes[b][r][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_lanesINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineE", .{ dot, bits, rows, pieces }));
            }
        }
        inline for (stream_rows, 0..) |rows, r| {
            inline for (stream_cbs, 0..) |cb, c| {
                k.stream[b][r][c] = if (cb * rows <= 16)
                    try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb0EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                else
                    null;
                k.pairs[b][r][c] = if (cb * rows <= 16)
                    try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb1EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                else
                    null;
                k.relu2[b][r][c] = if (cb * rows <= 16)
                    try m.function(
                        std.fmt.comptimePrint("_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}" ++
                            "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }),
                    )
                else
                    null;
                k.relu2_pair[b][r][c] = if (cb * rows <= 16)
                    try m.function(
                        std.fmt.comptimePrint("_ZN2tf4rocm29affine_dot2_stream_relu2_pairINS0_{s}" ++
                            "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineES3_iii", .{ dot, bits, rows, cb }),
                    )
                else
                    null;
            }
        }
    }
}
