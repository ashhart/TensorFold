//! Resolving the affine kernels from the tile and dot2 code objects, and their once-a-device tables.

const std = @import("std");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Caps = @import("../caps.zig").Caps;
const mlx_entries = @import("mlx_entries.zig");
const affine = @import("affine.zig");

const Error = runtime.Error;
const Kernels = affine.Kernels;
const Choices = affine.Choices;
const Registry = affine.Registry;
const tableFor = Kernels.tableFor;
const bit_widths = affine.bit_widths;
const row_counts = affine.row_counts;
const piece_counts = affine.piece_counts;
const stream_cbs = affine.stream_cbs;
const stream_rows = affine.stream_rows;
const kp_tiles = affine.kp_tiles;

/// `tiles` holds dot2_tiles.hip's kernels, `dot2` dot2.hip's; bf16 on the WMMA build, fp16 on RDNA2.
pub fn load(tiles_obj: Module, dot2_obj: Module, caps: Caps, choices: Choices) Error!Kernels {
    var k: Kernels = undefined;
    const wmma = caps.act == .bf16;
    k.caps = caps;
    k.wmma = wmma;
    k.matrix = wmma and caps.matrix == .wmma11;
    k.mode = choices.matrix;
    k.tile = if (choices.gemm) .gemm else .block;
    k.stream_on = choices.stream;
    k.stream_cb_force = 0;
    k.reg = Registry.init(tableFor(caps), &mlx_entries.all);
    k.reg.verify(k.env(), .mlx) catch return error.Invalid;
    // the GEMM tile's rows of x a lane keeps (tiles/gemm.hpp GemmShape): 8 for BF16, 16 for FP16
    if (wmma) try resolve(&k, "7DotBF16", "Li8E", tiles_obj) else try resolve(&k, "6DotF16", "Li16E", tiles_obj);
    const r = "_ZN2tf4rocm";
    k.span = .{ try dot2_obj.function(r ++ "16affine_dot2_spanILi1EEEvNS0_6AffineEPfi"), try dot2_obj.function(r ++ "16affine_dot2_spanILi8EEEvNS0_6AffineEPfi") };
    k.fold = try dot2_obj.function(r ++ "16affine_dot2_foldEPKfNS0_10GroupTableES3_Pfiii");
    k.wide = try dot2_obj.function(r ++ "16affine_dot2_wideENS0_6AffineE");
    k.tiled = try dot2_obj.function(r ++ "17affine_dot2_tiledENS0_6AffineE");
    k.reference = try dot2_obj.function(r ++ "18affine_dot2_kernelENS0_6AffineE");
    k.fill = try dot2_obj.function(r ++ "13fill_byte_lutEP6__half");
    k.moe_gu = if (k.matrix) try tiles_obj.function(r ++ "18affine_wmma_moe_guILi4EEEvNS0_6AffineEf") else null;
    k.moe_down = if (k.matrix) try tiles_obj.function(r ++ "20affine_wmma_moe_downILi4EEEvNS0_6AffineE") else null;
    return k;
}

fn resolve(k: *Kernels, comptime dot: []const u8, comptime shape: []const u8, m: Module) Error!void {
    @setEvalBranchQuota(400_000);
    inline for (.{ 4, 8 }, 0..) |rows, r| {
        inline for (stream_cbs, 0..) |cb, c| {
            k.held[r][c] = if ((rows == 4 and cb <= 4) or (rows == 8 and (cb == 2 or cb == 4)))
                try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_dot2_stream_heldINS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, rows, cb }))
            else
                null;
        }
    }
    k.stream_list = .{
        try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_dot2_stream_listINS0_{s}ELi4ELi4ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEiii", .{dot})),
        try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_dot2_stream_listINS0_{s}ELi4ELi4ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEiii", .{dot})),
    };
    inline for (bit_widths, 0..) |bits, b| {
        k.block[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_blockINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ dot, bits }));
        if (k.matrix) k.wmma_gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_gemmILi{d}EEEvNS0_6AffineE", .{bits}));
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
                k.pairs[b][r][c] = if (cb * rows <= 16 and cb >= 2)
                    try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb1EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                else
                    null;
                k.relu2[b][r][c] = if (cb * rows <= 16 and cb >= 2)
                    try m.function(
                        std.fmt.comptimePrint("_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}" ++
                            "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }),
                    )
                else
                    null;
                k.relu2_pair[b][r][c] = if (cb * rows <= 16 and cb >= 2)
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

/// fp16(bf16(code)) for every byte into the module's constant table: the RDNA2 fp16 tiles read it. Once a device.
pub fn fillByteLut(k: *const Kernels, r: *const runtime.Runtime, dot2_obj: Module) Error!void {
    if (k.wmma) return;
    const lut = try dot2_obj.global("_ZN2tf4rocm8kByteF16E");
    var args: hl.Args = .{};
    try args.add(lut.address);
    try hl.launch(k.fill, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, .{ .r = r, .handle = null }, &args);
    try runtime.check(r.api.hipStreamSynchronize(null));
}
