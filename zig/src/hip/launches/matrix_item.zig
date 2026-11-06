//! The routed prompt's matrix-core tile of few rows an item (tiles/matrix_item.hpp): its functions and its launch.

const std = @import("std");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const util = @import("util.zig");

/// The 16-row tiles a block holds, by tier: 64 rows.
pub const row_tiles = [_]usize{4};
pub const Functions = [affine.bit_widths.len][row_tiles.len]Function;

/// The tier's functions at a width, as the module names them.
pub fn load(m: Module, comptime bits: c_int) util.Error![row_tiles.len]Function {
    var out: [row_tiles.len]Function = undefined;
    inline for (row_tiles, 0..) |rm, i| {
        out[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_itemILi{d}ELi{d}EEEvNS0_6AffineE", .{ bits, rm }));
    }
    return out;
}

/// The rows of a tier's block.
pub fn rows(tier: usize) usize {
    return 16 * row_tiles[tier];
}

/// Tier `tier` over every item of a plan: 128 columns by the tier's rows a block, all waves across the columns.
pub fn go(k: *const affine.Kernels, c: affine.Call, tier: usize) util.Error!void {
    const b = std.mem.indexOfScalar(c_int, &affine.bit_widths, c.arg.bits) orelse return error.Invalid;
    if (c.s == util.counting) return;
    var args: hl.Args = .{};
    try args.add(c.arg);
    const grid = util.dim(util.cdiv(c.arg.n, 128), util.cdiv(c.arg.m, rows(tier)), c.items);
    try hl.launch(k.item[b][tier], .{ .grid = grid, .block = .{ .x = 256 } }, .{ .r = c.r, .handle = c.s }, &args);
}
