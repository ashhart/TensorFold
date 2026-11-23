//! gfx12's matrix-core affine tile on blocks of few rows (tiles/wmma12_affine.hpp): its functions and its launch.

const std = @import("std");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const util = @import("util.zig");
const kernels = @import("../kernels.zig");

/// A block's rows, columns and threads by tier: W12Affine64, W12Affine32, the K-parallel W12AffineKp (8 x 32 cols).
pub const row_tiers = [_]usize{ 64, 32, 16 };
pub const col_tiers = [_]usize{ 128, 128, 32 };
/// The waves that split k, for the K-parallel tier (0: a staged block); W12AffineKp's W and 16 TN columns.
const kp_waves = [_]usize{ 0, 0, 8 };
const threads = [_]c_uint{ 256, 128, 32 * kp_waves[kp_tier] };
/// The K-parallel tier, W12AffineKp.
pub const kp_tier = 2;
/// An item's most row blocks a launch has (0: one); the K-parallel tier loops past them, splitting a heavy item.
const row_blocks = [_]usize{ 0, 0, 2 };
pub const Functions = [affine.bit_widths.len][row_tiers.len]Function;

/// The tiers' functions at a width, as the module names them.
pub fn load(m: Module, comptime bits: c_int) util.Error![row_tiers.len]Function {
    var out: [row_tiers.len]Function = undefined;
    inline for (row_tiers, col_tiers, kp_waves, 0..) |rows, cols, waves, i| {
        out[i] = if (waves != 0)
            try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma12_kpILi{d}ELi{d}ELi{d}EEEvNS0_6AffineE", .{ bits, waves, cols / 16 }))
        else
            try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_wmma12_rowsILi{d}ELi{d}EEEvNS0_6AffineE", .{ bits, rows }));
    }
    return out;
}

/// The fewest blocks a plain product should give gfx1201's 64 compute units (measured there only; others keep 128).
const fill_target: ?usize = if (std.mem.eql(u8, kernels.arch, "gfx1201")) 64 else null;

/// A staged tier (64, then 32 rows) for a plain product with too few 128 x 128 blocks: same bytes. Null: 128 rows.
pub fn fillTier(m: c_int, n: c_int) ?usize {
    const target = fill_target orelse return null;
    const cols = util.cdiv(n, 128);
    if (cols * util.cdiv(m, 128) >= target) return null;
    return if (cols * util.cdiv(m, row_tiers[0]) >= target) 0 else 1;
}

/// Tier `tier` over every item of a plan (or the plain product): the tier's columns by its rows a block.
pub fn go(k: *const affine.Kernels, c: affine.Call, tier: usize) util.Error!void {
    const b = std.mem.indexOfScalar(c_int, &affine.bit_widths, c.arg.bits) orelse return error.Invalid;
    try goWith(k.rows12[b][tier], c, tier);
}

/// `f` (the tier's function, or a contract control in its place) on the tier's grid and block.
pub fn goWith(f: Function, c: affine.Call, tier: usize) util.Error!void {
    if (c.s == util.counting) return;
    var args: hl.Args = .{};
    try args.add(c.arg);
    const rows = util.cdiv(c.arg.m, row_tiers[tier]);
    const grid = util.dim(util.cdiv(c.arg.n, col_tiers[tier]), if (row_blocks[tier] == 0) rows else @min(rows, row_blocks[tier]), c.items);
    try hl.launch(f, .{ .grid = grid, .block = .{ .x = threads[tier] } }, .{ .r = c.r, .handle = c.s }, &args);
}
