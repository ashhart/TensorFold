//! The routed prompt's matrix-core tile of few rows an item (tiles/matrix_item.hpp): its functions and its launch.

const std = @import("std");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const util = @import("util.zig");
const registry = @import("core").registry;
const tuning = @import("../tuning/tuning.zig");

/// The 16-row tiles a block holds, by tier: 64 rows.
pub const row_tiles = [_]usize{4};
/// The most rows a tuning row's `few_rows` can cut at: what the tile sends to its K-parallel passes (kItemFewRows).
pub const few_rows = 8;
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
    try goWith(k.item[b][tier], c, tier);
}

/// `go` on a tile function of the tier's geometry: the tier's own, or a contract control in its place.
pub fn goWith(f: Function, c: affine.Call, tier: usize) util.Error!void {
    if (c.s == util.counting) return;
    var args: hl.Args = .{};
    try args.add(c.arg);
    const grid = util.dim(util.cdiv(c.arg.n, 128), util.cdiv(c.arg.m, rows(tier)), c.items);
    try hl.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, .{ .r = c.r, .handle = c.s }, &args);
}

// ---- the cut, from the tuning table to the launch

var cut_launched: c_int = -1;

fn keepCut(_: *const affine.Kernels, c: affine.Call) util.Error!void {
    cut_launched = c.arg.route.few;
}

fn anyEnv(_: registry.Env) bool {
    return true;
}

fn anyShape(_: registry.Shape) bool {
    return true;
}

fn toy(id: []const u8, op: registry.Op) affine.Entry {
    return .{ .id = id, .format = .mlx, .op = op, .path = .prefill, .family = .gemm, .caps = &anyEnv, .policy = &anyEnv, .fits = &anyShape, .launch = &keepCut };
}

test "a prompt launch carries its entry's cut from the tuning table, and none for a row without one" {
    const entries = [_]affine.Entry{ toy("toy.routed.prefill", .routed), toy("toy.project.prefill", .project) };
    const tuned = [_]tuning.Row{ .{ .id = "toy.routed.prefill", .cost = 1, .few_rows = few_rows }, .{ .id = "toy.project.prefill", .cost = 1 } };
    var k: affine.Kernels = undefined;
    k.wmma = true;
    k.matrix = true;
    k.mode = .auto;
    k.tile = .gemm;
    k.stream_on = true;
    k.reg = affine.Registry.init(&.{ .gfx = "toy", .rows = &tuned }, &entries);
    // the caller's route holds another cut: the table's replaces it, for a plan and for a plain product
    var a = std.mem.zeroes(affine.Arg);
    a.m, a.n, a.k, a.bits, a.group = .{ 4, 128, 128, 4, 32 };
    a.route = .{ .items = 0x1000, .few = 3 };
    try k.prefillLaunch(undefined, a, util.counting, 2, 4, 2);
    try std.testing.expectEqual(@as(c_int, few_rows), cut_launched);
    a.route = .{ .few = 3 };
    try k.prefillLaunch(undefined, a, util.counting, 1, 0, 0);
    try std.testing.expectEqual(@as(c_int, 0), cut_launched);
}
