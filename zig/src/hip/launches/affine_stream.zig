//! The stream tile's launch: the column rule, the held-codes tile and the routed work lists (tiles/stream.hpp).

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");

const Error = runtime.Error;
const Kernels = affine.Kernels;
const Arg = affine.Arg;
const StreamSides = affine.StreamSides;
const go = Kernels.go;
const cdiv = Kernels.cdiv;
const bitIndex = Kernels.bitIndex;
const stream_cbs = affine.stream_cbs;
const stream_rows = affine.stream_rows;
const stream_waves = affine.stream_waves;
const streams_wanted = affine.streams_wanted;
const stream_code_words = affine.stream_code_words;

/// A SwiGLU pair's lane carries at most this many code words (at least 2 columns): more waves in flight, same sums.
pub const stream_pair_code_words = 8;
/// The widest columns a lane of a routed plain product (the experts' down) takes, by row tile (stream_rows).
pub const stream_routed_cb_top = [stream_rows.len]c_int{ 8, 4, 4 };
/// The blocks on z of a routed work list (stream_list): each walks about live items / 16 of the plan's items
pub const stream_list_blocks = 16;
/// The columns below which one row of a plain product takes one column a lane; wider heads keep the waves rule.
pub const stream_one_col_n = 20_000;

/// Where an unrouted 4-bit plain product takes the held-codes tile: 3-4 rows on the rule's columns, 5-8 rows by width.
pub const HeldPick = struct { r: usize, c: usize, rows: c_int };
pub const held_wide_n = 16_000;
pub const held_narrow_lo = 1_000;
pub const held_narrow_hi = 4_096;
pub fn heldPick(bits: c_int, m: c_int, total_n: c_int, cb: c_int) ?HeldPick {
    if (bits != 4 or m < 3 or m > 8) return null;
    const ci = std.mem.indexOfScalar(c_int, &stream_cbs, cb) orelse return null;
    if (m <= 4) return if (cb <= 4) .{ .r = 0, .c = ci, .rows = 4 } else null;
    if (total_n >= held_wide_n) return .{ .r = 1, .c = 2, .rows = 8 };
    if (total_n > held_narrow_lo and total_n <= held_narrow_hi) return .{ .r = 1, .c = 1, .rows = 8 };
    return null;
}

test "heldPick: the held tile at 3 and 4 rows on the rule's columns, at 5 to 8 rows by width" {
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 1, 16480, 1));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 2, 248320, 8));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 0, .c = 0, .rows = 4 }), heldPick(4, 3, 2560, 1));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 0, .c = 2, .rows = 4 }), heldPick(4, 4, 16480, 4));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 1, .c = 2, .rows = 8 }), heldPick(4, 6, 16480, 4));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 1, .c = 1, .rows = 8 }), heldPick(4, 8, 2560, 1));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 5, 13952, 4));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 7, 324, 1));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(8, 3, 2560, 1));
}

pub fn streamWith(
    k: *const Kernels,
    r: *const runtime.Runtime,
    a: Arg,
    sides_in: StreamSides,
    total_n: c_int,
    items: c_int,
    s: abi.Stream,
    relu2: bool,
) Error!void {
    const b = bitIndex(a.bits).?;
    var sides = sides_in;
    // a pair's lane carries half as many output columns: a gate column and its up column make one
    const pair_div: c_int = if (sides.pair_cols > 0) 2 else 1;
    var lpc_log2: u5 = 0;
    while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
    const ri: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else 2;
    // the widest column count that still fills the card with waves, within the registers the lane's code words take
    var pick: usize = 0;
    var c: usize = stream_cbs.len;
    while (c > 0) {
        c -= 1;
        // one column a lane is the plain product's only (a pair's gate and up need two; relu^2 keeps its tiles)
        if (stream_cbs[c] * stream_rows[ri] > 16 or stream_cbs[c] * a.bits > stream_code_words or (stream_cbs[c] < 2 and (pair_div == 2 or relu2))) continue;
        if (pair_div == 2 and stream_cbs[c] > 2 and stream_cbs[c] * a.bits > stream_pair_code_words) continue;
        if (a.route.items != 0 and pair_div == 1 and !relu2 and stream_cbs[c] > stream_routed_cb_top[ri]) continue;
        // one row of an unrouted plain product under stream_one_col_n columns: one column a lane; wider keeps the rule
        if (ri == 0 and a.route.items == 0 and pair_div == 1 and !relu2 and total_n < stream_one_col_n and stream_cbs[c] > 1) continue;
        pick = c;
        const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[c], pair_div);
        if (@as(u64, cdiv(total_n, per_block)) * stream_waves * @as(u64, @intCast(items)) * cdiv(a.m, stream_rows[ri]) >= streams_wanted) break;
    }
    for (stream_cbs, 0..) |cb, i| {
        if (cb != k.stream_cb_force or cb * stream_rows[ri] > 16 or cb * a.bits > stream_code_words or (cb < 2 and (pair_div == 2 or relu2))) continue;
        pick = i;
    }
    if (a.route.items == 0 and pair_div == 1 and !relu2) {
        if (heldPick(a.bits, a.m, total_n, stream_cbs[pick])) |hp| {
            if (k.held[hp.r][hp.c]) |f| return heldGo(r, f, a, sides_in, total_n, items, s, lpc_log2, hp.rows, stream_cbs[hp.c]);
        }
    }
    const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[pick], pair_div);
    var blocks: c_uint = cdiv(total_n, per_block);
    if (sides.count > 0) {
        blocks = 0;
        for (0..@intCast(sides.count)) |i| {
            sides.first[i] = @intCast(blocks);
            blocks += cdiv(sides.n[i], per_block);
        }
        sides.first[@intCast(sides.count)] = @intCast(blocks);
    }
    // rows past the tile's: row blocks on x, column blocks on y (at most 65,535), so a column block reads codes once
    const row_blocks = cdiv(a.m, stream_rows[ri]);
    sides.row_blocks = if (row_blocks > 1 and blocks <= 65535) @intCast(row_blocks) else 0;
    // 5 to 8 rows of a routed 4-bit plan on the 4-row tile: the work list, no block for a dead item or a dead row block
    const list_cb: c_int = if (pair_div == 2) 2 else 4;
    if (a.route.items != 0 and !relu2 and sides.count == 0 and a.m > 4 and a.bits == 4 and ri == 2 and
        stream_cbs[pick] == list_cb and ((a.words | @as(u64, @intCast(@divExact(a.k * a.bits, 8)))) & 15) == 0)
    {
        var la: hl.Args = .{};
        try la.add(a);
        try la.add(sides);
        try la.add(@as(c_int, lpc_log2));
        try la.add(@as(c_int, switch (a.group) {
            32 => 0,
            64 => 1,
            else => 2,
        }));
        try la.add(items);
        const lf = k.stream_list[if (pair_div == 2) 0 else 1];
        return go(r, lf, .{ .x = blocks, .y = 1, .z = @intCast(@min(items, stream_list_blocks)) }, .{ .x = 32 * stream_waves }, s, &la);
    }
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(sides);
    try args.add(@as(c_int, lpc_log2));
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    const table = if (sides.pair_cols > 0) &k.pairs else if (relu2) &k.relu2 else &k.stream;
    const f = table[b][ri][pick].?;
    const grid: hl.Dim3 = if (sides.row_blocks > 1)
        .{ .x = row_blocks, .y = blocks, .z = @intCast(items) }
    else
        .{ .x = blocks, .y = row_blocks, .z = @intCast(items) };
    try go(r, f, grid, .{ .x = 32 * stream_waves }, s, &args);
}

/// A plain product on the held-codes tile: `rows` rows a block (every row of the launch in one row block at 5-8).
fn heldGo(r: *const runtime.Runtime, f: Function, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream, lpc_log2: u5, rows: c_int, cb: c_int) Error!void {
    const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * cb;
    var sides = sides_in;
    var blocks: c_uint = cdiv(total_n, per_block);
    if (sides.count > 0) {
        blocks = 0;
        for (0..@intCast(sides.count)) |i| {
            sides.first[i] = @intCast(blocks);
            blocks += cdiv(sides.n[i], per_block);
        }
        sides.first[@intCast(sides.count)] = @intCast(blocks);
    }
    sides.row_blocks = 0;
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(sides);
    try args.add(@as(c_int, lpc_log2));
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    try go(r, f, .{ .x = blocks, .y = cdiv(a.m, rows), .z = @intCast(items) }, .{ .x = 32 * stream_waves }, s, &args);
}
