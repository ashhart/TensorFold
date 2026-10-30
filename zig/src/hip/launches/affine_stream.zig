//! The stream tile's launch: the column rule (tiles/stream.hpp).

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
        if (stream_cbs[c] * stream_rows[ri] > 16 or stream_cbs[c] * a.bits > stream_code_words) continue;
        pick = c;
        const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[c], pair_div);
        if (@as(u64, cdiv(total_n, per_block)) * stream_waves * @as(u64, @intCast(items)) * cdiv(a.m, stream_rows[ri]) >= streams_wanted) break;
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
    try go(r, f, .{ .x = blocks, .y = cdiv(a.m, stream_rows[ri]), .z = @intCast(items) }, .{ .x = 32 * stream_waves }, s, &args);
}
