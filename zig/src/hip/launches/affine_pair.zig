//! Two relu^2 stream-tile products of one x in one launch: a routed plan's and a plain one's (tiles/stream.hpp).

const std = @import("std");
const affine = @import("affine.zig");
const hl = @import("../launch.zig");
const runtime = @import("../runtime.zig");
const abi = @import("../abi.zig");
const util = @import("util.zig");
const stream_pick = @import("stream_pick.zig");

const Kernels = affine.Kernels;
const Arg = affine.Arg;

fn cdiv(n: c_int, by: c_int) c_uint {
    return @intCast(@divTrunc(n + by - 1, by));
}

/// Plan `a` (`items` on z) and plain `b` as relu^2 into their `out16` in one launch; false unless both take the stream.
pub fn relu2PairGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, b: Arg, items: c_int, s: abi.Stream) !bool {
    if (a.out16 == 0 or b.out16 == 0 or a.out != 0 or b.out != 0) return false;
    if (a.route.items == 0 or b.route.items != 0) return false;
    if (a.k != b.k or a.bits != b.bits or a.group != b.group or a.m != b.m or a.fp16 != b.fp16) return false;
    if (a.scale.kind != b.scale.kind or a.bias.kind != b.bias.kind or items < 1) return false;
    var ra = a;
    ra.out16 = 0;
    const ea = k.reg.select(k.env(), .mlx, .routed, .decode, Kernels.shapeOf(ra, items)) orelse return false;
    var rb = b;
    rb.out16 = 0;
    var sb = Kernels.shapeOf(rb, 1);
    sb.round = true;
    const eb = k.choose(.project, .decode, sb) orelse return false;
    if (ea.family != .stream or eb.family != .stream) return false;
    const bi = std.mem.indexOfScalar(c_int, &affine.bit_widths, a.bits) orelse return false;
    var lpc_log2: u5 = 0;
    while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
    const ri = stream_pick.rowTile(a.m);
    // the routed product's own column count a lane, as its launch alone takes it (the bits do not depend on it)
    const tile = stream_pick.pick(k.caps.generation, a.m, lpc_log2, a.bits, a.n, items, false);
    const per = tile.per_block;
    const split = cdiv(a.n, per);
    const f = k.relu2_pair[bi][ri][tile.at] orelse return false;
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(b);
    try args.add(@as(c_int, @intCast(split)));
    try args.add(@as(c_int, lpc_log2));
    try args.add(@as(c_int, switch (a.group) {
        32 => 0,
        64 => 1,
        else => 2,
    }));
    const y = cdiv(a.m, affine.stream_rows[ri]);
    const grid: hl.Dim3 = .{ .x = split + cdiv(b.n, per), .y = y, .z = @intCast(items) };
    if (s == util.counting) return true;
    try hl.launch(f, .{ .grid = grid, .block = .{ .x = 32 * affine.stream_waves } }, .{ .r = r, .handle = s }, &args);
    return true;
}
