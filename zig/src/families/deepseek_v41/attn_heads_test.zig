//! CUDA decode attention at every world: each rank's own heads, the scratch partials over whole 16-head tiles.

const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const calls = @import("block_calls.zig");
const ds = @import("dspark_emit.zig");
const Config = @import("config.zig").Config;
const widths = @import("prefill_heads_test.zig").widths;
const attn = @import("dsv41_kernels").ops.attn;

/// Each attention launch in `cs` holds `H` heads in q, sink and out, its partials `n * 5 * HS` rows of 512 floats.
fn expectLaunches(cs: []const calls.Call, n: i64, H: i64, HS: i64) !usize {
    var found: usize = 0;
    for (cs) |c| {
        if (!std.mem.eql(u8, c.name, "tf_dsv41_attn_cuda_v1.attn")) continue;
        found += 1;
        try t.expectEqualSlices(i64, &.{ n, H, 512 }, c.args[0].arg.t.shape);
        try t.expectEqualSlices(i64, &.{H}, c.args[12].arg.t.shape);
        try t.expectEqualSlices(i64, &.{n * 5 * HS * 512}, c.args[14].arg.t.shape);
        try t.expectEqualSlices(i64, &.{n * 5 * HS}, c.args[15].arg.t.shape);
        try t.expectEqualSlices(i64, &.{n * 5 * HS}, c.args[16].arg.t.shape);
        try t.expectEqualSlices(i64, &.{ n, H, 512 }, c.args[17].arg.t.shape);
    }
    return found;
}

const Share = struct { world: u32, rank: u32, heads: i64, scratch: i64 };
const shares = [_]Share{
    .{ .world = 1, .rank = 0, .heads = 64, .scratch = 64 },
    .{ .world = 2, .rank = 1, .heads = 32, .scratch = 32 },
    .{ .world = 3, .rank = 0, .heads = 24, .scratch = 32 },
    .{ .world = 3, .rank = 1, .heads = 24, .scratch = 32 },
    .{ .world = 3, .rank = 2, .heads = 16, .scratch = 16 },
    .{ .world = 4, .rank = 3, .heads = 16, .scratch = 16 },
};

test "decode attention: each rank's launch takes its own heads, the scratch whole 16-head tiles" {
    const cfg: Config = .{};
    for (shares) |s| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const o: block.Options = .{ .world = s.world, .rank = s.rank };
        // an SWA layer and a compressed one
        const cs = try block.emit(a, &cfg, &w, o, &.{ 0, 2 }, 17, 2048, true);
        try t.expectEqual(@as(usize, 2), try expectLaunches(cs, 17, s.heads, s.scratch));
    }
}

test "DSpark attention: each rank's launch takes its own heads, the scratch whole 16-head tiles" {
    const cfg: Config = .{};
    for (shares) |s| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const o: block.Options = .{ .world = s.world, .rank = s.rank };
        const n: i64 = @intCast(cfg.dspark_block);
        try t.expect(try expectLaunches(try ds.emitPass(a, &cfg, &w, o, n), n, s.heads, s.scratch) > 0);
    }
}

test "attention tiles: a partial last tile only where the heads are not a multiple of 16" {
    for ([_][3]i64{ .{ 16, 1, 16 }, .{ 24, 2, 32 }, .{ 32, 2, 32 }, .{ 40, 3, 48 }, .{ 64, 4, 64 } }) |c| {
        try t.expectEqual(c[1], attn.tiles(c[0]));
        try t.expectEqual(c[2], attn.scratchHeads(c[0]));
    }
    // the instance check no longer refuses a partial tile; the launch picks the part kernels for it
    const a: attn.Args = .{ .R = 1, .H = 24, .ring = 256 };
    try t.expectEqual(@as(usize, 6), try attn.instance(a, 4, true, false));
    try t.expectError(error.Shape, attn.instance(.{ .R = 1, .H = 0, .ring = 256 }, 2, false, false));
}
