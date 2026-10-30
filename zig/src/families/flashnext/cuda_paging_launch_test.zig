//! Page-table forwarding through the actual host launch wrappers, including one-buffer and fused attention.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

const Capture = struct {
    var count: usize = 0;
    var symbols: [6]usize = @splat(0);
    var pages: [6]glue.Pg = @splat(.{});
    fn launch(f: cuda.abi.Function, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        symbols[count] = id;
        if (id % 2 == 0) {
            const at: usize = switch (id) {
                2 => 18,
                4 => 10,
                6 => 8,
                8, 10 => 18,
                12 => 15,
                else => unreachable,
            };
            pages[count] = @as(*const glue.Pg, @ptrCast(@alignCast(args.?[at].?))).*;
        }
        count += 1;
        return cuda.abi.success;
    }
};

fn fun(d: *const cuda.Driver, n: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(n) };
}

test "paged launches: prep, pool, tiled QSA, ahead, one-buffer and fused select their real paged symbols" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Capture.launch;
    var g: glue.Glue = undefined;
    g.f.prep = fun(&d, 1);
    g.f.prep_paged = fun(&d, 2);
    g.f.pool = fun(&d, 3);
    g.f.pool_paged = fun(&d, 4);
    g.f.scores_tile = fun(&d, 5);
    g.f.scores_paged = fun(&d, 6);
    g.f.chunks_ahead = fun(&d, 7);
    g.f.chunks_ahead_paged = fun(&d, 8);
    g.f.chunks_one = fun(&d, 9);
    g.f.chunks_one_paged = fun(&d, 10);
    g.f.fused = fun(&d, 11);
    g.f.fused_paged = fun(&d, 12);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]glue.Pg{ .{}, .{ .pt = 0x3000, .ps = 12 * 48 * 1184, .ptok = 48 } }) |pg| {
        Capture.count = 0;
        Capture.symbols = @splat(0);
        Capture.pages = @splat(.{});
        try g.attnPrep(stream, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1e-6, 1, pg);
        try g.qsaPool(stream, 1, 2, 3, 4, 5, 1e-6, 1, pg);
        try g.qsaScores(stream, 1, 2, 3, 4, 1024, 1, 600, pg);
        try g.attnChunks(stream, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 2, 5, true, pg);
        try g.attnChunks(stream, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 2, 5, true, pg);
        try g.attnFused(stream, 1, 2, 3, 4, 5, 6, 7, 8, 300, 2, pg);
        try std.testing.expectEqual(@as(usize, 6), Capture.count);
        for (Capture.symbols, Capture.pages, 0..) |id, actual, i| {
            try std.testing.expectEqual(2 * i + 1 + @intFromBool(pg.pt != 0), id);
            try std.testing.expectEqual(pg, actual);
        }
    }
}
