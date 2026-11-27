//! Production a/b prompt launches through a recording driver, with no CUDA or checkpoint opened.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const fixture = @import("qwen35_verify_host.zig");
const t = std.testing;
const Trace = struct {
    var ids: [1024]usize = undefined;
    var count: usize = 0;
    var parts: cuda.qlinear.PrefillParts = undefined;
    var dims: [4]c_int = undefined;
    var input: u64 = 0;
    var grid: u32 = 0;
    var threads: u32 = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        ids[count] = id;
        count += 1;
        if (id == 0x6153) {
            parts = @as(*const cuda.qlinear.PrefillParts, @ptrCast(@alignCast(args.?[1].?))).*;
            input = @as(*const u64, @ptrCast(@alignCast(args.?[0].?))).*;
            for (&dims, 0..) |*v, i| v.* = @as(*const c_int, @ptrCast(@alignCast(args.?[2 + i].?))).*;
            grid = gx;
            threads = bx;
        }
        return 0;
    }
};
fn function(d: *cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn weight(n: usize, i: usize) q.engine.Proj {
    return .{ .a4 = .{ .w = 10 + i, .s = 20 + i, .b = 30 + i, .n = n, .k = 5120, .npad = std.mem.alignForward(usize, n, 128) } };
}
test "prompt pair: actual adapter leaves qkv and z separate and binds one a/b launch at both large row counts" {
    var d: cuda.Driver = undefined;
    var e = fixture.engine(&d, true);
    d.api.cuLaunchKernel = Trace.launch;
    e.affine.prefill_fn = function(&d, 0x6151);
    e.affine.prefill22_fn = function(&d, 0x6152);
    e.affine.prefill_many_fn = function(&d, 0x6153);
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 1, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 4096, .linear = @splat(false) };
    const ls = [_]q.engine.Proj{ weight(10240, 0), weight(6144, 1), weight(48, 2), weight(48, 3) };
    const outs = [_]u64{ 0x20000, 0x30000, 0x40000, 0x50000 };
    for ([_]usize{ 2048, 4096 }) |rows| {
        Trace.count = 0;
        try e.projectMany(&ls, 0x10000, 0, &outs, rows, true, 0, false, true);
        try t.expectEqualSlices(usize, &.{ 0x6152, 0x6152, 0x6153 }, Trace.ids[0..Trace.count]);
        try t.expectEqual(@as(c_int, 2), Trace.parts.count);
        try t.expectEqual(@as(u64, 0x10000), Trace.input);
        try t.expectEqual([4]c_int{ @intCast(rows), 5120, 5120, 9 }, Trace.dims);
        try t.expectEqual(@as(u32, @intCast(rows / 64)), Trace.grid);
        try t.expectEqual(@as(u32, 256), Trace.threads);
        for (Trace.parts.p[0..2], 0..) |part, i| {
            try t.expectEqual(@as(c_int, 48), part.n);
            try t.expectEqual(@as(c_int, 128), part.npad);
            try t.expectEqual(@as(c_int, @intCast(i * rows / 128)), part.first);
            try t.expectEqual(outs[2 + i], part.out);
            try t.expectEqual(ls[2 + i].a4.w, part.w);
            try t.expectEqual(ls[2 + i].a4.s, part.scales);
            try t.expectEqual(ls[2 + i].a4.b, part.biases);
        }
    }
    for ([_]usize{ 257, 2047, 2048 }) |rows| for ([_]bool{ false, true }) |allow| for ([_]bool{ false, true }) |pdl| {
        if (rows >= 2048 and allow and pdl) continue;
        e.affine.pdl = pdl;
        Trace.count = 0;
        try e.projectMany(&ls, 0x10000, 0, &outs, rows, true, 0, false, allow);
        try t.expectEqual(@as(usize, 4), Trace.count);
        for (Trace.ids[0..Trace.count]) |id| try t.expect(id != 0x6153);
    };
}

test "prompt pair: down prefill22 opt-in selects only the exact dense27 shape" {
    var d: cuda.Driver = undefined;
    var e = fixture.engine(&d, true);
    d.api.cuLaunchKernel = Trace.launch;
    e.affine.prefill_fn = function(&d, 0x6151);
    e.affine.prefill22_fn = function(&d, 0x6152);
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 1, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 4096, .linear = @splat(false) };
    const dense = e.g;
    const Case = struct { n: usize, k: usize, rows: usize, on: bool = true, pdl: bool = true, big: bool = true, id: usize };
    const cases = [_]Case{
        .{ .n = 5120, .k = 17408, .rows = 2048, .on = false, .id = 0x6151 },
        .{ .n = 5120, .k = 17408, .rows = 2048, .id = 0x6152 },
        .{ .n = 5120, .k = 17408, .rows = 4096, .id = 0x6152 },
        .{ .n = 5120, .k = 17408, .rows = 2047, .id = 0x6151 },
        .{ .n = 6144, .k = 17408, .rows = 2048, .id = 0x6151 },
        .{ .n = 5120, .k = 8192, .rows = 2048, .id = 0x6151 },
        .{ .n = 5120, .k = 17408, .rows = 2048, .pdl = false, .id = 0x6151 },
        .{ .n = 5120, .k = 17408, .rows = 2048, .big = false, .id = 0x6151 },
        .{ .n = 17408, .k = 5120, .rows = 2048, .on = false, .id = 0x6152 },
        .{ .n = 17408, .k = 5120, .rows = 2048, .id = 0x6152 },
    };
    for (cases) |c| {
        e.g = dense;
        if (!c.big) e.g.layers = 32;
        e.affine.pdl = c.pdl;
        e.down_prefill22 = c.on;
        Trace.count = 0;
        const p: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = c.n, .k = c.k, .npad = std.mem.alignForward(usize, c.n, 128) } };
        try e.projAgain(p, 0x10000, 0x20000, c.rows, true);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(c.id, Trace.ids[0]);
    }
}
