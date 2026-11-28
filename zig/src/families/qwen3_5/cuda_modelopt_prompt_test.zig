//! MLX dispatch equals main at the actual driver ABI; ModelOpt alone takes the additional kernels.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig").Engine;
const geometry = @import("cuda_geometry.zig");
const Seg = @import("cuda_cache.zig").Seg;
const helper = @import("cuda_modelopt_prompt.zig");
const chain_rows = 64;
const Snapshot = struct { id: usize = 0, grid: [3]u32 = @splat(0), block: [3]u32 = @splat(0), shared: u32 = 0, args: [16]u64 = @splat(0) };
const Mock = struct {
    var now: Snapshot = .{};
    fn function(d: *const cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        now = .{ .id = id, .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .shared = sm };
        const n: usize = if (id == 1) 11 else if (id == 2) 16 else 8;
        for (0..n) |i| {
            const wide = if (id <= 2) i < 8 or (i >= 11 and i < 15) else i == 0 or i == 2 or i == 3 or i == 7;
            now.args[i] = if (wide) @as(*align(1) const u64, @ptrCast(args.?[i].?)).* else @as(*align(1) const u32, @ptrCast(args.?[i].?)).*;
        }
        return cuda.abi.success;
    }
};
fn baselineDelta(e: *E, layer: usize, rows: usize) !void {
    const g = e.g;
    const sg = std.mem.bytesToValue(Seg, e.staged.bytes[E.segs_at..][0..@sizeOf(Seg)]);
    const state = sg.rec + layer * g.v_heads * geometry.dk * geometry.dk * 4;
    var a: cuda.Args = .{};
    for ([_]u64{ e.q, e.k, e.v, e.g_, e.beta, state, state, e.y }) |v| a.add(v);
    for ([_]usize{ rows, g.k_heads, g.v_heads }) |v| a.add(@as(c_int, @intCast(v)));
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(g.v_heads), .y = @intCast(geometry.dk / chain_rows) }, .block = .{ .x = 2 * chain_rows } };
    try cuda.launch.launch(e.prompt_attn.chain, cfg, e.stream, &a);
}

test "MLX prompt isolation: recurrence launch and norm fallthrough equal the frozen main sequence" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var bytes: [E.segs_at + @sizeOf(Seg)]u8 align(16) = @splat(0);
    var sg = std.mem.zeroes(Seg);
    sg.rec = 0x100000;
    @memcpy(bytes[E.segs_at..][0..@sizeOf(Seg)], std.mem.asBytes(&sg));
    var e: E = undefined;
    e.staged = .{ .d = &d, .bytes = &bytes };
    e.stream = .{ .d = &d, .handle = null };
    e.prompt_attn.chain = Mock.function(&d, 1);
    e.q = 0x1000;
    e.k = 0x2000;
    e.v = 0x3000;
    e.g_ = 0x4000;
    e.beta = 0x5000;
    e.y = 0x6000;
    e.z = 0x7000;
    e.mix = 0x8000;
    e.g.k_heads = 16;
    e.g.v_heads = 48;
    e.g.eps = 1e-6;
    e.mo = null;
    for ([_]usize{ 1, 16, 64, 65, 1023, 1024, 2048 }) |rows| for ([_]usize{ 0, 3, 17 }) |layer| {
        try baselineDelta(&e, layer, rows);
        const before = Mock.now;
        try std.testing.expect(!try helper.delta(&e, layer, rows, 0x9000, 0xa000));
        try std.testing.expectEqualDeep(before, Mock.now);
        var mo: @import("cuda_modelopt.zig").Kernels = undefined;
        mo.glue.chain = .{ .old = Mock.function(&d, 1), .fused = Mock.function(&d, 2) };
        e.mo = mo;
        try std.testing.expect(try helper.delta(&e, layer, rows, 0x9000, 0xa000));
        try std.testing.expectEqual(@as(usize, 2), Mock.now.id);
        try std.testing.expectEqual([3]u32{ 48, 1, 1 }, Mock.now.grid);
        mo.glue.chain.fused = null;
        e.mo = mo;
        try std.testing.expect(!try helper.delta(&e, layer, rows, 0x9000, 0xa000));
        try std.testing.expectEqualDeep(before, Mock.now);
        e.mo = null;
    };
}
test "MLX prompt isolation: row router and large-row tiles retain main kernels grids and arguments" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var e: E = undefined;
    e.stream = .{ .d = &d, .handle = null };
    e.moe = .{ .router_fn = Mock.function(&d, 3), .router_tiles_fn = Mock.function(&d, 5), .topk_fn = Mock.function(&d, 6), .combine_fn = Mock.function(&d, 7) };
    for ([_]usize{ 1, 16, 256, 257, 512, 1024, 2048 }) |rows| for ([_]usize{ 17, 257 }) |experts| for ([_]usize{ 2048, 2055 }) |dims| {
        e.mo = null;
        e.g.hidden = dims;
        try e.moe.route(e.stream, 0x1000, dims, rows, 0x2000, 0x3000, dims, experts, 0xbeef);
        const before = Mock.now;
        try helper.route(&e, 0x1000, rows, 0x2000, 0x3000, experts, 0xbeef);
        try std.testing.expectEqualDeep(before, Mock.now);
        var mo: @import("cuda_modelopt.zig").Kernels = undefined;
        mo.glue.router = .{ .base = e.moe, .router_expert = Mock.function(&d, 4) };
        e.mo = mo;
        try helper.route(&e, 0x1000, rows, 0x2000, 0x3000, experts, 0xbeef);
        try std.testing.expectEqual(@as(usize, 4), Mock.now.id);
        try std.testing.expectEqual([3]u32{ @intCast((experts + 15) / 16), @intCast((rows + 15) / 16), 1 }, Mock.now.grid);
        mo.glue.router.?.router_expert = null;
        e.mo = mo;
        try helper.route(&e, 0x1000, rows, 0x2000, 0x3000, experts, 0xbeef);
        try std.testing.expectEqualDeep(before, Mock.now);
    };
}
