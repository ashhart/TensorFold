//! Real Nemotron host reads through the generic checked loader; CUDA transfers and completion are injected.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const weights = @import("cuda_weights.zig");
const living = @import("living");
const a = std.testing.allocator;
const F = struct {
    var download: usize = 0;
    var fail_download: usize = 0;
    var fail_wait = false;
    var pending = false;
    var drains: usize = 0;
    var writes: usize = 0;
    fn read(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        download += 1;
        if (download == fail_download) return cuda.abi.error_not_ready;
        pending = true;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        drains += 1;
        pending = false;
        return 0;
    }
    fn write(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        writes += 1;
        return 0;
    }
};
pub fn run(comptime Host: type) !void {
    const L = living.learner.Learner(Host);
    const neurons = @FieldType(L, "pockets").neurons;
    const d = 4;
    var up: [neurons * d]f32 = @splat(0);
    var down: [neurons * d]f32 = @splat(0.125);
    up[d] = 1.25;
    const original_up = up;
    const original_down = down;
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoHAsync_v2 = F.read;
    driver.api.cuMemcpyHtoDAsync_v2 = F.write;
    driver.api.cuStreamSynchronize = F.wait;
    driver.api.cuCtxSynchronize = F.context;
    var e: engine.Engine = undefined;
    e.c.hidden = d;
    e.c.layers = 3;
    e.c.shared_width = 64;
    e.c.kinds[0..3].* = .{ .moe, .moe, .moe };
    e.k.d = &driver;
    e.b.lane_part = 0;
    e.stream = .{ .d = &driver, .handle = null };
    var blocks: [3]weights.Block = undefined;
    for (&blocks) |*b| b.* = .{ .kind = .moe, .norm = 0, .moe = .{ .router = 0, .bias = 0, .experts = undefined, .pocket = .{ .up = @intFromPtr(&up), .down = @intFromPtr(&down), .n = neurons, .used = 2 } } };
    e.w.blocks = &blocks;
    var saved: ?@import("cuda_sites.zig").Sites = null;
    const model: Host.Ctx = .{ .e = &e, .dir = "", .living = &saved };
    const hooks: living.whole.Hooks(Host) = .{ .b = model, .gpa = a, .io = std.testing.io };
    var trainer: Host.Trainer = undefined;
    for (0..4) |failure| {
        F.download = 0;
        F.fail_download = if (failure < 2) failure + 1 else 0;
        F.fail_wait = failure == 2;
        F.pending = false;
        F.drains = 0;
        F.writes = 0;
        blocks[0].moe.pocket.?.n = if (failure == 3) neurons - 1 else neurons;
        var l = L.init(a, std.testing.io, model);
        defer l.deinit();
        const result = l.pockets.load(&trainer, hooks);
        try std.testing.expectError(if (failure == 3) error.PocketShape else error.NotReady, result);
        try std.testing.expectEqual(@as(usize, 0), l.pockets.moe.items.len);
        try std.testing.expectEqual(@as(usize, 0), l.pockets.used.items.len);
        try std.testing.expectEqual(@as(usize, 0), l.pockets.hidden);
        try std.testing.expect(!F.pending);
        try std.testing.expectEqual(@as(usize, 0), F.writes);
        try std.testing.expectEqual(@as(usize, 2), blocks[0].moe.pocket.?.used);
        if (failure == 2) try std.testing.expectEqual(@as(usize, 1), F.drains);
        try std.testing.expectEqualSlices(f32, &original_up, &up);
        try std.testing.expectEqualSlices(f32, &original_down, &down);
        F.fail_download = 0;
        F.fail_wait = false;
        blocks[0].moe.pocket.?.n = neurons;
        try l.pockets.load(&trainer, hooks);
        try std.testing.expectEqual(@as(usize, 2), l.pockets.moe.items.len);
        for (l.pockets.used.items) |used| try std.testing.expectEqual(@as(usize, 2), used);
        try std.testing.expectEqualSlices(f32, &up, l.pockets.up.items[0 .. neurons * d]);
        try std.testing.expectEqualSlices(f32, &down, l.pockets.down.items[0 .. neurons * d]);
        try std.testing.expectEqual(@as(usize, 0), F.writes);
    }
    blocks[0].moe.pocket = null;
    try std.testing.expectEqual(@as(usize, 0), try Host.pocketRead(model, 0, &up, &down));
    blocks[0].kind = .attention;
    try std.testing.expectError(error.NotAnMoeLayer, Host.pocketRead(model, 0, &up, &down));
}
