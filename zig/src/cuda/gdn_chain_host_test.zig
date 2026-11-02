//! Real chain dispatch and symbol binding recorded at the driver ABI, without a device.
const std = @import("std");
const cuda = @import("root.zig");
const chain = @import("gdn_chain.zig");
const Mock = struct {
    var id: usize = 0;
    var grid: [3]u32 = undefined;
    var block: [3]u32 = undefined;
    var args: [16]u64 = undefined;
    var bindings: usize = 0;
    fn bind(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const old = std.mem.eql(u8, std.mem.span(name), chain.old_symbol);
        if (!old and !std.mem.eql(u8, std.mem.span(name), chain.symbol)) return 1;
        out.* = @ptrFromInt(if (old) @as(usize, 1) else 2);
        bindings += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, _: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        id = @intFromPtr(f);
        grid = .{ gx, gy, gz };
        block = .{ bx, by, bz };
        if (sm != 0) return 1;
        for (0..@as(usize, if (id == 1) 11 else 16)) |i| args[i] = if (i < 8 or (i >= 11 and i < 15)) @as(*align(1) const u64, @ptrCast(p.?[i].?)).* else @as(*align(1) const u32, @ptrCast(p.?[i].?)).*;
        return 0;
    }
};
test "GDN chain port: off binds only the old symbol; on binds the fused kernel" {
    try std.testing.expect(chain.Chain.enabledValue(null));
    try std.testing.expect(chain.Chain.enabledValue("1"));
    try std.testing.expect(!chain.Chain.enabledValue("0"));
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Mock.bind;
    const mod: cuda.Module = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |on| {
        Mock.bindings = 0;
        const f = try chain.Chain.resolveMode(mod, on);
        try std.testing.expectEqual(@as(usize, if (on) 2 else 1), Mock.bindings);
        try std.testing.expectEqual(on, f.fused != null);
    }
}
test "GDN chain port: staged boundaries, family head groups, in-place state and optional sums keep the ABI" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const b: chain.Buffers = .{ .q = 0x1000, .k = 0x2000, .v = 0x3000, .g = 0x4000, .beta = 0x5000, .state = 0x6000, .last = 0x6000, .y = 0x7000, .z = 0x8000, .norm = 0x9000, .mix = 0xa000, .sums = 0xb000 };
    for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 64, 65, 512, 2048 }) |rows| for ([_][2]usize{ .{ 16, 32 }, .{ 16, 48 }, .{ 16, 64 }, .{ 16, 128 } }) |heads| for ([_]bool{ false, true }) |on| {
        const f: chain.Chain = .{ .old = .{ .d = &d, .handle = @ptrFromInt(1) }, .fused = if (on) .{ .d = &d, .handle = @ptrFromInt(2) } else null };
        try std.testing.expectEqual(on, try f.run(s, b, rows, heads[0], heads[1], 1e-6));
        try std.testing.expectEqual(@as(usize, if (on) 2 else 1), Mock.id);
        try std.testing.expectEqual([3]u32{ @intCast(heads[1]), if (on) 1 else 2, 1 }, Mock.grid);
        try std.testing.expectEqual([3]u32{ if (on) 256 else 128, 1, 1 }, Mock.block);
        try std.testing.expectEqualSlices(u64, &.{ b.q, b.k, b.v, b.g, b.beta, b.state, b.last, b.y }, Mock.args[0..8]);
        try std.testing.expectEqualSlices(u64, &.{ rows, heads[0], heads[1] }, Mock.args[8..11]);
        if (on) {
            try std.testing.expectEqualSlices(u64, &.{ b.z, b.norm, b.mix, b.sums }, Mock.args[11..15]);
            try std.testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 1e-6)))), Mock.args[15]);
            var no_sums = b;
            no_sums.sums = 0;
            no_sums.last = 0xc000;
            try std.testing.expect(try f.run(s, no_sums, rows, heads[0], heads[1], 1e-5));
            try std.testing.expectEqual(@as(u64, 0), Mock.args[14]);
            try std.testing.expectEqual(no_sums.last, Mock.args[6]);
        }
    };
    const f: chain.Chain = .{ .old = .{ .d = &d, .handle = @ptrFromInt(1) } };
    for ([_][3]usize{ .{ 0, 16, 32 }, .{ 16, 0, 32 }, .{ 16, 16, 0 }, .{ 16, 16, 33 } }) |sh| try std.testing.expectError(error.BadGdnChainShape, f.run(s, b, sh[0], sh[1], sh[2], 1e-6));
}
