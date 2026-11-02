//! Router symbols and grid order at the actual CUDA ABI, without executing a kernel.
const std = @import("std");
const cuda = @import("root.zig");
const Mock = struct {
    var id: usize = 0;
    var grid: [3]u32 = undefined;
    var args: [8]u64 = undefined;
    var bindings: usize = 0;
    fn bind(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const names = [_][]const u8{ "tf_moe_router", "tf_moe_router_expert_first", "tf_moe_topk", "tf_moe_combine" };
        for (names, 1..) |n, i| if (std.mem.eql(u8, n, std.mem.span(name))) {
            out.* = @ptrFromInt(i);
            bindings += 1;
            return 0;
        };
        return 1;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, _: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        id = @intFromPtr(f);
        grid = .{ gx, gy, gz };
        if (bx != 256 or by != 1 or bz != 1 or sm != 0) return 1;
        for (0..8) |i| args[i] = if (i == 0 or i == 2 or i == 3 or i == 7) @as(*align(1) const u64, @ptrCast(p.?[i].?)).* else @as(*align(1) const u32, @ptrCast(p.?[i].?)).*;
        return 0;
    }
};
test "MoE router port: off avoids the new symbol and restores the row-first grid" {
    try std.testing.expect(cuda.moe.Fns.expertFirst(null));
    try std.testing.expect(cuda.moe.Fns.expertFirst("1"));
    try std.testing.expect(!cuda.moe.Fns.expertFirst("0"));
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Mock.bind;
    d.api.cuLaunchKernel = Mock.launch;
    const mod: cuda.Module = .{ .d = &d, .handle = null };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |on| {
        Mock.bindings = 0;
        const f = try cuda.moe.Fns.resolveMode(mod, on);
        try std.testing.expectEqual(@as(usize, if (on) 4 else 3), Mock.bindings);
        for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 128, 512, 1024, 2048 }) |rows| for ([_]usize{ 1, 15, 16, 17, 255, 256, 257 }) |experts| for ([_]usize{ 128, 2048, 2055 }) |dims| for ([_]u64{ 0, 0xbeef }) |params| {
            try f.route(s, 0x1000, dims + 16, rows, 0x2000, 0x3000, dims, experts, params);
            try std.testing.expectEqual(@as(usize, if (on) 2 else 1), Mock.id);
            const rt: u32 = @intCast((rows + 15) / 16);
            const et: u32 = @intCast((experts + 15) / 16);
            try std.testing.expectEqual([3]u32{ if (on) et else rt, if (on) rt else et, 1 }, Mock.grid);
            try std.testing.expectEqualSlices(u64, &.{ 0x1000, dims + 16, 0x2000, 0x3000, rows, dims, experts, params }, &Mock.args);
        };
    }
}

test "MoE router port: expert-first coordinates cover each row and expert tile once" {
    for ([_][2]usize{ .{ 17, 257 }, .{ 33, 17 }, .{ 2048, 257 } }) |sh| {
        const rt = (sh[0] + 15) / 16;
        const et = (sh[1] + 15) / 16;
        const seen = try std.testing.allocator.alloc(bool, rt * et);
        defer std.testing.allocator.free(seen);
        @memset(seen, false);
        for (0..rt * et) |block| {
            const row = block / et;
            const expert = block % et;
            const at = row * et + expert;
            try std.testing.expect(!seen[at]);
            seen[at] = true;
        }
        for (seen) |v| try std.testing.expect(v);
    }
}
