const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const t = std.testing;
const Mock = struct {
    var fail = false;
    var copies: usize = 0;
    var zeros: usize = 0;
    fn zero(dst: u64, byte: u8, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        zeros += 1;
        @memset(@as([*]u8, @ptrFromInt(dst))[0..len], byte);
        return 0;
    }
    fn copy(c: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        if (fail) return cuda.abi.error_not_ready;
        for (0..c.height) |row| @memcpy(@as([*]u8, @ptrFromInt(c.dst + row * c.dst_pitch))[0..c.width], @as([*]const u8, @ptrFromInt(c.src + row * c.src_pitch))[0..c.width]);
        return 0;
    }
};

fn call(name: []const u8, args: *[2]calls.Named, src: calls.Tensor, dst: calls.Tensor) calls.Call {
    args.* = .{ .{ .arg = .{ .t = src } }, .{ .arg = .{ .t = dst } } };
    return .{ .triton = false, .glue = true, .name = name, .args = args };
}

test "prefill heads: production copy dispatch pads finite heads and extracts pitched groups without touching guards" {
    var d: cuda.Driver = undefined;
    d.api.cuMemsetD8Async = Mock.zero;
    d.api.cuMemcpy2DAsync_v2 = Mock.copy;
    Mock.fail = false;
    Mock.copies = 0;
    Mock.zeros = 0;
    var r: @import("run.zig").Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer r.deinit();
    var source: [3 * 32]u16 align(16) = undefined;
    for (&source, 0..) |*x, i| x.* = @intCast(i + 1);
    var padded: [3 * 64 + 16]u16 align(16) = @splat(0xA5A5);
    try r.external("source", @intFromPtr(&source));
    try r.external("padded", @intFromPtr(&padded));
    var args: [2]calls.Named = undefined;
    var c = call("glue.pf_heads_q", &args, .{ .role = .{ .buf = "source" }, .dt = .bf16, .shape = &.{ 3, 32 }, .stride = &.{ 32, 1 } }, .{ .role = .{ .buf = "padded" }, .dt = .bf16, .shape = &.{ 3, 64 }, .stride = &.{ 64, 1 }, .offset = 16 });
    try @import("forward_prefill_glue.zig").glueOne(undefined, &r, &c);
    for (0..3) |row| {
        try t.expectEqualSlices(u16, source[row * 32 ..][0..32], padded[8 + row * 64 ..][0..32]);
        try t.expect(std.mem.allEqual(u16, padded[8 + row * 64 + 32 ..][0..32], 0));
    }
    try t.expect(std.mem.allEqual(u16, padded[0..8], 0xA5A5));
    try t.expect(std.mem.allEqual(u16, padded[padded.len - 8 ..], 0xA5A5));
    var sink: [16]f32 align(16) = undefined;
    for (&sink, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) - 8;
    var sink32: [36]f32 align(16) = @splat(99);
    try r.external("sink", @intFromPtr(&sink));
    try r.external("sink32", @intFromPtr(&sink32));
    c = call("glue.pf_heads_sink", &args, .{ .role = .{ .buf = "sink" }, .dt = .f32, .shape = &.{16}, .stride = &.{1} }, .{ .role = .{ .buf = "sink32" }, .dt = .f32, .shape = &.{32}, .stride = &.{1}, .offset = 8 });
    try @import("forward_prefill_glue.zig").glueOne(undefined, &r, &c);
    try t.expectEqualSlices(f32, &sink, sink32[2..18]);
    try t.expect(std.mem.allEqual(f32, sink32[18..34], 0));
    try t.expect(sink32[0] == 99 and sink32[35] == 99);
    var full: [4 * 11 * 16]u16 align(16) = undefined;
    for (&full, 0..) |*x, i| x.* = @intCast(i);
    var real: [2 * 11 * 16]u16 align(16) = @splat(0xA5A5);
    try r.external("full", @intFromPtr(&full));
    try r.external("real", @intFromPtr(&real));
    c = call("glue.pf_heads_out", &args, .{ .role = .{ .buf = "full" }, .dt = .bf16, .shape = &.{ 4, 5, 16 }, .stride = &.{ 11 * 16, 16, 1 }, .offset = 3 * 16 * 2 }, .{ .role = .{ .buf = "real" }, .dt = .bf16, .shape = &.{ 2, 5, 16 }, .stride = &.{ 11 * 16, 16, 1 }, .offset = 3 * 16 * 2 });
    try @import("forward_prefill_glue.zig").glueOne(undefined, &r, &c);
    for (&real, 0..) |value, i| {
        const row = (i / 16) % 11;
        try t.expectEqual(if (row >= 3 and row < 8) full[i] else @as(u16, 0xA5A5), value);
    }
    try t.expectEqual(@as(usize, 3), Mock.copies);
    try t.expectEqual(@as(usize, 2), Mock.zeros);
    Mock.fail = true;
    defer Mock.fail = false;
    try t.expectError(error.NotReady, @import("forward_prefill_glue.zig").glueOne(undefined, &r, &c));
    args[1].arg.t.stride = &.{ 1, 16, 1 };
    try t.expectError(error.BadGlue, @import("forward_prefill_glue.zig").glueOne(undefined, &r, &c));
    try t.expectEqual(@as(usize, 4), Mock.copies);
}
