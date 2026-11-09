//! Real prompt-product download through an injected CUDA driver, with poisoned inactive neurons.
const std = @import("std");
const cuda = @import("cuda");
const usage = @import("lanes").pocket_usage;
const Read = @import("cuda_usage.zig").Read;
const kern = @import("cuda_kernels.zig");
const t = std.testing;
const F = struct {
    var reads: usize = 0;
    var waits: usize = 0;
    var fail_read = false;
    fn read(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        reads += 1;
        if (fail_read) return cuda.abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        waits += 1;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        @panic("usage must never launch a kernel or forward");
    }
};
test "pocket usage: actual CUDA read only copies existing products, excludes inactive poison and drains a failed copy" {
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoHAsync_v2 = F.read;
    driver.api.cuStreamSynchronize = F.wait;
    driver.api.cuLaunchKernel = F.launch;
    var kernels: kern.Kernels = undefined;
    kernels.d = &driver;
    const ops: kern.Ops = .{ .k = &kernels, .s = .{ .d = &driver, .handle = null } };
    var products: [3 * 64]f32 = @splat(std.math.nan(f32));
    products[0] = 0;
    products[64] = usage.activation_floor;
    products[128] = 1;
    const original = products;
    var memory: [3 * 64 * 4]u8 align(16) = undefined;
    var r: Read = .{ .trace = try usage.Capture.init(t.allocator, 4, 3), .stage = .{ .d = &driver, .bytes = &memory } };
    defer r.trace.deinit();
    try r.trace.start(0, 3);
    F.reads = 0;
    F.waits = 0;
    F.fail_read = false;
    try r.read(ops, 2, @intFromPtr(&products), 1);
    try t.expectEqual(@as(usize, 1), F.reads);
    try t.expectEqual(@as(usize, 1), F.waits);
    try t.expect(!(try r.trace.through(2)).has(2, 0));
    const all = try r.trace.through(3);
    try t.expect(all.complete and all.has(2, 0) and !all.has(2, 1));
    try t.expectEqualSlices(u8, std.mem.asBytes(&original), std.mem.asBytes(&products));
    F.fail_read = true;
    defer F.fail_read = false;
    try t.expectError(error.NotReady, r.read(ops, 2, @intFromPtr(&products), 1));
    try t.expectEqual(@as(usize, 2), F.waits);
    F.fail_read = false;
    // The next chunk includes its restored prefix, but never the rows after an interior mark.
    r.trace.restore(3, all);
    try r.trace.start(3, 1);
    try r.read(ops, 1, @intFromPtr(&products), 1);
    const next = try r.trace.through(4);
    try t.expect(next.complete and next.has(2, 0) and !next.has(1, 0));
    r.trace.restore(3, null);
    try t.expect(!(try r.trace.through(3)).complete);
    r.trace.restore(0, .{});
    try r.trace.start(0, 3);
    // Active poison cannot make an unknown usage count eligible for retirement.
    F.fail_read = false;
    try r.read(ops, 2, @intFromPtr(&products), 2);
    try t.expect(!(try r.trace.through(3)).complete);
}

test "pocket usage: warm end snapshots retain their completed prefix without a live read" {
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.usage = null;
    var s: @import("lanes").Stream = undefined;
    s.prompt_len = 7;
    var mask: usage.Masks = .{};
    mask.bits[3] = 2;
    s.pocket_usage = mask;
    const kept = (try @import("cuda_usage.zig").saved(&e, &s, 7)).?;
    try t.expect(kept.complete and kept.has(3, 1));
    try t.expect((try @import("cuda_usage.zig").saved(&e, &s, 6)) == null);
    try t.expect((try @import("cuda_usage.zig").saved(&e, null, 7)) == null);
}
