//! Exercise the fixture's actual poison-and-dispatch path with CPU buffers and no-op CUDA launches.
const std = @import("std");
const cuda = @import("cuda");
const fixture = @import("projection_f32_tests.zig");
const Rig = @import("quant_ref_tests.zig").RigT;

const Mock = struct {
    var destination: []u8 = &.{};
    var scratch: []u8 = &.{};
    var fills: usize = 0;
    var launches: usize = 0;
    var fresh = true;

    fn fill(ptr: cuda.abi.DevicePtr, byte: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const bytes: [*]u8 = @ptrFromInt(ptr);
        @memset(bytes[0..n], byte);
        fills += 1;
        return cuda.abi.success;
    }

    fn observe() cuda.abi.Result {
        for (destination) |byte| fresh = fresh and byte == 0xff;
        for (scratch) |byte| fresh = fresh and byte == 0xff;
        launches += 1;
        return cuda.abi.success;
    }

    fn plain(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return observe();
    }

    fn extended(_: *const cuda.abi.LaunchConfig, _: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return observe();
    }

    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn copy(dst: ?*anyopaque, src: cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
        const out: [*]u8 = @ptrCast(dst.?);
        const in: [*]const u8 = @ptrFromInt(src);
        @memcpy(out[0..n], in[0..n]);
        return cuda.abi.success;
    }
};

test "every F32 and BF16 fixture dispatch rejects a no-op after a valid stale prefix" {
    const a = std.testing.allocator;
    var driver: cuda.Driver = undefined;
    driver.api.cuMemsetD8Async = Mock.fill;
    driver.api.cuLaunchKernel = Mock.plain;
    driver.api.cuLaunchKernelEx = Mock.extended;
    driver.api.cuStreamSynchronize = Mock.sync;
    driver.api.cuMemcpyDtoH_v2 = Mock.copy;
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(1) };
    const f: cuda.Function = .{ .d = &driver, .handle = null };
    var lane: cuda.qmmf.Lane = .{
        .mod = .{ .d = &driver, .handle = null },
        .fns = @splat(@splat(f)),
        .f32_fns = @splat(@splat(f)),
        .reduce = @splat(f),
        .half = @splat(@splat(@splat(f))),
        .half_below = 0,
        .clusters = false,
    };
    const weight: cuda.qmmf.Weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 70, .k = 1024, .npad = 128 };
    const elements = 256 * weight.n;
    const expected = try a.alloc(f64, elements);
    defer a.free(expected);
    const tolerance = try a.alloc(f64, elements);
    defer a.free(tolerance);
    @memset(expected, 0);
    @memset(tolerance, 1);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var rig: Rig = .{
        .gpu = .{ .d = &driver, .ctx = undefined, .gpa = a, .io = std.testing.io },
        .a = arena.allocator(),
        .s = stream,
        .rng = undefined,
    };
    for ([_]cuda.qmmf.Out{ .f32, .bf16 }) |format| {
        const width: usize = if (format == .f32) 4 else 2;
        const destination = try a.alloc(u8, elements * width);
        defer a.free(destination);
        const scratch = try a.alloc(u8, 8 * elements * 4);
        defer a.free(scratch);
        const out: cuda.DeviceBuffer = .{ .d = &driver, .ptr = @intFromPtr(destination.ptr), .len = destination.len };
        const part: cuda.DeviceBuffer = .{ .d = &driver, .ptr = @intFromPtr(scratch.ptr), .len = scratch.len };
        Mock.destination = destination;
        Mock.scratch = scratch;
        for ([_]bool{ true, false }) |clusters| {
            lane.clusters = clusters;
            for ([_]usize{ 256, 1, 7, 17, 33 }) |rows| {
                @memset(destination, 0);
                @memset(scratch, 0);
                const count = rows * weight.n;
                try rig.close("valid stale prefix", out.ptr, format == .f32, expected[0..count], tolerance[0..count]);
                Mock.fills = 0;
                Mock.launches = 0;
                Mock.fresh = true;
                try fixture.dispatch(&lane, stream, 0, 1088, rows, weight, out, part, format);
                try std.testing.expectEqual(@as(usize, 2), Mock.fills);
                try std.testing.expect(Mock.launches > 0);
                try std.testing.expect(Mock.fresh);
                try std.testing.expectError(error.TestFailed, rig.close("no-op projection", out.ptr, format == .f32, expected[0..count], tolerance[0..count]));
            }
        }
    }
}
