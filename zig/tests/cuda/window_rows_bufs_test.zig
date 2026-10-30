//! The actual diagnostic owner under failing host allocation and a fake CUDA ABI. No driver is loaded.
const std = @import("std");
const cuda = @import("cuda");
const Bufs = @import("window_rows_tests.zig").Bufs;
const abi = cuda.abi;

const Fake = struct {
    var allocs: usize = 0;
    var frees: usize = 0;
    var fail_fill = false;
    fn alloc(p: *abi.DevicePtr, _: usize) callconv(.c) abi.Result {
        allocs += 1;
        p.* = 0x1000 + allocs * 0x1000;
        return abi.success;
    }
    fn free(_: abi.DevicePtr) callconv(.c) abi.Result {
        frees += 1;
        return abi.success;
    }
    fn copy(_: abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) abi.Result {
        return abi.success;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
    fn fill(_: abi.DevicePtr, _: u8, _: usize) callconv(.c) abi.Result {
        return if (fail_fill) abi.error_not_ready else abi.success;
    }
    fn fillAsync(p: abi.DevicePtr, value: u8, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        return fill(p, value, len);
    }
    fn driver() cuda.Driver {
        allocs = 0;
        frees = 0;
        fail_fill = false;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuMemsetD8_v2 = fill;
        d.api.cuMemsetD8Async = fillAsync;
        return d;
    }
};

fn buffers(d: *cuda.Driver, a: std.mem.Allocator) Bufs {
    return .{ .gpu = .{ .d = d, .ctx = undefined, .gpa = a, .io = std.testing.io } };
}

test "window diagnostic list OOM precedes either device allocation" {
    var d = Fake.driver();
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var b = buffers(&d, fail.allocator());
    defer b.deinit();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try std.testing.expectError(error.OutOfMemory, b.up(&.{ 1, 2 }));
    try std.testing.expectError(error.OutOfMemory, b.out(s, 16));
    try std.testing.expectEqual(@as(usize, 0), Fake.allocs);
    try std.testing.expectEqual(@as(usize, 0), b.list.items.len);
}

test "window diagnostic failed fill frees its unpublished buffer" {
    var d = Fake.driver();
    var b = buffers(&d, std.testing.allocator);
    defer b.deinit();
    Fake.fail_fill = true;
    try std.testing.expectError(error.NotReady, b.out(.{ .d = &d, .handle = null }, 16));
    try std.testing.expectEqual(@as(usize, 1), Fake.allocs);
    try std.testing.expectEqual(Fake.allocs, Fake.frees);
    try std.testing.expectEqual(@as(usize, 0), b.list.items.len);
}

test "window diagnostic buffers transfer ownership once and free once" {
    var d = Fake.driver();
    var b = buffers(&d, std.testing.allocator);
    _ = try b.up(&.{ 1, 2 });
    _ = try b.out(.{ .d = &d, .handle = null }, 16);
    try std.testing.expectEqual(@as(usize, 2), b.list.items.len);
    try std.testing.expectEqual(@as(usize, 0), Fake.frees);
    b.deinit();
    try std.testing.expectEqual(@as(usize, 2), Fake.allocs);
    try std.testing.expectEqual(Fake.allocs, Fake.frees);
}
