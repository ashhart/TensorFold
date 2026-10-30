//! Transfer a GLM device upload and its label together, or free both before returning an allocation failure.
const std = @import("std");
const cuda = @import("cuda");

pub fn keep(L: anytype, buffer: cuda.DeviceBuffer, comptime fmt: []const u8, args: anytype) !u64 {
    var b = buffer;
    errdefer b.free();
    const name = try std.fmt.allocPrint(L.gpa, fmt, args);
    errdefer L.gpa.free(name);
    try L.w.buffers.ensureUnusedCapacity(L.gpa, 1);
    try L.w.named.ensureUnusedCapacity(L.gpa, 1);
    L.w.buffers.appendAssumeCapacity(b);
    L.w.named.appendAssumeCapacity(.{ .name = name, .ptr = b.ptr, .len = b.len });
    return b.ptr;
}

test "a GLM upload and its label have one owner through every registration allocation failure" {
    const F = struct {
        var frees: usize = 0;
        fn alloc(p: *u64, _: usize) callconv(.c) cuda.abi.Result {
            p.* = 256;
            return cuda.abi.success;
        }
        fn free(_: u64) callconv(.c) cuda.abi.Result {
            frees += 1;
            return cuda.abi.success;
        }
        const Named = struct { name: []u8, ptr: u64, len: usize };
        const W = struct {
            buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
            named: std.ArrayList(Named) = .empty,
        };
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuMemAlloc_v2 = F.alloc;
    driver.api.cuMemFree_v2 = F.free;
    var at: usize = 0;
    while (true) : (at += 1) {
        F.frees = 0;
        var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = at });
        const gpa = fail.allocator();
        var w: F.W = .{};
        defer w.buffers.deinit(gpa);
        defer w.named.deinit(gpa);
        var loader = .{ .gpa = gpa, .w = &w };
        const buffer = try cuda.DeviceBuffer.alloc(&driver, 1);
        const result = keep(&loader, buffer, "upload.{d}", .{at});
        defer if (F.frees == 0 and w.buffers.items.len == 0) {
            var orphan = buffer;
            orphan.free();
        };
        if (fail.has_induced_failure) {
            try std.testing.expectError(error.OutOfMemory, result);
            try std.testing.expectEqual(@as(usize, 0), w.buffers.items.len);
            try std.testing.expectEqual(@as(usize, 0), w.named.items.len);
            try std.testing.expectEqual(@as(usize, 1), F.frees);
        } else {
            _ = try result;
            try std.testing.expectEqual(@as(usize, 1), w.buffers.items.len);
            try std.testing.expectEqual(@as(usize, 1), w.named.items.len);
            try std.testing.expectEqual(@as(usize, 0), F.frees);
            w.buffers.items[0].free();
            gpa.free(w.named.items[0].name);
            try std.testing.expectEqual(@as(usize, 1), F.frees);
            break;
        }
    }
    try std.testing.expect(at >= 3);
}
