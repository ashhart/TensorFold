//! Test allocator: count allocations, fail one selected call, and forward frees from existing containers.
const std = @import("std");
const A = std.mem.Allocator;
pub const Fault = struct {
    calls: usize = 0,
    failures: usize = 0,
    remaining: ?usize = null,
    pub fn arm(f: *Fault, nth: usize) void {
        f.remaining = nth;
    }
    pub fn allocator(f: *Fault) A {
        return .{ .ptr = f, .vtable = &.{ .alloc = alloc, .resize = A.noResize, .remap = A.noRemap, .free = free } };
    }
    fn alloc(ptr: *anyopaque, len: usize, align_: std.mem.Alignment, ret: usize) ?[*]u8 {
        const f: *Fault = @ptrCast(@alignCast(ptr));
        f.calls += 1;
        if (f.remaining) |n| {
            if (n == 1) {
                f.remaining = null;
                f.failures += 1;
                return null;
            }
            f.remaining = n - 1;
        }
        return std.testing.allocator.rawAlloc(len, align_, ret);
    }
    fn free(_: *anyopaque, bytes: []u8, align_: std.mem.Alignment, ret: usize) void {
        std.testing.allocator.rawFree(bytes, align_, ret);
    }
};
