//! Pages of a KV pool: 64 positions each, their ids and holders kept on the host; the device buffers are a family's.

const std = @import("std");
const radix = @import("engine_api").prompt_radix;
const Allocator = std.mem.Allocator;

/// Positions a page holds: the chunk of the recurrences, so a page edge is a chunk edge.
pub const tokens = 64;

/// Pages that hold `n` positions.
pub fn pagesFor(n: usize) usize {
    return (n + tokens - 1) / tokens;
}

/// Which page ids are free and how many holders each has: host bookkeeping only.
pub const Ids = struct {
    refs: []u32,
    free: std.ArrayList(u32) = .empty,
    /// Pages promised to streams that have not taken them yet.
    reserved: usize = 0,

    pub fn init(gpa: Allocator, count: usize) !Ids {
        const refs = try gpa.alloc(u32, count);
        errdefer gpa.free(refs);
        @memset(refs, 0);
        var ids: Ids = .{ .refs = refs };
        errdefer ids.free.deinit(gpa);
        try ids.free.ensureTotalCapacity(gpa, count);
        // the lowest id is taken first
        var i: u32 = @intCast(count);
        while (i > 0) {
            i -= 1;
            ids.free.appendAssumeCapacity(i);
        }
        return ids;
    }

    pub fn deinit(ids: *Ids, gpa: Allocator) void {
        ids.free.deinit(gpa);
        gpa.free(ids.refs);
    }

    /// A free page, now held once; null when none is free.
    pub fn take(ids: *Ids) ?u32 {
        const id = ids.free.pop() orelse return null;
        ids.refs[id] = 1;
        return id;
    }

    pub fn retain(ids: *Ids, id: u32) void {
        std.debug.assert(ids.refs[id] > 0);
        ids.refs[id] += 1;
    }

    /// One holder less; the page is free again when none is left.
    pub fn release(ids: *Ids, id: u32) void {
        std.debug.assert(ids.refs[id] > 0);
        ids.refs[id] -= 1;
        if (ids.refs[id] == 0) ids.free.appendAssumeCapacity(id);
    }

    pub fn holders(ids: *const Ids, id: u32) u32 {
        return ids.refs[id];
    }

    /// Pages free and not promised.
    pub fn available(ids: *const Ids) usize {
        return ids.free.items.len -| ids.reserved;
    }

    pub fn used(ids: *const Ids) usize {
        return ids.refs.len - ids.free.items.len;
    }

    /// The prompt cache's view of the pool: it holds pages beside the streams, each `bytes` over every layer.
    pub fn pages(ids: *Ids, bytes: u64) radix.Pages {
        return .{ .ptr = ids, .vtable = &vtable, .bytes = bytes };
    }

    fn of(ptr: *anyopaque) *Ids {
        return @ptrCast(@alignCast(ptr));
    }

    fn retainFn(ptr: *anyopaque, id: u32) void {
        of(ptr).retain(id);
    }

    fn releaseFn(ptr: *anyopaque, id: u32) void {
        of(ptr).release(id);
    }

    fn holdersFn(ptr: *anyopaque, id: u32) u32 {
        return of(ptr).holders(id);
    }

    fn availableFn(ptr: *anyopaque) usize {
        return of(ptr).available();
    }

    const vtable: radix.Pages.VTable = .{
        .retain = retainFn,
        .release = releaseFn,
        .holders = holdersFn,
        .available = availableFn,
    };
};

test "ids hand out the lowest page and free it when its last holder lets go" {
    var ids = try Ids.init(std.testing.allocator, 3);
    defer ids.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?u32, 0), ids.take());
    try std.testing.expectEqual(@as(?u32, 1), ids.take());
    ids.retain(0);
    ids.release(0);
    try std.testing.expectEqual(@as(usize, 1), ids.available());
    ids.release(0);
    try std.testing.expectEqual(@as(usize, 2), ids.available());
    try std.testing.expectEqual(@as(?u32, 0), ids.take());
    try std.testing.expectEqual(@as(?u32, 2), ids.take());
    try std.testing.expectEqual(@as(?u32, null), ids.take());
}

test "reserved pages are not available, and the prompt cache sees the same counts" {
    var ids = try Ids.init(std.testing.allocator, 4);
    defer ids.deinit(std.testing.allocator);
    ids.reserved = 3;
    try std.testing.expectEqual(@as(usize, 1), ids.available());
    ids.reserved = 9;
    try std.testing.expectEqual(@as(usize, 0), ids.available());
    const p = ids.pages(1024);
    try std.testing.expectEqual(@as(usize, 0), p.vtable.available(p.ptr));
    _ = ids.take();
    p.vtable.retain(p.ptr, 0);
    try std.testing.expectEqual(@as(u32, 2), p.vtable.holders(p.ptr, 0));
    p.vtable.release(p.ptr, 0);
    try std.testing.expectEqual(@as(u32, 1), ids.holders(0));
}

test "pages cover positions by 64" {
    try std.testing.expectEqual(@as(usize, 0), pagesFor(0));
    try std.testing.expectEqual(@as(usize, 1), pagesFor(64));
    try std.testing.expectEqual(@as(usize, 2), pagesFor(65));
}
