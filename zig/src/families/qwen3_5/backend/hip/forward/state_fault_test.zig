//! Host faults at Qwen HIP page-table publication and writable-copy bookkeeping.
const std = @import("std");
const hip = @import("hip");
const pages = @import("pages.zig");
const state = @import("state.zig");

const Host = struct {
    var uploads: ?usize = null;

    fn upload(dst: hip.abi.DevicePtr, src: [*]const u8, n: usize) callconv(.c) hip.abi.Result {
        if (uploads) |*left| {
            if (left.* == 0) return 1;
            left.* -= 1;
        }
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], src[0..n]);
        return 0;
    }

    fn alloc(out: *hip.abi.DevicePtr, n: usize) callconv(.c) hip.abi.Result {
        out.* = std.c.malloc(n) orelse return 1;
        return 0;
    }

    fn free(ptr: hip.abi.DevicePtr) callconv(.c) hip.abi.Result {
        std.c.free(ptr);
        return 0;
    }

    fn runtime() hip.Runtime {
        var r: hip.Runtime = .{ .lib = undefined, .api = undefined };
        r.api.hipMemcpyHtoD = upload;
        r.api.hipMalloc = alloc;
        r.api.hipFree = free;
        return r;
    }
};

const Fixture = struct {
    runtime: hip.Runtime,
    pool: pages.Pool,
    caches: state.Caches,

    fn init(f: *Fixture, count: usize) !void {
        const a = std.testing.allocator;
        f.runtime = Host.runtime();
        f.pool = .{
            .gpa = a,
            .d = &f.runtime,
            .count = count,
            .kv_heads = 0,
            .head_dim = 0,
            .act_bytes = 0,
            .keys = &.{},
            .values = &.{},
            .ids = try pages.Ids.init(a, count),
            .managed = true,
        };
        errdefer f.pool.ids.deinit(a);
        const desc = try hip.DeviceBuffer.alloc(&f.runtime, 64);
        f.caches = .{
            .layers = &.{},
            .total = count * pages.tokens,
            .desc = desc,
            .pool = &f.pool,
        };
    }

    fn deinit(f: *Fixture) void {
        f.caches.table.deinit(std.testing.allocator);
        f.caches.desc.free();
        f.pool.ids.deinit(std.testing.allocator);
    }

    fn oldShared(f: *Fixture) !struct { old: u32, fresh: u32 } {
        const old = f.pool.take().?;
        try f.caches.table.append(std.testing.allocator, old);
        f.pool.retain(old);
        const fresh = f.pool.ids.free.items[f.pool.ids.free.items.len - 1];
        return .{ .old = old, .fresh = fresh };
    }
};

fn rollbackSet(c: *state.Caches, ids: []const u32) !void {
    errdefer for (ids) |id| c.pool.release(id);
    try c.set(std.testing.allocator, 0, ids);
}

test "Qwen HIP page faults: failed set leaves host ownership with the old table" {
    var f: Fixture = undefined;
    try f.init(3);
    defer f.deinit();
    const a = std.testing.allocator;
    const old = f.pool.take().?;
    const incoming = f.pool.take().?;
    f.pool.retain(old);
    try f.caches.table.append(a, old);
    Host.uploads = 0;
    defer Host.uploads = null;
    try std.testing.expectError(error.HipFailed, rollbackSet(&f.caches, &.{incoming}));
    try std.testing.expectEqualSlices(u32, &.{old}, f.caches.table.items);
    try std.testing.expectEqual(@as(u32, 2), f.pool.ids.refs[old]);
    try std.testing.expectEqual(@as(u32, 0), f.pool.ids.refs[incoming]);
    Host.uploads = null;

    const replacement = f.pool.take().?;
    try f.caches.set(a, 0, &.{replacement});
    try std.testing.expectEqualSlices(u32, &.{replacement}, f.caches.table.items);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[old]);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[replacement]);
}

test "Qwen HIP page faults: failed writable upload leaves the installed page owned once" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    const ids = try f.oldShared();
    var out: std.ArrayList([3]u32) = .empty;
    defer out.deinit(std.testing.allocator);
    Host.uploads = 0;
    defer Host.uploads = null;
    try std.testing.expectError(error.HipFailed, f.caches.writable(&out, std.testing.allocator, 0, 1, null));
    try std.testing.expectEqualSlices(u32, &.{ids.fresh}, f.caches.table.items);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[ids.old]);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[ids.fresh]);
}

test "Qwen HIP page faults: failed writable result append leaves the installed page owned once" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    const ids = try f.oldShared();
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 0 });
    var out: std.ArrayList([3]u32) = .empty;
    defer out.deinit(failing.allocator());
    try std.testing.expectError(error.OutOfMemory, f.caches.writable(&out, failing.allocator(), 0, 1, null));
    try std.testing.expectEqualSlices(u32, &.{ids.fresh}, f.caches.table.items);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[ids.old]);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[ids.fresh]);
}
