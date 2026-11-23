//! Host faults at Nemotron HIP page adoption and the lane's later rollback.
const std = @import("std");
const hip = @import("hip");
const pages = @import("pages.zig");

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
    stream: pages.Stream,

    fn init(f: *Fixture, count: usize) !void {
        const a = std.testing.allocator;
        f.runtime = Host.runtime();
        f.pool = .{
            .gpa = a,
            .count = count,
            .kv_heads = 0,
            .head_kv = null,
            .head_dim = 0,
            .act = .bf16,
            .keys = &.{},
            .values = &.{},
            .ids = try @import("hip_serve").pages.Ids.init(a, count),
        };
        errdefer f.pool.ids.deinit(a);
        const desc = try hip.DeviceBuffer.alloc(&f.runtime, 64);
        f.stream = .{
            .gpa = a,
            .desc = desc,
            .table_at = 0,
            .cap = count * pages.tokens,
            .mem = desc,
            .states = &.{},
        };
    }

    fn deinit(f: *Fixture) void {
        f.stream.ids.deinit(std.testing.allocator);
        f.stream.desc.free();
        f.pool.ids.deinit(std.testing.allocator);
    }
};

fn adoptThenRefuseSend(s: *pages.Stream, pool: *pages.Pool, rollback: *std.ArrayList(u32)) !void {
    errdefer for (rollback.items) |id| pool.release(id);
    try s.adoptRollback(pool, rollback);
    return error.SendFailed;
}

test "Nemotron HIP page faults: failed adoption leaves the old table and rollback references" {
    var f: Fixture = undefined;
    try f.init(3);
    defer f.deinit();
    const old = f.pool.take().?;
    const incoming = f.pool.take().?;
    try f.stream.ids.append(std.testing.allocator, old);
    var rollback: std.ArrayList(u32) = .empty;
    defer rollback.deinit(std.testing.allocator);
    try rollback.append(std.testing.allocator, incoming);
    f.pool.retain(incoming);
    Host.uploads = 0;
    defer Host.uploads = null;
    try std.testing.expectError(error.HipFailed, adoptThenRefuseSend(&f.stream, &f.pool, &rollback));
    try std.testing.expectEqualSlices(u32, &.{old}, f.stream.ids.items);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[old]);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[incoming]);
    try std.testing.expectEqual(@as(usize, 1), rollback.items.len);
}

test "Nemotron HIP page faults: a late send failure does not release adopted pages twice" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    const incoming = f.pool.take().?;
    var rollback: std.ArrayList(u32) = .empty;
    defer rollback.deinit(std.testing.allocator);
    try rollback.append(std.testing.allocator, incoming);
    f.pool.retain(incoming);
    try std.testing.expectError(error.SendFailed, adoptThenRefuseSend(&f.stream, &f.pool, &rollback));
    try std.testing.expectEqualSlices(u32, &.{incoming}, f.stream.ids.items);
    try std.testing.expectEqual(@as(u32, 2), f.pool.ids.refs[incoming]);
    try std.testing.expectEqual(@as(usize, 0), rollback.items.len);
}

test "Nemotron HIP page faults: retain adoption rolls back failure and retains success" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    const old = f.pool.take().?;
    const incoming = f.pool.take().?;
    try f.stream.ids.append(std.testing.allocator, old);
    Host.uploads = 0;
    try std.testing.expectError(error.HipFailed, f.stream.adopt(&f.pool, &.{incoming}, true));
    Host.uploads = null;
    try std.testing.expectEqualSlices(u32, &.{old}, f.stream.ids.items);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[old]);
    try std.testing.expectEqual(@as(u32, 1), f.pool.ids.refs[incoming]);

    try f.stream.adopt(&f.pool, &.{incoming}, true);
    try std.testing.expectEqualSlices(u32, &.{incoming}, f.stream.ids.items);
    try std.testing.expectEqual(@as(u32, 0), f.pool.ids.refs[old]);
    try std.testing.expectEqual(@as(u32, 2), f.pool.ids.refs[incoming]);
}
