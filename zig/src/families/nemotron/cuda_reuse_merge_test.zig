//! Integration checks through the real snapshot mover and allocator; only CUDA copies and allocations are fake.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const t = std.testing;
fn download(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    return 0;
}
var device: [4][2 << 20]u8 align(16) = undefined;
var live: [4]bool = @splat(false);
var freed: usize = 0;
fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
    std.debug.assert(n <= device[0].len);
    for (&live, 0..) |*on, i| if (!on.*) {
        on.* = true;
        out.* = @intFromPtr(&device[i]);
        return 0;
    };
    return 2;
}
fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    return 0;
}
fn free(ptr: u64) callconv(.c) cuda.abi.Result {
    for (&live, 0..) |*on, i| if (ptr == @intFromPtr(&device[i])) {
        std.debug.assert(on.*);
        on.* = false;
        freed += 1;
        return 0;
    };
    unreachable;
}
pub fn Cases(comptime Engine: type, comptime Target: type, comptime config: anytype, comptime move: anytype, comptime blank: anytype, comptime drop: anytype, comptime save: anytype, comptime finish: anytype) type {
    return struct {
        test "kv merge: host snapshots follow shuffled KV pages and every layer's planes" {
            var d: cuda.Driver = undefined;
            d.api.cuMemcpyDtoHAsync_v2 = download;
            var e: Engine = undefined;
            e.c = config();
            e.c.layers = 2;
            e.c.kinds[0..2].* = .{ .attention, .attention };
            e.k.d = &d;
            e.stream = .{ .d = &d, .handle = null };
            e.head = null;
            e.snap_host = true;
            e.max_len = 1024;
            e.b.lane_part = 0;
            const row = state.kvRowBytes(e.c);
            const plane = state.chunk_keys * row;
            const page = e.pageBytes();
            const gpu = try t.allocator.alloc(u8, 4 * page);
            defer t.allocator.free(gpu);
            for (0..4) |id| for (0..4) |part| @memset(gpu[id * page + part * plane ..][0..plane], @intCast(id * 16 + part));
            var ids = [_]u32{ 3, 1 };
            var seq: state.Seq = undefined;
            seq.pages = &ids;
            e.bound = &seq;
            e.b.table = 1;
            e.b.k_cache = @intFromPtr(gpu.ptr);
            e.b.v_cache = e.b.k_cache + plane;
            e.b.ssm = 0;
            e.b.conv_base = 0;
            const rows = 513;
            const host = try t.allocator.alloc(u8, 4 * rows * row);
            defer t.allocator.free(host);
            const target: Target = .{ .e = &e, .head = null };
            try move(&target, rows, host, true, true, null);
            for ([_]usize{ 0, 2, 1, 3 }, 0..) |part, i| {
                try t.expect(std.mem.allEqual(u8, host[i * rows * row ..][0..plane], @intCast(48 + part)));
                try t.expect(std.mem.allEqual(u8, host[i * rows * row + plane ..][0..row], @intCast(16 + part)));
            }
        }
        test "kv merge: an exhausted host staging owner skips the mark without allocating device memory" {
            live = @splat(false);
            var d: cuda.Driver = undefined;
            d.api.cuStreamSynchronize = sync;
            d.api.cuMemAlloc_v2 = alloc;
            d.api.cuMemFree_v2 = free;
            var ctx: cuda.Context = undefined;
            ctx.d = &d;
            var e: Engine = undefined;
            e.gpa = t.allocator;
            e.ctx = &ctx;
            e.stream = .{ .d = &d, .handle = null };
            e.c = config();
            e.c.layers = 1;
            e.c.kinds[0] = .mamba;
            e.c.mamba_heads = 1;
            e.c.mamba_head_dim = 1;
            e.c.groups = 1;
            e.c.state = 1;
            e.snap_host = true;
            e.spares = .{};
            e.armed = .{ .open = .{ 0, 4 } };
            e.parity = 0;
            e.rows = 0;
            e.prev_keep = 0;
            const target: Target = .{ .e = &e, .head = null };
            const copy = save(@constCast(&target), null, 1) catch |err| {
                try t.expectEqual(error.NoSnapshotSlot, err);
                for (live) |on| try t.expect(!on);
                return;
            };
            defer drop(@constCast(&target), copy);
            return error.ExpectedBoundedStaging;
        }
        test "kv merge: failed host materialization keeps staging within the startup reservation" {
            live = @splat(false);
            const Fail = struct {
                fn host(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
                    return cuda.abi.error_not_ready;
                }
            };
            var d: cuda.Driver = undefined;
            d.api.cuStreamSynchronize = sync;
            d.api.cuMemAlloc_v2 = alloc;
            d.api.cuMemFree_v2 = free;
            d.api.cuMemHostAlloc = Fail.host;
            var ctx: cuda.Context = undefined;
            ctx.d = &d;
            var e: Engine = undefined;
            e.gpa = t.allocator;
            e.ctx = &ctx;
            e.stream = .{ .d = &d, .handle = null };
            e.c = config();
            e.c.layers = 1;
            e.c.kinds[0] = .mamba;
            e.c.mamba_heads = 1;
            e.c.mamba_head_dim = 1;
            e.c.groups = 1;
            e.c.state = 1;
            e.snap_host = true;
            e.spares = .{ .want = 1 };
            e.spares.fill(&d, 2 << 20);
            defer e.spares.deinit();
            e.armed = .{ .open = .{ 0, 4 } };
            e.parity = 0;
            e.rows = 0;
            e.prev_keep = 0;
            const target: Target = .{ .e = &e, .head = null };
            const copy = try save(@constCast(&target), null, 1);
            defer drop(@constCast(&target), copy);
            try finish(target);
            try t.expectEqual(@as(usize, 1), std.mem.count(bool, &live, &.{true}));
            try t.expectEqual(@as(usize, 0), e.spares.n);
        }
        test "kv merge: device snapshots borrow a slot and drop returns it without freeing its buffer" {
            live = @splat(false);
            freed = 0;
            var d: cuda.Driver = undefined;
            d.api.cuStreamSynchronize = sync;
            d.api.cuMemAlloc_v2 = alloc;
            d.api.cuMemFree_v2 = free;
            var ctx: cuda.Context = undefined;
            ctx.d = &d;
            var e: Engine = undefined;
            e.gpa = t.allocator;
            e.ctx = &ctx;
            e.stream = .{ .d = &d, .handle = null };
            e.snaps_configured = false;
            e.c = config();
            e.c.layers = 1;
            e.c.kinds[0] = .attention;
            e.snap_host = false;
            e.snaps = .{};
            e.spares = .{};
            e.armed = .{};
            e.parity = 0;
            e.rows = 0;
            try e.snaps.init(&d, 1, 64);
            defer e.snaps.deinit();
            defer e.spares.deinit();
            const ptr = e.snaps.at(0);
            const target: Target = .{ .e = &e, .head = null };
            const copy = try blank(&target, 1);
            var held = true;
            defer if (held) drop(@constCast(&target), copy);
            try t.expectEqual(ptr, copy.mamba.ptr);
            try t.expectEqual(@as(usize, 0), e.snaps.n);
            drop(@constCast(&target), copy);
            held = false;
            try t.expectEqual(@as(usize, 1), e.snaps.n);
            try t.expectEqual(@as(usize, 1), freed);
            try t.expect(live[0]);
        }
    };
}
