//! Production load ownership under an inert driver and real checkpoint mappings.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const weights = @import("cuda_weights.zig");
const abi = cuda.abi;
const Mock = struct {
    var calls: usize = 0;
    var fail: usize = 0;
    var active: [64]usize = @splat(0);
    var host: [64 << 20]u8 align(16) = undefined;
    var host_live: usize = 0;
    var sync_fail: bool = false;
    var wait_fail = false;
    var create_fail = false;
    var record_fail = false;
    var pending = false;
    var unsafe_frees: usize = 0;
    var unloads: usize = 0;
    var releases: usize = 0;
    var current: abi.Context = @ptrFromInt(0x3000);
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        calls += 1;
        if (calls == fail) return abi.error_not_ready;
        active[calls] = n;
        out.* = 0x1000 + calls;
        return 0;
    }
    fn free(p: u64) callconv(.c) c_int {
        if (pending) unsafe_frees += 1;
        active[p - 0x1000] = 0;
        return 0;
    }
    fn hostAlloc(p: *?*anyopaque, _: usize, _: c_uint) callconv(.c) c_int {
        p.* = &host;
        host_live += 1;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) c_int {
        host_live -= 1;
        return 0;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn copyAsync(p: u64, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) c_int {
        pending = true;
        return upload(p, src, n);
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) c_int {
        if (create_fail) return abi.error_not_ready;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) c_int {
        return if (record_fail) abi.error_not_ready else 0;
    }
    fn eventWait(_: abi.Event) callconv(.c) c_int {
        if (wait_fail) return abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn streamSync(_: abi.Stream) callconv(.c) c_int {
        if (wait_fail) return abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn sync() callconv(.c) c_int {
        if (sync_fail) return abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyHtoD_v2 = upload;
        d.api.cuMemcpyHtoDAsync_v2 = copyAsync;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventSynchronize = eventWait;
        d.api.cuEventQuery = eventWait;
        d.api.cuEventDestroy_v2 = eventWait;
        d.api.cuCtxSynchronize = sync;
        d.api.cuCtxSetCurrent = setCurrent;
        d.api.cuModuleUnload = unload;
        d.api.cuDevicePrimaryCtxRelease_v2 = release;
        d.api.cuEventDestroy_v2 = eventDestroy;
        d.api.cuStreamSynchronize = streamSync;
        return d;
    }
    fn setCurrent(c: abi.Context) callconv(.c) c_int {
        current = c;
        cuda.copy_source.setCurrent(c);
        return 0;
    }
    fn unload(_: abi.Module) callconv(.c) c_int {
        if (pending) unsafe_frees += 1;
        unloads += 1;
        return 0;
    }
    fn release(_: abi.Device) callconv(.c) c_int {
        if (pending) unsafe_frees += 1;
        releases += 1;
        return 0;
    }
    fn eventDestroy(_: abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn model() h3.Model {
        calls = 0;
        fail = 0;
        active = @splat(0);
        host_live = 0;
        sync_fail = false;
        wait_fail = false;
        create_fail = false;
        record_fail = false;
        pending = false;
        unsafe_frees = 0;
        unloads = 0;
        releases = 0;
        cuda.copy_source.setCurrent(current);
        var m: h3.Model = undefined;
        m.gpa = std.testing.allocator;
        m.driver = driver();
        m.blocks = &.{};
        m.load_stage = .{};
        m.load_failed = false;
        m.retained_next = null;
        return m;
    }
    fn live() usize {
        var n: usize = 0;
        for (active) |x| n += @intFromBool(x != 0);
        return n;
    }
    fn cleanup(m: *h3.Model) void {
        for (active, 0..) |n, i| if (n != 0) {
            var b: cuda.DeviceBuffer = .{ .d = &m.driver, .ptr = 0x1000 + i, .len = n };
            b.free();
        };
    }
};
fn fixture(dir: std.Io.Dir, qkv: usize, fc2_rows: usize) ![:0]u8 {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const names = [_][]const u8{ "attn.qkv_proj.weight", "attn.to_gate_compress.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight", "norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight" };
    const shapes = [_][2]usize{ .{ qkv, 1 }, .{ 128, 1 }, .{ 1, 128 }, .{ 2, 1 }, .{ fc2_rows, 1 }, .{ 1, 1 }, .{ 1, 1 }, .{ 1, 128 }, .{ 1, 128 } };
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.appendSlice(a, "{");
    var offset: usize = 0;
    for (names, shapes, 0..) |name, shape, i| {
        const n = shape[0] * shape[1] * 2;
        try header.print(a, "{s}\"blocks.0.{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ if (i == 0) "" else ",", name, shape[0], shape[1], offset, offset + n });
        offset += n;
    }
    try header.appendSlice(a, "}");
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    try bytes.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, bytes.items[0..8], header.items.len, .little);
    try bytes.appendSlice(a, header.items);
    try bytes.appendNTimes(a, 0, offset);
    try dir.writeFile(io, .{ .sub_path = "weights.safetensors", .data = bytes.items });
    const path = try dir.realPathFileAlloc(io, "weights.safetensors", a);
    defer a.free(path);
    return a.dupeSentinel(u8, path, 0);
}
test "h3 ownership: partial block allocations 2 through 5 preserve device counts with an inert context sentinel" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(tmp.dir, 384, 1);
    defer std.testing.allocator.free(path);
    for (2..6) |at| {
        var m = Mock.model();
        defer Mock.cleanup(&m);
        Mock.fail = at;
        const before = cuda.usage(false);
        try std.testing.expectError(error.NotReady, weights.read(&m, path.ptr));
        try std.testing.expectEqual(@as(usize, 0), Mock.live());
        try std.testing.expectEqual(@as(usize, 0), Mock.host_live);
        try std.testing.expectEqual(before.device, cuda.usage(false).device);
        try std.testing.expect(Mock.current == @as(abi.Context, @ptrFromInt(0x3000)));
    }
}
test "h3 metadata: nondivisible and mismatched projections are refused before device allocation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]usize{ 385, 381 }) |rows| {
        const path = try fixture(tmp.dir, rows, 1);
        defer std.testing.allocator.free(path);
        var m = Mock.model();
        defer Mock.cleanup(&m);
        const result = weights.read(&m, path.ptr);
        if (result) |_| weights.free(&m) else |_| {}
        try std.testing.expectError(error.BadGeometry, result);
        try std.testing.expectEqual(@as(usize, 0), Mock.calls);
    }
}

test "h3 ownership: synchronization failure after loading releases all weights" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(tmp.dir, 384, 1);
    defer std.testing.allocator.free(path);
    var m = Mock.model();
    m.ctx = .{ .d = &m.driver, .device = 3, .handle = Mock.current };
    defer if (Mock.live() > 0) weights.free(&m);
    Mock.sync_fail = true;
    const before = cuda.usage(false);
    try std.testing.expectError(error.NotReady, weights.readSynced(&m, path.ptr));
    try std.testing.expectEqual(@as(usize, 0), Mock.live());
    try std.testing.expectEqual(before.device, cuda.usage(false).device);
    try std.testing.expect(Mock.current == @as(abi.Context, @ptrFromInt(0x3000)));
}

test "h3 metadata: exact projection shape is checked before pinned and device allocation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(tmp.dir, 384, 2);
    defer std.testing.allocator.free(path);
    var m = Mock.model();
    defer Mock.cleanup(&m);
    try std.testing.expectError(error.WrongShape, weights.read(&m, path.ptr));
    try std.testing.expectEqual(@as(usize, 0), Mock.calls);
    try std.testing.expectEqual(@as(usize, 0), Mock.host_live);
}

fn blockFixture(dir: std.Io.Dir, ids: []const usize, missing_norm: ?usize) ![:0]u8 {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const names = [_][]const u8{ "attn.qkv_proj.weight", "attn.to_gate_compress.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight", "norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight" };
    const shapes = [_][2]usize{ .{ 384, 1 }, .{ 128, 1 }, .{ 1, 128 }, .{ 2, 1 }, .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 }, .{ 1, 128 }, .{ 1, 128 } };
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.appendSlice(a, "{");
    var offset: usize = 0;
    var count: usize = 0;
    for (ids) |id| for (names, shapes) |name, shape| {
        if (missing_norm == id and std.mem.eql(u8, name, "norm1.weight")) continue;
        const n = shape[0] * shape[1] * 2;
        try header.print(a, "{s}\"blocks.{d}.{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ if (count == 0) "" else ",", id, name, shape[0], shape[1], offset, offset + n });
        count += 1;
        offset += n;
    };
    try header.appendSlice(a, "}");
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    try bytes.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, bytes.items[0..8], header.items.len, .little);
    try bytes.appendSlice(a, header.items);
    try bytes.appendNTimes(a, 0, offset);
    try dir.writeFile(io, .{ .sub_path = "blocks.safetensors", .data = bytes.items });
    const path = try dir.realPathFileAlloc(io, "blocks.safetensors", a);
    defer a.free(path);
    return a.dupeSentinel(u8, path, 0);
}
fn badBlocks(ids: []const usize, missing: ?usize) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try blockFixture(tmp.dir, ids, missing);
    defer std.testing.allocator.free(path);
    var m = Mock.model();
    const result = weights.read(&m, path.ptr);
    if (result) |_| weights.free(&m) else |_| {}
    try std.testing.expectError(error.BadBlocks, result);
    try std.testing.expectEqual(@as(usize, 0), Mock.calls);
    try std.testing.expectEqual(@as(usize, 0), Mock.host_live);
}
test "h3 blocks: zero missing-first block fails before allocation" {
    try badBlocks(&.{0}, 0);
}
test "h3 blocks: gap block fails before allocation" {
    try badBlocks(&.{ 0, 2 }, null);
}
test "h3 blocks: missing-middle norm fails before allocation" {
    try badBlocks(&.{ 0, 1, 2 }, 1);
}
test "h3 blocks: valid one and two contiguous complete blocks load and release" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for (1..3) |n| {
        const path = try blockFixture(tmp.dir, (&[_]usize{ 0, 1 })[0..n], null);
        defer std.testing.allocator.free(path);
        var m = Mock.model();
        try weights.read(&m, path.ptr);
        try std.testing.expectEqual(n, m.blocks.len);
        try std.testing.expectEqual(5 * n, Mock.live());
        weights.free(&m);
        try std.testing.expectEqual(@as(usize, 0), Mock.live());
    }
}

test "h3 ownership: failed per-copy wait retains destinations until recovery" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(tmp.dir, 384, 1);
    defer std.testing.allocator.free(path);
    var m = Mock.model();
    m.ctx = .{ .d = &m.driver, .device = 3, .handle = Mock.current };
    defer {
        Mock.pending = false;
        Mock.wait_fail = false;
        Mock.sync_fail = false;
        cuda.copy_source.synchronizeContext(&m.driver) catch {};
        m.load_stage.deinit();
        if (m.blocks.len > 0) weights.free(&m);
    }
    Mock.wait_fail = true;
    Mock.sync_fail = true;
    try std.testing.expectError(error.NotReady, weights.readSynced(&m, path.ptr));
    try std.testing.expectEqual(@as(usize, 0), Mock.unsafe_frees);
    try std.testing.expectEqual(@as(usize, 5), Mock.live());
    Mock.wait_fail = false;
    Mock.sync_fail = false;
    try m.ctx.makeCurrent();
    try m.ctx.synchronize();
    weights.free(&m);
    try std.testing.expectEqual(@as(usize, 0), Mock.live());
}

test "h3 ownership: constructor keeps parents through failed wait and recovery for all post-submit event faults" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(tmp.dir, 384, 1);
    defer std.testing.allocator.free(path);
    for (0..3) |fault| {
        _ = Mock.model();
        const m = try @import("cuda_runtime_fake.zig").model();
        m.gpa.free(m.blocks);
        m.blocks = &.{};
        m.driver.api = Mock.driver().api;
        m.gemms.workspace = try cuda.DeviceBuffer.alloc(&m.driver, 16);
        Mock.wait_fail = true;
        Mock.sync_fail = true;
        Mock.create_fail = fault == 1;
        Mock.record_fail = fault == 2;
        try std.testing.expectError(error.NotReady, m.loadWeights(path.ptr));
        try std.testing.expect(Mock.pending);
        try std.testing.expectEqual(@as(usize, 6), Mock.live());
        try std.testing.expectEqual(@as(usize, 0), Mock.unsafe_frees);
        try std.testing.expectEqual(@as(usize, 0), Mock.unloads);
        try std.testing.expectEqual(@as(usize, 0), Mock.releases);
        try std.testing.expectEqual(@as(usize, 1), @import("cuda_load_retained.zig").recover());
        try std.testing.expectEqual(@as(usize, 6), Mock.live());
        try std.testing.expectError(error.LoadPending, h3.Model.load(std.testing.allocator, path.ptr));
        Mock.wait_fail = false;
        Mock.sync_fail = false;
        try std.testing.expectEqual(@as(usize, 0), @import("cuda_load_retained.zig").recover());
        try std.testing.expectEqual(@as(usize, 0), Mock.live());
        try std.testing.expectEqual(@as(usize, 0), Mock.host_live);
        try std.testing.expectEqual(@as(usize, 0), Mock.unsafe_frees);
        try std.testing.expectEqual(@as(usize, 1), Mock.unloads);
        try std.testing.expectEqual(@as(usize, 1), Mock.releases);
    }
}

test "h3 ownership: open refuses while detached failed models are recovering" {
    const retained = @import("cuda_load_retained.zig");
    const Barrier = @import("load_barrier_test.zig");
    const a = try @import("cuda_runtime_fake.zig").model();
    const b = try @import("cuda_runtime_fake.zig").model();
    _ = Mock.model();
    a.driver.api = Mock.driver().api;
    b.driver.api = Mock.driver().api;
    b.ctx.handle = @ptrFromInt(0x4000);
    b.ctx.device = 4;
    retained.keep(a);
    retained.keep(b);
    Barrier.reset();
    retained.Testing.detached = Barrier.pause;
    defer retained.Testing.detached = null;
    const Worker = struct {
        fn run() void {
            _ = retained.recover();
        }
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{});
    var joined = false;
    defer if (!joined) {
        Barrier.release();
        thread.join();
    };
    Barrier.wait();
    try std.testing.expectError(error.LoadPending, retained.admit());
    try std.testing.expectEqual(@as(usize, 0), Mock.releases);
    try std.testing.expectEqual(@as(usize, 0), Mock.unloads);
    Barrier.release();
    thread.join();
    joined = true;
    try std.testing.expectEqual(@as(usize, 2), Mock.releases);
    try std.testing.expectEqual(@as(usize, 2), Mock.unloads);
    try std.testing.expectEqual(@as(usize, 0), Mock.unsafe_frees);
    try std.testing.expectEqual(@as(usize, 0), retained.recover());
}
