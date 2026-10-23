//! The real post-load stage and real EXL3 Kit teardown with an inert CUDA ABI.
const std = @import("std");
const cuda = @import("cuda");
const Kit = @import("cuda_exl3_weights.zig").Kit;
const tail = @import("cuda_engine_tail.zig");
const t = std.testing;
const Fake = struct {
    var phase: usize = 0;
    var device: usize = 0;
    var host: usize = 0;
    var host_calls: usize = 0;
    var unloaded: usize = 0;
    var next: u64 = 0x10000;
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        if (phase == 1) return cuda.abi.error_not_ready;
        next += 0x10000;
        out.* = next;
        device += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        device -= 1;
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        host_calls += 1;
        if (phase == 2 or (phase == 3 and host_calls == 2)) return cuda.abi.error_not_ready;
        next += 0x10000;
        out.* = @ptrFromInt(next);
        host += 1;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        host -= 1;
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        unloaded += 1;
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (phase == 4) cuda.abi.error_not_ready else 0;
    }
};
const Engine = struct {
    gpa: std.mem.Allocator = t.allocator,
    d: *cuda.Driver,
    kit: ?*Kit,
    stream: cuda.Stream,
    g: struct { vocab: usize = 8 } = .{},
    verify_rows: usize = 2,
    tables: cuda.DeviceBuffer = undefined,
    staged: cuda.HostBuffer = undefined,
    host_logits: cuda.HostBuffer = undefined,
    pub fn scratch(_: *@This(), _: enum { allocate }) !usize {
        return 0;
    }
    pub fn finishWarm(_: *@This(), _: std.Io, _: i96) !void {
        return error.InjectedWarmFailure;
    }
};
fn kit(d: *cuda.Driver) !*Kit {
    const k = try t.allocator.create(Kit);
    const module: cuda.Module = .{ .d = d, .handle = @ptrFromInt(16) };
    k.dense = .{ .d = d, .lin = .{ .mod = module, .fns = undefined, .rot_in = undefined }, .xh = try cuda.DeviceBuffer.alloc(d, 64), .z = try cuda.DeviceBuffer.alloc(d, 128), .counters = try cuda.DeviceBuffer.alloc(d, 32) };
    k.experts = undefined;
    k.experts.mod = module;
    k.b16 = undefined;
    k.b16.mod = module;
    k.grouped = .{ .d = d, .kernels = &k.experts, .storage = try cuda.DeviceBuffer.alloc(d, 64) };
    return k;
}

test "EXL3 late init: table pinned logits wait and warm failures release every real Kit owner" {
    for (1..6) |phase| {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = Fake.alloc;
        d.api.cuMemFree_v2 = Fake.free;
        d.api.cuMemHostAlloc = Fake.hostAlloc;
        d.api.cuMemFreeHost = Fake.hostFree;
        d.api.cuModuleUnload = Fake.unload;
        d.api.cuStreamSynchronize = Fake.wait;
        Fake.phase = 0;
        Fake.host_calls = 0;
        Fake.unloaded = 0;
        const k = try kit(&d);
        var e: Engine = .{ .d = &d, .kit = k, .stream = .{ .d = &d, .handle = null } };
        Fake.phase = phase;
        const result = tail.finish(&e, t.io, 0);
        if (result) |_| return error.ExpectedLateFailure else |_| {}
        const released = Fake.unloaded;
        // Rescue the pre-fix owner before asserting, so a red run does not leak its own fixtures.
        if (released == 0) k.deinit(t.allocator);
        try t.expectEqual(@as(usize, 0), Fake.device);
        try t.expectEqual(@as(usize, 0), Fake.host);
        try t.expectEqual(@as(usize, 3), released);
    }
}
