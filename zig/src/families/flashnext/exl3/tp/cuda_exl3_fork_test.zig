const std = @import("std");
const cuda = @import("cuda");
const load = @import("../pack/cuda_exl3_load.zig");
const fork = @import("cuda_exl3_fork.zig");

const Fake = struct {
    var streams: usize = 0;
    var modules: usize = 0;
    var events: usize = 0;
    var buffers: usize = 0;
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn stream(_: cuda.abi.Stream) callconv(.c) c_int {
        streams += 1;
        return 0;
    }
    fn module(_: cuda.abi.Module) callconv(.c) c_int {
        modules += 1;
        return 0;
    }
    fn event(_: cuda.abi.Event) callconv(.c) c_int {
        events += 1;
        return 0;
    }
    fn alloc(out: *u64, _: usize) callconv(.c) c_int {
        out.* = 0x1000;
        return 0;
    }
    fn buffer(_: u64) callconv(.c) c_int {
        buffers += 1;
        return 0;
    }
    fn createStream(out: *cuda.abi.Stream, _: c_uint) callconv(.c) c_int {
        out.* = null;
        return 0;
    }
    fn createEvent(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        out.* = null;
        return 0;
    }
    fn driver() cuda.Driver {
        streams = 0;
        modules = 0;
        events = 0;
        buffers = 0;
        var d: cuda.Driver = undefined;
        d.api.cuStreamCreate = createStream;
        d.api.cuEventCreate = createEvent;
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamDestroy_v2 = stream;
        d.api.cuModuleUnload = module;
        d.api.cuEventDestroy_v2 = event;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = buffer;
        return d;
    }
};

fn attach(o: *load.Ops) !*fork.Fork {
    const f = try o.gpa.create(fork.Fork);
    f.tmp_bytes = 64;
    f.ev_fork = .{ .d = o.d, .handle = null };
    for (0..fork.branches) |i| {
        f.side[i] = .{ .d = o.d, .handle = null };
        f.dense[i] = .{ .d = o.d, .lin = .{ .mod = .{ .d = o.d, .handle = null }, .fns = undefined, .rot_in = undefined } };
        f.tmp[i] = try cuda.DeviceBuffer.alloc(o.d, 64);
        f.ev_join[i] = .{ .d = o.d, .handle = null };
    }
    o.fork = f;
    return f;
}

test "exl3: projection forks retire with their Ops owner and preserve other owners" {
    var d = Fake.driver();
    const left = try load.Ops.create(std.testing.allocator, &d);
    const right = try load.Ops.create(std.testing.allocator, &d);
    const l = try attach(left);
    const r = try attach(right);
    _ = left.retain();
    left.release();
    try std.testing.expect(left.fork == l and right.fork == r);
    try std.testing.expectEqual(@as(usize, 0), Fake.modules);
    left.release();
    try std.testing.expect(right.fork == r);
    try std.testing.expectEqual(fork.branches, Fake.modules);
    right.release();
    try std.testing.expectEqual(2 * fork.branches, Fake.streams);
    try std.testing.expectEqual(2 * fork.branches, Fake.modules);
    try std.testing.expectEqual(2 * (fork.branches + 1), Fake.events);
    try std.testing.expectEqual(2 * fork.branches, Fake.buffers);
}

test "exl3: fork reservation failure releases the current branch before publication" {
    if (cuda.kernels.available) return error.SkipZigTest;
    var d = Fake.driver();
    const o = try load.Ops.create(std.testing.allocator, &d);
    defer o.release();
    o.max_k = 128;
    try std.testing.expectError(error.BuiltWithoutKernels, fork.reserve(o));
    try std.testing.expect(o.fork == null);
    try std.testing.expectEqual(@as(usize, 1), Fake.streams);
    try std.testing.expectEqual(@as(usize, 1), Fake.events);
    try std.testing.expectEqual(@as(usize, 0), Fake.modules);
}
