//! Actual inline header/ID transport over recording CUDA and NCCL calls, without devices.
const std = @import("std");
const cuda = @import("cuda");
const wire = @import("glm").follow;
const abi = cuda.abi;
const t = std.testing;
const Params = ?[*]?*anyopaque;
const Recording = struct {
    var gpu: [384]u8 align(8) = undefined;
    var frames: [2][384]u8 = undefined;
    var lengths: [2]usize = undefined;
    var sent: usize = 0;
    var received: usize = 0;
    var receiving = false;
    var launches: usize = 0;
    var downloads: usize = 0;
    var uploads: usize = 0;
    var uploaded_offsets: [2]usize = undefined;
    var uploaded_lengths: [2]usize = undefined;
    var pinned: [4]?[]align(16) u8 = @splat(null);
    fn get(comptime T: type, params: Params, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(params.?[i].?))).*;
    }
    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, params: Params, _: Params) callconv(.c) abi.Result {
        const words = get(wire.Words, params, 2);
        const n = get(u32, params, 1);
        @memcpy(gpu[0 .. n * 4], std.mem.sliceAsBytes(words.w[0..n]));
        launches += 1;
        return 0;
    }
    fn broadcast(_: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: abi.Stream) callconv(.c) cuda.nccl.Result {
        if (receiving) {
            if (received >= sent or lengths[received] != n) return 1;
            @memcpy(gpu[0..n], frames[received][0..n]);
            received += 1;
        } else {
            if (sent == frames.len) return 1;
            @memcpy(frames[sent][0..n], gpu[0..n]);
            lengths[sent] = n;
            sent += 1;
        }
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) abi.Result {
        const out: [*]u8 = @ptrCast(dst.?);
        @memcpy(out[0..n], gpu[@intCast(src - 4096)..][0..n]);
        downloads += 1;
        return 0;
    }
    fn upload(dst: u64, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (uploads >= uploaded_offsets.len) return 1;
        const at: usize = @intCast(dst - 4096);
        @memcpy(gpu[at..][0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        uploaded_offsets[uploads] = at;
        uploaded_lengths[uploads] = n;
        uploads += 1;
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
        for (&pinned) |*slot| if (slot.* == null) {
            const bytes = t.allocator.alignedAlloc(u8, .@"16", n) catch return 1;
            slot.* = bytes;
            out.* = bytes.ptr;
            return 0;
        };
        return 1;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (&pinned) |*slot| if (slot.*) |bytes| {
            if (@intFromPtr(bytes.ptr) != @intFromPtr(ptr.?)) continue;
            t.allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        return 1;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn event(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn device(dev: *abi.Device) callconv(.c) abi.Result {
        dev.* = 0;
        return 0;
    }
    fn attribute(value: *c_int, _: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
        value.* = 0;
        return 0;
    }
};

test "GLM link transport: waited uploads and unwaited puts carry zero, 48 and 49 ids byte for byte" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuMemcpyHtoDAsync_v2 = Recording.upload;
    d.api.cuMemHostAlloc = Recording.hostAlloc;
    d.api.cuMemFreeHost = Recording.hostFree;
    d.api.cuEventCreate = Recording.eventCreate;
    d.api.cuEventRecord = Recording.eventRecord;
    d.api.cuEventQuery = Recording.event;
    d.api.cuEventSynchronize = Recording.event;
    d.api.cuEventDestroy_v2 = Recording.event;
    d.api.cuMemcpyDtoH_v2 = Recording.download;
    d.api.cuStreamSynchronize = Recording.sync;
    d.api.cuCtxGetDevice = Recording.device;
    d.api.cuDeviceGetAttribute = Recording.attribute;
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Recording.broadcast;
    var ids: [80]u32 = undefined;
    var got: [80]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + 17 * i);
    var link: wire.Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = .{ .d = &d, .handle = null }, .buf = .{ .d = &d, .ptr = 4096, .len = Recording.gpu.len, .borrowed = true }, .ids = &got, .gpa = t.allocator, .wait = false };
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();
    defer {
        cuda.copy_source.synchronizeStream(&d, null) catch {};
        cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    }
    for ([_]bool{ false, true }) |wait| for ([_]usize{ 0, 48, 49 }) |n| {
        link.wait = wait;
        Recording.sent = 0;
        Recording.received = 0;
        Recording.receiving = false;
        Recording.launches = 0;
        Recording.downloads = 0;
        Recording.uploads = 0;
        @memset(&Recording.gpu, 0xa5);
        @memset(&got, 0xdeadbeef);
        const h: wire.Header = .{ .kind = @backingInt(wire.Kind.draft), .depth = 7, .slot = 3, .seed = 123 };
        try link.send(.{ .d = &d, .handle = null }, h, ids[0..n]);
        try t.expectEqual(@as(usize, if (wait) 0 else 1), Recording.launches);
        try t.expectEqual(@as(usize, if (wait) (if (n == 0) 1 else 2) else 0), Recording.uploads);
        if (wait) {
            try t.expectEqual(@as(usize, 0), Recording.uploaded_offsets[0]);
            try t.expectEqual(@as(usize, 64), Recording.uploaded_lengths[0]);
            if (n > 0) {
                try t.expectEqual(@as(usize, 64), Recording.uploaded_offsets[1]);
                try t.expectEqual(n * 4, Recording.uploaded_lengths[1]);
            }
        }
        try t.expectEqual(@as(usize, if (n > 48) 2 else 1), Recording.sent);
        try t.expectEqual(@as(usize, 256), Recording.lengths[0]);
        if (n > 48) try t.expectEqual(@as(usize, 64 + n * 4), Recording.lengths[1]);
        Recording.receiving = true;
        @memset(&Recording.gpu, 0x5a);
        const message = try link.receive();
        var want = h;
        want.n = @intCast(n);
        try t.expectEqualDeep(want, message.h);
        try t.expectEqualSlices(u32, ids[0..n], message.ids);
        try t.expectEqual(@as(u32, 0xdeadbeef), got[n]);
        try t.expectEqual(Recording.sent, Recording.received);
        try t.expectEqual(@as(usize, if (n > 48) 2 else 1), Recording.downloads);
    };
}
