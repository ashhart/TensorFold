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
    var stream_syncs: usize = 0;
    var in_flight = false;
    var upload_sources: [2]?*const anyopaque = @splat(null);
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
        upload_sources[uploads] = src;
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
    fn query(_: abi.Event) callconv(.c) abi.Result {
        return if (in_flight) abi.error_not_ready else 0;
    }
    fn event(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        stream_syncs += 1;
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

fn makeDriver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuLaunchKernelEx = @import("glm").pdl.forwardEx(Recording.launch);
    d.func_name = null;
    d.api.cuMemcpyHtoDAsync_v2 = Recording.upload;
    d.api.cuMemHostAlloc = Recording.hostAlloc;
    d.api.cuMemFreeHost = Recording.hostFree;
    d.api.cuEventCreate = Recording.eventCreate;
    d.api.cuEventRecord = Recording.eventRecord;
    d.api.cuEventQuery = Recording.query;
    d.api.cuEventSynchronize = Recording.event;
    d.api.cuEventDestroy_v2 = Recording.event;
    d.api.cuMemcpyDtoH_v2 = Recording.download;
    d.api.cuStreamSynchronize = Recording.sync;
    d.api.cuCtxGetDevice = Recording.device;
    d.api.cuDeviceGetAttribute = Recording.attribute;
    return d;
}

fn makeLibrary() cuda.nccl.Library {
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Recording.broadcast;
    return lib;
}

fn clearCopies(d: *const cuda.Driver) void {
    cuda.copy_source.synchronizeStream(d, null) catch {};
    cuda.copy_source.dropSpares(d, cuda.copy_source.current());
}

fn socketPair() ![2]@import("sys").posix.socket_t {
    var pair: [2]@import("sys").posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    return pair;
}

test "GLM link transport: waited uploads and unwaited puts carry zero, 48 and 49 ids byte for byte" {
    var d = makeDriver();
    var lib = makeLibrary();
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

fn hostLink(d: *cuda.Driver, lib: *cuda.nccl.Library, fd: @import("sys").posix.socket_t, ids: []u32, staging: []u64) wire.Link {
    return .{ .d = d, .lib = lib, .comm = null, .stream = .{ .d = d, .handle = null }, .buf = .{ .d = d, .ptr = 4096, .len = Recording.gpu.len, .borrowed = true }, .ids = ids, .wire = staging, .gpa = t.allocator, .wait = false, .control = .{ .fd = fd, .max_bytes = Recording.gpu.len } };
}

test "GLM host control: the same follower header and ids with no NCCL, download, put or stream drain" {
    var d = makeDriver();
    var lib = makeLibrary();
    const pair = try socketPair();
    defer {
        for (pair) |fd| _ = std.c.close(fd);
    }
    var ids: [80]u32 = undefined;
    var got: [80]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + 17 * i);
    var leader_wire: [48]u64 = undefined;
    var follower_wire: [48]u64 = undefined;
    var sender = hostLink(&d, &lib, pair[0], &ids, &leader_wire);
    var receiver = hostLink(&d, &lib, pair[1], &got, &follower_wire);
    defer clearCopies(&d);
    for ([_]usize{ 0, 48, 49 }) |n| {
        Recording.sent = 0;
        Recording.received = 0;
        Recording.launches = 0;
        Recording.downloads = 0;
        Recording.uploads = 0;
        Recording.stream_syncs = 0;
        Recording.in_flight = false;
        @memset(&got, 0xdeadbeef);
        const h: wire.Header = .{ .kind = @backingInt(wire.Kind.draft), .depth = 7, .slot = 3, .seed = 123 };
        try sender.send(.{ .d = &d, .handle = null }, h, ids[0..n]);
        const message = try receiver.receive();
        var want = h;
        want.n = @intCast(n);
        try t.expectEqualDeep(want, message.h);
        try t.expectEqualSlices(u32, ids[0..n], message.ids);
        try t.expectEqual(@as(u32, 0xdeadbeef), got[n]);
        try t.expectEqual(@as(usize, 0), Recording.launches);
        try t.expectEqual(@as(usize, 0), Recording.sent);
        try t.expectEqual(@as(usize, 0), Recording.received);
        try t.expectEqual(@as(usize, 0), Recording.downloads);
        try t.expectEqual(@as(usize, 0), Recording.stream_syncs);
        try t.expectEqual(@as(usize, 1), Recording.uploads);
        try t.expectEqual(@as(usize, 0), Recording.uploaded_offsets[0]);
        try t.expectEqual(64 + n * 4, Recording.uploaded_lengths[0]);
        clearCopies(&d);
        Recording.stream_syncs = 0;
    }
    Recording.uploads = 0;
    const malformed: wire.Header = .{ .kind = @backingInt(wire.Kind.draft), .n = 1 };
    try sender.control.?.send(std.mem.asBytes(&malformed), &.{});
    try t.expectError(error.GlmControlFrame, receiver.receive());
    try t.expectEqual(@as(@import("sys").posix.socket_t, -1), receiver.control.?.fd);
    try t.expectEqual(@as(usize, 0), Recording.uploads);
}

test "GLM host control: a pending upload's pinned source is not overwritten by the next frame" {
    var d = makeDriver();
    var lib = makeLibrary();
    const pair = try socketPair();
    defer {
        for (pair) |fd| _ = std.c.close(fd);
    }
    defer {
        Recording.in_flight = false;
        clearCopies(&d);
    }
    var first_ids = [_]u32{ 11, 12, 13 };
    var second_ids = [_]u32{ 21, 22, 23, 24 };
    var sent: [8]u32 = undefined;
    var got: [8]u32 = undefined;
    var leader_wire: [48]u64 = undefined;
    var follower_wire: [48]u64 = undefined;
    var sender = hostLink(&d, &lib, pair[0], &sent, &leader_wire);
    var receiver = hostLink(&d, &lib, pair[1], &got, &follower_wire);
    const h: wire.Header = .{ .kind = @backingInt(wire.Kind.verify), .slot = 4 };
    var hh = h;
    hh.n = first_ids.len;
    var expected: [80]u8 align(8) = @splat(0);
    @memcpy(expected[0..64], std.mem.asBytes(&hh));
    @memcpy(expected[64..76], std.mem.sliceAsBytes(&first_ids));
    Recording.in_flight = true;
    Recording.uploads = 0;
    try sender.send(.{ .d = &d, .handle = null }, h, &first_ids);
    _ = try receiver.receive();
    try t.expectEqual(@as(usize, 1), Recording.uploads);
    const first_source = Recording.upload_sources[0].?;
    Recording.uploads = 0;
    try sender.send(.{ .d = &d, .handle = null }, h, &second_ids);
    _ = try receiver.receive();
    try t.expectEqual(@as(usize, 1), Recording.uploads);
    const second_source = Recording.upload_sources[0].?;
    try t.expect(first_source != second_source);
    const first_bytes = @as([*]const u8, @ptrCast(first_source))[0..76];
    try t.expectEqualSlices(u8, expected[0..76], first_bytes);
    try t.expectEqualSlices(u32, &second_ids, got[0..second_ids.len]);
}

test "GLM host control: frames preserve FIFO, a late peer arrives whole, and gaps or loss refuse" {
    const posix = @import("sys").posix;
    const pair = try socketPair();
    defer {
        for (pair) |fd| _ = std.c.close(fd);
    }
    var sender: cuda.tp_control.Channel = .{ .fd = pair[0], .max_bytes = 384 };
    var receiver: cuda.tp_control.Channel = .{ .fd = pair[1], .max_bytes = 384 };
    const first: [64]u8 = @splat(1);
    const second: [64]u8 = @splat(2);
    const ids = [_]u32{ 7, 8, 9 };
    var out: [384]u8 = undefined;
    try sender.send(&first, std.mem.sliceAsBytes(&ids));
    try sender.send(&second, &.{});
    var n = try receiver.receive(&out);
    try t.expectEqual(@as(usize, 76), n);
    try t.expectEqualSlices(u8, &first, out[0..64]);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&ids), out[64..76]);
    n = try receiver.receive(&out);
    try t.expectEqual(@as(usize, 64), n);
    try t.expectEqualSlices(u8, &second, out[0..64]);

    const Late = struct {
        channel: *cuda.tp_control.Channel,
        header: [64]u8,
        ids: [3]u32,
        err: ?anyerror = null,
        fn run(x: *@This()) void {
            std.Io.sleep(t.io, .fromMilliseconds(20), .awake) catch {};
            x.channel.send(&x.header, std.mem.sliceAsBytes(&x.ids)) catch |e| {
                x.err = e;
            };
        }
    };
    var late: Late = .{ .channel = &sender, .header = first, .ids = ids };
    const thread = try std.Thread.spawn(.{}, Late.run, .{&late});
    defer thread.join();
    n = try receiver.receive(&out);
    try t.expectEqual(@as(usize, 76), n);
    try t.expect(late.err == null);

    const bad_pair = try socketPair();
    defer {
        for (bad_pair) |fd| _ = std.c.close(fd);
    }
    var early: cuda.tp_control.Channel = .{ .fd = bad_pair[0], .max_bytes = 384 };
    var strict: cuda.tp_control.Channel = .{ .fd = bad_pair[1], .max_bytes = 384 };
    early.next = 2;
    try early.send(&first, &.{});
    try t.expectError(error.GlmControlOrder, strict.receive(&out));

    const lost = try socketPair();
    defer _ = std.c.close(lost[0]);
    var closed: cuda.tp_control.Channel = .{ .fd = lost[0], .max_bytes = 384 };
    _ = std.c.close(lost[1]);
    try t.expectError(error.GlmControlPeerClosed, closed.receive(&out));

    const failed = try socketPair();
    defer _ = std.c.close(failed[0]);
    var writer: cuda.tp_control.Channel = .{ .fd = failed[0], .max_bytes = 384 };
    _ = std.c.close(failed[1]);
    var refused = false;
    for (0..4) |_| {
        writer.send(&first, &.{}) catch {
            refused = true;
            break;
        };
    }
    try t.expect(refused);
    try t.expectError(error.GlmControlFrame, sender.send(first[0..63], &.{}));
    const p: cuda.tp_control.Prefix = .{ .magic = 0, .bytes = 64, .sequence = sender.next };
    const flags: u32 = if (@hasDecl(posix.MSG, "NOSIGNAL")) posix.MSG.NOSIGNAL else 0;
    try t.expectEqual(@as(isize, 16), std.c.send(pair[0], std.mem.asBytes(&p).ptr, 16, flags));
    try t.expectError(error.GlmControlFrame, receiver.receive(&out));
}
