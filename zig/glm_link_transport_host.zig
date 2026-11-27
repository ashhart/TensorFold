//! Actual inline header/ID transport over recording CUDA and NCCL calls, without devices.
const std = @import("std");
const cuda = @import("cuda");
const wire = @import("glm").follow;
const abi = cuda.abi;
const t = std.testing;
const Params = ?[*]?*anyopaque;
const Recording = struct {
    var gpu: [384]u8 align(8) = undefined;
    var frames: [4][384]u8 = undefined;
    var lengths: [4]usize = undefined;
    var sent: usize = 0;
    var received: usize = 0;
    var receiving = false;
    var launches: usize = 0;
    var downloads: usize = 0;
    var uploads: usize = 0;
    var uploaded_offsets: [2]usize = undefined;
    var launch_stream: abi.Stream = null;
    var broadcast_stream: abi.Stream = null;
    var sync_stream: abi.Stream = null;
    var uploaded_lengths: [2]usize = undefined;
    var pinned: [4]?[]align(16) u8 = @splat(null);
    fn get(comptime T: type, params: Params, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(params.?[i].?))).*;
    }
    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, stream: abi.Stream, params: Params, _: Params) callconv(.c) abi.Result {
        const words = get(wire.Words, params, 2);
        const n = get(u32, params, 1);
        @memcpy(gpu[0 .. n * 4], std.mem.sliceAsBytes(words.w[0..n]));
        launches += 1;
        launch_stream = stream;
        return 0;
    }
    fn broadcast(ptr: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, stream: abi.Stream) callconv(.c) cuda.nccl.Result {
        const offset = ptr - 4096;
        if (receiving) {
            if (received >= sent or lengths[received] != n) return 1;
            @memcpy(gpu[offset..][0..n], frames[received][0..n]);
            received += 1;
        } else {
            if (sent == frames.len) return 1;
            @memcpy(frames[sent][0..n], gpu[offset..][0..n]);
            lengths[sent] = n;
            sent += 1;
        }
        broadcast_stream = stream;
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
    fn sync(stream: abi.Stream) callconv(.c) abi.Result {
        sync_stream = stream;
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
    d.api.cuLaunchKernelEx = @import("glm").pdl.forwardEx(Recording.launch);
    d.func_name = null;
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

test "GLM control fold: default stays separate; folded two-rank steps and tensor routes stay exact" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuLaunchKernelEx = @import("glm").pdl.forwardEx(Recording.launch);
    d.func_name = null;
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
    var got: [80]u32 = undefined;
    var link: wire.Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = .{ .d = &d, .handle = null }, .buf = .{ .d = &d, .ptr = 4096, .len = Recording.gpu.len, .borrowed = true }, .ids = &got, .gpa = t.allocator, .wait = false };
    const control_handle: abi.Stream = @ptrFromInt(8);
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();
    defer {
        cuda.copy_source.synchronizeStream(&d, null) catch {};
        cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    }
    Recording.sent = 0;
    Recording.received = 0;
    Recording.receiving = false;
    Recording.launches = 0;
    try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.verify), .slot = 3 }, &.{ 11, 12 });
    try link.keep(.{ .d = &d, .handle = null }, 3, 2);
    try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.draft), .slot = 3, .depth = 2 }, &.{ 11, 12 });
    try t.expectEqual(@as(usize, 3), Recording.sent); // default off: verify, keep and next control as three compute-stream broadcasts
    try t.expectEqual(@as(abi.Stream, null), Recording.broadcast_stream);

    link.fold = true;
    link.control_buf = link.buf;
    link.control_stream = .{ .d = &d, .handle = control_handle };
    Recording.sent = 0;
    Recording.received = 0;
    Recording.receiving = false;
    try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.verify), .slot = 3 }, &.{ 11, 12 });
    try link.keep(.{ .d = &d, .handle = null }, 3, 2); // the next draft carries this slot and count
    try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.draft), .slot = 3, .depth = 2 }, &.{ 11, 12 });
    try t.expectEqual(@as(usize, 2), Recording.sent);
    try t.expectEqual(control_handle, Recording.broadcast_stream);
    try t.expectEqual(control_handle, Recording.launch_stream);
    Recording.receiving = true;
    const verify = try link.receive();
    try t.expectEqual(wire.Kind.verify, @as(wire.Kind, @fromBackingInt(@intCast(verify.h.kind))));
    try t.expect(link.received_keep == null);
    try t.expectEqualSlices(u32, &.{ 11, 12 }, verify.ids);
    const draft = try link.receive();
    try t.expectEqual(wire.Kind.draft, @as(wire.Kind, @fromBackingInt(@intCast(draft.h.kind))));
    try t.expectEqual(@as(u64, 2), link.received_keep.?.rows);
    try t.expectEqual(@as(u32, 3), link.received_keep.?.slot); // follower commits this slot before binding draft.slot
    try t.expectEqualSlices(u32, &.{ 11, 12 }, draft.ids);
    try t.expect(link.pending_keep == null);
    try t.expectEqual(control_handle, Recording.sync_stream);

    var boundary_ids: [80]u32 = undefined;
    for (&boundary_ids, 0..) |*id, i| id.* = @intCast(i + 100);
    for ([_]bool{ false, true }) |wait| for ([_]usize{ 44, 45 }) |n| {
        link.wait = wait;
        Recording.sent = 0;
        Recording.received = 0;
        Recording.uploads = 0;
        Recording.downloads = 0;
        Recording.launches = 0;
        Recording.receiving = false;
        link.pending_keep = .{ .slot = 3, .rows = 3 };
        try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.draft), .slot = 3, .depth = 2 }, boundary_ids[0..n]);
        try t.expectEqual(@as(usize, if (wait) 2 else 0), Recording.uploads);
        try t.expectEqual(@as(usize, if (wait) 0 else 1), Recording.launches);
        if (n > 44) try t.expectEqual(n * 4, Recording.lengths[1]);
        Recording.receiving = true;
        const boundary = try link.receive();
        try t.expectEqual(@as(u64, 3), link.received_keep.?.rows);
        try t.expectEqual(@as(u32, 3), link.received_keep.?.slot);
        try t.expectEqualSlices(u32, boundary_ids[0..n], boundary.ids);
        try t.expectEqual(@as(usize, if (n > 44) 2 else 1), Recording.downloads);
    };
    link.wait = false;

    // A follower that catches Cancelled receives the next control with no stale folded keep.
    Recording.sent = 0;
    Recording.received = 0;
    Recording.receiving = false;
    link.pending_keep = .{ .slot = 3, .rows = 5 };
    try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.prefill), .slot = 3 }, &.{21});
    try link.send(.{ .d = &d, .handle = null }, .{ .kind = @backingInt(wire.Kind.idle), .slot = 3 }, &.{});
    Recording.receiving = true;
    const cancelled = try link.receive();
    try t.expectEqual(wire.Kind.prefill, @as(wire.Kind, @fromBackingInt(@intCast(cancelled.h.kind))));
    try t.expectEqual(@as(u64, 5), link.received_keep.?.rows);
    try t.expectEqual(@as(u32, 3), link.received_keep.?.slot);
    _ = try link.receive();
    try t.expect(link.received_keep == null);

    // Tensor rows and disk takes never move to the control stream or its buffer path.
    Recording.receiving = false;
    try link.rows(4096, 8);
    try t.expectEqual(@as(abi.Stream, null), Recording.broadcast_stream);
    try link.from(0, 8);
    try t.expectEqual(@as(abi.Stream, null), Recording.broadcast_stream);
    try t.expectEqual(@as(abi.Stream, null), Recording.sync_stream);

    // A refused shared window emits nothing; its pending keep survives to the final end, without a trailing keep frame.
    var e: @import("glm").engine.Engine = undefined;
    e.dropped = .empty;
    e.slot = 3;
    e.m.glue.put_words = .{ .d = &d, .handle = null };
    const leader: wire.Leader = .{ .e = &e, .link = &link };
    Recording.sent = 0;
    Recording.received = 0;
    Recording.receiving = false;
    link.pending_keep = .{ .slot = 3, .rows = 9 };
    try t.expectError(error.WindowTooWide, leader.verifyShared(&.{}));
    try t.expectEqual(@as(usize, 0), Recording.sent);
    try t.expectEqual(@as(u64, 9), link.pending_keep.?.rows);
    try t.expectEqual(@as(u32, 3), link.pending_keep.?.slot);
    try leader.end();
    try t.expectEqual(@as(usize, 1), Recording.sent);
    try t.expect(link.pending_keep == null);
    Recording.receiving = true;
    const last = try link.receive();
    try t.expectEqual(wire.Kind.end, @as(wire.Kind, @fromBackingInt(@intCast(last.h.kind))));
    try t.expectEqual(@as(u64, 9), link.received_keep.?.rows);
    try t.expectEqual(@as(u32, 3), link.received_keep.?.slot);

    // Adjacent shared-window keeps: the fallback keeps slot 3; end carries slot 4 and the newest count.
    Recording.sent = 0;
    Recording.received = 0;
    Recording.receiving = false;
    try link.keep(.{ .d = &d, .handle = null }, 3, 4);
    try t.expectEqual(@as(usize, 0), Recording.sent);
    try link.keep(.{ .d = &d, .handle = null }, 4, 6);
    try t.expectEqual(@as(usize, 1), Recording.sent);
    try t.expectEqual(@as(u32, 4), link.pending_keep.?.slot);
    try t.expectEqual(@as(u64, 6), link.pending_keep.?.rows);
    e.slot = 4;
    try leader.end();
    try t.expectEqual(@as(usize, 2), Recording.sent);
    Recording.receiving = true;
    const fallback = try link.receive();
    try t.expectEqual(wire.Kind.keep, @as(wire.Kind, @fromBackingInt(@intCast(fallback.h.kind))));
    try t.expectEqual(@as(u32, 3), fallback.h.slot);
    try t.expectEqual(@as(u64, 4), fallback.h.a);
    try t.expect(link.received_keep == null);
    const adjacent_end = try link.receive();
    try t.expectEqual(wire.Kind.end, @as(wire.Kind, @fromBackingInt(@intCast(adjacent_end.h.kind))));
    try t.expectEqual(@as(u32, 4), link.received_keep.?.slot);
    try t.expectEqual(@as(u64, 6), link.received_keep.?.rows);
}

test "TF_GLM_CONTROL_FOLD: strict default and rank agreement" {
    const Reader = struct {
        fn unset(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_CONTROL_FOLD")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_CONTROL_FOLD")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_CONTROL_FOLD")) "yes" else null;
        }
    };
    const ranks = @import("glm").ranks;
    var problem: []const u8 = "";
    try t.expect(!(try ranks.Settings.read(Reader.unset, &problem)).control_fold);
    try t.expect((try ranks.Settings.read(Reader.on, &problem)).control_fold);
    try t.expect(!(try ranks.Settings.read(Reader.off, &problem)).control_fold);
    try t.expectError(error.BadSetting, ranks.Settings.read(Reader.bad, &problem));
    try t.expectEqualStrings("TF_GLM_CONTROL_FOLD: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]ranks.Fingerprint{ ranks.Fingerprint.of(2, 0, .{}, 8192, true, h, h, h), ranks.Fingerprint.of(2, 1, .{ .control_fold = true }, 8192, true, h, h, h) };
    try t.expectEqual(@as(?ranks.Mismatch, .{ .field = .control_fold, .rank = 1 }), ranks.mismatch(&all));
}
