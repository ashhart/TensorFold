//! Resize only the drafter's uncaptured scratch before serving, retaining the old layout on any fallible failure.
const std = @import("std");
const cuda = @import("cuda");
const most = @import("cuda_draft_frame.zig").max_streams;

pub fn resize(d: anytype, streams: usize) !void {
    return withBuffer(d, streams, cuda.DeviceBuffer);
}

fn withBuffer(d: anytype, streams: usize, comptime Buffer: type) !void {
    if (streams == 0 or streams > most) return error.BadDraftFrame;
    if (streams == d.frame_streams) return;
    const previous = d.frame_streams;
    d.frame_streams = streams;
    const bytes = d.layout(false) catch |err| {
        d.frame_streams = previous;
        return err;
    };
    d.frame_streams = previous;
    var fresh = try Buffer.alloc(d.e.d, bytes);
    errdefer fresh.free();
    try fresh.fill8(0, null);
    try d.e.stream.synchronize();
    var old = d.scratch;
    d.scratch = fresh;
    d.frame_streams = streams;
    _ = d.layout(true) catch unreachable; // the identical arithmetic succeeded above; no allocations or launches here
    old.free();
}

const Fake = struct {
    var fault: enum { none, allocate, fill, drain } = .none;
    var freed: usize = 0;
    const Buffer = struct {
        ptr: u64,
        len: usize,
        pub fn alloc(_: void, bytes: usize) !Buffer {
            if (fault == .allocate) return error.OutOfMemory;
            return .{ .ptr = 200, .len = bytes };
        }
        pub fn fill8(_: Buffer, _: u8, _: ?*anyopaque) !void {
            if (fault == .fill) return error.FillFailed;
        }
        pub fn free(_: *Buffer) void {
            freed += 1;
        }
    };
    const Stream = struct {
        pub fn synchronize(_: Stream) !void {
            if (fault == .drain) return error.DrainFailed;
        }
    };
    const Engine = struct { d: void = {}, stream: Stream = .{} };
    e: *Engine,
    frame_streams: usize = 1,
    scratch: Buffer = .{ .ptr = 100, .len = 16 },
    view: u64 = 100,
    pub fn layout(d: *Fake, assign: bool) !usize {
        if (assign) d.view = d.scratch.ptr;
        return d.frame_streams * 16;
    }
};

test "served scratch: allocation, fill and drain failures preserve the prior pointers and capacity" {
    for ([_]@TypeOf(Fake.fault){ .allocate, .fill, .drain, .none }) |fault| {
        Fake.fault = fault;
        Fake.freed = 0;
        var engine: Fake.Engine = .{};
        var d: Fake = .{ .e = &engine };
        const result = withBuffer(&d, 2, Fake.Buffer);
        if (fault == .none) {
            try result;
            try std.testing.expectEqual(@as(usize, 2), d.frame_streams);
            try std.testing.expectEqual(@as(u64, 200), d.view);
            try std.testing.expectEqual(@as(usize, 32), d.scratch.len);
        } else {
            if (fault == .allocate) try std.testing.expectError(error.OutOfMemory, result);
            if (fault == .fill) try std.testing.expectError(error.FillFailed, result);
            if (fault == .drain) try std.testing.expectError(error.DrainFailed, result);
            try std.testing.expectEqual(@as(usize, 1), d.frame_streams);
            try std.testing.expectEqual(@as(u64, 100), d.view);
            try std.testing.expectEqual(@as(usize, 16), d.scratch.len);
        }
        try std.testing.expectEqual(@as(usize, @intFromBool(fault != .allocate)), Fake.freed);
    }
}
