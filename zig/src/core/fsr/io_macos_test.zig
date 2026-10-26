const std = @import("std");
const io = @import("io_macos.zig");
const reader = @import("reader.zig");
extern "c" fn tf_fsr_aio_abi(c_uint) u64;

test "FSR macOS asynchronous IO ABI matches the actual system headers" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    const C = io.Darwin.Control;
    const values = [_]u64{ @sizeOf(C), @alignOf(C), @offsetOf(C, "fd"), @offsetOf(C, "offset"), @offsetOf(C, "buffer"), @offsetOf(C, "bytes"), @offsetOf(C, "priority"), @offsetOf(C, "event"), @offsetOf(C, "opcode"), @sizeOf(io.Darwin.Event), 0 };
    for (values, 0..) |expected, field| try std.testing.expectEqual(expected, tf_fsr_aio_abi(@intCast(field)));
}
const Mock = struct {
    pub const Control = struct { request: reader.Request, id: usize, ready: bool = false };
    var prepared: usize = 0;
    var submitted: usize = 0;
    var finished: usize = 0;
    var fallback: usize = 0;
    var fail_submit: usize = 99;
    var fail_result: usize = 99;
    var short_first: bool = false;
    var require_drained: bool = false;
    var all_queued_before_wait: bool = true;
    const data = "abcdefghijklmnopqrstuvwxyz0123456789";

    fn reset() void {
        prepared = 0;
        submitted = 0;
        finished = 0;
        fallback = 0;
        fail_submit = 99;
        fail_result = 99;
        short_first = false;
        require_drained = false;
        all_queued_before_wait = true;
    }
    pub fn prepare(c: *Control, fd: c_int, request: reader.Request) void {
        std.debug.assert(fd == 7);
        c.* = .{ .request = request, .id = prepared };
        prepared += 1;
    }
    pub fn submit(_: *Control) bool {
        if (submitted == fail_submit) return false;
        submitted += 1;
        return true;
    }
    pub fn status(c: *Control) c_int {
        if (!c.ready) return @backingInt(std.posix.E.INPROGRESS);
        return if (c.id == fail_result) @backingInt(std.posix.E.IO) else 0;
    }
    pub fn wait(c: *Control) void {
        if (submitted != @min(9, fail_submit)) all_queued_before_wait = false;
        c.ready = true;
    }
    pub fn result(c: *Control) isize {
        std.debug.assert(c.ready);
        finished += 1;
        if (c.id == fail_result) return -1;
        const n = c.request.destination.len - @as(usize, if (short_first and c.id == 0) 2 else 0);
        const at: usize = @intCast(c.request.span.offset);
        @memcpy(c.request.destination[0..n], data[at..][0..n]);
        return @intCast(n);
    }
    pub const File = struct {
        fd: c_int = 7,
        pub fn readAt(_: @This(), _: usize, dst: []u8, offset: u64) !usize {
            if (require_drained and finished != submitted) return error.FallbackBeforeDrain;
            fallback += 1;
            @memcpy(dst, data[@intCast(offset)..][0..dst.len]);
            return dst.len;
        }
    };
};

fn requests(memory: *[9][4]u8) [9]reader.Request {
    var out: [9]reader.Request = undefined;
    for (&out, memory, 0..) |*request, *dst, i|
        request.* = .{ .span = .{ .file = 0, .offset = i * 4, .length = 4 }, .destination = dst };
    return out;
}

test "FSR IO queues the bounded component group before waiting and restores short reads" {
    Mock.reset();
    Mock.short_first = true;
    var memory: [9][4]u8 = undefined;
    const work = requests(&memory);
    var stats: reader.Stats = .{};
    try io.group(Mock, @as([]const Mock.File, &.{Mock.File{}}), &work, &stats);
    try std.testing.expect(Mock.all_queued_before_wait);
    try std.testing.expectEqual(@as(usize, 9), Mock.finished);
    try std.testing.expectEqual(@as(usize, 1), Mock.fallback);
    try std.testing.expectEqual(@as(u64, 10), stats.calls);
    try std.testing.expectEqual(@as(u64, 36), stats.bytes);
    for (memory, 0..) |bytes, i| try std.testing.expectEqualStrings(Mock.data[i * 4 ..][0..4], &bytes);
}

test "FSR IO resource fallback and failures drain every submitted read before return" {
    var memory: [9][4]u8 = undefined;
    const work = requests(&memory);
    Mock.reset();
    Mock.fail_submit = 2;
    Mock.require_drained = true;
    var stats: reader.Stats = .{};
    try io.group(Mock, @as([]const Mock.File, &.{Mock.File{}}), &work, &stats);
    try std.testing.expectEqual(@as(usize, 2), Mock.finished);
    try std.testing.expectEqual(@as(usize, 7), Mock.fallback);
    for (memory, 0..) |bytes, i| try std.testing.expectEqualStrings(Mock.data[i * 4 ..][0..4], &bytes);
    Mock.reset();
    Mock.fail_result = 0;
    stats = .{};
    try std.testing.expectError(error.ReadFailed, io.group(Mock, @as([]const Mock.File, &.{Mock.File{}}), &work, &stats));
    try std.testing.expectEqual(@as(usize, 9), Mock.finished);
    try std.testing.expectEqual(@as(usize, 0), Mock.fallback);
}

test "FSR IO validates the entire group before submitting its first read" {
    Mock.reset();
    var memory: [9][4]u8 = undefined;
    var work = requests(&memory);
    work[8].span.file = 1;
    var stats: reader.Stats = .{};
    try std.testing.expectError(error.NoFile, io.group(Mock, @as([]const Mock.File, &.{Mock.File{}}), &work, &stats));
    try std.testing.expectEqual(@as(usize, 0), Mock.submitted);
}
