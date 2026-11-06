//! Submit an owned keep before publishing its completed tap rows; the caller owns every GPU resource.
const std = @import("std");
pub fn prefix(gathered: []const u32, path: []const u32) bool {
    return path.len > 0 and path.len <= gathered.len and std.mem.eql(u32, path, gathered[0..path.len]);
}
/// Shared logits are complete at the existing verify wait; keep writes only cache state on the ready path.
pub fn restoreHead(logits: []u16, width: usize, row: u32) !void {
    if (width == 0 or logits.len % width != 0 or row >= logits.len / width) return error.BadHeadRow;
    if (row == 0) return;
    @memcpy(logits[0..width], logits[@as(usize, row) * width ..][0..width]);
}
pub fn run(c: anytype, ready: bool) !void {
    const cb = try c.encode(ready);
    cb.commit();
    if (ready) c.hold(cb) else {
        cb.wait();
        try c.check(cb);
    }
    c.publish() catch |err| {
        if (ready) c.drain() catch {};
        return err;
    };
}
const Recording = struct {
    waits: usize = 0,
    commits: usize = 0,
    retained: bool = false,
    published: bool = false,
    drained: usize = 0,
    fail_check: bool = false,
    fail_publish: bool = false,
    const Buffer = struct {
        r: *Recording,
        pub fn commit(b: Buffer) void {
            b.r.commits += 1;
        }
        pub fn wait(b: Buffer) void {
            b.r.waits += 1;
        }
    };
    pub fn encode(r: *Recording, _: bool) !Buffer {
        return .{ .r = r };
    }
    pub fn check(r: *Recording, _: Buffer) !void {
        if (r.fail_check) return error.GpuFailed;
    }
    pub fn hold(r: *Recording, _: Buffer) void {
        r.retained = true;
    }
    pub fn publish(r: *Recording) !void {
        if (r.fail_publish) return error.BadTaps;
        r.published = true;
    }
    pub fn drain(r: *Recording) !void {
        r.drained += 1;
        r.retained = false;
    }
};
test "qwen27 pass: prefetched taps publish with no blocking keep wait before the next drafter" {
    var r: Recording = .{};
    try run(&r, true);
    try std.testing.expectEqual(@as(usize, 1), r.commits);
    try std.testing.expectEqual(@as(usize, 0), r.waits);
    try std.testing.expect(r.retained and r.published);
    try r.drain();
    try std.testing.expect(!r.retained);
}
test "qwen27 pass: a non-prefetched keep waits and refuses publication on failure" {
    var r: Recording = .{ .fail_check = true };
    try std.testing.expectError(error.GpuFailed, run(&r, false));
    try std.testing.expectEqual(@as(usize, 1), r.waits);
    try std.testing.expect(!r.published and !r.retained);
}
test "qwen27 pass: publication failure drains the submitted asynchronous keep" {
    var r: Recording = .{ .fail_publish = true };
    try std.testing.expectError(error.BadTaps, run(&r, true));
    try std.testing.expectEqual(@as(usize, 1), r.drained);
    try std.testing.expect(!r.retained and !r.published);
}
test "qwen27 pass: the core may keep a prefix, but a changed path needs its own tap gather" {
    try std.testing.expect(prefix(&.{ 0, 2, 7 }, &.{ 0, 2 }));
    try std.testing.expect(prefix(&.{ 0, 2, 7 }, &.{ 0, 2, 7 }));
    try std.testing.expect(!prefix(&.{ 0, 2, 7 }, &.{ 0, 3 }));
    try std.testing.expect(!prefix(&.{ 0, 2, 7 }, &.{ 0, 2, 7, 8 }));
    try std.testing.expect(!prefix(&.{ 0, 2, 7 }, &.{}));
}

test "qwen27 pass: completed shared logits restore the authoritative row without a GPU submission" {
    var words = [_]u16{ 0, 1, 2, 0x8000, 0x7f80, 0x7fc1, 0xffff, 0x0001, 0x3f80 };
    const untouched = words;
    try restoreHead(&words, 3, 2);
    try std.testing.expectEqualSlices(u16, untouched[6..9], words[0..3]);
    try std.testing.expectEqualSlices(u16, untouched[3..], words[3..]);
    const restored = words;
    try restoreHead(&words, 3, 0);
    try std.testing.expectEqualSlices(u16, &restored, &words);
    try std.testing.expectError(error.BadHeadRow, restoreHead(&words, 3, 3));
    try std.testing.expectError(error.BadHeadRow, restoreHead(&words, 2, 1));
    try std.testing.expectError(error.BadHeadRow, restoreHead(&words, 0, 0));
    try std.testing.expectEqualSlices(u16, &restored, &words);
}
