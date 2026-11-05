//! File bytes and mmap bytes deliberately differ, so the actual upload hook must use the owned file producer.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const Fake = @import("upload_fake").Fake;
const t = std.testing;
const Reader = struct {
    pub const alignment = 4096;
    var reads: usize = 0;
    var last: usize = 0;
    var closes: usize = 0;
    var broken = false;
    pub fn fits(_: usize) usize {
        return 7;
    }
    pub fn span(at: u64, len: usize) usize {
        return @intCast(std.mem.alignForward(u64, at + len, alignment) - std.mem.alignBackward(u64, at, alignment));
    }
    pub const File = struct {
        pub fn open(_: []const u8) !File {
            return .{};
        }
        pub fn close(_: *File) void {
            closes += 1;
        }
        pub fn read(_: File, into: []align(4096) u8, at: u64, n: usize) ![]u8 {
            try t.expect(Fake.pending == null);
            reads += 1;
            if (broken) return error.ReadFailed;
            for (into[3..][0..n], 0..) |*b, i| b.* = @truncate(at + i);
            last = @intFromPtr(into.ptr) + 3;
            return into[3..][0..n];
        }
    };
};
const Source = cuda.checkpoint_upload.Source(Reader);
fn open(d: *cuda.Driver, mapped: []const u8) !Source {
    Reader.reads = 0;
    Reader.closes = 0;
    Reader.broken = false;
    const files = [_]struct { path: []const u8, map: struct { memory: []const u8 } }{.{ .path = "fixture", .map = .{ .memory = mapped } }};
    return Source.open(t.allocator, d, &.{ .files = .{ .items = &files } }, true);
}
test "weight upload: actual Qwen hook reads one pinned stage, unaligned pieces and tail" {
    var d = Fake.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(61));
    const mapped: [32]u8 = @splat(0xcc);
    var source = try open(&d, &mapped);
    defer source.deinit();
    var into = .{ .weight_source = @as(?*Source, &source) };
    const dst: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = 128, .borrowed = true };
    try weights.uploadAt(&into, dst, 5, mapped[1..20]);
    for (Fake.gpu[5..24], 1..) |b, i| try t.expectEqual(@as(u8, @intCast(i)), b);
    try t.expectEqual(@as(usize, 3), Reader.reads);
    try t.expectEqual(@as(usize, 3), Fake.copies);
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expectEqual(Reader.last, Fake.last_source);
    try t.expectEqual(@as(usize, 0), Fake.early);
}
test "weight upload: Qwen disabled and temporary vectors keep the ordinary uploader" {
    var d = Fake.driver();
    var source = try Source.open(t.allocator, &d, @as(*const struct { files: struct { items: []const struct { path: []const u8, map: struct { memory: []const u8 } } } }, undefined), false);
    defer source.deinit();
    try t.expectEqual(@as(usize, 0), source.maps.items.len);
    var into = .{ .weight_source = @as(?*Source, &source) };
    const bytes = [_]u8{ 9, 8, 7 };
    try weights.uploadAt(&into, .{ .d = &d, .ptr = 0x1000, .len = 16, .borrowed = true }, 2, &bytes);
    try t.expectEqualSlices(u8, &bytes, Fake.gpu[2..5]);
    try t.expectEqual(@as(usize, 0), Fake.allocs);
}
test "weight upload: failed completion outlives checkpoint and later recovers" {
    var d = Fake.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(62));
    const mapped: [32]u8 = @splat(0xcc);
    var source = try open(&d, &mapped);
    const before = cuda.copy_source.retained();
    Fake.fail = true;
    try t.expectError(error.NotReady, source.upload(.{ .d = &d, .ptr = 0x1000, .len = 64, .borrowed = true }, 0, mapped[1..8]));
    source.deinit();
    try t.expectEqual(@as(usize, 1), Reader.closes);
    try t.expectEqual(@as(usize, 0), Fake.frees);
    try t.expectEqual(before + 1, cuda.copy_source.retained());
    Fake.fail = false;
    try cuda.copy_source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 1), Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.early);
    try t.expectEqual(before, cuda.copy_source.retained());
}
test "weight upload: producer error and bounds refuse before DMA" {
    var d = Fake.driver();
    const mapped: [32]u8 = @splat(0xcc);
    var source = try open(&d, &mapped);
    defer source.deinit();
    const dst: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = 16, .borrowed = true };
    try t.expectError(error.Invalid, source.upload(dst, 15, mapped[0..2]));
    Reader.broken = true;
    try t.expectError(error.ReadFailed, source.upload(dst, 0, mapped[0..2]));
    try t.expectEqual(@as(usize, 0), Fake.copies);
}
test "weight upload: Qwen takes one full slot at open, never grows it between tensor pieces" {
    var d = Fake.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(63));
    const mapped: [8192]u8 = @splat(0xcc);
    var source = try open(&d, &mapped);
    defer source.deinit();
    const at_open = Fake.allocs;
    const dst: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = 128, .borrowed = true };
    try t.expect(try source.upload(dst, 0, mapped[4090..4091]));
    try t.expect(try source.upload(dst, 1, mapped[4094..4113]));
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expectEqual(@as(usize, 1), at_open);
    try t.expectEqual(@as(usize, 0), Fake.frees);
    for (Fake.gpu[1..20], 4094..) |b, i| try t.expectEqual(@as(u8, @truncate(i)), b);
}
