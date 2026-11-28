//! EXL3 uploads whose copy is accepted and whose wait fails: each destination stays until a wait proves completion.
const std = @import("std");
const up = @import("exl3_upload.zig");
const mock = @import("exl3_mock.zig");
const copy_source = @import("copy_source.zig");

const Tensor = struct { bytes: []const u8 };
const Part = struct { trellis: Tensor, suh: Tensor, svh: Tensor, bias: ?Tensor = null, suh_packed: bool = false };
const Book = enum { inst3, mcg, mul1 };
const group = .{ .k = @as(usize, 128), .n = @as(usize, 128), .halves = @as(u16, 8), .codebook = Book.mcg };

/// Nothing freed while a copy was pending, every buffer freed after, the wait's error returned, no source retained.
fn settled(u: *const up.Uploader) !void {
    try std.testing.expectEqual(@as(usize, 0), mock.bad_frees);
    try std.testing.expect(!mock.pending);
    try std.testing.expectEqual(mock.allocations, mock.frees);
    try std.testing.expectEqual(@as(usize, 0), u.bufs.items.len);
    try std.testing.expectEqual(@as(usize, 0), copy_source.retained());
}

fn dense(fail_copy: usize, waits: usize, slab: usize) !void {
    var d = mock.init();
    mock.fail_copy = fail_copy;
    mock.fail_waits = waits;
    var u: up.Uploader = .{ .d = &d, .gpa = std.testing.allocator, .max_buffers = 8, .slab = slab };
    defer u.deinit();
    const parts: Part = .{ .trellis = .{ .bytes = &.{ 1, 2 } }, .suh = .{ .bytes = &.{ 3, 4 } }, .svh = .{ .bytes = &.{ 5, 6 } } };
    try std.testing.expectError(error.NotReady, u.dense(parts, group));
    try settled(&u);
}

test "a dense group whose first copy's wait fails keeps that buffer until its stream completes the copy" {
    try dense(1, 1, 0);
}

test "a dense group whose second copy's wait fails twice keeps that buffer until the context completes the copy" {
    try dense(2, 2, 0);
}

test "a new slab whose first copy's wait fails is drained, freed and never registered" {
    try dense(1, 1, 1024);
    try std.testing.expectEqual(@as(usize, 1), mock.allocations);
}

test "a copy carved into an open slab whose wait fails drains before rollback frees the slab" {
    try dense(2, 2, 1024);
    try std.testing.expectEqual(@as(usize, 1), mock.allocations);
}

test "a failed carve leaves the uploader without the slab it could not fill" {
    var d = mock.init();
    mock.fail_copy = 1;
    mock.fail_waits = 1;
    var u: up.Uploader = .{ .d = &d, .gpa = std.testing.allocator, .max_buffers = 8, .slab = 1024 };
    defer u.deinit();
    const small: [100]u8 = @splat(7);
    try std.testing.expectError(error.NotReady, u.dev(&small));
    try std.testing.expectEqual(@as(?usize, null), u.slab_at);
    try std.testing.expectEqual(@as(usize, 0), u.bytes);
    try settled(&u);
    _ = try u.dev(&small); // the next upload opens a fresh slab
    try std.testing.expectEqual(@as(usize, 2), mock.allocations);
    try std.testing.expectEqual(@as(usize, 1024), u.bytes);
}

test "an expert layer whose fifth copy's wait fails frees the completed copies and keeps the fifth until it completes" {
    var d = mock.init();
    mock.fail_copy = 5;
    mock.fail_waits = 1;
    var u: up.Uploader = .{ .d = &d, .gpa = std.testing.allocator, .max_buffers = 18 };
    defer u.deinit();
    const signs: [16]u8 = @splat(255);
    const p: [2]up.Proj = @splat(.{ .trellis = &.{ 0, 0 }, .suh = &signs, .svh = &signs, .k2 = 8, .signs = true });
    try std.testing.expectError(error.NotReady, up.layer(&u, std.testing.allocator, 128, 128, .mul1, 0, 0, &p, &p, &p));
    try std.testing.expectEqual(@as(usize, 5), mock.copy_count);
    try settled(&u);
}
