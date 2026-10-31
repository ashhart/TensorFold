const std = @import("std");
const Stream = @import("core/lanes/stream.zig").Stream;
const control = @import("families/lfm3/prefill_control.zig");
const Probe = struct {
    stream: ?*Stream = null,
    cancel_at: ?u32 = null,
    yield_at: ?u32 = null,
    cancel_checks: usize = 0,
    yield_checks: usize = 0,
    marks: [4]u32 = @splat(0),
    marked: usize = 0,
    fn cast(p: *anyopaque) *Probe {
        return @ptrCast(@alignCast(p));
    }
    fn cancelled(p: *anyopaque) bool {
        const x = cast(p);
        x.cancel_checks += 1;
        return if (x.cancel_at) |at| x.stream.?.fill_at.load(.monotonic) >= at else false;
    }
    fn yielded(p: *anyopaque) bool {
        const x = cast(p);
        x.yield_checks += 1;
        return if (x.yield_at) |at| x.stream.?.fill_at.load(.monotonic) >= at else false;
    }
    fn kept(p: *anyopaque, _: *Stream, at: u32) void {
        const x = cast(p);
        x.marks[x.marked] = at;
        x.marked += 1;
    }
};
fn make(x: *Probe, prompt: []const u32) !Stream {
    return Stream.init(std.testing.allocator, .{ .id = "prefill-control", .prompt = prompt, .max_new = 1, .cancel_check = .{ .ptr = x, .check = Probe.cancelled }, .yield_check = .{ .ptr = x, .check = Probe.yielded }, .reuse = .{ .hook = .{ .ptr = x, .at = Probe.kept } } });
}
test "LFM3 prefill cancellation takes priority and no empty prefix is kept" {
    var p: Probe = .{ .cancel_at = 0, .yield_at = 0 };
    var s = try make(&p, &.{ 1, 2, 3 });
    defer s.deinit(std.testing.allocator);
    p.stream = &s;
    try std.testing.expectError(error.Cancelled, control.before(&s, 0));
    try std.testing.expectEqual(@as(usize, 1), p.cancel_checks);
    try std.testing.expectEqual(@as(usize, 0), p.yield_checks);
    try std.testing.expectEqual(@as(usize, 0), p.marked);
}
test "LFM3 prefill background yield remains distinct from cancellation" {
    var p: Probe = .{ .yield_at = 0 };
    var s = try make(&p, &.{ 1, 2, 3 });
    defer s.deinit(std.testing.allocator);
    p.stream = &s;
    try std.testing.expectError(error.Yielded, control.before(&s, 0));
    try std.testing.expectEqual(@as(usize, 1), p.cancel_checks);
    try std.testing.expectEqual(@as(usize, 1), p.yield_checks);
}
test "LFM3 prefill interruption keeps only a newly completed partial prefix" {
    var ids: [300]u32 = @splat(1);
    var p: Probe = .{ .cancel_at = 128 };
    var s = try make(&p, &ids);
    defer s.deinit(std.testing.allocator);
    p.stream = &s;
    control.completed(&s, 128);
    try std.testing.expectError(error.Cancelled, control.before(&s, 128));
    try std.testing.expectEqual(@as(u32, 128), p.marks[0]);
    s.cached = 128;
    try std.testing.expectError(error.Cancelled, control.before(&s, 128));
    control.completed(&s, 300);
    try std.testing.expectError(error.Cancelled, control.before(&s, 300));
    try std.testing.expectEqual(@as(usize, 1), p.marked);
}
test "LFM3 prefill restored and completed positions publish before callbacks" {
    var ids: [300]u32 = @splat(1);
    var p: Probe = .{ .yield_at = 256 };
    var s = try make(&p, &ids);
    defer s.deinit(std.testing.allocator);
    p.stream = &s;
    s.cached = 128;
    control.completed(&s, 128);
    try control.before(&s, 128);
    try std.testing.expectEqual(@as(u32, 128), s.fill_at.load(.monotonic));
    control.completed(&s, 256);
    try std.testing.expectError(error.Yielded, control.before(&s, 256));
    try std.testing.expectEqual(@as(u32, 256), p.marks[0]);
    try std.testing.expectEqual(@as(usize, 2), p.cancel_checks);
    try std.testing.expectEqual(@as(usize, 2), p.yield_checks);
}
test "LFM3 prefill completed background work ignores yield but honors final cancellation" {
    var p: Probe = .{ .yield_at = 3 };
    var s = try make(&p, &.{ 1, 2, 3 });
    defer s.deinit(std.testing.allocator);
    p.stream = &s;
    control.completed(&s, 3);
    try control.finished(&s);
    try std.testing.expectEqual(@as(usize, 0), p.yield_checks);
    p.cancel_at = 3;
    try std.testing.expectError(error.Cancelled, control.finished(&s));
    try std.testing.expectEqual(@as(usize, 0), p.marked);
}
