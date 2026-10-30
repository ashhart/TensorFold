//! The real CUDA prompt cancellation hook needs no device to name the last completed prefix.
const std = @import("std");
const lanes = @import("lanes");
const t = std.testing;

pub fn cancelled(comptime Lanes: type, backend: anytype) !void {
    const Probe = struct {
        calls: usize = 0,
        at: u32 = 0,
        fn kept(ctx: *anyopaque, _: *lanes.Stream, at: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            p.calls += 1;
            p.at = at;
        }
    };
    var state: Lanes = .{ .gpa = t.allocator, .io = t.io, .e = undefined, .capacity = 32, .target = undefined };
    defer state.fills.deinit(t.allocator);
    var probe: Probe = .{};
    var s = try lanes.Stream.init(t.allocator, .{ .id = "prefix", .prompt = &(@as([16]u32, @splat(3))), .max_new = 1, .reuse = .{ .hook = .{ .ptr = &probe, .at = Probe.kept } } });
    defer s.deinit(t.allocator);
    const b = backend(&state);
    const cancel = b.vtable.cancel_prefill orelse return error.MissingCancelHook;
    try cancel(b.ptr, &s);
    try t.expectEqual(@as(usize, 0), probe.calls);
    s.cached = 4;
    s.fill_at.store(12, .monotonic);
    try state.fills.put(t.allocator, &s, .{ .at = 8, .media = null });
    try cancel(b.ptr, &s);
    try t.expectEqual(@as(usize, 1), probe.calls);
    try t.expectEqual(@as(u32, 8), probe.at);
    try t.expect(state.fills.contains(&s)); // the host still owns the fill until its snapshot and release complete
    for ([_]usize{ 0, 4, 16 }) |at| {
        state.fills.getPtr(&s).?.at = at;
        try cancel(b.ptr, &s);
    }
    try t.expectEqual(@as(usize, 1), probe.calls);
}
