//! A step recorded as graphs cut at its collectives, replayed with new inputs: the collectives run between the graphs.
const std = @import("std");
const rig = @import("rig.zig");
const collective = @import("../collective.zig");
const split = @import("../graph_split.zig");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");

/// One rank's "all-gather": a copy of `send` into `recv` on the stream, counted.
const Copy = struct {
    r: *const runtime.Runtime,
    calls: usize = 0,

    const vtable: collective.Comm.VTable = .{ .all_reduce = never, .all_gather = gather, .broadcast = never2, .close = close };

    fn gather(ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: collective.Dtype, s: abi.Stream) collective.Error!void {
        const c: *Copy = @ptrCast(@alignCast(ptr));
        if (dtype != .f32) return error.CollectiveFailed;
        c.calls += 1;
        runtime.check(c.r.api.hipMemcpyDtoDAsync(@ptrFromInt(recv), @ptrFromInt(send), 4 * n, s)) catch return error.CollectiveFailed;
    }

    fn never(_: *anyopaque, _: u64, _: u64, _: usize, _: collective.Dtype, _: abi.Stream) collective.Error!void {
        return error.CollectiveFailed;
    }

    fn never2(_: *anyopaque, _: u64, _: u64, _: usize, _: collective.Dtype, _: usize, _: abi.Stream) collective.Error!void {
        return error.CollectiveFailed;
    }

    fn close(_: *anyopaque) void {}
};

test "a step cut at its collective: nothing runs while it records, each replay runs the graphs and the call in order" {
    const t = try rig.Rig.open(1 << 20);
    defer t.close();
    const n = 1000;
    var a = try t.alloc(4 * n);
    defer a.free();
    var b = try t.alloc(4 * n);
    defer b.free();
    var c = try t.alloc(4 * n);
    defer c.free();
    var d = try t.alloc(4 * n);
    defer d.free();
    var host: [n]f32 = undefined;
    for (&host, 0..) |*x, i| x.* = @floatFromInt(i);
    try runtime.check(t.r.api.hipMemcpyHtoD(@ptrFromInt(a.base()), @ptrCast(&host), 4 * n));
    const zeros: [n]f32 = @splat(0);
    try runtime.check(t.r.api.hipMemcpyHtoD(@ptrFromInt(d.base()), @ptrCast(&zeros), 4 * n));
    var copy: Copy = .{ .r = &t.r };
    const inner: collective.Comm = .{ .ptr = &copy, .vtable = &Copy.vtable, .rank = 0, .world = 1 };
    var rec: split.Recorder = .{ .gpa = rig.gpa, .stream = t.stream, .inner = inner };
    try rec.begin();
    const via = rec.comm().?;
    // a -> b, the call b -> c, c -> d
    try t.on.tf_cast(@ptrFromInt(a.base()), 0, @ptrFromInt(b.base()), 0, n, t.stream.handle);
    try via.allGather(b.base(), c.base(), n, .f32, t.stream.handle);
    try t.on.tf_cast(@ptrFromInt(c.base()), 0, @ptrFromInt(d.base()), 0, n, t.stream.handle);
    var s = (try rec.finish()) orelse return error.TestUnexpectedResult;
    defer s.deinit();
    try t.stream.synchronize();
    try std.testing.expectEqual(@as(usize, 3), s.pieces.len);
    try std.testing.expectEqual(@as(usize, 0), copy.calls);
    var got: [n]f32 = undefined;
    try runtime.check(t.r.api.hipMemcpyDtoH(@ptrCast(&got), @ptrFromInt(d.base()), 4 * n));
    for (got) |x| try std.testing.expectEqual(@as(f32, 0), x);
    for (0..3) |round| {
        for (&host, 0..) |*x, i| x.* = @floatFromInt(i * (round + 2));
        try runtime.check(t.r.api.hipMemcpyHtoD(@ptrFromInt(a.base()), @ptrCast(&host), 4 * n));
        try s.launchOn(t.stream);
        try t.stream.synchronize();
        try runtime.check(t.r.api.hipMemcpyDtoH(@ptrCast(&got), @ptrFromInt(d.base()), 4 * n));
        try std.testing.expectEqualSlices(f32, &host, &got);
    }
    try std.testing.expectEqual(@as(usize, 3), copy.calls);
}
