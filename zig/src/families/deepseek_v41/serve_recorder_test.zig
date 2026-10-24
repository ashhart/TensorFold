//! The trace wrapper must not swallow erase requests meant for the underlying shared cache.
const std = @import("std");
const api = @import("engine_api");
const Recorder = @import("serve_recorder.zig").Recorder;
const Fake = struct {
    calls: u32 = 0,
    fn engine(f: *@This()) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .forget_prompts = forget } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .prompt_cache = true };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn forget(ctx: *anyopaque, span: []const u32) api.ForgetError!api.prompt_cache.Forgotten {
        const f: *@This() = @ptrCast(@alignCast(ctx));
        if (!std.mem.eql(u32, span, &.{ 7, 11, 13 })) return error.Busy;
        f.calls += 1;
        return .{ .states = 2 };
    }
};

test "DeepSeek trace forwards forget and reports its retained token record" {
    var f: Fake = .{};
    var rec: Recorder = .{ .gpa = std.testing.allocator, .io = std.testing.io, .inner = f.engine(), .path = "operator-trace.jsonl" };
    defer rec.deinit();
    const got = try rec.engine().forgetPrompts(&.{ 7, 11, 13 });
    try std.testing.expectEqual(@as(u32, 1), f.calls);
    try std.testing.expectEqual(@as(u32, 2), got.states);
    try std.testing.expectEqual(@as(u32, 1), got.failed);
    try std.testing.expect(!got.complete());
    rec.path = null;
    const direct = try rec.engine().forgetPrompts(&.{ 7, 11, 13 });
    try std.testing.expectEqual(@as(u32, 2), f.calls);
    try std.testing.expect(direct.complete());
}
