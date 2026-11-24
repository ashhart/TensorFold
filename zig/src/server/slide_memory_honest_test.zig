//! Folded memory's lessons report what was done: stored and compiled, never a recall check that did not run.
const std = @import("std");
const api = @import("engine_api");
const slide_memory = @import("slide_memory.zig");
const Server = @import("server.zig").Server;
const TestText = @import("stream_preflight_test.zig").TestText;
const t = std.testing;

/// An engine whose memory ops all succeed and find no conflict; it serves no generation.
const Script = struct {
    ops: u32 = 0,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined, .memory_op = op } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn op(ctx: *anyopaque, _: *const api.MemoryOp) api.MemoryResult {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.ops += 1;
        return .{ .units = 1, .affected = 1 };
    }
};

test "folded memory: a saved lesson records no recall verdict, because no recall check ran" {
    const env = @import("sys").env;
    try t.expectEqual(@as(c_int, 0), env.setenv("TF_SLIDE_FAST_CUES", "1", 1)); // no model-written cues: no generation
    defer _ = env.unsetenv("TF_SLIDE_FAST_CUES");
    var script: Script = .{};
    var text: TestText = .{};
    var srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const id = slide_memory.learnBackground(srv, "The fictional observatory access code is ORBIT-417.", "chat", .{}) orelse return error.TestUnexpectedResult;
    try t.expect(script.ops > 0);
    const f = srv.slide.find(id) orelse return error.TestUnexpectedResult;
    try t.expectEqual(@as(u32, 0), f.checks);
    try t.expect(f.recalled == null);
}
