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
    tags: [32]std.meta.Tag(api.MemoryOp) = undefined, // each op in order, the first 32
    unstaged_teach: u32 = 0,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined, .memory_op = op } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn op(ctx: *anyopaque, o: *const api.MemoryOp) api.MemoryResult {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        if (s.ops < s.tags.len) s.tags[s.ops] = std.meta.activeTag(o.*);
        if (o.* == .teach_chunk and !o.teach_chunk.staged) s.unstaged_teach += 1;
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

test "Learning persistence: a lesson's chunks are taught unseen and committed in one publish-and-save" {
    const env = @import("sys").env;
    try t.expectEqual(@as(c_int, 0), env.setenv("TF_SLIDE_FAST_CUES", "1", 1));
    defer _ = env.unsetenv("TF_SLIDE_FAST_CUES");
    var script: Script = .{};
    var text: TestText = .{};
    var srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    _ = slide_memory.learnBackground(srv, "The fictional observatory access code is ORBIT-417.", "chat", .{}) orelse return error.TestUnexpectedResult;
    try t.expectEqual(@as(u32, 0), script.unstaged_teach); // nothing a request could route against before the publish
    var last_teach: ?usize = null;
    var publish: ?usize = null;
    var save_count: usize = 0;
    for (script.tags[0..@min(script.ops, script.tags.len)], 0..) |tag, i| switch (tag) {
        .teach_chunk => last_teach = i,
        .publish_parent => {
            try t.expect(publish == null);
            publish = i;
        },
        .save => save_count += 1,
        else => {},
    };
    try t.expect(last_teach.? < publish.?);
    try t.expectEqual(@as(usize, 0), save_count); // publish_parent includes durable commit and any tombstones
}

test "Learning persistence: an atomic publish failure does not issue compensating live or durable writes" {
    const FailPublish = struct {
        ops: u32 = 0,
        tags: [32]std.meta.Tag(api.MemoryOp) = undefined,
        fn engine(s: *@This()) api.Engine {
            return .{ .ctx = s, .vtable = &.{ .info = info, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined, .memory_op = op } };
        }
        fn info(_: *anyopaque) api.Info {
            return .{ .context_window = 4096 };
        }
        fn op(ctx: *anyopaque, o: *const api.MemoryOp) api.MemoryResult {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            if (s.ops < s.tags.len) s.tags[s.ops] = std.meta.activeTag(o.*);
            s.ops += 1;
            if (o.* == .publish_parent) return .{ .err = "injected pre-commit failure" };
            return .{ .units = 1, .affected = 1 };
        }
    };
    const env = @import("sys").env;
    try t.expectEqual(@as(c_int, 0), env.setenv("TF_SLIDE_FAST_CUES", "1", 1));
    defer _ = env.unsetenv("TF_SLIDE_FAST_CUES");
    var script: FailPublish = .{};
    var text: TestText = .{};
    var srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    try t.expect(slide_memory.learnBackground(srv, "The fictional observatory access code is ORBIT-417.", "chat", .{}) == null);
    var publishes: usize = 0;
    var compensating_ops: usize = 0;
    for (script.tags[0..@min(script.ops, script.tags.len)]) |tag| switch (tag) {
        .publish_parent => publishes += 1,
        .save, .forget_parent, .supersede_parent => compensating_ops += 1,
        else => {},
    };
    try t.expectEqual(@as(usize, 1), publishes);
    try t.expectEqual(@as(usize, 0), compensating_ops);
}
