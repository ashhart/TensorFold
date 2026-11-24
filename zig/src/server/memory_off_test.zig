//! Learning off: a chat request carries no folded-memory query and the server encodes no memory header.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const TestText = @import("stream_preflight_test.zig").TestText;
const t = std.testing;

/// An engine that serves no learning (Info.folded_memory false, as without --slide).
const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 64 };
    }
};

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

test "memory off: a chat request has no memory query or strip, and the server encodes no memory header" {
    var script: Script = .{};
    var text: TestText = .{};
    var srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    try t.expectEqual(@as(usize, 0), srv.empty_system_prefix.len);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const messages = (try json.parse(a, "[{\"role\": \"user\", \"content\": \"what is the code?\"}]")).ok;
    const p = try chat.prepare(srv, &cx, .{ .messages = messages, .fields = (try json.parse(a, "{}")).ok }, Gone{});
    defer srv.release(p.preparing, p.request);
    try t.expectEqual(@as(usize, 0), p.request.memory_query.len);
    try t.expectEqual(@as(u32, 0), p.request.memory_strip);
    try t.expect(p.prompt_len > 0);
}
