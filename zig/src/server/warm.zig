//! After a chat reply, the next turn's shared prompt prefilled in the background, so that turn resumes past the reply.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const prompt_mod = @import("prompt.zig");
const chat = @import("chat.zig");
const log = @import("log.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;

/// One background prompt pass and the arena holding its request until the engine finishes it.
const Pass = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    request: api.Request,

    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        if (e.* != .finished) return;
        const p: *Pass = @ptrCast(@alignCast(ptr));
        var arena = p.arena;
        p.gpa.destroy(p);
        arena.deinit();
    }
};

/// Prefill the conversation, `message` (the reply as sent) and the next turn's opening, on `warm_turns` engines.
pub fn reply(srv: *Server, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8) void {
    if (!srv.info.warm_turns or input.prompt != null or input.messages != .array) return;
    start(srv, input, message, thinking, effort) catch |e| log.line("next turn: no background prefill ({s})", .{@errorName(e)});
}

fn start(srv: *Server, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8) !void {
    const p = try srv.gpa.create(Pass);
    p.* = .{ .gpa = srv.gpa, .arena = .init(srv.gpa), .request = undefined };
    var handed = false;
    defer if (!handed) {
        p.arena.deinit();
        srv.gpa.destroy(p);
    };
    const a = p.arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const warm_prompt = try shared(srv, &cx, input, message, thinking, effort);
    const prompt = warm_prompt.ids;
    if (prompt.len == 0 or (srv.info.context_window > 0 and prompt.len >= srv.info.context_window)) return;
    p.request = .{ .prompt = prompt, .max_tokens = 0, .background = true, .history_len = @intCast(prompt.len), .rewind_len = @intCast(prompt_mod.rewindLen(srv, &cx, .{ .array = warm_prompt.messages }, input.messages.array.len, input.tools, prompt, thinking, effort)), .chunks = try srv.chunks.starts(a, prompt), .decode_spans = try prompt_mod.replySpans(srv, &cx, prompt) };
    const id = srv.next_id.fetchAdd(1, .monotonic);
    try srv.engine.submit(id, &p.request, .{ .ctx = p, .event = Pass.event });
    handed = true;
}

/// The tokens every next user or tool turn shares: two renderings agree up to that turn's text.
fn shared(srv: *Server, cx: *errors.Cx, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8) !struct { ids: []const u32, messages: []Value } {
    const before = input.messages.array;
    var ids: [2][]const u32 = undefined;
    var first: []Value = undefined;
    for ([_][]const u8{ "A", "B" }, &ids, 0..) |text, *out, i| {
        const list = try cx.a.alloc(Value, before.len + 2);
        @memcpy(list[0..before.len], before);
        list[before.len] = message;
        const user = try json.newObject(cx.a);
        try user.put(cx.a, "role", .{ .string = "user" });
        try user.put(cx.a, "content", .{ .string = text });
        list[before.len + 1] = .{ .object = user };
        if (i == 0) first = list;
        out.* = try prompt_mod.renderIds(srv, cx, .{ .array = list }, input.tools, thinking, effort, true);
    }
    const n = std.mem.indexOfDiff(u32, ids[0], ids[1]) orelse return .{ .ids = &.{}, .messages = first };
    return .{ .ids = ids[0][0..n], .messages = first };
}
