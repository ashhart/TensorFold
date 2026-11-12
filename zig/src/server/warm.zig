//! After a chat reply, the next turn's shared prompt prefilled in the background, so that turn resumes past the reply.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const prompt_mod = @import("prompt.zig");
const images = @import("images.zig");
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
pub fn reply(srv: *Server, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8, prompt_len: usize) void {
    if (!srv.info.warm_turns or input.prompt != null or input.messages != .array) return;
    start(srv, input, message, thinking, effort, prompt_len) catch |e| log.line("next turn: no background prefill ({s})", .{@errorName(e)});
}

fn start(srv: *Server, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8, prompt_len: usize) !void {
    const p = try srv.gpa.create(Pass);
    p.* = .{ .gpa = srv.gpa, .arena = .init(srv.gpa), .request = undefined };
    var handed = false;
    defer if (!handed) {
        p.arena.deinit();
        srv.gpa.destroy(p);
    };
    const a = p.arena.allocator();
    const clear = input.fields.get("clear_thinking");
    var cx: errors.Cx = .{ .a = a, .clear_thinking = if (clear != null and clear.? == .bool) clear.?.bool else null };
    const warm_prompt = try shared(srv, &cx, input, try sentBack(a, input.messages.array, message), thinking, effort);
    var prompt = warm_prompt.ids;
    // an image conversation's pass holds its images as its turns did (data URLs only: no URL is fetched twice)
    const pictures = try images.expand(srv.info.vision, null, srv.text, &cx, .{ .array = warm_prompt.messages }, prompt, prompt.len);
    if (pictures) |x| prompt = x.ids;
    if (prompt.len == 0 or (srv.info.context_window > 0 and prompt.len >= srv.info.context_window)) return;
    const messages: Value = .{ .array = warm_prompt.messages };
    // cut as the next turn's request cuts (chat.prepare), so a planned family's kept end is that request's state
    const system_len = prompt_mod.systemPrefixLen(srv, &cx, messages, input.tools, prompt, thinking, effort);
    // its states count as the reply's request's: they supersede none of that turn's own until a later turn does
    p.request = .{ .prompt = prompt, .max_tokens = 0, .background = true, .born = @intCast(prompt_len), .history_len = @intCast(prompt.len), .rewind_len = if (pictures != null) 0 else @intCast(prompt_mod.rewindLen(srv, &cx, messages, input.messages.array.len, input.tools, prompt, thinking, effort)), .chunks = try chat.requestChunks(srv, a, prompt, system_len), .decode_spans = try prompt_mod.replySpans(srv, &cx, prompt) };
    if (pictures) |x| {
        p.request.images = x.images;
        p.request.cache_key = x.cache_key;
    }
    const id = srv.next_id.fetchAdd(1, .monotonic);
    try srv.engine.submit(id, &p.request, .{ .ctx = p, .event = Pass.event });
    handed = true;
}

/// The reply as the next turn sends it back: with its reasoning only when an earlier reply came back with its own.
fn sentBack(a: std.mem.Allocator, history: []const Value, message: Value) !Value {
    if (!message.has("reasoning_content") or keepsReasoning(history)) return message;
    const copy = try json.copyObject(a, message.object);
    _ = copy.orderedRemove("reasoning_content");
    return .{ .object = copy };
}

/// Whether a client sends replies back with their reasoning, from its history; a first turn's client is taken not to.
fn keepsReasoning(history: []const Value) bool {
    for (history) |m| {
        if (!std.mem.eql(u8, m.strField("role") orelse "", "assistant")) continue;
        if (m.strField("reasoning_content")) |r| if (r.len > 0) return true;
    }
    return false;
}

/// The tokens every next user or tool turn shares: two renderings agree up to that turn's text.
fn shared(srv: *Server, cx: *errors.Cx, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8) !struct { ids: []const u32, messages: []Value } {
    const before = input.messages.array;
    const call_id = firstCallId(message);
    var ids: [2][]const u32 = undefined;
    var first: []Value = undefined;
    for ([_][]const u8{ "A", "B" }, &ids, 0..) |text, *out, i| {
        const list = try cx.a.alloc(Value, before.len + 2);
        @memcpy(list[0..before.len], before);
        list[before.len] = message;
        // a reply that called tools is answered by their results: the next turn opens as its first call's result
        const next = try json.newObject(cx.a);
        try next.put(cx.a, "role", .{ .string = if (call_id != null) "tool" else "user" });
        if (call_id) |id| try next.put(cx.a, "tool_call_id", .{ .string = id });
        try next.put(cx.a, "content", .{ .string = text });
        list[before.len + 1] = .{ .object = next };
        if (i == 0) first = list;
        out.* = try prompt_mod.renderIds(srv, cx, .{ .array = list }, input.tools, thinking, effort, true);
    }
    const n = std.mem.indexOfDiff(u32, ids[0], ids[1]) orelse return .{ .ids = &.{}, .messages = first };
    return .{ .ids = ids[0][0..n], .messages = first };
}

/// The reply's first tool call's id when it made calls (the next turn opens with that call's result); else null.
fn firstCallId(message: Value) ?[]const u8 {
    const calls = message.get("tool_calls") orelse return null;
    if (calls != .array or calls.array.len == 0) return null;
    return calls.array[0].strField("id") orelse "";
}
