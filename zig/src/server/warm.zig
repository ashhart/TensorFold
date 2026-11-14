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
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    request: api.Request,

    fn event(ptr: *anyopaque, id: api.Id, e: *const api.Event) void {
        if (e.* != .finished) return;
        const p: *Pass = @ptrCast(@alignCast(ptr));
        live.untrack(p.io, id);
        var arena = p.arena;
        p.gpa.destroy(p);
        arena.deinit();
    }
};

/// Forgets begun (POST /v1/imprint/forget, the prompt cache flush), process-wide: a request reads it as it begins.
pub var forgets: std.atomic.Value(u64) = .init(0);

/// The warm passes the engine holds, by id, so that a forget can cancel them.
pub var live: Live = .{};

pub const Live = struct {
    mutex: std.Io.Mutex = .init,
    ids: std.ArrayList(api.Id) = .empty,

    const gpa = std.heap.smp_allocator;

    pub fn track(l: *Live, io: std.Io, id: api.Id) !void {
        l.mutex.lockUncancelable(io);
        defer l.mutex.unlock(io);
        try l.ids.append(gpa, id);
    }

    pub fn untrack(l: *Live, io: std.Io, id: api.Id) void {
        l.mutex.lockUncancelable(io);
        defer l.mutex.unlock(io);
        if (std.mem.indexOfScalar(api.Id, l.ids.items, id)) |i| _ = l.ids.swapRemove(i);
    }
};

/// A forget begins: replies running now prefill no next turn, and warm passes in flight are cancelled before it drops.
pub fn forgetting(srv: *Server) void {
    _ = forgets.fetchAdd(1, .seq_cst);
    live.mutex.lockUncancelable(srv.io);
    const ids = Live.gpa.dupe(api.Id, live.ids.items) catch &.{}; // the engine's cancel runs outside this lock
    live.mutex.unlock(srv.io);
    defer Live.gpa.free(ids);
    for (ids) |id| srv.engine.cancel(id);
}

/// Prefill the conversation, `message` (the reply as sent) and the next turn's opening, on `warm_turns` engines.
pub fn reply(srv: *Server, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8, prompt_len: usize, began: u64) void {
    if (!srv.info.warm_turns or input.prompt != null or input.messages != .array) return;
    if (forgets.load(.seq_cst) != began) return; // a forget began while it ran: its next turn starts cold
    start(srv, input, message, thinking, effort, prompt_len, began) catch |e| log.line("next turn: no background prefill ({s})", .{@errorName(e)});
}

fn start(srv: *Server, input: chat.Input, message: Value, thinking: bool, effort: ?[]const u8, prompt_len: usize, began: u64) !void {
    const p = try srv.gpa.create(Pass);
    p.* = .{ .gpa = srv.gpa, .io = srv.io, .arena = .init(srv.gpa), .request = undefined };
    var handed = false;
    defer if (!handed) {
        p.arena.deinit();
        srv.gpa.destroy(p);
    };
    const a = p.arena.allocator();
    const clear = input.fields.get("clear_thinking");
    var cx: errors.Cx = .{ .a = a, .clear_thinking = if (clear != null and clear.? == .bool) clear.?.bool else null };
    const asked = input.fields.get("reasoning_effort");
    cx.named_effort = @import("fields.zig").namedEffort(if (asked != null and asked.? == .string) asked.?.string else null, srv.effort_levels);
    const warm_prompt = try shared(srv, &cx, input, message, thinking, effort);
    var prompt = warm_prompt.ids;
    // an image conversation's pass holds its images as its turns did (data URLs only: no URL is fetched twice)
    const pictures = try images.expand(srv.info.vision, null, srv.text, &cx, .{ .array = warm_prompt.messages }, prompt, prompt.len);
    if (pictures) |x| prompt = x.ids;
    if (prompt.len == 0 or (srv.info.context_window > 0 and prompt.len >= srv.info.context_window)) return;
    // its states count as the reply's request's: they supersede none of that turn's own until a later turn does
    p.request = .{ .prompt = prompt, .max_tokens = 0, .background = true, .born = @intCast(prompt_len), .history_len = @intCast(prompt.len), .rewind_len = if (pictures != null) 0 else @intCast(prompt_mod.rewindLen(srv, &cx, .{ .array = warm_prompt.messages }, input.messages.array.len, input.tools, prompt, thinking, effort)), .chunks = try srv.chunks.starts(a, prompt), .decode_spans = try prompt_mod.replySpans(srv, &cx, prompt) };
    if (pictures) |x| {
        p.request.images = x.images;
        p.request.cache_key = x.cache_key;
    }
    const id = srv.next_id.fetchAdd(1, .monotonic);
    try live.track(srv.io, id);
    srv.engine.submit(id, &p.request, .{ .ctx = p, .event = Pass.event }) catch |e| {
        live.untrack(srv.io, id);
        return e;
    };
    handed = true;
    if (forgets.load(.seq_cst) != began) srv.engine.cancel(id); // a forget began as it was made, past its cancels
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
