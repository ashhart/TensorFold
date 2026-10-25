//! Cortheon's reply loop: rounds with the model's server tool calls run between them, checks at the end, one reply.
const std = @import("std");
const json = @import("json");
const errors = @import("../errors.zig");
const chat = @import("../chat.zig");
const tool_parse = @import("../tool_parse.zig");
const compact = @import("../compact.zig");
const Server = @import("../server.zig").Server;
const Turn = @import("cortheon.zig").Turn;
const tools = @import("tools.zig");
const checks = @import("checks.zig");
const research = @import("research.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Generation rounds in one reply, the check round included.
const max_rounds = 8;

/// The whole reply to `input` (research first when the turn wants it); `input` ends as the model read it.
pub fn whole(t: *Turn, srv: *Server, cx: *errors.Cx, input: *chat.Input, gone: anytype) chat.Failure!chat.Reply {
    if (t.research) try research.run(t, srv, cx, input, null, gone);
    return rounds(t, srv, cx, input, try prepare(srv, cx, input.*, gone), null, gone);
}

/// The streamed reply from `first` (prepared before the stream opened); it owns `first` from here.
pub fn stream(t: *Turn, srv: *Server, cx: *errors.Cx, input: *chat.Input, first: chat.Prepared, sink: chat.Sink, gone: anytype) chat.Failure!chat.Reply {
    var prepared = first;
    if (t.research) {
        srv.release(prepared.preparing, prepared.request);
        try research.run(t, srv, cx, input, sink, gone);
        prepared = try prepare(srv, cx, input.*, gone);
    }
    return rounds(t, srv, cx, input, prepared, sink, gone);
}

/// A round's request, compacted first when the server compacts.
fn prepare(srv: *Server, cx: *errors.Cx, input: chat.Input, gone: anytype) chat.Failure!chat.Prepared {
    return (try compact.prepare(srv, cx, input, gone)).prepared;
}

fn rounds(t: *Turn, srv: *Server, cx: *errors.Cx, input: *chat.Input, first: chat.Prepared, sink: ?chat.Sink, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    var prepared = first;
    var filter: Filter = .{ .a = a, .inner = sink };
    var completion: usize = 0;
    var reasoning: usize = 0;
    var round: u32 = 0;
    while (true) : (round += 1) {
        filter.startRound();
        var reply = try chat.generate(srv, cx, prepared, if (sink != null) filter.sink() else null, gone);
        t.rounds += 1;
        completion += reply.completion_tokens;
        reasoning += reply.reasoning_tokens;
        const parsed = try tool_parse.parse(a, reply.content, input.tools, null);
        var server: std.ArrayList(Value) = .empty;
        var client = false;
        if (parsed.calls) |calls| for (calls) |c| {
            if (tools.isServer(tools.callName(c))) try server.append(a, c) else client = true;
        };
        const more = round + 1 < max_rounds and !std.mem.eql(u8, reply.finish_reason, "length");
        if (server.items.len > 0 and t.calls_left > 0 and more) {
            try filter.endRound(false);
            if (sink != null) try t.addProse(a, parsed.content); // a stream already showed it; a whole reply drops the narration
            try runCalls(t, a, input, server.items, sink, gone);
            prepared = try prepare(srv, cx, input.*, gone);
            continue;
        }
        if (more and !client and server.items.len == 0) if (check(t, input.messages.array, parsed.content)) |c| {
            try filter.endRound(false);
            try t.addProse(a, parsed.content);
            try t.checks.append(a, c.name);
            try t.progress(a, sink, "check: {s}", .{c.name});
            try t.hide(a, input, try note(a, c.note));
            prepared = try prepare(srv, cx, input.*, gone);
            continue;
        };
        try filter.endRound(true);
        reply.content = try t.visible(a, reply.content);
        reply.completion_tokens = completion;
        reply.reasoning_tokens = reasoning;
        return reply;
    }
}

/// The check this reply's end calls for, once a user turn: tests after an edit, then sources for researched answers.
fn check(t: *Turn, messages: []const Value, prose: []const u8) ?checks.Check {
    if (t.client_tools and t.task.deliverable == .code_change and checks.untested(messages) and t.state.memory.firstCheck(t.user_key ^ 1))
        return .{ .name = "tests", .note = checks.tests_note };
    const researched = t.fetches > 0 and (t.task.deliverable == .research_answer or t.task.deliverable == .answer or t.task.deliverable == .decision);
    if (researched and !checks.cites(prose) and !checks.cites(t.prose.items) and t.state.memory.firstCheck(t.user_key ^ 2))
        return .{ .name = "sources", .note = checks.sources_note };
    return null;
}

fn note(a: Allocator, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "system" });
    try o.put(a, "content", .{ .string = text });
    return .{ .object = o };
}

/// Runs the model's server tool calls (within the reply's budget) and adds them and their results to `input`.
fn runCalls(t: *Turn, a: Allocator, input: *chat.Input, calls: []const Value, sink: ?chat.Sink, gone: anytype) chat.Failure!void {
    const allowed = calls[0..@min(calls.len, t.calls_left)];
    try t.hide(a, input, try research.assistantCalls(a, allowed));
    for (allowed) |c| {
        if (gone.check()) return error.Cancelled;
        const is_search = std.mem.eql(u8, tools.callName(c), tools.search_name);
        const subject = (try tools.argument(a, c, if (is_search) "query" else "url")) orelse "";
        if (is_search) try t.progress(a, sink, "searching the web for \"{s}\"", .{subject}) else try t.progress(a, sink, "reading {s}", .{subject});
        if (is_search) t.focus = subject;
        const ran = try tools.run(t.state.web, a, c, t.today, 6000, try std.mem.concat(a, u8, &.{ t.focus, " ", t.goal }), &t.seen);
        try t.noteSources(a, ran);
        if (!ran.ok) try t.progress(a, sink, "{s}", .{ran.text[0..@min(ran.text.len, 200)]});
        try t.hide(a, input, try research.toolMessage(a, c, ran.text));
    }
    t.calls_left -= @intCast(allowed.len);
    if (t.calls_left == 0) try t.hide(a, input, try note(a, "Cortheon: the web budget for this reply is spent. Answer now from the evidence above, and say what is still unsettled."));
}

/// The client's view of the stream: Cortheon's calls never reach it, other calls wait for the round's end.
const Filter = struct {
    a: Allocator,
    inner: ?chat.Sink,
    calls: std.ArrayList(Call) = .empty,
    held: std.ArrayList(Value) = .empty, // client call deltas, sent once the round turns out to be the last
    clients: u32 = 0, // client calls this round, which number the ones the client sees
    wrote: bool = false, // an earlier round sent content
    wrote_round: bool = false,

    const Call = struct { index: i64, server: bool, client: u32 };

    fn sink(f: *Filter) chat.Sink {
        return .{ .ctx = f, .call = call, .begin = begin };
    }

    fn begin(ctx: *anyopaque, prompt_len: usize) error{Closed}!void {
        const f: *Filter = @ptrCast(@alignCast(ctx));
        const inner = f.inner orelse return;
        if (inner.begin) |start| try start(inner.ctx, prompt_len);
    }

    fn startRound(f: *Filter) void {
        f.calls.clearRetainingCapacity();
        f.held.clearRetainingCapacity();
        f.clients = 0;
        f.wrote = f.wrote or f.wrote_round;
        f.wrote_round = false;
    }

    /// The round ended: its held client calls go out when it is the last round, else they are dropped with it.
    fn endRound(f: *Filter, last: bool) chat.Failure!void {
        const inner = f.inner orelse return;
        if (last) for (f.held.items) |d| inner.call(inner.ctx, d) catch return error.Cancelled;
        f.held.clearRetainingCapacity();
    }

    fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
        const f: *Filter = @ptrCast(@alignCast(ctx));
        const inner = f.inner.?;
        if (delta == .string) {
            if (delta.string.len == 0) return;
            if (f.wrote and !f.wrote_round) try inner.call(inner.ctx, .{ .string = "\n\n" });
            f.wrote_round = true;
            return inner.call(inner.ctx, delta);
        }
        const list = delta.get("tool_calls") orelse return inner.call(inner.ctx, delta);
        if (list != .array) return;
        for (list.array) |one| {
            if (one != .object) continue;
            const index = (one.get("index") orelse continue).int64() orelse continue;
            const known = for (f.calls.items) |c| {
                if (c.index == index) break c;
            } else blk: {
                const c: Call = .{ .index = index, .server = tools.isServer(tools.callName(one)), .client = f.clients };
                if (!c.server) f.clients += 1;
                f.calls.append(f.a, c) catch return error.Closed;
                break :blk c;
            };
            if (known.server) continue;
            const copy = json.copyObject(f.a, one.object) catch return error.Closed;
            copy.put(f.a, "index", json.intValue(f.a, known.client) catch return error.Closed) catch return error.Closed;
            const wrapped = tool_parse.wrapCalls(f.a, .{ .object = copy }) catch return error.Closed;
            f.held.append(f.a, wrapped) catch return error.Closed;
        }
    }
};
