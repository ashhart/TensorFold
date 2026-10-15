//! What a lesson asks the model and how replies become examples: chat replies, continuations, token ids.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

pub const think_reply = 320; // a thinking reply a check reads at most: reasoning, then what the user sees

pub const think_budget = 224; // reasoning a check allows before the reply is forced

pub fn encode(srv: *Server, cx: *Cx, al: Allocator, text: []const u8) ![]const u32 {
    return srv.text.encode(al, text, false) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the tokenizer cannot encode a lesson's text");
}

/// The turn's end after a reply that finished, none after one cut short.
pub fn ending(reply: chat.Reply, end: []const u32) []const u32 {
    return if (std.mem.eql(u8, reply.finish_reason, "stop")) end else &.{};
}

/// `question` as the model reads it when asked (no thinking, after a system note when given), then `answer` and `end`.
pub fn example(srv: *Server, cx: *Cx, keep: Allocator, system: ?[]const u8, question: []const u8, answer: []const u8, end: []const u32) !api.Example {
    const a = cx.a;
    var asked: std.ArrayList(Value) = .empty;
    if (system) |s| try asked.append(a, try message(a, "system", s));
    try asked.append(a, try message(a, "user", question));
    const head = try prompt.renderIds(srv, cx, .{ .array = asked.items }, &.{}, false, null, true);
    const body = srv.text.encode(a, answer, false) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the tokenizer cannot encode an answer");
    return .{ .ids = try std.mem.concat(keep, u32, &.{ head, body, end }), .start = @intCast(head.len), .decide = prompt.decideOf(srv, head) };
}

/// `question` as the model reads it when asked (after a system message when given): its prompt rows only.
pub fn promptOnly(srv: *Server, cx: *Cx, keep: Allocator, system: ?[]const u8, question: []const u8) !api.Example {
    const a = cx.a;
    var asked: std.ArrayList(Value) = .empty;
    if (system) |s| try asked.append(a, try message(a, "system", s));
    try asked.append(a, try message(a, "user", question));
    const head = try prompt.renderIds(srv, cx, .{ .array = asked.items }, &.{}, false, null, true);
    if (head.len < 3) return cx.refuse("a question renders too short to route");
    return .{ .ids = try keep.dupe(u32, head), .start = @intCast(head.len - 1), .decide = prompt.decideOf(srv, head) };
}

/// `question` as a router reads it after an unrelated exchange (`before`: the user's turn and the reply).
pub fn promptAfter(srv: *Server, cx: *Cx, keep: Allocator, before: [2][]const u8, question: []const u8) !api.Example {
    const a = cx.a;
    const asked = [_]Value{ try message(a, "user", before[0]), try message(a, "assistant", before[1]), try message(a, "user", question) };
    const head = try prompt.renderIds(srv, cx, .{ .array = @constCast(&asked) }, &.{}, false, null, true);
    if (head.len < 3) return cx.refuse("a question renders too short to route");
    return .{ .ids = try keep.dupe(u32, head), .start = @intCast(head.len - 1), .decide = prompt.decideOf(srv, head) };
}

/// One greedy reply without thinking to `user`, after a system note when given.
pub fn ask(srv: *Server, cx: *Cx, system: ?[]const u8, user: []const u8, max_tokens: i64, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    var list: std.ArrayList(Value) = .empty;
    if (system) |s| try list.append(a, try message(a, "system", s));
    try list.append(a, try message(a, "user", user));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .float = 0 });
    const prepared = try chat.prepare(srv, cx, .{ .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = max_tokens, .temperature = 0 }, gone);
    return chat.generate(srv, cx, prepared, null, gone);
}

/// One greedy reply as `ask` gives it, from the model as shipped (no lesson acts): a lesson's writer hears none.
pub fn askBase(srv: *Server, cx: *Cx, system: ?[]const u8, user: []const u8, max_tokens: i64, gone: anytype) chat.Failure!chat.Reply {
    return askBaseAt(srv, cx, system, user, max_tokens, 0, gone);
}

/// `askBase` sampled at `temperature`, seeded by the prompt so it repeats: other words than the greedy reply's.
pub fn askBaseAt(srv: *Server, cx: *Cx, system: ?[]const u8, user: []const u8, max_tokens: i64, temperature: f64, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    var list: std.ArrayList(Value) = .empty;
    if (system) |s| try list.append(a, try message(a, "system", s));
    try list.append(a, try message(a, "user", user));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .float = temperature });
    var prepared = try chat.prepare(srv, cx, .{ .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = max_tokens, .temperature = temperature }, gone);
    prepared.request.decide = null;
    prepared.request.untaught = true;
    return chat.generate(srv, cx, prepared, null, gone);
}

/// One greedy reply with thinking on, its reasoning bounded; its content is what the user reads after the reasoning.
pub fn askThinking(srv: *Server, cx: *Cx, user: []const u8, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    const asked = try a.dupe(Value, &.{try message(a, "user", user)});
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = true });
    try fields.put(a, "temperature", .{ .float = 0 });
    try fields.put(a, "thinking_budget", try json.intValue(a, think_budget));
    const prepared = try chat.prepare(srv, cx, .{ .messages = .{ .array = asked }, .fields = .{ .object = fields }, .max_tokens = think_reply, .temperature = 0 }, gone);
    return chat.generate(srv, cx, prepared, null, gone);
}

/// One greedy continuation of `ids` as they stand, past any template, `max_tokens` at most.
pub fn continueIds(srv: *Server, cx: *Cx, ids: []const u32, max_tokens: i64, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    const fields = try json.newObject(a);
    try fields.put(a, "temperature", .{ .float = 0 });
    const prepared = try chat.prepare(srv, cx, .{ .prompt = .{ .ids = ids }, .fields = .{ .object = fields }, .max_tokens = max_tokens, .temperature = 0 }, gone);
    return chat.generate(srv, cx, prepared, null, gone);
}

pub fn message(a: Allocator, role: []const u8, content: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}
