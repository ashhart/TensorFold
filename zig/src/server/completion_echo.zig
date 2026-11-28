//! ``echo`` on /v1/completions, beside vLLM's ``prompt_logprobs``: the prompt's own text and rows in a whole reply.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const chat = @import("chat.zig");
const logprobs = @import("logprobs.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

/// ``prompt_logprobs`` and ``echo`` from a body into `input`, each refused by name where it cannot be served.
pub fn plan(srv: *Server, cx: *Cx, body: Value, input: *chat.Input, completions: bool, streamed: bool) errors.Refused!void {
    const rows = srv.info.capability(.prompt_logprobs);
    input.prompt_logprobs = try fields.promptLogprobs(cx, body, completions, streamed, rows);
    const asked = (if (completions) body.field("echo") else null) orelse return; // chat routes do not read it
    if (asked != .bool) return cx.refuse("echo must be a boolean or null");
    if (!asked.bool) return;
    if (streamed) return cx.refuse("echo is supported on /v1/completions without streaming only");
    if (body.get("messages") != null) return cx.refuse("echo is supported with a prompt, not messages");
    if (!rows) return cx.refuse("echo is not supported by this model or backend");
    input.echo = true;
    input.prompt_only = noTokens(body);
    if (input.prompt_only) input.max_tokens = 1; // the engine still draws one token, which the reply drops
}

/// The body asks for no reply tokens: a ``max_tokens`` or ``max_completion_tokens`` of 0, and neither asks for more.
fn noTokens(body: Value) bool {
    if (json.truthyField(body, "max_tokens") or json.truthyField(body, "max_completion_tokens")) return false;
    return body.field("max_tokens") != null or body.field("max_completion_tokens") != null;
}

/// ``logprobs.project``, then the prompt's rows where they were asked: vLLM's form, and echo's ahead of the reply.
pub fn project(srv: *Server, cx: *Cx, reply: *chat.Reply, input: chat.Input, calls: ?[]Value) errors.Refused!void {
    try logprobs.project(srv, cx, reply, input.tools, calls);
    if (input.prompt_logprobs) |n| if (reply.prompt_rows) |rows| {
        reply.prompt_logprobs = try logprobs.promptValue(srv, cx.a, rows, n);
    };
    if (input.echo) try echo(srv, cx, reply, input);
}

/// The prompt ahead of the reply: its text, and with ``logprobs`` its tokens' entries, the first with none.
fn echo(srv: *Server, cx: *Cx, reply: *chat.Reply, input: chat.Input) errors.Refused!void {
    const a = cx.a;
    const ids = (try @import("prompt.zig").prepare(srv, cx, input, false, null)).ids; // the ids the engine read
    reply.content = try std.mem.concat(a, u8, &.{ try srv.text.decode(a, ids), reply.content });
    const n = input.logprobs orelse return;
    const rows = reply.prompt_rows orelse return cx.other("the engine gave no rows for the echoed prompt");
    const shown = reply.logprobs orelse return cx.other("the reply has no logprob entries to echo the prompt before");
    if (ids.len == 0 or rows.len != ids.len - 1) return cx.other("the engine's prompt rows are out of step with the prompt");
    const after = shown.object.get("content").?.array;
    const all = try a.alloc(Value, ids.len + after.len);
    const first = try json.newObject(a);
    try first.put(a, "token", .{ .string = try srv.text.decode(a, ids[0..1]) });
    try first.put(a, "logprob", .null);
    try first.put(a, "top_logprobs", .null);
    all[0] = .{ .object = first };
    for (all[1..ids.len], ids[1..], rows) |*slot, id, row| {
        var best = row;
        best.count = @min(row.count, n); // the engine may hold more for prompt_logprobs' count
        slot.* = logprobs.rowValue(srv, a, id, best, &.{}, false, .{}) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Failed => cx.other("the engine's prompt rows are out of step with the prompt"),
        };
    }
    @memcpy(all[ids.len..], after);
    try shown.object.put(a, "content", .{ .array = all });
}
