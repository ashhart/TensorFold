//! A reply that continues the final assistant message: the template's render without a generation prompt, cut there.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const prompt = @import("prompt.zig");
const images = @import("images.zig");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

/// Written after the final message's text in a second render, to find where the template puts that text.
const probe = "\u{2063}probe";

pub const thinking_on = "continuing a final assistant message needs thinking off, or the reply would open in a reasoning block: send chat_template_kwargs {\"enable_thinking\": false} (on /v1/messages, thinking {\"type\": \"disabled\"})";
pub const rewritten = "this chat template does not end the final assistant message with its text as sent, so the reply cannot continue it";
pub const calls = "a final assistant message with tool calls cannot be continued: send its text alone";
pub const not_assistant = "continue_final_message needs the last message to be an assistant message, which the reply continues";
pub const with_generation = "continue_final_message needs add_generation_prompt false or unset: the reply continues the final message";
pub const trailing = "final assistant content cannot end with trailing whitespace";

/// A chat body's ``continue_final_message`` (``add_generation_prompt`` false beside it): false when absent or false.
pub fn asked(cx: *Cx, body: Value, messages: Value) errors.Refused!bool {
    const flag = body.field("continue_final_message") orelse return false;
    if (flag != .bool) return cx.refuse("continue_final_message must be a boolean");
    if (!flag.bool) return false;
    if (body.field("add_generation_prompt")) |g| if (g != .bool or g.bool) return cx.refuse(with_generation);
    _ = try final(cx, messages);
    return true;
}

/// Whether a Messages conversation ends with the assistant's prefill, which the reply continues, as Messages does.
pub fn continues(cx: *Cx, messages: Value) errors.Refused!bool {
    const list = if (messages == .array) messages.array else return false;
    if (list.len == 0 or !std.mem.eql(u8, list[list.len - 1].strField("role") orelse "", "assistant")) return false;
    var text: std.ArrayList(u8) = .empty;
    const content = list[list.len - 1].get("content") orelse Value.null;
    if (content == .string) try text.appendSlice(cx.a, content.string);
    if (content == .array) for (content.array) |part| try text.appendSlice(cx.a, part.strField("text") orelse "");
    if (reply_text.pyRstrip(text.items).len != text.items.len) return cx.refuse(trailing);
    return true;
}

/// The final message's text, or the refusal naming why the conversation cannot be continued.
fn final(cx: *Cx, messages: Value) errors.Refused![]const u8 {
    const list = if (messages == .array) messages.array else &.{};
    const last = if (list.len > 0) list[list.len - 1] else Value.null;
    if (!std.mem.eql(u8, last.strField("role") orelse "", "assistant")) return cx.refuse(not_assistant);
    if (json.truthyField(last, "tool_calls")) return cx.refuse(calls);
    return last.strField("content") orelse cx.refuse("the final assistant message's content must be text to be continued");
}

/// The prompt that continues the final assistant message: its turn as the template renders it, up to its text's end.
pub fn ids(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8) errors.Refused![]const u32 {
    if (thinking and srv.info.capability(.reasoning)) return cx.refuse(thinking_on);
    const list = try prompt.templateMessages(srv, cx, messages);
    const text = try final(cx, list);
    const options = prompt.renderOptions(srv, cx, tools, thinking, effort, false);
    var problem: []const u8 = "";
    const whole = prompt.renderText(srv, cx.a, list, options, &problem) catch |e| return failed(cx, e, problem);
    const marked = try withText(cx, list, try std.mem.concat(cx.a, u8, &.{ text, probe }));
    const other = prompt.renderText(srv, cx.a, marked, options, &problem) catch |e| return failed(cx, e, problem);
    const at = cut(whole, other.text, text) orelse return cx.refuse(rewritten);
    const kept = try whole.slice(cx.a, 0, at);
    return kept.encode(srv.text, cx.a, &problem) catch |e| failed(cx, e, problem);
}

/// Where the message's text ends in `whole`: the probe renders there, only template text follows and the text ends.
fn cut(whole: model_text.Rendered, marked: []const u8, text: []const u8) ?usize {
    const at = std.mem.indexOfDiff(u8, whole.text, marked) orelse return null;
    if (marked.len != whole.text.len + probe.len or !std.mem.eql(u8, marked[at..][0..probe.len], probe)) return null;
    if (!std.mem.eql(u8, marked[at + probe.len ..], whole.text[at..]) or !whole.isTemplate(at, whole.text.len)) return null;
    // a template may drop the leading whitespace (Qwen's and GLM's strip the text), never the rest
    const kept = whole.text[0..at];
    return if (std.mem.endsWith(u8, kept, text) or std.mem.endsWith(u8, kept, reply_text.pyLstrip(text))) at else null;
}

/// A continued request's prompt with its cache hints: the conversation before the continued message, and its rewind.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!prompt.Rendered {
    const prompt_ids = try ids(srv, cx, input.messages, input.tools, thinking, effort);
    const history_len = before(srv, cx, input.messages, input.tools, thinking, effort, prompt_ids);
    const fetch = if (srv.config.vision_urls) images.Fetch.network(srv.io) else null;
    if (try images.expandWith(srv.info.vision, fetch, srv.text, cx, input.messages, prompt_ids, history_len, input.image_bytes)) |x|
        return .{ .ids = x.ids, .history_len = x.history_len, .images = x.images, .cache_key = x.cache_key, .frames = x.frames };
    const count = input.messages.array.len;
    const rewind_len = if (srv.info.prompt_cache) prompt.rewindLen(srv, cx, input.messages, count, input.tools, prompt_ids, thinking, effort) else 0;
    return .{ .ids = prompt_ids, .history_len = history_len, .rewind_len = rewind_len };
}

/// The rendered conversation before the continued message, which a next turn repeats; zero unless it leads the prompt.
fn before(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, prompt_ids: []const u32) usize {
    const list = messages.array;
    if (list.len < 2) return 0;
    var scratch: Cx = .{ .a = cx.a, .clear_thinking = cx.clear_thinking, .named_effort = cx.named_effort, .template_fields = cx.template_fields };
    const history = prompt.renderIds(srv, &scratch, .{ .array = list[0 .. list.len - 1] }, tools, thinking, effort, false) catch return 0;
    return if (history.len > 0 and history.len < prompt_ids.len and std.mem.startsWith(u32, prompt_ids, history)) history.len else 0;
}

/// `list` with its final message's text replaced by `text`.
fn withText(cx: *Cx, list: Value, text: []const u8) errors.Refused!Value {
    const out = try cx.a.dupe(Value, list.array);
    const last = try json.copyObject(cx.a, out[out.len - 1].object);
    try last.put(cx.a, "content", .{ .string = text });
    out[out.len - 1] = .{ .object = last };
    return .{ .array = out };
}

fn failed(cx: *Cx, e: model_text.Error, problem: []const u8) errors.Refused {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.fail(.request, "{s}", .{problem}),
    };
}

/// The final message a continued reply extends in the next turn: its text, then the reply's, as one assistant message.
pub fn joined(a: std.mem.Allocator, final_message: Value, reply: Value) std.mem.Allocator.Error!Value {
    const o = try json.copyObject(a, final_message.object);
    try o.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ final_message.strField("content") orelse "", reply.strField("content") orelse "" }) });
    if (reply.get("tool_calls")) |c| try o.put(a, "tool_calls", c);
    return .{ .object = o };
}

test "assistant prefill: the cut lies at the text's end, and a probe the template does not keep verbatim finds none" {
    const whole: model_text.Rendered = .{ .text = "<a>Sure<end>" };
    try std.testing.expectEqual(@as(?usize, 7), cut(whole, "<a>Sure" ++ probe ++ "<end>", "Sure"));
    try std.testing.expectEqual(@as(?usize, 7), cut(.{ .text = "<a>Sure" }, "<a>Sure" ++ probe, "\n Sure")); // stripped lead
    try std.testing.expectEqual(@as(?usize, null), cut(whole, "<a>SURE" ++ probe ++ "<end>", "Sure")); // rewritten
    try std.testing.expectEqual(@as(?usize, null), cut(whole, "<a>Sure " ++ probe ++ "<end>", "Sure ")); // trailing space dropped
    try std.testing.expectEqual(@as(?usize, null), cut(whole, "<a>Sure<end>", "Sure")); // the message not rendered
    try std.testing.expectEqual(@as(?usize, null), cut(whole, "<a>Sure" ++ probe ++ "<end>", "Shore")); // other text
    const after: model_text.Rendered = .{ .text = whole.text, .spans = &.{ .{ 3, 7 }, .{ 8, 10 } } };
    try std.testing.expectEqual(@as(?usize, null), cut(after, "<a>Sure" ++ probe ++ "<end>", "Sure")); // request text after
}
