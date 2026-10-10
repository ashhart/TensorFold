//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const images = @import("images.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

pub const Rendered = struct { ids: []const u32, history_len: usize = 0, rewind_len: usize = 0 };

pub const isTitle = messages_mod.isTitleRequest;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    const normalized = try messages_mod.toolArguments(cx, try messages_mod.normalizeWith(cx, messages, srv.late_system, srv.needs_user_after_tool, srv.info.vision != null));
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
    };
    var ids = render(srv, cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    return ids;
}

/// The template's ids for `messages`; with image input, a marker the messages' text spells is ordinary text (images.zig).
fn render(srv: *Server, a: std.mem.Allocator, messages: Value, options: model_text.RenderOptions, problem: *[]const u8) model_text.Error![]u32 {
    const v = srv.info.vision orelse return srv.text.renderIds(a, messages, options, problem);
    const markers = try images.markerText(srv.text, a, v);
    const text = try srv.text.render(a, try images.shield(a, messages, markers), options, problem);
    return images.encodeShielded(srv.text, a, text, markers);
}

/// The template's generation prompt (thinking on or off): what a probe conversation renders past its history.
fn generationPrompt(srv: *Server, cx: *Cx, thinking: bool) errors.Refused![]const u32 {
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "x" });
    const list = try cx.a.alloc(Value, 1);
    list[0] = .{ .object = user };
    const with = try renderIds(srv, cx, .{ .array = list }, &.{}, thinking, null, true);
    const without = try renderIds(srv, cx, .{ .array = list }, &.{}, thinking, null, false);
    if (without.len >= with.len or !std.mem.eql(u32, with[0..without.len], without)) return &.{};
    return with[without.len..];
}

/// Earlier replies' decoded rows: from each turn's generation prompt end through its closing stop token.
pub fn replySpans(srv: *Server, cx: *Cx, prompt: []const u32) errors.Refused![]const [2]u32 {
    if (srv.eos.len == 0) return &.{};
    var probe: Cx = .{ .a = cx.a }; // a template that refuses the probe leaves the request without spans, not failed
    const openers = [2][]const u32{ generationPrompt(srv, &probe, false) catch &.{}, generationPrompt(srv, &probe, true) catch &.{} };
    var spans: std.ArrayList([2]u32) = .empty;
    var i: usize = 0;
    while (i < prompt.len) {
        var start: ?usize = null;
        var at_min: usize = prompt.len;
        for (openers) |o| if (o.len > 0) if (std.mem.indexOfPos(u32, prompt, i, o)) |at| if (at < at_min) {
            at_min = at;
            start = at + o.len;
        };
        const first = start orelse break;
        const end = std.mem.indexOfAnyPos(u32, prompt, first, srv.eos) orelse break;
        try spans.append(cx.a, .{ @intCast(first), @intCast(end + 1) });
        i = end + 1;
    }
    return spans.items;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    const prompt = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, true);
    const history = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, false);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    const count = if (input.messages == .array) input.messages.array.len else 0;
    const rewind_len = if (srv.info.prompt_cache) rewindLen(srv, cx, input.messages, count, input.tools, prompt, thinking, effort) else 0; // one more render
    return .{ .ids = prompt, .history_len = history_len, .rewind_len = rewind_len };
}

/// Where the latest user message's text starts (of the first `real_count`): the prompt and a probe there agree to it.
pub fn rewindLen(srv: *Server, cx: *Cx, messages: Value, real_count: usize, tools: []const Value, prompt: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array or real_count > messages.array.len) return 0;
    const user_index = for (0..real_count) |n| {
        const i = real_count - 1 - n;
        const role = messages.array[i].strField("role") orelse continue;
        if (std.mem.eql(u8, role, "user")) break i;
    } else return 0;
    if (messages.array[user_index] != .object) return 0;
    const list = cx.a.dupe(Value, messages.array) catch return 0;
    const user = json.copyObject(cx.a, messages.array[user_index].object) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    list[user_index] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const probe = renderIds(srv, &scratch, .{ .array = list }, tools, thinking, effort, true) catch return 0;
    const i = std.mem.indexOfDiff(u32, prompt, probe) orelse return 0;
    return if (i < prompt.len) i else 0;
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}
