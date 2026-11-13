//! The next turn's background prefill renders the reply as its client will send it back, reasoning included or not.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const late = @import("late_system_test.zig");

/// Renders each message as `[role]`, `<r>reasoning</r>` when it has some, then its content and tool calls.
const Shown = struct {
    fn text(t: *@This()) model_text.Text {
        var base: late.DropLate = .{};
        var vt = base.text().vtable.*;
        vt.render = render;
        const held = std.testing.allocator.create(model_text.Text.VTable) catch unreachable;
        held.* = vt;
        return .{ .ctx = t, .vtable = held };
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        if (messages != .array) return .{ .text = out.items };
        for (messages.array) |m| {
            try out.print(a, "[{s}]", .{late.textOf(m, "role")});
            const r = late.textOf(m, "reasoning_content");
            if (r.len > 0) try out.print(a, "<r>{s}</r>", .{r});
            try out.print(a, "{s}\n", .{late.textOf(m, "content")});
            if (m.get("tool_calls")) |calls| {
                try out.appendSlice(a, try json.stringify(a, calls, .{ .compact = true }));
                try out.append(a, '\n');
            }
        }
        return .{ .text = out.items };
    }
};

/// late_system_test's Capture with the reasoning capability: replies split their reasoning at `</think>`.
const Thinking = struct {
    inner: late.Capture,

    fn engine(e: *@This()) api.Engine {
        var vt = e.inner.engine().vtable.*;
        vt.info = info;
        const held = std.testing.allocator.create(api.Engine.VTable) catch unreachable;
        held.* = vt;
        return .{ .ctx = &e.inner, .vtable = held };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .warm_turns = true, .reasoning = true };
    }
};

const reply = "Because.</think>Before.\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}</tool_call>";
const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]}";

fn warmed(e: *Thinking) ![]u8 {
    const text = try std.testing.allocator.alloc(u8, e.inner.background_prompt.len);
    for (e.inner.background_prompt, text) |id, *byte| byte.* = @intCast(id);
    return text;
}

test "warm turns: the guess carries the reply's reasoning unless the client's history drops it" {
    var shown: Shown = .{};
    const t = shown.text();
    defer std.testing.allocator.destroy(t.vtable);
    var backend: Thinking = .{ .inner = .{ .reply = reply, .warm_turns = true } };
    const eng = backend.engine();
    defer std.testing.allocator.destroy(eng.vtable);
    defer std.testing.allocator.free(backend.inner.prompt);
    defer std.testing.allocator.free(backend.inner.background_prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, eng, t, .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // a first turn: no assistant message yet, so the guess takes the reasoning the client was sent
    const first = try late.exchange(srv, "/v1/chat/completions", "{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"go\"}]" ++ tools);
    defer std.testing.allocator.free(first.body);
    try std.testing.expectEqual(@as(u16, 200), first.status);
    const message = (try json.parse(a, first.body)).ok.get("choices").?.array[0].get("message").?;
    try std.testing.expectEqualStrings("Because.", message.get("reasoning_content").?.string);
    const w1 = try warmed(&backend);
    defer std.testing.allocator.free(w1);
    try std.testing.expect(std.mem.indexOf(u8, w1, "[assistant]<r>Because.</r>Before.") != null);

    // a client that sent reasoning back keeps getting it in the guess
    const echoing = "{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"go\"},{\"role\":\"assistant\",\"content\":\"Before.\",\"reasoning_content\":\"Earlier.\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"x\"}]" ++ tools;
    const second = try late.exchange(srv, "/v1/chat/completions", echoing);
    defer std.testing.allocator.free(second.body);
    try std.testing.expectEqual(@as(u16, 200), second.status);
    const w2 = try warmed(&backend);
    defer std.testing.allocator.free(w2);
    try std.testing.expect(std.mem.indexOf(u8, w2, "[assistant]<r>Earlier.</r>Before.") != null);
    try std.testing.expect(std.mem.indexOf(u8, w2, "[assistant]<r>Because.</r>Before.") != null);

    // a client whose history dropped reasoning gets a guess without it, as it will send this reply back
    const dropping = "{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"go\"},{\"role\":\"assistant\",\"content\":\"Before.\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"x\"}]" ++ tools;
    const third = try late.exchange(srv, "/v1/chat/completions", dropping);
    defer std.testing.allocator.free(third.body);
    try std.testing.expectEqual(@as(u16, 200), third.status);
    const w3 = try warmed(&backend);
    defer std.testing.allocator.free(w3);
    try std.testing.expect(std.mem.indexOf(u8, w3, "<r>Because.</r>") == null);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, w3, "[assistant]Before."));
}

test "warm turns: echoesReasoning reads the latest assistant message, and a conversation with none echoes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const echoes = @import("warm.zig").echoesReasoning;
    try std.testing.expect(echoes((try json.parse(a, "[{\"role\":\"user\",\"content\":\"go\"}]")).ok));
    try std.testing.expect(echoes((try json.parse(a, "[{\"role\":\"assistant\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\",\"reasoning_content\":\"\"},{\"role\":\"tool\",\"content\":\"x\"}]")).ok));
    try std.testing.expect(!echoes((try json.parse(a, "[{\"role\":\"assistant\",\"content\":\"a\",\"reasoning_content\":\"r\"},{\"role\":\"assistant\",\"content\":\"b\"},{\"role\":\"user\",\"content\":\"x\"}]")).ok));
    try std.testing.expect(!echoes(.null));
}
