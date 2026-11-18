//! A final assistant message continued through the Messages and chat routes, on checkpoint templates (prefill.zig).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const prefill = @import("prefill.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const exchange = @import("late_system_test.zig").exchange;
const family = @import("thinking_family_fixture.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

/// An engine that keeps each request's prompt, history and drafts switch, then writes `reply` and stops.
const Script = struct {
    reply: []const u32 = &.{},
    cache: bool = false,
    prompts: [8][2048]u32 = undefined,
    lens: [8]usize = @splat(0),
    history: [8]u32 = undefined,
    drafts: [8]bool = undefined,
    count: usize = 0,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 65536, .prompt_cache = s.cache, .warm_turns = s.cache };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        if (s.count < s.prompts.len) {
            @memcpy(s.prompts[s.count][0..request.prompt.len], request.prompt);
            s.lens[s.count] = request.prompt.len;
            s.history[s.count] = request.history_len;
            s.drafts[s.count] = request.drafts;
            s.count += 1;
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        if (request.max_tokens > 0 and s.reply.len > 0) sink.event(sink.ctx, id, &.{ .tokens = s.reply });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn seen(s: *const Script, i: usize) []const u32 {
        return s.prompts[i][0..s.lens[i]];
    }
};

/// A served checkpoint template over the lossless ASCII fixture vocabulary.
const Served = struct {
    fixture: @import("hf_eos_fixture.zig").Fixture,
    hf: *HfText,
    srv: *Server,
    script: *Script,

    fn init(a: Allocator, source: []const u8, config: []const u8, kind: []const u8, script: *Script) !Served {
        var f = try family.init(t.allocator, t.io, source, config, kind);
        errdefer f.deinit();
        var problem: []const u8 = "";
        const hf = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
        errdefer hf.deinit();
        const srv = try Server.init(t.allocator, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        return .{ .fixture = f, .hf = hf, .srv = srv, .script = script };
    }

    fn deinit(s: *Served) void {
        s.srv.deinit();
        s.hf.deinit();
        s.fixture.deinit();
    }

    /// One request through the server's routes: its status and the JSON body it answered.
    fn post(s: *Served, a: Allocator, path: []const u8, body: []const u8) !struct { u16, json.Value } {
        const sent = try exchange(s.srv, path, body);
        defer t.allocator.free(sent.body);
        return .{ sent.status, (try json.parse(a, try a.dupe(u8, sent.body))).ok };
    }
};

const Template = struct { name: []const u8, source: []const u8, config: []const u8, kind: []const u8, cut: []const u8 };

/// Each checkpoint's render of `turns`, cut where the assistant's "Sure: {" ends.
const templates = [_]Template{
    .{ .name = "Qwen3.8", .source = @embedFile("fixtures/thinking/qwen38.jinja"), .config = @embedFile("fixtures/thinking/qwen38-tokenizer_config.json"), .kind = "qwen3_5", .cut = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nSure: {" },
    .{ .name = "GLM-5.3", .source = @embedFile("fixtures/thinking/glm53.jinja"), .config = @embedFile("fixtures/thinking/glm53-tokenizer_config.json"), .kind = "glm5_next", .cut = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think></think>Sure: {" },
    .{ .name = "LFM2", .source = @embedFile("fixtures/thinking/lfm2.jinja"), .config = @embedFile("fixtures/thinking/lfm2-tokenizer_config.json"), .kind = "lfm2_moe", .cut = "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nSure: {" },
};
const turns = "[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Sure: {\"}]";
const continuation = " \"a\": 1}";
const off = "\"chat_template_kwargs\":{\"enable_thinking\":false}";

fn messages(a: Allocator, text: []const u8) !json.Value {
    return (try json.parse(a, text)).ok;
}

test "assistant prefill: Qwen3.8, GLM-5.3 and LFM2 templates are cut where the final assistant message's text ends" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (templates) |tmpl| {
        var script: Script = .{};
        var s = try Served.init(a, tmpl.source, tmpl.config, tmpl.kind, &script);
        defer s.deinit();
        var cx: errors.Cx = .{ .a = a };
        const ids = try prefill.ids(s.srv, &cx, try messages(a, turns), &.{}, false, null);
        t.expectEqualStrings(tmpl.cut, try s.srv.text.decode(a, ids)) catch |e| {
            std.debug.print("{s}: the cut differs\n", .{tmpl.name});
            return e;
        };
        if (std.mem.eql(u8, tmpl.name, "LFM2")) continue; // its earlier turns carry no think block, unlike its generation prompt
        // Qwen3.8's and GLM-5.3's: the thinking-off reply's own prompt, then the prefill's text
        const user = try messages(a, "[{\"role\":\"user\",\"content\":\"Hi\"}]");
        const plain = try prompt.renderIds(s.srv, &cx, user, &.{}, false, null, true);
        try t.expectEqualSlices(u32, try std.mem.concat(a, u32, &.{ plain, try s.srv.text.encode(a, "Sure: {", false) }), ids);
    }
}

test "assistant prefill: a Messages prefill is continued: content holds the continuation, usage and count_tokens the prefill" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (templates) |tmpl| {
        var script: Script = .{};
        var s = try Served.init(a, tmpl.source, tmpl.config, tmpl.kind, &script);
        defer s.deinit();
        script.reply = try s.srv.text.encode(a, continuation, false);
        var cx: errors.Cx = .{ .a = a };
        const ids = try prefill.ids(s.srv, &cx, try messages(a, turns), &.{}, false, null);
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,\"messages\":{s}}}", .{turns});
        const status, const reply = try s.post(a, "/v1/messages", body);
        try t.expectEqual(@as(u16, 200), status);
        try t.expectEqualSlices(u32, ids, script.seen(0));
        try t.expect(script.drafts[0]); // drafting stays on: a continued prompt is an ordinary prompt
        const content = reply.get("content").?.array;
        try t.expectEqual(@as(usize, 1), content.len);
        try t.expectEqualStrings(continuation, content[0].strField("text").?);
        try t.expectEqual(@as(?i64, @intCast(ids.len)), reply.get("usage").?.get("input_tokens").?.int64());
        try t.expectEqual(@as(?i64, @intCast(script.reply.len)), reply.get("usage").?.get("output_tokens").?.int64());
        const counted_status, const counted = try s.post(a, "/v1/messages/count_tokens", body);
        try t.expectEqual(@as(u16, 200), counted_status);
        try t.expectEqual(@as(?i64, @intCast(ids.len)), counted.get("input_tokens").?.int64());
        // the same ids as a raw completion prompt reach the engine unchanged
        var list: std.ArrayList(u8) = .empty;
        for (ids, 0..) |id, i| try list.print(a, "{s}{d}", .{ if (i == 0) "" else ",", id });
        const raw_status, _ = try s.post(a, "/v1/completions", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,\"prompt\":[{s}]}}", .{list.items}));
        try t.expectEqual(@as(u16, 200), raw_status);
        try t.expectEqualSlices(u32, script.seen(0), script.seen(1));
    }
}

test "assistant prefill: chat completions continue the final message only with continue_final_message" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var s = try Served.init(a, templates[0].source, templates[0].config, templates[0].kind, &script);
    defer s.deinit();
    script.reply = try s.srv.text.encode(a, continuation, false);
    var cx: errors.Cx = .{ .a = a };
    const continued = try prefill.ids(s.srv, &cx, try messages(a, turns), &.{}, false, null);
    const new_turn = try prompt.renderIds(s.srv, &cx, try messages(a, turns), &.{}, false, null, true);
    const cases = [_]struct { extra: []const u8, ids: []const u32 }{
        .{ .extra = "", .ids = new_turn }, // as before: the reply opens a turn after the assistant's
        .{ .extra = ",\"add_generation_prompt\":false", .ids = new_turn },
        .{ .extra = ",\"continue_final_message\":false", .ids = new_turn },
        .{ .extra = ",\"continue_final_message\":true", .ids = continued },
        .{ .extra = ",\"continue_final_message\":true,\"add_generation_prompt\":false", .ids = continued },
    };
    for (cases, 0..) |c, i| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,\"messages\":{s},{s}{s}}}", .{ turns, off, c.extra });
        const status, const reply = try s.post(a, "/v1/chat/completions", body);
        try t.expectEqual(@as(u16, 200), status);
        try t.expectEqualSlices(u32, c.ids, script.seen(i));
        const message = reply.get("choices").?.array[0].get("message").?;
        try t.expectEqualStrings(continuation, message.strField("content").?);
        try t.expectEqual(@as(?i64, @intCast(c.ids.len)), reply.get("usage").?.get("prompt_tokens").?.int64());
    }
}

test "assistant prefill: refusals name why the final message cannot be continued, before the engine sees it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var s = try Served.init(a, templates[0].source, templates[0].config, templates[0].kind, &script);
    defer s.deinit();
    const tool = "\"tools\":[{\"name\":\"look\",\"input_schema\":{\"type\":\"object\"}}]";
    const chat_tool = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"look\",\"parameters\":{\"type\":\"object\"}}}]";
    const flag = "\"continue_final_message\":true";
    const cases = [_]struct { path: []const u8, body: []const u8, message: []const u8 }{
        .{ .path = "/v1/messages", .body = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Sure: \"}]", .message = prefill.trailing },
        .{ .path = "/v1/messages", .body = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Sure\"},{\"type\":\"text\",\"text\":\"\\n\"}]}]", .message = prefill.trailing },
        .{ .path = "/v1/messages", .body = "\"messages\":" ++ turns ++ ",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":4}", .message = prefill.thinking_on },
        .{ .path = "/v1/messages/count_tokens", .body = "\"messages\":" ++ turns ++ ",\"thinking\":{\"type\":\"adaptive\"}", .message = prefill.thinking_on },
        .{ .path = "/v1/messages", .body = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"look\",\"input\":{}}]}]," ++ tool, .message = prefill.calls },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," ++ flag ++ "," ++ off, .message = prefill.not_assistant },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":" ++ turns ++ "," ++ flag ++ ",\"add_generation_prompt\":true," ++ off, .message = prefill.with_generation },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":" ++ turns ++ ",\"continue_final_message\":\"yes\"," ++ off, .message = "continue_final_message must be a boolean" },
        // the request itself turns thinking on: a switch, DeepSeek's form, the top-level toggle or a named effort
        .{ .path = "/v1/chat/completions", .body = "\"messages\":" ++ turns ++ "," ++ flag ++ ",\"chat_template_kwargs\":{\"enable_thinking\":true}", .message = prefill.thinking_on },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":" ++ turns ++ "," ++ flag ++ ",\"chat_template_kwargs\":{\"thinking\":{\"type\":\"enabled\"}}", .message = prefill.thinking_on },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":" ++ turns ++ "," ++ flag ++ ",\"thinking\":true", .message = prefill.thinking_on },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":" ++ turns ++ "," ++ flag ++ ",\"reasoning_effort\":\"high\"", .message = prefill.thinking_on },
        .{ .path = "/v1/messages", .body = "\"messages\":" ++ turns ++ ",\"thinking\":{\"type\":\"adaptive\"}", .message = prefill.thinking_on },
        .{ .path = "/v1/chat/completions", .body = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"look\",\"arguments\":\"{}\"}}]}]," ++ flag ++ "," ++ off ++ "," ++ chat_tool, .message = prefill.calls },
        // Qwen3.8's template trims the message: its trailing space would be lost, so the reply cannot continue it
        .{ .path = "/v1/chat/completions", .body = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Sure: \"}]," ++ flag ++ "," ++ off, .message = prefill.rewritten },
    };
    for (cases) |c| {
        const status, const reply = try s.post(a, c.path, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,{s}}}", .{c.body}));
        try t.expectEqual(@as(u16, 400), status);
        try t.expectEqualStrings(c.message, reply.get("error").?.strField("message").?);
    }
    try t.expectEqual(@as(usize, 0), script.count);
}

test "assistant prefill: a template that rewrites or drops the final message's text is refused; one that strips its lead is cut" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sources = [_][]const u8{
        "{% for m in messages %}<{{ m.role }}>{{ m.content | upper }}{% endfor %}",
        "{% for m in messages %}{% if not (loop.last and m.role == 'assistant') %}<{{ m.role }}>{{ m.content }}{% endif %}{% endfor %}",
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% if loop.last %}({{ m.content }}){% endif %}{% endfor %}",
    };
    for (sources) |source| {
        var script: Script = .{};
        var s = try Served.init(a, source, "{}", "test", &script);
        defer s.deinit();
        var cx: errors.Cx = .{ .a = a };
        try t.expectError(error.Refused, prefill.ids(s.srv, &cx, try messages(a, turns), &.{}, false, null));
        try t.expectEqualStrings(prefill.rewritten, cx.message);
    }
    var script: Script = .{};
    var s = try Served.init(a, templates[0].source, templates[0].config, templates[0].kind, &script);
    defer s.deinit();
    var cx: errors.Cx = .{ .a = a };
    const lead = try messages(a, "[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"\\n\\nSure: {\"}]");
    try t.expectEqualStrings(templates[0].cut, try s.srv.text.decode(a, try prefill.ids(s.srv, &cx, lead, &.{}, false, null)));
}

test "assistant prefill: the prompt cache keeps the history before the continued message, and the next turn's warm pass joins it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // an explicit off and the server's default (thinking on, which a continued request does not take) cache alike
    for ([_][]const u8{ "," ++ off, "" }) |extra| {
        var script: Script = .{ .cache = true };
        var s = try Served.init(a, templates[0].source, templates[0].config, templates[0].kind, &script);
        defer s.deinit();
        script.reply = try s.srv.text.encode(a, continuation, false);
        var cx: errors.Cx = .{ .a = a };
        const user = try messages(a, "[{\"role\":\"user\",\"content\":\"Hi\"}]");
        const history = try prompt.renderIds(s.srv, &cx, user, &.{}, false, null, false);
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,\"messages\":{s},\"continue_final_message\":true{s}}}", .{ turns, extra });
        const status, _ = try s.post(a, "/v1/chat/completions", body);
        try t.expectEqual(@as(u16, 200), status);
        try t.expectEqualStrings(templates[0].cut, try s.srv.text.decode(a, script.seen(0)));
        try t.expectEqual(@as(u32, @intCast(history.len)), script.history[0]);
        try t.expectEqual(@as(usize, 2), script.count);
        // the warm pass renders the next turn's shared prompt: the continued message and the reply as one turn
        const next = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nSure: {" ++ continuation ++ "<|im_end|>\n<|im_start|>user\n";
        try t.expectEqualStrings(next, try s.srv.text.decode(a, script.seen(1)));
    }
}

test "assistant prefill: a server's default thinking leaves a continued request thinking off, its reply all content" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (templates) |tmpl| {
        var script: Script = .{};
        var s = try Served.init(a, tmpl.source, tmpl.config, tmpl.kind, &script);
        defer s.deinit();
        try t.expect(s.srv.config.enable_thinking); // the server's default: replies think unless a request turns it off
        script.reply = try s.srv.text.encode(a, continuation, false);
        var cx: errors.Cx = .{ .a = a };
        const ids = try prefill.ids(s.srv, &cx, try messages(a, turns), &.{}, false, null);
        // a thinking reply would read the continuation as reasoning: it has no close marker
        const chat_status, const chat = try s.post(a, "/v1/chat/completions", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,\"messages\":{s},\"continue_final_message\":true}}", .{turns}));
        try t.expectEqual(@as(u16, 200), chat_status);
        try t.expectEqualSlices(u32, ids, script.seen(0));
        const message = chat.get("choices").?.array[0].get("message").?;
        try t.expectEqualStrings(continuation, message.strField("content").?);
        try t.expect(message.get("reasoning_content") == null);
        try t.expect(!chat.get("tensorfold").?.get("enable_thinking").?.bool);
        const status, const reply = try s.post(a, "/v1/messages", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":16,\"messages\":{s}}}", .{turns}));
        try t.expectEqual(@as(u16, 200), status);
        try t.expectEqualSlices(u32, ids, script.seen(1));
        const content = reply.get("content").?.array;
        try t.expectEqual(@as(usize, 1), content.len);
        try t.expectEqualStrings("text", content[0].strField("type").?);
        try t.expectEqualStrings(continuation, content[0].strField("text").?);
    }
}

test "assistant prefill: DeepSeek-V4's encoder is cut after the message's text, its plain thinking-off prompt and the text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try @import("hf_eos_fixture.zig").Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    const vocab = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = vocab });
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v4\"}" });
    var problem: []const u8 = "";
    const hf = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{};
    const srv = try Server.init(t.allocator, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    const whole = try hf.text().render(a, try messages(a, turns), .{ .add_generation_prompt = false }, &problem);
    try t.expectEqualStrings("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>Hi<\u{ff5c}Assistant\u{ff5c}></think>Sure: {<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", whole.text);
    var cx: errors.Cx = .{ .a = a };
    const ids = try prefill.ids(srv, &cx, try messages(a, turns), &.{}, false, null);
    const plain = try prompt.renderIds(srv, &cx, try messages(a, "[{\"role\":\"user\",\"content\":\"Hi\"}]"), &.{}, false, null, true);
    try t.expectEqualSlices(u32, try std.mem.concat(a, u32, &.{ plain, try srv.text.encode(a, "Sure: {", false) }), ids);
}

test "assistant prefill: Kimi's renderer is cut inside the response it opens, its plain thinking-off prompt and the text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const kimi = try @import("kimi_text_test.zig").synthetic(t.allocator);
    defer kimi.deinit();
    var script: Script = .{};
    const srv = try Server.init(t.allocator, t.io, script.engine(), kimi.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const ids = try prefill.ids(srv, &cx, try messages(a, turns), &.{}, false, null);
    try t.expectEqualStrings("<|open|>message role=\"user\"<|sep|>Hi<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>Sure: {", try srv.text.decode(a, ids));
    const plain = try prompt.renderIds(srv, &cx, try messages(a, "[{\"role\":\"user\",\"content\":\"Hi\"}]"), &.{}, false, null, true);
    try t.expectEqualSlices(u32, try std.mem.concat(a, u32, &.{ plain, try srv.text.encode(a, "Sure: {", false) }), ids);
}

test "assistant prefill: real tokenizers (TF_TOKENIZER_DIRS) end the cut with the message's own tokens after its history" {
    const list = std.c.getenv("TF_TOKENIZER_DIRS") orelse {
        std.debug.print("assistant prefill: skipped, TF_TOKENIZER_DIRS names no tokenizer directories\n", .{});
        return error.SkipZigTest;
    };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const conversation = "[{\"role\":\"user\",\"content\":\"What is two plus two?\"},{\"role\":\"assistant\",\"content\":\"The answer is\"}]";
    var it = std.mem.tokenizeScalar(u8, std.mem.span(list), std.fs.path.delimiter);
    while (it.next()) |dir| {
        var problem: []const u8 = "";
        const hf = try HfText.load(t.allocator, t.io, dir, null, a, &problem);
        defer hf.deinit();
        var script: Script = .{};
        const srv = try Server.init(t.allocator, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        defer srv.deinit();
        var cx: errors.Cx = .{ .a = a };
        const name = std.fs.path.basename(dir);
        const ids = prefill.ids(srv, &cx, try messages(a, conversation), &.{}, false, null) catch |e| {
            std.debug.print("assistant prefill: {s}: {s}\n", .{ name, cx.message });
            return e;
        };
        const own = try srv.text.encodePlain(a, "The answer is");
        const history = try prompt.renderIds(srv, &cx, try messages(a, "[{\"role\":\"user\",\"content\":\"What is two plus two?\"}]"), &.{}, false, null, false);
        std.debug.print("assistant prefill: {s}: {d} prompt tokens, its last {d} the message's own, after {d} of history\n", .{ name, ids.len, own.len, history.len });
        try t.expect(std.mem.endsWith(u32, ids, own));
        try t.expect(history.len + own.len < ids.len and std.mem.startsWith(u32, ids, history));
    }
}
