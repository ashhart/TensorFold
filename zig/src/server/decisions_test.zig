//! Request checks for /v1/decisions, and a fake engine hook that scores one choice.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const openai = @import("openai.zig");
const model_text = @import("model_text.zig");
const decisions = @import("decisions.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;

const ByteText = struct {
    open_think: bool = false,

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(ctx: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        const t: *ByteText = @ptrCast(@alignCast(ctx));
        return a.dupe(u8, if (t.open_think) "prompt<think>" else "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

const Backend = struct {
    window: u32 = 0,
    hooked: bool = false,
    calls: usize = 0,
    prompt_len: usize = 0,
    label_len: usize = 0,

    fn engine(b: *Backend) api.Engine {
        return .{ .ctx = b, .vtable = if (b.hooked) &hooked_vt else &bare_vt };
    }

    fn info(ctx: *anyopaque) api.Info {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .context_window = b.window };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64) error{Failed}!f64 {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.calls += 1;
        b.prompt_len = prompt.len;
        b.label_len = labels.len;
        if (logits.len > 0) logits[0] = 0;
        if (logits.len > 1) logits[1] = 3;
        return 4;
    }
};

const bare_vt: api.Engine.VTable = .{ .info = Backend.info, .submit = Backend.submit, .cancel = Backend.cancel, .status = Backend.status, .memory = Backend.memory };
const hooked_vt: api.Engine.VTable = .{ .info = Backend.info, .submit = Backend.submit, .cancel = Backend.cancel, .status = Backend.status, .memory = Backend.memory, .score = Backend.score };

fn serve(text: *ByteText, backend: *Backend) !*Server {
    return Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
}

fn messageOf(srv: *Server, a: Allocator, body: []const u8) ![]const u8 {
    const raw = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, decisions.reply(srv, &cx, raw));
    return cx.message;
}

test "a decisions body is refused with 0.6.6's sentences" {
    var text: ByteText = .{};
    var backend: Backend = .{};
    const srv = try serve(&text, &backend);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const raw = (try json.parse(a, "{\"input\":\"x\"}")).ok;
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, decisions.reply(srv, &cx, raw));
    try std.testing.expectEqualStrings("questions must contain at least one question", cx.message);
    const err = try openai.wrapError(a, try openai.errorBody(a, &cx, null));
    try std.testing.expectEqualStrings("{\"error\": {\"message\": \"questions must contain at least one question\", \"type\": \"invalid_request_error\"}}", try json.stringify(a, err, .{}));

    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{\"input\":\"x\",\"temperature\":0}", "temperature must be a number above 0" },
        .{ "{\"input\":\"x\",\"temperature\":true}", "temperature must be a finite number or null" },
        .{ "{\"input\":\"x\",\"temperature\":\"0\"}", "temperature must be a number above 0" },
        .{ "{\"input\":\"x\",\"temperature\":\"hot\"}", "temperature must be a finite number or null" },
        .{ "{\"input\":\"x\",\"temperature\":\"1\"}", "questions must contain at least one question" },
        .{ "{\"input\":\"x\",\"prompt_format_version\":2}", "prompt_format_version 2 is not served, this server uses version 1" },
        .{ "{\"input\":\"x\",\"prompt_format_version\":false}", "prompt_format_version False is not served, this server uses version 1" },
        .{ "{\"input\":\"x\",\"prompt_format_version\":\"1\"}", "prompt_format_version 1 is not served, this server uses version 1" },
        .{ "{\"input\":\"x\",\"prompt_format_version\":1.0}", "questions must contain at least one question" },
        .{ "{\"input\":\"x\",\"prompt_format_version\":null}", "questions must contain at least one question" },
        .{ "{\"input\":\"x\",\"prompt_format_version\":true}", "questions must contain at least one question" },
        .{ "{\"zzz\":1,\"aaa\":true}", "unknown field 'aaa'" },
        .{ "{\"input\":\"x\",\"chat_template_kwargs\":[]}", "chat_template_kwargs must be an object" },
        .{ "{\"input\":\"x\",\"chat_template_kwargs\":{\"enable_thinking\":true}}", "decisions need enable_thinking false or unset" },
        .{ "{\"input\":\"x\",\"chat_template_kwargs\":{\"enable_thinking\":false}}", "questions must contain at least one question" },
        .{ "{\"input\":\"x\",\"questions\":[1]}", "each question must be an object" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"  \"}]}", "a question id must not be blank" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\"},{\"id\":\"q\"}]}", "question id 'q' is repeated" },
        .{ "{\"input\":\"  \",\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"p\",\"options\":[{\"name\":\"a\"},{\"name\":\"b\"}]}]}", "input must not be blank" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"nope\"}]}", "question 'q': unknown question type 'nope'" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\"}]}", "question 'q': unknown question type None" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"p\",\"options\":[{\"name\":\"a\"}]}]}", "question 'q': a choice needs 2 to 26 options" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"p\",\"options\":[{\"name\":\"Red\"},{\"name\":\"red\"}]}]}", "question 'q': option name 'red' repeats another name" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"p\",\"options\":[{\"name\":\"a\\nb\"},{\"name\":\"c\"}]}]}", "question 'q': option names must not be blank or contain line breaks" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"  \",\"options\":[{\"name\":\"a\"},{\"name\":\"b\"}]}]}", "question 'q': a question must not be blank" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"p\",\"options\":[{\"name\":\"a\",\"extra\":1},{\"name\":\"b\"}]}]}", "question 'q': unknown field 'extra'" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"yes_no\",\"question\":\"  \"}]}", "question 'q': a question must not be blank" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"score\",\"question\":\"rate\",\"levels\":[\"only\"]}]}", "question 'q': a score needs 2 to 10 levels" },
        .{ "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"score\",\"question\":\"rate\",\"levels\":[\"good\",\" \"]}]}", "question 'q': a level must not be blank" },
        .{ "[]", "'list' object has no attribute 'get'" },
    };
    for (cases) |case| try std.testing.expectEqualStrings(case[1], try messageOf(srv, a, case[0]));
}

const choice_body = "{\"input\":\"x\",\"model\":\"m\",\"return_prompt_token_ids\":true,\"questions\":[{\"id\":\"q\",\"type\":\"choice\",\"question\":\"pick\",\"options\":[{\"name\":\"red\"},{\"name\":\"blue\",\"description\":\"sky\"}]}]}";

test "the engine hook scores a choice and a missing hook is refused" {
    var text: ByteText = .{};
    var backend: Backend = .{ .hooked = true };
    const srv = try serve(&text, &backend);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const raw = (try json.parse(a, choice_body)).ok;
    var cx: errors.Cx = .{ .a = a };
    const payload = try decisions.reply(srv, &cx, raw);
    try std.testing.expectEqualStrings("decisions", payload.get("object").?.string);
    try std.testing.expectEqualStrings("m", payload.get("model").?.string);
    try std.testing.expectEqualStrings("1", payload.get("prompt_format_version").?.int);
    const answer = payload.get("answers").?.get("q").?;
    try std.testing.expectEqualStrings("choice", answer.get("type").?.string);
    try std.testing.expectEqualStrings("blue", answer.get("choice").?.string);
    try std.testing.expect(answer.get("probabilities").?.get("blue").?.float > answer.get("probabilities").?.get("red").?.float);
    try std.testing.expectEqual(@as(usize, 2), answer.get("label_token_ids").?.array.len);
    try std.testing.expectEqualStrings("65", answer.get("label_token_ids").?.array[0].int);
    try std.testing.expectEqualStrings("66", answer.get("label_token_ids").?.array[1].int);
    try std.testing.expectEqual(@as(usize, 6), answer.get("prompt_token_ids").?.array.len);
    const usage = payload.get("usage").?;
    try std.testing.expectEqualStrings("6", usage.get("prompt_tokens").?.int);
    try std.testing.expectEqualStrings("0", usage.get("completion_tokens").?.int);
    try std.testing.expectEqualStrings("6", usage.get("total_tokens").?.int);
    try std.testing.expectEqual(@as(usize, 1), backend.calls);
    try std.testing.expectEqual(@as(usize, 6), backend.prompt_len);
    try std.testing.expectEqual(@as(usize, 2), backend.label_len);

    const score_raw = (try json.parse(a, "{\"input\":\"x\",\"questions\":[{\"id\":\"s\",\"type\":\"score\",\"question\":\"rate\",\"levels\":[\"bad\",\"good\"]}]}")).ok;
    const scored = try decisions.reply(srv, &cx, score_raw);
    const score = scored.get("answers").?.get("s").?;
    try std.testing.expectEqualStrings("score", score.get("type").?.string);
    try std.testing.expect(score.get("choice") == null);
    try std.testing.expect(score.get("score").?.float > 0.9);

    try std.testing.expectEqualStrings("question 'q': the answer label 'yes' is not one distinct token after the chat prompt for this tokenizer, so this model is not supported", try messageOf(srv, a, "{\"input\":\"x\",\"questions\":[{\"id\":\"q\",\"type\":\"yes_no\",\"question\":\"rain\"}]}"));

    text.open_think = true;
    try std.testing.expectEqualStrings("question 'q': the chat template leaves a reasoning block open at the answer position", try messageOf(srv, a, choice_body));
}

test "a prompt past the context window is refused before scoring" {
    var text: ByteText = .{};
    var backend: Backend = .{ .hooked = true, .window = 4 };
    const srv = try serve(&text, &backend);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("question 'q': the prompt has 6 tokens, which does not fit the context length of 4 tokens", try messageOf(srv, arena.allocator(), choice_body));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
}

test "an engine with no score hook refuses after the checks pass" {
    var text: ByteText = .{};
    var backend: Backend = .{};
    const srv = try serve(&text, &backend);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("decision scoring is not supported by this engine yet", try messageOf(srv, arena.allocator(), choice_body));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
}
