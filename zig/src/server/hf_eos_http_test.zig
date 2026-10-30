//! Actual HfText and OpenAI HTTP-handler EOS paths with an inert scripted engine, without sockets or a GPU.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const fixture = @import("hf_eos_fixture.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const tokens = @import("tokens.zig");
const Allocator = std.mem.Allocator;

const Script = struct {
    a: Allocator,
    ids: []const u32,
    seen_eos: []const u32 = &.{},
    seen_prompt: []const u32 = &.{},
    reasoning: bool = true,
    reason: api.Reason = .length,
    chunk: usize = 0, // tokens an event (0: all in one)
    structured: bool = false, // the request carried a grammar
    after: ?u32 = null, // the token the grammar starts after

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 262144, .reasoning = s.reasoning };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen_eos = s.a.dupe(u32, request.eos) catch return error.Busy;
        s.seen_prompt = s.a.dupe(u32, request.prompt) catch return error.Busy;
        s.structured = request.structure != null;
        if (request.structure) |st| s.after = st.after;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        const step = if (s.chunk == 0) s.ids.len else s.chunk;
        var at: usize = 0;
        while (at < s.ids.len) : (at += step) sink.event(sink.ctx, id, &.{ .tokens = s.ids[at..@min(s.ids.len, at + step)] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = s.reason } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    a: Allocator,
    status: ?u16 = null,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,
    opened: bool = false,
    done: bool = false,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.opened = true;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        if (value) |v| {
            r.events.append(r.a, v) catch return error.Closed;
        } else r.done = true;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status_code;
        r.whole = value;
    }
};

fn check(first: u32, declared: bool, ignore: bool, streaming: bool, leading: bool) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, io, if (declared) fixture.generation else null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, io, f.path, null, a, &problem);
    defer text.deinit();
    const ids = if (leading) [_]u32{ 1, first, 2 } else [_]u32{ first, 1, 2 };
    var script = Script{ .a = a, .ids = &ids };
    const srv = try Server.init(gpa, io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":3,\"stream\":{s},\"ignore_eos\":{s}}}", .{ if (streaming) "true" else "false", if (ignore) "true" else "false" }))).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    const ends = !ignore and (first == 127906 or declared);
    const expected_text: []const u8 = if (ends) (if (leading) "O" else "") else if (leading)
        (if (first == 127901) "O<|endoftext|>K" else "O<|im_end|>K")
    else if (first == 127901) "<|endoftext|>OK" else "<|im_end|>OK";
    const expected_ids: []const u32 = if (ends) ids[0..@as(usize, if (leading) 2 else 1)] else &ids;
    const expected_sha = tokens.tokenSha256(expected_ids);
    var content: std.ArrayList(u8) = .empty;
    var final: ?json.Value = response.whole;
    if (streaming) {
        try std.testing.expect(response.opened and response.done and response.whole == null);
        for (response.events.items) |value| {
            const choices = value.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            const choice = choices.array[0];
            try content.appendSlice(a, json.strOr(choice.get("text")));
            if (choice.get("finish_reason")) |reason| {
                if (reason == .string) final = value;
            }
        }
    } else {
        try std.testing.expectEqual(@as(?u16, 200), response.status);
        try content.appendSlice(a, json.strOr(response.whole.?.get("choices").?.array[0].get("text")));
    }
    const value = final orelse return error.MissingFinalReply;
    const choice = value.get("choices").?.array[0];
    try std.testing.expectEqualStrings(expected_text, content.items);
    try std.testing.expectEqualStrings(if (ends) "stop" else "length", json.strOr(choice.get("finish_reason")));
    try std.testing.expectEqual(@as(i64, @intCast(expected_ids.len)), value.get("usage").?.get("completion_tokens").?.int64().?);
    try std.testing.expectEqualStrings(&expected_sha, json.strOr(value.get("tensorfold").?.get("token_sha256")));
    try std.testing.expectEqualSlices(u32, if (ignore) &.{} else if (declared) &.{ 127906, 127901 } else &.{127906}, script.seen_eos);
}

test "EOS policy: HTTP stops on both checkpoint EOS ids with exact usage hashes and streamed whole parity" {
    for ([_]u32{ 127906, 127901 }) |first| for ([_]bool{ false, true }) |streaming|
        try check(first, true, false, streaming, false);
}

test "EOS policy: HTTP ignore_eos preserves tokens and a pad not declared as EOS keeps decoding" {
    for ([_]u32{ 127906, 127901 }) |first| for ([_]bool{ false, true }) |streaming|
        try check(first, true, true, streaming, false);
    for ([_]bool{ false, true }) |streaming| try check(127901, false, false, streaming, false);
}

test "EOS policy: HTTP preserves ordinary answer text before either declared EOS" {
    for ([_]u32{ 127906, 127901 }) |first| for ([_]bool{ false, true }) |streaming|
        try check(first, true, false, streaming, true);
}

/// One structured chat reply with ignore_eos, on the DeepSeek-V4.1 test tokenizer, `chunk` tokens an event.
fn grammarStop(thinking: bool, streaming: bool, chunk: usize) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, io, null);
    defer f.deinit();
    const vocab = try std.Io.Dir.cwd().readFileAlloc(io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
    // a Fuse decoder: without one its pieces decode joined by spaces
    const fused = try std.mem.replaceOwned(u8, a, vocab, "{\"version\": \"1.0\",", "{\"version\": \"1.0\", \"decoder\": {\"type\": \"Fuse\"},");
    try f.tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = fused });
    try f.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    try f.tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<｜end▁of▁sentence｜>\"}" });
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, io, f.path, null, a, &problem);
    defer text.deinit();
    const fake = lanes.grammar_fake;
    const old = fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var compiler = fake.compiler();
    // the lane core ends the reply on the grammar's stop token; with thinking on, an end token before `</think>` stays
    const end = text.text().tokenId("<｜end▁of▁sentence｜>").?;
    const close = text.text().tokenId("</think>").?;
    const reply = [_]u32{ 'h', 'm', end, 'o', 'k', close, '{', '}', end };
    const ids: []const u32 = if (thinking) &reply else reply[6..];
    var script = Script{ .a = a, .ids = ids, .reason = .stop, .chunk = chunk };
    const srv = try Server.init(gpa, io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = thinking,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"H\"}}],\"max_tokens\":{d},\"stream\":{},\"ignore_eos\":true,\"response_format\":{{\"type\":\"json_object\"}}}}", .{ ids.len, streaming }))).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, true, body);
    var content: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    var final: ?json.Value = response.whole;
    var pieces: [2]usize = .{ 0, 0 }; // reasoning and content deltas
    if (streaming) {
        try std.testing.expect(response.opened and response.done and response.whole == null);
        for (response.events.items) |value| {
            const choices = value.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            const choice = choices.array[0];
            if (choice.get("delta")) |d| {
                const said: [2][]const u8 = .{ d.strField("reasoning_content") orelse "", d.strField("content") orelse "" };
                try reasoning.appendSlice(a, said[0]);
                try content.appendSlice(a, said[1]);
                for (&pieces, said) |*n, x| n.* += @intFromBool(x.len > 0);
            }
            if (choice.get("finish_reason")) |reason| {
                if (reason == .string) final = value;
            }
        }
        // each event's text streams with it, none left to the finish: one delta for each event that adds some
        const step = if (chunk == 0) ids.len else chunk;
        const answer_at: usize = if (thinking) 6 else 0;
        try std.testing.expectEqual(if (thinking) (4 / step) + 1 else 0, pieces[0]);
        try std.testing.expectEqual(((answer_at + 1) / step) - (answer_at / step) + 1, pieces[1]);
    } else {
        try std.testing.expectEqual(@as(?u16, 200), response.status);
        const message = response.whole.?.get("choices").?.array[0].get("message").?;
        try content.appendSlice(a, message.strField("content") orelse "");
        try reasoning.appendSlice(a, message.strField("reasoning_content") orelse "");
    }
    const value = final orelse return error.MissingFinalReply;
    try std.testing.expect(script.structured);
    try std.testing.expectEqual(@as(?u32, if (thinking) close else null), script.after);
    try std.testing.expectEqualSlices(u32, &.{}, script.seen_eos);
    try std.testing.expectEqualSlices(u32, &.{end}, srv.eos);
    try std.testing.expectEqualStrings("{}", content.items);
    try std.testing.expectEqualStrings(if (thinking) "hm<｜end▁of▁sentence｜>ok" else "", reasoning.items);
    try std.testing.expectEqualStrings("stop", json.strOr(value.get("choices").?.array[0].get("finish_reason")));
    try std.testing.expectEqual(@as(i64, @intCast(ids.len)), value.get("usage").?.get("completion_tokens").?.int64().?);
}

test "EOS policy: with ignore_eos a grammar's stop token ends the reply outside its text, and an earlier end token stays" {
    // every chunk size: the reasoning's end token, `</think>` and the grammar's stop land together or apart
    for ([_]bool{ false, true }) |thinking| for ([_]bool{ false, true }) |streaming| for (0..10) |chunk|
        try grammarStop(thinking, streaming, chunk);
}

fn tokenIds(streaming: bool, asked: bool) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, io, f.path, null, a, &problem);
    defer text.deinit();
    const ids = [_]u32{ 127901, 1, 2 };
    var script = Script{ .a = a, .ids = &ids };
    const srv = try Server.init(gpa, io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":3,\"stream\":{s},\"return_token_ids\":{s}}}", .{ if (streaming) "true" else "false", if (asked) "true" else "false" }))).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    var final: ?json.Value = response.whole;
    if (streaming) {
        try std.testing.expect(response.opened and response.done and response.whole == null);
        for (response.events.items) |value| {
            const choices = value.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            const reason = choices.array[0].get("finish_reason");
            if (reason != null and reason.? == .string) final = value;
        }
    } else {
        try std.testing.expectEqual(@as(?u16, 200), response.status);
    }
    const value = final orelse return error.MissingFinalReply;
    const tensorfold = value.get("tensorfold") orelse return error.MissingTensorfold;
    if (!asked) {
        try std.testing.expect(tensorfold.get("token_ids") == null);
        return;
    }
    const got = tensorfold.get("token_ids") orelse return error.MissingTokenIds;
    try std.testing.expect(got == .array);
    try std.testing.expectEqual(ids.len, got.array.len);
    for (got.array, ids) |v, want| try std.testing.expectEqual(@as(i64, @intCast(want)), v.int64().?);
    try std.testing.expectEqualStrings(&tokens.tokenSha256(&ids), json.strOr(tensorfold.get("token_sha256")));
}

test "return_token_ids: the ids equal the reply's tokens and their sha, whole and streamed" {
    for ([_]bool{ false, true }) |streaming| try tokenIds(streaming, true);
}

test "return_token_ids: unset keeps the tensorfold block without token_ids" {
    for ([_]bool{ false, true }) |streaming| try tokenIds(streaming, false);
}

/// A chat reply ("OK", or Harmony channels then "OK") under `asked` (null: default, thinking on) and Info.reasoning.
fn answerOnly(streaming: bool, reasoning: bool, asked: ?bool, harmony: bool) !void {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, std.testing.io, fixture.generation);
    defer f.deinit();
    try f.tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "chat_template.jinja",
        .data = "{{ messages[0]['content'] }}{% if add_generation_prompt %}{{ 'K' if (enable_thinking is defined and not enable_thinking) else '' }}{% endif %}",
    });
    if (harmony) try f.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = harmony_tokenizer });
    var problem: []const u8 = "";
    const hf = try HfText.load(gpa, std.testing.io, f.path, null, a, &problem);
    defer hf.deinit();
    const ids: []const u32 = if (harmony) &.{ 3, 4, 5, 6, 3, 7, 5, 1, 2 } else &.{ 1, 2, 127906 };
    var script: Script = .{ .a = a, .ids = ids, .reasoning = reasoning };
    const srv = try Server.init(gpa, std.testing.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true }, null);
    defer srv.deinit();
    const kwargs = if (asked) |on| try std.fmt.allocPrint(a, ",\"chat_template_kwargs\":{{\"enable_thinking\":{}}}", .{on}) else "";
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"H\"}}],\"max_tokens\":{d},\"stream\":{}{s}}}", .{ ids.len, streaming, kwargs }))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response: Reply = .{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, true, body);
    const thinking = asked orelse true;
    try std.testing.expectEqualSlices(u32, if (thinking) &.{0} else &.{ 0, 2 }, script.seen_prompt);
    var content: std.ArrayList(u8) = .empty;
    var thought: std.ArrayList(u8) = .empty;
    var final = response.whole;
    if (streaming) {
        try std.testing.expect(response.opened and response.done);
        for (response.events.items) |event| {
            if (event.get("usage")) |usage| if (usage != .null) {
                final = event;
            };
            const choices = event.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            if (choices.array[0].get("delta")) |delta| {
                try content.appendSlice(a, json.strOr(delta.get("content")));
                try thought.appendSlice(a, json.strOr(delta.get("reasoning_content")));
            }
        }
    } else {
        try std.testing.expectEqual(@as(?u16, 200), response.status);
        const message = response.whole.?.get("choices").?.array[0].get("message").?;
        try content.appendSlice(a, json.strOr(message.get("content")));
        try thought.appendSlice(a, json.strOr(message.get("reasoning_content")));
    }
    const split = reasoning and thinking;
    const answer = if (harmony) "<|channel|>analysis<|message|>plan<|channel|>final<|message|>OK" else "OK";
    try std.testing.expectEqualStrings(if (split) "" else answer, content.items);
    try std.testing.expectEqualStrings(if (split) answer else "", thought.items);
    const value = final orelse return error.MissingFinalReply;
    try std.testing.expectEqual(thinking, value.get("tensorfold").?.get("enable_thinking").?.bool);
    if (!split) try std.testing.expectEqual(@as(i64, 0), value.get("usage").?.get("completion_tokens_details").?.get("reasoning_tokens").?.int64().?);
}

test "Info.reasoning false: default or explicit thinking renders its prompt but the reply is all answer content" {
    for ([_]bool{ false, true }) |streaming| for ([_]?bool{ null, true, false }) |asked|
        try answerOnly(streaming, false, asked, false);
}

test "Info.reasoning default: the reply splits as before under default, explicit on and explicit off thinking" {
    for ([_]bool{ false, true }) |streaming| for ([_]?bool{ null, true, false }) |asked|
        try answerOnly(streaming, true, asked, false);
}

test "Info.reasoning false: Harmony channel text stays answer content, streamed and whole" {
    for ([_]bool{ false, true }) |streaming|
        try answerOnly(streaming, false, null, true);
}

/// The fixture tokenizer with Harmony channel pieces as ordinary tokens 3-7.
const harmony_tokenizer =
    \\{"model":{"type":"BPE","vocab":{"H":0,"O":1,"K":2,"<|channel|>":3,"analysis":4,"<|message|>":5,"plan":6,"final":7,
    \\"<|endoftext|>":127901,"<|im_end|>":127906},"merges":[]},"decoder":{"type":"Fuse"},
    \\"added_tokens":[{"id":127901,"content":"<|endoftext|>","special":true},{"id":127906,"content":"<|im_end|>","special":true}]}
;
