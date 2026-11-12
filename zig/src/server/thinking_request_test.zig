//! Explicit thinking-off requests are either represented in the prompt or refused before submission.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const fixture = @import("hf_eos_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const exchange = @import("late_system_test.zig").exchange;
const Allocator = std.mem.Allocator;

const Script = struct {
    seen: bool = false,
    prompt: [4096]u32 = undefined,
    count: usize = 0,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        _ = ctx;
        return .{ .context_window = 262144 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen = true;
        s.count = request.prompt.len;
        @memcpy(s.prompt[0..s.count], request.prompt);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{2} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
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
    whole: ?json.Value = null,
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        _ = ctx;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        _ = ctx;
        _ = value;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.whole = value;
        r.status = status_code;
    }
};

const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

fn capability(srv: *Server, a: Allocator) !bool {
    var wire: Wire = .{ .a = a };
    var c: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "GET", .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    try @import("status_routes.zig").models(srv, &c, a);
    const at = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.MissingHttpHead;
    const body = (try json.parse(a, wire.bytes.items[at + 4 ..])).ok;
    var found: ?bool = null;
    for (body.get("data").?.array) |entry| {
        const caps = entry.get("capabilities") orelse return error.MissingThinkingCapability;
        const applicable = caps.get("thinking_applicable") orelse return error.MissingThinkingApplicability;
        try std.testing.expectEqual(srv.close_id != null, applicable.bool);
        try std.testing.expectEqual(srv.info.capability(.thinking_applicable), applicable.bool);
        const bit = caps.get("thinking_switch") orelse return error.MissingThinkingCapability;
        try std.testing.expect(bit == .bool);
        if (found) |previous| try std.testing.expectEqual(previous, bit.bool);
        found = bit.bool;
    }
    try std.testing.expectEqual(srv.info.capability(.thinking_switch), found orelse return error.MissingModel);
    return found.?;
}

const plain = "H";
const dangling = "H{% if add_generation_prompt %}<think>{% endif %}";
const closed = "H{% if add_generation_prompt %}<think></think>{% endif %}";
const switched = "H{% if add_generation_prompt %}{% if enable_thinking %}<think>{% else %}</think>{% endif %}{% endif %}";
const unclosed_switch = "{% if not enable_thinking %}O{% endif %}H";
const off = ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
const refusal = "thinking off is not supported by this chat template: it has no thinking switch or closable generation prefix";

const Source = enum { client, completion, ask, ask_in, messages, responses };

fn check(template: []const u8, close_token: bool, server_thinking: bool, extra: []const u8, code: u16, ids: []const u32) !void {
    return checkSource(template, close_token, server_thinking, extra, code, ids, .client);
}

fn checkSource(template: []const u8, close_token: bool, server_thinking: bool, extra: []const u8, code: u16, ids: []const u32, source: Source) !void {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    if (close_token) try f.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0,"O":1,"K":2,"<think>":300,"</think>":301,"<|im_end|>":127906},"merges":[]},"decoder":{"type":"Fuse"},"added_tokens":[{"id":300,"content":"<think>","special":true},{"id":301,"content":"</think>","special":true},{"id":127906,"content":"<|im_end|>","special":true}]}
    });
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = template });
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer text.deinit();
    var script: Script = .{};
    const srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{ "m", "alias" },
        .enable_thinking = server_thinking,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const can_off = try capability(srv, a);
    try t.expect(!script.seen);
    if (code == 400) try t.expect(!can_off) else if (source == .client and std.mem.eql(u8, extra, off)) try t.expect(can_off);
    if (source == .completion) try t.expect(!can_off);
    if (source == .ask or source == .ask_in) {
        var cx: @import("errors.zig").Cx = .{ .a = a };
        const Stay = struct {
            pub fn check(_: @This()) bool {
                return false;
            }
        };
        const reply = if (source == .ask)
            try @import("slide_ask.zig").ask(srv, &cx, null, "H", 2, Stay{})
        else
            try @import("slide_ask.zig").askIn(srv, &cx, &.{.{ "H", "K" }}, "H", 2, true, Stay{});
        try t.expectEqualStrings("K", reply.content);
        try t.expect(script.seen);
        try t.expectEqualSlices(u32, ids, script.prompt[0..script.count]);
        return;
    }
    if (source == .messages or source == .responses) return checkRoute(srv, a, &script, source, extra, code, ids);
    const input = if (source == .completion) "\"prompt\":\"H\"" else "\"messages\":[{\"role\":\"user\",\"content\":\"H\"}]";
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",{s},\"max_tokens\":2,\"return_token_ids\":true{s}}}", .{ input, extra }))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response: Reply = .{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, source == .client, body);
    try t.expectEqual(@as(?u16, code), response.status);
    const whole = response.whole.?;
    if (code == 400) {
        try t.expect(!script.seen);
        try t.expectEqualStrings(refusal, whole.get("error").?.strField("message").?);
    } else {
        try t.expect(script.seen);
        try t.expectEqualSlices(u32, ids, script.prompt[0..script.count]);
        const tokens = whole.get("tensorfold").?.get("token_ids").?.array;
        try t.expectEqual(@as(usize, 1), tokens.len);
        try t.expectEqual(@as(?i64, 2), tokens[0].int64());
    }
}

/// A Messages request (and its count_tokens) or a Responses request through its route, as a client sends it.
fn checkRoute(srv: *Server, a: Allocator, script: *Script, source: Source, extra: []const u8, code: u16, ids: []const u32) !void {
    const t = std.testing;
    const body = if (source == .messages)
        try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{{\"role\":\"user\",\"content\":\"H\"}}]{s}}}", .{extra})
    else
        try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_output_tokens\":2,\"store\":false,\"input\":\"H\"{s}}}", .{extra});
    const sent = try exchange(srv, if (source == .messages) "/v1/messages" else "/v1/responses", body);
    defer t.allocator.free(sent.body);
    try t.expectEqual(code, sent.status);
    try t.expectEqual(code != 400, script.seen);
    if (code == 400) {
        try t.expectEqualStrings(refusal, (try json.parse(a, sent.body)).ok.get("error").?.strField("message").?);
    } else try t.expectEqualSlices(u32, ids, script.prompt[0..script.count]);
    if (source != .messages) return;
    const counted = try exchange(srv, "/v1/messages/count_tokens", body);
    defer t.allocator.free(counted.body);
    try t.expectEqual(code, counted.status);
    const reply = (try json.parse(a, counted.body)).ok;
    if (code == 400) {
        try t.expectEqualStrings(refusal, reply.get("error").?.strField("message").?);
    } else try t.expectEqual(@as(?i64, @intCast(ids.len)), reply.get("input_tokens").?.int64());
}

test "thinking request: Messages without thinking is served with main's ids, and count_tokens counts them" {
    for ([_]bool{ false, true }) |default| {
        try checkSource(plain, true, default, "", 200, &.{0}, .messages);
        try checkSource("H<think></think>H", true, default, "", 200, &.{ 0, 300, 301, 0 }, .messages);
        try checkSource(dangling, true, default, "", 200, &.{ 0, 300, 301 }, .messages);
    }
}

test "thinking request: Messages' own disabled thinking is refused where the template can't honour it, on count_tokens too" {
    for ([_]bool{ false, true }) |default| {
        try checkSource(plain, true, default, ",\"thinking\":{\"type\":\"disabled\"}", 400, &.{}, .messages);
        try checkSource(switched, true, default, ",\"thinking\":{\"type\":\"disabled\"}", 200, &.{ 0, 301 }, .messages);
        try checkSource(plain, true, default, ",\"thinking\":{\"type\":\"adaptive\"}", 200, &.{0}, .messages);
    }
}

test "thinking request: Responses without reasoning keeps main's prompt under either server default, and its own effort none is refused" {
    for ([_]bool{ false, true }) |default| {
        try checkSource(plain, true, default, "", 200, &.{0}, .responses);
        try checkSource(plain, true, default, ",\"reasoning\":{\"effort\":\"none\"}", 400, &.{}, .responses);
    }
}

test "thinking request: off without a template switch or closable prefix refuses by name" {
    for ([_][]const u8{ off, ",\"thinking\":false", ",\"thinking\":{\"type\":\"disabled\"}", ",\"reasoning_effort\":\"none\"", ",\"chat_template_kwargs\":{\"thinking\":false}", ",\"chat_template_kwargs\":{\"reasoning_effort\":\"none\"}" }) |extra|
        try check(plain, true, true, extra, 400, &.{});
    try check("H<think></think>H", true, true, off, 400, &.{});
}

test "thinking request: omission, enabled requests and nonreasoning templates keep working" {
    try check(plain, true, true, "", 200, &.{0});
    try check(plain, true, false, "", 200, &.{0});
    try check(plain, true, true, ",\"thinking\":true", 200, &.{0});
    try check(plain, false, true, off, 200, &.{0});
}

test "thinking request: switched, already closed and closable prefixes keep their exact ids" {
    for ([_]bool{ false, true }) |default| {
        try check(switched, true, default, off, 200, &.{ 0, 301 });
        try check(unclosed_switch, true, default, off, 200, &.{ 1, 0 });
        try check(closed, true, default, off, 200, &.{ 0, 300, 301 });
        try check(dangling, true, default, off, 200, &.{ 0, 300, 301 });
    }
}

test "thinking request: a raw completion's prompt ids are its own, whatever thinking field it carries" {
    for ([_][]const u8{ "", off, ",\"thinking\":false", ",\"reasoning_effort\":\"none\"" }) |extra|
        try checkSource(plain, true, true, extra, 200, &.{0}, .completion);
}

test "thinking request: internal lesson ask keeps main behavior without an off switch" {
    try checkSource(plain, true, true, "", 200, &.{0}, .ask);
}

test "thinking request: internal lesson askIn keeps main behavior without an off switch" {
    try checkSource(plain, true, true, "", 200, &.{0}, .ask_in);
}

fn checkpoint(source: []const u8, config: []const u8, kind: []const u8, expected: []const u8) !void {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try @import("thinking_family_fixture.zig").init(t.allocator, t.io, source, config, kind);
    defer f.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer hf.deinit();
    try t.expectEqualStrings(source, hf.source);
    const text = hf.text();
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"H\"}]")).ok;
    const raw = try text.render(a, messages, .{ .enable_thinking = false }, &problem);
    try t.expectEqualStrings(expected, raw.text);
    const raw_ids = try raw.encode(text, a, &problem);
    try t.expectEqualStrings(expected, try text.decode(a, raw_ids));
    try t.expectEqual(@as(u32, 300), raw_ids[raw_ids.len - 1]);
    const closed_ids = try std.mem.concat(a, u32, &.{ raw_ids, &.{301} });
    try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ expected, "</think>" }), try text.decode(a, closed_ids));
    for ([_]bool{ false, true }) |default| {
        var script: Script = .{};
        const srv = try Server.init(t.allocator, t.io, script.engine(), text, .{
            .served_name = "m",
            .model_ids = &.{ "m", "alias" },
            .enable_thinking = default,
            .use_drafts = false,
        }, null);
        defer srv.deinit();
        try t.expect(!srv.thinking_switch);
        try t.expect(try capability(srv, a));
        try t.expect(!script.seen);
        var cx: @import("errors.zig").Cx = .{ .a = a };
        // The legacy entry point is used by main and by internal requests; it adds the same closing token.
        const legacy = try @import("prompt.zig").renderIds(srv, &cx, messages, &.{}, false, null, true);
        try t.expectEqualSlices(u32, closed_ids, legacy);
        for ([_][]const u8{ off, ",\"thinking\":false", ",\"thinking\":{\"type\":\"disabled\"}", ",\"reasoning_effort\":\"none\"", ",\"chat_template_kwargs\":{\"thinking\":false}", ",\"chat_template_kwargs\":{\"reasoning_effort\":\"none\"}" }) |extra|
            try checkpointReply(srv, a, &script, extra, legacy);
        try checkpointReply(srv, a, &script, "", if (default) raw_ids else legacy);
        try checkpointReply(srv, a, &script, ",\"thinking\":true", raw_ids);
    }
}

fn checkpointReply(srv: *Server, a: Allocator, script: *Script, extra: []const u8, ids: []const u32) !void {
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"H\"}}],\"max_tokens\":2,\"return_token_ids\":true{s}}}", .{extra}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response: Reply = .{ .a = a };
    script.seen = false;
    openai.run(srv, a, response.out(), .{ .conn = &conn }, true, body);
    try std.testing.expectEqual(@as(?u16, 200), response.status);
    try std.testing.expect(script.seen);
    try std.testing.expectEqualSlices(u32, ids, script.prompt[0..script.count]);
    const whole = response.whole.?;
    const message = whole.get("choices").?.array[0].get("message").?;
    const is_closed = ids[ids.len - 1] == 301;
    try std.testing.expectEqualStrings(if (is_closed) "K" else "", message.strField("content").?);
    if (is_closed) {
        try std.testing.expect(message.strField("reasoning_content") == null);
    } else try std.testing.expectEqualStrings("K", message.strField("reasoning_content").?);
    const tokens = whole.get("tensorfold").?.get("token_ids").?.array;
    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(?i64, 2), tokens[0].int64());
}

test "thinking request: checkpoint GLM 5.3 off closes its generation prefix and preserves main ids" {
    try checkpoint(@embedFile("fixtures/thinking/glm53.jinja"), @embedFile("fixtures/thinking/glm53-tokenizer_config.json"), "glm5_next", "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>H<|assistant|><think>");
}

test "thinking request: checkpoint LFM2 off closes its generation prefix and preserves main ids" {
    try checkpoint(@embedFile("fixtures/thinking/lfm2.jinja"), @embedFile("fixtures/thinking/lfm2-tokenizer_config.json"), "lfm2_moe", "<|startoftext|><|im_start|>user\nH<|im_end|>\n<|im_start|>assistant\n<think>");
}

/// GLM 5.3's template with thinking off, served: the prompt the engine gets carries `word` from `effort`.
fn effortOff(server_effort: ?[]const u8, extra: []const u8, effort: ?[]const u8, word: []const u8) !void {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config = @embedFile("fixtures/thinking/glm53-tokenizer_config.json");
    var f = try @import("thinking_family_fixture.zig").init(t.allocator, t.io, @embedFile("fixtures/thinking/glm53.jinja"), config, "glm5_next");
    defer f.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"H\"}]")).ok;
    const raw = try text.render(a, messages, .{ .enable_thinking = false, .reasoning_effort = effort }, &problem);
    try t.expect(std.mem.indexOf(u8, raw.text, word) != null);
    const ids = try std.mem.concat(a, u32, &.{ try raw.encode(text, a, &problem), &.{301} });
    var script: Script = .{};
    const srv = try Server.init(t.allocator, t.io, script.engine(), text, .{
        .served_name = "m",
        .model_ids = &.{ "m", "alias" },
        .use_drafts = false,
        .reasoning_effort = server_effort,
    }, null);
    defer srv.deinit();
    try checkpointReply(srv, a, &script, extra, ids);
}

test "thinking request: GLM 5.3 off renders a named effort, and main's Max when none is named" {
    try effortOff(null, off ++ ",\"reasoning_effort\":\"high\"", "high", "Reasoning Effort: High");
    try effortOff(null, ",\"thinking\":false,\"reasoning_effort\":\"low\"", "low", "Reasoning Effort: Low");
    try effortOff(null, off, null, "Reasoning Effort: Max");
    try effortOff(null, ",\"reasoning_effort\":\"none\"", null, "Reasoning Effort: Max");
    try effortOff("high", off, "high", "Reasoning Effort: High"); // --reasoning-effort names it for every request
}

test "thinking request: a named effort has no fallback, so Kimi's off prompt keeps none" {
    const fields = @import("fields.zig");
    const levels = [_][]const u8{ "low", "high", "max" };
    try std.testing.expect(fields.namedEffort(null, null, &levels) == null); // effortFor would give Kimi "max"
    try std.testing.expect(fields.namedEffort("none", "high", &levels) == null);
    try std.testing.expectEqualStrings("high", fields.namedEffort("high", null, &levels).?);
    try std.testing.expectEqualStrings("low", fields.namedEffort(null, "low", &levels).?);
    try std.testing.expectEqualStrings("high", fields.namedEffort("high", "low", &levels).?);
}
