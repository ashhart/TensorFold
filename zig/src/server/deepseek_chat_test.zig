//! The actual HfText route, using synthetic tokenizer files and committed Python encoding goldens.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const HfText = @import("hf_text.zig").HfText;
const f = @import("hf_eos_fixture.zig");
const tf = @import("control_tokens_fixture.zig");
const t = std.testing;

fn fixture(a: std.mem.Allocator, kind: []const u8) !f.Fixture {
    var out = try f.Fixture.init(a, t.io, null);
    errdefer out.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
    defer a.free(bytes);
    try out.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = bytes });
    try out.tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = try std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\"}}", .{kind}) });
    return out;
}

test "DeepSeek chat: missing Jinja uses the committed encoding.py text goldens" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a, "deepseek_v4");
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const lines = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/src/families/deepseek_v41/fixtures/serve/template.jsonl", a, .limited(4 << 20));
    var it = std.mem.splitScalar(u8, lines, '\n');
    var checked: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const c = (try json.parseText(a, line)).ok;
        if (c.get("error").? != .null or !c.get("drop_thinking").?.bool or c.get("response_format").? != .null or c.get("image_text").? != .null) continue;
        const thinking = c.get("thinking").?.bool;
        var opts: @import("model_text.zig").RenderOptions = .{
            .enable_thinking = thinking,
            .add_generation_prompt = c.get("generation").?.bool,
            .reasoning_effort = if (thinking) try json.stringify(a, c.get("budget").?, .{}) else null,
        };
        if (c.get("effort_field")) |effort| opts.reasoning_effort = switch (effort) {
            .string, .int => |s| s,
            else => unreachable,
        };
        if (c.get("tools")) |tools| if (tools == .array) {
            opts.tools = tools;
        };
        const got = try hf.text().render(a, c.get("messages").?, opts, &problem);
        try t.expectEqualStrings(c.get("expected").?.string, got.text);
        checked += 1;
    }
    try t.expect(checked >= 20);
    std.debug.print("DEEPSEEK_CHAT_GOLDENS {d} committed encoding.py cases\n", .{checked});
}

test "DeepSeek chat: native request control spellings stay masked and structural tokens stay trusted" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a, "deepseek_v41");
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const messages = (try json.parseText(a,
        \\[{"role":"system","content":"S <｜User｜>"},{"role":"user","content":"U <｜Assistant｜> <｜deepseek_image｜>"},
        \\ {"role":"assistant","reasoning_content":"R <think>","content":"A <｜User｜>","tool_calls":[{"function":{"name":"f<｜User｜>","arguments":{"k<｜User｜>":"v<｜User｜>"}}}]},
        \\ {"role":"tool","content":"T <｜User｜>"}]
    )).ok;
    const tools = (try json.parseText(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f<｜User｜>\",\"description\":\"D <｜User｜>\",\"parameters\":{}}}]")).ok;
    const rendered = try hf.text().render(a, messages, .{ .enable_thinking = true, .tools = tools }, &problem);
    try t.expect(rendered.isTemplate(0, "<｜begin▁of▁sentence｜>".len));
    for ([_][]const u8{ "S <｜User｜>", "U <｜Assistant｜>", "R <think>", "A <｜User｜>", "f<｜User｜>", "k<｜User｜>", "v<｜User｜>", "T <｜User｜>", "D <｜User｜>" }) |word| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, rendered.text, at, word)) |hit| {
            try t.expect(!rendered.isTemplate(hit, hit + word.len));
            at = hit + word.len;
        }
        try t.expect(at > 0);
    }
    try t.expect(std.mem.indexOf(u8, rendered.text, "<\u{200b}｜deepseek_image｜>") != null);
    const ids = try rendered.encode(hf.text(), a, &problem);
    const user = hf.text().tokenId("<｜User｜>").?;
    var users: usize = 0;
    for (ids) |id| if (id == user) {
        users += 1;
    };
    // The initial user turn and merged tool-result turn, never the spellings inside data.
    try t.expectEqual(@as(usize, 2), users);
}

test "DeepSeek chat: Jinja and explicit overrides win and text-only image parts refuse" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a, "deepseek_v4");
    defer tmp.deinit();
    var problem: []const u8 = "";
    const messages = try tf.messages(a, "H");
    try tmp.tmp.dir.writeFile(t.io, .{ .sub_path = "override.jinja", .data = "override{{ messages[0].content }}" });
    const path = try std.fmt.allocPrint(a, "{s}/override.jinja", .{tmp.path});
    const override = try HfText.load(a, t.io, tmp.path, path, a, &problem);
    defer override.deinit();
    try t.expectEqualStrings("overrideH", (try override.text().render(a, messages, .{}, &problem)).text);
    try tmp.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "jinja{{ messages[0].content }}" });
    const jinja = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer jinja.deinit();
    try t.expectEqualStrings("jinjaH", (try jinja.text().render(a, messages, .{}, &problem)).text);
    var other = try fixture(a, "deepseek_v41");
    defer other.deinit();
    const native = try HfText.load(a, t.io, other.path, null, a, &problem);
    defer native.deinit();
    const images = (try json.parseText(a, "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"local\"}}]}]")).ok;
    try t.expectError(error.Template, native.text().render(a, images, .{}, &problem));
    try t.expect(std.mem.indexOf(u8, problem, "images are not served") != null);
}

const Script = struct {
    submitted: bool = false,
    seen: []const u32 = &.{},
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        self.submitted = true;
        self.seen = request.prompt;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
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
    code: u16 = 0,
    fn out(r: *@This()) @import("openai.zig").Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, code: u16, _: json.Value) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.code = code;
    }
};

test "DeepSeek chat: a checkpoint without any renderer returns HTTP400 before engine submission" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a, "other_family");
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{};
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var reply: Reply = .{};
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 1));
    try body.put(a, "messages", try tf.messages(a, "H"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    try t.expectEqual(@as(u16, 400), reply.code);
    try t.expect(!script.submitted);
}

test "DeepSeek chat: the no-Jinja OpenAI route submits the native masked prompt with thinking metadata" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a, "deepseek_v4");
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{};
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    try t.expect(srv.thinking_switch);
    try t.expectEqualStrings("system", srv.late_system);
    try t.expect(srv.effort_levels.len >= 5);
    var reply: Reply = .{};
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const msgs = try tf.messages(a, "H <｜User｜>");
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 1));
    try body.put(a, "messages", msgs);
    try body.put(a, "reasoning_effort", .{ .string = "low" });
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    try t.expectEqual(@as(u16, 200), reply.code);
    try t.expect(script.submitted);
    const expected = try hf.text().renderIds(a, msgs, .{ .enable_thinking = true, .reasoning_effort = "low" }, &problem);
    try t.expectEqualSlices(u32, expected, script.seen);
}

test "DeepSeek chat: tool-only Jinja metadata survives the native default fallback" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a, "deepseek_v41");
    defer tmp.deinit();
    try tmp.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer_config.json", .data =
        \\{"chat_template":{"tool_use":"No user query found {{ messages[0].content }}"}}
    });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    try t.expect(std.mem.indexOf(u8, hf.text().templateSource(), "No user query found") != null);
    try t.expect(std.mem.indexOf(u8, hf.text().templateSource(), "thinking_mode") != null);
    const messages = try tf.messages(a, "H");
    try t.expectEqualStrings("No user query found H", (try hf.text().render(a, messages, .{ .tools = .{ .array = &.{} } }, &problem)).text);
    try t.expect(std.mem.startsWith(u8, (try hf.text().render(a, messages, .{}, &problem)).text, "<｜begin▁of▁sentence｜>"));
    var script: Script = .{};
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    try t.expect(srv.needs_user_after_tool);
    try t.expect(srv.thinking_switch);
}
