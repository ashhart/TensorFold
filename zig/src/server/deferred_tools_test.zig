//! Deferred tools and tool references from the Messages translation through normalization, templates and the routes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const messages = @import("messages.zig");
const model_text = @import("model_text.zig");
const deferred_tools = @import("deferred_tools.zig");
const translate = @import("anthropic_translate.zig").translate;
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const exchange = @import("late_system_test.zig").exchange;
const family = @import("thinking_family_fixture.zig");
const Value = json.Value;
const Cx = errors.Cx;
const t = std.testing;

const glm = .{ @embedFile("fixtures/thinking/glm53.jinja"), "glm5_next" };
const config = @embedFile("fixtures/thinking/glm53-tokenizer_config.json");
const plain = "{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}<|{{ m.role }}|>{{ m.content }}" ++
    "{% if m.tool_calls %}{{ m.tool_calls | tojson }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
const off_text = "{\"name\": \"off\", \"description\": \"E\", \"parameters\": {\"type\": \"object\"}}";
const tools_src =
    \\[{"type":"function","function":{"name":"on","description":"D","parameters":{"type":"object"}}},
    \\ {"type":"function","function":{"name":"off","description":"E","parameters":{"type":"object"},"defer_loading":true}}]
;

fn parse(a: std.mem.Allocator, text: []const u8) !Value {
    return (try json.parse(a, text)).ok;
}

fn str(a: std.mem.Allocator, v: Value) ![]const u8 {
    return json.stringify(a, v, .{ .ascii = false, .compact = true });
}

test "deferred tools: Messages translation carries defer_loading and tool_reference, and refuses what Anthropic does" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const head = "{\"model\":\"m\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"tools\":[";
    const chat = try translate(&cx, try parse(a,
        \\{"model":"m","max_tokens":16,"tools":[{"name":"on","input_schema":{"type":"object"}},
        \\ {"name":"off","description":"E","input_schema":{"type":"object"},"defer_loading":true}],
        \\ "messages":[{"role":"user","content":"H"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"on","input":{}}]},
        \\ {"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"found"},{"type":"tool_reference","tool_name":"off"}]}]}]}
    ), false);
    try t.expectEqualStrings(
        \\[{"type":"function","function":{"name":"on","parameters":{"type":"object"}}},{"type":"function","function":{"name":"off","parameters":{"type":"object"},"description":"E","defer_loading":true}}]
    , try str(a, chat.get("tools").?));
    try t.expectEqualStrings(
        \\{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"found"},{"type":"tool_reference","name":"off"}]}
    , try str(a, chat.get("messages").?.array[2]));
    // defer_loading false is the default, so the request is main's own
    const off = try translate(&cx, try parse(a, head ++ "{\"name\":\"t\",\"input_schema\":{},\"defer_loading\":false}]}"), false);
    const bare = try translate(&cx, try parse(a, head ++ "{\"name\":\"t\",\"input_schema\":{}}]}"), false);
    try t.expectEqualStrings(try str(a, bare), try str(a, off));
    const refused = [_][2][]const u8{
        .{ "{\"name\":\"t\",\"input_schema\":{},\"defer_loading\":\"yes\"}]}", "tool.defer_loading must be a boolean" },
        .{ "{\"name\":\"t\",\"input_schema\":{},\"defer_loading\":true}]}", "At least one tool must have defer_loading=false. All tools cannot be deferred." },
        .{ "{\"type\":\"tool_search_tool_regex_20251119\",\"name\":\"tool_search_tool_regex\"}]}", "only client-defined function tools are supported" },
    };
    inline for (refused) |case| {
        try t.expectError(error.Refused, translate(&cx, try parse(a, head ++ case[0]), false));
        try t.expectEqualStrings(case[1], cx.message);
        try t.expectEqual(@as(u16, 400), cx.status());
    }
    const nameless = try parse(a,
        \\{"model":"m","max_tokens":16,"tools":[{"name":"on","input_schema":{}}],"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"tool_reference"}]}]}]}
    );
    try t.expectError(error.Refused, translate(&cx, nameless, false));
    try t.expectEqualStrings("tool_reference.tool_name must be a nonempty string", cx.message);
}

test "deferred tools: normalization keeps a tool result's references on both passes and refuses one elsewhere" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const once = try messages.normalize(&cx, try parse(a,
        \\[{"role":"user","content":"H"},{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"a"},{"type":"tool_reference","name":"off","x":1}]},
        \\ {"role":"tool","tool_call_id":"c2","content":[{"type":"text","text":"b"},{"type":"text","text":"c"}]}]
    ), "system", false);
    const want =
        \\[{"role":"user","content":"H"},{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"a"},{"type":"tool_reference","name":"off"}]},{"role":"tool","tool_call_id":"c2","content":"bc"}]
    ;
    try t.expectEqualStrings(want, try str(a, once));
    try t.expectEqualStrings(want, try str(a, try messages.normalize(&cx, once, "system", false)));
    for ([_][]const u8{
        \\[{"role":"tool","tool_call_id":"c1","content":[{"type":"tool_reference"}]}]
        ,
        \\[{"role":"tool","tool_call_id":"c1","content":[{"type":"tool_reference","name":""}]}]
        ,
        \\[{"role":"user","content":[{"type":"tool_reference","name":"off"}]}]
    }, [_][]const u8{ "a tool_reference part must name a tool", "a tool_reference part must name a tool", "this server accepts text parts only; image, audio and video inputs are unsupported" }) |raw, message| {
        try t.expectError(error.Refused, messages.normalize(&cx, try parse(a, raw), "system", false));
        try t.expectEqualStrings(message, cx.message);
    }
}

test "deferred tools: references resolve against every offered tool, as text where the template cannot read them" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const tools = try parse(a, tools_src);
    const history = try parse(a,
        \\[{"role":"user","content":"H"},{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"found"},{"type":"tool_reference","name":"off"}]}]
    );
    const kept = try deferred_tools.resolve(&cx, history, tools, true);
    try t.expectEqualStrings(try str(a, history), try str(a, kept));
    const spelled = try deferred_tools.resolve(&cx, history, tools, false);
    try t.expectEqualStrings("found\n" ++ off_text, spelled.array[1].get("content").?.string);
    try t.expectEqualStrings("H", spelled.array[0].get("content").?.string);
    // a reference to a tool the request never offered is refused by name, as Anthropic refuses it
    const unknown = try parse(a,
        \\[{"role":"tool","tool_call_id":"c1","content":[{"type":"tool_reference","name":"missing"}]}]
    );
    for ([_]bool{ true, false }) |reads| {
        try t.expectError(error.Refused, deferred_tools.resolve(&cx, unknown, tools, reads));
        try t.expectEqualStrings("Tool reference 'missing' not found in available tools", cx.message);
        try t.expectEqual(errors.Kind.request, cx.kind);
    }
    const plain_history = try parse(a,
        \\[{"role":"user","content":"H"},{"role":"tool","tool_call_id":"c1","content":"done"}]
    );
    try t.expectEqual(plain_history.array.ptr, (try deferred_tools.resolve(&cx, plain_history, tools, false)).array.ptr);
    const listed = (try deferred_tools.shown(a, tools)).?;
    try t.expectEqual(@as(usize, 1), listed.array.len);
    try t.expectEqualStrings("on", listed.array[0].get("function").?.get("name").?.string);
    const loaded = try parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"on\",\"defer_loading\":false}}]");
    try t.expectEqual(loaded.array.ptr, (try deferred_tools.shown(a, loaded)).?.array.ptr);
    try t.expect(try deferred_tools.shown(a, .{ .array = tools.array[1..] }) == null);
}

/// A chat as the routes render it: normalized, references resolved, normalized again, call arguments as objects.
fn rendered(cx: *Cx, text: model_text.Text, raw: []const u8, tools: Value, generation: bool) ![]const u8 {
    const once = try messages.normalize(cx, try parse(cx.a, raw), "system", false);
    const resolved = try deferred_tools.resolve(cx, once, tools, text.deferred_native);
    const ready = try messages.toolArguments(cx, try messages.normalize(cx, resolved, "system", false));
    var problem: []const u8 = "";
    const out = text.render(cx.a, ready, .{ .tools = tools, .add_generation_prompt = generation }, &problem) catch |e| {
        std.debug.print("render: {s}\n", .{problem});
        return e;
    };
    return out.text;
}

test "deferred tools: GLM-5.3 renders them its own way, others get the spec at the reference, and earlier text is kept" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const call = "[{\"role\":\"user\",\"content\":\"H\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\"," ++
        "\"type\":\"function\",\"function\":{\"name\":\"on\",\"arguments\":\"{}\"}}]}";
    const loaded = call ++ ",{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":[{\"type\":\"tool_reference\",\"name\":\"off\"}]}]";
    for ([_][2][]const u8{ glm, .{ plain, "plain_test" } }, [_]bool{ true, false }) |case, native| {
        var f = try family.init(t.allocator, t.io, case[0], config, case[1]);
        defer f.deinit();
        var problem: []const u8 = "";
        const hf = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
        defer hf.deinit();
        const text = hf.text();
        try t.expectEqual(native, text.deferred_native);
        var cx: Cx = .{ .a = a };
        const tools = try parse(a, tools_src);
        const without = try rendered(&cx, text, call ++ "]", .{ .array = tools.array[0..1] }, false);
        // a deferred tool is out of the prompt until a reference loads it, which leaves the text before it as it was
        try t.expectEqualStrings(without, try rendered(&cx, text, call ++ "]", tools, false));
        try t.expectEqualStrings(without, (try rendered(&cx, text, call ++ "]", tools, true))[0..without.len]);
        const got = try rendered(&cx, text, loaded, tools, true);
        try t.expect(std.mem.startsWith(u8, got, without));
        try t.expect(std.mem.indexOf(u8, without, "\"off\"") == null);
        const after = got[without.len..];
        if (native) {
            try t.expect(std.mem.indexOf(u8, after, "<tool_response><tools>\n" ++ off_text ++ "\n</tools></tool_response>") != null);
        } else {
            try t.expect(std.mem.indexOf(u8, after, "<|tool|>" ++ off_text ++ "<|assistant|>") != null);
        }
    }
}

const Script = struct {
    reply: []const u32 = &.{},
    seen: bool = false,
    count: usize = 0,
    prompt: [8192]u32 = undefined,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 65536 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen = true;
        s.count = request.prompt.len;
        @memcpy(s.prompt[0..s.count], request.prompt);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (s.reply) |token| sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
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

const request_head =
    \\{"model":"m","max_tokens":256,"tools":[{"name":"on","description":"D","input_schema":{"type":"object"}},
    \\ {"name":"off","description":"E","input_schema":{"type":"object"},"defer_loading":true}],
    \\ "messages":[{"role":"user","content":"H"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"on","input":{}}]},
    \\ {"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"tool_reference","tool_name":"
;

test "deferred tools: a Messages request loads a tool by reference, count_tokens agrees, and its later call is returned" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // GLM-5.3 lists the translated spec in its own key order; the text form names, describes, then gives parameters
    const glm_text = "<tool_response><tools>\n{\"name\": \"off\", \"parameters\": {\"type\": \"object\"}, \"description\": \"E\"}\n";
    for ([_][2][]const u8{ glm, .{ plain, "plain_test" } }, [_][]const u8{ glm_text, off_text }) |case, loaded| {
        var f = try family.init(t.allocator, t.io, case[0], config, case[1]);
        defer f.deinit();
        var problem: []const u8 = "";
        const hf = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
        defer hf.deinit();
        const text = hf.text();
        var script: Script = .{ .reply = try text.encode(a, "<tool_call>off<arg_key>x</arg_key><arg_value>v</arg_value></tool_call>", false) };
        const srv = try Server.init(t.allocator, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
        defer srv.deinit();
        const body = request_head ++ "off\"}]}]}]}";
        const sent = try exchange(srv, "/v1/messages", body);
        defer t.allocator.free(sent.body);
        try t.expectEqual(@as(u16, 200), sent.status);
        const prompt = try text.decode(a, script.prompt[0..script.count]);
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, prompt, loaded));
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, prompt, "\"off\""));
        try t.expect(std.mem.indexOf(u8, prompt, "defer_loading") == null);
        const reply = try parse(a, sent.body);
        try t.expectEqualStrings("tool_use", reply.get("stop_reason").?.string);
        const use = reply.get("content").?.array[0];
        try t.expectEqualStrings("tool_use", use.get("type").?.string);
        try t.expectEqualStrings("off", use.get("name").?.string);
        try t.expectEqualStrings("{\"x\":\"v\"}", try str(a, use.get("input").?));
        const counted = try exchange(srv, "/v1/messages/count_tokens", body);
        defer t.allocator.free(counted.body);
        try t.expectEqual(@as(u16, 200), counted.status);
        try t.expectEqual(@as(?i64, @intCast(script.count)), (try parse(a, counted.body)).get("input_tokens").?.int64());
        script.seen = false;
        for ([_][]const u8{ "/v1/messages", "/v1/messages/count_tokens" }) |path| {
            const missing = try exchange(srv, path, request_head ++ "missing\"}]}]}]}");
            defer t.allocator.free(missing.body);
            try t.expectEqual(@as(u16, 400), missing.status);
            const problem_reply = (try parse(a, missing.body)).get("error").?;
            try t.expectEqualStrings("invalid_request_error", problem_reply.get("type").?.string);
            try t.expectEqualStrings("Tool reference 'missing' not found in available tools", problem_reply.get("message").?.string);
        }
        try t.expect(!script.seen);
    }
}
