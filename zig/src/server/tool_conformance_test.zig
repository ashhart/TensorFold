//! Chat template families: each renders its tools section, a call it writes parses back, a bad one stays text.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const f = @import("control_tokens_fixture.zig");
const errors = @import("errors.zig");
const HfText = @import("hf_text.zig").HfText;
const messages_mod = @import("messages.zig");
const Server = @import("server.zig").Server;
const tool_lfm2 = @import("tool_lfm2.zig");
const tool_minimax = @import("tool_minimax.zig");
const tool_parse = @import("tool_parse.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const Value = json.Value;

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const tools_json =
    \\[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path"]}}}]
;

/// `body` parsed as the request's messages, rendered through the family's template, tools offered.
fn render(a: Allocator, template_src: []const u8, body: []const u8, tools: Value) ![]const u8 {
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = template_src });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{};
    const srv = try Server.init(a, std.testing.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const raw = (try json.parseText(a, body)).ok;
    const normalized = try messages_mod.normalizeWith(&cx, raw, srv.late_system, srv.needs_user_after_tool, srv.media());
    const with_args = try messages_mod.toolArguments(&cx, normalized);
    const r = try srv.text.render(a, with_args, .{ .tools = tools, .add_generation_prompt = true }, &problem);
    return r.text;
}

const Rewrite = enum { none, lfm2, minimax };

/// `output` as the model wrote it: any family rewrite, then the finished-reply parser.
fn callsOf(a: Allocator, output: []const u8, tools: []const Value, rewrite: Rewrite) !tool_parse.Parsed {
    const text = switch (rewrite) {
        .none => output,
        .lfm2 => try tool_lfm2.dsml(a, output, tools, true),
        .minimax => try tool_minimax.dsml(a, output, tools),
    };
    return tool_parse.parse(a, text, tools, null);
}

/// `v`'s function object, or null.
fn fnOf(v: Value) ?*const json.Object {
    const o = if (v == .object) v.object else return null;
    const fn_v = o.get("function") orelse return null;
    return if (fn_v == .object) fn_v.object else null;
}

/// The parsed calls' names and arguments in order, as a canonical line each: `name=JSON`.
fn digest(a: Allocator, parsed: tool_parse.Parsed) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    const calls = parsed.calls orelse return out.items;
    for (calls) |call| {
        const fobj = fnOf(call) orelse continue;
        try out.appendSlice(a, fobj.get("name").?.string);
        try out.append(a, '=');
        if (fobj.get("arguments")) |args| {
            const s = if (args == .string) args.string else try json.stringify(a, args, .{ .ascii = false });
            try out.appendSlice(a, s);
        }
        try out.append(a, '\n');
    }
    return out.items;
}

const args_esc = "{\"path\":\"notes.txt\",\"content\":\"l\\u00edne\\n\\u65e5\\u672c\\u8a9e \\\"q\\\" {\\\"a\\\":1}\"}";

/// One family: its template, the pieces its tools section and its rendered call must carry, its model-output samples.
const Family = struct {
    name: []const u8,
    template: ?[]const u8 = null,
    tools_parts: []const []const u8 = &.{},
    call_part: ?[]const u8 = null,
    result_part: ?[]const u8 = null,
    output: []const u8 = "",
    rewrite: Rewrite = .none,
    two: []const u8 = "",
    malformed: []const u8 = "",
};

const families = [_]Family{
    .{ // GLM-5.3: <tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>
        .name = "glm",
        .template = @embedFile("fixtures/thinking/glm53.jinja"),
        .tools_parts = &.{ "# Tools", "<tools>", "read_file", "path" },
        .call_part = "<tool_call>read_file<arg_key>path</arg_key><arg_value>",
        .result_part = "file body",
        .output = "Checking.\n<tool_call>read_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value></tool_call>",
        .two = "<tool_call>read_file<arg_key>path</arg_key><arg_value>a</arg_value></tool_call><tool_call>read_file<arg_key>path</arg_key><arg_value>b</arg_value></tool_call>",
        .malformed = "<tool_call>read_file<arg_key>path</arg_key><arg_value>notes.txt",
    },
    .{ // Qwen3.8: <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        .name = "qwen",
        .template = @embedFile("fixtures/thinking/qwen38.jinja"),
        .tools_parts = &.{ "read_file", "path" },
        .call_part = "<tool_call>\n<function=read_file>",
        .result_part = "file body",
        .output = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>",
        .two = "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>",
        .malformed = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt",
    },
    .{ // Dory shares the Qwen3-Coder XML call form.
        .name = "dory",
        .template = @embedFile("fixtures/dory/chat_template.jinja"),
        .tools_parts = &.{ "read_file", "path" },
        .call_part = "<tool_call>\n<function=read_file>",
        .result_part = "file body",
        .output = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>",
        .two = "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>",
        .malformed = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt",
    },
    .{ // LFM2: <|tool_call_start|>[name(k=v)]<|tool_call_end|>, rewritten as DSML before parsing.
        .name = "lfm2",
        .template = @embedFile("fixtures/thinking/lfm2.jinja"),
        .tools_parts = &.{ "List of tools: [", "read_file" },
        .call_part = "<|tool_call_start|>[read_file(",
        .result_part = "file body",
        .output = "<|tool_call_start|>[read_file(path=\"notes.txt\")]<|tool_call_end|>",
        .rewrite = .lfm2,
        .two = "<|tool_call_start|>[read_file(path=\"a\"), read_file(path=\"b\")]<|tool_call_end|>",
        .malformed = "<|tool_call_start|>[read_file(path=",
    },
    .{ // ChatML's control tokens: tools and calls as JSON in the template; models answer <tool_call>{json}</tool_call>.
        .name = "chatml",
        .template = @embedFile("fixtures/control_tokens/chatml.jinja"),
        .tools_parts = &.{"read_file"},
        .call_part = "call_1",
        .result_part = "file body",
        .output = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"notes.txt\"}}\n</tool_call>",
        .two = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"b\"}}\n</tool_call>",
        .malformed = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {",
    },
    .{ // DeepSeek's control tokens; its models answer in DSML.
        .name = "deepseek",
        .template = @embedFile("fixtures/control_tokens/deepseek.jinja"),
        .tools_parts = &.{"read_file"},
        .call_part = "call_1",
        .result_part = "file body",
        .output = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read_file\"><｜DSML｜parameter name=\"path\" string=\"true\">notes.txt</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        .two = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read_file\"><｜DSML｜parameter name=\"path\" string=\"true\">a</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name=\"read_file\"><｜DSML｜parameter name=\"path\" string=\"true\">b</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        .malformed = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read_file\"><｜DSML｜parameter name=\"path\" string=\"true\">notes.txt",
    },
    .{ // GLM's control tokens; same tag calls as the thinking template.
        .name = "glm-control",
        .template = @embedFile("fixtures/control_tokens/glm.jinja"),
        .tools_parts = &.{"read_file"},
        .call_part = "call_1",
        .result_part = "file body",
        .output = "<tool_call>read_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value></tool_call>",
        .two = "<tool_call>read_file<arg_key>path</arg_key><arg_value>a</arg_value></tool_call><tool_call>read_file<arg_key>path</arg_key><arg_value>b</arg_value></tool_call>",
        .malformed = "<tool_call>read_file<arg_key>path</arg_key><arg_value>notes.txt",
    },
    .{ // MiniMax-M3's prefixed XML, rewritten as DSML before parsing; its template ships with the model.
        .name = "minimax",
        .rewrite = .minimax,
        .output = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"read_file\">]<]minimax[>[<path>notes.txt]<]minimax[>[</path>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        .two = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"read_file\">]<]minimax[>[<path>a]<]minimax[>[</path>]<]minimax[>[</invoke>]<]minimax[>[<invoke name=\"read_file\">]<]minimax[>[<path>b]<]minimax[>[</path>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        .malformed = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"read_file\">]<]minimax[>[<path>notes.txt",
    },
    .{ // Gemma's Pythonic calls inside a tool_call envelope; the template ships with the model.
        .name = "gemma",
        .output = "<tool_call>call:read_file{path:\"notes.txt\"}</tool_call>",
        .two = "<tool_call>call:read_file{path:\"a\"}</tool_call><tool_call>call:read_file{path:\"b\"}</tool_call>",
        .malformed = "<tool_call>call:read_file{path:",
    },
};

test "tool conformance: every chat template family renders tools, round-trips calls and never invents one" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parseText(a, tools_json)).ok.array;
    const arg_line = "read_file={\"path\":\"notes.txt\"}\n";
    for (families) |fam| {
        if (fam.template) |src| {
            // (a) one tool definition plus a user turn renders the tools section as the template defines it
            const defs = try render(a, src, "[{\"role\":\"user\",\"content\":\"Read notes.\"}]", .{ .array = tools });
            for (fam.tools_parts) |part| {
                if (std.mem.indexOf(u8, defs, part) == null) std.debug.print("{s} tools section missing {s}\n", .{ fam.name, part });
                try testing.expect(std.mem.indexOf(u8, defs, part) != null);
            }
            // (b) an assistant tool call plus its tool result renders the family's markup
            const hist = "[{\"role\":\"user\",\"content\":\"Read notes.\"},{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":" ++ args_esc ++ "}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"file body\"}]";
            const past = try render(a, src, hist, .{ .array = tools });
            try testing.expect(std.mem.indexOf(u8, past, fam.call_part.?) != null);
            try testing.expect(std.mem.indexOf(u8, past, fam.result_part.?) != null);
        }
        // (b) the family's call markup parses back with the same name and arguments
        const one = try callsOf(a, fam.output, tools, fam.rewrite);
        try testing.expectEqualStrings(arg_line, try digest(a, one));
        // (c) two calls in one turn keep their order
        const pair = try callsOf(a, fam.two, tools, fam.rewrite);
        try testing.expectEqualStrings("read_file={\"path\":\"a\"}\nread_file={\"path\":\"b\"}\n", try digest(a, pair));
        // (d) a malformed or truncated call is error or plain text, never a different call
        const bad = try callsOf(a, fam.malformed, tools, fam.rewrite);
        try testing.expectEqualStrings("", try digest(a, bad));
        try testing.expect(std.mem.indexOf(u8, try digest(a, bad), "read_file") == null);
    }
}
