//! Required tools through both request parsers, the real feature gate and the shared constrained lane core.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const json = @import("json");
const errors = @import("errors.zig");
const mt = @import("model_text.zig");
const Server = @import("server.zig").Server;
const openai = @import("openai.zig");
const responses = @import("responses_translate.zig");
const tools = @import("tool_grammar.zig");
const parse = @import("tool_parse.zig");
const a = std.testing.allocator;
const specs = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

const Text = struct {
    form: enum { xml, json, gemma, unknown } = .json,
    fn text(t: *Text) mt.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = source } };
    }
    fn encode(_: *anyopaque, g: std.mem.Allocator, s: []const u8, _: bool) mt.Error![]u32 {
        const ids = try g.alloc(u32, s.len);
        for (s, ids) |ch, *id| id.* = if (ch == '\n') 95 else @as(u32, ch) - 32;
        return ids;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 95) try out.append(g, @intCast(id + 32)) else if (id == 95) try out.append(g, '\n');
        return out.toOwnedSlice(g);
    }
    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        return if (std.mem.eql(u8, s, "<tool_call>")) 28 else null;
    }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(undefined, g, &.{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 97;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{96};
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }
    fn render(ptr: *anyopaque, g: std.mem.Allocator, _: json.Value, _: mt.RenderOptions, _: *[]const u8) mt.Error!mt.Rendered {
        const t: *Text = @ptrCast(@alignCast(ptr));
        return .{ .text = try (g.dupe(u8, switch (t.form) {
            .xml => "<tool_call><function=tfprobe_fn><parameter=tfprobe_k>v</parameter></function></tool_call>",
            .json => "<tool_call>{\"name\":\"tfprobe_fn\",\"arguments\":{\"tfprobe_k\":\"v\"}}</tool_call>",
            .gemma => "<|tool_call>call:tfprobe_fn{tfprobe_k:<|\"|>v<|\"|>}<tool_call|>",
            .unknown => "prompt",
        })) };
    }
};
const Stub = struct {
    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .name = "qwen27" };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

// The compiler boundary records the production grammar; no xgrammar library or GPU is loaded in this fixture.
const Compile = struct {
    fn compiler(c: *Compile) api.grammar.Compiler {
        var out: api.grammar.Compiler = undefined;
        out.h = c;
        out.words = 4;
        out.mutex = .init;
        out.f.compile = compile;
        out.f.free = free;
        out.f.close = free;
        return out;
    }
    fn compile(ptr: *anyopaque, kind: c_int, _: [*]const u8, _: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
        std.debug.assert(kind == 5);
        return ptr;
    }
    fn free(_: *anyopaque) callconv(.c) void {}
};
fn value(g: std.mem.Allocator, s: []const u8) !json.Value {
    return (try json.parse(g, s)).ok;
}

test "required tools: unsupported required tools name the actual backend before any grammar or admission" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const f = try value(cx.a, "{\"tool_call_required\":true}");
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 200 };
    const offered = (try value(cx.a, specs)).array;
    try std.testing.expectError(error.Refused, srv.checkFeatures(&cx, f, true, false, &.{1}, offered, &request));
    try std.testing.expectEqual(@as(u16, 400), cx.status());
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "qwen27") != null);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "grammar") != null);
    try std.testing.expect(request.structure == null and request.call == null);
    srv.info.name = "lanes";
    srv.config.identity.backend = "cuda";
    try std.testing.expectError(error.Refused, srv.checkFeatures(&cx, f, true, false, &.{1}, offered, &request));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "cuda") != null);
}

test "required tools: Gemma and unknown templates can constrain required calls on a grammar backend" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    for ([_]@FieldType(Text, "form"){ .xml, .json, .gemma, .unknown }) |form| {
        var text: Text = .{ .form = form };
        const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        defer srv.deinit();
        srv.info.structures = true;
        srv.grammars = &compiler;
        var cx: errors.Cx = .{ .a = g };
        var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 200 };
        try srv.checkFeatures(&cx, try value(g, "{\"tool_call_required\":true}"), true, false, &.{1}, (try value(g, specs)).array, &request);
        try std.testing.expect(request.structure != null);
    }
}

fn body(cx: *errors.Cx, response: bool, choice: []const u8) !json.Value {
    const History = struct {
        pub fn conversation(_: @This(), c: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
            return c.refuse("this fixture has no stored response");
        }
    };
    const b = try json.newObject(cx.a);
    if (response) {
        try b.put(cx.a, "input", .{ .string = "hi" });
        const offered = (try value(cx.a, specs)).array;
        const flat = try cx.a.alloc(json.Value, offered.len);
        for (offered, flat) |tool, *t| {
            const f = tool.get("function").?.object;
            try f.put(cx.a, "type", .{ .string = "function" });
            t.* = .{ .object = f };
        }
        try b.put(cx.a, "tools", .{ .array = flat });
    } else {
        try b.put(cx.a, "messages", try value(cx.a, "[{\"role\":\"user\",\"content\":\"hi\"}]"));
        try b.put(cx.a, "tools", try value(cx.a, specs));
    }
    try b.put(cx.a, "tool_choice", try value(cx.a, choice));
    try b.put(cx.a, "parallel_tool_calls", .{ .bool = false });
    return if (response) (try responses.translate(cx, .{ .object = b }, History{})).chat else .{ .object = b };
}

test "required tools: both routes parse required and named calls and enforce a complete tool reply with drafted equals plain" {
    const witness = @import("tool_required_witness.zig");
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    for ([_]bool{ false, true }) |response| for ([_]bool{ false, true }) |named| for ([_]@FieldType(Text, "form"){ .xml, .json, .gemma, .unknown }) |form| {
        var text: Text = .{ .form = form };
        const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        defer srv.deinit();
        srv.info.structures = true;
        srv.grammars = &compiler;
        var cx: errors.Cx = .{ .a = g };
        const choice = if (!named) "\"required\"" else if (response) "{\"type\":\"function\",\"name\":\"now\"}" else "{\"type\":\"function\",\"function\":{\"name\":\"now\"}}";
        const input = try openai.chatInput(srv, &cx, try body(&cx, response, choice));
        try std.testing.expect(input.fields.get("tool_call_required").?.truthy());
        try std.testing.expectEqual(@as(usize, if (named) 1 else 2), input.tools.len);
        var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 512 };
        try srv.checkFeatures(&cx, input.fields, true, false, &.{1}, input.tools, &request);
        const structure = request.structure.?;
        const root = try value(g, structure.text);
        const format = root.get("format").?;
        try std.testing.expectEqualStrings("sequence", format.strField("type").?);
        try std.testing.expectEqual(@as(usize, 3), format.get("elements").?.array.len);
        const reply = try witness.text(g, root);
        try std.testing.expect(std.mem.startsWith(u8, reply, if (form == .gemma) "<|tool_call>call:" else "<tool_call>"));
        if (named) try std.testing.expect(std.mem.indexOf(u8, structure.text, "lookup") == null);
        const want = try Text.encode(undefined, g, reply, false);
        for ([_]u64{ 1235, 1236 }) |seed| {
            const plain = try witness.run(want, 0, seed);
            defer a.free(plain.tokens);
            for ([_]u32{ 1, 4 }) |depth| {
                const drafted = try witness.run(want, depth, seed);
                defer a.free(drafted.tokens);
                try std.testing.expectEqualSlices(u32, plain.tokens, drafted.tokens);
                try std.testing.expect(drafted.drafted > 0);
            }
            const decoded = try Text.decode(undefined, g, plain.tokens);
            const result = try parse.parse(g, decoded, input.tools, 1);
            try std.testing.expectEqualStrings("", result.content);
            try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
            try std.testing.expectEqualStrings(if (named) "now" else "lookup", result.calls.?[0].get("function").?.strField("name").?);
        }
    };
}

test "required tools: required without tools, unknown names and competing grammars are refused while auto and none stay optional" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    for ([_]bool{ false, true }) |response| {
        const missing = try body(&cx, response, "\"required\"");
        _ = missing.object.swapRemove("tools");
        try std.testing.expectError(error.Refused, openai.chatInput(srv, &cx, missing));
        const unknown = if (response) "{\"type\":\"function\",\"name\":\"absent\"}" else "{\"type\":\"function\",\"function\":{\"name\":\"absent\"}}";
        try std.testing.expectError(error.Refused, openai.chatInput(srv, &cx, try body(&cx, response, unknown)));
        const conflict = try body(&cx, response, "\"required\"");
        try conflict.object.put(cx.a, "guided_regex", .{ .string = "x" });
        try std.testing.expectError(error.Refused, openai.chatInput(srv, &cx, conflict));
        for ([_][]const u8{ "\"auto\"", "\"none\"" }) |choice| {
            const input = try openai.chatInput(srv, &cx, try body(&cx, response, choice));
            try std.testing.expect(input.fields.get("tool_call_required") == null);
            var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 20 };
            try srv.checkFeatures(&cx, input.fields, input.tools.len > 0, false, &.{1}, input.tools, &request);
            try std.testing.expect(request.structure == null);
        }
    }
}

test "required tools: Gemma constrained JSON arguments preserve punctuation inside string values" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    const reply = "<|tool_call>call:lookup{\"q\":\"a,b:c {d:e}\"}<tool_call|>";
    const result = try parse.parse(g, reply, (try value(g, specs)).array, 1);
    try std.testing.expect(result.calls != null);
    const arguments = try value(g, result.calls.?[0].get("function").?.strField("arguments").?);
    try std.testing.expectEqualStrings("a,b:c {d:e}", arguments.strField("q").?);
}

test "qwen27 tool calls: required and named calls use the masked lane fake through both routes" {
    const witness = @import("tool_required_witness.zig");
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    for ([_]bool{ false, true }) |response| for ([_]bool{ false, true }) |named| for ([_]@FieldType(Text, "form"){ .xml, .json, .gemma, .unknown }) |form| {
        var text: Text = .{ .form = form };
        const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        defer srv.deinit();
        srv.info.structures = @import("qwen27_masks").supported;
        srv.grammars = &compiler;
        var cx: errors.Cx = .{ .a = g };
        const choice = if (!named) "\"required\"" else if (response) "{\"type\":\"function\",\"name\":\"now\"}" else "{\"type\":\"function\",\"function\":{\"name\":\"now\"}}";
        const input = try openai.chatInput(srv, &cx, try body(&cx, response, choice));
        try std.testing.expect(input.fields.get("tool_call_required").?.truthy());
        try std.testing.expectEqual(@as(usize, if (named) 1 else 2), input.tools.len);
        var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 512 };
        try srv.checkFeatures(&cx, input.fields, true, false, &.{1}, input.tools, &request);
        const structure = request.structure.?;
        const root = try value(g, structure.text);
        const format = root.get("format").?;
        try std.testing.expectEqualStrings("sequence", format.strField("type").?);
        try std.testing.expectEqual(@as(usize, 3), format.get("elements").?.array.len);
        const reply = try witness.text(g, root);
        try std.testing.expect(std.mem.startsWith(u8, reply, if (form == .gemma) "<|tool_call>call:" else "<tool_call>"));
        if (named) try std.testing.expect(std.mem.indexOf(u8, structure.text, "lookup") == null);
        const want = try Text.encode(undefined, g, reply, false);
        for ([_]u64{ 1235, 1236 }) |seed| {
            const plain = try witness.runTarget(want, 0, seed, @import("qwen27_mask_fake").Adapter);
            defer a.free(plain.tokens);
            for ([_]u32{ 1, 4 }) |depth| {
                const drafted = try witness.runTarget(want, depth, seed, @import("qwen27_mask_fake").Adapter);
                defer a.free(drafted.tokens);
                try std.testing.expectEqualSlices(u32, plain.tokens, drafted.tokens);
                try std.testing.expect(drafted.drafted > 0);
            }
            const decoded = try Text.decode(undefined, g, plain.tokens);
            const result = try parse.parse(g, decoded, input.tools, 1);
            try std.testing.expectEqualStrings("", result.content);
            try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
            try std.testing.expectEqualStrings(if (named) "now" else "lookup", result.calls.?[0].get("function").?.strField("name").?);
        }
    };
}
