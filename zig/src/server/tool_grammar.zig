//! tool_choice "required" or a named function as a grammar: calls in the template's markup, arguments in their schema.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const lfm2 = @import("tool_grammar_lfm2.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The template's call envelope, with XML parameters or JSON arguments.
pub const Style = enum { xml, json, gemma, glm };

const probe = "tfprobe_fn";
const key = "tfprobe_k";

/// Blanks around and between calls: the template's newlines, never an endless run.
const blanks = "[ \\t\\r\\n]{0,8}";

/// The most blank characters between an argument's JSON tokens (grammar.py BLANKS).
const value_blanks = 32;

/// The style of `rendered`, a turn that calls `probe` with {key: "v"}; null: markup this grammar does not write.
pub fn style(rendered: []const u8) ?Style {
    if (std.mem.indexOf(u8, rendered, "<|tool_call>call:" ++ probe ++ "{") != null) return .gemma;
    const at = std.mem.lastIndexOf(u8, rendered, "<tool_call>") orelse return null;
    const text = rendered[at..];
    if (std.mem.indexOf(u8, text, "<function=" ++ probe ++ ">") != null and std.mem.indexOf(u8, text, "<parameter=" ++ key ++ ">") != null) return .xml;
    if (std.mem.indexOf(u8, text, "\"" ++ probe ++ "\"") != null and std.mem.indexOf(u8, text, "\"arguments\"") != null) return .json;
    if (std.mem.indexOf(u8, text, "<tool_call>" ++ probe ++ "<arg_key>" ++ key ++ "</arg_key>") != null) return .glm;
    return null;
}

/// The grammar for a reply that must call one of `tools`: one call, or (`parallel`) one or more, then the end.
pub fn structure(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, thinking: bool) errors.Refused!api.Structure {
    const probe_turn = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"" ++ probe ++ "\", \"arguments\": {\"" ++ key ++ "\": \"v\"}}}]}]")).ok;
    var scratch: Cx = .{ .a = cx.a };
    const rendered = if (prompt.renderIds(srv, &scratch, probe_turn, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => "",
    };
    const s = style(rendered) orelse .json; // unrecognized markup uses the canonical call form the parser accepts
    const text = if (lfm2.isStyle(rendered, probe, key)) try lfm2.build(cx.a, tools, parallel) else try build(cx.a, s, tools, parallel);
    const field = try std.fmt.allocPrint(cx.a, "tool_choice \"required\" or a named function on backend '{s}'", .{srv.backendName()});
    return srv.compiledStructure(cx, .structural_tag, text, field, thinking);
}

/// The structural tag's JSON text (xgrammar's form).
pub fn build(a: Allocator, s: Style, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const calls = try a.alloc(Value, tools.len);
    for (tools, calls) |t, *c| c.* = try call(a, s, try tool_specs.toolName(a, t), try parameters(a, t));
    const one: Value = if (calls.len == 1) calls[0] else try node(a, "or", &.{.{ "elements", .{ .array = calls } }});
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    var parts: std.ArrayList(Value) = .empty;
    try parts.appendSlice(a, &.{ ws, one });
    if (parallel) try parts.append(a, try node(a, "star", &.{.{ "content", try node(a, "sequence", &.{.{ "elements", try list(a, &.{ ws, one }) }}) }}));
    try parts.append(a, ws);
    const format = try node(a, "sequence", &.{.{ "elements", .{ .array = parts.items } }});
    return json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// One call to `name`: <tool_call>, the name and its arguments in the template's style, </tool_call>.
fn call(a: Allocator, s: Style, name: []const u8, schema: Value) Allocator.Error!Value {
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    const elements = switch (s) {
        .xml => try list(a, &.{
            try constant(a, "<tool_call>"),                                        ws,
            try constant(a, try std.fmt.allocPrint(a, "<function={s}>", .{name})), try arguments(a, schema, "qwen_xml"),
            try constant(a, "</function>"),                                        ws,
            try constant(a, "</tool_call>"),
        }),
        .gemma => try list(a, &.{ try constant(a, try std.fmt.allocPrint(a, "<|tool_call>call:{s}", .{name})), try arguments(a, schema, "json"), try constant(a, "<tool_call|>") }),
        .json => blk: {
            const props = try json.newObject(a);
            const fixed = try json.newObject(a);
            try fixed.put(a, "const", .{ .string = name });
            try props.put(a, "name", .{ .object = fixed });
            try props.put(a, "arguments", schema);
            const whole = try json.newObject(a);
            try whole.put(a, "type", .{ .string = "object" });
            try whole.put(a, "properties", .{ .object = props });
            try whole.put(a, "required", try list(a, &.{ .{ .string = "name" }, .{ .string = "arguments" } }));
            break :blk try list(a, &.{ try constant(a, "<tool_call>"), ws, try arguments(a, .{ .object = whole }, "json"), ws, try constant(a, "</tool_call>") });
        },
        // GLM's template writes NAME<arg_key>K</arg_key><arg_value>V</arg_value>... with no blanks: xgrammar's glm_xml
        .glm => try list(a, &.{
            try constant(a, try std.fmt.allocPrint(a, "<tool_call>{s}", .{name})), try arguments(a, schema, "glm_xml"),
            try constant(a, "</tool_call>"),
        }),
    };
    return node(a, "sequence", &.{.{ "elements", elements }});
}

/// The tool's parameter schema (OpenAI's function.parameters, a bare tool's parameters or input_schema).
fn parameters(a: Allocator, tool: Value) Allocator.Error!Value {
    const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
    for ([_][]const u8{ "parameters", "input_schema" }) |k| if (function.get(k)) |p| if (p == .object) return p;
    const empty = try json.newObject(a);
    try empty.put(a, "type", .{ .string = "object" });
    try empty.put(a, "properties", .{ .object = try json.newObject(a) });
    return .{ .object = empty };
}

fn arguments(a: Allocator, schema: Value, style_name: []const u8) Allocator.Error!Value {
    return node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "style", .{ .string = style_name } }, .{ "max_whitespace_cnt", try json.intValue(a, value_blanks) } });
}

fn constant(a: Allocator, text: []const u8) Allocator.Error!Value {
    return node(a, "const_string", &.{.{ "value", .{ .string = text } }});
}

fn list(a: Allocator, items: []const Value) Allocator.Error!Value {
    return .{ .array = try a.dupe(Value, items) };
}

/// {"type": kind, fields...}.
fn node(a: Allocator, kind: []const u8, entries: []const struct { []const u8, Value }) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = kind });
    for (entries) |e| try o.put(a, e[0], e[1]);
    return .{ .object = o };
}

test "the probe call's markup picks the style" {
    try std.testing.expectEqual(Style.xml, style("<think></think><tool_call>\n<function=tfprobe_fn>\n<parameter=tfprobe_k>\nv\n</parameter>\n</function>\n</tool_call>\n").?);
    try std.testing.expectEqual(Style.json, style("<tool_call>\n{\"name\": \"tfprobe_fn\", \"arguments\": {\"tfprobe_k\": \"v\"}}\n</tool_call>").?);
    try std.testing.expectEqual(Style.gemma, style("<|tool_call>call:tfprobe_fn{tfprobe_k:<|\"|>v<|\"|>}<tool_call|>").?);
    try std.testing.expectEqual(Style.glm, style("</think><tool_call>tfprobe_fn<arg_key>tfprobe_k</arg_key><arg_value>v</arg_value></tool_call>").?);
}

test "a required call's grammar names every offered tool and its schema; one call without parallel" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"now\"}}]")).ok;
    const many = try build(a, .xml, tools.array, true);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<function=get_weather>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<function=now>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"style\":\"qwen_xml\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"required\":[\"city\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"type\":\"star\"") != null);
    const single = try build(a, .json, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"type\":\"star\"") == null and std.mem.indexOf(u8, single, "\"type\":\"or\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"const\":\"get_weather\"") != null);
    const glm = try build(a, .glm, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, glm, "\"value\":\"<tool_call>get_weather\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, glm, "\"style\":\"glm_xml\"") != null);
}

test "GLM required tools keep their name, schema and parallel policy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const picked = style("<tool_call>tfprobe_fn<arg_key>tfprobe_k</arg_key><arg_value>v</arg_value></tool_call>") orelse return error.TestExpectedStyle;
    try std.testing.expectEqualStrings("glm", @tagName(picked));
    const tools = (try json.parse(a, "[{\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}},{\"name\":\"clock\"}]")).ok;
    const single = try build(a, picked, tools.array[0..1], false);
    for ([_][]const u8{ "<tool_call>weather", "glm_xml", "city", "</tool_call>" }) |text| try std.testing.expect(std.mem.indexOf(u8, single, text) != null);
    try std.testing.expect(std.mem.indexOf(u8, single, "clock") == null);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"type\":\"star\"") == null);
    const many = try build(a, picked, tools.array, true);
    try std.testing.expect(std.mem.indexOf(u8, many, "<tool_call>clock") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"type\":\"star\"") != null);
}
