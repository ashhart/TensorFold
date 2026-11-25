//! Inkling 2's required calls in its template's own form: each call a model message, then the end of sampling.
const std = @import("std");
const json = @import("json");
const tg = @import("tool_grammar.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const open = "<|message_model|>";
pub const invoke = "<|content_invoke_tool_json|>";
pub const end = "<|end_message|>";

/// Whether `rendered`, the template's turn calling `probe`, writes Inkling 2's own call message.
pub fn isNative(rendered: []const u8, comptime probe: []const u8) bool {
    return std.mem.indexOf(u8, rendered, open ++ probe ++ invoke) != null;
}

/// The structural tag: one call message, or (`parallel`) one or more back to back; the stop is the end of sampling.
pub fn build(a: Allocator, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const calls = try a.alloc(Value, tools.len);
    for (tools, calls) |tool, *c| c.* = try call(a, try tool_specs.toolName(a, tool), try tg.parameters(a, tool));
    const one = if (calls.len == 1) calls[0] else try tg.node(a, "or", &.{.{ "elements", .{ .array = calls } }});
    const format = if (!parallel) one else try tg.node(a, "sequence", &.{.{ "elements", try tg.list(a, &.{ one, try tg.node(a, "star", &.{.{ "content", one }}) }) }});
    return json.stringify(a, try tg.node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`, args in `schema`.
fn call(a: Allocator, name: []const u8, schema: Value) Allocator.Error!Value {
    const fixed = try json.newObject(a);
    try fixed.put(a, "const", .{ .string = name });
    const props = try json.newObject(a);
    try props.put(a, "name", .{ .object = fixed });
    try props.put(a, "args", schema);
    const whole = try json.newObject(a);
    try whole.put(a, "type", .{ .string = "object" });
    try whole.put(a, "properties", .{ .object = props });
    try whole.put(a, "required", try tg.list(a, &.{ .{ .string = "name" }, .{ .string = "args" } }));
    const opener = try std.mem.concat(a, u8, &.{ open, name, invoke });
    return tg.node(a, "sequence", &.{.{ "elements", try tg.list(a, &.{ try tg.constant(a, opener), try tg.arguments(a, .{ .object = whole }, "json"), try tg.constant(a, end) }) }});
}

test "Inkling 2 reply: the native required grammar writes each call as its own model message, repeated only in parallel" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]")).ok;
    const single = try build(a, tools.array, false);
    for ([_][]const u8{ "<|message_model|>get_weather<|content_invoke_tool_json|>", "\"args\"", "\"const\":\"get_weather\"", "\"city\"", "<|end_message|>" }) |text|
        try std.testing.expect(std.mem.indexOf(u8, single, text) != null);
    // no generic envelope, no blanks rule and no repeat without parallel calls
    for ([_][]const u8{ "<tool_call>", "\"regex\"", "\"type\":\"star\"" }) |text| try std.testing.expect(std.mem.indexOf(u8, single, text) == null);
    try std.testing.expect(std.mem.indexOf(u8, try build(a, tools.array, true), "\"type\":\"star\"") != null);
    try std.testing.expect(isNative("<|message_model|>tfprobe_fn<|content_invoke_tool_json|>{\"name\":\"tfprobe_fn\",\"args\":{}}<|end_message|>", "tfprobe_fn"));
    try std.testing.expect(!isNative("<tool_call>{\"name\":\"tfprobe_fn\",\"arguments\":{}}</tool_call>", "tfprobe_fn"));
}
