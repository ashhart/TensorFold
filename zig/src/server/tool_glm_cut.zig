//! A GLM call max_tokens cut (``NAME<arg_key>K</arg_key><arg_value>V``...): its arguments as an XML stream cuts them.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const tool_parse = @import("tool_parse.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const key_open = "<arg_key>";
const key_close = "</arg_key>";
const value_open = "<arg_value>";
const value_close = "</arg_value>";

/// A reply max_tokens cut inside a GLM call: the whole calls before it, then it, cut; null when it ends in none.
pub fn cutCalls(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error!?[]Value {
    const start = tool_parse.openCall(text) orelse return null;
    const cut = try cutCall(a, text[start + "<tool_call>".len ..], tools) orelse return null;
    const before = (try tool_parse.parse(a, text[0..start], tools, null)).calls orelse &.{};
    return try std.mem.concat(a, Value, &.{ before, &.{cut} });
}

/// The call an open envelope's ``written`` payload begins, arguments as far as written; null unless it names a tool.
fn cutCall(a: Allocator, written: []const u8, tools: []const Value) Allocator.Error!?Value {
    const block = try tool_parse.repairGlmKeys(a, written);
    var pos = std.mem.indexOf(u8, block, key_open) orelse return null; // the name is whole once its first key opens
    const name = (try offered(a, reply_text.pyStrip(block[0..pos]), tools)) orelse return null;
    const schemas = try tool_params.schemas(a, tools);
    var args: std.ArrayList(u8) = .empty;
    while (true) {
        pos = block.len - reply_text.pyLstrip(block[pos..]).len;
        if (!std.mem.startsWith(u8, block[pos..], key_open)) {
            if (cutTag(block[pos..], key_open) or cutTag(block[pos..], "</tool_call>")) break;
            return null; // text after an argument: not GLM's markup
        }
        const key_start = pos + key_open.len;
        const key_end = std.mem.indexOfPos(u8, block, key_start, key_close) orelse break; // cut inside a key
        var at = block.len - reply_text.pyLstrip(block[key_end + key_close.len ..]).len;
        if (!std.mem.startsWith(u8, block[at..], value_open)) {
            if (cutTag(block[at..], value_open)) break;
            return null;
        }
        at += value_open.len;
        const key = reply_text.pyStrip(block[key_start..key_end]);
        const schema = try tool_params.schemaOf(&schemas, name, key, a);
        const typed = try tool_params.schemaTakesJson(a, schema);
        try args.appendSlice(a, if (args.items.len > 0) "," else "{");
        try args.appendSlice(a, try json.quote(a, key, .{ .ascii = false }));
        try args.append(a, ':');
        const value_end = std.mem.indexOfPos(u8, block, at, value_close) orelse {
            const value = block[at .. block.len - reply_text.partialTag(block[at..], value_close)];
            const quoted = try json.quote(a, value, .{ .ascii = false });
            try args.appendSlice(a, if (typed) value else quoted[0 .. quoted.len - 1]); // its closing quote never came
            break;
        };
        const value = block[at..value_end];
        const whole = try tool_params.decode(a, value, schema, false); // as tool_parse reads a GLM value
        try args.appendSlice(a, if (typed) try json.stringify(a, whole, .{ .ascii = false }) else try json.quote(a, value, .{ .ascii = false }));
        pos = value_end + value_close.len;
    }
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    try function.put(a, "arguments", .{ .string = args.items });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
    try call.put(a, "type", .{ .string = "function" });
    try call.put(a, "function", .{ .object = function });
    return .{ .object = call };
}

/// ``rest`` is all the reply wrote of ``tag`` before the cut (nothing included).
fn cutTag(rest: []const u8, tag: []const u8) bool {
    return rest.len < tag.len and std.mem.startsWith(u8, tag, rest);
}

/// The offered tool's spelling of a GLM call's one-line ``name``, as ``tool_parse.parse`` matches it; null if none.
fn offered(a: Allocator, name: []const u8, tools: []const Value) Allocator.Error!?[]const u8 {
    if (name.len == 0 or std.mem.indexOfAny(u8, name, "<>\n\r") != null) return null;
    for (tools) |tool| {
        const spelled = try tool_specs.toolName(a, tool);
        if (spelled.len > 0 and std.ascii.eqlIgnoreCase(spelled, name)) return spelled;
    }
    return null;
}
