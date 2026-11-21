//! Raw-string arguments and custom/function namespaces for parsed tool calls.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const tool_params = @import("tool_params.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Error = @import("tool_dsml.zig").Error;

/// Canonical spellings of offered custom tools; comparisons follow the parser's case folding.
pub fn names(a: Allocator, tools: []const Value) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (tools) |tool| {
        const kind = tool.get("type") orelse continue;
        if (kind != .string or !std.mem.eql(u8, kind.string, "custom")) continue;
        try out.append(a, try tool_specs.toolName(a, tool));
    }
    return out.items;
}

pub fn functions(a: Allocator, tools: []const Value) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (tools) |tool| {
        const kind = tool.get("type") orelse continue;
        if (kind != .string or !std.mem.eql(u8, kind.string, "function")) continue;
        try out.append(a, try tool_specs.toolName(a, tool));
    }
    return out.items;
}

pub fn isCustom(offered: []const []const u8, name: []const u8) bool {
    for (offered) |candidate| if (std.ascii.eqlIgnoreCase(candidate, name)) return true;
    return false;
}

/// Ordinary functions retain the JSON-object contract; custom tools preserve their one input string exactly.
pub fn arguments(a: Allocator, name: []const u8, args: Value, custom: []const []const u8, function_names: []const []const u8, complete: bool) Error!Value {
    if (isCustom(custom, name)) {
        if (args == .string) return args;
        if (!isCustom(function_names, name)) return error.Invalid;
    }
    const object = try jsonObject(a, args);
    if (complete and !tool_params.finite(object)) return error.Invalid;
    return object;
}

/// ``_tool_json_object``: arguments as an object, Invalid where a function call is malformed.
pub fn jsonObject(a: Allocator, v: Value) Error!Value {
    switch (v) {
        .object => return v,
        .null => return .{ .object = try json.newObject(a) },
        .string => |s| {
            const doc = std.mem.trim(u8, s, " \t\r\n\x0b\x0c");
            if (doc.len == 0) return .{ .object = try json.newObject(a) };
            return try tool_params.parseObject(a, doc) orelse error.Invalid;
        },
        else => return error.Invalid,
    }
}

/// A custom call carries both the native chat function view and its Responses custom input view.
pub fn openaiCall(a: Allocator, name: []const u8, args: Value, custom: []const []const u8) Allocator.Error!Value {
    const is_custom = isCustom(custom, name) and args == .string;
    const text = if (is_custom) args.string else try json.stringify(a, args, .{ .ascii = false, .compact = true });
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    try function.put(a, "arguments", .{ .string = text });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
    try call.put(a, "type", .{ .string = if (is_custom) "custom" else "function" });
    try call.put(a, "function", .{ .object = function });
    if (is_custom) {
        const payload = try json.newObject(a);
        try payload.put(a, "name", .{ .string = name });
        try payload.put(a, "input", .{ .string = text });
        try call.put(a, "custom", .{ .object = payload });
    }
    return .{ .object = call };
}
