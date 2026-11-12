//! Native XTML rendering, adapted from Moonshot AI's K3.1 release encoding_k3.py at revision 6d0a0202.

// Copyright (c) 2026 Moonshot AI; release license: LICENSES/Kimi-K3.1-Pro.txt.
const std = @import("std");
const json = @import("json");
const values = @import("kimi_values.zig");
const V = json.Value;
const A = std.mem.Allocator;

pub const Segment = struct { text: []const u8, allow_special: bool = false, request: bool = false };
pub const Options = struct {
    thinking: bool = true,
    add_generation_prompt: bool = true,
    thinking_effort: ?[]const u8 = null,
    tools: V = .null,
    tool_choice: V = .null,
    response_format: V = .null,
    response_schema: V = .null,
};
pub const efforts = [_][]const u8{ "low", "medium", "high", "max" };
const Attr = struct { key: []const u8, value: []const u8, request: bool = false };

const Builder = struct {
    a: A,
    list: std.ArrayList(Segment) = .empty,

    fn add(b: *Builder, text: []const u8, allow_special: bool, request: bool) !void {
        if (text.len > 0) try b.list.append(b.a, .{ .text = text, .allow_special = allow_special, .request = request });
    }
    fn plain(b: *Builder, text: []const u8) !void {
        try b.add(text, false, false);
    }
    fn data(b: *Builder, text: []const u8) !void {
        try b.add(text, false, true);
    }
    fn control(b: *Builder, text: []const u8) !void {
        try b.add(text, true, false);
    }
    fn open(b: *Builder, tag: []const u8, attrs: []const Attr) !void {
        try b.control("<|open|>");
        try b.plain(tag);
        for (attrs) |attr| {
            try b.plain(try std.mem.concat(b.a, u8, &.{ " ", attr.key }));
            try b.plain("=\"");
            const amp = try std.mem.replaceOwned(u8, b.a, attr.value, "&", "&amp;");
            const escaped = try std.mem.replaceOwned(u8, b.a, amp, "\"", "&quot;");
            try b.add(escaped, false, attr.request);
            try b.plain("\"");
        }
        try b.control("<|sep|>");
    }
    fn close(b: *Builder, tag: []const u8) !void {
        try b.control("<|close|>");
        try b.plain(tag);
        try b.control("<|sep|>");
    }
    fn end(b: *Builder) !void {
        try b.control("<|end_of_msg|>");
    }
    fn content(b: *Builder, v: V) !void {
        if (v == .string) return b.data(v.string);
        if (v == .null) return;
        if (v != .array) return error.BadKimiContent;
        for (v.array) |part| {
            if (values.fieldIs(part, "type", "image") or values.fieldIs(part, "type", "image_url")) return error.KimiVisionUnsupported;
            try b.data(part.strField("text") orelse return error.BadKimiContent);
        }
    }
    fn internal(b: *Builder, kind: []const u8, body: []const u8, request: bool) !void {
        try b.open("message", &.{ .{ .key = "role", .value = "system" }, .{ .key = "type", .value = kind } });
        try b.add(std.mem.trim(u8, body, " \t\r\n"), false, request);
        try b.close("message");
        try b.end();
    }
    fn tools(b: *Builder, tools_: V, dynamic: bool) !void {
        const encoded = try json.stringify(b.a, try values.sorted(b.a, tools_), .{ .ascii = false, .compact = true });
        const prefix = if (dynamic)
            "## New Tools Available\nThe system dynamically extends the toolset via lazy-loading.\n" ++
                "You have access to all existing and extended tools.\nHere are the specs for the extended tools.\n\n```json\n"
        else
            "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n";
        try b.internal("tool-declare", try std.mem.concat(b.a, u8, &.{ prefix, encoded, "\n```" }), true);
    }
    fn assistant(b: *Builder, message: V, thinking: bool) !void {
        if (thinking) {
            try b.open("think", &.{});
            if (message.firstTruthy(&.{ "reasoning_content", "reasoning" })) |reason| try b.data(try values.text(b.a, reason));
            try b.close("think");
        }
        try b.open("response", &.{});
        try b.content(message.get("content") orelse .null);
        try b.close("response");
        const calls = message.get("tool_calls") orelse return;
        if (!calls.truthy()) return;
        if (calls != .array) return error.BadKimiCalls;
        try b.open("tools", &.{});
        for (calls.array, 0..) |call, index| {
            const function = call.get("function") orelse call;
            try b.open("call", &.{
                .{ .key = "tool", .value = function.strField("name") orelse return error.BadKimiCalls, .request = true },
                .{ .key = "index", .value = try std.fmt.allocPrint(b.a, "{d}", .{index + 1}) },
            });
            const args = try values.arguments(b.a, function.get("arguments") orelse .null);
            if (args.raw) |raw| {
                try b.open("json", &.{.{ .key = "type", .value = "object" }});
                try b.data(raw);
                try b.close("json");
            } else {
                for (args.items) |arg| {
                    try b.open("argument", &.{
                        .{ .key = "key", .value = arg.key, .request = true },
                        .{ .key = "type", .value = arg.kind },
                    });
                    try b.data(arg.text);
                    try b.close("argument");
                }
            }
            try b.close("call");
        }
        try b.close("tools");
    }
};

pub fn render(a: A, messages: V, o: Options) ![]const Segment {
    if (messages != .array) return error.BadKimiMessages;
    const ordered = try values.orderResults(a, messages.array);
    var b: Builder = .{ .a = a };
    if (o.tools.truthy()) try b.tools(o.tools, false);
    if (o.thinking) if (o.thinking_effort) |effort| {
        var valid = false;
        for (efforts) |e| valid = valid or std.mem.eql(u8, e, effort);
        if (!valid) return error.BadKimiEffort;
        const body = try std.mem.concat(a, u8, &.{
            "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), ",
            "supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=",
            effort,
            "`.",
        });
        try b.internal("thinking-effort", body, false);
    };
    var calls: []const V = &.{};
    var tool_index: usize = 0;
    for (ordered) |message| {
        const role = message.strField("role") orelse return error.BadKimiMessages;
        if (std.mem.eql(u8, role, "system") and json.truthyField(message, "tools")) {
            try b.tools(message.get("tools").?, true);
            continue;
        }
        var attrs: std.ArrayList(Attr) = .empty;
        try attrs.append(a, .{ .key = "role", .value = role });
        if (std.mem.eql(u8, role, "tool")) {
            tool_index += 1;
            const tool = message.strField("tool") orelse message.strField("name") orelse blk: {
                if (tool_index > calls.len) return error.MissingKimiToolName;
                const call = calls[tool_index - 1];
                break :blk (call.get("function") orelse call).strField("name") orelse return error.MissingKimiToolName;
            };
            try attrs.append(a, .{ .key = "tool", .value = tool, .request = true });
            try attrs.append(a, .{ .key = "index", .value = try std.fmt.allocPrint(a, "{d}", .{tool_index}) });
        } else {
            const known = std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "user") or std.mem.eql(u8, role, "assistant");
            if (!known) return error.BadKimiRole;
            if (message.firstTruthy(&.{"name"})) |name| {
                if (name != .string) return error.BadKimiName;
                try attrs.append(a, .{ .key = "name", .value = name.string, .request = true });
            }
        }
        try b.open("message", attrs.items);
        if (std.mem.eql(u8, role, "assistant")) {
            const cs = message.get("tool_calls") orelse .null;
            calls = if (cs == .array) cs.array else &.{};
            tool_index = 0;
            try b.assistant(message, o.thinking);
        } else try b.content(message.get("content") orelse .null);
        try b.close("message");
        try b.end();
    }
    if (eq(o.tool_choice, "required"))
        try b.internal("tool-choice", "The system is invoked with `tool_choice=required`.\nYou MUST call tools in the next message.", false)
    else if (eq(o.tool_choice, "none"))
        try b.internal("tool-choice", "The system is invoked with `tool_choice=none`.\nYou MUST NOT call any tools in the next message.", false);
    const format = o.response_format.strField("type") orelse o.response_format.str() orelse "";
    if (std.mem.eql(u8, format, "json_object")) {
        try b.internal("response-format", "The system is invoked with `response_format=json_object`.\n" ++
            "Your response must be raw JSON data without markdown code blocks (```json) or any additional formatting.", false);
    } else if (std.mem.eql(u8, format, "json_schema")) {
        const js = o.response_format.get("json_schema") orelse .null;
        const schema = if (o.response_schema != .null) o.response_schema else js.get("schema") orelse js.get("json_schema") orelse js;
        const encoded = try json.stringify(a, try values.sorted(a, schema), .{ .ascii = false, .compact = true });
        try b.internal("response-format", try std.mem.concat(a, u8, &.{
            "The system is invoked with `response_format=json_schema`.\n",
            "Your response must be raw JSON data without markdown code blocks (```json) or any additional formatting.\n",
            "The JSON data must match the following schema:\n```json\n",
            encoded,
            "\n```",
        }), true);
    }
    if (o.add_generation_prompt) {
        try b.open("message", &.{.{ .key = "role", .value = "assistant" }});
        try b.open(if (o.thinking) "think" else "response", &.{});
    }
    return b.list.items;
}

fn eq(v: V, s: []const u8) bool {
    return v == .string and std.mem.eql(u8, v.string, s);
}
