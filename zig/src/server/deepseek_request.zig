//! The native DeepSeek Chat Completions input shapes; other renderers keep their own contract.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");

fn string(v: ?json.Value) bool {
    return v != null and v.? == .string;
}

fn name(v: ?json.Value) bool {
    if (!string(v) or v.?.string.len == 0 or v.?.string.len > 64) return false;
    for (v.?.string) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return false;
    return true;
}

pub fn message(cx: *errors.Cx, m: json.Value) errors.Refused!void {
    if (m.get("role")) |role| if (role == .string and std.mem.eql(u8, role.string, "user")) {
        const content = m.get("content");
        if (content == null or content.? == .null) return cx.refuse("a native DeepSeek user message must contain text or text parts");
    };
    if (m.field("tool_calls")) |calls| {
        if (calls != .array) return cx.refuse("tool_calls must be a list");
        for (calls.array) |call| {
            if (call != .object or !string(call.get("id"))) return cx.refuse("a tool call must contain an id string");
            const kind = call.get("type");
            if (!string(kind) or !std.mem.eql(u8, kind.?.string, "function")) return cx.refuse("a tool call type must be function");
            const f = call.get("function");
            if (f == null or f.? != .object or !name(f.?.get("name")) or !string(f.?.get("arguments")))
                return cx.refuse("a tool call function must contain name and arguments strings");
        }
    }
    if (m.get("role")) |role| if (role == .string and std.mem.eql(u8, role.string, "tool") and !string(m.get("tool_call_id")))
        return cx.refuse("a tool message must contain a tool_call_id string");
}

fn tools(cx: *errors.Cx, value: ?json.Value) errors.Refused!void {
    const list = value orelse return;
    if (list == .null) return;
    if (list != .array) return cx.refuse("tools must be a list");
    for (list.array) |tool| {
        const kind = tool.get("type");
        const f = tool.get("function");
        if (!string(kind) or !std.mem.eql(u8, kind.?.string, "function") or f == null or f.? != .object or !name(f.?.get("name")))
            return cx.refuse("tools must contain function objects with a 1-64 character alphanumeric, underscore or hyphen name");
    }
}

pub fn active(cx: *errors.Cx, value: ?json.Value, choice: ?json.Value) errors.Refused![]json.Value {
    try tools(cx, value);
    return @import("tool_specs.zig").active(cx, value, choice);
}
