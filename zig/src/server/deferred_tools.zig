//! Deferred tools (``defer_loading``) and the ``tool_reference`` parts a tool result loads them with.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const textPart = @import("messages.zig").textPart;

/// A template that names both fields renders them itself (GLM-5.3); any other gets the text form.
pub fn native(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "tool_reference") != null and std.mem.indexOf(u8, source, "defer_loading") != null;
}

/// A tool's function object: an OpenAI ``{"function": {...}}`` entry's, or a bare spec.
fn function(tool: Value) ?Value {
    if (tool != .object) return null;
    const inner = tool.get("function") orelse return tool;
    return if (inner == .object) inner else null;
}

fn deferred(tool: Value) bool {
    const f = function(tool) orelse return false;
    const flag = f.get("defer_loading") orelse return false;
    return flag == .bool and flag.bool;
}

/// The tools a template that cannot read ``defer_loading`` lists: the deferred ones left out, none as null.
pub fn shown(a: Allocator, tools: ?Value) Allocator.Error!?Value {
    const list = tools orelse return null;
    if (list != .array) return tools;
    for (list.array) |tool| {
        if (deferred(tool)) break;
    } else return tools;
    var kept: std.ArrayList(Value) = .empty;
    for (list.array) |tool| if (!deferred(tool)) try kept.append(a, tool);
    return if (kept.items.len > 0) Value{ .array = kept.items } else null;
}

fn spec(tools: ?Value, name: []const u8) ?Value {
    const list = tools orelse return null;
    if (list != .array) return null;
    for (list.array) |tool| {
        const f = function(tool) orelse continue;
        const n = f.get("name") orelse continue;
        if (n == .string and std.mem.eql(u8, n.string, name)) return f;
    }
    return null;
}

/// A referenced tool as text: its name, description and parameters as a template's ``tojson`` writes them.
fn spelled(a: Allocator, f: Value) Allocator.Error![]const u8 {
    const o = try json.newObject(a);
    for ([_][]const u8{ "name", "description", "parameters" }) |key| if (f.get(key)) |v| try o.put(a, key, v);
    return json.stringify(a, .{ .object = o }, .{ .ascii = false });
}

fn hasReference(content: Value) bool {
    if (content != .array) return false;
    for (content.array) |part| if (part.typeIs("tool_reference")) return true;
    return false;
}

/// Every ``tool_reference`` checked against all offered tools; a template that can't read one gets its spec as text.
pub fn resolve(cx: *Cx, messages: Value, tools: ?Value, template_reads: bool) errors.Refused!Value {
    if (messages != .array) return messages;
    for (messages.array) |m| {
        if (m.get("content")) |c| if (hasReference(c)) break;
    } else return messages;
    const a = cx.a;
    const out = try a.dupe(Value, messages.array);
    for (out) |*slot| {
        const content = slot.get("content") orelse continue;
        if (!hasReference(content)) continue;
        var parts: std.ArrayList(Value) = .empty;
        var texts: std.ArrayList([]const u8) = .empty;
        for (content.array) |part| {
            if (!part.typeIs("tool_reference")) {
                try parts.append(a, part);
                if (part.typeIs("text")) if (part.get("text")) |t| if (t == .string) try texts.append(a, t.string);
                continue;
            }
            const n = part.get("name") orelse Value.null;
            if (n != .string or n.string.len == 0) return cx.refuse("a tool_reference part must name a tool");
            const name = n.string;
            const f = spec(tools, name) orelse return cx.fail(.request, "Tool reference '{s}' not found in available tools", .{name});
            if (template_reads) {
                try parts.append(a, part);
                continue;
            }
            const text = try spelled(a, f);
            try parts.append(a, try textPart(a, text));
            try texts.append(a, text);
        }
        const copy = try json.copyObject(a, slot.object);
        const all_text = texts.items.len == parts.items.len;
        try copy.put(a, "content", if (!template_reads and all_text) .{ .string = try std.mem.join(a, "\n", texts.items) } else .{ .array = parts.items });
        slot.* = .{ .object = copy };
    }
    return .{ .array = out };
}
