//! Responses function/custom tools, deferred references and tool_choice as chat tool specifications.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Choice = struct { choice: ?Value, offered: ?[]Value };

fn kindOf(v: Value) ?Value {
    return if (v == .object) v.get("type") else null;
}

fn kind(v: Value) []const u8 {
    const k = kindOf(v) orelse return "";
    return if (k == .string) k.string else "";
}

fn deferred(tool: Value) bool {
    const value = tool.get("defer_loading") orelse return false;
    return value == .bool and value.bool;
}

fn name(tool: Value) []const u8 {
    if (std.mem.eql(u8, kind(tool), "function")) return tool.get("function").?.strField("name") orelse "";
    return tool.strField("name") orelse "";
}

fn customFormat(cx: *Cx, tool: Value, out: *json.Object) errors.Refused!void {
    const format = tool.get("format") orelse return;
    if (format != .object) return cx.refuse("a custom tool's format must be a text or grammar object");
    const f = kind(format);
    if (std.mem.eql(u8, f, "text")) {
        if (format.object.count() != 1) return cx.refuse("a text custom-tool format has only type: text");
    } else if (std.mem.eql(u8, f, "grammar")) {
        const syntax = format.strField("syntax") orelse return cx.refuse("a grammar custom-tool format needs syntax: lark");
        const definition = format.strField("definition") orelse return cx.refuse("a grammar custom-tool format needs a definition string");
        if (!std.mem.eql(u8, syntax, "lark")) return cx.refuse("custom-tool grammar syntax must be lark");
        if (definition.len == 0 or format.object.count() != 3) return cx.refuse("a grammar custom-tool format needs only type, syntax and a non-empty definition");
    } else return cx.fail(.request, "custom-tool format {s} is not supported: use text or grammar", .{pyrepr.repr(cx.a, kindOf(format)) catch return error.OutOfMemory});
    try out.put(cx.a, "format", format);
}

pub fn tools(cx: *Cx, raw: ?Value) errors.Refused!?[]Value {
    const value = raw orelse return null;
    if (value == .null) return null;
    if (value != .array) return cx.refuse("tools must be a list");
    var out: std.ArrayList(Value) = .empty;
    for (value.array, 0..) |tool, i| {
        if (tool != .object) return cx.fail(.request, "tools[{d}] must be an object", .{i});
        const tool_name = tool.strField("name") orelse "";
        if (tool_name.len == 0) return cx.refuse("a function or custom tool needs a name");
        if (tool.get("defer_loading")) |d| if (d != .bool) return cx.refuse("defer_loading must be a boolean");
        if (std.mem.eql(u8, kind(tool), "function")) {
            const function = try json.newObject(cx.a);
            for ([_][]const u8{ "name", "description", "parameters", "strict" }) |k| if (tool.field(k)) |v| try function.put(cx.a, k, v);
            const t = try json.newObject(cx.a);
            try t.put(cx.a, "type", .{ .string = "function" });
            try t.put(cx.a, "function", .{ .object = function });
            if (json.truthyField(tool, "defer_loading")) try t.put(cx.a, "defer_loading", .{ .bool = true });
            try out.append(cx.a, .{ .object = t });
        } else if (std.mem.eql(u8, kind(tool), "custom")) {
            const t = try json.newObject(cx.a);
            try t.put(cx.a, "type", .{ .string = "custom" });
            for ([_][]const u8{ "name", "description" }) |k| if (tool.field(k)) |v| try t.put(cx.a, k, v);
            try customFormat(cx, tool, t);
            if (json.truthyField(tool, "defer_loading")) try t.put(cx.a, "defer_loading", .{ .bool = true });
            try out.append(cx.a, .{ .object = t });
        } else return cx.fail(.request, "tools of type {s} are not supported: this server runs function and custom tools", .{pyrepr.repr(cx.a, kindOf(tool)) catch return error.OutOfMemory});
    }
    return out.items;
}

fn appendDistinct(cx: *Cx, kept: *std.ArrayList(Value), tool: Value) errors.Refused!void {
    for (kept.items) |existing| if (std.mem.eql(u8, kind(existing), kind(tool)) and std.mem.eql(u8, name(existing), name(tool))) return;
    try kept.append(cx.a, tool);
}

fn referenced(offered: []const Value, reference: Value) ?Value {
    const reference_name = reference.strField("name") orelse return null;
    for (offered) |tool| if (std.mem.eql(u8, kind(tool), kind(reference)) and std.mem.eql(u8, name(tool), reference_name)) return tool;
    return null;
}

fn withoutDeferred(cx: *Cx, offered: ?[]Value) errors.Refused!?[]Value {
    const all = offered orelse return null;
    var kept: std.ArrayList(Value) = .empty;
    for (all) |tool| if (!deferred(tool)) try kept.append(cx.a, tool);
    return kept.items;
}

fn unknown(cx: *Cx, reference: Value) errors.Refused!void {
    return cx.fail(.request, "tool_choice references unknown {s} tool '{s}'", .{ kind(reference), reference.strField("name") orelse "" });
}

pub fn toolChoice(cx: *Cx, raw: ?Value, offered: ?[]Value) errors.Refused!Choice {
    if (raw == null or raw.? == .null) return .{ .choice = null, .offered = try withoutDeferred(cx, offered) };
    const choice = raw.?;
    if (choice == .string) for ([_][]const u8{ "none", "auto", "required" }) |s| if (std.mem.eql(u8, s, choice.string)) return .{ .choice = choice, .offered = try withoutDeferred(cx, offered) };
    const k = kind(choice);
    if ((std.mem.eql(u8, k, "function") or std.mem.eql(u8, k, "custom")) and choice.strField("name") != null) {
        const choice_name = choice.strField("name").?;
        const selected = referenced(offered orelse &.{}, choice);
        if (selected == null) {
            try unknown(cx, choice);
            unreachable;
        }
        const active = try cx.a.alloc(Value, 1);
        active[0] = selected.?;
        if (std.mem.eql(u8, k, "custom")) return .{ .choice = choice, .offered = active };
        const function = try json.newObject(cx.a);
        try function.put(cx.a, "name", .{ .string = choice_name });
        const c = try json.newObject(cx.a);
        try c.put(cx.a, "type", .{ .string = "function" });
        try c.put(cx.a, "function", .{ .object = function });
        return .{ .choice = .{ .object = c }, .offered = active };
    }
    const mode: Value = choice.get("mode") orelse .{ .string = "auto" };
    const mode_ok = mode == .string and (std.mem.eql(u8, mode.string, "auto") or std.mem.eql(u8, mode.string, "required"));
    if (std.mem.eql(u8, k, "allowed_tools") and mode_ok) {
        const allowed = choice.get("tools") orelse return cx.refuse("allowed_tools needs a list of function or custom tool references");
        if (allowed != .array) return cx.refuse("allowed_tools may name function or custom tools only");
        var kept: std.ArrayList(Value) = .empty;
        for (allowed.array) |reference| {
            if (reference != .object or (!std.mem.eql(u8, kind(reference), "function") and !std.mem.eql(u8, kind(reference), "custom"))) return cx.refuse("allowed_tools may name function or custom tools only");
            const selected = referenced(offered orelse &.{}, reference);
            if (selected == null) {
                try unknown(cx, reference);
                unreachable;
            }
            try appendDistinct(cx, &kept, selected.?);
        }
        return .{ .choice = mode, .offered = kept.items };
    }
    return cx.refuse("tool_choice must be none, auto, required, a named tool or allowed_tools");
}
