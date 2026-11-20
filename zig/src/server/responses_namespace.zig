//! Responses namespace tools: one level of groups, flattened to <namespace>__<tool> for the chat tools that run them.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const spells = @import("responses_translate.zig").spells;
const Choice = @import("responses_tools.zig").Choice;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

fn kindOf(v: Value) ?Value {
    return if (v == .object) v.get("type") else null;
}

/// A tool that asks to be loaded later: this server runs every tool it is given now.
fn deferred(cx: *Cx, tool: Value) errors.Refused {
    return cx.fail(.request, "tool {s} sets defer_loading, which is not supported: send the tool without it", .{try pyrepr.repr(cx.a, tool.get("name"))});
}

/// Whether the declared tools hold a namespace: what makes the flat path pay for the namespace-only checks.
pub fn hasNamespace(declared: ?Value) bool {
    if (declared == null or declared.? != .array) return false;
    for (declared.?.array) |tool| if (tool.typeIs("namespace")) return true;
    return false;
}

/// Whether ``name`` is one of the namespace groups these tools declare.
fn offeredGroup(declared: ?Value, name: []const u8) bool {
    if (declared == null or declared.? != .array) return false;
    for (declared.?.array) |tool| {
        if (!tool.typeIs("namespace")) continue;
        const group = tool.strField("name") orelse continue;
        if (std.mem.eql(u8, group, name)) return true;
    }
    return false;
}

/// A raw function_call item's namespace as the group it came from: only a declared group's name answers for it.
pub fn inputNamespaces(cx: *Cx, items: []const Value, declared: ?Value) errors.Refused!void {
    for (items) |item| {
        if (item != .object or !item.typeIs("function_call")) continue;
        const ns = item.get("namespace") orelse continue; // absent or null keeps the name as it was sent
        if (ns != .string) continue; // messages() gives that item its own refusal
        if (!offeredGroup(declared, ns.string)) return cx.fail(.request, "namespace {s} is not among offered: send the name of a namespace tool", .{try pyrepr.repr(cx.a, ns)});
    }
}

/// A flattened tool's name, refused by name when another tool carries it already.
fn uniqueName(cx: *Cx, names: *std.ArrayList([]const u8), name: []const u8) errors.Refused!void {
    for (names.items) |taken| if (std.mem.eql(u8, taken, name)) return cx.fail(.request, "two tools are named {s}: each tool needs its own name", .{try pyrepr.repr(cx.a, Value{ .string = name })});
    try names.append(cx.a, name);
}

/// One function tool as the chat function that runs it; a namespace child is flattened as <ns>__<tool> with both notes.
fn functionTool(cx: *Cx, names: ?*std.ArrayList([]const u8), out: *std.ArrayList(Value), tool: Value, ns_name: []const u8, ns_desc: []const u8) errors.Refused!void {
    if (names != null and json.truthyField(tool, "defer_loading")) return deferred(cx, tool);
    const name = tool.strField("name") orelse "";
    if (name.len == 0) return cx.refuse("a function tool needs a name");
    const function = try json.newObject(cx.a);
    for ([_][]const u8{ "name", "description", "parameters", "strict" }) |k| if (tool.field(k)) |v| try function.put(cx.a, k, v);
    var flat = name;
    if (ns_name.len > 0) {
        const prefix = try std.fmt.allocPrint(cx.a, "{s}__", .{ns_name});
        if (std.mem.startsWith(u8, name, prefix)) return cx.refuse("a namespace child's name must not start with its namespace and '__'");
        if (tool.get("description")) |d| if (d != .null and d != .string) return cx.refuse("a namespace tool's child needs a string description");
        flat = try std.fmt.allocPrint(cx.a, "{s}__{s}", .{ ns_name, name });
        const own = tool.strField("description") orelse "";
        try function.put(cx.a, "name", .{ .string = flat });
        try function.put(cx.a, "description", .{ .string = if (own.len == 0) ns_desc else try std.fmt.allocPrint(cx.a, "{s}: {s}", .{ ns_desc, own }) });
    }
    if (names) |taken| try uniqueName(cx, taken, flat);
    const t = try json.newObject(cx.a);
    try t.put(cx.a, "type", .{ .string = "function" });
    try t.put(cx.a, "function", .{ .object = function });
    try out.append(cx.a, .{ .object = t });
}

/// A namespace tool's children as flat function tools: OpenAI's one level of namespacing, and only one.
fn namespaceTools(cx: *Cx, names: *std.ArrayList([]const u8), out: *std.ArrayList(Value), ns: Value) errors.Refused!void {
    if (json.truthyField(ns, "defer_loading")) return deferred(cx, ns);
    const ns_name = ns.strField("name") orelse "";
    if (ns_name.len == 0) return cx.refuse("a namespace tool needs a name");
    const ns_desc = ns.strField("description") orelse "";
    if (ns_desc.len == 0) return cx.refuse("a namespace tool needs a description");
    const children = ns.get("tools") orelse return cx.refuse("a namespace tool needs a list of function tools");
    if (children != .array) return cx.refuse("a namespace tool's tools must be a list");
    if (children.array.len == 0) return cx.refuse("a namespace tool needs at least one function tool");
    for (children.array) |child| {
        const kind = kindOf(child);
        if (kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "namespace")) return cx.refuse("a namespace tool cannot hold another namespace");
        if (kind == null or kind.? != .string or !std.mem.eql(u8, kind.?.string, "function")) return cx.fail(.request, "tools of type {s} are not supported: this server runs function tools only", .{try pyrepr.repr(cx.a, kind)});
        try functionTool(cx, names, out, child, ns_name, ns_desc);
    }
}

pub fn tools(cx: *Cx, raw: ?Value) errors.Refused!?[]Value {
    const value = raw orelse return null;
    if (value == .null) return null;
    if (value != .array) return cx.refuse("tools must be a list");
    var out: std.ArrayList(Value) = .empty;
    var names: std.ArrayList([]const u8) = .empty; // every name the tools end up with, so none is given twice
    const taken = if (hasNamespace(value)) &names else null; // flat-only lists keep main's unchecked names
    for (value.array) |tool| {
        if (tool.typeIs("namespace")) {
            try namespaceTools(cx, &names, &out, tool);
            continue;
        }
        if (!tool.typeIs("function")) return cx.fail(.request, "tools of type {s} are not supported: this server runs function tools only", .{try pyrepr.repr(cx.a, kindOf(tool))});
        try functionTool(cx, taken, &out, tool, "", "");
    }
    return out.items;
}

/// The name a tool_choice selector picks among the declared tools: null for none, "" for an ambiguous one.
fn selectorName(a: Allocator, declared: ?Value, selector: []const u8, ns: ?[]const u8) errors.Refused!?[]const u8 {
    if (declared == null or declared.? != .array) return null;
    for (declared.?.array) |tool| { // a composed engine spelling is unique, so it answers before any own-name match
        if (!tool.typeIs("namespace")) continue;
        const group = tool.strField("name") orelse continue;
        if (ns) |want| if (!std.mem.eql(u8, want, group)) continue;
        const children = tool.get("tools") orelse continue;
        if (children != .array) continue;
        for (children.array) |child| {
            if (!child.typeIs("function")) continue;
            const own = child.strField("name") orelse continue;
            if (!spells(selector, group, own)) continue;
            return try std.fmt.allocPrint(a, "{s}__{s}", .{ group, own });
        }
    }
    var picked: ?[]const u8 = null;
    var hits: usize = 0;
    for (declared.?.array) |tool| {
        if (tool.typeIs("function")) {
            if (ns != null) continue; // a top-level tool has no namespace, so an explicit one never picks it
            const own = tool.strField("name") orelse continue;
            if (std.mem.eql(u8, own, selector)) return own; // the tool's own name answers for itself
            continue;
        }
        if (!tool.typeIs("namespace")) continue;
        const group = tool.strField("name") orelse continue;
        if (ns) |want| if (!std.mem.eql(u8, want, group)) continue;
        const children = tool.get("tools") orelse continue;
        if (children != .array) continue;
        for (children.array) |child| {
            if (!child.typeIs("function")) continue;
            const own = child.strField("name") orelse continue;
            if (std.mem.eql(u8, own, selector)) {
                hits += 1;
                picked = try std.fmt.allocPrint(a, "{s}__{s}", .{ group, own });
            }
        }
    }
    return if (hits == 0) null else if (hits > 1) "" else picked;
}

pub fn toolChoice(cx: *Cx, raw: ?Value, offered: ?[]Value, declared: ?Value) errors.Refused!Choice {
    const choice = raw orelse return .{ .choice = null, .offered = offered };
    if (choice == .null) return .{ .choice = null, .offered = offered };
    if (choice == .string) for ([_][]const u8{ "none", "auto", "required" }) |s| if (std.mem.eql(u8, s, choice.string)) return .{ .choice = choice, .offered = offered };
    const kind = kindOf(choice);
    const k = if (kind != null and kind.? == .string) kind.?.string else "";
    const namespaced = hasNamespace(declared); // only these requests read the namespace a selector carries
    if (std.mem.eql(u8, k, "function") and choice.strField("name") != null) {
        const given = choice.strField("name").?;
        const found = try selectorName(cx.a, declared, given, if (namespaced) choice.strField("namespace") else null);
        if (found) |name| if (name.len == 0) return cx.fail(.request, "tool_choice {s} is ambiguous: name it as <namespace>__<tool>", .{try pyrepr.repr(cx.a, choice.get("name").?)});
        // An unresolved name keeps its own spelling, which is the name tool_specs refuses it by.
        const resolved = found orelse given;
        const function = try json.newObject(cx.a);
        try function.put(cx.a, "name", .{ .string = resolved });
        const c = try json.newObject(cx.a);
        try c.put(cx.a, "type", .{ .string = "function" });
        try c.put(cx.a, "function", .{ .object = function });
        return .{ .choice = .{ .object = c }, .offered = offered };
    }
    const mode: Value = choice.get("mode") orelse .{ .string = "auto" };
    const mode_ok = mode == .string and (std.mem.eql(u8, mode.string, "auto") or std.mem.eql(u8, mode.string, "required"));
    if (std.mem.eql(u8, k, "allowed_tools") and mode_ok) {
        const allowed = choice.get("tools");
        const list: []const Value = if (allowed != null and allowed.? == .array) allowed.?.array else &.{};
        if (allowed != null and allowed.?.truthy() and allowed.? != .array) return cx.refuse("allowed_tools may name function tools only");
        for (list) |t| {
            const tk = kindOf(t);
            if (tk == null or tk.? != .string or !std.mem.eql(u8, tk.?.string, "function")) return cx.refuse("allowed_tools may name function tools only");
        }
        var resolved: std.ArrayList([]const u8) = .empty;
        for (list) |t| {
            const pick = t.strField("name") orelse if (namespaced) return cx.refuse("allowed_tools may name function tools only") else continue;
            // An entry no tool carries keeps nothing.
            const name = try selectorName(cx.a, declared, pick, if (namespaced) t.strField("namespace") else null) orelse continue;
            if (name.len == 0) return cx.fail(.request, "tool_choice {s} is ambiguous: name it as <namespace>__<tool>", .{try pyrepr.repr(cx.a, t.get("name").?)});
            try resolved.append(cx.a, name);
        }
        var kept: std.ArrayList(Value) = .empty;
        for (offered orelse &.{}) |t| {
            const name = t.get("function").?.get("name").?;
            for (resolved.items) |allow| if (name == .string and std.mem.eql(u8, allow, name.string)) {
                try kept.append(cx.a, t);
                break;
            };
        }
        return .{ .choice = mode, .offered = kept.items };
    }
    return cx.refuse("tool_choice must be none, auto, required, a function or allowed_tools of functions");
}
