//! Offered tools and tool_choice as the template should see them (``tools.active_tool_specs``).
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Python's ``str()`` of a decoded JSON scalar (containers approximate with JSON).
pub fn pyStr(a: Allocator, v: Value) Allocator.Error![]const u8 {
    return switch (v) {
        .string => |s| s,
        .int => |t| t,
        .bool => |b| if (b) "True" else "False",
        .null => "None",
        .float => |f| blk: {
            var buf: [40]u8 = undefined;
            break :blk try a.dupe(u8, json.floatRepr(&buf, f));
        },
        else => try json.stringify(a, v, .{}),
    };
}

/// ``str(value or "").strip()``.
fn nameText(a: Allocator, v: ?Value) Allocator.Error![]const u8 {
    const value = v orelse return "";
    if (!value.truthy()) return "";
    return std.mem.trim(u8, try pyStr(a, value), " \t\r\n\x0b\x0c");
}

pub fn toolName(a: Allocator, tool: Value) Allocator.Error![]const u8 {
    if (tool != .object) return "";
    if (tool.get("function")) |function| if (function == .object) return nameText(a, function.get("name"));
    return nameText(a, tool.get("name"));
}

fn lowerEq(text: []const u8, want: []const u8) bool {
    return std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " \t\r\n\x0b\x0c"), want);
}

pub fn choiceDisables(choice: ?Value) bool {
    const c = choice orelse return false;
    if (c == .string) return lowerEq(c.string, "none");
    if (c == .object) {
        const value = c.firstTruthy(&.{ "type", "mode" }) orelse return false;
        return value == .string and lowerEq(value.string, "none");
    }
    return false;
}

/// OpenAI's "required" or a named function: the reply must call a tool.
pub fn choiceRequiresCall(a: Allocator, choice: ?Value) Allocator.Error!bool {
    const c = choice orelse return false;
    if (c == .string) return lowerEq(c.string, "required");
    if (c == .object) {
        const value = if (c.firstTruthy(&.{ "type", "mode" })) |v| try pyStr(a, v) else "";
        if (lowerEq(value, "required")) return true;
        const function = c.get("function");
        return lowerEq(value, "function") and function != null and function.? == .object;
    }
    return false;
}

fn isFunctionChoice(a: Allocator, c: Value) Allocator.Error!bool {
    if (c != .object) return false;
    const t = c.get("type") orelse return false;
    if (!t.truthy()) return false;
    return std.ascii.eqlIgnoreCase(try pyStr(a, t), "function");
}

/// The tools the template offers: none for "none", only the named one for a named function.
pub fn active(cx: *Cx, tools: ?Value, choice: ?Value) errors.Refused![]Value {
    var specs: []Value = &.{};
    if (tools) |t| if (t != .null) {
        if (t != .array) return cx.other("tools must be a list");
        for (t.array, 0..) |tool, i| {
            if (tool != .object) return cx.fail(.other, "tools[{d}] must be an object", .{i});
            if ((try toolName(cx.a, tool)).len == 0) return cx.fail(.other, "tools[{d}] must include a function name", .{i});
        }
        specs = t.array;
    };
    if (choiceDisables(choice)) return &.{};
    if (specs.len == 0) {
        if (try choiceRequiresCall(cx.a, choice)) return cx.other("tool_choice requires a tool call, but the request offers no tools");
        return &.{};
    }
    const c = choice orelse return specs;
    if (!try isFunctionChoice(cx.a, c)) return specs;
    const function = c.get("function") orelse return cx.other("tool_choice function must include a function object");
    if (function != .object) return cx.other("tool_choice function must include a function object");
    const requested = try nameText(cx.a, function.get("name"));
    if (requested.len == 0) return cx.other("tool_choice function must include a name");
    var named: std.ArrayList(Value) = .empty;
    for (specs) |tool| if (std.mem.eql(u8, try toolName(cx.a, tool), requested)) try named.append(cx.a, tool);
    if (named.items.len == 0) return cx.fail(.other, "tool_choice requested unknown tool '{s}'", .{requested});
    return named.items;
}
