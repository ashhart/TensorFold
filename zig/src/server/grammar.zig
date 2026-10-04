//! Structured-output fields checked as ``engine.grammar.request_spec`` checks them; the engine enforces them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const fields = [_][]const u8{ "response_format", "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs" };

pub const Kind = @FieldType(api.Structure, "kind");
pub const Spec = struct { kind: Kind, text: []const u8 = "", field: []const u8 = "response_format" };

/// The JSON decoder's own message, without its position.
fn bareMessage(full: []const u8) []const u8 {
    const at = std.mem.lastIndexOf(u8, full, ": line ") orelse return full;
    return full[0..at];
}

fn schemaText(cx: *Cx, raw: Value, where: []const u8) errors.Refused![]const u8 {
    var value = raw;
    if (raw == .string) value = switch (try json.parseText(cx.a, raw.string)) {
        .ok => |v| v,
        .err => |e| return cx.fail(.request, "{s} is not valid JSON: {s}", .{ where, bareMessage(e) }),
    };
    if (value == .bool) value = if (value.bool) .{ .object = try json.newObject(cx.a) } else .null;
    if (value != .object) return cx.fail(.request, "{s} must be a JSON schema object", .{where});
    return json.stringify(cx.a, value, .{ .ascii = false });
}

fn text(cx: *Cx, v: Value, where: []const u8) errors.Refused![]const u8 {
    if (v != .string or v.string.len == 0) return cx.fail(.request, "{s} must be a non-empty string", .{where});
    return v.string;
}

fn choices(cx: *Cx, v: Value, where: []const u8) errors.Refused![]const u8 {
    const ok = v == .array and v.array.len > 0 and for (v.array) |c| {
        if (c != .string or c.string.len == 0) break false;
    } else true;
    if (!ok) return cx.fail(.request, "{s} must be a non-empty list of non-empty strings", .{where});
    return json.stringify(cx.a, v, .{ .ascii = false });
}

fn read(cx: *Cx, kind: Kind, v: Value, where: []const u8) errors.Refused![]const u8 {
    return switch (kind) {
        .json_schema => schemaText(cx, v, where),
        .choice => choices(cx, v, where),
        else => text(cx, v, where),
    };
}

/// The body's structured-output request, null for plain text; a refusal when it is malformed.
pub fn spec(cx: *Cx, body: Value) errors.Refused!?Spec {
    if (body.field("response_format")) |rf| {
        if (rf != .object) return cx.refuse("response_format must be an object such as {\"type\": \"json_object\"}");
        const kind = rf.get("type");
        const name = if (kind != null and kind.? == .string) kind.?.string else "";
        if (kind != null and kind.? == .string and std.mem.eql(u8, name, "json_object")) return .{ .kind = .json };
        if (kind != null and kind.? == .string and std.mem.eql(u8, name, "json_schema")) {
            const js = rf.get("json_schema");
            if (js == null or js.? != .object or js.?.field("schema") == null) return cx.refuse("response_format json_schema needs json_schema.schema (a JSON schema object)");
            return .{ .kind = .json_schema, .text = try schemaText(cx, js.?.get("schema").?, "response_format json_schema.schema") };
        }
        if (!(kind != null and kind.? == .string and std.mem.eql(u8, name, "text")))
            return cx.fail(.request, "response_format type must be text, json_object or json_schema, not {s}", .{try pyrepr.repr(cx.a, kind)});
    }
    const guided = [_]struct { []const u8, Kind }{ .{ "guided_json", .json_schema }, .{ "guided_regex", .regex }, .{ "guided_choice", .choice }, .{ "guided_grammar", .grammar } };
    for (guided) |g| if (body.field(g[0])) |v| return .{ .kind = g[1], .text = try read(cx, g[1], v, g[0]), .field = g[0] };
    const so = body.field("structured_outputs") orelse return null;
    if (so != .object) return cx.refuse("structured_outputs must be an object such as {\"json\": {...}}");
    var given: std.ArrayList([]const u8) = .empty;
    for (so.object.keys(), so.object.values()) |k, v| if (v != .null and !(v == .bool and !v.bool)) try given.append(cx.a, k);
    std.mem.sort([]const u8, given.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    const known = [_]struct { []const u8, Kind }{ .{ "json", .json_schema }, .{ "regex", .regex }, .{ "choice", .choice }, .{ "grammar", .grammar } };
    var other: std.ArrayList([]const u8) = .empty;
    for (given.items) |k| {
        const is_known = for (known) |kn| {
            if (std.mem.eql(u8, kn[0], k)) break true;
        } else std.mem.eql(u8, k, "json_object");
        if (!is_known) try other.append(cx.a, k);
    }
    if (other.items.len > 0) return cx.fail(.request, "structured_outputs {s} is not supported: use json, json_object, regex, choice or grammar", .{try std.mem.join(cx.a, ", ", other.items)});
    for (given.items) |k| for (known) |kn| if (std.mem.eql(u8, kn[0], k))
        return .{ .kind = kn[1], .text = try read(cx, kn[1], so.get(k).?, try std.fmt.allocPrint(cx.a, "structured_outputs.{s}", .{k})), .field = "structured_outputs" };
    if (json.truthyField(so, "json_object")) return .{ .kind = .json, .field = "structured_outputs" };
    return null;
}

/// ``grammar.refusal``: a malformed grammar, or one beside a required call, refused before any header.
pub fn refusal(cx: *Cx, body: Value) errors.Refused!void {
    const s = try spec(cx, body) orelse return;
    const tools = body.get("tools");
    if (tools != null and tools.?.truthy() and try tool_specs.choiceRequiresCall(cx.a, body.get("tool_choice")))
        return cx.fail(.request, "{s} cannot be combined with tool_choice \"required\" or a named function: send one", .{s.field});
}
