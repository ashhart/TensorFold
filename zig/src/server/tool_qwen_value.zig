//! Qwen XML framing is ambiguous when an exact string itself starts or ends with a line break.
const std = @import("std");
const json = @import("json");
const params = @import("tool_params.zig");
const kinds = @import("tool_kinds.zig");
const A = std.mem.Allocator;
const V = json.Value;

pub fn leading(text: []const u8) []const u8 {
    if (std.mem.startsWith(u8, text, "\r\n")) return text[2..];
    return if (std.mem.startsWith(u8, text, "\n")) text[1..] else text;
}

pub fn trailing(text: []const u8) []const u8 {
    if (std.mem.endsWith(u8, text, "\r\n")) return text[0 .. text.len - 2];
    return if (std.mem.endsWith(u8, text, "\n")) text[0 .. text.len - 1] else text;
}

const Match = struct {
    a: A,
    root: V,
    left: usize = 256,

    fn both(x: ?bool, y: ?bool) ?bool {
        if (x == false or y == false) return false;
        return if (x == true or y == true) true else null;
    }

    fn check(m: *Match, schema: V, text: []const u8, depth: usize) A.Error!?bool {
        if (schema != .object or depth == 32 or m.left == 0) return null;
        m.left -= 1;
        var matched: ?bool = null;
        if (schema.get("const")) |v| matched = v == .string and std.mem.eql(u8, v.string, text);
        if (schema.get("enum")) |vs| if (vs == .array) {
            var found = false;
            for (vs.array) |v| if (v == .string and std.mem.eql(u8, v.string, text)) {
                found = true;
                break;
            };
            matched = both(matched, found);
        };
        if (schema.get("$ref")) |ref| if (ref == .string) {
            if (try kinds.resolvePointer(m.a, m.root, ref.string)) |target|
                matched = both(matched, try m.check(target, text, depth + 1));
        };
        if (schema.get("allOf")) |branches| if (branches == .array) {
            for (branches.array) |branch| matched = both(matched, try m.check(branch, text, depth + 1));
        };
        for ([_][]const u8{ "anyOf", "oneOf" }) |name| if (schema.get(name)) |branches| if (branches == .array) {
            var accepted: usize = 0;
            var unknown = false;
            for (branches.array) |branch| {
                const result = try m.check(branch, text, depth + 1);
                unknown = unknown or result == null;
                accepted += @intFromBool(result == true);
            }
            const result: ?bool = if (unknown) null else if (std.mem.eql(u8, name, "oneOf")) accepted == 1 else accepted > 0;
            matched = both(matched, result);
        };
        return matched;
    }
};

fn matches(a: A, schema: params.Schema, text: []const u8) A.Error!?bool {
    var m: Match = .{ .a = a, .root = schema.root };
    return m.check(schema.value, text, 0);
}

/// Only a literal constraint can distinguish a data newline from an optional XML framing newline.
pub fn constrained(a: A, schema: params.Schema) A.Error!bool {
    const accepted = (try kinds.schemaKinds(a, schema.value, schema.root)) orelse return false;
    if (!accepted.contains(.string)) return false;
    return try matches(a, schema, "") != null;
}

pub fn decode(a: A, raw: []const u8, schema: params.Schema) A.Error!V {
    const first = leading(raw);
    const legacy = trailing(first);
    var selected = legacy;
    for ([_][]const u8{ legacy, first, trailing(raw), raw }) |candidate| {
        if (try matches(a, schema, candidate) == true) {
            selected = candidate;
            break;
        }
    }
    return params.decode(a, selected, schema, true);
}
