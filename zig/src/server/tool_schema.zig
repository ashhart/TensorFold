//! Completed LFM2 arguments checked against the schema constraints its grammar cannot express.
const std = @import("std");
const json = @import("json");
const tool_kinds = @import("tool_kinds.zig");
const tool_params = @import("tool_params.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Keywords this small post-validator deliberately refuses instead of pretending to enforce.
const unsupported = [_][]const u8{
    "pattern",           "format",        "multipleOf",            "uniqueItems",      "contains",     "minContains", "maxContains",
    "patternProperties", "propertyNames", "dependentRequired",     "dependentSchemas", "dependencies", "if",          "then",
    "else",              "not",           "unevaluatedProperties", "unevaluatedItems", "prefixItems",
};

/// Keys whose value maps names (a property's, a definition's) to subschemas: the names are never keywords.
const named = [_][]const u8{ "properties", "$defs", "definitions", "patternProperties" };
/// Keys whose value is data, not a schema: nothing inside them is a keyword.
const data = [_][]const u8{ "enum", "const", "default", "examples" };

fn listed(key: []const u8, names: []const []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, key, name)) return true;
    return false;
}

/// Whether every constraint in this schema has a post-validation implementation, judged at keyword positions only.
pub fn supported(schema: Value) bool {
    if (schema == .array) {
        for (schema.array) |item| if (!supported(item)) return false;
        return true;
    }
    if (schema != .object) return true;
    for (schema.object.keys(), schema.object.values()) |key, value| {
        if (listed(key, &unsupported)) return false;
        if (listed(key, &data)) continue;
        if (listed(key, &named)) {
            if (value == .object) for (value.object.values()) |sub| if (!supported(sub)) return false;
            continue;
        }
        if (!supported(value)) return false;
    }
    return true;
}

/// Whether `value` satisfies `schema`, including required children and local references.
pub fn valid(a: Allocator, value: Value, schema: Value) Allocator.Error!bool {
    if (!supported(schema) or !tool_params.finiteDecoded(value)) return false;
    return node(a, value, schema, schema, 0);
}

fn node(a: Allocator, value: Value, schema: Value, root: Value, depth: usize) Allocator.Error!bool {
    if (depth > 64 or schema != .object) return depth <= 64;
    if (schema.get("$ref") != null) {
        const target = (try tool_kinds.resolved(a, schema, root)) orelse return false;
        var own = try json.copyObject(a, schema.object);
        _ = own.orderedRemove("$ref");
        if (!try node(a, value, target, root, depth + 1)) return false;
        return node(a, value, .{ .object = own }, root, depth + 1);
    }
    if (try tool_kinds.schemaKinds(a, schema, root)) |kinds| {
        const scalar_fallback = depth == 1 and value != .object and value != .array and !kinds.contains(.object) and !kinds.contains(.array);
        if (!scalar_fallback and !accepts(kinds, value)) return false;
    }
    if (schema.get("const")) |fixed| if (!equal(value, fixed)) return false;
    if (schema.get("enum")) |choices| {
        if (choices != .array) return false;
        var any = false;
        for (choices.array) |choice| {
            any = any or equal(value, choice);
        }
        if (!any) return false;
    }
    for ([_][]const u8{ "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" }) |key| if (schema.get(key)) |limit| {
        const n = number(value) orelse return false;
        const bound = number(limit) orelse return false;
        const below = std.mem.startsWith(u8, key, "minimum") or std.mem.startsWith(u8, key, "exclusiveMinimum");
        const exclusive = std.mem.startsWith(u8, key, "exclusive");
        if (below and (n < bound or (exclusive and n == bound))) return false;
        if (!below and (n > bound or (exclusive and n == bound))) return false;
    };
    if (value == .string) {
        const len = std.unicode.utf8CountCodepoints(value.string) catch return false;
        if (schema.get("minLength")) |limit| if (try integer(limit)) |n| if (len < n) return false;
        if (schema.get("maxLength")) |limit| if (try integer(limit)) |n| if (len > n) return false;
    }
    if (value == .array) {
        if (schema.get("minItems")) |limit| if (try integer(limit)) |n| if (value.array.len < n) return false;
        if (schema.get("maxItems")) |limit| if (try integer(limit)) |n| if (value.array.len > n) return false;
        if (schema.get("items")) |items| for (value.array) |item| if (!try node(a, item, items, root, depth + 1)) return false;
    }
    if (value == .object) {
        const properties = schema.get("properties");
        if (schema.get("minProperties")) |limit| if (try integer(limit)) |n| if (value.object.count() < n) return false;
        if (schema.get("maxProperties")) |limit| if (try integer(limit)) |n| if (value.object.count() > n) return false;
        if (schema.get("required")) |required| {
            if (required != .array) return false;
            for (required.array) |key| if (key != .string or value.object.get(key.string) == null) return false;
        }
        if (properties != null and properties.? == .object) for (properties.?.object.keys(), properties.?.object.values()) |key, child| {
            if (value.object.get(key)) |item| if (!try node(a, item, child, root, depth + 1)) return false;
        };
        if (schema.get("additionalProperties")) |additional| for (value.object.keys(), value.object.values()) |key, item| {
            if (properties != null and properties.? == .object and properties.?.object.get(key) != null) continue;
            if (additional == .bool and !additional.bool) return false;
            if (additional == .object and !try node(a, item, additional, root, depth + 1)) return false;
        };
    }
    if (schema.get("anyOf")) |branches| {
        var any = false;
        if (branches == .array) for (branches.array) |branch| {
            any = any or try node(a, value, branch, root, depth + 1);
        };
        if (!any) return false;
    }
    if (schema.get("oneOf")) |branches| {
        if (branches != .array) return false;
        var matches: usize = 0;
        for (branches.array) |branch| {
            matches += @intFromBool(try node(a, value, branch, root, depth + 1));
        }
        if (matches != 1) return false;
    }
    if (schema.get("allOf")) |branches| {
        if (branches != .array) return false;
        for (branches.array) |branch| if (!try node(a, value, branch, root, depth + 1)) return false;
    }
    return true;
}

fn accepts(kinds: tool_kinds.Kinds, value: Value) bool {
    return switch (value) {
        .array => kinds.contains(.array),
        .object => kinds.contains(.object),
        .bool => kinds.contains(.boolean),
        .int => kinds.contains(.integer),
        .float => |f| kinds.contains(.number) or (kinds.contains(.integer) and @trunc(f) == f),
        .null => kinds.contains(.null),
        .string => kinds.contains(.string),
    };
}

fn integer(value: Value) Allocator.Error!?usize {
    if (value != .int) return null;
    const n = value.int64() orelse return null;
    return if (n >= 0) @intCast(n) else null;
}

fn number(value: Value) ?f64 {
    return switch (value) {
        .int => |text| std.fmt.parseFloat(f64, text) catch null,
        .float => |f| if (std.math.isFinite(f)) f else null,
        else => null,
    };
}

fn equal(left: Value, right: Value) bool {
    if (std.meta.activeTag(left) != std.meta.activeTag(right)) return false;
    return switch (left) {
        .null => true,
        .bool => |b| b == right.bool,
        .int => |text| std.mem.eql(u8, text, right.int),
        .float => |f| f == right.float,
        .string => |s| std.mem.eql(u8, s, right.string),
        .array => |items| array_equal: {
            if (items.len != right.array.len) break :array_equal false;
            for (items, right.array) |a, b| {
                if (!equal(a, b)) break :array_equal false;
            }
            break :array_equal true;
        },
        .object => |obj| object_equal: {
            if (obj.count() != right.object.count()) break :object_equal false;
            for (obj.keys(), obj.values()) |key, item| {
                const other = right.object.get(key) orelse break :object_equal false;
                if (!equal(item, other)) break :object_equal false;
            }
            break :object_equal true;
        },
    };
}
