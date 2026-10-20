//! The issue's tool schemas for a parameter declared as a union, and the calls that carry them.
const std = @import("std");
const json = @import("json");
const V = json.Value;
const A = std.mem.Allocator;

/// The shell tool's request parameter as the issue declares it: one object.
pub const object_schema = "{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"run\"]},\"command\":{\"type\":\"string\"}},\"required\":[\"action\",\"command\"]}";

/// The issue's second schema: a oneOf of two objects.
pub const one_of_schema = "{\"oneOf\":[{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"run\"]},\"command\":{\"type\":\"string\"}},\"required\":[\"action\",\"command\"]},{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"poll\"]},\"handle\":{\"type\":\"string\"}},\"required\":[\"action\",\"handle\"]}]}";

/// The issue's two objects under anyOf.
pub const any_of_schema = "{\"anyOf\":[{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"run\"]},\"command\":{\"type\":\"string\"}},\"required\":[\"action\",\"command\"]},{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"poll\"]},\"handle\":{\"type\":\"string\"}},\"required\":[\"action\",\"handle\"]}]}";

/// A union of a string and an object.
pub const string_or_object_schema = "{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\"}}}]}";

/// A list of types, the other spelling of a union.
pub const type_list_schema = "{\"type\":[\"object\",\"null\"]}";

/// A union whose branch is itself a union.
pub const nested_schema = "{\"oneOf\":[{\"anyOf\":[{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\"}}}]},{\"type\":\"object\",\"properties\":{\"handle\":{\"type\":\"string\"}}}]}";

/// A plain string parameter.
pub const string_schema = "{\"type\":\"string\"}";

/// A union whose branches are all strings.
pub const string_union_schema = "{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"string\"}]}";

/// One schema, one value the model wrote, and the arguments the client must see.
pub const Case = struct {
    schema: []const u8,
    value: []const u8,
    expected: []const u8,
};

pub const cases = [_]Case{
    .{ .schema = object_schema, .value = "{\"action\": \"run\", \"command\": \"cat probe.txt\"}", .expected = "{\"request\":{\"action\":\"run\",\"command\":\"cat probe.txt\"}}" },
    .{ .schema = one_of_schema, .value = "{\"action\": \"run\", \"command\": \"cat probe.txt\"}", .expected = "{\"request\":{\"action\":\"run\",\"command\":\"cat probe.txt\"}}" },
    .{ .schema = any_of_schema, .value = "{\"action\": \"run\", \"command\": \"cat probe.txt\"}", .expected = "{\"request\":{\"action\":\"run\",\"command\":\"cat probe.txt\"}}" },
    .{ .schema = string_or_object_schema, .value = "{\"action\": \"run\"}", .expected = "{\"request\":{\"action\":\"run\"}}" },
    .{ .schema = string_or_object_schema, .value = "cat probe.txt", .expected = "{\"request\":\"cat probe.txt\"}" },
    .{ .schema = string_or_object_schema, .value = "123", .expected = "{\"request\":\"123\"}" },
    .{ .schema = type_list_schema, .value = "{\"action\": \"run\"}", .expected = "{\"request\":{\"action\":\"run\"}}" },
    .{ .schema = type_list_schema, .value = "null", .expected = "{\"request\":null}" },
    .{ .schema = nested_schema, .value = "{\"action\": \"run\"}", .expected = "{\"request\":{\"action\":\"run\"}}" },
    .{ .schema = string_schema, .value = "{\"action\": \"run\"}", .expected = "{\"request\":\"{\\\"action\\\": \\\"run\\\"}\"}" },
    .{ .schema = string_union_schema, .value = "{\"action\": \"run\"}", .expected = "{\"request\":\"{\\\"action\\\": \\\"run\\\"}\"}" },
};

/// The shell tool the issue offers, with its request parameter declared by `schema`.
pub fn tools(a: A, schema: []const u8) A.Error![]V {
    const text = try std.mem.concat(a, u8, &.{ "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"Run a shell command\",\"parameters\":{\"type\":\"object\",\"required\":[\"request\"],\"properties\":{\"request\":", schema, "}}}}]" });
    return (try json.parse(a, text)).ok.array;
}

/// The reply that writes `value` into the request parameter.
pub fn reply(a: A, value: []const u8) A.Error![]const u8 {
    return std.fmt.allocPrint(a, "<tool_call>\n<function=shell>\n<parameter=request>\n{s}\n</parameter>\n</function>\n</tool_call>", .{value});
}
