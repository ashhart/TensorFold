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
    /// The members of the parameters root the schema's references point into.
    defs: []const u8 = "",
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
    return toolsWith(a, schema, "");
}

/// The shell tool with its request parameter declared by `schema`, and `defs` beside the properties.
pub fn toolsWith(a: A, schema: []const u8, defs: []const u8) A.Error![]V {
    const comma = if (defs.len == 0) "" else ",";
    const text = try std.mem.concat(a, u8, &.{ "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"Run a shell command\",\"parameters\":{\"type\":\"object\",\"required\":[\"request\"],\"properties\":{\"request\":", schema, "}", comma, defs, "}}}]" });
    return (try json.parse(a, text)).ok.array;
}

/// The reply that writes `value` into the request parameter.
pub fn reply(a: A, value: []const u8) A.Error![]const u8 {
    return std.fmt.allocPrint(a, "<tool_call>\n<function=shell>\n<parameter=request>\n{s}\n</parameter>\n</function>\n</tool_call>", .{value});
}

/// A union of a string and an integer.
pub const string_or_integer_schema = "{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}";

/// Pydantic's discriminated union of two actions, by reference.
pub const pydantic_union_schema = "{\"oneOf\":[{\"$ref\":\"#/$defs/Run\"},{\"$ref\":\"#/$defs/Poll\"}],\"discriminator\":{\"propertyName\":\"action\",\"mapping\":{\"run\":\"#/$defs/Run\",\"poll\":\"#/$defs/Poll\"}}}";

/// Pydantic's Optional[Model], by reference.
pub const pydantic_optional_schema = "{\"anyOf\":[{\"$ref\":\"#/$defs/Address\"},{\"type\":\"null\"}]}";

/// A reference into the definitions of the parameters root.
pub const definitions_schema = "{\"$ref\":\"#/definitions/Thing\"}";

/// A reference that points at another reference.
pub const chain_schema = "{\"$ref\":\"#/$defs/First\"}";

/// A reference that points back at itself.
pub const cycle_schema = "{\"$ref\":\"#/$defs/Loop\"}";

/// The definitions the cycle points at.
pub const cycle_defs = "\"$defs\":{\"Loop\":{\"$ref\":\"#/$defs/Loop\"}}";

/// A reference to a definition that is not there.
pub const missing_schema = "{\"$ref\":\"#/$defs/Missing\"}";

/// An allOf whose branches agree on an object.
pub const all_of_object_schema = "{\"allOf\":[{\"type\":\"object\"},{\"required\":[\"a\"]}]}";

/// An allOf whose branches disagree.
pub const all_of_disagreeing_schema = "{\"allOf\":[{\"type\":\"object\"},{\"type\":\"array\"}]}";

/// An enum of integers.
pub const enum_schema = "{\"enum\":[1,2]}";

/// A const object.
pub const const_schema = "{\"const\":{\"action\":\"run\"}}";

/// An object type with a oneOf of two constraints.
pub const object_with_one_of_schema = "{\"type\":\"object\",\"oneOf\":[{\"required\":[\"a\"]},{\"required\":[\"b\"]}]}";

/// A quoted string keeps the text the model wrote, whatever the schema says.
pub const quoted_cases = [_]Case{
    .{ .schema = string_schema, .value = "\"hello\"", .expected = "{\"request\":\"\\\"hello\\\"\"}" },
    .{ .schema = string_or_object_schema, .value = "\"hello\"", .expected = "{\"request\":\"\\\"hello\\\"\"}" },
    .{ .schema = string_or_integer_schema, .value = "\"hello\"", .expected = "{\"request\":\"\\\"hello\\\"\"}" },
};

/// The schemas that point at definitions of the parameters root.
pub const ref_cases = [_]Case{
    .{
        .schema = pydantic_union_schema,
        .defs = "\"$defs\":{\"Run\":{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"run\"]},\"command\":{\"type\":\"string\"}},\"required\":[\"action\",\"command\"]},\"Poll\":{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"poll\"]},\"handle\":{\"type\":\"string\"}},\"required\":[\"action\",\"handle\"]}}",
        .value = "{\"action\": \"run\", \"command\": \"cat probe.txt\"}",
        .expected = "{\"request\":{\"action\":\"run\",\"command\":\"cat probe.txt\"}}",
    },
    .{
        .schema = pydantic_optional_schema,
        .defs = "\"$defs\":{\"Address\":{\"type\":\"object\",\"properties\":{\"street\":{\"type\":\"string\"}}}}",
        .value = "{\"street\": \"Main Street\"}",
        .expected = "{\"request\":{\"street\":\"Main Street\"}}",
    },
    .{
        .schema = definitions_schema,
        .defs = "\"definitions\":{\"Thing\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}}}}",
        .value = "{\"id\": 7}",
        .expected = "{\"request\":{\"id\":7}}",
    },
    .{
        .schema = chain_schema,
        .defs = "\"$defs\":{\"First\":{\"$ref\":\"#/$defs/Second\"},\"Second\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}}}}",
        .value = "{\"id\": 7}",
        .expected = "{\"request\":{\"id\":7}}",
    },
    .{
        .schema = cycle_schema,
        .defs = cycle_defs,
        .value = "{\"id\": 7}",
        .expected = "{\"request\":\"{\\\"id\\\": 7}\"}",
    },
    .{
        .schema = missing_schema,
        .value = "{\"id\": 7}",
        .expected = "{\"request\":\"{\\\"id\\\": 7}\"}",
    },
};

/// The schemas that name their kinds through their combiners.
pub const combined_cases = [_]Case{
    .{ .schema = all_of_object_schema, .value = "{\"a\": 1}", .expected = "{\"request\":{\"a\":1}}" },
    .{ .schema = all_of_disagreeing_schema, .value = "{\"a\": 1}", .expected = "{\"request\":\"{\\\"a\\\": 1}\"}" },
    .{ .schema = enum_schema, .value = "5", .expected = "{\"request\":5}" },
    .{ .schema = const_schema, .value = "{\"action\": \"run\"}", .expected = "{\"request\":{\"action\":\"run\"}}" },
    .{ .schema = object_with_one_of_schema, .value = "{\"a\": 1}", .expected = "{\"request\":{\"a\":1}}" },
};

/// A number and an integer union.
pub const number_any_of_schema = "{\"type\":\"number\",\"anyOf\":[{\"type\":\"integer\"},{\"type\":\"null\"}]}";

/// A number and an integer intersection.
pub const number_all_of_schema = "{\"allOf\":[{\"type\":\"number\"},{\"type\":\"integer\"}]}";

/// The schemas where a number and an integer meet.
pub const number_cases = [_]Case{
    .{ .schema = number_any_of_schema, .value = "5", .expected = "{\"request\":5}" },
    .{ .schema = number_all_of_schema, .value = "5", .expected = "{\"request\":5}" },
};

/// A union of two constraints that name different kinds.
pub const one_and_any_schema = "{\"oneOf\":[{\"type\":\"object\"},{\"type\":\"string\"}],\"anyOf\":[{\"type\":\"object\"},{\"type\":\"array\"}]}";

/// A type list and an enum that disagree.
pub const type_and_enum_schema = "{\"type\":[\"object\",\"null\"],\"enum\":[null]}";

/// The schemas where every keyword constrains the value.
pub const keyword_cases = [_]Case{
    .{ .schema = one_and_any_schema, .value = "[1]", .expected = "{\"request\":\"[1]\"}" },
    .{ .schema = one_and_any_schema, .value = "{\"a\": 1}", .expected = "{\"request\":{\"a\":1}}" },
    .{ .schema = type_and_enum_schema, .value = "{\"a\": 1}", .expected = "{\"request\":\"{\\\"a\\\": 1}\"}" },
    .{ .schema = type_and_enum_schema, .value = "null", .expected = "{\"request\":null}" },
};

/// A reference whose tilde escapes nothing.
pub const bad_escape_schema = "{\"$ref\":\"#/$defs/a~xb\"}";

/// A reference with a percent escape.
pub const percent_escape_schema = "{\"$ref\":\"#/$defs/a%20b\"}";

/// The definitions the two references point into.
pub const pointer_defs = "\"$defs\":{\"a~b\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}}},\"a b\":{\"type\":\"object\",\"properties\":{\"y\":{\"type\":\"integer\"}}}}";

/// The schemas whose references are pointers.
pub const pointer_cases = [_]Case{
    .{ .schema = bad_escape_schema, .defs = pointer_defs, .value = "{\"x\": 1}", .expected = "{\"request\":\"{\\\"x\\\": 1}\"}" },
    .{ .schema = percent_escape_schema, .defs = pointer_defs, .value = "{\"y\": 2}", .expected = "{\"request\":{\"y\":2}}" },
};

/// A reference to a definition that names no kinds.
pub const free_schema = "{\"$ref\":\"#/$defs/Free\"}";

/// An object type with a reference to a definition that names no kinds.
pub const typed_free_schema = "{\"type\":\"object\",\"$ref\":\"#/$defs/Free\"}";

/// The definition that names no kinds.
pub const free_defs = "\"$defs\":{\"Free\":{\"description\":\"anything\"}}";

/// The schemas whose reference names no kinds, and the ones that name nothing at all.
pub const unconstrained_cases = [_]Case{
    .{ .schema = typed_free_schema, .defs = free_defs, .value = "{\"a\": 1}", .expected = "{\"request\":{\"a\":1}}" },
    .{ .schema = free_schema, .defs = free_defs, .value = "{\"a\": 1}", .expected = "{\"request\":\"{\\\"a\\\": 1}\"}" },
    .{ .schema = cycle_schema, .defs = cycle_defs, .value = "{\"id\": 7}", .expected = "{\"request\":\"{\\\"id\\\": 7}\"}" },
    .{ .schema = missing_schema, .value = "{\"id\": 7}", .expected = "{\"request\":\"{\\\"id\\\": 7}\"}" },
};
