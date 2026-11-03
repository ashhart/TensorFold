//! Content, role and reasoning deltas shared by the reply and wire endpoints.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;
/// ``{key: text}``: a streamed delta (role, content or reasoning_content).
pub fn deltaOf(a: Allocator, key: []const u8, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, key, .{ .string = text });
    return .{ .object = o };
}

pub fn role(a: Allocator) Allocator.Error!Value {
    return deltaOf(a, "role", "assistant");
}
