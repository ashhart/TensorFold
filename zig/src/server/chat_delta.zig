//! Content, role and reasoning deltas shared by the reply and wire endpoints.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("json").Value;
pub const deltaOf = @import("reply_text.zig").deltaOf;

pub fn role(a: Allocator) Allocator.Error!Value {
    return deltaOf(a, "role", "assistant");
}
