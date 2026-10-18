//! The release refusal reads controlled values without changing the process environment.
const std = @import("std");
pub const experimental = @import("sys").experimental;
pub var value: ?[:0]const u8 = null;
pub fn getenv(name: [*:0]const u8) ?[:0]const u8 {
    return if (std.mem.eql(u8, std.mem.span(name), "TF_EXPERIMENTAL")) value else null;
}
