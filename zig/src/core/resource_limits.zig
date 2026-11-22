//! The native server's opt-in pre-resource input limits.
const std = @import("std");

pub const switch_name = "TENSORFOLD_HTTP_STRICT_INPUT_LIMITS";

/// Off by default: enabling the limits can refuse prompts main accepts.
pub fn enabled() bool {
    const raw = std.c.getenv("TENSORFOLD_HTTP_STRICT_INPUT_LIMITS") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}
