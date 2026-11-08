//! Tensor-only contract probes report an unchecked family and leave the supported probes running.
const std = @import("std");

pub fn tensor(available: bool, name: []const u8) bool {
    if (available) return true;
    std.debug.print("UNSUPPORTED {s}: tensor units required; coverage unchecked\n", .{name});
    return false;
}
