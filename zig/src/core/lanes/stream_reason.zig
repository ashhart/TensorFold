//! Why a stream ended: as a reply does, on its budget, failed, or cancelled.
const std = @import("std");

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};
