//! What a HIP family's contract entry is, apart from the list of them, so the pure selector needs no family code.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");

/// How much of its gates 1-4 an entry runs: a quick check or every shape.
pub const Profile = enum { smoke, full };

/// A family's image, the gate-5 records it covers by launch name (none: every record of `group`), and its gates 1-4.
pub const Entry = struct {
    group: hip.kernels.Group,
    records: []const []const u8 = &.{},
    run: *const fn (std.mem.Allocator, std.Io, Profile) anyerror!core.contract.gates.Tally,
};
