//! The drafter flags: a neural drafter's checkpoint and the bits its draft head reads it at.
const cli = @import("cli.zig");

/// The drafter flags, spliced into cli.flags where the Python table lists them.
pub const flags = [_]cli.Flag{
    .{ .name = "--drafter", .native = true },
    .{ .name = "--drafter-bits", .choices = &.{ "0", "4", "8" }, .native = true },
};
