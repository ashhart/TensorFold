//! The std options a server binary takes from its backend: the backend's own when it declares them, else the defaults.

const std = @import("std");

pub fn of(comptime Backend: type) std.Options {
    return if (@hasDecl(Backend, "std_options")) Backend.std_options else .{};
}

test "a backend's std options pass through, and a backend without them keeps the defaults" {
    const Plain = struct {};
    const Rccl = struct {
        pub const std_options: std.Options = .{ .signal_stack_size = null };
    };
    try std.testing.expectEqual((std.Options{}).signal_stack_size, of(Plain).signal_stack_size);
    try std.testing.expectEqual(@as(?usize, null), of(Rccl).signal_stack_size);
}
