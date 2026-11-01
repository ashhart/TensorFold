//! A server binary's std options: its backend's own when it declares them, else the defaults, with the server's log.

const std = @import("std");

pub fn of(comptime Backend: type) std.Options {
    var o: std.Options = if (@hasDecl(Backend, "std_options")) Backend.std_options else .{};
    o.logFn = @import("log.zig").stdLog;
    return o;
}

test "a backend's std options pass through, and a backend without them keeps the defaults" {
    const Plain = struct {};
    const Rccl = struct {
        pub const std_options: std.Options = .{ .signal_stack_size = null };
    };
    try std.testing.expectEqual((std.Options{}).signal_stack_size, of(Plain).signal_stack_size);
    try std.testing.expectEqual(@as(?usize, null), of(Rccl).signal_stack_size);
}
