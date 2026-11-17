//! `zig build test-device`: the machine scan's parsers, the support table, the profile and the summary, with no GPU.
test {
    _ = @import("parse_test.zig");
    _ = @import("support_test.zig");
    _ = @import("profile_test.zig");
    _ = @import("report_test.zig");
    _ = @import("read.zig");
}

test "device scan: this host's own quick scan reads without a driver and finds its RAM" {
    const std = @import("std");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const m = try @import("read.zig").quick(arena.allocator(), std.testing.io);
    try std.testing.expect(m.ram > 0);
    try std.testing.expect(m.arch.len > 0);
}
