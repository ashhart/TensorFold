//! Wrapper-specific feature retention decorates a normal backend without rebuilding its unrelated fields.
const std = @import("std");
const be = @import("backend.zig");

pub fn inherit(result: be.Backend, inner: be.Backend) be.Backend {
    var decorated = result;
    decorated.restored_features = inner.restored_features;
    decorated.needs_full_keep = inner.needs_full_keep;
    return decorated;
}

/// The original constructor remains authoritative for every field except these inherited cache capabilities.
pub fn Wrapped(comptime T: type, comptime make: anytype, comptime field: []const u8) type {
    return struct {
        pub fn backend(x: *T) be.Backend {
            return inherit(make(x), @field(x, field));
        }
    };
}

test "backend cache capabilities: decorators preserve constructor fields and both independent flags" {
    const Probe = struct {
        target: be.Backend,
        inner: be.Backend,
        fn make(_: *@This()) be.Backend {
            return .{ .ptr = @ptrFromInt(0x1000), .vtable = @ptrFromInt(0x2000), .first_before_draft = true, .penalties = true, .chunk_invariant = false };
        }
    };
    for ([_]bool{ false, true }) |restored| for ([_]bool{ false, true }) |full| {
        const inner: be.Backend = .{ .ptr = @ptrFromInt(0x3000), .vtable = @ptrFromInt(0x4000), .restored_features = restored, .needs_full_keep = full, .chunk_invariant = true };
        var probe: Probe = .{ .target = inner, .inner = inner };
        for ([_]be.Backend{ Wrapped(Probe, Probe.make, "target").backend(&probe), Wrapped(Probe, Probe.make, "inner").backend(&probe) }) |actual| {
            var expected = Probe.make(&probe);
            expected.restored_features = restored;
            expected.needs_full_keep = full;
            try std.testing.expect(std.meta.eql(expected, actual));
            try std.testing.expectEqual(restored, actual.restored_features);
            try std.testing.expectEqual(full, actual.needs_full_keep);
            try std.testing.expectEqual(@as(usize, 0x1000), @intFromPtr(actual.ptr));
            try std.testing.expectEqual(@as(usize, 0x2000), @intFromPtr(actual.vtable));
            try std.testing.expect(actual.first_before_draft and actual.penalties and !actual.chunk_invariant);
        }
        try std.testing.expectEqual(restored, probe.target.restored_features);
        try std.testing.expectEqual(full, probe.inner.needs_full_keep);
    };
}
