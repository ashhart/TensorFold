//! Native CUDA wrapper capability flags, exercised through the production backend adapter.
const std = @import("std");
const lanes = @import("lanes");

pub fn check(comptime Host: type, comptime backend: anytype) !void {
    var target: lanes.fake.Fake = .{ .gpa = std.testing.allocator, .prefill_chunks = 3 };
    defer target.deinit();
    var h: Host = undefined;
    h.inner = target.stepped();
    try std.testing.expect(backend(&h).vtable.prefill_step != null);
    try std.testing.expect(!backend(&h).first_before_draft);
    h.inner.first_before_draft = true;
    try std.testing.expect(backend(&h).first_before_draft);
    try std.testing.expect(!backend(&h).penalties);
    h.inner.penalties = true;
    try std.testing.expect(backend(&h).penalties);
    try std.testing.expect(backend(&h).chunk_invariant); // the fake declares it; the wrapper must carry it
    h.inner.chunk_invariant = false;
    try std.testing.expect(!backend(&h).chunk_invariant);
}
