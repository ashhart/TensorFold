//! VerbsWire on fake verbs whose completions lag: what it holds in flight and what a failed transport leaves behind.
const std = @import("std");
const rv = @import("roce_verbs.zig");
const fake = @import("roce_fake.zig");
const Roce = @import("roce.zig").Roce;
const Fake = fake.Fake;
const slot = fake.slot;

test "chunked posts: the wire's own in-flight count and peak match the HCAs', and reach the 2 MiB cap" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(slot));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.slot_bytes = slot;
    Fake.reset(3);
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    for (1..9) |seq| {
        try w.post(@intCast(seq & 1), @intCast(seq), slot);
        try std.testing.expectEqual(Fake.inflight, w.inflight());
        try std.testing.expect(w.inflight() <= rv.max_inflight);
    }
    while (w.inflight() > 0) try w.reap();
    try std.testing.expectEqual(@as(u64, 0), Fake.inflight);
    try std.testing.expectEqual(Fake.max_inflight, w.peak.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2 << 20), w.peak.load(.monotonic));
}
