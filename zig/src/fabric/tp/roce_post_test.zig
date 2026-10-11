//! VerbsWire.post on fake verbs: a post that would leave its slot or region is refused before any verb.
const std = @import("std");
const fake = @import("roce_fake.zig");
const Roce = @import("roce.zig").Roce;
const Fake = fake.Fake;

test "a post past its slot is refused before any verb; a full slot posts inside the region" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(fake.slot));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.reset(1);
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    try std.testing.expectError(error.OutOfRegion, w.post(1, 1, fake.slot + 16));
    try std.testing.expectError(error.OutOfRegion, w.post(2, 1, 16));
    try std.testing.expectEqual(@as(u64, 0), Fake.posts);
    try w.post(1, 1, fake.slot);
    try std.testing.expectEqual(@as(u64, 2), Fake.posts);
    try std.testing.expectEqual(@as(u64, 0), Fake.outside);
}
