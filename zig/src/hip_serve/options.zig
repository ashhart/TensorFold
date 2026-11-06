//! Native HIP open settings shared by family loaders and the host that admits them.
const api = @import("engine_api");

pub const Options = struct {
    window: usize,
    streams: usize,
    fixed: bool = false,
    rank: u32 = 0,
    world: u32 = 1,
    cache_gib: ?f64 = null,
    cache_over_cap: bool = false,
    vision: ?api.VisionOpen = null,
};

test "HIP native options: existing callers remain text-only and image limits retain explicit overrides" {
    const std = @import("std");
    const text: Options = .{ .window = 8192, .streams = 2 };
    try std.testing.expect(text.vision == null and !text.fixed and text.world == 1);
    const image: Options = .{ .window = 8192, .streams = 2, .vision = .{ .image_tokens = 6144, .image_tokens_set = true, .max_images = 2 } };
    try std.testing.expectEqual(@as(u32, 6144), image.vision.?.image_tokens);
    try std.testing.expect(image.vision.?.image_tokens_set);
}
