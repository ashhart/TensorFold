const std = @import("std");
const api = @import("engine_api");
const serve = @import("hip_serve");
const fam = @import("hip_families.zig");
const Allocator = std.mem.Allocator;

test "HIP optional family offers: image metadata comes from the loaded owner's offer" {
    try std.testing.expect(fam.offerOf(.{}).vision == null);
    try std.testing.expect(fam.offerOf(.{ .offer = serve.Offer{} }).vision == null);
    const Stub = struct {
        fn prepare(_: *anyopaque, _: Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
    };
    var owner: u8 = 0;
    const v: api.Vision = .{ .ctx = &owner, .prepare = Stub.prepare, .image_token = 9, .image_tokens = 6144, .max_images = 2 };
    const offered = fam.offerOf(.{ .offer = serve.Offer{ .vision = v } }).vision.?;
    try std.testing.expectEqual(v.ctx, offered.ctx);
    try std.testing.expectEqual(v.image_token, offered.image_token);
    try std.testing.expectEqual(v.image_tokens, offered.image_tokens);
}
