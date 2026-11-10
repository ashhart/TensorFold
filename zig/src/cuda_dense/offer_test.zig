//! The tower variant as the host reads it: its options from the serve flags, its vision offer, its tower's bytes.
const std = @import("std");
const api = @import("engine_api");
const offer = @import("vision_offer");
const options = @import("cuda_options");
const native = @import("native.zig");
const fixtures = @import("tiny_test.zig");
const t = std.testing;

const V = native.Native(fixtures.TinyTower);
const N = native.Native(fixtures.Tiny);

test "the host's offer reads the tower's marks, limits, prepare and words through Loaded.vision" {
    var tower: fixtures.Fake = .{ .limits = .{ .image_tokens = 128, .max_images = 3 } };
    var loaded: V.Loaded = undefined;
    loaded.vision = &tower;
    const v = offer.offer(loaded.vision, null).?;
    try t.expectEqual(@as(u32, 500), v.image_token);
    try t.expectEqualSlices(u32, &.{ 501, 500, 502 }, v.markers);
    try t.expect(v.wrap == null and v.video == null); // the template writes the open and close marks itself
    try t.expectEqualStrings("image", v.part_type);
    try t.expect(v.image_tokens == 128 and v.max_images == 3);
    const bytes = "\x09abc";
    const p = try v.prepare(v.ctx, t.allocator, bytes, 64);
    defer t.allocator.free(p.pixels);
    try t.expect(p.gh == 4 and p.gw == 4 and p.tokens == 4 and p.pixels[0] == 9);
    try t.expectEqual(std.hash.Wyhash.hash(0, bytes), p.hash);
    try t.expectError(error.FakeImage, v.prepare(v.ctx, t.allocator, "", 64));
    try t.expectStringStartsWith(v.explain.?(error.FakeImage).?, "the fake tower");
    loaded.vision = null; // served without --vision: no offer
    try t.expect(offer.offer(loaded.vision, null) == null);
}

test "the serve flags set the tower's limits: its own default image tokens unless --vision-image-tokens names them" {
    var o: V.Options = .{ .context = 4096, .drafts = true };
    var why: []const u8 = "";
    try t.expect(options.set(V, .{ .dir = "", .model_type = "tiny_dense", .vision = .{ .max_images = 3 } }, &o, &why));
    try t.expectEqual(V.Options{ .context = 4096, .drafts = true, .vision = .{ .image_tokens = 64, .max_images = 3 } }, o);
    try t.expect(options.set(V, .{ .dir = "", .model_type = "tiny_dense", .vision = .{ .image_tokens = 1000, .image_tokens_set = true } }, &o, &why));
    try t.expect(o.vision.?.image_tokens == 1000 and o.vision.?.max_images == 4);
    var plain: V.Options = .{ .context = 4096, .drafts = true };
    try t.expect(options.set(V, .{ .dir = "", .model_type = "tiny_dense" }, &plain, &why));
    try t.expect(plain.vision == null);
}

test "the host counts the tower's bytes with --vision on rank 0, none without it or for a text-only recipe" {
    const a = t.allocator;
    const io = t.io;
    const with: api.Open = .{ .dir = "", .model_type = "tiny_dense", .vision = .{} };
    try t.expectEqual(try V.visionBytes(a, io, ""), try offer.weightBytes(V, a, io, with));
    try t.expectEqual(@as(u64, 0), try offer.weightBytes(V, a, io, .{ .dir = "", .model_type = "tiny_dense" }));
    try t.expectEqual(@as(u64, 0), try offer.weightBytes(N, a, io, with));
}
