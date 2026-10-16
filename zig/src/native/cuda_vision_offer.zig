//! What a CUDA family with an image tower offers the server: its processor, image token, markers and limits.
const std = @import("std");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const qwen35 = @import("qwen35");
const Allocator = std.mem.Allocator;

/// Info.vision for a family's tower (Nemotron Omni's or Qwen's); null without --vision.
pub fn offer(v: anytype) ?api.Vision {
    const x = v orelse return null;
    const T = @typeInfo(@TypeOf(x)).pointer.child;
    const prepare = struct {
        fn f(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
            const t: *T = @ptrCast(@alignCast(ctx));
            const p = try t.prepare(a, bytes, max_tokens);
            return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
        }
    }.f;
    if (comptime T == qwen35.native.vision.Vision) {
        // the template writes Qwen's vision_start and vision_end around the image token itself
        const image = x.shared.image_token;
        return .{ .ctx = x, .prepare = prepare, .image_token = image, .markers = &x.markers, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = T.explain };
    }
    return .{ .ctx = x, .prepare = prepare, .image_token = x.marks.image, .markers = &x.mark_ids, .wrap = .{ x.marks.open, x.marks.close }, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = nemotron.native.vision.Vision.explain };
}
