//! What a CUDA family with an image tower offers the server: its processor, image token, markers and limits.
const std = @import("std");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const Allocator = std.mem.Allocator;

/// Info.vision for a Nemotron Omni engine's tower; null without --vision.
pub fn offer(v: ?*nemotron.native.vision.Vision) ?api.Vision {
    const x = v orelse return null;
    const prepare = struct {
        fn f(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
            const t: *nemotron.native.vision.Vision = @ptrCast(@alignCast(ctx));
            const p = try t.prepare(a, bytes, max_tokens);
            return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
        }
    }.f;
    return .{ .ctx = x, .prepare = prepare, .image_token = x.marks.image, .markers = &x.mark_ids, .wrap = .{ x.marks.open, x.marks.close }, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = nemotron.native.vision.Vision.explain };
}
