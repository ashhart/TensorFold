//! What a CUDA family with an image tower offers the server: its processor, image token, markers and limits.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// Info.vision for a family's tower (Nemotron Omni's, GLM-5.3-Flash's); null without --vision.
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
    // a template that writes each image's begin and end itself: the server expands the image token alone
    const wrap: ?[2]u32 = if (@hasDecl(T, "template_wraps")) null else .{ x.marks.open, x.marks.close };
    return .{ .ctx = x, .prepare = prepare, .image_token = x.marks.image, .markers = &x.mark_ids, .wrap = wrap, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = T.explain };
}
