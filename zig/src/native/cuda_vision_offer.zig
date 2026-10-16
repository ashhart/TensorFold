//! What a CUDA family with an image tower offers the server: its processor, image token, markers and limits.
const std = @import("std");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const Allocator = std.mem.Allocator;
const Vision = nemotron.native.vision.Vision;
const video = nemotron.native.vision.video;

/// Info.vision for a Nemotron Omni engine's tower; null without --vision. `open` gives a video offer its limits.
pub fn offer(v: ?*Vision, open: ?api.VisionOpen) ?api.Vision {
    const x = v orelse return null;
    const prepare = struct {
        fn f(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
            const t: *Vision = @ptrCast(@alignCast(ctx));
            const p = try t.prepare(a, bytes, max_tokens);
            return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
        }
    }.f;
    var out: api.Vision = .{ .ctx = x, .prepare = prepare, .image_token = x.marks.image, .markers = &x.mark_ids, .wrap = .{ x.marks.open, x.marks.close }, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = Vision.explain };
    // video: the checkpoint's temporal projection, and a reserved token for the template's `<video>`
    if (x.cfg.video_temporal == 2 and x.marks.video != null) if (open) |o| {
        out.video = .{
            .prepare = prepareVideo,
            .video_token = x.marks.video.?,
            .video_text = "<video>",
            .group_open = x.marks.open,
            .group_close = x.marks.close,
            .markers = &x.mark_ids,
            .video_tokens = @min(o.video_tokens, x.limits.video_tokens),
            .max_videos = o.max_videos,
            .max_frames = o.max_frames,
        };
    };
    return out;
}

/// The server's decoder behind the tower's video source.
const Decoding = struct {
    src: api.VideoSource,

    fn decode(ctx: *anyopaque, indices: []const u32, sink: video.Sink) anyerror!void {
        const d: *Decoding = @ptrCast(@alignCast(ctx));
        return d.src.decode(d.src.ctx, indices, .{ .ctx = sink.ctx, .take = sink.take });
    }
};

fn prepareVideo(ctx: *anyopaque, a: Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
    const t: *Vision = @ptrCast(@alignCast(ctx));
    var d: Decoding = .{ .src = src };
    const p = try t.prepareVideo(a, .{
        .frames = src.frames,
        .rate_num = src.rate_num,
        .rate_den = src.rate_den,
        .width = src.width,
        .height = src.height,
        .ctx = &d,
        .decode = Decoding.decode,
    }, max_tokens, max_frames);
    const groups = try a.alloc(api.PreparedImage, p.groups.len);
    for (groups, p.groups) |*o, g| o.* = .{ .pixels = g.pixels, .gh = g.gh, .gw = g.gw, .tokens = g.tokens, .hash = g.hash };
    return .{ .groups = groups, .times = p.times, .labels = p.labels };
}
