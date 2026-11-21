//! The --vision-max-images, --vision-max-videos, --vision-image-tokens and --vision-video-tokens flags.
const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("cli.zig");

/// A vision limit flag's value in its range and its effect on `out`; false for any other flag.
pub fn flag(a: Allocator, out: *cli.Args, name: []const u8, v: []const u8, u: *cli.Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--vision-max-images")) {
        const n = try cli.int(u, a, name, v);
        if (n < 1 or n > 64) return cli.fail(u, a, "argument --vision-max-images: expected 1 to 64 images: '{s}'", .{v});
        out.vision_max_images = @intCast(n);
    } else if (std.mem.eql(u8, name, "--vision-max-videos")) {
        const n = try cli.int(u, a, name, v);
        if (n < 1 or n > 16) return cli.fail(u, a, "argument --vision-max-videos: expected 1 to 16 videos: '{s}'", .{v});
        out.vision_max_videos = @intCast(n);
    } else if (std.mem.eql(u8, name, "--vision-video-tokens")) {
        const n = try cli.int(u, a, name, v);
        if (n < 64 or n > 262144) return cli.fail(u, a, "argument --vision-video-tokens: expected 64 to 262144 tokens: '{s}'", .{v});
        out.vision_video_tokens = @intCast(n);
    } else if (std.mem.eql(u8, name, "--vision-image-tokens")) {
        const n = try cli.int(u, a, name, v);
        if (n < 16 or n > 262144) return cli.fail(u, a, "argument --vision-image-tokens: expected 16 to 262144 tokens: '{s}'", .{v});
        out.vision_image_tokens = @intCast(n);
        out.vision_image_tokens_set = true;
    } else return false;
    return true;
}
