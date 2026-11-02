//! Three-coordinate image RoPE indices; adjacent-pair rotary is applied by the device graph.
const std = @import("std");
const c = @import("config.zig");
const Geometry = @import("processor.zig").Geometry;
pub const Error = error{ BadImageSpan, ContextExceeded };
pub const Result = struct { delta: i32, next: i32 };

/// Coordinates are token-major. Every image must have exactly one delimited run of merged rows.
pub fn fill(ids: []const u32, images: []const Geometry, out: [][3]i32) Error!Result {
    if (ids.len != out.len or ids.len > c.context) return error.ContextExceeded;
    var pos: usize = 0;
    var at: usize = 0;
    var image: usize = 0;
    while (at < ids.len) {
        if (ids[at] == c.image_token or ids[at] == c.image_end) return error.BadImageSpan;
        out[at] = @splat(@intCast(pos));
        const start_image = ids[at] == c.image_start;
        at += 1;
        pos += 1;
        if (!start_image) continue;
        if (image == images.len) return error.BadImageSpan;
        const g = images[image];
        if (g.h == 0 or g.w == 0 or g.h % 28 != 0 or g.w % 28 != 0 or g.h > 16384 or g.w > 16384)
            return error.BadImageSpan;
        const gh = g.h / 28;
        const gw = g.w / 28;
        const count = gh * gw;
        if (count >= ids.len - at) return error.BadImageSpan;
        for (0..count) |i| {
            if (ids[at + i] != c.image_token) return error.BadImageSpan;
            out[at + i] = .{ @intCast(pos), @intCast(pos + i / gw), @intCast(pos + i % gw) };
        }
        at += count;
        pos += @max(gh, gw);
        if (ids[at] != c.image_end) return error.BadImageSpan;
        out[at] = @splat(@intCast(pos));
        pos += 1;
        at += 1;
        image += 1;
    }
    if (image != images.len or pos > c.context) return error.BadImageSpan;
    return .{ .delta = @as(i32, @intCast(pos)) - @as(i32, @intCast(ids.len)), .next = @intCast(pos) };
}
