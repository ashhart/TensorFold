//! Validate a server image's content geometry before allocating or launching the Parse tower.
const c = @import("config.zig");
const processor = @import("processor.zig");

pub fn geometry(image: anytype) !processor.Geometry {
    if (image.at != 0) return error.ParseImageMustBePrefix;
    const content = image.content_size orelse return error.ParseContentSizeRequired;
    if (image.gh == 0 or image.gh > c.max_height / c.patch or image.gw == 0 or
        image.gw > c.max_width / c.patch or image.gw % 4 != 0) return error.BadParseImage;
    if (content[0] == 0 or content[1] == 0 or content[0] > image.gh * 16 or content[1] > image.gw * 16)
        return error.BadParseImage;
    const g: processor.Geometry = .{ .content = .{ .h = content[0], .w = content[1] }, .padded = .{ .h = @as(usize, image.gh) * 16, .w = @as(usize, image.gw) * 16 } };
    if ((g.content.h + 15) / 16 != image.gh or (g.content.w + 63) / 64 * 4 != image.gw or
        g.tokens() != image.tokens or g.patches() * 768 != image.pixels.len) return error.BadParseImage;
    return g;
}
