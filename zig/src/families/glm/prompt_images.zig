//! The shared image rows belong to one prompt at a time; another prompt can rebuild them when it resumes.
const lanes = @import("lanes");
const Stream = lanes.Stream;
const Image = lanes.stream.Image;

/// The images (in prompt order) whose rows a pass from `from` still embeds: a restored state holds those ending by it.
pub fn past(images: []const Image, from: usize) []const Image {
    for (images, 0..) |img, k| if (@as(usize, img.at) + img.tokens > from) return images[k..];
    return images[images.len..];
}

/// The images ending after `from` (the restore point, then each chunk's start) through the tower, unless held.
pub fn prepare(b: anytype, s: *Stream, slot: u32, from: usize) !void {
    const todo = past(s.images, from);
    if (todo.len == 0 or b.image_owner == s) return;
    b.image_owner = null;
    b.endImages();
    try b.encodeImages(slot, todo);
    b.image_owner = s;
}
