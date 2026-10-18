//! The shared image rows belong to one prompt at a time; another prompt can rebuild them when it resumes.
const Stream = @import("lanes").Stream;
pub fn prepare(b: anytype, s: *Stream, slot: u32) !void {
    if (s.images.len == 0 or b.image_owner == s) return;
    b.image_owner = null;
    b.endImages();
    try b.encodeImages(slot, s.images);
    b.image_owner = s;
}
