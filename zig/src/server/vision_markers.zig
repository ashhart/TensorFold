//! Vision parsing accepts only added-token markers, which request provenance can exclude from matching.
const api = @import("engine_api");
const Text = @import("model_text.zig").Text;

/// The first unsafe marker, including wrappers and optional video timestamps.
pub fn firstUnsafe(text: Text, vision: api.Vision) ?u32 {
    if (!text.isAdded(vision.image_token)) return vision.image_token;
    for (vision.markers) |id| if (!text.isAdded(id)) return id;
    if (vision.wrap) |ids| for (ids) |id| {
        if (!text.isAdded(id)) return id;
    };
    if (vision.video) |video| {
        if (!text.isAdded(video.video_token)) return video.video_token;
        if (!text.isAdded(video.group_open)) return video.group_open;
        if (!text.isAdded(video.group_close)) return video.group_close;
        for (video.markers) |id| if (!text.isAdded(id)) return id;
        if (video.time_first) |id| if (!text.isAdded(id)) return id;
    }
    return null;
}
