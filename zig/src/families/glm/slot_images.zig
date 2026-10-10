//! A stream's images through the tower into the engine's image rows, which its prompt chunks inject over their
//! placeholders (forward.zig's `inject`); the tower remembers whose rows it holds until the pass ends.
const std = @import("std");
const mtl = @import("metal");
const vision_mod = @import("vision.zig");
const Slots = @import("slots.zig").Slots;
const Image = @import("lanes").stream.Image;

/// Slot `i`'s images through the tower into the engine's image rows, injected over their placeholders by the chunks
/// that follow until `end`. One image a command buffer: the tower's host-written scratch is shared.
pub fn encode(sl: *Slots, i: u32, imgs: []const Image) !void {
    const e = sl.e;
    const v = e.vision orelse return error.NoVision;
    if (imgs.len > e.vision_limits.max_images) return error.TooManyImages;
    if (i >= sl.slots.len or !sl.slots[i].used) return error.SlotOutOfStep;
    v.spans.clearRetainingCapacity();
    try sl.flush();
    e.sync();
    var row: u32 = 0;
    for (imgs) |img| {
        if (row + img.tokens > e.vision_limits.image_tokens or img.gh * img.gw != img.tokens * vision_mod.patches_per_token) return error.ImageTooLarge;
        if (img.at + img.tokens > sl.slots[i].prompt_len) return error.ImageOutOfPrompt;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const b = e.begin();
        try v.encode(b.enc, img.pixels, img.gh, img.gw, e.vision_rows.at(@as(usize, row) * e.c.hidden * 2), null);
        try e.finish(b.cb, b.enc);
        try v.spans.append(v.gpa, .{ .at = img.at, .n = img.tokens, .row = row });
        row += img.tokens;
    }
    v.inject = .{ .slot = i, .v = .{ .rows = e.vision_rows, .spans = v.spans.items } };
}

/// The stream's prompt pass is over: the chunks after it inject nothing.
pub fn end(sl: *Slots) void {
    if (sl.e.vision) |v| v.inject = null;
}
