//! Where each message the chat template renders begins, in tokens: a byte start is kept only where the ids agree.
const std = @import("std");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

/// The token offsets in `ids` (`rendered`'s encoding) of the byte starts `bytes`: sorted, past 0, before the end.
pub fn tokens(t: model_text.Text, a: Allocator, rendered: model_text.Rendered, ids: []const u32, bytes: []const usize) model_text.Error![]const u32 {
    if (bytes.len == 0 or rendered.segments.len > 0) return &.{};
    var out: std.ArrayList(u32) = .empty;
    var from: usize = 0;
    var at: usize = 0;
    var problem: []const u8 = "";
    for (bytes) |b| {
        if (b <= from or b >= rendered.text.len) continue;
        const piece = (try rendered.slice(a, from, b)).encode(t, a, &problem) catch |e| return if (e == error.OutOfMemory) e else &.{};
        // a start inside a token of the whole encoding: its piece joins the next one, and no start is kept there
        if (at + piece.len >= ids.len or !std.mem.eql(u32, ids[at..][0..piece.len], piece)) continue;
        at += piece.len;
        from = b;
        try out.append(a, @intCast(at));
    }
    return out.items;
}
