//! Target window IDs on the GLM peer wire; rank zero chooses, followers replay the same words.
const std = @import("std");
pub const Part = struct { slot: u32, ids: []const u32 };
pub fn window(pending: u32, drafts: []const u32, out: []u32) ![]u32 {
    if (out.len == 0 or drafts.len >= out.len) return error.WindowTooWide;
    out[0] = pending;
    @memcpy(out[1..][0..drafts.len], drafts);
    return out[0 .. drafts.len + 1];
}
pub fn shared(parts: anytype, out: []u32, max_streams: usize, row_limit: usize) ![]const u32 {
    if (parts.len == 0 or parts.len > max_streams or parts.len > out.len / 2) return error.WindowTooWide;
    var at: usize = 2 * parts.len;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len > row_limit -| rows or p.ids.len > out.len -| at) return error.WindowTooWide;
        out[2 * i] = p.slot;
        out[2 * i + 1] = std.math.cast(u32, p.ids.len) orelse return error.WindowTooWide;
        @memcpy(out[at..][0..p.ids.len], p.ids);
        at += p.ids.len;
        rows += p.ids.len;
    }
    return out[0..at];
}
pub fn following(ids: []const u32, count: usize, out: []Part, row_limit: usize) ![]Part {
    if (count == 0 or count > out.len or count > ids.len / 2) return error.BadStep;
    var at = 2 * count;
    var rows: usize = 0;
    for (out[0..count], 0..) |*p, i| {
        const n = ids[2 * i + 1];
        if (n == 0 or n > ids.len -| at or n > row_limit -| rows) return error.BadStep;
        p.* = .{ .slot = ids[2 * i], .ids = ids[at..][0..n] };
        at += n;
        rows += n;
    }
    if (at != ids.len) return error.BadStep;
    return out[0..count];
}
