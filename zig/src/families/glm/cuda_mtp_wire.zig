//! Shared MTP rank messages retain each stream's own metadata and chance-buffer intent.
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const follow = @import("cuda_follow.zig");
const Header = follow.Header;
const Kind = follow.Kind;
pub const words = 2 * engine.max_shared + engine.max_shared * fw.window_rows;
const draft_shared_words = words;

/// A slot and packed keep/length/depth/prompt/chance flags per part, then its follow ids.
pub fn encodeDraftShared(parts: []const engine.DraftPart, buf: *[draft_shared_words]u32) !struct { h: Header, ids: []const u32 } {
    if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
    var n: usize = 2 * parts.len;
    var has_chances = false;
    for (parts, 0..) |p, i| {
        if (p.depth > engine.max_drafts) return error.TooDeep;
        if (p.kept > fw.window_rows or p.follow.len == 0 or p.follow.len > fw.window_rows) return error.WindowTooWide;
        if (p.from_prompt and p.follow.len != 1) return error.WindowTooWide;
        if (p.out.len < p.depth) return error.WindowTooWide;
        if (p.chances) |c| if (c.len < p.depth) return error.WindowTooWide;
        if (n + p.follow.len > buf.len) return error.WindowTooWide;
        has_chances = has_chances or p.chances != null;
        buf[2 * i] = p.slot;
        buf[2 * i + 1] = @as(u32, @intCast(p.kept)) | (@as(u32, @intCast(p.follow.len)) << 8) | (@as(u32, @intCast(p.depth)) << 16) | (@as(u32, @intFromBool(p.from_prompt)) << 24) | (@as(u32, @intFromBool(p.chances != null)) << 25);
        @memcpy(buf[n..][0..p.follow.len], p.follow);
        n += p.follow.len;
    }
    return .{ .h = .{ .kind = @backingInt(Kind.draft_shared), .depth = @intCast(parts.len), .n = @intCast(n), .flags = @as(u32, @intFromBool(has_chances)) << 1 }, .ids = buf[0..n] };
}

/// Decode each part without turning another stream's chance request on for it.
pub fn decodeDraftShared(h: Header, ids: []const u32, parts: *[engine.max_shared]engine.DraftPart, outs: *[engine.max_shared][engine.max_drafts]u32, chances: *[engine.max_shared][engine.max_drafts]f64, follow_buf: *[engine.max_shared * fw.window_rows]u32) ![]const engine.DraftPart {
    if (h.kind != @backingInt(Kind.draft_shared) or h.flags & ~@as(u32, 2) != 0 or h.n != ids.len) return error.BadStep;
    const k = h.depth;
    if (k == 0 or k > engine.max_shared or ids.len < 2 * k) return error.BadStep;
    const header_chances = h.flags & 2 != 0;
    var any_chances = false;
    var at: usize = 2 * k;
    var f_at: usize = 0;
    for (parts[0..k], 0..) |*p, i| {
        const meta = ids[2 * i + 1];
        if (meta >> 26 != 0) return error.BadStep;
        const want_chances = meta & (1 << 25) != 0;
        any_chances = any_chances or want_chances;
        const kept: usize = meta & 0xff;
        const len: usize = (meta >> 8) & 0xff;
        const depth: usize = (meta >> 16) & 0xff;
        const from_prompt = ((meta >> 24) & 1) != 0;
        if (kept > fw.window_rows or len == 0 or len > fw.window_rows or depth > engine.max_drafts or (from_prompt and len != 1) or at + len > ids.len or f_at + len > follow_buf.len) return error.BadStep;
        @memcpy(follow_buf[f_at..][0..len], ids[at..][0..len]);
        p.* = .{ .slot = ids[2 * i], .kept = kept, .from_prompt = from_prompt, .follow = follow_buf[f_at..][0..len], .depth = depth, .out = outs[i][0..depth], .chances = if (want_chances) chances[i][0..depth] else null };
        at += len;
        f_at += len;
    }
    if (at != ids.len or any_chances != header_chances) return error.BadStep;
    return parts[0..k];
}
