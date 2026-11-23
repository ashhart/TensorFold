//! Where each message the chat template renders begins, in tokens: only at proven boundaries of the prompt's encoding.
const std = @import("std");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

/// Token offsets of the byte starts where the tokens before spell the text exactly and each side re-encodes alone.
pub fn tokens(t: model_text.Text, a: Allocator, rendered: model_text.Rendered, ids: []const u32, bytes: []const usize) model_text.Error![]const u32 {
    if (bytes.len == 0 or rendered.segments.len > 0) return &.{};
    const edges = try tokenEdges(t, a, rendered.text, ids, bytes);
    if (edges.len == 0) return &.{};
    var out: std.ArrayList(u32) = .empty;
    var problem: []const u8 = "";
    var cursor: usize = 0;
    var lo: Edge = .{ .byte = 0, .token = 0 };
    var lo_ok = false; // the piece before `lo` encodes alone to its ids
    for (0..edges.len + 1) |j| {
        const hi: Edge = if (j < edges.len) edges[j] else .{ .byte = rendered.text.len, .token = ids.len };
        const piece = try slice(a, rendered, &cursor, lo.byte, hi.byte);
        const got: []const u32 = piece.encode(t, a, &problem) catch |e| if (e == error.OutOfMemory) return e else &.{};
        const ok = got.len > 0 and std.mem.eql(u32, got, ids[lo.token..hi.token]);
        if (j > 0 and lo_ok and ok) try out.append(a, @intCast(lo.token));
        lo = hi;
        lo_ok = ok;
    }
    return out.items;
}

const Edge = struct { byte: usize, token: usize };

/// The candidates in `bytes` where a token of `ids` begins, while the tokens' own bytes still spell `text` exactly.
fn tokenEdges(t: model_text.Text, a: Allocator, text: []const u8, ids: []const u32, bytes: []const usize) Allocator.Error![]const Edge {
    var edges: std.ArrayList(Edge) = .empty;
    var seen: std.AutoHashMapUnmanaged(u32, ?[]const u8) = .empty; // each distinct token's bytes, decoded once
    var at: usize = 0;
    var next: usize = 0;
    for (ids, 0..) |id, i| {
        while (next < bytes.len and bytes[next] < at) next += 1;
        if (i > 0 and next < bytes.len and bytes[next] == at) try edges.append(a, .{ .byte = at, .token = i });
        if (next == bytes.len) break;
        // a fused unknown, a normalizer or a context-dependent decoder ends the proof here
        const slot = try seen.getOrPut(a, id);
        if (!slot.found_existing) slot.value_ptr.* = try t.tokenBytes(a, id);
        const own = slot.value_ptr.* orelse break;
        if (own.len == 0 or !std.mem.startsWith(u8, text[at..], own)) break;
        at += own.len;
    }
    return edges.items;
}

/// `r` from `start` to `end`, its request ranges searched from `cursor` on, since pieces come in order.
fn slice(a: Allocator, r: model_text.Rendered, cursor: *usize, start: usize, end: usize) Allocator.Error!model_text.Rendered {
    while (cursor.* < r.spans.len and r.spans[cursor.*][1] <= start) cursor.* += 1;
    var spans: std.ArrayList(model_text.Span) = .empty;
    for (r.spans[cursor.*..]) |p| {
        if (p[0] >= end) break;
        try spans.append(a, .{ @max(start, p[0]) - start, @min(end, p[1]) - start });
    }
    return .{ .text = r.text[start..end], .spans = spans.items };
}
