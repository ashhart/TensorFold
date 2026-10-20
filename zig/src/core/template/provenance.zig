//! Request byte ranges travel with strings, independently of their contents.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Half-open byte offsets, sorted and disjoint. Adjacent ranges may remain separate.
pub const Span = [2]usize;

pub const Str = struct {
    s: []const u8,
    safe: bool = false,
    spans: []const Span = &.{},

    pub fn request(a: Allocator, s: []const u8) Allocator.Error!Str {
        return derived(a, s, s.len > 0);
    }

    pub fn derived(a: Allocator, s: []const u8, request_bytes: bool) Allocator.Error!Str {
        if (!request_bytes or s.len == 0) return .{ .s = s };
        const spans = try a.alloc(Span, 1);
        spans[0] = .{ 0, s.len };
        return .{ .s = s, .spans = spans };
    }

    pub fn overlaps(s: Str, start: usize, end: usize) bool {
        if (start == end) return false;
        var lo: usize = 0;
        var hi = s.spans.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (s.spans[mid][1] <= start) lo = mid + 1 else hi = mid;
        }
        for (s.spans[lo..]) |p| {
            if (p[0] >= end) break;
            if (p[1] > start) return true;
        }
        return false;
    }

    pub fn slice(s: Str, a: Allocator, start: usize, end: usize) Allocator.Error!Str {
        std.debug.assert(start <= end and end <= s.s.len);
        if (start == end) return .{ .s = s.s[start..end], .safe = s.safe };
        var spans: std.ArrayList(Span) = .empty;
        var lo: usize = 0;
        var hi = s.spans.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (s.spans[mid][1] <= start) lo = mid + 1 else hi = mid;
        }
        for (s.spans[lo..]) |p| {
            if (p[0] >= end) break;
            if (p[1] > start) try spans.append(a, .{ @max(start, p[0]) - start, @min(end, p[1]) - start });
        }
        return .{ .s = s.s[start..end], .safe = s.safe, .spans = spans.items };
    }

    /// A borrowed substring of this string, including an empty substring at its end.
    pub fn part(s: Str, a: Allocator, bytes: []const u8) Allocator.Error!Str {
        const start = @intFromPtr(bytes.ptr) - @intFromPtr(s.s.ptr);
        return s.slice(a, start, start + bytes.len);
    }
};

/// Monotone substring operations visit each source range once, even for fragmented request strings.
pub const Cursor = struct {
    source: Str,
    index: usize = 0,

    pub fn part(c: *Cursor, a: Allocator, bytes: []const u8) Allocator.Error!Str {
        const start = @intFromPtr(bytes.ptr) - @intFromPtr(c.source.s.ptr);
        const end = start + bytes.len;
        if (start == end) return .{ .s = bytes, .safe = c.source.safe };
        while (c.index < c.source.spans.len and c.source.spans[c.index][1] <= start) c.index += 1;
        var spans: std.ArrayList(Span) = .empty;
        for (c.source.spans[c.index..]) |p| {
            if (p[0] >= end) break;
            if (p[1] > start) try spans.append(a, .{ @max(start, p[0]) - start, @min(end, p[1]) - start });
        }
        return .{ .s = bytes, .safe = c.source.safe, .spans = spans.items };
    }
};

pub const Buffer = struct {
    text: std.ArrayList(u8) = .empty,
    spans: std.ArrayList(Span) = .empty,
    span_steps: usize = 0,

    pub fn literal(b: *Buffer, a: Allocator, s: []const u8) Allocator.Error!void {
        try b.text.appendSlice(a, s);
    }

    pub fn append(b: *Buffer, a: Allocator, s: Str) Allocator.Error!void {
        const offset = b.text.items.len;
        try b.text.appendSlice(a, s.s);
        for (s.spans) |p| {
            // Never mutate an earlier range: an accumulator's prior value may still refer to it.
            try b.spans.append(a, .{ offset + p[0], offset + p[1] });
            b.span_steps += 1;
        }
    }

    pub fn str(b: *const Buffer) Str {
        return .{ .s = b.text.items, .spans = b.spans.items };
    }
};
