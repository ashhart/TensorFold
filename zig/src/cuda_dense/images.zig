//! A request's images on the dense engine: spans checked, encoded into the lane's rows at its begin, cut by chunk.
const std = @import("std");
const cuda = @import("cuda");
const recipe = @import("recipe.zig");
const Inject = @import("pass.zig").Inject;

const Image = recipe.Image;

/// One image's rows: prompt rows [first, first + rows), its features from row `at` of the lane's buffer.
pub const Span = struct { first: usize, rows: usize, at: usize };

/// A lane's images: their spans, the lane's feature buffer, and the injects of the chunk running now.
pub const Held = struct {
    spans: []Span = &.{},
    local: []Inject = &.{},
    base: u64 = 0, // the lane's feature rows: [image_tokens, hidden] bf16
    hidden: usize = 0,

    /// Images ascending and disjoint inside the prompt, every row of each `mark`, their tokens within `budget`.
    pub fn check(images: []const Image, prompt: []const u32, mark: u32, budget: usize) error{DenseImageSpan}!void {
        var end: usize = 0;
        var total: usize = 0;
        for (images) |im| {
            const first: usize = im.at;
            if (im.tokens == 0 or first < end or first >= prompt.len or im.tokens > prompt.len - first) return error.DenseImageSpan;
            end = first + im.tokens;
            total += im.tokens;
            if (total > budget) return error.DenseImageSpan;
            for (prompt[first..end]) |id| if (id != mark) return error.DenseImageSpan;
        }
    }

    /// The spans of checked `images`: each image's feature rows after the ones before it in the buffer at `base`.
    pub fn init(gpa: std.mem.Allocator, images: []const Image, base: u64, hidden: usize) !Held {
        const spans = try gpa.alloc(Span, images.len);
        errdefer gpa.free(spans);
        const local = try gpa.alloc(Inject, images.len);
        var at: usize = 0;
        for (images, spans) |im, *sp| {
            sp.* = .{ .first = im.at, .rows = im.tokens, .at = at };
            at += im.tokens;
        }
        return .{ .spans = spans, .local = local, .base = base, .hidden = hidden };
    }

    pub fn deinit(h: *Held, gpa: std.mem.Allocator) void {
        gpa.free(h.spans);
        gpa.free(h.local);
        h.* = .{};
    }

    /// Bytes from `base` to span `sp`'s feature row `row`.
    fn offset(h: *const Held, sp: Span, row: usize) u64 {
        return (sp.at + row) * h.hidden * 2;
    }

    /// Each image of `src` through tower `t` in order on `s`, one call an image, a cancel check before each.
    pub fn encode(h: *const Held, comptime T: type, t: *T, s: cuda.Stream, src: anytype) !void {
        if (src.images.len != h.spans.len) return error.DenseImageSpan;
        for (src.images, h.spans) |*im, sp| {
            if (src.isCancelled()) return error.Cancelled;
            try T.encode(t, s, im, h.base + h.offset(sp, 0), null);
        }
    }

    /// The injects of chunk rows [at, end): each image's rows inside it, its source offset into that image's rows.
    pub fn intersect(h: *Held, at: usize, end: usize) []const Inject {
        var n: usize = 0;
        for (h.spans) |sp| {
            const first = @max(at, sp.first);
            const last = @min(end, sp.first + sp.rows);
            if (last <= first) continue;
            h.local[n] = .{ .row = first - at, .rows = last - first, .src = h.base + h.offset(sp, first - sp.first) };
            n += 1;
        }
        return h.local[0..n];
    }
};
