//! A prompt's image rows: where each image's span sits, its aligner rows, and what each prompt row of a chunk takes.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const vision = @import("vision.zig");
const prep = @import("vision_prep.zig");
const vt = @import("vision_tower.zig");

pub const Image = host.lanes.stream.Image;
const Ref = vision.Ref;

/// One image's span in the prompt: positions [at, at + tokens), its aligner grid, and its first row in the features.
pub const Span = struct { at: u32, tokens: u32, lh: u32, lw: u32, feat: u32 };

pub const Rows = struct {
    a: std.mem.Allocator,
    pipes: *const vision.Pipelines,
    spans: []Span,
    /// the aligner rows of every image the prompt still has to run: bf16 [rows][dim], image after image
    feat: mtl.Buffer,
    delims: Ref,
    dim: u32,
    /// the embedding row an image position takes before its rows are merged in
    embed_id: u32,
    /// the chunk's merge sources, int32 [chunk]
    src: mtl.Buffer,
    chunk: u32,

    /// The spans of `images` (sorted, apart, as long as their grids say) in a prompt of `n` ids.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, pipes: *const vision.Pipelines, cfg: vision.Config, delims: Ref, images: []const Image, n: u32, chunk: u32, feat_rows: u32) !Rows {
        const spans = try a.alloc(Span, images.len);
        errdefer a.free(spans);
        var end: u32 = 0;
        var row: u32 = 0;
        for (images, spans) |im, *s| {
            const g = prep.llmGrid(im.gh, im.gw, cfg.ratio);
            if (im.tokens != prep.spanTokens(g[0], g[1]) or im.at < end or @as(u64, im.at) + im.tokens > n) return error.BadImageSpan;
            if (im.tokens > chunk) return error.ImageSpanTooLong;
            s.* = .{ .at = im.at, .tokens = im.tokens, .lh = g[0], .lw = g[1], .feat = row };
            end = im.at + im.tokens;
            row += g[0] * g[1];
        }
        const feat = try device.buffer(@max(@as(usize, feat_rows) * cfg.out_dim * 2, 256), mtl.ResourceOptions.shared);
        errdefer feat.deinit();
        const src = try device.buffer(@max(@as(usize, chunk) * 4, 256), mtl.ResourceOptions.shared);
        return .{ .a = a, .pipes = pipes, .spans = spans, .feat = feat, .delims = delims, .dim = cfg.out_dim, .embed_id = cfg.image_token_id, .src = src, .chunk = chunk };
    }

    pub fn deinit(r: *Rows) void {
        r.a.free(r.spans);
        r.feat.deinit();
        r.src.deinit();
    }

    /// The rows of the images reaching past position `from` (the others are in the state), run through `tower`.
    pub fn build(a: std.mem.Allocator, device: mtl.Device, tower: *vt.Tower, images: []const Image, n: u32, chunk: u32, from: u32) !Rows {
        const cfg = tower.w.cfg;
        var run: std.ArrayList(Image) = .empty;
        defer run.deinit(a);
        var feat_rows: u32 = 0;
        for (images) |im| if (im.at + im.tokens > from) {
            try run.append(a, im);
            const g = prep.llmGrid(im.gh, im.gw, cfg.ratio);
            feat_rows += g[0] * g[1];
        };
        var rows = try init(a, device, &tower.kit.v, cfg, tower.w.delims, run.items, n, chunk, feat_rows);
        errdefer rows.deinit();
        const dim: usize = cfg.out_dim;
        for (run.items, rows.spans) |im, sp| {
            try tower.run(im.pixels, im.gh, im.gw, .{});
            const out = tower.output();
            const bytes = @as(usize, sp.lh) * sp.lw * dim * 2;
            @memcpy(rows.feat.contents()[@as(usize, sp.feat) * dim * 2 ..][0..bytes], (out.buf.contents() + out.off)[0..bytes]);
        }
        return rows;
    }

    /// The aligner rows of every span, in the order the spans were given.
    pub fn totalRows(spans: []const Span) u32 {
        var n: u32 = 0;
        for (spans) |s| n += s.lh * s.lw;
        return n;
    }

    fn spanOf(r: *const Rows, pos: u32) ?*const Span {
        for (r.spans) |*s| if (pos >= s.at and pos < s.at + s.tokens) return s;
        return null;
    }

    /// Whether prompt position `pos` is an image position.
    pub fn isImage(r: *const Rows, pos: u32) bool {
        return r.spanOf(pos) != null;
    }

    /// The end of the chunk [a, b): pulled back to a span's start, or pushed past it, when it would cut one.
    pub fn chunkEnd(r: *const Rows, a: u32, b: u32) !u32 {
        if (r.spanOf(a)) |s| if (s.at != a) return error.ImageSpanSplit;
        const s = r.spanOf(b) orelse return b;
        if (s.at == b) return b;
        if (s.at > a) return s.at;
        if (s.at + s.tokens > a + r.chunk) return error.ImageSpanTooLong;
        return s.at + s.tokens;
    }

    /// For rows [a, b): the merge sources into `src`, and the image flags (1 on an image position) into `flags`.
    pub fn describe(r: *const Rows, a: u32, b: u32, flags: []u8) void {
        const src: []i32 = r.src.slice(i32, b - a);
        for (a..b) |q| {
            const i = q - a;
            const s = r.spanOf(@intCast(q)) orelse {
                src[i] = -4;
                flags[i] = 0;
                continue;
            };
            flags[i] = 1;
            const k = prep.kindAt(@as(u32, @intCast(q)) - s.at, s.lh, s.lw);
            src[i] = switch (k[0]) {
                .start => -1,
                .end => -2,
                .newline => -3,
                .row => @intCast(s.feat + k[1]),
            };
        }
    }

    /// The merge of rows [a, b) into the streams' first buffer (after the embedding, before the first layer).
    pub fn merge(r: *const Rows, e: mtl.ComputeEncoder, streams: Ref, a: u32, b: u32) void {
        vision.merge(r.pipes, e, .{ .buf = r.feat }, r.delims, .{ .buf = r.src }, streams, b - a, r.dim);
    }

    /// The ids the embedding reads for rows of `ids`: image positions take the image token's row.
    pub fn embedId(r: *const Rows, id: u32) u32 {
        return if (prep.isVid(id)) r.embed_id else id;
    }
};

test "a span's chunk end: pulled back to the start, or pushed past the span" {
    var r: Rows = undefined;
    var spans = [_]Span{.{ .at = 10, .tokens = 6, .lh = 1, .lw = 3, .feat = 0 }};
    r.spans = &spans;
    r.chunk = 8;
    try std.testing.expectEqual(@as(u32, 10), try r.chunkEnd(0, 12));
    try std.testing.expectEqual(@as(u32, 9), try r.chunkEnd(0, 9));
    try std.testing.expectEqual(@as(u32, 16), try r.chunkEnd(10, 12));
    try std.testing.expectEqual(@as(u32, 20), try r.chunkEnd(16, 20));
    try std.testing.expectError(error.ImageSpanSplit, r.chunkEnd(12, 14));
}
