//! Transcribe's audio on CUDA: a prompt pass's audio rows, spliced after each embed as image rows are.
const std = @import("std");
const cuda = @import("cuda");
const radio = @import("radio");

pub const Audio = struct {
    gpa: std.mem.Allocator,
    rows: radio.Rows,
    spans: []radio.Span = &.{}, // the prompt pass in progress: each clip's rows from its first <|audio|> position

    /// A rows buffer of `limit` rows of the language model's width.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, limit: u32, hidden: usize) !*Audio {
        const a = try gpa.create(Audio);
        errdefer gpa.destroy(a);
        a.* = .{ .gpa = gpa, .rows = try radio.Rows.init(d, limit, hidden) };
        return a;
    }

    pub fn deinit(a: *Audio) void {
        a.gpa.free(a.spans);
        a.rows.deinit();
        a.gpa.destroy(a);
    }

    /// One clip's bf16 rows (`rows.hidden` each) from host memory, standing at prompt positions [at, at + n).
    pub fn set(a: *Audio, s: cuda.Stream, at: u32, bf16: []const u8) !void {
        const w = a.rows.hidden * 2;
        if (bf16.len == 0 or bf16.len % w != 0) return error.AudioRows;
        const n: u32 = @intCast(bf16.len / w);
        if (n > a.rows.limit) return error.AudioRows;
        try a.rows.d.check(a.rows.d.api.cuMemcpyHtoDAsync_v2(a.rows.buf.ptr, bf16.ptr, bf16.len, s.handle), "cuMemcpyHtoDAsync");
        a.gpa.free(a.spans);
        a.spans = try a.gpa.dupe(radio.Span, &.{.{ .at = at, .n = n, .row = 0, .image = 0 }});
    }

    /// Prompt rows [lo, hi) embedded at `emb` take the audio rows they cross.
    pub fn splice(a: *const Audio, s: cuda.Stream, lo: usize, hi: usize, emb: u64) !void {
        if (a.spans.len == 0) return;
        try a.rows.splice(s, a.spans, @intCast(lo), @intCast(hi), emb);
    }
};
