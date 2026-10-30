//! Transcribe's audio on CUDA: the encoder (nemo_audio) and a prompt pass's audio rows, spliced as image rows are.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const radio = @import("radio");
const nemo_audio = @import("nemo_audio");

/// Audio rows a request may hold at most: the card's hour of 16 kHz audio (12.5 rows a second).
pub const hour_rows: u32 = @intCast(nemo_audio.rows(3600 * nemo_audio.Config.sample_rate));

/// The rows buffer for a `context`-token window: an hour of audio, or the window when it holds less.
pub fn limitFor(context: usize) u32 {
    return @intCast(@min(context, hour_rows));
}
/// Clips a request may hold.
pub const max_clips = 8;

pub const Audio = struct {
    gpa: std.mem.Allocator,
    rows: radio.Rows,
    encoder: ?*nemo_audio.Encoder = null, // null: rows set from the host (a check's reference rows)
    spans: []radio.Span = &.{}, // the prompt pass in progress: each clip's rows from its first <|audio|> position

    /// A buffer of `limit` rows of the language model's width, and the encoder of the checkpoint at `dir` if given.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: ?[]const u8, limit: u32, hidden: usize) !*Audio {
        const a = try gpa.create(Audio);
        errdefer gpa.destroy(a);
        a.* = .{ .gpa = gpa, .rows = try radio.Rows.init(d, limit, hidden) };
        errdefer a.rows.deinit();
        if (dir) |path| {
            a.encoder = try nemo_audio.Encoder.init(gpa, io, d, path);
            if (a.encoder.?.c.text_hidden != hidden) return error.AudioRowShape;
        }
        return a;
    }

    pub fn deinit(a: *Audio) void {
        a.gpa.free(a.spans);
        if (a.encoder) |x| x.deinit();
        a.rows.deinit();
        a.gpa.destroy(a);
    }

    /// Device bytes the encoder and the rows buffer hold.
    pub fn bytes(a: *const Audio) u64 {
        return a.rows.buf.len + if (a.encoder) |x| x.bytes() else 0;
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

    /// A prompt pass from row `from`: the clips whose rows end past it, through the encoder into the rows buffer.
    pub fn begin(a: *Audio, s: cuda.Stream, clips: []const lanes.stream.Audio, from: usize) !void {
        a.end();
        const enc = a.encoder orelse return error.NoAudioEncoder;
        var spans: std.ArrayList(radio.Span) = .empty;
        errdefer spans.deinit(a.gpa);
        var row: u32 = 0;
        for (clips, 0..) |c, i| {
            if (c.at + c.tokens <= from) continue; // a kept prefix holds it already
            if (row + c.tokens > a.rows.limit) return error.AudioRows;
            const sp: radio.Span = .{ .at = c.at, .n = c.tokens, .row = row, .image = @intCast(i) };
            if (try enc.encode(s, c.samples, a.rows.at(sp), null) != c.tokens) return error.AudioRows;
            try spans.append(a.gpa, sp);
            row += c.tokens;
        }
        a.spans = try spans.toOwnedSlice(a.gpa);
    }

    /// The pass ended: no rows splice into later passes until another begins.
    pub fn end(a: *Audio) void {
        a.gpa.free(a.spans);
        a.spans = &.{};
    }

    /// Prompt rows [lo, hi) embedded at `emb` take the audio rows they cross.
    pub fn splice(a: *const Audio, s: cuda.Stream, lo: usize, hi: usize, emb: u64) !void {
        if (a.spans.len == 0) return;
        try a.rows.splice(s, a.spans, @intCast(lo), @intCast(hi), emb);
    }

    /// A WAV clip as the encoder reads it: its samples, its rows and a hash of its samples (the prompt cache's key).
    pub fn prepare(a: *const Audio, alloc: std.mem.Allocator, wav: []const u8, max_rows: u32) !struct { samples: []f32, tokens: u32, hash: u64 } {
        _ = a;
        const samples = try nemo_audio.mel.decodeWav(alloc, wav);
        errdefer alloc.free(samples);
        if (samples.len < nemo_audio.Config.hop) return error.AudioTooShort;
        const n = nemo_audio.rows(samples.len);
        if (n > max_rows) return error.AudioTooLong;
        return .{ .samples = samples, .tokens = @intCast(n), .hash = std.hash.Wyhash.hash(0x6175_6469_6f, std.mem.sliceAsBytes(samples)) };
    }
};
