//! Transcribe's audio on CUDA: a prompt pass's clips through the audio encoder into rows, spliced as image rows are.
const std = @import("std");
const cuda = @import("cuda");
const radio = @import("radio");
const nemo_audio = @import("nemo_audio");

/// A clip in a prompt: its samples (16 kHz mono) and the prompt position of its first `<|audio|>` row.
pub const Clip = struct { samples: []const f32, at: u32 };

/// A WAV file as the encoder reads it: its samples, the rows it takes, a hash of the samples for the prompt cache.
pub const Prepared = struct {
    samples: []f32,
    rows: u32,
    hash: u64,

    pub fn deinit(p: *Prepared, a: std.mem.Allocator) void {
        a.free(p.samples);
    }
};

/// The rows a request's audio may take in all: an hour of it.
pub fn limit() u32 {
    return nemo_audio.Config.limit();
}

/// A WAV file's samples and rows; error.AudioTooShort or error.AudioTooLong past what the encoder takes.
pub fn prepare(a: std.mem.Allocator, data: []const u8) !Prepared {
    const samples = try nemo_audio.wav.decode(a, data);
    errdefer a.free(samples);
    if (samples.len / nemo_audio.Config.hop < 2) return error.AudioTooShort;
    if (samples.len > nemo_audio.Config.max_seconds * nemo_audio.Config.rate) return error.AudioTooLong;
    const hash = std.hash.Wyhash.hash(0x6e656d6f61756469, std.mem.sliceAsBytes(samples));
    return .{ .samples = samples, .rows = @intCast(nemo_audio.Config.rows(samples.len)), .hash = hash };
}

/// The clips a prompt pass from `from` reads, in prompt order, their rows back to back from row 0.
pub fn plan(a: std.mem.Allocator, clips: []const Clip, from: u32) ![]radio.Span {
    var out: std.ArrayList(radio.Span) = .empty;
    errdefer out.deinit(a);
    var row: u32 = 0;
    var end: u32 = 0;
    for (clips, 0..) |c, i| {
        const n: u32 = @intCast(nemo_audio.Config.rows(c.samples.len));
        if (c.at < end or n == 0) return error.AudioSpans; // in order, apart, each with rows
        end = std.math.add(u32, c.at, n) catch return error.AudioSpans;
        if (end <= from) continue; // a restored prefix holds these rows' states already
        try out.append(a, .{ .at = c.at, .n = n, .row = row, .image = @intCast(i) });
        row = std.math.add(u32, row, n) catch return error.AudioSpans;
    }
    return out.toOwnedSlice(a);
}

pub const Audio = struct {
    gpa: std.mem.Allocator,
    rows: radio.Rows,
    spans: []radio.Span = &.{}, // the prompt pass in progress: each clip's rows from its first <|audio|> position
    encoder: ?*nemo_audio.Encoder = null, // null: rows come from the host (`set`)

    /// A rows buffer of `limit` rows of the language model's width.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rows_limit: u32, hidden: usize) !*Audio {
        const a = try gpa.create(Audio);
        errdefer gpa.destroy(a);
        a.* = .{ .gpa = gpa, .rows = try radio.Rows.init(d, rows_limit, hidden) };
        return a;
    }

    /// The checkpoint's audio encoder and a rows buffer for an hour of audio in the language model's width.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, hidden: usize) !*Audio {
        const enc = try gpa.create(nemo_audio.Encoder);
        errdefer gpa.destroy(enc);
        enc.* = try nemo_audio.Encoder.init(gpa, io, d, dir);
        errdefer enc.deinit();
        if (enc.c.output != hidden) return error.AudioWidth;
        const a = try init(gpa, d, limit(), hidden);
        a.encoder = enc;
        return a;
    }

    pub fn deinit(a: *Audio) void {
        a.end();
        if (a.encoder) |e| {
            e.deinit();
            a.gpa.destroy(e);
        }
        a.rows.deinit();
        a.gpa.destroy(a);
    }

    /// Device bytes the encoder and the rows buffer hold.
    pub fn bytes(a: *const Audio) u64 {
        return a.rows.buf.len + if (a.encoder) |e| e.bytes else 0;
    }

    /// A prompt pass from row `from`: the clips whose rows end past it, through the encoder into the rows buffer.
    pub fn begin(a: *Audio, s: cuda.Stream, clips: []const Clip, from: usize) !void {
        a.end();
        if (clips.len == 0) return;
        const enc = a.encoder orelse return error.NoAudioEncoder;
        a.spans = try plan(a.gpa, clips, @intCast(from));
        errdefer a.end();
        if (radio.rows.total(a.spans) > a.rows.limit) return error.AudioRows;
        for (a.spans) |sp| {
            const n = try enc.encode(s, clips[sp.image].samples, a.rows.at(sp), null);
            if (n != sp.n) return error.AudioRows;
        }
    }

    /// The pass ended: no rows splice into later passes until another begins.
    pub fn end(a: *Audio) void {
        if (a.spans.len > 0) a.gpa.free(a.spans);
        a.spans = &.{};
    }

    /// One clip's bf16 rows (`rows.hidden` each) from host memory, standing at prompt positions [at, at + n).
    pub fn set(a: *Audio, s: cuda.Stream, at: u32, bf16: []const u8) !void {
        const w = a.rows.hidden * 2;
        if (bf16.len == 0 or bf16.len % w != 0) return error.AudioRows;
        const n: u32 = @intCast(bf16.len / w);
        if (n > a.rows.limit) return error.AudioRows;
        try a.rows.d.check(a.rows.d.api.cuMemcpyHtoDAsync_v2(a.rows.buf.ptr, bf16.ptr, bf16.len, s.handle), "cuMemcpyHtoDAsync");
        a.end();
        a.spans = try a.gpa.dupe(radio.Span, &.{.{ .at = at, .n = n, .row = 0, .image = 0 }});
    }

    /// Prompt rows [lo, hi) embedded at `emb` take the audio rows they cross.
    pub fn splice(a: *const Audio, s: cuda.Stream, lo: usize, hi: usize, emb: u64) !void {
        if (a.spans.len == 0) return;
        try a.rows.splice(s, a.spans, @intCast(lo), @intCast(hi), emb);
    }
};

test "audio plan: clips in prompt order, rows back to back, a restored prefix's clip left out" {
    const a = std.testing.allocator;
    const one: [30560]f32 = @splat(0); // 24 rows
    const two: [56080]f32 = @splat(0); // 44 rows
    const clips = [_]Clip{ .{ .samples = &one, .at = 43 }, .{ .samples = &two, .at = 80 } };
    const all = try plan(a, &clips, 0);
    defer a.free(all);
    try std.testing.expectEqual(@as(usize, 2), all.len);
    try std.testing.expectEqual(radio.Span{ .at = 80, .n = 44, .row = 24, .image = 1 }, all[1]);
    const later = try plan(a, &clips, 67);
    defer a.free(later);
    try std.testing.expectEqual(radio.Span{ .at = 80, .n = 44, .row = 0, .image = 1 }, later[0]);
    const overlapping = [_]Clip{ .{ .samples = &one, .at = 43 }, .{ .samples = &two, .at = 50 } };
    try std.testing.expectError(error.AudioSpans, plan(a, &overlapping, 0));
}

test "audio limit: an hour of audio's rows" {
    try std.testing.expectEqual(@as(u32, 45000), limit());
}

test "audio begin: a pass refused after planning leaves no spans for the next one" {
    var enc: nemo_audio.Encoder = undefined;
    var a: Audio = .{ .gpa = std.testing.allocator, .rows = .{ .d = undefined, .buf = undefined, .hidden = 8, .limit = 10 }, .encoder = &enc };
    const one: [30560]f32 = @splat(0); // 24 rows, over the limit of 10
    const clips = [_]Clip{.{ .samples = &one, .at = 43 }};
    try std.testing.expectError(error.AudioRows, a.begin(undefined, &clips, 0));
    try std.testing.expectEqual(@as(usize, 0), a.spans.len);
}
