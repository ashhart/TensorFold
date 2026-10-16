//! Text to audio on Metal: frames generated in bursts, each checked burst decoded by the codec after it.
const std = @import("std");
const cfg = @import("../config.zig");
const common = @import("../common.zig");
const wav = @import("../wav.zig");
const gpu = @import("gpu.zig");
const eng = @import("engine.zig");
const codec_ = @import("codec.zig");

pub const Sink = common.Sink;
pub const StreamOptions = common.StreamOptions;
pub const Stats = common.Stats;
const ms = common.ms;
const W = cfg.frame_width;
const spf = codec_.samples_per_frame;

pub const Tts = struct {
    gpa: std.mem.Allocator,
    e: *eng.Engine,
    c: *codec_.Codec,
    audio: u64,
    max_chunk: usize,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, e: *eng.Engine, codec_path: []const u8, max_chunk: usize) !*Tts {
        const t = try gpa.create(Tts);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        t.e = e;
        t.max_chunk = @max(max_chunk, 1);
        t.c = try codec_.Codec.load(gpa, io, e.g, codec_path, .{});
        errdefer t.c.destroy();
        t.audio = try e.g.alloc(t.max_chunk * spf * 4);
        // warm the codec's pipelines: one silent chunk
        try t.c.reset();
        const zero = try e.g.alloc(4 * W * 4);
        try t.c.decode(zero, 4, t.audio);
        try e.flush();
        return t;
    }

    pub fn destroy(t: *Tts) void {
        t.c.destroy();
        t.gpa.destroy(t);
    }

    /// Codes [frames, 10] of a voice reference: WAV bytes (any rate, resampled as torchaudio does).
    pub fn encodeWav(t: *Tts, bytes: []const u8) ![][10]i32 {
        const a = try wav.parse(t.gpa, bytes);
        defer a.deinit(t.gpa);
        const x = try wav.resample(t.gpa, a.samples, a.rate, codec_.sample_rate);
        defer t.gpa.free(x);
        return t.c.encode(x);
    }

    /// Decode frames_out rows [from, to) of the current request and hand the samples to the sink.
    fn flush(t: *Tts, from: usize, to: usize, sink: Sink) !void {
        var at = from;
        while (at < to) {
            const n = @min(t.max_chunk, to - at);
            try t.c.decode(t.e.frames_out + at * W * 4, n, t.audio);
            try t.e.flush();
            try sink.audio(sink.ctx, std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(t.e.g.host(t.audio, n * spf * 4)))));
            at += n;
        }
    }

    /// Generate and stream: returns the frames (caller frees) and the stats.
    pub fn speak(t: *Tts, io: std.Io, rows: []const [W]i32, smp: eng.Sampling, max_frames: usize, so: StreamOptions, sink: Sink) !struct { frames: [][W]i32, stats: Stats } {
        const e = t.e;
        const t0 = eng.nowNs(io);
        try e.prefill(rows, null);
        try t.c.reset();
        try e.flush();
        const t1 = eng.nowNs(io);
        const limit = @min(max_frames, e.o.max_frames, e.o.max_ctx - rows.len - 1);
        const host = try t.gpa.alloc([W]i32, limit);
        errdefer t.gpa.free(host);
        var made: usize = 0;
        var eos = false;
        var first_audio: f64 = 0;
        const chunk = @max(1, @min(so.chunk, t.max_chunk));
        var next: usize = @max(1, @min(so.first_chunk, chunk));
        while (made < limit and !eos) {
            const burst = @min(next, limit - made);
            next = @min(chunk, next + @max(1, next / 4));
            for (0..burst) |_| try e.frameStep(smp, true, null);
            try e.download(e.frames_out + made * W * 4, std.mem.sliceAsBytes(host[made .. made + burst]));
            var good = made + burst;
            for (host[made .. made + burst], made..) |f, i| if (f[0] == e.c.ids.eos) {
                eos = true;
                good = i;
                break;
            };
            if (good > made) {
                try t.flush(made, good, sink);
                if (made == 0) first_audio = ms(t0, eng.nowNs(io));
            }
            made = good;
        }
        const t2 = eng.nowNs(io);
        return .{ .frames = host[0..made], .stats = .{
            .frames = made,
            .eos = eos,
            .prompt_rows = rows.len,
            .prefill_ms = ms(t0, t1),
            .first_audio_ms = first_audio,
            .total_ms = ms(t0, t2),
            .audio_s = @as(f64, @floatFromInt(made * spf)) / codec_.sample_rate,
        } };
    }
};

/// The codec alone, without the backbone: voice references to codes, and frames to audio for --codec-check.
pub const CodecOnly = struct {
    gpa: std.mem.Allocator,
    g: *gpu.Gpu,
    c: *codec_.Codec,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !*CodecOnly {
        const o = try gpa.create(CodecOnly);
        errdefer gpa.destroy(o);
        o.gpa = gpa;
        o.g = try gpu.Gpu.open(gpa, @import("drama3_metallib").bytes);
        errdefer o.g.close();
        o.c = try codec_.Codec.load(gpa, io, o.g, path, .{});
        return o;
    }

    pub fn close(o: *CodecOnly) void {
        o.c.destroy();
        o.g.close();
        o.gpa.destroy(o);
    }

    pub fn buffer(o: *CodecOnly, bytes: usize) !u64 {
        return o.g.alloc(bytes);
    }

    pub fn upload(o: *CodecOnly, dst: u64, bytes: []const u8) !void {
        try o.g.put(dst, bytes);
    }

    /// Waits for the GPU, then copies the bytes at `src` out.
    pub fn download(o: *CodecOnly, src: u64, out: []u8) !void {
        try o.g.get(src, out);
    }

    pub fn sync(o: *CodecOnly) !void {
        try o.g.flush();
    }

    pub fn reset(o: *CodecOnly) !void {
        try o.c.reset();
    }

    pub fn decode(o: *CodecOnly, frames: u64, nf: usize, audio: u64) !void {
        try o.c.decode(frames, nf, audio);
    }

    pub fn encode(o: *CodecOnly, x44: []const f32) ![][10]i32 {
        return o.c.encode(x44);
    }

    pub fn weightBytes(o: *const CodecOnly) usize {
        return o.c.weight_bytes;
    }

    pub fn setProbe(o: *CodecOnly, p: ?codec_.Probe) void {
        o.c.probe = p;
    }
};
