//! Text to audio on one GPU: frames generated in bursts, each checked burst decoded by the codec while the next runs.
const std = @import("std");
const cuda = @import("cuda");
const eng = @import("engine.zig");
const codec_ = @import("codec.zig");
const wav = @import("wav.zig");
const cfg = @import("config.zig");
const kern = @import("kernels.zig");
const common = @import("common.zig");

pub const Sink = common.Sink;
pub const StreamOptions = common.StreamOptions;
pub const Stats = common.Stats;
const ms = common.ms;

const W = cfg.frame_width;

pub const Tts = struct {
    gpa: std.mem.Allocator,
    e: *eng.Engine,
    c: *codec_.Codec,
    cs: cuda.Stream,
    audio_dev: cuda.DeviceBuffer,
    audio_host: []f32,
    max_chunk: usize,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, e: *eng.Engine, codec_path: []const u8, max_chunk: usize) !*Tts {
        const t = try gpa.create(Tts);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        t.e = e;
        t.max_chunk = @max(max_chunk, 1);
        t.c = try codec_.Codec.load(gpa, io, &e.driver, &e.k, codec_path, .{});
        errdefer t.c.destroy();
        t.cs = try cuda.Stream.init(&e.driver, true);
        t.audio_dev = try cuda.DeviceBuffer.alloc(&e.driver, t.max_chunk * codec_.samples_per_frame * 4);
        t.audio_host = try gpa.alloc(f32, t.max_chunk * codec_.samples_per_frame);
        // warm the codec (cuBLAS heuristics for its shapes): one silent chunk
        try t.c.reset(t.cs);
        var zero = try cuda.DeviceBuffer.alloc(&e.driver, 4 * W * 4);
        defer zero.free();
        try zero.fill8(0, null);
        try t.c.decode(t.cs, zero.ptr, 4, t.audio_dev.ptr);
        try t.cs.synchronize();
        return t;
    }

    pub fn destroy(t: *Tts) void {
        t.gpa.free(t.audio_host);
        t.audio_dev.free();
        t.cs.deinit();
        t.c.destroy();
        t.gpa.destroy(t);
    }

    /// Codes [frames, 10] of a voice reference: WAV bytes (any rate, resampled as torchaudio does), or samples.
    pub fn encodeWav(t: *Tts, bytes: []const u8) ![][10]i32 {
        const a = try wav.parse(t.gpa, bytes);
        defer a.deinit(t.gpa);
        return t.encodeSamples(a.samples, a.rate);
    }

    pub fn encodeSamples(t: *Tts, samples: []const f32, rate: u32) ![][10]i32 {
        const x = try wav.resample(t.gpa, samples, rate, codec_.sample_rate);
        defer t.gpa.free(x);
        return t.c.encode(t.cs, x);
    }

    /// Decode frames [n, 11] (host) to samples [n * 2048] (host), from a fresh codec state.
    pub fn decodeAll(t: *Tts, frames: []const [W]i32, out: []f32) !void {
        try t.c.reset(t.cs);
        var fb = try cuda.DeviceBuffer.alloc(&t.e.driver, @max(frames.len, 1) * W * 4);
        defer fb.free();
        try fb.upload(0, std.mem.sliceAsBytes(frames));
        var done: usize = 0;
        while (done < frames.len) {
            const n = @min(t.max_chunk, frames.len - done);
            try t.c.decode(t.cs, fb.ptr + done * W * 4, n, t.audio_dev.ptr);
            try t.audio_dev.downloadAsync(0, std.mem.sliceAsBytes(out[done * codec_.samples_per_frame ..][0 .. n * codec_.samples_per_frame]), t.cs.handle);
            try t.cs.synchronize();
            done += n;
        }
    }

    /// Decode frames_out rows [from, to) of the current request on the codec stream and hand the samples to the sink.
    fn flush(t: *Tts, from: usize, to: usize, sink: Sink) !void {
        var at = from;
        while (at < to) {
            const n = @min(t.max_chunk, to - at);
            try t.c.decode(t.cs, t.e.frames_out + at * W * 4, n, t.audio_dev.ptr);
            const out = t.audio_host[0 .. n * codec_.samples_per_frame];
            try t.audio_dev.downloadAsync(0, std.mem.sliceAsBytes(out), t.cs.handle);
            try t.cs.synchronize();
            try sink.audio(sink.ctx, out);
            at += n;
        }
    }

    /// Generate and stream: returns the frames (caller frees) and the stats.
    pub fn speak(t: *Tts, io: std.Io, rows: []const [W]i32, smp: eng.Sampling, max_frames: usize, so: StreamOptions, sink: Sink) !struct { frames: [][W]i32, stats: Stats } {
        const e = t.e;
        const t0 = eng.nowNs(io);
        try e.prefill(rows, null);
        try t.c.reset(t.cs);
        const t1 = eng.nowNs(io);
        const limit = @min(max_frames, e.o.max_frames, e.o.max_ctx - rows.len - 1);
        const host = try t.gpa.alloc([W]i32, limit);
        errdefer t.gpa.free(host);
        var made: usize = 0; // frames generated and checked
        var decoded: usize = 0; // frames handed to the codec
        var eos = false;
        var first_audio: f64 = 0;
        const chunk = @max(1, @min(so.chunk, t.max_chunk));
        var next: usize = @max(1, @min(so.first_chunk, chunk));
        while (made < limit and !eos) {
            const burst = @min(next, limit - made);
            next = @min(chunk, next + @max(1, next / 4));
            for (0..burst) |_| try e.frameStep(smp, true, null);
            // the frames checked so far decode while this burst generates
            if (decoded < made) {
                try t.flush(decoded, made, sink);
                if (decoded == 0) first_audio = ms(t0, eng.nowNs(io));
                decoded = made;
            }
            try e.download(e.frames_out + made * W * 4, std.mem.sliceAsBytes(host[made .. made + burst]));
            var good = made + burst;
            for (host[made .. made + burst], made..) |f, i| if (f[0] == e.c.ids.eos) {
                eos = true;
                good = i;
                break;
            };
            made = good;
            if (decoded == 0 and made > 0) { // the first audio as soon as the first burst is checked
                try t.flush(0, made, sink);
                first_audio = ms(t0, eng.nowNs(io));
                decoded = made;
            }
        }
        if (decoded < made) {
            try t.flush(decoded, made, sink);
            if (decoded == 0) first_audio = ms(t0, eng.nowNs(io));
        }
        const t2 = eng.nowNs(io);
        return .{ .frames = host[0..made], .stats = .{
            .frames = made,
            .eos = eos,
            .prompt_rows = rows.len,
            .prefill_ms = ms(t0, t1),
            .first_audio_ms = first_audio,
            .total_ms = ms(t0, t2),
            .audio_s = @as(f64, @floatFromInt(made * codec_.samples_per_frame)) / codec_.sample_rate,
        } };
    }
};

/// The codec alone, without the backbone: voice references to codes, and frames to audio for --codec-check.
pub const CodecOnly = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: *kern.Kernels,
    c: *codec_.Codec,
    bufs: std.ArrayList(cuda.DeviceBuffer),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !*CodecOnly {
        const o = try gpa.create(CodecOnly);
        errdefer gpa.destroy(o);
        o.gpa = gpa;
        o.bufs = .empty;
        o.driver = try cuda.Driver.open();
        o.ctx = try cuda.Context.init(&o.driver, 0);
        o.stream = try cuda.Stream.init(&o.driver, true);
        o.k = try gpa.create(kern.Kernels);
        o.k.* = try kern.Kernels.load(&o.driver, &o.ctx);
        o.c = try codec_.Codec.load(gpa, io, &o.driver, o.k, path, .{});
        return o;
    }

    pub fn close(o: *CodecOnly) void {
        for (o.bufs.items) |*b| b.free();
        o.bufs.deinit(o.gpa);
        o.c.destroy();
        o.k.unload();
        o.gpa.destroy(o.k);
        o.stream.deinit();
        o.ctx.deinit();
        o.driver.close();
        o.gpa.destroy(o);
    }

    pub fn buffer(o: *CodecOnly, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(&o.driver, @max(bytes, 256));
        try o.bufs.append(o.gpa, b);
        return b.ptr;
    }

    pub fn upload(o: *CodecOnly, dst: u64, bytes: []const u8) !void {
        var b: cuda.DeviceBuffer = .{ .d = &o.driver, .ptr = dst, .len = bytes.len, .borrowed = true };
        try b.upload(0, bytes);
    }

    /// Waits for the stream, then copies device bytes at `src` out.
    pub fn download(o: *CodecOnly, src: u64, out: []u8) !void {
        try o.stream.synchronize();
        var b: cuda.DeviceBuffer = .{ .d = &o.driver, .ptr = src, .len = out.len, .borrowed = true };
        try b.download(0, out);
    }

    pub fn sync(o: *CodecOnly) !void {
        try o.stream.synchronize();
    }

    pub fn reset(o: *CodecOnly) !void {
        try o.c.reset(o.stream);
    }

    /// Enqueue the decode of `nf` device frames [nf, 11] into nf * 2048 fp32 samples at `audio`.
    pub fn decode(o: *CodecOnly, frames: u64, nf: usize, audio: u64) !void {
        try o.c.decode(o.stream, frames, nf, audio);
    }

    pub fn encode(o: *CodecOnly, x44: []const f32) ![][10]i32 {
        return o.c.encode(o.stream, x44);
    }

    pub fn weightBytes(o: *const CodecOnly) usize {
        return o.c.weight_bytes;
    }

    pub fn setProbe(o: *CodecOnly, p: ?codec_.Probe) void {
        o.c.probe = p;
    }
};
