//! ZONOS3's codec decoder on one GPU: code frames [n][10] in any chunks (the trunk keeps its state), 44.1 kHz fp32 out.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("codec_kernels.zig");
const wts = @import("codec_weights.zig");
const stage_layer = @import("codec_layers.zig");

const Buf = cuda.DeviceBuffer;
const W = wts.width;

pub const sample_rate = 44100;
pub const frame_samples = 2940; // output samples a code frame
const internal_frame = 2944; // 44160 Hz samples a code frame
const head_rate = 16; // head frames a code frame

pub const Options = struct {
    max_frames: usize = 2048, // code frames a request
    pass_frames: usize = 32, // code frames a pass through the trunk

    pub fn validate(o: Options) !void {
        if (o.max_frames == 0 or o.max_frames > 8192) return error.BadFrameBudget;
        if (o.pass_frames == 0 or o.pass_frames > o.max_frames) return error.BadPassFrames;
    }
};

/// A capture of a pass under the reference's dump names: device fp32 rows [rows, width].
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void,
};

pub const Codec = struct {
    gpa: std.mem.Allocator,
    o: Options,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: kern.Kernels,
    w: *wts.Weights,
    bufs: std.ArrayList(Buf),
    counted: ?usize, // set: allocations are counted, not made
    frames: usize = 0, // code frames decoded
    emitted: usize = 0, // samples handed out
    // state
    sem_hist: [wts.sem_blocks]u64, // [W][6]
    head_hist: [wts.head_blocks]u64,
    hist: u64, // every history in one block (zeroed per request)
    hist_bytes: usize,
    kv: [5][][2]u64, // per stage and layer: K, V [max rows of the stage][d]
    spec_frames: [2]u64, // windowed head frames [16 * max_frames][n_fft]
    // scratch (R = 16 * pass_frames rows)
    codes: u64, // i32 [pass][10]
    xa: u64, // [R][W] (or [pass][1280])
    xb: u64,
    n: u64,
    o_: u64,
    att: u64,
    big: u64, // [R][3072]
    // the tail, whole request
    branches: u64, // [2][internal samples]
    mixed: u64, // [internal samples]
    audio: u64, // [samples]

    /// The codec on a context the caller made current, launching on `stream`.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, generator: []const u8, head: []const u8, o: Options) !*Codec {
        try o.validate();
        const c = try gpa.create(Codec);
        errdefer gpa.destroy(c);
        c.* = undefined;
        c.gpa = gpa;
        c.o = o;
        c.d = d;
        c.stream = stream;
        c.frames = 0;
        c.emitted = 0;
        c.bufs = .empty;
        c.counted = null;
        for (&c.kv) |*layers| layers.* = &.{};
        c.k = try kern.Kernels.load(d);
        errdefer c.k.unload();
        c.w = try wts.load(gpa, io, d, generator, head);
        errdefer {
            c.w.deinit();
            gpa.destroy(c.w);
        }
        errdefer c.freeState();
        try c.allocState();
        try c.reset();
        return c;
    }

    pub fn destroy(c: *Codec) void {
        c.stream.synchronize() catch {};
        c.freeState();
        c.w.deinit();
        c.gpa.destroy(c.w);
        c.k.unload();
        c.gpa.destroy(c);
    }

    fn freeState(c: *Codec) void {
        for (c.bufs.items) |*b| b.free();
        c.bufs.deinit(c.gpa);
        for (c.kv) |layers| c.gpa.free(layers);
    }

    fn alloc(c: *Codec, bytes: usize) !u64 {
        if (c.counted) |*n| {
            n.* += @max(bytes, 256);
            return 0;
        }
        var b = try Buf.alloc(c.d, @max(bytes, 256));
        errdefer b.free();
        try c.bufs.append(c.gpa, b);
        return b.ptr;
    }

    /// Device bytes of the state and workspaces `o` allocates: the allocation code itself, counting.
    pub fn stateBytes(o: Options) usize {
        var c: Codec = undefined;
        c.o = o;
        c.counted = 0;
        c.allocState() catch unreachable;
        return c.counted.?;
    }

    fn allocState(c: *Codec) !void {
        const T = c.o.max_frames;
        const R = head_rate * c.o.pass_frames;
        const one = W * 6 * 4;
        c.hist_bytes = (wts.sem_blocks + wts.head_blocks) * one;
        c.hist = try c.alloc(c.hist_bytes);
        for (&c.sem_hist, 0..) |*h, i| h.* = c.hist + i * one;
        for (&c.head_hist, 0..) |*h, i| h.* = c.hist + (wts.sem_blocks + i) * one;
        for (&c.kv, wts.stages, 0..) |*layers, shape, si| {
            const bytes = (T << @intCast(si)) * shape.d * 4;
            if (c.counted != null) {
                for (0..2 * shape.layers) |_| _ = try c.alloc(bytes);
                continue;
            }
            layers.* = try c.gpa.alloc([2]u64, shape.layers);
            for (layers.*) |*pair| pair.* = .{ try c.alloc(bytes), try c.alloc(bytes) };
        }
        for (&c.spec_frames, wts.n_ffts) |*f, n| f.* = try c.alloc(head_rate * T * n * 4);
        c.codes = try c.alloc(c.o.pass_frames * 10 * 4);
        c.xa = try c.alloc(R * W * 4);
        c.xb = try c.alloc(R * W * 4);
        c.n = try c.alloc(R * W * 4);
        c.o_ = try c.alloc(R * W * 4);
        c.att = try c.alloc(R * W * 4);
        c.big = try c.alloc(R * 3072 * 4);
        c.branches = try c.alloc(2 * T * internal_frame * 4);
        c.mixed = try c.alloc(T * internal_frame * 4);
        c.audio = try c.alloc(T * frame_samples * 4);
    }

    fn buf(c: *Codec, ptr: u64) Buf {
        return .{ .d = c.d, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    pub fn download(c: *Codec, src: u64, out: []u8) !void {
        try c.buf(src).downloadAsync(0, out, c.stream.handle);
        try c.stream.synchronize();
    }

    /// A new request: zero histories, no frames. Cache rows are written before they are read.
    pub fn reset(c: *Codec) !void {
        var b = c.buf(c.hist);
        b.len = c.hist_bytes;
        try b.fill8(0, c.stream.handle);
        try c.stream.synchronize();
        c.frames = 0;
        c.emitted = 0;
    }

    /// The next code frames (every code below 4096). Under a probe they must fit one pass.
    pub fn decode(c: *Codec, codes: []const [10]i32, probe: ?Probe) !void {
        if (c.frames + codes.len > c.o.max_frames) return error.FrameBudget;
        if (probe != null and codes.len > c.o.pass_frames) return error.ProbeNeedsOnePass;
        for (codes) |row| for (row) |code| if (code < 0 or code >= wts.codebook_size) return error.BadCode;
        var at: usize = 0;
        while (at < codes.len) {
            const t = @min(c.o.pass_frames, codes.len - at);
            try c.pass(codes[at..][0..t], probe);
            at += t;
        }
    }

    fn convnext(c: *Codec, b: wts.Block, hist: u64, x: u64, rows: usize) !void {
        try stage_layer.convnext(&c.k, c.stream, .{ .n = c.n, .o = c.o_, .att = c.att, .big = c.big }, b, hist, x, rows);
    }

    fn layer(c: *Codec, shape: wts.StageShape, L: wts.Layer, kv: [2]u64, x: u64, rows: usize, pos0: usize) !void {
        try stage_layer.layer(&c.k, c.stream, .{ .n = c.n, .o = c.o_, .att = c.att, .big = c.big }, shape, L, kv, x, rows, pos0);
    }

    fn pass(c: *Codec, codes: []const [10]i32, probe: ?Probe) !void {
        const k = &c.k;
        const s = c.stream;
        const w = c.w;
        const t = codes.len;
        try c.buf(c.codes).uploadAsync(0, std.mem.sliceAsBytes(codes), s.handle);
        // the latent: the semantic code through its head, plus the nine residual codes
        var x = c.xa;
        var y = c.xb;
        try k.gather(s, c.codes, 0, w.sem_table, x, W, t);
        for (w.sem, c.sem_hist) |b, h| try c.convnext(b, h, x, t);
        try k.linear(s, w.sem_out_w, w.sem_out_b, x, W, W, wts.latent_dim, y, wts.latent_dim, t);
        try k.latent(s, c.codes, w.res_tables, y, wts.latent_dim, t);
        if (probe) |p| try p.f(p.ctx, "latent", y, t, wts.latent_dim);
        std.mem.swap(u64, &x, &y);
        // the stages: x holds the stage's input rows
        var rows = t;
        for (wts.stages, w.stages, c.kv, 0..) |shape, st, kv, si| {
            const pos0 = c.frames << @intCast(si);
            try k.linear(s, st.in_proj, 0, x, shape.in, shape.in, shape.d, y, shape.d, rows);
            for (st.layers, kv) |L, pair| try c.layer(shape, L, pair, y, rows, pos0);
            if (shape.out == 0) {
                std.mem.swap(u64, &x, &y);
                if (probe) |p| try p.f(p.ctx, "stage_8", x, rows, shape.d);
                break;
            }
            try k.linear(s, st.out_proj, 0, y, shape.d, shape.d, shape.out, x, shape.out, rows);
            if (probe) |p| {
                var nb: [16]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "stage_{d}", .{2 * si}), x, rows, shape.out);
            }
            rows *= 2;
            try k.patchUp(s, x, y, shape.out, rows);
            std.mem.swap(u64, &x, &y);
        }
        // the head: ConvNeXt blocks, a norm, then each branch's spectrum to windowed frames
        for (w.head, c.head_hist) |b, h| try c.convnext(b, h, x, rows);
        try k.layernorm(s, x, w.head_ln_w, w.head_ln_b, c.n, W, 1e-6, rows);
        if (probe) |p| try p.f(p.ctx, "trunk", c.n, rows, W);
        const row0 = c.frames * head_rate;
        for (wts.n_ffts, 0..) |n, b| {
            try k.linear(s, w.outs_w[b], w.outs_b[b], c.n, W, W, n + 2, c.big, n + 2, rows);
            if (probe) |p| try p.f(p.ctx, if (b == 0) "y0" else "y1", c.big, rows, n + 2);
            try k.spec(s, c.big, n / 2 + 1, rows);
            try k.linear(s, w.synth[b], 0, c.big, n + 2, n + 2, n, c.spec_frames[b] + row0 * n * 4, n, rows);
        }
        c.frames += t;
    }

    /// The samples no later frame can change since the last call (2940 F - 1470 after F frames); `final` adds the rest.
    pub fn emit(c: *Codec, gpa: std.mem.Allocator, final: bool, probe: ?Probe) ![]f32 {
        const k = &c.k;
        const s = c.stream;
        const cap = c.o.max_frames * internal_frame;
        const total = c.frames * internal_frame; // internal samples the decoded frames cover
        const end = if (final) c.frames * frame_samples else (c.frames * frame_samples) -| (frame_samples / 2);
        if (end <= c.emitted) return gpa.alloc(f32, 0);
        // output [emitted, end) reads the mix from the first block's start to the last block's taps, 63 more a branch
        const m0 = (wts.resample_in * (c.emitted / wts.resample_out)) -| 17;
        const m1 = @min(total, wts.resample_in * ((end - 1) / wts.resample_out) + wts.resample_taps - 17);
        const b0 = m0 -| 63;
        const b1 = @min(total, m1 + 63);
        // before the end, samples past the decoded frames are never read; at the end they read as zero
        const bound = if (final) total else cap;
        for (wts.n_ffts, 0..) |n, b| {
            const out = c.branches + (b * cap + b0) * 4;
            try k.ola(s, c.spec_frames[b], c.w.win2[b], c.frames * head_rate, n, wts.hop, b0, b1 - b0, out);
            if (probe) |p| try p.f(p.ctx, if (b == 0) "s0" else "s1", out, 1, b1 - b0);
        }
        try k.mix(s, c.branches, cap, bound, c.w.mixer, m0, m1 - m0, c.mixed + m0 * 4);
        if (probe) |p| try p.f(p.ctx, "mixer", c.mixed + m0 * 4, 1, m1 - m0);
        try k.resample(s, c.mixed, bound, c.w.resampler, c.emitted, end - c.emitted, c.audio);
        if (probe) |p| try p.f(p.ctx, "audio", c.audio, 1, end - c.emitted);
        const pcm = try gpa.alloc(f32, end - c.emitted);
        errdefer gpa.free(pcm);
        try c.download(c.audio, std.mem.sliceAsBytes(pcm));
        c.emitted = end;
        return pcm;
    }

    /// Every decoded frame's samples not yet emitted (caller frees): after no emit, the request's whole audio.
    pub fn finish(c: *Codec, gpa: std.mem.Allocator, probe: ?Probe) ![]f32 {
        return c.emit(gpa, true, probe);
    }
};
