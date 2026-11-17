//! ZONOS3's codec decoder on Metal: code frames [n][10] in any chunks (the trunk keeps its state), 44.1 kHz fp32 out.
const std = @import("std");
const gpu = @import("gpu.zig");
const k = @import("codec_kernels.zig");
const wts = @import("codec_weights.zig");
const stage_layer = @import("codec_layers.zig");
const metallib = @import("zonos3_metallib").bytes;

const Gpu = gpu.Gpu;
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
    g: *Gpu,
    own_gpu: bool, // a standalone codec closes its device
    first_slot: usize, // the device's buffers from here on are this codec's
    w: *wts.Weights,
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

    /// The codec on `g`, after everything else made on it so far; destroy it before anything made on `g` before it.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, generator: []const u8, head: []const u8, o: Options) !*Codec {
        try o.validate();
        const c = try gpa.create(Codec);
        errdefer gpa.destroy(c);
        c.* = undefined;
        c.gpa = gpa;
        c.o = o;
        c.g = g;
        c.own_gpu = false;
        c.first_slot = g.slots();
        c.frames = 0;
        c.emitted = 0;
        for (&c.kv) |*layers| layers.* = &.{};
        errdefer g.releaseFrom(c.first_slot);
        c.w = try wts.load(gpa, io, g, generator, head);
        errdefer {
            c.w.deinit();
            gpa.destroy(c.w);
        }
        errdefer for (c.kv) |layers| gpa.free(layers);
        try c.allocState();
        try c.reset();
        return c;
    }

    /// The codec on its own device.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, generator: []const u8, head: []const u8, o: Options) !*Codec {
        const g = try Gpu.open(gpa, metallib);
        errdefer g.close();
        const c = try create(gpa, io, g, generator, head, o);
        c.own_gpu = true;
        return c;
    }

    pub fn destroy(c: *Codec) void {
        for (c.kv) |layers| c.gpa.free(layers);
        c.w.deinit();
        c.gpa.destroy(c.w);
        if (c.own_gpu) c.g.close() else c.g.releaseFrom(c.first_slot);
        c.gpa.destroy(c);
    }

    fn alloc(c: *Codec, bytes: usize) !u64 {
        return c.g.alloc(bytes);
    }

    /// Bytes of the state and workspaces `o` allocates: the allocation code itself, counting.
    pub fn stateBytes(o: Options) usize {
        var g: Gpu = undefined;
        g.counted = 0;
        var c: Codec = undefined;
        c.o = o;
        c.g = &g;
        c.allocState() catch unreachable;
        return g.counted.?;
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
            if (c.g.counted != null) {
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

    pub fn download(c: *Codec, src: u64, out: []u8) !void {
        try c.g.get(src, out);
    }

    /// A new request: zero histories, no frames. Cache rows are written before they are read.
    pub fn reset(c: *Codec) !void {
        try c.g.flush();
        @memset(c.g.host(c.hist, c.hist_bytes), 0);
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

    fn convnext(c: *Codec, b: wts.Block, hist: u64, x: u64, rows: usize) void {
        stage_layer.convnext(c.g, .{ .n = c.n, .o = c.o_, .att = c.att, .big = c.big }, b, hist, x, rows);
    }

    fn layer(c: *Codec, shape: wts.StageShape, L: wts.Layer, kv: [2]u64, x: u64, rows: usize, pos0: usize) void {
        stage_layer.layer(c.g, .{ .n = c.n, .o = c.o_, .att = c.att, .big = c.big }, shape, L, kv, x, rows, pos0);
    }

    fn pass(c: *Codec, codes: []const [10]i32, probe: ?Probe) !void {
        const g = c.g;
        const w = c.w;
        const t = codes.len;
        try g.put(c.codes, std.mem.sliceAsBytes(codes));
        // the latent: the semantic code through its head, plus the nine residual codes
        var x = c.xa;
        var y = c.xb;
        k.gather(g, c.codes, 0, w.sem_table, x, W, t);
        for (w.sem, c.sem_hist) |b, h| c.convnext(b, h, x, t);
        k.linear(g, w.sem_out_w, w.sem_out_b, x, W, W, wts.latent_dim, y, wts.latent_dim, t);
        k.latent(g, c.codes, w.res_tables, y, wts.latent_dim, t);
        if (probe) |p| try p.f(p.ctx, "latent", y, t, wts.latent_dim);
        std.mem.swap(u64, &x, &y);
        // the stages: x holds the stage's input rows
        var rows = t;
        for (wts.stages, w.stages, c.kv, 0..) |shape, st, kv, si| {
            const pos0 = c.frames << @intCast(si);
            k.linear(g, st.in_proj, 0, x, shape.in, shape.in, shape.d, y, shape.d, rows);
            for (st.layers, kv) |L, pair| c.layer(shape, L, pair, y, rows, pos0);
            if (shape.out == 0) {
                std.mem.swap(u64, &x, &y);
                if (probe) |p| try p.f(p.ctx, "stage_8", x, rows, shape.d);
                break;
            }
            k.linear(g, st.out_proj, 0, y, shape.d, shape.d, shape.out, x, shape.out, rows);
            if (probe) |p| {
                var nb: [16]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "stage_{d}", .{2 * si}), x, rows, shape.out);
            }
            rows *= 2;
            k.patchUp(g, x, y, shape.out, rows);
            std.mem.swap(u64, &x, &y);
        }
        // the head: ConvNeXt blocks, a norm, then each branch's spectrum to windowed frames
        for (w.head, c.head_hist) |b, h| c.convnext(b, h, x, rows);
        k.layernorm(g, x, w.head_ln_w, w.head_ln_b, c.n, W, 1e-6, rows);
        if (probe) |p| try p.f(p.ctx, "trunk", c.n, rows, W);
        const row0 = c.frames * head_rate;
        for (wts.n_ffts, 0..) |n, b| {
            k.linear(g, w.outs_w[b], w.outs_b[b], c.n, W, W, n + 2, c.big, n + 2, rows);
            if (probe) |p| try p.f(p.ctx, if (b == 0) "y0" else "y1", c.big, rows, n + 2);
            k.spec(g, c.big, n / 2 + 1, rows);
            k.linear(g, w.synth[b], 0, c.big, n + 2, n + 2, n, c.spec_frames[b] + row0 * n * 4, n, rows);
        }
        c.frames += t;
    }

    /// The samples no later frame can change since the last call (2940 F - 1470 after F frames); `final` adds the rest.
    pub fn emit(c: *Codec, gpa: std.mem.Allocator, final: bool, probe: ?Probe) ![]f32 {
        const g = c.g;
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
            k.ola(g, c.spec_frames[b], c.w.win2[b], c.frames * head_rate, n, wts.hop, b0, b1 - b0, out);
            if (probe) |p| try p.f(p.ctx, if (b == 0) "s0" else "s1", out, 1, b1 - b0);
        }
        k.mix(g, c.branches, cap, bound, c.w.mixer, m0, m1 - m0, c.mixed + m0 * 4);
        if (probe) |p| try p.f(p.ctx, "mixer", c.mixed + m0 * 4, 1, m1 - m0);
        k.resample(g, c.mixed, bound, c.w.resampler, c.emitted, end - c.emitted, c.audio);
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
