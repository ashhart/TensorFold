//! Drama 3's codec (Fish S2 modded DAC) on Metal in fp32: codec_net.zig's schedules on the drama3 metallib.
const std = @import("std");
const net = @import("../codec_net.zig");
const source = @import("../codec_load.zig");
const gpu = @import("gpu.zig");

const Gpu = gpu.Gpu;

pub const samples_per_frame = net.samples_per_frame;
pub const sample_rate = net.sample_rate;
pub const D = net.D;
pub const Stage = net.Stage;
pub const Tf = net.Tf;
pub const Probe = net.Probe;
pub const Options = net.Options;

fn u(x: usize) u32 {
    return @intCast(x);
}

const ElemArgs = extern struct { total: u32, n: u32, flag: u32 };

pub const Codec = struct {
    gpa: std.mem.Allocator,
    owns_weights: bool,
    state_start: usize,
    state_end: usize,
    g: *Gpu,
    o: Options,
    w: net.Weights,
    state: net.Decoder,
    probe: ?Probe = null,
    enc_bufs: ?EncBufs = null, // the encoder's buffers, kept for the next clip that fits

    pub const EncBufs = struct { frames: usize, sc: [4]u64, stbuf: u64, kv: [8][2]u64, codes: u64 };

    pub fn load(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, path: []const u8, o: Options) !*Codec {
        const c = try gpa.create(Codec);
        errdefer gpa.destroy(c);
        c.* = undefined;
        c.gpa = gpa;
        c.owns_weights = true;
        c.g = g;
        const first_buffer = g.bufs.items.len;
        errdefer g.releaseFrom(first_buffer);
        c.o = o;
        c.probe = null;
        c.enc_bufs = null;
        var file = try source.Source.open(gpa, io, path);
        defer file.close();
        c.w = try net.Weights.load(gpa, &file, Upload{ .g = g });
        errdefer c.w.deinit(gpa);
        c.state_start = c.g.bufs.items.len;
        errdefer c.g.releaseFrom(c.state_start);
        try c.allocDecoder();
        c.state_end = c.g.bufs.items.len;
        return c;
    }

    /// The buffers stay with the Gpu (it frees them at close); this frees the host side.
    pub fn fork(base: *Codec) !*Codec {
        const c = try base.gpa.create(Codec);
        errdefer base.gpa.destroy(c);
        c.* = base.*;
        c.owns_weights = false;
        c.enc_bufs = null;
        c.state_start = c.g.bufs.items.len;
        errdefer c.g.releaseFrom(c.state_start);
        try c.allocDecoder();
        c.state_end = c.g.bufs.items.len;
        return c;
    }

    pub fn poison(c: *Codec) !void {
        try c.g.flush();
        for (c.g.bufs.items[c.state_start..c.state_end]) |buffer| @memset(buffer.contents()[0..buffer.length()], 0xa5);
    }

    pub fn destroy(c: *Codec) void {
        if (c.owns_weights) c.w.deinit(c.gpa);
        c.gpa.destroy(c);
    }

    fn allocDecoder(c: *Codec) !void {
        c.state = try net.Decoder.alloc(&c.w, c.o, c.g);
        try c.reset();
    }

    /// Forget the stream: zero every carried row (the transformer ring is rewritten before it is read).
    pub fn reset(c: *Codec) !void {
        try c.g.flush();
        c.state.pos = 0;
        for (c.state.stages()) |st| c.zeroMargin(st);
    }

    /// Host write: the caller has flushed the GPU work that reads this stage.
    pub fn zeroMargin(c: *Codec, st: Stage) void {
        if (st.margin == 0) return;
        @memset(c.g.host(st.buf, st.margin * st.cols * 4), 0);
    }

    fn run(c: *Codec) Run {
        return .{ .g = c.g, .tmp = c.state.carry_tmp };
    }

    /// Enqueue the decode of `nf` frames (device int32 rows [nf, 11], codes in columns 1..10) into nf * 2048 samples.
    pub fn decode(c: *Codec, frames: u64, nf: usize, audio: u64) !void {
        try net.decode(c.run(), &c.w, &c.state, c.o, frames, nf, audio);
    }

    /// Codes [frames, 10] of a 44.1 kHz mono clip (padded to whole frames), as DAC.encode makes them.
    pub fn encode(c: *Codec, audio: []const f32) ![][10]i32 {
        try @import("../model.zig").reference(audio, 44100);
        const nf = (audio.len + samples_per_frame - 1) / samples_per_frame;
        if (nf == 0) return error.EmptyAudio;
        const L0 = nf * samples_per_frame;
        var e: Clip = .{ .c = c, .b = try c.clipBuffers(nf) };
        // audio in, padded with zeros to whole frames
        try c.g.flush();
        const in = c.g.host(e.b.sc[0], L0 * 4);
        @memset(in, 0);
        @memcpy(in[0 .. audio.len * 4], std.mem.sliceAsBytes(audio));
        try net.encode(c.run(), &e, &c.w, e.b.sc, e.b.stbuf, nf);
        try c.g.flush();
        const out = try c.gpa.alloc([10]i32, nf);
        @memcpy(std.mem.sliceAsBytes(out), c.g.host(e.b.codes, nf * 40));
        return out;
    }

    /// Buffers for clips up to `frames` frames, kept on the codec so the next clip that fits reuses them.
    fn clipBuffers(c: *Codec, nf: usize) !EncBufs {
        if (c.enc_bufs) |b| if (b.frames >= nf) return b;
        const frames = std.mem.alignForward(usize, nf, 64);
        const L0 = frames * samples_per_frame;
        const big = L0 * 64 + 64 * 1024;
        var b: EncBufs = .{ .frames = frames, .sc = undefined, .stbuf = undefined, .kv = undefined, .codes = undefined };
        for (&b.sc) |*p| p.* = try c.g.alloc(big * 4);
        b.stbuf = try c.g.alloc(big * 4);
        // the transformers' caches: the most rows any of them takes (an encoder block's output, or the frames)
        var kv_rows = frames;
        var rows = L0;
        for (c.w.enc) |B| {
            rows /= B.s;
            if (B.tf != null) kv_rows = @max(kv_rows, rows);
        }
        for (&b.kv) |*p| p.* = .{ try c.g.alloc(kv_rows * D * 4), try c.g.alloc(kv_rows * D * 4) };
        b.codes = try c.g.alloc(frames * 10 * 4);
        c.enc_bufs = b;
        return b;
    }
};

const Upload = struct {
    g: *Gpu,

    pub fn put(up: Upload, bytes: []const u8) !u64 {
        const p = try up.g.alloc(bytes.len);
        @memcpy(up.g.host(p, bytes.len), bytes);
        return p;
    }
};

/// A clip's encoder buffers on Metal, from the codec's kept set.
const Clip = struct {
    c: *Codec,
    b: Codec.EncBufs,

    /// A stage over `buf` with its carried rows zeroed (after the GPU work before it).
    pub fn stage(e: *Clip, buf: u64, margin: usize, cols: usize) !Stage {
        try e.c.g.flush();
        const st: Stage = .{ .buf = buf, .margin = margin, .cols = cols };
        e.c.zeroMargin(st);
        return st;
    }

    pub fn kv(e: *Clip, layers: usize, _: usize) ![]const [2]u64 {
        return e.b.kv[0..layers];
    }

    pub fn codes(e: *Clip, _: usize) !u64 {
        return e.b.codes;
    }

    pub fn probe(e: *Clip, name: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        if (e.c.probe) |p| {
            try e.c.g.flush();
            try p.f(p.ctx, name, ptr, rows, cols);
        }
    }
};

/// The codec's dispatches, as codec_net.zig's schedules call them.
const Run = struct {
    g: *Gpu,
    tmp: u64,

    /// fp32 Y [t, o] (rows ldy apart) = X [t, cin] (rows ldx apart) W [o, cin]^T + beta Y.
    pub fn sgemm(r: Run, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32) !void {
        r.gemmTaps(w, o, cin, x, ldx, y, ldy, t, beta, 1, 0, 0);
    }

    /// sgemm summed over `taps` weight blocks [o, cin], tap j reading X rows (taps - 1 - j) * dil earlier.
    fn gemmTaps(r: Run, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32, taps: usize, dil: usize, base: usize) void {
        const A = extern struct { rows: u32, o: u32, cin: u32, ldx: u32, ldy: u32, beta: f32, taps: u32, dil: u32, base: u32 };
        const a: A = .{ .rows = u(t), .o = u(o), .cin = u(cin), .ldx = u(ldx), .ldy = u(ldy), .beta = beta, .taps = u(taps), .dil = u(dil), .base = u(base) };
        r.g.go(.c_gemm, .{ (o + 63) / 64, (t + 63) / 64, 1 }, 256, &.{ x, w, y }, a);
    }

    fn elem(r: Run, p: gpu.P, bufs: []const u64, total: usize, n: usize, flag: bool) void {
        r.g.go(p, .{ (total + 255) / 256, 1, 1 }, 256, bufs, ElemArgs{ .total = u(total), .n = u(n), .flag = @intFromBool(flag) });
    }

    pub fn bias(r: Run, y: u64, b: u64, rows: usize, n: usize, tanh: bool) !void {
        r.elem(.c_bias, &.{ y, b }, rows * n, n, tanh);
    }

    pub fn snake(r: Run, x: u64, y: u64, rows: usize, cols: usize, alpha: u64) !void {
        r.elem(.c_snake, &.{ x, y, if (alpha != 0) alpha else x }, rows * cols, cols, alpha != 0);
    }

    /// `n` floats.
    pub fn copy(r: Run, src: u64, dst: u64, n: usize) !void {
        r.g.copy(src, dst, n * 4);
    }

    /// Causal convolution from a stage: out [rows, o] = b + sum_j W_j x[t - (K - 1 - j) dil].
    pub fn conv(r: Run, st: Stage, rows: usize, taps: u64, kk: usize, dil: usize, o: usize, out: u64, b: u64, tanh: bool) !void {
        std.debug.assert((kk - 1) * dil <= st.margin);
        r.gemmTaps(taps, o, st.cols, st.buf, st.cols, out, o, rows, 0, kk, dil, st.margin);
        try r.bias(out, b, rows, o, tanh);
    }

    pub fn rmsnorm(r: Run, x: u64, w: u64, y: u64, rows: usize) !void {
        const A = extern struct { D: u32, eps: f32 };
        r.g.go(.c_rmsnorm, .{ rows, 1, 1 }, 256, &.{ x, w, y }, A{ .D = D, .eps = 1e-5 });
    }

    pub fn addGamma(r: Run, x: u64, y: u64, gamma: u64, rows: usize, n: usize) !void {
        r.elem(.c_add_gamma, &.{ x, y, if (gamma != 0) gamma else y }, rows * n, n, gamma != 0);
    }

    pub fn ropeKv(r: Run, qkv: u64, kv: [2]u64, pos0: usize, ring: usize, rows: usize) !void {
        const R = extern struct { pos0: u32, ring: u32 };
        r.g.go(.c_rope_kv, .{ rows, 1, 1 }, 512, &.{ qkv, kv[0], kv[1] }, R{ .pos0 = u(pos0), .ring = u(ring) });
    }

    pub fn attn(r: Run, qkv: u64, kv: [2]u64, out: u64, pos0: usize, window: usize, ring: usize, rows: usize) !void {
        const A = extern struct { pos0: u32, window: u32, ring: u32 };
        const a: A = .{ .pos0 = u(pos0), .window = u(window), .ring = u(ring) };
        r.g.go(.c_attn, .{ rows, 2, 1 }, 256, &.{ qkv, kv[0], kv[1], out }, a);
    }

    pub fn siluMul(r: Run, x: u64, y: u64, rows: usize) !void {
        r.g.go(.c_silu_mul, .{ (3 * D + 255) / 256, rows, 1 }, 256, &.{ x, y }, @as(u32, 3 * D));
    }

    pub fn dwconv7(r: Run, src: u64, dw: u64, dw_b: u64, out: u64, rows: usize) !void {
        r.g.go(.c_dwconv7, .{ (D + 255) / 256, rows, 1 }, 256, &.{ src, dw, dw_b, out }, @as(u32, D));
    }

    pub fn layernorm(r: Run, x: u64, w: u64, b: u64, y: u64, rows: usize) !void {
        const A = extern struct { D: u32, eps: f32 };
        r.g.go(.c_layernorm, .{ rows, 1, 1 }, 256, &.{ x, w, b, y }, A{ .D = D, .eps = 1e-6 });
    }

    pub fn gelu(r: Run, x: u64, total: usize) !void {
        const n: u32 = u(total);
        r.g.go(.c_gelu, .{ (n + 255) / 256, 1, 1 }, 256, &.{x}, n);
    }

    pub fn vqGather(r: Run, frames: u64, table: u64, out: u64, n: usize) !void {
        const G = extern struct { fs: u32, D: u32 };
        r.g.go(.c_vq_gather, .{ n, 1, 1 }, 256, &.{ frames, table, out }, G{ .fs = 11, .D = D });
    }

    pub fn vqEncode(r: Run, x: u64, w: *const net.Weights, codes: u64, nf: usize) !void {
        const bufs = [_]u64{ x, w.vq_in_w, w.vq_in_b, w.vq_cbn, w.vq_table, codes };
        r.g.go(.c_vq_encode, .{ nf, 1, 1 }, 256, &bufs, @as(u32, D));
    }
};
