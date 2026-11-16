//! ZONOS3's codec encoder on one GPU, the acoustic branch: a lane of 44.1 kHz audio in, its latent rows at 15 Hz out.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("codec_kernels.zig");
const layers = @import("codec_layers.zig");
const wts = @import("encoder_weights.zig");

const Buf = cuda.DeviceBuffer;

pub const frame_samples = 2940; // 44.1 kHz samples a code frame
pub const internal_frame = 2944; // 44.16 kHz samples a code frame
pub const lane_frames = 300; // a lane is at most 20 s; a longer clip continues in lanes with 10 s of left context
pub const latent_dim = 1024;
const head_rows = 16; // 240 Hz rows a code frame

/// A capture under the reference's dump names: device fp32 rows [rows, width].
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void,

    fn at(p: Probe, comptime f: []const u8, args: anytype, ptr: u64, rows: usize, width: usize) !void {
        var nb: [64]u8 = undefined;
        try p.f(p.ctx, try std.fmt.bufPrint(&nb, f, args), ptr, rows, width);
    }
};

const names = .{ "ze_resample_in", "ze_conv", "ze_snake", "ze_add", "ze_patch_down" };
const F = enum(u8) { resample_in, conv, snake, add, patch_down };

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: kern.Kernels,
    mod: cuda.Module,
    fns: [names.len]cuda.Function,
    w: *wts.Weights,
    bufs: std.ArrayList(Buf),
    audio: u64, // the lane's 44.1 kHz samples
    wave: u64, // its 44.16 kHz samples, whole frames
    p: u64, // front-end rows, three buffers a block's steps rotate through
    q: u64,
    r: u64,
    sc: layers.Scratch,
    kv: [2]u64,
    latent: u64, // [frames][latent_dim], the last lane's

    /// The encoder on a context the caller made current, launching on `stream`.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, generator: []const u8) !*Encoder {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.d = d;
        e.stream = stream;
        e.bufs = .empty;
        e.k = try kern.Kernels.load(d);
        errdefer e.k.unload();
        e.mod = try cuda.Module.load(d, cuda.kernels.zonos3_encoder);
        errdefer e.mod.unload();
        inline for (names, 0..) |name, j| e.fns[j] = try e.mod.function(name);
        e.w = try wts.load(gpa, io, d, generator);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        errdefer e.freeState();
        const samples = lane_frames * internal_frame;
        const rows = lane_frames * head_rows;
        const front = samples * wts.blocks[0].channels * 4;
        e.audio = try e.alloc(lane_frames * frame_samples * 4);
        e.wave = try e.alloc(samples * 4);
        e.p = try e.alloc(front);
        e.q = try e.alloc(front);
        e.r = try e.alloc(front);
        e.sc = .{ .n = try e.alloc(rows * 768 * 4), .o = try e.alloc(rows * 768 * 4), .att = try e.alloc(rows * 768 * 4), .big = try e.alloc(rows * 3072 * 4) };
        e.kv = .{ try e.alloc(rows * 768 * 4), try e.alloc(rows * 768 * 4) };
        e.latent = try e.alloc(lane_frames * latent_dim * 4);
        return e;
    }

    pub fn destroy(e: *Encoder) void {
        e.stream.synchronize() catch {};
        e.freeState();
        e.w.deinit();
        e.gpa.destroy(e.w);
        e.mod.unload();
        e.k.unload();
        e.gpa.destroy(e);
    }

    fn freeState(e: *Encoder) void {
        for (e.bufs.items) |*b| b.free();
        e.bufs.deinit(e.gpa);
    }

    fn alloc(e: *Encoder, bytes: usize) !u64 {
        var b = try Buf.alloc(e.d, @max(bytes, 256));
        errdefer b.free();
        try e.bufs.append(e.gpa, b);
        return b.ptr;
    }

    fn buf(e: *Encoder, ptr: u64) Buf {
        return .{ .d = e.d, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    pub fn download(e: *Encoder, src: u64, out: []u8) !void {
        try e.buf(src).downloadAsync(0, out, e.stream.handle);
        try e.stream.synchronize();
    }

    fn go(e: *Encoder, which: F, gx: usize, gy: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(e.fns[@backingInt(which)], .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = 256 }, .shared = @intCast(shared) }, e.stream, a);
    }

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    fn l(x: usize) i64 {
        return @intCast(x);
    }

    /// out [t_out][c_out] from x [t][c_in]: `kernel` taps `dilation` apart, `pad` zeros before the first input.
    fn conv(e: *Encoder, x: u64, c_in: usize, t: usize, w: wts.Conv, kernel: usize, dilation: usize, stride: usize, pad: usize, out: u64, c_out: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(c_in));
        a.add(i(t));
        a.add(w.w);
        a.add(w.b);
        a.add(i(kernel));
        a.add(i(dilation));
        a.add(i(stride));
        a.add(i(pad));
        a.add(out);
        a.add(i(c_out));
        try e.go(.conv, t / stride, (c_out + 7) / 8, kernel * c_in * 4, &a);
    }

    fn snake(e: *Encoder, x: u64, alpha: u64, c: usize, t: usize, y: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(alpha);
        a.add(i(c));
        a.add(l(t * c));
        a.add(y);
        try e.go(.snake, (t * c + 255) / 256, 1, 0, &a);
    }

    fn add(e: *Encoder, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(l(n));
        try e.go(.add, (n + 255) / 256, 1, 0, &a);
    }

    /// in [rows * h][c] to out [rows][c * h].
    fn patchDown(e: *Encoder, in: u64, out: u64, c: usize, h: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(i(c));
        a.add(i(h));
        try e.go(.patch_down, (c * h + 255) / 256, rows, 0, &a);
    }

    /// A lane's audio (44.1 kHz, at most 20 s, from the lane's start) to `latent`; the frames it covers.
    pub fn acoustic(e: *Encoder, audio: []const f32, probe: ?Probe) !usize {
        const frames = (audio.len + frame_samples - 1) / frame_samples;
        if (frames == 0 or frames > lane_frames) return error.BadLane;
        const s = e.stream;
        const k = &e.k;
        const w = e.w;
        var t = frames * internal_frame;
        // 44.16 kHz, zero to the end of the last frame
        try e.buf(e.audio).uploadAsync(0, std.mem.sliceAsBytes(audio), s.handle);
        var wave = e.buf(e.wave);
        wave.len = t * 4;
        try wave.fill8(0, s.handle);
        {
            const count = (audio.len * wts.resample_out + wts.resample_in - 1) / wts.resample_in;
            var a: cuda.Args = .{};
            a.add(e.audio);
            a.add(l(audio.len));
            a.add(w.resampler);
            a.add(l(count));
            a.add(e.wave);
            try e.go(.resample_in, (count + 255) / 256, 1, 0, &a);
        }
        if (probe) |p| try p.f(p.ctx, "internal_audio", e.wave, t, 1);
        // the front end: x holds the block's rows, y and z are scratch
        var x = e.p;
        var y = e.q;
        var z = e.r;
        try e.conv(e.wave, 1, t, w.first, 7, 1, 1, 6, x, wts.blocks[0].channels);
        if (probe) |p| try p.f(p.ctx, "lane.0.enc.in", x, t, wts.blocks[0].channels);
        for (w.blocks, wts.blocks, 0..) |b, shape, bi| {
            const c = shape.channels;
            for (b.units, wts.dilations, 0..) |u, dilation, ui| {
                try e.snake(x, u.alpha0, c, t, y);
                try e.conv(y, c, t, u.wide, 7, dilation, 1, 6 * dilation, z, c);
                try e.snake(z, u.alpha1, c, t, y);
                try e.conv(y, c, t, u.point, 1, 1, 1, 0, z, c);
                try e.add(x, z, t * c);
                if (probe) |p| try p.at("lane.0.enc.b{d}.res{d}", .{ bi, ui }, x, t, c);
            }
            try e.snake(x, b.alpha, c, t, y);
            try e.conv(y, c, t, b.down, 2 * shape.stride, 1, shape.stride, shape.stride, z, 2 * c);
            t /= shape.stride;
            std.mem.swap(u64, &x, &z);
            if (probe) |p| try p.at("lane.0.enc.b{d}.conv", .{bi}, x, t, 2 * c);
        }
        var rows = t / wts.patch;
        try e.patchDown(x, y, wts.front_width, wts.patch, rows);
        std.mem.swap(u64, &x, &y);
        // the stages: x holds a stage's input rows
        for (wts.stages, w.stages, 0..) |shape, st, si| {
            try k.linear(s, st.in_proj, 0, x, shape.in, shape.in, shape.d, y, shape.d, rows);
            if (probe) |p| try p.at("lane.0.stage.{d}.in", .{si}, y, rows, shape.d);
            for (st.layers, 0..) |L, li| {
                try layers.layer(k, s, e.sc, shape, L, e.kv, y, rows, 0);
                if (probe) |p| try p.at("lane.0.stage.{d}.l{d}", .{ si, li }, y, rows, shape.d);
            }
            try k.linear(s, st.out_proj, 0, y, shape.d, shape.d, shape.out, x, shape.out, rows);
            if (probe) |p| try p.at("lane.0.stage.{d}", .{si}, x, rows, shape.out);
            if (si + 1 == wts.stages.len) break;
            rows /= 2;
            try e.patchDown(x, y, shape.out, 2, rows);
            std.mem.swap(u64, &x, &y);
        }
        try e.buf(e.latent).copyFrom(0, x, frames * latent_dim * 4, s.handle);
        if (probe) |p| try p.f(p.ctx, "lane.0.acoustic", e.latent, frames, latent_dim);
        return frames;
    }
};
