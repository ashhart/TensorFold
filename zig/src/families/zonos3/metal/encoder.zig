//! ZONOS3's codec encoder on Metal, the acoustic branch: a lane of 44.1 kHz audio in, its latent rows at 15 Hz out.
const std = @import("std");
const gpu = @import("gpu.zig");
const ek = @import("encoder_kernels.zig");
const ck = @import("codec_kernels.zig");
const layers = @import("codec_layers.zig");
const wts = @import("encoder_weights.zig");

const Gpu = gpu.Gpu;

pub const frame_samples = 2940; // 44.1 kHz samples a code frame
pub const internal_frame = 2944; // 44.16 kHz samples a code frame
pub const lane_frames = 300; // a lane is at most 20 s; a longer clip continues in lanes with 10 s of left context
pub const clip_frames = 900; // a reference clip is at most 60 s
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

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
    w: *wts.Weights,
    audio: u64, // the clip's 44.1 kHz samples
    wave: u64, // its 44.16 kHz samples, whole frames
    frames: usize = 0, // the loaded clip's frames
    p: u64, // front-end rows, three buffers a block's steps rotate through
    q: u64,
    r: u64,
    sc: layers.Scratch,
    kv: [2]u64,
    latent: u64, // [frames][latent_dim], the last lane's

    /// The encoder on `g`, after everything else made on it so far; destroy it before anything made on `g` before it.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, generator: []const u8) !*Encoder {
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.g = g;
        const owner = g.own(e);
        defer _ = g.own(owner);
        errdefer g.release(e);
        e.w = try wts.load(gpa, io, g, generator);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        const samples = lane_frames * internal_frame;
        const rows = lane_frames * head_rows;
        const front = samples * wts.blocks[0].channels * 4;
        e.frames = 0;
        e.audio = try g.alloc(clip_frames * frame_samples * 4);
        e.wave = try g.alloc(clip_frames * internal_frame * 4);
        e.p = try g.alloc(front);
        e.q = try g.alloc(front);
        e.r = try g.alloc(front);
        e.sc = .{ .n = try g.alloc(rows * 768 * 4), .o = try g.alloc(rows * 768 * 4), .att = try g.alloc(rows * 768 * 4), .big = try g.alloc(rows * 3072 * 4) };
        e.kv = .{ try g.alloc(rows * 768 * 4), try g.alloc(rows * 768 * 4) };
        e.latent = try g.alloc(lane_frames * latent_dim * 4);
        return e;
    }

    pub fn destroy(e: *Encoder) void {
        e.g.release(e);
        e.w.deinit();
        e.gpa.destroy(e.w);
        e.gpa.destroy(e);
    }

    pub fn download(e: *Encoder, src: u64, out: []u8) !void {
        try e.g.get(src, out);
    }

    /// A clip (44.1 kHz, at most 60 s) resampled whole to 44.16 kHz, zero to the end of its last frame; its frames.
    pub fn load(e: *Encoder, audio: []const f32, probe: ?Probe) !usize {
        const frames = (audio.len + frame_samples - 1) / frame_samples;
        if (frames == 0 or frames > clip_frames) return error.BadClip;
        const g = e.g;
        const t = frames * internal_frame;
        try g.put(e.audio, std.mem.sliceAsBytes(audio));
        @memset(g.host(e.wave, t * 4), 0);
        const count = (audio.len * wts.resample_out + wts.resample_in - 1) / wts.resample_in;
        ek.resampleIn(g, e.audio, audio.len, e.w.resampler, count, e.wave);
        if (probe) |p| try p.f(p.ctx, "internal_audio", e.wave, t, 1);
        e.frames = frames;
        return frames;
    }

    /// A lane of the loaded clip, `frames` frames from frame `first` (at most 20 s), to `latent`.
    pub fn acoustic(e: *Encoder, first: usize, frames: usize, probe: ?Probe) !void {
        if (frames == 0 or frames > lane_frames or first + frames > e.frames) return error.BadLane;
        const g = e.g;
        const w = e.w;
        var t = frames * internal_frame;
        // the front end: x holds the block's rows, y and z are scratch
        var x = e.p;
        var y = e.q;
        var z = e.r;
        ek.conv(g, e.wave + first * internal_frame * 4, 1, t, w.first.w, w.first.b, 7, 1, 1, 6, x, wts.blocks[0].channels);
        if (probe) |p| try p.f(p.ctx, "lane.0.enc.in", x, t, wts.blocks[0].channels);
        for (w.blocks, wts.blocks, 0..) |b, shape, bi| {
            const c = shape.channels;
            for (b.units, wts.dilations, 0..) |u, dilation, ui| {
                ek.snake(g, x, u.alpha0, c, t * c, y);
                ek.conv(g, y, c, t, u.wide.w, u.wide.b, 7, dilation, 1, 6 * dilation, z, c);
                ek.snake(g, z, u.alpha1, c, t * c, y);
                ek.conv(g, y, c, t, u.point.w, u.point.b, 1, 1, 1, 0, z, c);
                ek.add(g, x, z, t * c);
                if (probe) |p| try p.at("lane.0.enc.b{d}.res{d}", .{ bi, ui }, x, t, c);
            }
            ek.snake(g, x, b.alpha, c, t * c, y);
            ek.conv(g, y, c, t, b.down.w, b.down.b, 2 * shape.stride, 1, shape.stride, shape.stride, z, 2 * c);
            t /= shape.stride;
            std.mem.swap(u64, &x, &z);
            if (probe) |p| try p.at("lane.0.enc.b{d}.conv", .{bi}, x, t, 2 * c);
        }
        var rows = t / wts.patch;
        ek.patchDown(g, x, y, wts.front_width, wts.patch, rows);
        std.mem.swap(u64, &x, &y);
        // the stages: x holds a stage's input rows
        for (wts.stages, w.stages, 0..) |shape, st, si| {
            ck.linear(g, st.in_proj, 0, x, shape.in, shape.in, shape.d, y, shape.d, rows);
            if (probe) |p| try p.at("lane.0.stage.{d}.in", .{si}, y, rows, shape.d);
            for (st.layers, 0..) |L, li| {
                layers.layer(g, e.sc, shape, L, e.kv, rows, y, rows, 0);
                if (probe) |p| try p.at("lane.0.stage.{d}.l{d}", .{ si, li }, y, rows, shape.d);
            }
            ck.linear(g, st.out_proj, 0, y, shape.d, shape.d, shape.out, x, shape.out, rows);
            if (probe) |p| try p.at("lane.0.stage.{d}", .{si}, x, rows, shape.out);
            if (si + 1 == wts.stages.len) break;
            rows /= 2;
            ek.patchDown(g, x, y, shape.out, 2, rows);
            std.mem.swap(u64, &x, &y);
        }
        try g.flush();
        @memcpy(g.host(e.latent, frames * latent_dim * 4), g.host(x, frames * latent_dim * 4));
        if (probe) |p| try p.f(p.ctx, "lane.0.acoustic", e.latent, frames, latent_dim);
    }
};
