//! A reference clip to its codes on Metal: the codec encoder's acoustic branch, the SSL teacher and the quantizer.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;
const enc = @import("encoder.zig");
const teach = @import("teacher.zig");
const quant = @import("quantizer.zig");
const features = @import("../features.zig");
const release = @import("../release.zig");
const dec = @import("codec_weights.zig");

pub const codebooks = quant.codebooks;
pub const rate = 44100;
/// The shortest clip the release clones from, and the longest it keeps.
pub const min_samples = rate / 2;
pub const max_frames = enc.clip_frames;
/// Frames of what came before that a lane after the first carries: 10 s.
const context = enc.lane_frames / 2;

pub const Clip = struct {
    gpa: std.mem.Allocator,
    encoder: *enc.Encoder,
    teacher: *teach.Teacher,
    quantizer: *quant.Quantizer,

    /// On `g` after all else made on it; `root`: the release, `decoder`: its weights. Destroy before older ones on `g`.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, root: []const u8, generator: []const u8, decoder: *const dec.Weights) !*Clip {
        const c = try gpa.create(Clip);
        errdefer gpa.destroy(c);
        const teacher_dir = try std.fs.path.join(gpa, &.{ root, release.teacher_dir });
        defer gpa.free(teacher_dir);
        c.gpa = gpa;
        c.encoder = try enc.Encoder.create(gpa, io, g, generator);
        errdefer c.encoder.destroy();
        c.teacher = try teach.Teacher.create(gpa, io, g, teacher_dir);
        errdefer c.teacher.destroy();
        c.quantizer = try quant.Quantizer.create(gpa, io, g, generator, decoder);
        return c;
    }

    pub fn destroy(c: *Clip) void {
        c.quantizer.destroy();
        c.teacher.destroy();
        c.encoder.destroy();
        c.gpa.destroy(c);
    }

    /// Device bytes of weights this holds.
    pub fn bytes(c: *const Clip) usize {
        return c.encoder.w.bytes + c.teacher.bytes + c.quantizer.w.bytes;
    }

    /// A clip's codes [frames][10] (caller frees): 44.1 kHz mono, at least half a second; past 60 s is not read.
    pub fn encode(c: *Clip, gpa: std.mem.Allocator, clip: []const f32) ![][codebooks]i32 {
        if (clip.len < min_samples) return error.ClipTooShort;
        const audio = clip[0..@min(clip.len, max_frames * enc.frame_samples)];
        const frames = try c.encoder.load(audio, null);
        const out = try gpa.alloc([codebooks]i32, frames);
        errdefer gpa.free(out);
        const lane_samples = enc.lane_frames * enc.frame_samples;
        const padded = try gpa.alloc(f32, lane_samples);
        defer gpa.free(padded);
        const lane = try gpa.alloc([codebooks]i32, enc.lane_frames);
        defer gpa.free(lane);
        const ssl = try gpa.alloc(f32, teach.max_rows * teach.d);
        defer gpa.free(ssl);
        var start: usize = 0;
        while (start < frames) {
            // the first lane starts the clip; a later one begins `context` frames back and keeps what follows them
            const carried: usize = if (start == 0) 0 else context;
            const end = @min(frames, start + enc.lane_frames - carried);
            const first = start - carried;
            const n = end - first;
            try c.encoder.acoustic(first, n, null);
            // the teacher hears the lane's own samples, zero to its 20 s
            const from = first * enc.frame_samples;
            const have = @min(audio.len, end * enc.frame_samples) - from;
            @memset(padded, 0);
            @memcpy(padded[0..have], audio[from..][0..have]);
            const low = try features.resample(gpa, padded);
            defer gpa.free(low);
            const feats = try features.rows(gpa, low);
            defer gpa.free(feats);
            const rows = feats.len / teach.feature_width;
            const ptr = try c.teacher.run(feats, null);
            try c.teacher.download(ptr, std.mem.sliceAsBytes(ssl[0 .. rows * teach.d]));
            const pooled = try quant.pool(gpa, ssl[0 .. rows * teach.d], n);
            defer gpa.free(pooled);
            try c.quantizer.run(c.encoder.latent, pooled, lane[0..n], null);
            @memcpy(out[start..end], lane[carried..n]);
            start = end;
        }
        return out;
    }
};
