//! What every backend shares: the AR's sampling knobs, its result and trace, a song's timings, the clock.
const std = @import("std");
const cfg = @import("config.zig");
const wts = @import("weights.zig");

pub const Sampling = struct {
    ar_cfg: f32 = 1.5,
    top_k: i32 = 50,
    seed: u64 = 0, // the AR stage's own seed (config.deriveSeed(song seed, "ar"))
};

/// Per-frame logits for parity (teacher forcing): code 0 [2, 16385] and the depth heads [7, 2, 1024], fp32.
pub const Trace = struct {
    c0: std.ArrayList(f32) = .empty,
    depth: std.ArrayList(f32) = .empty,
};

pub const Result = struct {
    states: []u16, // bf16 [frames, 8 * 4096] of the kept frames (frame 0 dropped, as the reference)
    frames: usize,
    stopped: bool,
    prompt_tokens: usize,
    prefill_ms: f64,
    ar_ms: f64,
    frames_per_s: f64,
};

pub const Progress = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, stage: []const u8, done: usize, total: usize) void,
};

pub const Song = struct {
    audio: []f32, // [2, samples]
    samples: usize,
    frames: usize,
    stopped: bool,
    prompt_tokens: usize,
    prefill_ms: f64,
    ar_ms: f64,
    frames_per_s: f64,
    dit_ms: f64,
    dav_ms: f64,
    total_ms: f64,
    latent_frames: usize,
    peak_device_gb: f64,
};

/// Weight bytes a frame reads: body and lm_head once, the depth decoder 7 times, each head once.
pub fn frameBytes(a: *const wts.Ar) f64 {
    const per: f64 = if (a.format == .int8) 1 else 2;
    var n: f64 = 0;
    const add = struct {
        fn m(x: wts.Mat) f64 {
            return @floatFromInt(x.n * x.k);
        }
    }.m;
    for (a.body) |L| n += add(L.qkv) + add(L.o) + add(L.gate_up) + add(L.down);
    n += add(a.lm_head);
    var dn: f64 = add(a.projection);
    for (a.depth) |L| dn += add(L.qkv) + add(L.o) + add(L.gate_up) + add(L.down);
    n += @as(f64, cfg.codebooks - 1) * dn;
    for (a.heads) |h| n += add(h);
    return n * per;
}

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

pub fn ms(a: i128, b: i128) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e6;
}
