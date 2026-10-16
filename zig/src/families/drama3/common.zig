//! What both Drama 3 backends share: request settings, the parity probe, streaming stats and the clock.
const std = @import("std");

pub const Options = struct {
    max_ctx: usize = 4096, // positions (prompt rows + frames)
    max_rows: usize = 1024, // prompt rows a chunk
    attn_chunk: usize = 512, // query rows a prompt attention GEMM
    max_frames: usize = 2048,
    threads: usize = 16,
    pdl: bool = true, // CUDA: GEMVs launched with programmatic dependent launch
    warmup: bool = true, // a short prompt and frame at open: kernels, heuristics and the frame step ready
    split: bool = true, // CUDA: tower codes 2..9 on a second stream beside the next backbone row
};

pub const Sampling = struct {
    temperature: f32 = 1.0,
    top_k: i32 = 30,
    top_p: f32 = 0.9,
    seed: u64 = 1,

    pub fn greedy(s: Sampling) bool {
        return s.temperature <= 0 or s.top_k == 1;
    }
};

/// Per-layer probe of a pass under the oracle's dump names, called with device rows [t, width] in bf16 or fp32.
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, f32_: bool) anyerror!void,
};

pub const Sink = struct {
    ctx: *anyopaque,
    audio: *const fn (ctx: *anyopaque, samples: []const f32) anyerror!void,
};

pub const StreamOptions = struct {
    first_chunk: usize = 4, // frames before the first audio (4 = 186 ms of audio)
    chunk: usize = 16, // frames a later chunk, reached by 1.25x steps from first_chunk (generation runs ~1.5x
    // real time, so each chunk is made within the audio already handed over: no gap when played as it arrives)
};

pub const Stats = struct {
    frames: usize,
    eos: bool,
    prompt_rows: usize,
    prefill_ms: f64,
    first_audio_ms: f64, // request start -> the first samples handed to the sink
    total_ms: f64,
    audio_s: f64,

    pub fn fps(s: Stats) f64 {
        return @as(f64, @floatFromInt(s.frames)) / ((s.total_ms - s.prefill_ms) / 1e3);
    }
};

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

pub fn ms(a: i128, b: i128) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e6;
}
