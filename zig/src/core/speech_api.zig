/// Mono PCM written before a successful speech request's finished event; `silence_mid`: speech ended in a long silence.
pub const SpeechResult = struct { samples: usize = 0, reason: @import("engine_api.zig").Reason = .stop, silence_mid: bool = false };

/// Caller-owned until finished, even on cancellation; a streamed request's pcm[0..n] is the caller's after ``audio`` n.
pub const SpeechRequest = struct {
    prompt: []const u32,
    frames: []const [11]i32 = &.{},
    pcm: []f32,
    result: *SpeechResult,
    seed: u64,
    max_frames: u32 = 1024,
    temperature: f32 = 1,
    top_p: f32 = 0.9,
    top_k: u32 = 30,
    /// Publishes decoded prefixes; whole replies also decode incrementally when a speech guard is enabled.
    stream: bool = false,
    /// End of the invariant reference prompt, excluding request text; 0 disables reuse; backend saves no lookahead row.
    reference_rows: u32 = 0,
};

pub const SpeechInfo = struct {
    sample_rate: u32,
    frame_length: u32,
    max_frames: u32,
    max_prompt_tokens: u32,
};
