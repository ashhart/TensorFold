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
    /// Plain text frame rows for reference-conditioned semantic CFG; empty means no guidance.
    unconditioned: []const [11]i32 = &.{},
    cfg_gamma: f32 = 1,
    /// A speaker embedding for the family's speaker row; empty: the prompt rows carry the voice, or there is none.
    speaker: []const f32 = &.{},
    /// A reference clip to clone the voice from: mono samples at the family's rate; empty: none.
    reference_audio: []const f32 = &.{},
    /// The family's condition rows by bucket, in prompt order; empty: every one unknown.
    conditions: []const u32 = &.{},
    /// What the request's steering controls add to its generated frames; null: nothing.
    steering: ?*const SpeechSteering = null,
    /// Generate under LACI: the reading of the text watched, a skip or a stall rolled back and regenerated.
    laci: ?*const SpeechLaci = null,
};

/// The alignment heads of the request's language: attention heads whose attention over the text follows the speech.
pub const SpeechLaci = struct {
    heads: []const Head,
    /// The pace prior, in generated frames a text token.
    frames_per_token: f64,
    /// A short text: the prompt holds a filler sentence after it, and the reply is cut where its reading starts.
    short: ?Short = null,

    /// `filler_token`: the filler's first text row (0 is BOS); the rest are the cut's settings and silence weights.
    pub const Short = struct { filler_token: u32, lookback: u32, min_voiced: u32, max_frames: u32, max_retries: u32, threshold: i32, weights: []const i8 };

    /// `radius`: the half-width of the head's window while a regeneration holds it to the text it failed on.
    pub const Head = struct { layer: u32, head: u32, radius: i64 };
};

/// Deltas on a family's hidden sources, a row of the model's width for each of `sources` in every delta.
pub const SpeechSteering = struct {
    sources: []const u32,
    /// Added to every generated frame; empty: none.
    constant: []const f32 = &.{},
    /// Added at full weight for `onset` frames, ramping over `ramp` frames to `hold` times it; empty: none.
    released: []const f32 = &.{},
    onset: u32 = 0,
    ramp: u32 = 0,
    hold: f64 = 1,
    spans: []const Span = &.{},
    /// Guidance on the deltas through a twin stream without them; 1: off.
    guidance: f32 = 1,

    /// Full weight on frames [start, end), a linear ramp of `ramp` frames on both sides.
    pub const Span = struct { start: f32, end: f32, ramp: f32, delta: []const f32 };
};

pub const SpeechInfo = struct {
    sample_rate: u32,
    frame_length: u32,
    max_frames: u32,
    max_prompt_tokens: u32,
};
