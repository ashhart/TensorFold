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
    /// One step of a speech edit in place of a spoken reply; null: none.
    edit: ?*const SpeechEdit = null,
};

/// A speech edit's steps on the engine: the server plans the edit between them.
pub const SpeechEdit = struct {
    op: Op,
    /// encode: room for the clip's codes, a frame a row. decode: the codes to speak.
    codes: [][10]i32 = &.{},
    /// fill: room for the rows the fill generates, the finishing row included.
    rows: [][11]i32 = &.{},
    /// fill: rows the span-end id stays masked for.
    min_frames: u32 = 0,
    /// place: the transcript's words that hold a letter or digit, and room for what is found of each.
    words: []const []const u8 = &.{},
    spans_ms: [][2]u32 = &.{},
    presence: []f32 = &.{},
    out: *Out,

    /// encode: `reference_audio` to codes. fill: `frames` to rows. decode: `codes` to `pcm`. place: `words` timed.
    pub const Op = enum { encode, fill, decode, place };
    /// `frames`: codes written (encode). `rows`: rows written, and the aligned span-end frame, -1 without one (fill).
    pub const Out = struct { frames: usize = 0, rows: usize = 0, end_frame: i64 = -1 };
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
    /// The spans follow the model's reading position instead of the character rate; null: the rate.
    follow: ?Follow = null,

    /// Each span's text positions [a, b) (0 is BOS), the gate's lead and lag in tokens, and the heads that read.
    pub const Follow = struct { ranges: []const [2]u32, lead: f64, lag: f64, heads: []const SpeechLaci.Head };

    /// Full weight on frames [start, end), a linear ramp of `ramp` frames on both sides.
    pub const Span = struct { start: f32, end: f32, ramp: f32, delta: []const f32 };
};

pub const SpeechInfo = struct {
    sample_rate: u32,
    frame_length: u32,
    max_frames: u32,
    max_prompt_tokens: u32,
};
