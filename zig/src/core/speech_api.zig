//! One speech synthesis: the request, its result, what the engine offers, and why a submit is refused.

/// Mono PCM written before a successful speech request's finished event.
pub const Result = struct { samples: usize = 0 };

/// Caller-owned until finished, even on cancellation; a streamed request's pcm[0..n] is the caller's after ``audio`` n.
pub const Request = struct {
    prompt: []const u32,
    pcm: []f32,
    result: *Result,
    seed: u64,
    max_frames: u32 = 1024,
    temperature: f32 = 1,
    top_p: f32 = 0.9,
    top_k: u32 = 30,
    /// Decode while generating and report each final prefix with ``audio`` events; false: one decode after EOS.
    stream: bool = false,
};

pub const Info = struct {
    sample_rate: u32,
    frame_length: u32,
    max_frames: u32,
    max_prompt_tokens: u32,
};

pub const SubmitError = error{ Closed, Busy, InvalidRequest, Unsupported };
