//! A reply to decode: its prompt, limits, images and stop check (``engine_api.Request``).
const lanes = @import("lanes");
const CallGate = @import("engine_request_controls.zig").CallGate;
const Structure = @import("engine_request_controls.zig").Structure;
const Image = lanes.stream.Image;
const Sampling = lanes.Sampling;
const Penalties = lanes.Penalties;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// A reply to decode. The request and every slice in it stay valid until its ``finished`` event.
pub const Request = struct {
    prompt: []const u32,
    embedding: ?lanes.embedding.Batch = null,
    readout: ?*const lanes.readout.Spec = null, // an answer from the prompt's rows (Info.readouts engines only)
    max_tokens: u32,
    sampling: ?Sampling = null,
    penalties: Penalties = .{},
    /// Tokens that end the reply as ``stop`` (the token is delivered); empty when the request ignores EOS.
    eos: []const u32 = &.{},
    stop: ?Stop = null,
    /// false: one token a round and no drafts, the serial reference drafted replies must equal.
    drafts: bool = true,
    /// Background work yields to foreground arrivals.
    background: bool = false,
    /// The prompt length its kept states count as (0: its own); a pass prefilling a reply carries that request's.
    born: u32 = 0,
    /// Prompt prefix lengths worth keeping for a later turn: the rendered history, then shared system blocks.
    history_len: u32 = 0,
    shared_prefixes: []const u32 = &.{},
    /// Where the latest user message's text starts: a state kept before it lets an edit of that turn resume (0: none).
    rewind_len: u32 = 0,
    /// End of the latest user message in this prompt (0: unknown); background reply passes leave it unknown.
    user_end: u32 = 0,
    /// Prompt chunk starts after 0 (Python PrefillPlan); empty selects the engine's plan.
    chunks: []const u32 = &.{},
    /// Decoded intervals [start,end), sorted, nonempty and strictly separated; an empty list uses prompt arithmetic.
    decode_spans: []const [2]u32 = &.{},
    /// Most reply tokens inside a think block before the engine writes ``think_close`` (0: no limit).
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: ?u32 = null,
    /// Reply tokens after ``think_end`` at most, within ``max_tokens`` (0: ``max_tokens`` alone; Info.answer_limit).
    answer_max: u32 = 0,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
    /// With ``loop_guard``: the reply ends where the guard fires, with no forced close and no answer after it.
    loop_stop: bool = false,
    /// The prompt row (the assistant header's last) where routed lessons decide whether they act (null: none acts).
    decide: ?u32 = null,
    /// The model as shipped: no learned neuron acts on its rows, no prompt state is kept or reused (a lesson's writer).
    untaught: bool = false,
    /// Logprob rows for each reply token with this many best tokens (0..20), when ``Info.logprobs``; null: none.
    logprobs: ?u8 = null,
    /// Rows for each prompt token after the first, with ranks and this many best tokens, when ``Info.prompt_logprobs``.
    prompt_logprobs: ?u8 = null,
    /// Images whose rows replace placeholder tokens of ``prompt``, in order; only when ``Info.vision`` is set.
    images: []const Image = &.{},
    /// The prompt as the prompt cache matches it (each image's placeholders by its content); empty: ``prompt``.
    cache_key: []const u32 = &.{},

    /// The prompt as the prompt cache matches it: an image request's placeholders stand for their images.
    pub fn cacheKey(r: *const Request) []const u32 {
        return if (r.cache_key.len > 0) r.cache_key else r.prompt;
    }

    /// No token is drawn: the backend answers from the prompt's rows (an embedding or a readout).
    pub fn promptOnly(r: *const Request) bool {
        return r.embedding != null or r.readout != null;
    }

    /// The prompt length its kept states count as: its own, or for a warm pass the request whose reply it prefills.
    pub fn bornLen(r: *const Request) u32 {
        return if (r.born > 0) r.born else @intCast(r.prompt.len);
    }
};
