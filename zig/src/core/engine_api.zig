//! The engine a server drives (``Engine``), and ``LaneHost``: the lane core served on one thread, rounds left to it.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// The server's name for one request, unique while the process lives.
pub const tfvideo = @import("tfvideo.zig");

pub const Id = u64;

/// One request's sampling (Python's exact_sampling.Sampling); a request without one decodes greedily.
pub const Sampling = lanes.Sampling;

/// One image of a request (``Request.images``): the tower's input and the placeholder rows its output replaces.
pub const Image = lanes.stream.Image;

/// A prompt's reproducible seed (``seed_for``), for requests that name none.
pub const seedFor = lanes.sampling.seedFor;

/// A committed token's log probability under the target's raw distribution, with the best tokens at its position.
pub const LogprobRow = lanes.LogprobRow;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
};

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar, structural_tag },
    /// The schema (JSON text), regex, choices (a JSON array), EBNF or structural tag; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
    /// The server's compiler, which has compiled `text` once already (the engine's compile is then a cache hit).
    compiler: ?*grammar.Compiler = null,
};

pub const grammar = lanes.grammar;

/// A reply to decode. The request and every slice in it stay valid until its ``finished`` event.
pub const Request = struct {
    prompt: []const u32,
    max_tokens: u32,
    sampling: ?Sampling = null,
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
    /// Where the prompt's prefill chunks start after 0 (Python's PrefillPlan at ``Info.prefill_step``); empty: the engine's own.
    chunks: []const u32 = &.{},
    /// Decoded intervals [start,end), sorted, nonempty and strictly separated; an empty list uses prompt arithmetic.
    decode_spans: []const [2]u32 = &.{},
    /// Most reply tokens inside a think block before the engine writes ``think_close`` (0: no limit).
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: ?u32 = null,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
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

    /// The prompt length its kept states count as: its own, or for a warm pass the request whose reply it prefills.
    pub fn bornLen(r: *const Request) u32 {
        return if (r.born > 0) r.born else @intCast(r.prompt.len);
    }
};

/// An image prepared for the engine's tower (``Vision.prepare``): patches, patch grid, tokens and a content hash.
pub const PreparedImage = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// Image input an engine serves (``Info.vision``): image bytes to tower input, and what a request may hold.
pub const Vision = struct {
    ctx: *anyopaque,
    /// The image's bytes (any format the platform decodes) at most ``max_tokens`` tokens; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!PreparedImage,
    /// The token the chat template writes once per image, expanded to the image's tokens.
    image_token: u32,
    /// The tokens the template wraps an image in (the image token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Tokens the server writes around each image's rows (Nemotron Omni's <img> and </img>); null: the template's own.
    wrap: ?[2]u32 = null,
    /// Image tokens one request may hold in all, and images.
    image_tokens: u32,
    max_images: u32,
    /// Video input, when the engine's family reads it.
    video: ?VideoOffer = null,
    /// The engine's words for a prepare error the server cannot word (a decoder library it lacks); null: the server's.
    explain: ?*const fn (err: anyerror) ?[]const u8 = null,
};

/// A video's facts as the server's decoder reads them, and its frames on request.
pub const VideoSource = struct {
    frames: u64, // the stream's frame count
    rate_num: u64, // its average frame rate, num / den
    rate_den: u64,
    width: u32,
    height: u32,
    ctx: *anyopaque,
    /// The frames at ``indices`` (ascending, distinct), each to ``sink`` in order.
    decode: *const fn (ctx: *anyopaque, indices: []const u32, sink: FrameSink) anyerror!void,
};

/// Where a decoded frame goes: RGB rows of ``width`` * 3 bytes, ``stride`` bytes apart.
pub const FrameSink = struct {
    ctx: *anyopaque,
    take: *const fn (ctx: *anyopaque, k: usize, rgb: []const u8, width: u32, height: u32, stride: usize) anyerror!void,
};

/// A video prepared for the tower: each group of frames an image, and each group's time in seconds.
pub const PreparedVideo = struct { groups: []PreparedImage, times: []f64 };

/// Video input an engine serves (``Vision.video``): how a decoded video becomes images and how the prompt holds them.
pub const VideoOffer = struct {
    /// ``src`` at most ``max_tokens`` tokens and ``max_frames`` sampled frames; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, src: VideoSource, max_tokens: u32, max_frames: u32) anyerror!PreparedVideo,
    /// The token the chat template writes once per video, expanded to its groups.
    video_token: u32,
    /// The tokens around each group's image tokens; the group's time follows as text.
    group_open: u32,
    group_close: u32,
    /// The tokens the template wraps a video in (the video token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Video tokens one request may hold in all, videos, and frames sampled a video.
    video_tokens: u32,
    max_videos: u32,
    max_frames: u32,
};

/// --vision: image and video input's limits (``Open.vision``); image_tokens_set: the flag given, over a family default.
pub const VisionOpen = struct { image_tokens: u32 = 4096, image_tokens_set: bool = false, max_images: u32 = 4, video_tokens: u32 = 16384, max_videos: u32 = 2, max_frames: u32 = 256 };

pub const Reason = enum { stop, length, cancelled, failed };

/// What a finished reply's rounds did, for its runtime fields and /metrics.
pub const Stats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    min_rows: u32 = 0,
    prefill_widths: []const u32 = &.{},
    prefill_raised: []const bool = &.{},
    /// The loop period that ended or interrupted the think block, if any.
    loop_period: ?u32 = null,
    /// Prompt prefill duration measured by the engine host, in seconds.
    prefill_seconds: ?f64 = null,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
};

pub const Event = union(enum) {
    /// The prompt is in the cache: ``cached`` of its tokens came from a kept prefix.
    prefilled: u32,
    /// The next ``tokens`` event's logprob rows, one a token (``logprobs`` requests); valid only during the call.
    logprobs: []const LogprobRow,
    /// The prompt's rows, one a prompt token after the first, right after ``prefilled`` (``prompt_logprobs`` requests).
    prompt_logprobs: []const LogprobRow,
    /// Tokens committed for this request, in order; valid only during the call.
    tokens: []const u32,
    /// The reply ended; nothing follows. ``message`` says why a failed reply failed.
    finished: struct { reason: Reason, stats: Stats = .{}, message: []const u8 = "" },
};

/// Where a request's events go. Called on the engine's thread: it must return at once and not re-enter the engine.
pub const Sink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, id: Id, event: *const Event) void,
};

/// What the engine serves, fixed once it has loaded.
pub const Info = struct {
    name: []const u8 = "lanes",
    lanes: u32 = 1,
    /// Prompt plus reply tokens one request may use (0: no limit).
    context_window: u32 = 0,
    context_fitted: bool = false,
    /// What the engine enforces; the server refuses requests that need more.
    call_gates: bool = false,
    /// The engine observes loop_guard requests in its lane stream.
    loop_guard: bool = false,
    /// The engine enforces a token budget while the think block is open.
    thinking_budget: bool = true,
    structures: bool = false,
    /// Requests may ask for ``logprobs`` (the engine's lanes give the target's rows).
    logprobs: bool = false,
    /// Requests may ask for ``prompt_logprobs`` (a prompt pass from 0 that reads every row).
    prompt_logprobs: bool = false,
    /// Prompt rows a prefill chunk at most, for the server's chunk starts (0: the engine cuts prompts itself).
    prefill_step: u32 = 0,
    /// The engine keeps prompt states between requests, so ``prefilled`` counts the tokens it restored.
    prompt_cache: bool = false,
    /// The policy the engine runs under, one line (`/health` shows it); empty: none.
    policy: []const u8 = "",
    /// A line the server prints once at startup (the engine's memory plan); empty: none.
    startup: []const u8 = "",
    /// A prompt-only request (max_tokens 0) keeps its end state: the server prefills each reply for the next turn.
    warm_turns: bool = false,
    /// The engine decodes plain whatever a request asks: no drafter loaded, or this chip is faster plain.
    plain_only: bool = false,
    /// The immutable retained-prefix plan applied at startup, not a process memory limit or current cache occupancy.
    prompt_cache_plan: ?PromptCachePlan = null,
    /// The engine scores decisions (`score`) and writes no text: the server refuses text requests.
    decides: bool = false,
    /// Decisions may carry pictures (`Decision.pictures`).
    pictures: bool = false,
    /// Image input (--vision); null: text only.
    vision: ?Vision = null,
    /// The engine decodes greedily only: the server refuses a request with a temperature before admitting it.
    greedy_only: bool = false,
    /// Whether a request's caches fit beside the running streams' now (null: a free lane does); else it waits in order.
    admit: ?Admit = null,
};

pub const Admit = struct {
    ctx: *anyopaque,
    fits: *const fn (ctx: *anyopaque, prompt: u32, max_tokens: u32) bool,
};

pub const PromptCachePlan = struct {
    source: enum { physical_footprint, explicit, metal_working_set },
    budget_bytes: u64,
    explicit_budget: bool = false,
    over_cap: ?bool = null,
    ram_bytes: ?u64 = null,
    ready_footprint_bytes: ?u64 = null,
    cap_bytes: ?u64 = null,
    room_bytes: ?u64 = null,
    margin_bytes: ?u64 = null,
};

/// A checkpoint family an engine reads: its config ``model_type`` and weight formats, as gate entries name them.
pub const Family = struct { model_type: []const u8, formats: []const []const u8 };

/// The math an NVFP4 checkpoint runs: its own (FP4 rows into NVFP4 layers, FP8 into FP8 ones), or bf16 rows throughout.
pub const Precision = enum { checkpoint, full };

/// What a server asks of the engine it opens: the checkpoint, and the serve flags an engine reads.
pub const Open = struct {
    dir: []const u8,
    model_type: []const u8,
    context: ?i64 = null,
    lanes: u32 = 8,
    /// --parallel named a number: an engine whose memory fits fewer streams refuses instead of serving fewer.
    lanes_fixed: bool = false,
    drafts: bool = true,
    /// --drafter: a draft model's directory for families that load one (Qwen3.8-27B's DFlash2); null: none.
    drafter: ?[]const u8 = null,
    /// --drafter-bits: 4 prepares the drafter's weights in 4 bits; 0 keeps them bf16.
    drafter_bits: u8 = 4,
    /// --record: each stream's committed rows and tapped features written under this folder, for drafter training.
    record: ?[]const u8 = null,
    speed_up: ?[]const u8 = null,
    prompt_cache_gib: ?f64 = null,
    prompt_cache_over_cap: bool = false,
    /// --learn: where shared prompt states are kept on disk for later sessions and servers (null: off).
    learn: ?[]const u8 = null,
    /// --learn-min-free-gib: filesystem space kept free before any learned-state write.
    learn_min_free_gib: f64 = 4,
    /// --learn-gib: what learned states may take on disk, every model and build together.
    learn_gib: f64 = 32,
    /// --learn-conversations: every kept state written behind to disk too, not only shared cuts (under `learn`).
    learn_conversations: bool = false,
    /// --learn-daily-gib: what learned states may write a day under the root, every model together.
    learn_daily_gib: f64 = 32,
    /// --slide: Sliding Weights learns into the served weights, live (off: learn requests are refused).
    slide: bool = false,
    /// --recall: the server attaches Recall's record once the engine loads; an engine that reads none refuses first.
    recall: bool = false,
    /// --device and --segments (CUDA); null: the backend's environment fallback, then its default.
    device: ?u32 = null,
    segments: ?u32 = null,
    /// --precision (CUDA): an NVFP4 checkpoint's math, NVIDIA's own or full precision; null: the family's default.
    precision: ?Precision = null,
    /// --mtp-drafts and --mtp-confidence; null: the engine's defaults.
    mtp_drafts: ?u32 = null,
    mtp_confidence: ?f64 = null,
    /// Kept prompt entries (--checkpoint-slots); null: the engine's plan.
    keep: ?i64 = null,
    /// Tensor parallelism: this process is `rank` of `tp`; rank 0 listens on `master`:`master_port` for the others.
    tp: u32 = 1,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    /// --policy: the engine's policy as `key=value,...`; empty: its defaults.
    policy: []const u8 = "",
    /// --vision: load the checkpoint's image tower (null: text only; an engine without one refuses).
    vision: ?VisionOpen = null,
};

/// A packed decision: a trunk (pictures expand at `at`), a branch a question, and each branch's labels (`counts`).
pub const Decision = struct {
    trunk: []const u32,
    at: usize = 0,
    pictures: []const []const u8 = &.{}, // data URLs, in prompt order
    branches: []const []const u32 = &.{},
    counts: []const usize = &.{},
    labels: []const u32 = &.{},
    logits: []f64 = &.{},
    read: usize = 0, // out: the trunk's tokens with its pictures expanded, plus every branch's
    problem: []const u8 = "", // out: why the engine refused (error.Refused), in `arena`
    arena: ?std.mem.Allocator = null, // the request's, for what the engine hands back
};

/// An opened engine; ``close`` stops its thread and frees its backend.
pub const Opened = struct {
    engine: Engine,
    close: *const fn (ctx: *anyopaque) void,
    ctx: *anyopaque,
    /// A drain's deadline: rounds end at the next boundary, what is left fails with `reason`; null: `close` alone.
    halt: ?*const fn (ctx: *anyopaque, reason: []const u8) void = null,
};

pub const Memory = struct { active: u64 = 0, cache: u64 = 0, peak: u64 = 0, pinned: u64 = 0, limit: ?u64 = null };

/// A backend's words for its own refusals (a request it cannot serve); null: the error's name.
pub const Explain = struct {
    ctx: ?*anyopaque = null,
    text: *const fn (ctx: ?*anyopaque, err: anyerror) ?[]const u8,
};

/// A backend's own memory counts for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const MemorySource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque, reset_peak: bool) ?Memory,
};

pub const Status = struct {
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    /// Filling prompts' total tokens and those still to prefill (null: no prompt is going in).
    fill_prompt_tokens: ?u32 = null,
    fill_tokens_left: ?u32 = null,
    /// A prompt pass is running whose backend reports no `fill_at`: the totals stay null rather than guessed.
    fill_unknown: bool = false,
    /// Changes whenever the filling set does: a reader pairs its drain samples with one fill.
    fill_run: u64 = 0,
};

pub const SubmitError = error{ Closed, Busy, InvalidSpans, Stale }; // Stale: a learner's put-back has not reached the device

/// What a reply refused or ended for `Stale` says, a submit's error and a queued job's failure alike.
pub const stale_message = "a Living Weights delete could not be put back on the device: send the delete again to retry it, or restart the server";

/// A worked example for the learner: token ids, the answer from `start`; `decide`: its assistant header's last row.
pub const Example = struct { ids: []const u32, start: u32, decide: u32, subject: u32 = 0, value: u32 = 0 }; // subject: the row of its last token; value: the row before its answer's value (0: none)

/// One fact's lesson (Sliding Weights): answers, held-out answers, near misses, and prompts that must not move.
pub const LearnRequest = struct {
    train: []const Example = &.{},
    think: []const Example = &.{}, // the facts' answers with thinking on, learned beside `train` (one of each a step)
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{}, // answers of every kind this lesson's change must leave as they are
    asked: []const Example = &.{}, // more wordings of the fact's questions, for its router (prompts only)
    unasked: []const Example = &.{}, // questions it must not answer, for its router (prompts only)
    kept: []const Example = &.{}, // questions the last lesson took, with the answers they had (a reroute's)
    undo: bool = false, // instead take the last lesson's change back, as if it never ran
    steps: u32 = 60, // bounded steps this time at most
    more: bool = false, // more steps on the last lesson's rows, which are not captured again
    reroute: bool = false, // the last lesson's router fitted again with `unasked` among what it must not take
    drop: bool = false, // the last lesson taken out whole
    save: bool = false, // every kept lesson written beside the model's shards, each with its router
    retire: []const Neuron = &.{}, // instead zero these neurons of facts a newer fact replaces
    unretire: bool = false, // instead put back the neurons the last retire zeroed (the newer fact was not learned)
    forget: []const Neuron = &.{}, // instead zero these neurons' input weights (a fact deleted on request), kept to put back
    unforget: bool = false, // instead put back, bit for bit, what the last forget changed (a kept fact did not come back)
};

/// A neuron a pocket lesson placed (Sliding Weights): its layer, its index in that layer's pocket, the row it keys on.
pub const Neuron = struct { layer: u16, neuron: u16, key: NeuronKey };
pub const NeuronKey = enum { reply, subject, value };

/// A pocket lesson's stage, as it runs: its rows read, its neurons fitted, their outputs, its held-out answers judged.
pub const LearnPhase = enum { reading, fitting, values, checking };

/// Where a pocket lesson stands: its unit `at` of `of` (0: unknown), its outputs' loss, its neurons once placed.
pub const LearnProgress = struct { phase: LearnPhase, at: u32 = 0, of: u32 = 0, loss: f32 = 0, neurons: []const Neuron = &.{} };

/// A lesson's outcome in order, ending with `done` (a message says why it failed); slices live only during the call.
pub const LearnEvent = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32, neurons: []const Neuron = &.{} }, // neurons: a pocket's
    progress: LearnProgress,
    done: struct { message: []const u8 = "" },
};

/// Where a learn request's events go; called on the engine's thread, so it must return at once.
pub const LearnSink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, event: *const LearnEvent) void,
};

pub const LearnError = error{ Closed, Busy, Unsupported };

/// A family's learner, which LaneHost steps one bounded unit at a time while no stream decodes.
pub const Learner = struct {
    ctx: *anyopaque,
    /// Start a job; its events, `done` last, go to `sink` from later steps.
    begin: *const fn (ctx: *anyopaque, request: *const LearnRequest, sink: LearnSink) anyerror!void,
    /// One unit of the job: `changed` when weights moved (kept prompt states are then dropped), `done` when it ended.
    step: *const fn (ctx: *anyopaque) Step,
    /// End the open job now; it still sends `done`, with the reason.
    abort: *const fn (ctx: *anyopaque) void,

    /// stale: the device lacks a put-back the learner holds, so replies are refused until a later step clears it.
    pub const Step = struct { done: bool, changed: bool, stale: bool = false };
};

pub const Engine = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        info: *const fn (ctx: *anyopaque) Info,
        submit: *const fn (ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void,
        /// Ends a request between rounds; its ``finished`` (cancelled) still arrives. Unknown ids are ignored.
        cancel: *const fn (ctx: *anyopaque, id: Id) void,
        status: *const fn (ctx: *anyopaque, out: *Status, stream_tokens: []u32) void,
        /// Device memory in bytes, the peak since the last reset; null when the backend keeps no count.
        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
        /// The engine's queue as a keepalive target, or null when the backend is not Metal.
        keepalive: ?*const fn (ctx: *anyopaque) ?keepalive.Target = null,
        /// One logit per label at the row after `prompt`, and the vocabulary logsumexp when `mass` (else 0).
        score: ?*const fn (ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 = null,
        /// Each branch's label logits at its last row after the decision's shared trunk.
        decide: ?*const fn (ctx: *anyopaque, d: *Decision) error{ Failed, Refused }!void = null,
        /// Queue `request` for the family's learner; it stays valid until `done`. Null: this engine does not learn.
        learn: ?*const fn (ctx: *anyopaque, request: *const LearnRequest, sink: LearnSink) LearnError!void = null,
        /// Recall's record for every later request's drafts and writes, before any request (null: no record read).
        recall: ?*const fn (ctx: *anyopaque, store: *lanes.recall_store.Store) void = null,
        /// Queue a question about kept states for the engine thread (it stays valid until `done`); false: refused.
        imprint: ?*const fn (ctx: *anyopaque, ask: *ImprintAsk) bool = null,
        /// A planned stop's first half: no more work, disk writes finished under a deadline, the engine thread joined.
        drain: ?*const fn (ctx: *anyopaque) void = null,
    };

    pub fn info(e: Engine) Info {
        return e.vtable.info(e.ctx);
    }
    pub fn submit(e: Engine, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        try @import("cache_modes.zig").validate(request.decode_spans);
        return e.vtable.submit(e.ctx, id, request, sink);
    }
    pub fn cancel(e: Engine, id: Id) void {
        e.vtable.cancel(e.ctx, id);
    }
    pub fn status(e: Engine, out: *Status, stream_tokens: []u32) void {
        e.vtable.status(e.ctx, out, stream_tokens);
    }
    pub fn memory(e: Engine, reset_peak: bool) ?Memory {
        return e.vtable.memory(e.ctx, reset_peak);
    }
    /// The engine's queue as a keepalive target, or null when the backend is not Metal.
    pub fn keepaliveTarget(e: Engine) ?keepalive.Target {
        const f = e.vtable.keepalive orelse return null;
        return f(e.ctx);
    }

    /// Hands the engine Recall's record; false when it drafts from none (its own copies only).
    pub fn attachRecall(e: Engine, store: *lanes.recall_store.Store) bool {
        const f = e.vtable.recall orelse return false;
        f(e.ctx, store);
        return true;
    }

    /// Queues `ask` for the engine thread; false when the engine answers none (its `done` never comes).
    pub fn imprintAsk(e: Engine, ask: *ImprintAsk) bool {
        const f = e.vtable.imprint orelse return false;
        return f(e.ctx, ask);
    }

    /// Finishes the engine's disk writes before the process ends; a no-op where it keeps none.
    pub fn drain(e: Engine) void {
        if (e.vtable.drain) |f| f(e.ctx);
    }

    /// The decision hook, or Unsupported when this engine does not score.
    pub fn score(e: Engine, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{ Failed, Unsupported }!f64 {
        const f = e.vtable.score orelse return error.Unsupported;
        return f(e.ctx, prompt, labels, logits, mass);
    }

    pub fn decide(e: Engine, d: *Decision) error{ Failed, Refused, Unsupported }!void {
        const f = e.vtable.decide orelse return error.Unsupported;
        return f(e.ctx, d);
    }

    /// The Sliding Weights hook, or Unsupported when this engine does not learn.
    pub fn learn(e: Engine, request: *const LearnRequest, sink: LearnSink) LearnError!void {
        const f = e.vtable.learn orelse return error.Unsupported;
        return f(e.ctx, request, sink);
    }
};

/// A backend's own driver for a lone greedy stream (the GPU round), which LaneHost runs while the stream is alone.
pub const Lone = struct {
    ctx: *anyopaque,
    sampled: bool = false, // it drives sampled streams too
    logprobs: bool = false, // it gives each token its logprob row too
    structures: bool = false, // it masks each window's rows by the stream's grammar too
    min_budget: u32 = 0, // thinking budgets above this and loop guards: it hands the stream over before a forced close (0: never)
    resumes: bool = false, // it takes the lane core's stream back between rounds (LoneHooks.resumed); else only opens
    /// Prefills (or resumes) `s` and decodes it until it finishes (false) or `hooks.yield` hands it to the lane core.
    run: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool,
};

/// What a lone driver tells the host between rounds: tokens landed; and asks: hand the stream over now?
pub const LoneHooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
    resumed: bool = false, // the stream comes from the lane core between rounds: no prompt pass, its drafts the core's
};

/// The lane core served to the HTTP threads (lane_host.zig).
pub const LaneHost = @import("lane_host.zig").LaneHost;
/// An opt-in family round planner for the lane host, in place of its `prefill_step` chunks (lane_rounds.zig).
pub const Rounds = @import("lane_host.zig").Rounds;

/// Exact prompt reuse between requests, for any family (prompt_cache.zig).
pub const prompt_cache = @import("prompt_cache.zig");

/// The same for a paged cache over the backend's own KV pages (prompt_radix.zig).
pub const prompt_radix = @import("prompt_radix.zig");

/// Decoded prompt rows as canonical spans, for a cache keyed by the arithmetic that made each row (cache_modes.zig).
pub const cache_modes = @import("cache_modes.zig");

/// Learned prompt-cache states on disk (prompt_imprint.zig), and when a shared cut goes there (prompt_learn.zig).
pub const prompt_imprint = @import("prompt_imprint.zig");
pub const prompt_learn = @import("prompt_learn.zig");

/// A question about a conversation's kept states, in RAM and on disk (rows_ask.zig).
pub const ImprintAsk = @import("rows_ask.zig").Ask;
pub const ImprintOp = @import("rows_ask.zig").Op;
pub const ImprintDisk = @import("rows_ask.zig").Disk;

/// Recall's record of the ids earlier requests read and wrote, for drafts (not the prompt cache's learned states).
pub const recall_store = lanes.recall_store;

/// The idle keepalive's ticker and target contract.
pub const keepalive = @import("keepalive.zig");

/// A model folder's original files kept before --slide first teaches it, its hold and its rollback (originals.zig).
pub const originals = @import("originals.zig");

test {
    _ = @import("lane_host.zig");
    _ = @import("lane_host_test.zig");
    _ = @import("lane_rounds_test.zig");
    _ = @import("lane_halt_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prompt_radix.zig");
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
    _ = @import("originals_test.zig");
}
