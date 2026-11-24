//! The native server's configuration, kept beside the HTTP connection loop.
const std = @import("std");
const Value = @import("json").Value;
const http_conn = @import("http_conn.zig");
const request_guard = @import("request_guard.zig");
const Identity = @import("identity.zig").Identity;

/// auto compacts near the window. fraction compacts past that share of it.
pub const CompactAt = union(enum) { auto, fraction: f64 };

pub const Config = struct {
    browser: request_guard.Policy = .{},
    allowed_controls: ?@import("drama_allow.zig").Policy = null,
    redraw: @import("speech_redraw.zig").Policy = .{},
    sse_keepalive_s: f64 = 15,
    served_name: []const u8,
    model_dir: []const u8 = "",
    /// --voices, --default-voice, --request-voices and --manage-voices; the last two off keep the startup catalog.
    voice: @import("voice_cli.zig").Config = .{},
    /// The served name first, then aliases, without repeats.
    model_ids: []const []const u8,
    /// Ids a request may name for background priority by default (--name-priority); each is in model_ids.
    background_ids: []const []const u8 = &.{},
    default_max_tokens: i64 = 4096,
    enable_thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    /// With thinking on, reply tokens the reasoning may use beyond the request's max_tokens (0: none).
    thinking_reserve: i64 = 0,
    loop_guard: bool = false,
    /// --loop-guard-stop: a guarded reply ends where the guard fires, with a loop error and no answer or calls.
    loop_stop: bool = false,
    tool_think_close: bool = false, // --tool-think-close: a required or named call closes thinking in time
    keep_warm_s: i64 = 900, // seconds the idle keepalive runs after the last request ends (0: off)
    /// Seconds of idle past the last request before the engine closes; the next request reloads it (0: never).
    idle_unload_s: i64 = 0,
    dashboard: bool = false,
    /// Sampling when a request names none (generation_config.json and the serve flags), or null for greedy.
    default_sampling: ?Value = null,
    /// --instruct-sampling's set, for a sampled chat request with thinking off (family_sampling.resolveInstruct).
    instruct_sampling: ?Value = null,
    use_drafts: bool = true,
    /// --vision-urls: image parts may name public HTTPS URLs, fetched before admission (media_fetch.zig).
    vision_urls: bool = false,
    /// --vision-pdf: file/input_file and data:application/pdf parts are rendered via PDFium (pdf.zig).
    vision_pdf: bool = false,
    seed_salt: i64 = 0,
    silence_mid: bool = false,
    silence_guard: bool = false,
    cfg: ?@import("drama_cfg.zig").Options = null,
    /// The flock a voice holds while it synthesizes, so batch work on this GPU gives way (gpu_turn.zig).
    speech_lock: []const u8 = @import("gpu_turn.zig").path,
    prompt_seed: bool = false, // --prompt-seed: historical prompt-keyed sampling when a request omits its seed
    cache_salt_per_key: bool = false, // --cache-salt-per-key: the matched API key salts each request's prompt cache
    /// TENSORFOLD_REQUEST_LOG: where chat and completion bodies are appended; null: nowhere.
    request_log: ?[]const u8 = null,
    /// TENSORFOLD_REQUEST_LOG_REPLIES=1: each finished reply is appended beside its request; default 0.
    request_log_replies: bool = false,
    journal: ?*@import("request_journal.zig").Journal = null, // serve owns TENSORFOLD_REQUEST_JOURNAL
    /// TENSORFOLD_PHASE_METRICS=1: /metrics adds the queue phase's histogram; unset, its families are as before.
    phase_metrics: bool = false,
    /// The Spark server's keys in each reply's ``tensorfold`` block (spark_wire.zig): DeepSeek-V4, or TENSORFOLD_WIRE.
    spark_stats: bool = false,
    /// Null keeps every reply identical to a server that has no compaction.
    compact_at: ?CompactAt = null,
    /// Tokens of the recent tail kept whole. Null uses min(20000, a quarter of the window).
    compact_keep: ?u32 = null,
    /// Directory of the stored note, or null when notes are not stored.
    compact_memory: ?[]const u8 = null,
    /// --slide-graph: the file holding the Sliding Weights fact graph; null keeps the graph in memory.
    slide_graph: ?[]const u8 = null,
    slide_drift_gate: bool = false,
    slide_take_out_check: bool = false, // TF_SLIDE_TAKE_OUT_CHECK=1 with --slide, read once at start
    slide_forget_fail: u32 = 0, // a fact a delete fails to check, to test its put-back (0: none; test-hook builds)
    /// --cortheon: Cortheon's program and server-run web research for chat requests; off changes no reply.
    cortheon: bool = false,
    /// BRAVE_SEARCH_API_KEY: Cortheon searches through Brave instead of DuckDuckGo.
    cortheon_search_key: ?[]const u8 = null,
    /// --learn-conversations: /v1/imprint answers; without it each of its routes refuses and names the flag.
    saved_conversations: bool = false,
    timeouts: http_conn.Timeouts = .{},
    max_connections: u32 = @import("http_admission.zig").default_limit,
    /// --max-connections-per-address, --trusted-proxy: one address's share from three quarters full.
    peering: @import("http_admission.zig").Peering = .{},
    /// --max-inflight, --max-inflight-per-key, --requests-per-second, --request-burst; all zero: no admission.
    admission: @import("request_admission.zig").Limits = .{},
    /// The open /health names these after its other keys; empty fields are left out.
    identity: Identity = .{},
    /// The process environment GET /v1/benchmark/manifest reads (revision and the hardware probe); null: none.
    environ: ?*const std.process.Environ.Map = null,
    /// TENSORFOLD_TOOL_ID_KEY_FILE's key (tool_id_key.zig), which `Server.init` takes and zeroes; null: a random key.
    tool_id_key: ?*@import("ids.zig").Key = null,
};
