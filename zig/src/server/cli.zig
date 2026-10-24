//! ``tensorfold serve`` flags: one table for parsing them and for ``capabilities --json``.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    value, // --flag VALUE or --flag=VALUE
    store_true, // --flag
    append, // repeatable --flag VALUE
};

pub const Flag = struct {
    name: []const u8,
    kind: Kind = .value,
    /// argparse's choices; empty: any value of the flag's type.
    choices: []const []const u8 = &.{},
    /// This binary honours the flag (and lists it in ``capabilities``); the rest are refused as unsupported.
    native: bool = false,
    /// The values it honours when only some of the choices.
    native_values: ?[]const []const u8 = null,
};

const backend_values: []const []const u8 = if (builtin.os.tag == .macos) &.{ "auto", "mlx" } else if (served("--backend=rocm")) &.{ "auto", "rocm" } else &.{ "auto", "cuda" };

/// The flags past the common ones the binary's backend serves, from its root's `serves` (test roots: none).
const native_serves: []const []const u8 = if (@hasDecl(@import("root"), "serves")) @import("root").serves else &.{};

fn served(comptime name: []const u8) bool {
    return servedBy(native_serves, name);
}

fn servedBy(list: []const []const u8, name: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

/// The CUDA build's own flags and variables (the Metal build refuses them).
const cuda_build = builtin.os.tag == .linux;

/// Python serve flags plus this binary's native-only flags; supported flags are marked native.
pub const flags = [_]Flag{
    .{ .name = "--host", .native = true },
    .{ .name = "--allowed-host", .kind = .append, .native = true },
    .{ .name = "--allowed-origin", .kind = .append, .native = true },
    .{ .name = "--allowed-controls", .native = true },
    .{ .name = "--port", .native = true },
    .{ .name = "--name", .native = true },
    .{ .name = "--chat-template", .native = true },
    .{ .name = "--voices", .native = true },
    .{ .name = "--default-voice", .native = true },
    .{ .name = "--alias", .kind = .append, .native = true },
    .{ .name = "--api-key", .kind = .append, .native = true },
    .{ .name = "--api-key-file", .native = true },
    .{ .name = "--metrics-open", .kind = .store_true, .native = true },
    .{ .name = "--dashboard", .kind = .store_true, .native = true },
    .{ .name = "--vision", .kind = .store_true, .native = true },
    .{ .name = "--vision-urls", .kind = .store_true, .native = true },
    .{ .name = "--vision-offload", .kind = .store_true },
    .{ .name = "--vision-max-images", .native = true },
    .{ .name = "--vision-max-videos", .native = true },
    .{ .name = "--vision-image-tokens", .native = true },
    .{ .name = "--vision-video-tokens", .native = true },
    .{ .name = "--context", .native = true },
    .{ .name = "--max-frames", .native = true },
    .{ .name = "--speed-up", .native = !cuda_build },
    .{ .name = "--max-tokens", .native = true },
    .{ .name = "--temperature", .native = true },
    .{ .name = "--prompt-seed", .kind = .store_true, .native = true },
    .{ .name = "--top-p", .native = true },
    .{ .name = "--top-k", .native = true },
    .{ .name = "--min-p", .native = true },
    .{ .name = "--thinking", .kind = .store_true, .native = true },
    .{ .name = "--no-thinking", .kind = .store_true, .native = true },
    .{ .name = "--reasoning-effort", .choices = &.{ "low", "medium", "high", "xhigh" }, .native = true },
    .{ .name = "--thinking-budget", .native = true },
    .{ .name = "--thinking-reserve", .native = true },
    .{ .name = "--loop-guard", .kind = .store_true, .native = true },
    .{ .name = "--no-drafts", .kind = .store_true, .native = true },
    .{ .name = "--keep-warm", .native = true },
    .{ .name = "--silence-floor-dbfs", .native = true },
    .{ .name = "--silence-seconds", .native = true },
    .{ .name = "--silence-onset-s", .native = true },
    .{ .name = "--silence-onset-voiced-s", .native = true },
    .{ .name = "--silence-mid-s", .native = true },
    .{ .name = "--silence-mid-min-voiced-s", .native = true },
    .{ .name = "--silence-redraws", .native = true },
    .{ .name = "--silence-redraw-budget-s", .native = true },
    .{ .name = "--silence-redraw-controls", .native = true },
    .{ .name = "--short-reply-ratio", .native = true },
    .{ .name = "--short-reply-chars-per-s", .native = true },
    .{ .name = "--register-redraw-st", .native = true },
    .{ .name = "--register-redraw-high", .native = true },
    .{ .name = "--register-high-hz", .native = true },
    .{ .name = "--idle-unload", .native = true },
    .{ .name = "--compact-at", .native = true },
    .{ .name = "--compact-keep", .native = true },
    .{ .name = "--compact-memory", .native = true },
    .{ .name = "--slide", .kind = .store_true, .native = true },
    .{ .name = "--slide-graph", .native = true },
    .{ .name = "--drafter", .native = true },
    .{ .name = "--drafter-bits", .choices = &.{ "0", "4" }, .native = true },
    .{ .name = "--record", .native = true },
    .{ .name = "--kv-cache", .choices = &.{ "bf16", "fp8" }, .native = true },
    .{ .name = "--prompt-precision", .choices = &.{ "bf16", "int8" }, .native = true },
    .{ .name = "--mtp-drafts", .native = served("--mtp-drafts") },
    .{ .name = "--mtp-confidence", .native = served("--mtp-confidence") },
    .{ .name = "--lane-kernels", .choices = &.{ "auto", "on", "off" } },
    .{ .name = "--prompt-cache-gib", .native = true },
    .{ .name = "--prompt-cache-over-cap", .kind = .store_true, .native = true },
    .{ .name = "--checkpoint-slots", .native = served("--checkpoint-slots") },
    .{ .name = "--spill-gib" },
    .{ .name = "--snapshot-dir", .native = true, .native_values = &.{"none"} },
    .{ .name = "--max-snapshots", .native = true, .native_values = &.{"0"} },
    .{ .name = "--parallel", .native = true },
    .{ .name = "--decode-share" },
    .{ .name = "--prefill-pass" },
    .{ .name = "--pass-cache-gib" },
    .{ .name = "--mlx-cache-gib" },
    .{ .name = "--ssd-experts" },
    .{ .name = "--ple-on-ssd", .kind = .store_true },
    .{ .name = "--no-update-check", .kind = .store_true, .native = true },
    .{ .name = "--backend", .choices = &.{ "auto", "mlx", "cuda", "rocm" }, .native = true, .native_values = backend_values },
    .{ .name = "--tp", .choices = &.{ "1", "2", "4", "8" }, .native = served("--tp") },
    .{ .name = "--rank", .native = served("--tp") },
    .{ .name = "--master", .native = served("--tp") },
    .{ .name = "--master-port", .native = served("--tp") },
    // the GPU engine's policy switches as key=value pairs, repeatable
    .{ .name = "--policy", .kind = .append, .native = served("--policy") },
    .{ .name = "--kv-dtype", .choices = &.{ "bf16", "int8", "int4" } },
    .{ .name = "--prefill-fp8", .kind = .store_true },
    .{ .name = "--no-prefill-fp8", .kind = .store_true },
    .{ .name = "--precision", .choices = &.{ "checkpoint", "full" }, .native = cuda_build },
    // the CUDA CLI's own: the GPU ordinal, and whole prompt chunks a call runs as staggered segments
    .{ .name = "--device", .native = cuda_build },
    .{ .name = "--segments", .native = cuda_build },
    // the native server's own: shared prompt states (a harness) kept on disk for later sessions and servers
    .{ .name = "--learn", .kind = .store_true, .native = true },
    .{ .name = "--learn-dir", .native = true },
    .{ .name = "--learn-gib", .native = true },
    .{ .name = "--learn-min-free-gib", .native = true },
    .{ .name = "--learn-conversations", .kind = .store_true, .native = true },
    .{ .name = "--learn-daily-gib", .native = true },
    // the native server's own: drafts copied from a record of the token ids earlier requests read and wrote
    .{ .name = "--recall", .kind = .store_true, .native = true },
    .{ .name = "--recall-dir", .native = true },
    .{ .name = "--recall-gib", .native = true },
    .{ .name = "--recall-read-only", .kind = .store_true, .native = true },
    // the native server's own: Cortheon's reasoning program and server-run web research for every chat request
    .{ .name = "--cortheon", .kind = .store_true, .native = true },
    .{ .name = "--notes", .native = true },
};

/// The variables this binary honours as the Python engine does, then the CUDA build's.
pub const env = [_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_NO_LIVE", "TENSORFOLD_SEED_SALT", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_NO_UPDATE_CHECK", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "BRAVE_SEARCH_API_KEY" } ++
    (if (cuda_build) [_][]const u8{ "TF_CUDA_DEVICE", "TF_CUDA_SEGMENTS", "TENSORFOLD_CUDA_KERNELS", "TENSORFOLD_MEMORY_RESERVE_GIB", "TENSORFOLD_CUDA_MEMORY_LIMIT_GB" } else [_][]const u8{});

pub const Args = struct {
    notes: ?[]const u8 = null,
    model: []const u8 = "",
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    name: []const u8 = "",
    chat_template: ?[]const u8 = null,
    voices: ?[]const u8 = null,
    default_voice: []const u8 = "default",
    alias: []const []const u8 = &.{},
    api_key: []const []const u8 = &.{},
    allowed_host: []const []const u8 = &.{},
    allowed_origin: []const []const u8 = &.{},
    allowed_controls: ?@import("drama_allow.zig").Policy = null,
    api_key_file: ?[]const u8 = null,
    metrics_open: bool = false,
    dashboard: bool = false,
    context: ?i64 = null,
    max_frames: ?u32 = null,
    silence: api.speech_guard.Config = .{},
    redraw: @import("speech_redraw.zig").Policy = .{},
    speed_up: ?[]const u8 = null, // speed-up mode: this Mac's settings for the two-Mac link (Flash Next and the GLMs)
    prompt_cache_gib: ?f64 = null, // kept prompt states' budget (0: none; null: what 70% of RAM leaves past the loaded server)
    prompt_cache_over_cap: bool = false, // a --prompt-cache-gib past that is kept, not refused
    learn: bool = false, // keep shared prompt states on disk (--learn-dir: where; it implies --learn)
    learn_dir: ?[]const u8 = null,
    learn_min_free_gib: f64 = 4,
    learn_gib: f64 = 32, // disk for learned states, every model and build together
    learn_conversations: bool = false, // every kept state on disk, not only shared cuts (implies --learn)
    learn_daily_gib: f64 = 32, // what learned states may write a day, every model together
    recall: bool = false, // draft copies from Recall's record of earlier requests (its other flags imply it)
    recall_dir: ?[]const u8 = null, // its directory (null: the tokenizer's own under ~/.cache/tensorfold/recall)
    recall_gib: f64 = 1.0 / 16.0, // the record's disk; its RAM is about twice as much once full
    recall_read_only: bool = false, // read the record, add nothing: what gates and benchmarks run with
    max_tokens: i64 = 4096,
    temperature: ?f64 = null,
    prompt_seed: bool = false,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
    thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    thinking_reserve: i64 = 0,
    loop_guard: bool = false,
    no_drafts: bool = false,
    keep_warm: i64 = 900, // seconds the idle keepalive runs after the last request ends (0: off)
    idle_unload: i64 = 0, // seconds of idle past the last request before the engine closes (0: never; the next request reloads)
    compact_auto: bool = false,
    compact_fraction: ?f64 = null,
    compact_keep: ?u32 = null,
    compact_memory: ?[]const u8 = null,
    drafter: ?[]const u8 = null,
    drafter_bits: u8 = 4,
    record: ?[]const u8 = null, // drafter training rows written under this folder
    kv_cache: []const u8 = "bf16", // fp8: e4m3 attention caches with a scale a row, about twice the context
    prompt_precision: []const u8 = "bf16", // int8: prompt matmuls quantize their inputs to int8 a 64-input group
    slide: bool = false, // Sliding Weights learns what /v1/slide/learn is told into the served weights, live
    slide_graph: ?[]const u8 = null, // the Sliding Weights fact graph's file (null: memory only)
    parallel: []const u8 = "auto",
    backend: []const u8 = "auto",
    device: ?u32 = null,
    segments: ?u32 = null,
    precision: ?api.Precision = null, // an NVFP4 checkpoint's math on CUDA: NVIDIA's own (checkpoint) or full precision
    mtp_drafts: ?u32 = null,
    mtp_confidence: ?f64 = null,
    checkpoint_slots: ?i64 = null,
    tp: u32 = 1,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    policy: []const u8 = "", // every --policy joined by commas
    cortheon: bool = false, // Cortheon's program, research before building, and cortheon_search/cortheon_fetch run by the server
    vision: bool = false, // image input (GLM-5.3-Flash): data-URL image_url parts in user messages
    vision_urls: bool = false, // with --vision, public HTTPS image URLs too (media_fetch.zig)
    vision_max_images: u32 = 4,
    vision_max_videos: u32 = 2,
    vision_image_tokens: u32 = 4096,
    vision_image_tokens_set: bool = false, // --vision-image-tokens given: a family's own default no longer applies
    vision_video_tokens: u32 = 16384, // the visual tokens a request's videos share (the processor's max_image_tokens)
};

/// A usage error's message (argparse's ``error:`` line); the caller exits 2.
pub const Usage = struct { message: []const u8 = "" };

fn find(name: []const u8) ?Flag {
    for (flags) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

fn fail(u: *Usage, a: Allocator, comptime fmt: []const u8, args: anytype) error{ Usage, OutOfMemory } {
    u.message = try std.fmt.allocPrint(a, fmt, args);
    return error.Usage;
}

/// ``serve`` arguments as the switch passes them: the model and full flag names.
pub fn parse(a: Allocator, argv: []const []const u8, u: *Usage) error{ Usage, OutOfMemory }!Args {
    var out: Args = .{};
    var repeated: Repeated = .{};
    var model: ?[]const u8 = null;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (!std.mem.startsWith(u8, token, "--")) {
            if (model != null) return fail(u, a, "unrecognized arguments: {s}", .{token});
            model = token;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = find(name) orelse return fail(u, a, "unrecognized arguments: {s}", .{token});
        var value: ?[]const u8 = null;
        if (flag.kind != .store_true) {
            if (eq) |e| value = token[e + 1 ..] else {
                i += 1;
                if (i >= argv.len) return fail(u, a, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
            if (flag.choices.len > 0) for (flag.choices) |c| {
                if (std.mem.eql(u8, c, value.?)) break;
            } else return fail(u, a, "argument {s}: invalid choice: '{s}'", .{ name, value.? });
        } else if (eq != null) return fail(u, a, "argument {s}: ignored explicit argument '{s}'", .{ name, token[eq.? + 1 ..] });
        if (!flag.native) return fail(u, a, "{s} is not served by the native engine yet; the Python engine 0.6.6 serves it: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{name});
        if (flag.native_values) |allowed| if (value) |v| for (allowed) |x| {
            if (std.mem.eql(u8, x, v)) break;
        } else return fail(u, a, "{s} {s} is not served by the native engine; the Python engine 0.6.6 serves it: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{ name, v });
        try apply(a, &out, name, value, u, &repeated);
    }
    out.model = model orelse return fail(u, a, "the following arguments are required: model", .{});
    try ranks(u, a, out);
    out.silence.validate() catch |err|
        return fail(u, a, "invalid silence guard configuration: {s}", .{@errorName(err)});
    out.redraw.validate(out.silence) catch |err| return fail(u, a, "invalid redraw configuration: {s}", .{@errorName(err)});
    if (out.vision_urls and !out.vision) return fail(u, a, "--vision-urls needs --vision", .{});
    if (out.idle_unload < 0 or out.idle_unload > std.math.maxInt(i64) / std.time.ns_per_s) return fail(u, a, "argument --idle-unload: expected a seconds count from 0 (off) under {d}: '{d}'", .{ std.math.maxInt(i64) / std.time.ns_per_s, out.idle_unload });
    out.alias = repeated.alias.items;
    out.api_key = repeated.keys.items;
    if (repeated.hosts.items.len > 64 or repeated.origins.items.len > 64) return fail(u, a, "at most 64 --allowed-host and 64 --allowed-origin entries are allowed", .{});
    out.allowed_host = repeated.hosts.items;
    out.allowed_origin = repeated.origins.items;
    return out;
}

fn int(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!i64 {
    return std.fmt.parseInt(i64, std.mem.trim(u8, v, " "), 10) catch fail(u, a, "argument {s}: invalid int value: '{s}'", .{ name, v });
}

fn float(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    return @import("fields.zig").pyFloat(v) orelse fail(u, a, "argument {s}: invalid float value: '{s}'", .{ name, v });
}

/// A finite GiB count from 0 (off) to below 2^34, whose bytes fit a u64.
fn gib(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    const g = try float(u, a, name, v);
    if (!std.math.isFinite(g) or g < 0 or g >= 1 << 34) return fail(u, a, "argument {s}: expected GiB from 0 (off) to below 2^34: '{s}'", .{ name, v });
    return g;
}

const Repeated = struct { alias: std.ArrayList([]const u8) = .empty, keys: std.ArrayList([]const u8) = .empty, hosts: std.ArrayList([]const u8) = .empty, origins: std.ArrayList([]const u8) = .empty };

fn apply(a: Allocator, out: *Args, name: []const u8, value: ?[]const u8, u: *Usage, repeated: *Repeated) error{ Usage, OutOfMemory }!void {
    const v = value orelse "";
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--silence-floor-dbfs")) {
        out.silence.floor_dbfs = try float(u, a, name, v);
        return;
    }
    if (is(name, "--silence-seconds")) {
        out.silence.seconds = try float(u, a, name, v);
        return;
    }
    if (is(name, "--silence-mid-s") or is(name, "--silence-mid-min-voiced-s")) {
        const seconds = try float(u, a, name, v);
        if (is(name, "--silence-mid-s")) out.silence.mid_seconds = seconds else out.silence.mid_min_voiced_s = seconds;
        return;
    }
    if (is(name, "--allowed-controls")) {
        out.allowed_controls = @import("drama_allow.zig").parse(v) catch
            return fail(u, a, "argument --allowed-controls: expected comma-separated UTF-8 names without square brackets: '{s}'", .{v});
        return;
    }
    if (is(name, "--allowed-host")) {
        if (!@import("request_guard.zig").validHost(v)) return fail(u, a, "argument --allowed-host: expected a hostname or bracketed IP without a port: '{s}'", .{v});
        return repeated.hosts.append(a, v);
    }
    if (is(name, "--allowed-origin")) {
        if (!@import("request_guard.zig").validOrigin(v)) return fail(u, a, "argument --allowed-origin: expected http(s)://HOST[:PORT] with no path or userinfo: '{s}'", .{v});
        return repeated.origins.append(a, v);
    }
    if (try cudaFlag(a, out, name, v, u) or try gpuFlag(a, out, name, v, u)) return;
    if (try recallFlag(a, out, name, v, u)) return;
    if (is(name, "--voices")) {
        out.voices = v;
        return;
    }
    if (is(name, "--default-voice")) {
        if (!std.mem.eql(u8, v, "default") and !@import("drama_voice.zig").validName(v))
            return fail(u, a, "invalid --default-voice: {s}", .{v});
        out.default_voice = v;
        return;
    }
    if (@import("notes_flags.zig").parse(out, name, v)) return;
    if (is(name, "--chat-template")) {
        out.chat_template = v;
        return;
    }
    if (is(name, "--max-frames")) {
        const n = try int(u, a, name, v);
        if (n < 1 or n > 8190) return fail(u, a, "argument --max-frames: expected 1 through 8190: '{s}'", .{v});
        out.max_frames = @intCast(n);
        return;
    }
    if (is(name, "--silence-onset-s")) out.silence.onset_seconds = try float(u, a, name, v) else if (is(name, "--silence-onset-voiced-s")) out.silence.onset_voiced_seconds = try float(u, a, name, v) else if (is(name, "--host")) out.host = v else if (is(name, "--port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --port: invalid port: '{s}'", .{v});
        out.port = @intCast(p);
    } else if (is(name, "--name")) out.name = v else if (is(name, "--alias")) try repeated.alias.append(a, v) else if (is(name, "--api-key")) try repeated.keys.append(a, v) else if (is(name, "--api-key-file")) out.api_key_file = v else if (is(name, "--metrics-open")) out.metrics_open = true else if (is(name, "--dashboard")) out.dashboard = true else if (is(name, "--context")) out.context = try int(u, a, name, v) else if (is(name, "--speed-up")) out.speed_up = v else if (is(name, "--prompt-cache-gib")) out.prompt_cache_gib = try gib(u, a, name, v) else if (is(name, "--prompt-cache-over-cap")) out.prompt_cache_over_cap = true else if (is(name, "--learn")) out.learn = true else if (is(name, "--learn-dir")) {
        out.learn = true;
        out.learn_dir = v;
    } else if (is(name, "--learn-min-free-gib")) {
        out.learn_min_free_gib = try gib(u, a, name, v);
    } else if (is(name, "--learn-gib")) {
        out.learn = true;
        out.learn_gib = try gib(u, a, name, v);
    } else if (is(name, "--learn-conversations")) {
        out.learn = true;
        out.learn_conversations = true;
    } else if (is(name, "--learn-daily-gib")) {
        out.learn_daily_gib = try gib(u, a, name, v);
    } else if (is(name, "--prompt-seed")) {
        out.prompt_seed = true;
    } else if (is(name, "--max-tokens")) out.max_tokens = try int(u, a, name, v) else if (is(name, "--temperature")) out.temperature = try float(u, a, name, v) else if (is(name, "--top-p")) out.top_p = try float(u, a, name, v) else if (is(name, "--top-k")) out.top_k = try int(u, a, name, v) else if (is(name, "--min-p")) out.min_p = try float(u, a, name, v) else if (is(name, "--thinking")) out.thinking = true else if (is(name, "--no-thinking")) out.thinking = false else if (is(name, "--reasoning-effort")) out.reasoning_effort = v else if (is(name, "--thinking-budget")) out.thinking_budget = try int(u, a, name, v) else if (is(name, "--thinking-reserve")) out.thinking_reserve = try int(u, a, name, v) else if (is(name, "--loop-guard")) out.loop_guard = true else if (is(name, "--drafter")) out.drafter = v else if (is(name, "--drafter-bits")) out.drafter_bits = @intCast(try int(u, a, name, v)) else if (is(name, "--record")) out.record = v else if (is(name, "--kv-cache")) out.kv_cache = v else if (is(name, "--prompt-precision")) out.prompt_precision = v else if (is(name, "--no-drafts")) out.no_drafts = true else if (is(name, "--keep-warm")) out.keep_warm = try int(u, a, name, v) else if (is(name, "--idle-unload")) out.idle_unload = try int(u, a, name, v) else if (is(name, "--compact-at")) {
        if (std.mem.eql(u8, v, "auto")) out.compact_auto = true else {
            const f = try float(u, a, name, v);
            if (!(f > 0 and f <= 1)) return fail(u, a, "argument --compact-at: expected auto or a fraction in (0, 1]: '{s}'", .{v});
            out.compact_fraction = f;
        }
    } else if (is(name, "--compact-keep")) {
        const n = try int(u, a, name, v);
        if (n < 0) return fail(u, a, "argument --compact-keep: expected a token count from 0: '{s}'", .{v});
        out.compact_keep = @intCast(n);
    } else if (is(name, "--compact-memory")) out.compact_memory = v else if (is(name, "--slide")) out.slide = true else if (is(name, "--slide-graph")) out.slide_graph = v else if (is(name, "--parallel")) out.parallel = v else if (is(name, "--backend")) out.backend = v else if (is(name, "--cortheon")) out.cortheon = true else if (is(name, "--vision")) out.vision = true else if (is(name, "--vision-urls")) out.vision_urls = true else if (is(name, "--vision-max-images")) {
        const n = try int(u, a, name, v);
        if (n < 1 or n > 64) return fail(u, a, "argument --vision-max-images: expected 1 to 64 images: '{s}'", .{v});
        out.vision_max_images = @intCast(n);
    } else if (is(name, "--vision-max-videos")) {
        const n = try int(u, a, name, v);
        if (n < 1 or n > 16) return fail(u, a, "argument --vision-max-videos: expected 1 to 16 videos: '{s}'", .{v});
        out.vision_max_videos = @intCast(n);
    } else if (is(name, "--vision-video-tokens")) {
        const n = try int(u, a, name, v);
        if (n < 64 or n > 262144) return fail(u, a, "argument --vision-video-tokens: expected 64 to 262144 tokens: '{s}'", .{v});
        out.vision_video_tokens = @intCast(n);
    } else if (is(name, "--vision-image-tokens")) {
        const n = try int(u, a, name, v);
        if (n < 16 or n > 262144) return fail(u, a, "argument --vision-image-tokens: expected 16 to 262144 tokens: '{s}'", .{v});
        out.vision_image_tokens = @intCast(n);
        out.vision_image_tokens_set = true;
    } else if (is(name, "--silence-redraws")) {
        out.redraw.max = std.math.cast(u8, try int(u, a, name, v)) orelse return fail(u, a, "argument --silence-redraws: expected 0 to 255 redraws: '{s}'", .{v});
    } else if (is(name, "--silence-redraw-budget-s")) {
        out.redraw.budget_s = try float(u, a, name, v);
    } else if (is(name, "--short-reply-ratio")) out.redraw.short_ratio = try float(u, a, name, v) else if (is(name, "--short-reply-chars-per-s")) out.redraw.short_chars_per_s = try float(u, a, name, v) else if (is(name, "--silence-redraw-controls")) {
        out.redraw.controls = @import("drama_allow.zig").parse(v) catch return fail(u, a, "argument --silence-redraw-controls: expected comma-separated UTF-8 names without square brackets, or an empty list: '{s}'", .{v});
    } else if (is(name, "--register-redraw-st")) out.redraw.register.st = try float(u, a, name, v) else if (is(name, "--register-redraw-high")) out.redraw.register.high = try float(u, a, name, v) else if (is(name, "--register-high-hz")) out.redraw.register.high_hz = try float(u, a, name, v);
}

/// --recall and the flags that imply it; false for any other flag.
fn recallFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--recall-dir")) {
        out.recall_dir = v;
    } else if (std.mem.eql(u8, name, "--recall-gib")) {
        out.recall_gib = try gib(u, a, name, v);
        const least = "argument --recall-gib: at least 1/128 GiB (two 4 MiB chunks): '{s}'";
        if (out.recall_gib < 1.0 / 128.0) return fail(u, a, least, .{v});
    } else if (std.mem.eql(u8, name, "--recall-read-only")) {
        out.recall_read_only = true;
    } else if (!std.mem.eql(u8, name, "--recall")) return false;
    out.recall = true;
    return true;
}

/// The CUDA build's --device, --segments and --precision; false for any other flag.
fn cudaFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--precision")) {
        out.precision = std.meta.stringToEnum(api.Precision, v) orelse return fail(u, a, "argument --precision: checkpoint or full: '{s}'", .{v});
    } else if (std.mem.eql(u8, name, "--device")) {
        out.device = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --device: a GPU ordinal from 0: '{s}'", .{v});
    } else if (std.mem.eql(u8, name, "--segments")) {
        const n = try int(u, a, name, v);
        out.segments = if (n >= 1) @intCast(@min(n, std.math.maxInt(u32))) else return fail(u, a, "argument --segments: a count from 1: '{s}'", .{v});
    } else return false;
    return true;
}

/// A rank names one of the `tp` processes.
fn ranks(u: *Usage, a: Allocator, out: Args) error{ Usage, OutOfMemory }!void {
    if (out.rank >= out.tp) return fail(u, a, "argument --rank: {d} is not below --tp {d}", .{ out.rank, out.tp });
}

/// The GPU engine's tensor-parallel, MTP, kept-entry and policy flags; false for any other flag.
fn gpuFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--tp")) out.tp = @intCast(try int(u, a, name, v)) else if (is(name, "--rank")) {
        out.rank = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --rank: a rank from 0: '{s}'", .{v});
    } else if (is(name, "--master")) out.master = v else if (is(name, "--master-port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --master-port: invalid port: '{s}'", .{v});
        out.master_port = @intCast(p);
    } else if (is(name, "--mtp-drafts")) {
        const n = try int(u, a, name, v);
        if (n < 0 or n > 64) return fail(u, a, "argument --mtp-drafts: a count from 0 to 64: '{s}'", .{v});
        out.mtp_drafts = @intCast(n);
    } else if (is(name, "--mtp-confidence")) {
        const c = try float(u, a, name, v);
        if (!(c >= 0 and c <= 1)) return fail(u, a, "argument --mtp-confidence: a probability in [0, 1]: '{s}'", .{v});
        out.mtp_confidence = c;
    } else if (is(name, "--checkpoint-slots")) {
        const n = try int(u, a, name, v);
        if (n < 0) return fail(u, a, "argument --checkpoint-slots: a count from 0: '{s}'", .{v});
        out.checkpoint_slots = n;
    } else if (is(name, "--policy")) {
        if (std.mem.indexOfScalar(u8, v, '=') == null) return fail(u, a, "argument --policy: expected key=value: '{s}'", .{v});
        out.policy = if (out.policy.len == 0) v else try std.fmt.allocPrint(a, "{s},{s}", .{ out.policy, v });
    } else return false;
    return true;
}

/// ``--parallel``: "auto" is up to 8 requests at once; a number caps it.
pub fn parallel(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    if (std.ascii.eqlIgnoreCase(t, "auto")) return 8;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    return @intCast(@max(1, @min(n, 4096)));
}

/// ``--parallel`` named a number (an engine that fits fewer refuses), not "auto" (it serves what fits).
pub fn parallelFixed(text: []const u8) bool {
    return !std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " "), "auto") and parallel(text) != null;
}

/// What a server asks of its engine, from its flags.
pub fn request(a: Allocator, dir: []const u8, model_type: []const u8, args: Args) !api.Open {
    return .{
        .dir = dir,
        .model_type = model_type,
        .context = args.context,
        .max_frames = args.max_frames,
        .silence = args.silence,
        .lanes = parallel(args.parallel) orelse 8,
        .lanes_fixed = parallelFixed(args.parallel),
        .drafts = !args.no_drafts,
        .drafter = args.drafter,
        .drafter_bits = args.drafter_bits,
        .record = args.record,
        .kv8 = std.mem.eql(u8, args.kv_cache, "fp8"),
        .prompt_int8 = std.mem.eql(u8, args.prompt_precision, "int8"),
        .speed_up = args.speed_up,
        .prompt_cache_gib = args.prompt_cache_gib,
        .prompt_cache_over_cap = args.prompt_cache_over_cap,
        .learn = if (args.learn) args.learn_dir orelse try api.prompt_imprint.defaultRoot(a) else null,
        .learn_gib = args.learn_gib,
        .learn_min_free_gib = args.learn_min_free_gib,
        .learn_conversations = args.learn and args.learn_conversations,
        .learn_daily_gib = args.learn_daily_gib,
        .slide = args.slide,
        .recall = args.recall,
        .device = args.device,
        .segments = args.segments,
        .precision = args.precision,
        .mtp_drafts = args.mtp_drafts,
        .mtp_confidence = args.mtp_confidence,
        .keep = args.checkpoint_slots,
        .tp = args.tp,
        .rank = args.rank,
        .master = args.master,
        .master_port = args.master_port,
        .policy = args.policy,
        .vision = if (args.vision) .{ .image_tokens = args.vision_image_tokens, .image_tokens_set = args.vision_image_tokens_set, .max_images = args.vision_max_images, .max_videos = args.vision_max_videos, .video_tokens = args.vision_video_tokens } else null,
    };
}

/// What ``capabilities --json`` reports about the engine side: its version, chip, backends and families.
pub const Engines = struct {
    version: []const u8,
    chip: ?[]const u8 = null,
    optional_libraries: []const @import("sys").optional_libraries.Status = &.{},
    backends: []const []const u8 = &.{},
    /// model_type to the weight formats it reads, as gate entries name them.
    families: []const api.Family = &.{},
};

/// The native capabilities document: the table's native flags and the honoured variables.
pub fn capabilities(w: *std.Io.Writer, e: Engines) !void {
    try w.print("{{\"schema\": 1, \"engine\": \"zig\", \"version\": \"{s}\", \"chip\": ", .{e.version});
    if (e.chip) |c| try w.print("\"{s}\"", .{c}) else try w.writeAll("null");
    try w.writeAll(", \"backends\": [");
    for (e.backends, 0..) |b, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", b });
    try w.writeAll("], \"families\": {");
    for (e.families, 0..) |f, i| {
        try w.print("{s}\"{s}\": [", .{ if (i > 0) ", " else "", f.model_type });
        for (f.formats, 0..) |fmt, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", fmt });
        try w.writeAll("]");
    }
    try w.writeAll("}, \"serve\": {\"flags\": {");
    var first = true;
    for (flags) |f| {
        if (!f.native) continue;
        try w.print("{s}\"{s}\": {{", .{ if (first) "" else ", ", f.name });
        first = false;
        const values = f.native_values orelse f.choices;
        if (values.len > 0) {
            try w.writeAll("\"values\": [");
            for (values, 0..) |v, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", v });
            try w.writeAll("]");
        }
        try w.writeAll("}");
    }
    try w.writeAll("}}, \"env\": [");
    for (env, 0..) |name, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", name });
    try w.writeAll("], \"optional_libraries\": ");
    try std.json.Stringify.value(e.optional_libraries, .{}, w);
    try w.writeAll("}\n");
}

const cli = @This();
/// The private flag helpers cli_test.zig checks.
pub const Test = if (builtin.is_test) struct {
    pub const cuda_build = cli.cuda_build;
    pub const native_serves = cli.native_serves;
    pub const servedBy = cli.servedBy;
    pub const cudaFlag = cli.cudaFlag;
    pub const gpuFlag = cli.gpuFlag;
    pub const ranks = cli.ranks;
} else struct {};
