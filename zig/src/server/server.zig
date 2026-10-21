//! The server: one engine and tokenizer behind the HTTP routes, one thread a connection.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const auth = @import("auth.zig");
const request_guard = @import("request_guard.zig");
const metrics = @import("metrics.zig");
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const http_conn = @import("http_conn.zig");
const listener_mod = @import("listener.zig");
const routes = @import("routes.zig");
const responses = @import("responses.zig");
const grammar = @import("grammar.zig");
const clock = @import("clock.zig");
const tool_grammar = @import("tool_grammar.zig");
const chunk_plan = @import("chunk_plan.zig");
const log = @import("log.zig");
const live = @import("live.zig");
const slide_graph = @import("slide_graph.zig");
const slide_lesson = @import("slide_lesson.zig");
const cortheon = @import("cortheon/cortheon.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// auto compacts near the window. fraction compacts past that share of it.
pub const CompactAt = union(enum) { auto, fraction: f64 };

pub const Identity = @import("identity.zig").Identity;

/// The most a kept-alive connection's arena keeps between requests; a longer request's memory goes back.
pub const connection_arena_limit: usize = 1 << 20;

pub const Config = struct {
    browser: request_guard.Policy = .{},
    allowed_controls: ?@import("drama_allow.zig").Policy = null,
    redraw: @import("speech_redraw.zig").Policy = .{},
    sse_keepalive_s: f64 = 15,
    served_name: []const u8,
    model_dir: []const u8 = "",
    voices_dir: ?[]const u8 = null,
    default_voice: []const u8 = "default",
    /// The served name first, then aliases, without repeats.
    model_ids: []const []const u8,
    default_max_tokens: i64 = 4096,
    enable_thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    /// With thinking on, reply tokens the reasoning may use beyond the request's max_tokens (0: none).
    thinking_reserve: i64 = 0,
    loop_guard: bool = false,
    /// Seconds the idle keepalive runs after the last request ends (0: off).
    keep_warm_s: i64 = 900,
    dashboard: bool = false,
    /// Sampling when a request names none (generation_config.json and the serve flags), or null for greedy.
    default_sampling: ?Value = null,
    use_drafts: bool = true,
    /// --vision-urls: image parts may name public HTTPS URLs, fetched before admission (media_fetch.zig).
    vision_urls: bool = false,
    seed_salt: i64 = 0,
    silence_mid: bool = false,
    /// --prompt-seed keeps historical prompt-keyed sampling when a request omits its seed.
    prompt_seed: bool = false,
    /// TENSORFOLD_REQUEST_LOG: where chat and completion bodies are appended; null: nowhere.
    request_log: ?[]const u8 = null,
    /// Null keeps every reply identical to a server that has no compaction.
    compact_at: ?CompactAt = null,
    /// Tokens of the recent tail kept whole. Null uses min(20000, a quarter of the window).
    compact_keep: ?u32 = null,
    /// Directory of the stored note, or null when notes are not stored.
    compact_memory: ?[]const u8 = null,
    /// --slide-graph: the file holding the Sliding Weights fact graph; null keeps the graph in memory.
    slide_graph: ?[]const u8 = null,
    /// A fact whose check a delete fails, to test its put-back (0: none); only -Dslide-test-hooks builds set it.
    slide_forget_fail: u32 = 0,
    /// --cortheon: Cortheon's program and server-run web research for chat requests; off changes no reply.
    cortheon: bool = false,
    /// BRAVE_SEARCH_API_KEY: Cortheon searches through Brave instead of DuckDuckGo.
    cortheon_search_key: ?[]const u8 = null,
    /// --learn-conversations: /v1/imprint answers; without it each of its routes refuses and names the flag.
    saved_conversations: bool = false,
    timeouts: http_conn.Timeouts = .{},
    max_connections: u32 = @import("http_admission.zig").default_limit,
    /// The open /health names these after its other keys; empty fields are left out.
    identity: Identity = .{},
    /// The process environment GET /v1/benchmark/manifest reads (rank count, revision, the hardware probe); null: none.
    environ: ?*const std.process.Environ.Map = null,
};

pub const Server = struct {
    gpa: Allocator,
    io: std.Io,
    engine: api.Engine,
    info: api.Info,
    text: model_text.Text,
    config: Config,
    voices: []const @import("drama_voice.zig").Voice = &.{},
    keys: ?*auth.Store,
    late_system: []const u8 = "system",
    needs_user_after_tool: bool = false,
    markers: reply_text.Markers = reply_text.think_markers,
    close_id: ?u32 = null, // the close marker's own token, when the marker is one token
    chunks: chunk_plan.Plan = .{},
    effort_levels: []const []const u8 = &.{},
    thinking_switch: bool = false,
    eos: []const u32 = &.{},
    think_close: []const u32 = &.{},
    think_close_end: ?u32 = null,
    metrics: metrics.Metrics,
    store: responses.Store,
    next_id: std.atomic.Value(u64) = .init(1),
    /// Connections open now; a stop waits for them before freeing what they read.
    open_connections: std.atomic.Value(u32) = .init(0),
    draining: std.atomic.Value(bool) = .init(false),
    generating: std.atomic.Value(u32) = .init(0),
    /// The idle keepalive on the engine's queue, armed by --keep-warm; null when off or not Metal.
    keepalive: ?*api.keepalive.Keepalive = null,
    preparing: std.atomic.Value(i64) = .init(0),
    speech_inflight: std.atomic.Value(u32) = .init(0),
    arena: std.heap.ArenaAllocator,
    /// Structured output's compiler, opened by the first structured request (Python's grammar.compiler).
    grammars: ?*api.grammar.Compiler = null,
    grammar_problem: ?[]const u8 = null,
    grammar_mutex: std.Io.Mutex = .init,
    /// The facts the learner was taught and whether the weights recall them.
    slide: slide_graph.Graph,
    teacher: slide_lesson.Teacher, // what every Sliding Weights lesson shares: keep prompts, the turn's end
    /// The folder's snapshot made or hashed whole by this process (under teacher.learning): later lessons trust it.
    snapshot_checked: bool = false,
    /// Cortheon's web reader and what replies hid from their clients; null without --cortheon.
    cortheon: ?*cortheon.State = null,
    /// Recall's record (--recall), which GET and DELETE /v1/recall read and empty; main opens and closes it.
    recall: ?*api.recall_store.Store = null,

    /// Reads what the template and tokenizer decide once: late system role, think markers, efforts, the forced close.
    pub fn init(gpa: Allocator, io: std.Io, engine: api.Engine, text: model_text.Text, config: Config, keys: ?*auth.Store) !*Server {
        const srv = try gpa.create(Server);
        srv.* = .{ .gpa = gpa, .io = io, .engine = engine, .info = engine.info(), .text = text, .config = config, .keys = keys, .metrics = .{ .gpa = gpa }, .store = .{ .gpa = gpa }, .arena = .init(gpa), .slide = .init(gpa, io, config.slide_graph), .teacher = .init(gpa) };
        errdefer srv.deinit(); // a later step's failure stops the keepalive before the caller closes the engine
        if (srv.info.embeddings != null and srv.info.text_generation) return error.InvalidEmbeddingCapabilities;
        if (srv.info.vision) |vision| if (@import("vision_markers.zig").firstUnsafe(text, vision)) |id| {
            log.line("--vision: marker token {d} is not an added token in this tokenizer; image and video input cannot be served safely", .{id});
            return error.InvalidVisionMarkers;
        };
        if (config.voices_dir) |dir| {
            if (srv.info.speech == null) return error.VoicesRequireSpeech;
            srv.voices = try @import("drama_voice.zig").load(srv.arena.allocator(), io, dir);
        }
        _ = try @import("drama_voice.zig").select(srv.voices, config.default_voice, "default");
        if (!srv.info.text_generation) {
            if (config.cortheon) return error.CortheonRequiresTextGeneration;
            return srv;
        }
        if (config.cortheon) srv.cortheon = try cortheon.State.init(gpa, io, "TensorFold-Cortheon (+https://github.com/ashhart/TensorFold)", config.cortheon_search_key);
        if (config.keep_warm_s > 0) if (engine.keepaliveTarget()) |target| {
            srv.keepalive = api.keepalive.Keepalive.start(gpa, io, target, @as(i64, config.keep_warm_s) * std.time.ns_per_s) catch |e| blk: {
                log.line("idle keepalive off: {s}", .{@errorName(e)});
                break :blk null;
            };
        };
        const a = srv.arena.allocator();
        srv.eos = try a.dupe(u32, text.eosIds());
        if (text.tokenId(reply_text.channel_markers.close) != null) srv.markers = reply_text.channel_markers;
        srv.close_id = text.tokenId(srv.markers.close);
        const source = text.templateSource();
        srv.effort_levels = try fields.effortLevels(a, source);
        srv.thinking_switch = std.mem.indexOf(u8, source, "enable_thinking") != null or
            std.mem.indexOf(u8, source, "thinking_mode") != null;
        srv.late_system = try lateSystem(a, text);
        srv.needs_user_after_tool = needsUserAfterTool(source);
        if (srv.info.vision != null) {
            const template_images = templateImages(srv);
            if (!template_images) log.line("warning: --vision: the chat template writes no image placeholder for an image part, so image requests will fail; serve with --chat-template naming a template that has an image path", .{});
        }
        if (text.tokenId("</think>")) |end| {
            const lead = try text.encode(a, "\n", false);
            const trail = try text.encode(a, "\n\n", false);
            srv.think_close = try std.mem.concat(a, u32, &.{ lead, &.{end}, trail });
            srv.think_close_end = end;
        }
        if (srv.info.prefill_step > 0) {
            srv.chunks = try chunk_plan.markers(srv, a, srv.info.prefill_step);
            var step: [32]u8 = undefined;
            log.line("prompt chunks of up to {s} tokens, cut at replies {d}+ tokens apart", .{ live.commas(&step, srv.info.prefill_step), srv.chunks.min_chunk });
        }
        return srv;
    }

    pub fn stopKeepalive(srv: *Server) void {
        if (srv.keepalive) |k| k.stop();
        srv.keepalive = null;
    }

    pub fn deinit(srv: *Server) void {
        srv.stopKeepalive();
        if (srv.grammars) |g| g.close();
        srv.metrics.deinit();
        srv.arena.deinit();
        srv.store.deinit();
        srv.slide.deinit();
        srv.teacher.deinit();
        if (srv.cortheon) |c| c.deinit();
        srv.gpa.destroy(srv);
    }

    /// The effort the template hears: the request's name, or this server's default.
    pub fn effortFor(srv: *const Server, explicit: ?[]const u8) ?[]const u8 {
        const e = explicit orelse srv.config.reasoning_effort orelse return null;
        return fields.coerceEffort(e, srv.effort_levels);
    }

    /// Request seeds override configured keys; otherwise use prompt keying or fresh entropy.
    pub fn resolveSampling(srv: *Server, cx: *Cx, f: Value, temperature: f64, prompt_ids: []const u32) errors.Refused!?api.Sampling {
        _ = temperature; // the request's own temperature field decides, as in the Python server
        const options = try json.newObject(cx.a);
        const defaults = if (srv.config.default_sampling) |d| try fields.parseNumbers(cx, d) else null;
        if (defaults) |d| try merge(options, cx, d);
        const supplied = try fields.parseNumbers(cx, f);
        try merge(options, cx, supplied);
        const temp = number(options.get("temperature")) orelse 0;
        if (temp <= 0) return null;
        if (srv.info.greedy_only) return cx.refuse(if ((if (defaults) |d| number(d.get("temperature")) else null) orelse 0 > 0)
            "this engine decodes greedily only and this server samples by default: send temperature 0, or serve with --temperature 0"
        else
            "this engine decodes greedily only: send temperature 0 or leave it out");
        const seed: u64 = if (options.get("seed")) |s| seedBits(s) else if (srv.config.prompt_seed)
            api.seedFor(prompt_ids, srv.config.seed_salt)
        else fresh: {
            var bits: u64 = undefined;
            srv.io.randomSecure(std.mem.asBytes(&bits)) catch
                return cx.fail(.server, "could not obtain a random request seed", .{});
            break :fresh bits & 0x7fff_ffff_ffff_ffff;
        };
        return .{
            .seed = seed,
            .temperature = @floatCast(temp),
            .top_k = @intCast(@min(@max(0, (if (options.get("top_k")) |k| k.int64() else null) orelse 0), std.math.maxInt(u32))),
            .top_p = @floatCast(number(options.get("top_p")) orelse 1.0),
            .min_p = @floatCast(number(options.get("min_p")) orelse 0.0),
        };
    }

    /// A specific engine name, or the server's backend identity when the shared host uses its default name.
    pub fn backendName(srv: *const Server) []const u8 {
        return if (std.mem.eql(u8, srv.info.name, "lanes") and srv.config.identity.backend.len > 0) srv.config.identity.backend else srv.info.name;
    }

    /// The grammar a request asks for, or its refusal when the engine cannot enforce it.
    pub fn checkFeatures(srv: *Server, cx: *Cx, f: Value, has_tools: bool, thinking: bool, prompt_ids: []const u32, tools: []const Value, request: *api.Request) errors.Refused!void {
        _ = prompt_ids;
        const required = if (f.get("tool_call_required")) |r| r.truthy() else false;
        if (has_tools and required) {
            if (!srv.info.structures) return cx.fail(.request, "backend '{s}' cannot enforce tool_choice \"required\" or a named function: it has no token grammar support; send \"auto\" or \"none\"", .{srv.backendName()});
            const one_call = if (f.get("tool_call_single")) |x| x.truthy() else false;
            request.structure = try tool_grammar.structure(srv, cx, tools, !one_call, thinking);
        }
        if (try grammar.spec(cx, f)) |s| {
            if (!srv.info.structures) return cx.refuse("structured output (response_format and the guided_* fields) is not supported by this engine yet");
            request.structure = try srv.compiledStructure(cx, s.kind, s.text, s.field, thinking);
        }
    }

    /// A request's grammar, compiled once here (a 400 with xgrammar's reason when it cannot be enforced).
    pub fn compiledStructure(srv: *Server, cx: *Cx, kind: grammar.Kind, text: []const u8, field: []const u8, thinking: bool) errors.Refused!api.Structure {
        const c = srv.grammarCompiler() orelse return cx.fail(.request, "{s}: {s}", .{ if (kind == .structural_tag) field else "structured output", srv.grammar_problem.? });
        var err: [512]u8 = undefined;
        const compiled = c.compile(srv.io, @fromBackingInt(@intCast(@backingInt(kind))), text, &err) catch
            return cx.fail(.request, "{s}: the grammar cannot be enforced: {s}", .{ field, std.mem.sliceTo(&err, 0) });
        compiled.free(); // kept in the compiler's cache for the engine's compile
        return .{ .kind = kind, .text = text, .after = if (thinking) srv.text.tokenId(srv.markers.close) else null, .compiler = c };
    }

    /// The model's grammar compiler: tokenizer.json, the text model's vocab_size, the eos ids.
    fn grammarCompiler(srv: *Server) ?*api.grammar.Compiler {
        srv.grammar_mutex.lockUncancelable(srv.io);
        defer srv.grammar_mutex.unlock(srv.io);
        if (srv.grammars) |g| return g;
        if (srv.grammar_problem != null) return null;
        const a = srv.arena.allocator();
        srv.grammar_problem = blk: {
            const dir = std.Io.Dir.cwd();
            const tok = dir.readFileAlloc(srv.io, std.fs.path.join(a, &.{ srv.config.model_dir, "tokenizer.json" }) catch break :blk "out of memory", a, .limited(256 << 20)) catch break :blk "the checkpoint has no tokenizer.json";
            const cfg_text = dir.readFileAlloc(srv.io, std.fs.path.join(a, &.{ srv.config.model_dir, "config.json" }) catch break :blk "out of memory", a, .limited(16 << 20)) catch break :blk "the checkpoint has no config.json";
            const cfg = switch (json.parse(a, cfg_text) catch break :blk "out of memory") {
                .ok => |v| v,
                .err => break :blk "config.json is not JSON",
            };
            const vocab = grammar.vocabSize(cfg) orelse break :blk "structured output needs the checkpoint's config.json vocab_size";
            var err: [512]u8 = undefined;
            const c = a.create(api.grammar.Compiler) catch break :blk "out of memory";
            c.* = api.grammar.Compiler.open(srv.io, a, tok, vocab, srv.eos, &err) catch |e| break :blk switch (e) {
                error.NoGrammarLibrary => @import("sys").optional_libraries.spec(.grammar).message,
                else => a.dupe(u8, std.mem.sliceTo(&err, 0)) catch "out of memory",
            };
            srv.grammars = c;
            log.line("structured output: xgrammar over {d} tokens", .{vocab});
            break :blk null;
        };
        return srv.grammars;
    }

    /// Counts one finished request for /metrics.
    pub fn noteRequest(srv: *Server, prompt_len: usize, cached: usize, generated: usize, drafted: u64, accepted: u64, rounds: u64, received: i96, first: ?i96, last: ?i96, prefill: ?f64) void {
        const now = clock.nowNs(srv.io);
        const tpot = srv.metrics.tpotValue(first, last, generated);
        srv.metrics.note(srv.io, prompt_len, cached, generated, drafted, accepted, rounds, @max(0, clock.seconds(now - received)), if (first) |f| clock.seconds(f - received) else null, if (first) |f| @max(0, clock.seconds(now - f)) else null, prefill, tpot);
    }

    /// Accepts connections until ``stop`` is set, each on its own thread.
    pub fn serve(srv: *Server, listener: listener_mod.Listener, stop: *const std.atomic.Value(bool)) void {
        const loopback_bind = if (listener.boundAddress()) |a| auth.loopbackAddress(a) else true;
        while (listener.accept(stop)) |c| {
            // counted before its thread runs, so a zero count once this loop ends means no connection is left
            if (!srv.admitConnection(c.fd)) continue;
            const t = std.Thread.spawn(.{ .stack_size = 8 << 20 }, connection, .{ srv, c, loopback_bind }) catch {
                _ = srv.open_connections.fetchSub(1, .acq_rel);
                _ = platform.posix.system.close(c.fd);
                continue;
            };
            t.detach();
        }
    }

    pub fn admitConnection(srv: *Server, fd: platform.posix.socket_t) bool {
        const admission = @import("http_admission.zig");
        if (admission.acquire(&srv.open_connections, srv.config.max_connections)) return true;
        admission.reject(fd);
        return false;
    }

    fn connection(srv: *Server, accepted: listener_mod.Listener.Accepted, loopback_bind: bool) void {
        defer _ = srv.open_connections.fetchSub(1, .acq_rel);
        defer _ = platform.posix.system.close(accepted.fd);
        var peer_buf: [64]u8 = undefined;
        const peer = listener_mod.peerText(&peer_buf, accepted.peer);
        var conn = http_conn.Conn.init(srv.gpa, accepted.fd, peer) catch return;
        defer conn.deinit();
        conn.timeouts = srv.config.timeouts;
        var browser = srv.config.browser;
        browser.loopback_bind = loopback_bind;
        conn.auth_enabled = srv.keys != null and srv.keys.?.enabled();
        if (conn.auth_enabled) conn.hook = .{ .ctx = srv, .call = countReply };
        var arena: std.heap.ArenaAllocator = .init(srv.gpa);
        defer arena.deinit();
        var first = true;
        while (true) {
            _ = arena.reset(.{ .retain_with_limit = connection_arena_limit });
            const a = arena.allocator();
            const outcome = conn.readRequest(a, !first) catch break;
            first = false;
            switch (outcome) {
                .closed => break,
                .handled => if (conn.close) break else continue,
                .ready => {},
            }
            if (!request_guard.allow(&conn, browser, a)) break;
            if (!srv.authenticate(&conn, a)) break;
            if (conn.header("Expect")) |expect| if (std.ascii.eqlIgnoreCase(expect, "100-continue") and std.mem.order(u8, conn.version, "HTTP/1.1") != .lt) conn.sendContinue();
            srv.dispatch(&conn, a);
            if (conn.close or conn.broken) break;
        }
    }

    pub fn dispatch(srv: *Server, conn: *http_conn.Conn, a: Allocator) void {
        routes.dispatch(srv, conn, a);
        if (!conn.response_complete) conn.close = true;
    }

    fn countReply(ctx: *anyopaque, conn: *http_conn.Conn, code: u16) void {
        const srv: *Server = @ptrCast(@alignCast(ctx));
        srv.metrics.httpRequest(srv.io, conn.key_label orelse "unauthenticated", code);
    }

    /// The key gate before any route: 401 for a gated path without a valid key (``Authenticated._authenticate``).
    fn authenticate(srv: *Server, conn: *http_conn.Conn, a: Allocator) bool {
        const keys = srv.keys orelse return true;
        if (std.mem.eql(u8, conn.method, "OPTIONS") or !keys.enabled()) return true;
        conn.key_label = keys.match(a, single(conn, "Authorization"), single(conn, "x-api-key"));
        if (!auth.gatedRequest(conn.method, conn.path, keys.metrics_open) or conn.key_label != null) return true;
        const path = auth.routePath(conn.path);
        const messages = for ([_][]const u8{ "/v1/messages", "/v1/messages/count_tokens", "/messages", "/messages/count_tokens" }) |p| {
            if (std.mem.eql(u8, path, p)) break true;
        } else false;
        const body = if (messages)
            "{\"type\": \"error\", \"error\": {\"type\": \"authentication_error\", \"message\": \"A valid API key is required\"}}"
        else
            "{\"error\": {\"message\": \"A valid API key is required\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": \"invalid_api_key\"}}";
        conn.close = true;
        conn.startResponse(401, null) catch return false;
        conn.addHeader("WWW-Authenticate", "Bearer") catch return false;
        conn.addHeader("Content-Type", "application/json") catch return false;
        var len: [24]u8 = undefined;
        conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return false;
        conn.addHeader("Connection", "close") catch return false;
        conn.finish(if (std.mem.eql(u8, conn.method, "HEAD")) "" else body) catch {};
        return false;
    }
};

/// A header that appears exactly once; none or several read as absent (as the key match reads them).
fn single(conn: *const http_conn.Conn, name: []const u8) ?[]const u8 {
    if (conn.headerCount(name) != 1) return null;
    return conn.header(name);
}

fn merge(into: *json.Object, cx: *Cx, from: Value) errors.Refused!void {
    for (from.object.keys(), from.object.values()) |k, v| if (v != .null) try into.put(cx.a, k, v);
}

fn number(v: ?Value) ?f64 {
    const x = v orelse return null;
    return if (x == .bool) null else x.float64();
}

/// A request seed as the engine's u64: negative and huge ints wrap as two's complement bits.
fn seedBits(v: Value) u64 {
    if (v != .int) return 0;
    if (std.fmt.parseInt(i128, v.int, 10)) |n| return @truncate(@as(u128, @bitCast(n))) else |_| return 0;
}

/// ``template_late_system``: "system" when the template keeps a later system message in place, else "user".
fn lateSystem(a: Allocator, text: model_text.Text) ![]const u8 {
    const probe_text = "tensorfold-late-system-probe";
    const probe = (try json.parse(a, "[{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"u\"}, {\"role\": \"assistant\", \"content\": \"a\"}, {\"role\": \"system\", \"content\": \"tensorfold-late-system-probe\"}, {\"role\": \"user\", \"content\": \"v\"}]")).ok;
    var problem: []const u8 = "";
    const rendered = text.render(a, probe, .{ .add_generation_prompt = false }, &problem) catch return "user";
    return if (std.mem.indexOf(u8, rendered.text, probe_text) != null) "system" else "user";
}

/// With image input: whether a user turn holding an image part renders an image placeholder (else a startup warning).
fn templateImages(srv: *Server) bool {
    const v = srv.info.vision orelse return true;
    var arena: std.heap.ArenaAllocator = .init(srv.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const probe = (json.parse(a, "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,AA==\"}}, {\"type\": \"text\", \"text\": \"x\"}]}]") catch return true).ok;
    const ids = @import("prompt.zig").renderIds(srv, &cx, probe, &.{}, false, null, true) catch return true; // a render failure is the request's to report
    for (ids) |t| if (t == v.image_token) return true;
    return false;
}

/// Templates requiring a user query gain a user turn after trailing tool results replace their last user turn.
fn needsUserAfterTool(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "No user query found") != null;
}
