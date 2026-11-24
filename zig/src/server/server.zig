//! The server: one engine and tokenizer behind the HTTP routes, one thread a connection.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const auth = @import("auth.zig");
const scope = @import("auth_scope.zig");
const request_guard = @import("request_guard.zig");
const metrics = @import("metrics.zig");
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const recipient_reply = @import("recipient_reply.zig");
const http_conn = @import("http_conn.zig");
const admission = @import("http_admission.zig");
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
const idle_unload = @import("idle_unload.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
/// auto compacts near the window. fraction compacts past that share of it.
pub const CompactAt = @import("server_config.zig").CompactAt;
pub const Identity = @import("identity.zig").Identity;
/// The most a kept-alive connection's arena keeps between requests; a longer request's memory goes back.
pub const connection_arena_limit: usize = 1 << 20;

pub const Config = @import("server_config.zig").Config;

/// --name-priority: background by default when the id the request names is configured (openai.replyModel parity).
pub fn namedBackground(config: Config, f: Value) bool {
    var asked = config.served_name;
    if (f.get("model")) |m| if (m == .string) for (config.model_ids) |known| {
        if (std.mem.eql(u8, known, m.string)) asked = m.string;
    };
    for (config.background_ids) |id| if (std.mem.eql(u8, id, asked)) return true;
    return false;
}
/// Name the compiled structure with the client's own field when its route carried one (a borrowed static literal).
pub fn nameStructure(format: ?[]const u8, request: *api.Request) void {
    const name = format orelse return;
    if (request.structure) |*st| st.field = name;
}

pub const Server = struct {
    gpa: Allocator,
    io: std.Io,
    engine: api.Engine,
    info: api.Info,
    text: model_text.Text,
    config: Config,
    voices: @import("drama_voice_store.zig").Store = .{},
    zonos: ?*@import("zonos_speech.zig").State = null, // a ZONOS3 Mini release's voices; Drama 3's stay in `voices`
    keys: ?*auth.Store,
    late_system: []const u8 = "system",
    needs_user_after_tool: bool = false,
    markers: reply_text.Markers = reply_text.think_markers,
    /// Replies are messages to recipients (`to=self` reasoning, `to=user` answer, a tool's name a call).
    recipients: bool = false,
    close_id: ?u32 = null, // the close marker's own token, when the marker is one token
    chunks: chunk_plan.Plan = .{},
    effort_levels: []const []const u8 = &.{},
    /// The effort a template with no thinking switch hears with thinking off: its lowest level, not its default.
    off_effort: ?[]const u8 = null,
    thinking_switch: bool = false,
    video_decoder: bool = false, // libtfvideo loaded in `init`: with the video offer it decides video input (#3004)
    eos: []const u32 = &.{},
    think_close: []const u32 = &.{},
    think_close_end: ?u32 = null,
    /// Template's empty system turn prefix (`<|im_start|>system\n<|im_end|>\n`), stripped when Folded Memory fires.
    empty_system_prefix: []const u32 = &.{},
    metrics: metrics.Metrics,
    /// Scheduler load the engine publishes as it works, and this server's identity and activity (load_ledger.zig).
    load: api.load_ledger.Ledger,
    store: responses.Store,
    background_jobs: @import("responses_background.zig").Manager = .{}, // background: true responses still running
    ws_cache_bytes: std.atomic.Value(usize) = .init(0), // Responses socket cache bytes (ws_cache.zig)
    tool_ids: @import("ids.zig").Key, // every tool-call id's and cache salt's key: the config's, else this server's own
    next_id: std.atomic.Value(u64) = .init(1),
    /// Connections open now; a stop waits for them before freeing what they read.
    open_connections: std.atomic.Value(u32) = .init(0),
    /// Open connections by remote address, for the per-address share of the slots.
    peers: admission.Peers = .{},
    draining: std.atomic.Value(bool) = .init(false),
    generating: std.atomic.Value(u32) = .init(0),
    /// Model requests holding a reservation, when config.admission is on.
    admission: @import("request_admission.zig").Controller = .{},
    /// The idle keepalive on the engine's queue, armed by --keep-warm; null when off or not Metal.
    keepalive: ?*api.keepalive.Keepalive = null,
    /// The idle-unload manager (--idle-unload); null when off. Its thread parks the engine; a wake reloads it.
    idle_manager: ?*idle_unload.Manager = null,
    preparing: std.atomic.Value(i64) = .init(0),
    speech_inflight: std.atomic.Value(u32) = .init(0),
    arena: std.heap.ArenaAllocator,
    /// Structured output's compiler, opened by the first structured request (Python's grammar.compiler).
    grammars: ?*api.grammar.Compiler = null,
    grammar_problem: ?[]const u8 = null,
    tool_grammars: ?*api.grammar.Compiler = null, // Inkling 2's required calls: the eos ids and its end of message
    tool_grammar_problem: ?[]const u8 = null,
    grammar_mutex: std.Io.Mutex = .init,
    /// The facts the learner was taught and whether the weights recall them.
    slide: slide_graph.Graph,
    teacher: slide_lesson.Teacher, // what every Sliding Weights lesson shares: keep prompts, the turn's end
    /// Non-blocking in-flight chat learner for Folded Memory Weights v5 (started when --slide is on).
    inflight: @import("slide_inflight.zig").Worker = .{},
    /// The folder's snapshot made or hashed whole by this process (under teacher.learning): later lessons trust it.
    snapshot_checked: bool = false,
    /// Cortheon's web reader and what replies hid from their clients; null without --cortheon.
    cortheon: ?*cortheon.State = null,
    /// Recall's record (--recall), which GET and DELETE /v1/recall read and empty; main opens and closes it.
    recall: ?*api.recall_store.Store = null,
    witness: ?*@import("witness.zig").State = null, // serve owns GET /v1/execution/witness state; null: off

    /// Reads what the template and tokenizer decide once: late system role, think markers, efforts, the forced close.
    pub fn media(srv: *const Server) @import("messages.zig").Media {
        return @import("input_types.zig").media(srv);
    }

    pub fn init(gpa: Allocator, io: std.Io, engine: api.Engine, text: model_text.Text, config: Config, keys: ?*auth.Store) !*Server {
        const tool_ids = if (config.tool_id_key) |key| key.take() else try @import("ids.zig").Key.random(io);
        const srv = try gpa.create(Server);
        srv.* = .{ .tool_ids = tool_ids, .gpa = gpa, .io = io, .engine = engine, .info = engine.info(), .text = text, .config = config, .keys = keys, .metrics = .{ .gpa = gpa, .phases = config.phase_metrics }, .load = .init(io), .store = .{ .gpa = gpa }, .arena = .init(gpa), .slide = .init(gpa, io, config.slide_graph), .teacher = .init(gpa) };
        srv.config.tool_id_key = null; // its key is zeroed: `tool_ids` is the one copy
        errdefer srv.deinit(); // a later step's failure stops the keepalive before the caller closes the engine
        @import("stale_temps.zig").discard(io, config); // a crash's temporaries beside the note and the graph
        if (srv.info.embeddings != null and srv.info.capability(.text_generation)) return error.InvalidEmbeddingCapabilities;
        if (srv.info.readouts != null and (srv.info.capability(.text_generation) or srv.info.embeddings != null)) return error.InvalidReadoutCapabilities;
        if (srv.info.vision) |vision| if (@import("vision_markers.zig").firstUnsafe(text, vision)) |id| {
            log.line("--vision: marker token {d} is not an added token in this tokenizer; image and video input cannot be served safely", .{id});
            return error.InvalidVisionMarkers;
        };
        try @import("input_types.zig").start(srv); // the video decoder and the audio locator's token
        try @import("speech_voices.zig").start(srv, gpa, io, config);
        if (!srv.info.capability(.text_generation)) {
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
        const minimax = text.tokenId(reply_text.mm_markers.close) != null;
        if (minimax) srv.markers = reply_text.mm_markers else if (srv.info.reasoning_optional and std.mem.eql(u8, srv.markers.close, reply_text.think_markers.close)) srv.markers.optional = true;
        srv.close_id = text.tokenId(srv.markers.close);
        srv.recipients = recipient_reply.detect(text);
        const source = text.templateSource();
        srv.effort_levels = try fields.effortLevels(a, source);
        if (srv.recipients and srv.effort_levels.len > 0) srv.off_effort = fields.coerceEffort("minimal", srv.effort_levels);
        // Inkling 2's template has no enable_thinking: its off is reasoning_effort "none" (effort level 0)
        if (@import("chat_inkling2.zig").enabled(srv)) srv.off_effort = "none";
        // MiniMax-M3's template decides on its own; thinking off closes the block in the prompt (prompt.zig)
        srv.thinking_switch = minimax or std.mem.indexOf(u8, source, "enable_thinking") != null or
            std.mem.indexOf(u8, source, "thinking_mode") != null;
        srv.late_system = try lateSystem(a, text);
        srv.needs_user_after_tool = needsUserAfterTool(source);
        if (srv.info.vision != null) {
            const template_images = templateImages(srv);
            if (!template_images) log.line("warning: --vision: the chat template writes no image placeholder for an image part, so image requests will fail; serve with --chat-template naming a template that has an image path", .{});
        }
        if (text.tokenId(if (minimax) reply_text.mm_markers.close else "</think>")) |end| {
            const lead = try text.encode(a, "\n", false);
            const trail = try text.encode(a, "\n\n", false);
            srv.think_close = try std.mem.concat(a, u32, &.{ lead, &.{end}, trail });
            srv.think_close_end = end;
        }
        // the header a memory prompt strips, encoded only where the engine serves learning
        if (srv.info.folded_memory) srv.empty_system_prefix = text.encode(a, "<|im_start|>system\n<|im_end|>\n", false) catch &.{};
        srv.info.thinking_applicable = srv.close_id != null;
        srv.info.thinking_switch = try @import("prompt.zig").thinkingSwitch(srv);
        if (srv.info.prefill_step > 0) {
            srv.chunks = try chunk_plan.markers(srv, a, srv.info.prefill_step);
            srv.chunks.min_chunk = srv.info.prefill_min_chunk;
            var step: [32]u8 = undefined;
            log.line("prompt chunks of up to {s} tokens, cut at replies {d}+ tokens apart", .{ live.commas(&step, srv.info.prefill_step), srv.chunks.min_chunk });
        }
        if (srv.info.folded_memory) srv.inflight.start(gpa, io, srv);
        return srv;
    }

    pub fn stopKeepalive(srv: *Server) void {
        if (srv.keepalive) |k| k.stop();
        srv.keepalive = null;
    }

    pub fn deinit(srv: *Server) void {
        srv.inflight.shutdown(); // drain any pending in-flight chat learning before engine/slide tear down
        srv.background_jobs.shutdown(srv); // they stop and end before the engine, store and counts they use go
        srv.load.detach(false); // an engine still attached is still open: it stops writing before anything here frees
        srv.stopKeepalive();
        if (srv.grammars) |g| g.close();
        if (srv.tool_grammars) |g| if (g != srv.grammars) g.close();
        srv.metrics.deinit();
        srv.voices.deinit();
        srv.arena.deinit();
        srv.store.deinit();
        srv.slide.deinit();
        srv.teacher.deinit();
        if (srv.cortheon) |c| c.deinit();
        srv.gpa.destroy(srv);
    }

    /// The effort the template hears: the request's name, or this server's default.
    pub fn effortFor(srv: *const Server, explicit: ?[]const u8) ?[]const u8 {
        const e = explicit orelse srv.config.reasoning_effort orelse (if (srv.text.vtable.format == .kimi) "max" else return null);
        return fields.coerceEffort(e, srv.effort_levels);
    }

    /// The defaults a request starts from: --instruct-sampling's set for a thinking-off chat that samples under it.
    pub fn samplingDefaults(srv: *const Server, instruct: bool, f: Value) ?Value {
        const defaults = srv.config.default_sampling;
        if (instruct) if (srv.config.instruct_sampling) |set| {
            const t = number(f.get("temperature")) orelse number(set.get("temperature")) orelse
                (if (defaults) |d| number(d.get("temperature")) else null) orelse 0;
            if (t > 0) return set;
        };
        return defaults;
    }

    /// Request seeds override configured keys; otherwise use prompt keying or fresh entropy.
    pub fn resolveSampling(srv: *Server, cx: *Cx, f: Value, temperature: f64, prompt_ids: []const u32, instruct: bool) errors.Refused!?api.Sampling {
        _ = temperature; // the request's own temperature field decides, as in the Python server
        const options = try json.newObject(cx.a);
        const defaults = if (srv.samplingDefaults(instruct, f)) |d| try fields.parseNumbers(cx, d) else null;
        if (defaults) |d| try merge(options, cx, d);
        const supplied = try fields.parseNumbers(cx, f);
        try merge(options, cx, supplied);
        const temp = number(options.get("temperature")) orelse 0;
        if (temp <= 0) return null;
        if (srv.info.capability(.greedy_only)) return cx.refuse(if ((if (defaults) |d| number(d.get("temperature")) else null) orelse 0 > 0)
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

    /// The request's penalties over configured defaults; unsupported non-default values are refused by name.
    pub fn resolvePenalties(srv: *Server, cx: *Cx, f: Value, instruct: bool) errors.Refused!api.Penalties {
        const options = try json.newObject(cx.a);
        if (srv.samplingDefaults(instruct, f)) |d| try merge(options, cx, try fields.parseNumbers(cx, d));
        try merge(options, cx, try fields.parseNumbers(cx, f));
        const presence = number(options.get("presence_penalty")) orelse 0;
        const frequency = number(options.get("frequency_penalty")) orelse 0;
        if (!srv.info.capability(.penalties) and (presence != 0 or frequency != 0))
            return cx.fail(.request, "presence_penalty and frequency_penalty are not supported by backend '{s}': send 0 or leave them out", .{srv.backendName()});
        return .{ .presence = @floatCast(presence), .frequency = @floatCast(frequency) };
    }

    /// A specific engine name, or the server's backend identity when the shared host uses its default name.
    pub fn backendName(srv: *const Server) []const u8 {
        return if (std.mem.eql(u8, srv.info.name, "lanes") and srv.config.identity.backend.len > 0) srv.config.identity.backend else srv.info.name;
    }

    /// The grammar a request asks for, or its refusal when the engine cannot enforce it.
    pub fn checkFeatures(srv: *Server, cx: *Cx, f: Value, has_tools: bool, thinking: bool, prompt_ids: []const u32, tools: []const Value, request: *api.Request) errors.Refused!void {
        _ = prompt_ids;
        errdefer srv.release(false, request.*); // a refused request frees what it compiled here
        const required = if (f.get("tool_call_required")) |r| r.truthy() else false;
        const one_call = if (f.get("tool_call_single")) |x| x.truthy() else false;
        const all = has_tools and !required and (tool_grammar.autoAllEnv() catch return cx.fail(.server, "invalid TF_DSV41_TOOL_GRAMMAR: expected 0 or 1", .{}));
        if (has_tools and required) {
            if (!srv.info.capability(.structures)) return cx.fail(.request, "backend '{s}' cannot enforce tool_choice \"required\" or a named function: it has no token grammar support; send \"auto\" or \"none\"", .{srv.backendName()});
            request.structure = try tool_grammar.structure(srv, cx, tools, !one_call, thinking);
        } else if (has_tools and srv.info.capability(.structures) and tool_grammar.autoWanted(tools, all) and srv.grammarCompiler() != null) {
            // auto with a strict tool (any, TENSORFOLD_TOOL_GRAMMAR=1): text or calls; no grammar library, no grammar
            request.structure = try tool_grammar.optionalStructure(srv, cx, tools, all, !one_call, thinking);
        }
        if (try grammar.spec(cx, f)) |s| {
            if (!srv.info.capability(.structures)) return cx.refuse("structured output (response_format and the guided_* fields) is not supported by this engine yet");
            if (s.text.len > grammar.max_text) return cx.fail(.request, "{s}: the grammar is {d} bytes, more than the {d} KiB a grammar may be", .{ s.field, s.text.len, grammar.max_text / 1024 });
            const st = try srv.compiledStructure(cx, s.kind, s.text, s.field, thinking);
            if (request.structure) |tool| tool.free();
            request.structure = st;
        }
        try budgetStructure(cx, f, request);
    }

    /// Give back a request's preparing count, and its grammar once the engine is done with it (or it was never sent).
    pub fn release(srv: *Server, preparing: bool, request: ?api.Request) void {
        if (preparing) _ = srv.preparing.fetchSub(1, .acq_rel);
        if (request) |r| if (r.structure) |st| st.free();
    }

    /// A request's grammar, compiled once here and handed to the engine (a 400 with xgrammar's reason if it cannot be).
    pub fn compiledStructure(srv: *Server, cx: *Cx, kind: grammar.Kind, text: []const u8, field: []const u8, thinking: bool) errors.Refused!api.Structure {
        return srv.compiledWith(cx, srv.grammarCompiler(), srv.grammar_problem, kind, text, field, thinking);
    }

    /// A required tool call's grammar: on Inkling 2 its own compiler, which may also end at a call's end of message.
    pub fn compiledToolStructure(srv: *Server, cx: *Cx, text: []const u8, field: []const u8, thinking: bool) errors.Refused!api.Structure {
        if (!@import("chat_inkling2.zig").enabled(srv)) return srv.compiledStructure(cx, .structural_tag, text, field, thinking);
        const c = srv.compilerFor(&srv.tool_grammars, &srv.tool_grammar_problem, true);
        return srv.compiledWith(cx, c, srv.tool_grammar_problem, .structural_tag, text, field, thinking);
    }

    fn compiledWith(srv: *Server, cx: *Cx, compiler: ?*api.grammar.Compiler, problem: ?[]const u8, kind: grammar.Kind, text: []const u8, field: []const u8, thinking: bool) errors.Refused!api.Structure {
        try grammar.checkNesting(cx, kind, text, field);
        const c = compiler orelse return cx.fail(.request, "{s}: {s}", .{ if (kind == .structural_tag) field else "structured output", problem.? });
        var err: [512]u8 = undefined;
        const compiled = c.compile(srv.io, @fromBackingInt(@intCast(@backingInt(kind))), text, &err) catch
            return cx.fail(.request, "{s}: the grammar cannot be enforced: {s}", .{ field, std.mem.sliceTo(&err, 0) });
        return .{ .kind = kind, .field = field, .text = text, .after = if (thinking) srv.text.tokenId(srv.markers.close) else null, .compiler = c, .compiled = compiled };
    }

    /// grammar_budget: a boolean; true keeps a grammar's reply completable within max_tokens (none without a grammar).
    fn budgetStructure(cx: *Cx, f: Value, request: *api.Request) errors.Refused!void {
        const v = f.get("grammar_budget") orelse return;
        if (v != .bool) return cx.refuse("grammar_budget must be a boolean");
        if (!v.bool) return;
        const st = if (request.structure) |*s| s else return;
        const g = st.compiled orelse return cx.refuse("grammar_budget: the grammar was not compiled here");
        try @import("grammar_budget.zig").require(cx, g);
        st.budget = true;
    }

    /// The model's grammar compiler: tokenizer.json, the text model's vocab_size, the eos ids.
    fn grammarCompiler(srv: *Server) ?*api.grammar.Compiler {
        return srv.compilerFor(&srv.grammars, &srv.grammar_problem, false);
    }

    /// A compiler opened once into `slot`; `calls`: its ends also take Inkling 2's end of message (chat_inkling2).
    fn compilerFor(srv: *Server, slot: *?*api.grammar.Compiler, problem: *?[]const u8, calls: bool) ?*api.grammar.Compiler {
        srv.grammar_mutex.lockUncancelable(srv.io);
        defer srv.grammar_mutex.unlock(srv.io);
        if (slot.*) |g| return g;
        if (problem.* != null) return null;
        const a = srv.arena.allocator();
        problem.* = blk: {
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
            const ends = if (calls) @import("chat_inkling2.zig").grammarEnds(srv, a, srv.eos) catch break :blk "out of memory" else srv.eos;
            c.* = api.grammar.Compiler.open(srv.io, a, tok, vocab, ends, &err) catch |e| break :blk switch (e) {
                error.NoGrammarLibrary => @import("sys").optional_libraries.spec(.grammar).message,
                else => a.dupe(u8, std.mem.sliceTo(&err, 0)) catch "out of memory",
            };
            slot.* = c;
            log.line("structured output: xgrammar over {d} tokens", .{vocab});
            break :blk null;
        };
        return slot.*;
    }

    /// Counts one finished request for /metrics.
    pub fn noteRequest(srv: *Server, prompt_len: usize, cached: usize, generated: usize, drafted: u64, accepted: u64, rounds: u64, received: i96, first: ?i96, last: ?i96, prefill: ?f64, queued: ?f64) void {
        const now = clock.nowNs(srv.io);
        const tpot = srv.metrics.tpotValue(first, last, generated);
        srv.metrics.note(srv.io, prompt_len, cached, generated, drafted, accepted, rounds, @max(0, clock.seconds(now - received)), if (first) |f| clock.seconds(f - received) else null, if (first) |f| @max(0, clock.seconds(now - f)) else null, prefill, tpot);
        if (queued) |q| srv.metrics.noteQueue(srv.io, q);
    }

    /// Accepts connections until ``stop`` is set, each on its own thread.
    pub fn serve(srv: *Server, listener: listener_mod.Listener, stop: *const std.atomic.Value(bool)) void {
        const loopback_bind = if (listener.boundAddress()) |a| auth.loopbackAddress(a) else true;
        while (listener.accept(stop)) |c| {
            // counted before its thread runs, so a zero count once this loop ends means no connection is left
            if (!srv.admitConnection(c.fd)) continue;
            const slot = admission.peerSlot(&srv.peers, srv.io, &srv.open_connections, srv.config.max_connections, srv.config.peering, c.fd, c.peer) orelse continue;
            const t = std.Thread.spawn(.{ .stack_size = 8 << 20 }, connection, .{ srv, c, loopback_bind, slot }) catch {
                admission.release(&srv.peers, srv.io, &srv.open_connections, slot);
                _ = platform.posix.system.close(c.fd);
                continue;
            };
            t.detach();
        }
    }

    pub fn admitConnection(srv: *Server, fd: platform.posix.socket_t) bool {
        if (admission.acquire(&srv.open_connections, srv.config.max_connections)) return true;
        admission.reject(fd);
        return false;
    }

    fn connection(srv: *Server, accepted: listener_mod.Listener.Accepted, loopback_bind: bool, slot: u16) void {
        defer admission.release(&srv.peers, srv.io, &srv.open_connections, slot);
        defer _ = platform.posix.system.close(accepted.fd);
        var peer_buf: [64]u8 = undefined;
        const peer = listener_mod.peerText(&peer_buf, accepted.peer);
        var conn = http_conn.Conn.init(srv.gpa, accepted.fd, peer) catch return;
        defer conn.deinit();
        conn.timeouts = srv.config.timeouts;
        conn.shed = .{ .open = &srv.open_connections, .limit = srv.config.max_connections };
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
        const matched = keys.matchIdentity(a, single(conn, "Authorization"), single(conn, "x-api-key"));
        if (matched) |m| {
            conn.key_label = m.label;
            conn.key_digest = m.digest;
            conn.key_scopes = m.scopes;
        }
        if (!auth.gatedRequest(conn.method, conn.path, keys.metrics_open)) return true;
        const path = auth.routePath(conn.path);
        // a Responses WebSocket upgrade opens with an inference key; the socket checks each response.create again
        const upgrade = @import("ws_upgrade.zig").requested(conn) and std.mem.eql(u8, responses.route(path) orelse "-", "");
        if (matched != null and (scope.allows(conn.key_scopes, conn.method, conn.path) or (upgrade and scope.allowsUpgrade(conn.key_scopes)))) return true;
        const messages = for ([_][]const u8{ "/v1/messages", "/v1/messages/count_tokens", "/messages", "/messages/count_tokens" }) |p| {
            if (std.mem.eql(u8, path, p)) break true;
        } else false;
        if (matched != null) {
            // a key without this route's scope is refused before its body is read or a parked model wakes
            @import("auth_scope_reply.zig").deny(conn, messages, .route);
            return false;
        }
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

const lateSystem = @import("late_system.zig").lateSystem;

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

test {
    _ = @import("server_queue_phase_test.zig");
}
