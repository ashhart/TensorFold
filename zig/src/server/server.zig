//! The server: one engine and tokenizer behind the HTTP routes, one thread a connection.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const auth = @import("auth.zig");
const metrics = @import("metrics.zig");
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const http_conn = @import("http_conn.zig");
const listener_mod = @import("listener.zig");
const routes = @import("routes.zig");
const responses = @import("responses.zig");
const grammar = @import("grammar.zig");
const clock = @import("clock.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const chunk_plan = @import("chunk_plan.zig");
const log = @import("log.zig");
const live = @import("live.zig");
const slide_graph = @import("slide_graph.zig");
const slide_lesson = @import("slide_lesson.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// auto compacts near the window. fraction compacts past that share of it.
pub const CompactAt = union(enum) { auto, fraction: f64 };

pub const Config = struct {
    served_name: []const u8,
    model_dir: []const u8 = "",
    /// The served name first, then aliases, without repeats.
    model_ids: []const []const u8,
    default_max_tokens: i64 = 4096,
    enable_thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    loop_guard: bool = false,
    /// Seconds the idle keepalive runs after the last request ends (0: off).
    keep_warm_s: i64 = 900,
    dashboard: bool = false,
    /// Sampling when a request names none (generation_config.json and the serve flags), or null for greedy.
    default_sampling: ?Value = null,
    use_drafts: bool = true,
    seed_salt: i64 = 0,
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
    timeouts: http_conn.Timeouts = .{},
};

pub const Server = struct {
    gpa: Allocator,
    io: std.Io,
    engine: api.Engine,
    info: api.Info,
    text: model_text.Text,
    config: Config,
    keys: ?*auth.Store,
    late_system: []const u8 = "system",
    needs_user_after_tool: bool = false,
    markers: reply_text.Markers = reply_text.think_markers,
    chunks: chunk_plan.Plan = .{},
    effort_levels: []const []const u8 = &.{},
    eos: []const u32 = &.{},
    think_close: []const u32 = &.{},
    think_close_end: ?u32 = null,
    metrics: metrics.Metrics,
    store: responses.Store,
    next_id: std.atomic.Value(u64) = .init(1),
    /// Connections open now; a stop waits for them before freeing what they read.
    open_connections: std.atomic.Value(u32) = .init(0),
    /// The idle keepalive on the engine's queue, armed by --keep-warm; null when off or not Metal.
    keepalive: ?*api.keepalive.Keepalive = null,
    preparing: std.atomic.Value(i64) = .init(0),
    arena: std.heap.ArenaAllocator,
    /// Structured output's compiler, opened by the first structured request (Python's grammar.compiler).
    grammars: ?*api.grammar.Compiler = null,
    grammar_problem: ?[]const u8 = null,
    grammar_mutex: std.Io.Mutex = .init,
    /// The facts the learner was taught and whether the weights recall them.
    slide: slide_graph.Graph,
    teacher: slide_lesson.Teacher, // what every Sliding Weights lesson shares: keep prompts, the turn's end

    /// Reads what the template and tokenizer decide once: late system role, think markers, efforts, the forced close.
    pub fn init(gpa: Allocator, io: std.Io, engine: api.Engine, text: model_text.Text, config: Config, keys: ?*auth.Store) !*Server {
        const srv = try gpa.create(Server);
        srv.* = .{ .gpa = gpa, .io = io, .engine = engine, .info = engine.info(), .text = text, .config = config, .keys = keys, .metrics = .{ .gpa = gpa }, .store = .{ .gpa = gpa }, .arena = .init(gpa), .slide = .init(gpa, io, config.slide_graph), .teacher = .init(gpa) };
        if (config.keep_warm_s > 0) if (engine.keepaliveTarget()) |target| {
            srv.keepalive = api.keepalive.Keepalive.start(gpa, io, target, @as(i64, config.keep_warm_s) * std.time.ns_per_s) catch |e| blk: {
                log.line("idle keepalive off: {s}", .{@errorName(e)});
                break :blk null;
            };
        };
        const a = srv.arena.allocator();
        srv.eos = try a.dupe(u32, text.eosIds());
        if (text.tokenId(reply_text.channel_markers.close) != null) srv.markers = reply_text.channel_markers;
        srv.effort_levels = try fields.effortLevels(a, text.templateSource());
        srv.late_system = try lateSystem(a, text);
        srv.needs_user_after_tool = needsUserAfterTool(text.templateSource());
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

    pub fn deinit(srv: *Server) void {
        if (srv.keepalive) |k| k.stop(); // before the engine's queue goes away
        if (srv.grammars) |g| g.close();
        srv.metrics.deinit();
        srv.arena.deinit();
        srv.store.deinit();
        srv.slide.deinit();
        srv.teacher.deinit();
        srv.gpa.destroy(srv);
    }

    /// The effort the template hears: the request's name, or this server's default.
    pub fn effortFor(srv: *const Server, explicit: ?[]const u8) ?[]const u8 {
        const e = explicit orelse srv.config.reasoning_effort orelse return null;
        return fields.coerceEffort(e, srv.effort_levels);
    }

    /// ``_resolve_sampling``: omitted or null fields keep the defaults; an omitted seed is keyed to the prompt.
    pub fn resolveSampling(srv: *Server, cx: *Cx, f: Value, temperature: f64, prompt_ids: []const u32) errors.Refused!?api.Sampling {
        _ = temperature; // the request's own temperature field decides, as in the Python server
        const options = try json.newObject(cx.a);
        const defaults = if (srv.config.default_sampling) |d| try fields.parseNumbers(cx, d) else null;
        if (defaults) |d| try merge(options, cx, d);
        try merge(options, cx, try fields.parseNumbers(cx, f));
        const temp = number(options.get("temperature")) orelse 0;
        if (temp <= 0) return null;
        if (srv.info.greedy_only) return cx.refuse(if ((if (defaults) |d| number(d.get("temperature")) else null) orelse 0 > 0)
            "this engine decodes greedily only and this server samples by default: send temperature 0, or serve with --temperature 0"
        else
            "this engine decodes greedily only: send temperature 0 or leave it out");
        const seed: u64 = if (options.get("seed")) |s| seedBits(s) else api.seedFor(prompt_ids, srv.config.seed_salt);
        return .{
            .seed = seed,
            .temperature = @floatCast(temp),
            .top_k = @intCast(@min(@max(0, (if (options.get("top_k")) |k| k.int64() else null) orelse 0), std.math.maxInt(u32))),
            .top_p = @floatCast(number(options.get("top_p")) orelse 1.0),
            .min_p = @floatCast(number(options.get("min_p")) orelse 0.0),
        };
    }

    /// The call gate and grammar a request asks for, or its refusal when the engine cannot enforce them.
    pub fn checkFeatures(srv: *Server, cx: *Cx, f: Value, has_tools: bool, thinking: bool, prompt_ids: []const u32, tools: []const Value, request: *api.Request) errors.Refused!void {
        _ = prompt_ids;
        const required = if (f.get("tool_call_required")) |r| r.truthy() else false;
        if (has_tools and required) {
            const form = try srv.callForm(cx);
            const opener = if (form) |fm| srv.text.tokenId(fm.opener) else null;
            if (opener == null) return cx.refuse("tool_choice \"required\" or a named function needs a chat template that marks tool calls (<tool_call> or <|tool_call>), and this one does not: send \"auto\"");
            if (!srv.info.call_gates) return cx.refuse("tool_choice \"required\" or a named function is not supported by this engine yet: send \"auto\"");
            var names: std.ArrayList([]const u8) = .empty;
            if (form.?.lead != null) for (tools) |t| try names.append(cx.a, try tool_specs.toolName(cx.a, t));
            const think_open = if (srv.markers.open.len > 0) blk: {
                const ids = srv.text.encode(cx.a, srv.markers.open, false) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Template => &[_]u32{},
                };
                break :blk if (ids.len > 0) ids[0] else null;
            } else srv.text.tokenId("<think>");
            request.call = .{ .opener = opener.?, .lead = form.?.lead orelse "", .tail = form.?.tail orelse "", .names = names.items, .think_open = think_open, .think_end = srv.text.tokenId(srv.markers.close) };
        }
        if (try grammar.spec(cx, f)) |s| {
            if (!srv.info.structures) return cx.refuse("structured output (response_format and the guided_* fields) is not supported by this engine yet");
            const c = srv.grammarCompiler() orelse return cx.fail(.request, "structured output: {s}", .{srv.grammar_problem.?});
            var err: [512]u8 = undefined;
            const compiled = c.compile(srv.io, @fromBackingInt(@intCast(@backingInt(s.kind))), s.text, &err) catch
                return cx.fail(.request, "{s}: the grammar cannot be enforced: {s}", .{ s.field, std.mem.sliceTo(&err, 0) });
            compiled.free(); // kept in the compiler's cache for the engine's compile
            request.structure = .{ .kind = s.kind, .text = s.text, .after = if (thinking) srv.text.tokenId(srv.markers.close) else null, .compiler = c };
        }
    }

    /// The model's grammar compiler: tokenizer.json, config.json's vocab_size (text_config first), the eos ids.
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
            var vocab: ?u32 = null;
            for ([_]?json.Value{ cfg.get("text_config"), cfg }) |part| if (vocab == null) if (part) |p| if (p.get("vocab_size")) |v| if (v.int64()) |n| {
                vocab = @intCast(n);
            };
            if (vocab == null) break :blk "structured output needs the checkpoint's config.json vocab_size";
            var err: [512]u8 = undefined;
            const c = a.create(api.grammar.Compiler) catch break :blk "out of memory";
            c.* = api.grammar.Compiler.open(tok, vocab.?, srv.eos, &err) catch |e| break :blk switch (e) {
                error.NoGrammarLibrary => "the server cannot load " ++ api.grammar.library ++ " (xgrammar; zig build grammar)",
                else => a.dupe(u8, std.mem.sliceTo(&err, 0)) catch "out of memory",
            };
            srv.grammars = c;
            log.line("structured output: xgrammar over {d} tokens", .{vocab.?});
            break :blk null;
        };
        return srv.grammars;
    }

    const Form = struct { opener: []const u8, lead: ?[]const u8, tail: ?[]const u8 };

    /// (opener, lead, tail) of this template's tool calls, read from a rendered probe call.
    fn callForm(srv: *Server, cx: *Cx) errors.Refused!?Form {
        const probe = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"tfprobe_fn\", \"arguments\": {}}}]}]")).ok;
        var scratch: Cx = .{ .a = cx.a };
        const text = if (prompt.renderIds(srv, &scratch, probe, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |_| "";
        if (std.mem.lastIndexOf(u8, text, "tfprobe_fn")) |at| {
            var best: ?usize = null;
            var opener: []const u8 = "";
            for (reply_text.calls) |c| if (std.mem.lastIndexOf(u8, text[0..at], c[0])) |s| {
                if (best == null or s > best.? or (s == best.? and std.mem.order(u8, c[0], opener) == .gt)) {
                    best = s;
                    opener = c[0];
                }
            };
            if (best) |s| {
                const end = at + "tfprobe_fn".len;
                return .{ .opener = opener, .lead = text[s + opener.len .. at], .tail = if (end < text.len) text[end .. end + 1] else "" };
            }
        }
        for (reply_text.calls) |c| if (srv.text.tokenId(c[0]) != null) return .{ .opener = c[0], .lead = null, .tail = null };
        return null;
    }

    /// Counts one finished request for /metrics.
    pub fn noteRequest(srv: *Server, prompt_len: usize, cached: usize, generated: usize, drafted: u64, accepted: u64, rounds: u64, received: i96, first: ?i96, last: ?i96, prefill: ?f64) void {
        const now = clock.nowNs(srv.io);
        const tpot = srv.metrics.tpotValue(first, last, generated);
        srv.metrics.note(srv.io, prompt_len, cached, generated, drafted, accepted, rounds, @max(0, clock.seconds(now - received)), if (first) |f| clock.seconds(f - received) else null, if (first) |f| @max(0, clock.seconds(now - f)) else null, prefill, tpot);
    }

    /// Accepts connections until ``stop`` is set, each on its own thread.
    pub fn serve(srv: *Server, listener: listener_mod.Listener, stop: *const std.atomic.Value(bool)) void {
        while (listener.accept(stop)) |c| {
            const t = std.Thread.spawn(.{ .stack_size = 8 << 20 }, connection, .{ srv, c }) catch {
                _ = std.posix.system.close(c.fd);
                continue;
            };
            t.detach();
        }
    }

    fn connection(srv: *Server, accepted: listener_mod.Listener.Accepted) void {
        _ = srv.open_connections.fetchAdd(1, .acq_rel);
        defer _ = srv.open_connections.fetchSub(1, .acq_rel);
        defer _ = std.posix.system.close(accepted.fd);
        var peer_buf: [64]u8 = undefined;
        const peer = listener_mod.peerText(&peer_buf, accepted.peer);
        var conn = http_conn.Conn.init(srv.gpa, accepted.fd, peer) catch return;
        defer conn.deinit();
        conn.timeouts = srv.config.timeouts;
        conn.auth_enabled = srv.keys != null and srv.keys.?.enabled();
        if (conn.auth_enabled) conn.hook = .{ .ctx = srv, .call = countReply };
        var arena: std.heap.ArenaAllocator = .init(srv.gpa);
        defer arena.deinit();
        while (true) {
            _ = arena.reset(.retain_capacity);
            const a = arena.allocator();
            const outcome = conn.readRequest(a, true) catch break;
            switch (outcome) {
                .closed => break,
                .handled => if (conn.close) break else continue,
                .ready => {},
            }
            if (!srv.authenticate(&conn, a)) break;
            if (conn.header("Expect")) |expect| if (std.ascii.eqlIgnoreCase(expect, "100-continue") and std.mem.order(u8, conn.version, "HTTP/1.1") != .lt) conn.sendContinue();
            routes.dispatch(srv, &conn, a);
            if (conn.close or conn.broken) break;
        }
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
        if (!auth.gated(conn.path, keys.metrics_open) or conn.key_label != null) return true;
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
        conn.finish(body) catch {};
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
    return if (std.mem.indexOf(u8, rendered, probe_text) != null) "system" else "user";
}

/// ``needs_user_after_tool``: the template raises "No user query found" when the conversation has no user query, which a conversation whose user turn became tool results no longer has; such templates gain a user turn after a trailing tool message.
fn needsUserAfterTool(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "No user query found") != null;
}
