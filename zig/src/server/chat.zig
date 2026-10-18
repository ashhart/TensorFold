//! One reply, as ``ChatApp.chat`` makes it: render, submit to the engine, stream text, reasoning and calls.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields_mod = @import("fields.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const prompt_mod = @import("prompt.zig");
const log = @import("log.zig");
const ids = @import("ids.zig");
const clock = @import("clock.zig");
const Server = @import("server.zig").Server;
const family_mod = @import("family.zig");
const chunk_plan = @import("chunk_plan.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const chat_generation = @import("chat_generation.zig");
const Generation = chat_generation.Generation;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };

/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
    /// ``logprobs: true`` with this many ``top_logprobs``; null: not asked.
    logprobs: ?u8 = null,
    /// ``/v1/completions``' ``prompt_logprobs``: rows for the prompt with this many alternatives; null: not asked.
    prompt_logprobs: ?u8 = null,
    /// The request body (a family reads fields of its own from it, such as ``response_format``).
    body: Value = .null,
    /// ``parallel_tool_calls: false``: the reply keeps its first call (a family applies it while streaming).
    single_call: bool = false,
};

/// A streamed piece: content text (a string) or a delta object (reasoning or tool calls).
pub const Sink = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, delta: Value) error{Closed}!void,
    /// Called only after admission, with cancellation and mailbox draining already guarded.
    begin: ?*const fn (ctx: *anyopaque, prompt_len: usize) error{Closed}!void = null,
};

pub const Reply = struct {
    content: []const u8,
    stop_sequence: ?[]const u8,
    reasoning: ?[]const u8,
    tool_calls_streamed: bool,
    finish_reason: []const u8,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    runtime: Value,
    speculative: Value,
    /// The template switches the prompt was rendered with (the next turn's prefill renders with them too).
    thinking: bool = false,
    effort: ?[]const u8 = null,
    /// ``choices[0].logprobs`` (``{"content": [...]}``) for a request that asked; null otherwise.
    logprobs: ?Value = null,
    /// ``choices[0].prompt_logprobs`` (vLLM's form) for a request that asked; null otherwise.
    prompt_logprobs: ?Value = null,
    /// A family's parsed calls (OpenAI tool-call objects); null: the route parses the content itself.
    calls: ?[]Value = null,
};

pub const Failure = error{ Refused, Cancelled, Failed, OutOfMemory };

const nowNs = clock.nowNs;
const seconds = clock.seconds;

/// Python's ``bool(fields.get(key))`` on the sampling fields.
fn flag(f: Value, key: []const u8) ?bool {
    const v = f.get(key) orelse return null;
    if (v == .null) return null;
    return v.truthy();
}

/// A request rendered and checked but not submitted; it holds the preparing count until generate or release.
pub const Prepared = struct {
    input: Input,
    request: api.Request,
    prompt_len: usize,
    received: i96,
    thinking: bool,
    effort: ?[]const u8,
    sampled: bool, // a sampling config, which the reply labels "exact"; greedy otherwise
    drafts: bool,
    stops: fields_mod.Stops,
    preparing: bool,
    images_omitted: u32 = 0,
    /// the prompt's prepared images (a family's), held until the reply ends
    family_images: ?family_mod.Images = null,
};

/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.
pub fn prepare(srv: *Server, cx: *Cx, input: Input, gone: anytype) Failure!Prepared {
    const a = cx.a;
    const io = srv.io;
    const received = nowNs(io);
    const f = input.fields;
    const stops_opt = try fields_mod.stopOptions(cx, f);
    var limit: i64 = @max(1, if (input.max_tokens) |m| (if (m != 0) m else srv.config.default_max_tokens) else srv.config.default_max_tokens);
    const priority = f.get("priority");
    const background = prompt_mod.isTitle(input.messages, input.tools.len > 0) or
        (priority != null and priority.? == .string and std.mem.eql(u8, priority.?.string, "background"));
    const preparing = !background;
    if (preparing) _ = srv.preparing.fetchAdd(1, .acq_rel);
    errdefer release(srv, preparing);
    if (gone.check()) return error.Cancelled;
    var thinking = flag(f, "enable_thinking") orelse srv.config.enable_thinking;
    if (input.prompt != null) thinking = false;
    const asked: ?[]const u8 = if (f.get("reasoning_effort")) |e| (if (e == .string) e.string else null) else null;
    const effort: ?[]const u8 = if (srv.family) |fam| asked orelse srv.config.reasoning_effort orelse fam.defaultEffort() else srv.effortFor(asked);
    const rendered = try prompt_mod.prepare(srv, cx, input, thinking, effort);
    errdefer if (rendered.family_images) |im| im.release();
    if (gone.check()) return error.Cancelled;
    if (rendered.ids.len == 0) return cx.refuse("rendered prompt is empty");
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0) {
        const room = window - n;
        if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.context_fitted) ", the most this server's memory budget fits" else "", n });
        if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
        limit = @min(limit, room);
    }
    const system_len: usize = if (input.prompt != null) 0 else prompt_mod.systemPrefixLen(srv, cx, input.messages, input.tools, rendered.ids, thinking, effort);
    var shared: std.ArrayList(u32) = .empty;
    if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
        if (cut >= 512) try shared.append(a, @intCast(cut));
    };
    const sampling = try srv.resolveSampling(cx, f, input.temperature, rendered.ids);
    const draft_field = f.get("draft");
    const drafts = srv.config.use_drafts and !(draft_field != null and draft_field.? == .bool and !draft_field.?.bool);
    var request: api.Request = .{
        .prompt = rendered.ids,
        .max_tokens = @intCast(@min(limit, std.math.maxInt(u32))),
        .sampling = sampling,
        .eos = if (stops_opt.ignore_eos) &.{} else srv.eos,
        .drafts = drafts,
        .background = background,
        .history_len = @intCast(rendered.history_len),
        .rewind_len = @intCast(rendered.rewind_len),
        .shared_prefixes = shared.items,
        // a cut before the conversation's own text: fresh sessions resume their whole harness
        .chunks = try chunk_plan.withCut(a, try srv.chunks.starts(a, rendered.ids), if (srv.chunks.step > 0) @intCast(@max(system_len, 1) - 1) else 0, rendered.ids.len, srv.chunks.min_chunk),
        // from the tokens alone, for every request: cached states are keyed by tokens
        .decode_spans = try prompt_mod.replySpans(srv, cx, rendered.ids),
        .tools_json = if (input.tools.len > 0) try json.stringify(a, .{ .array = @constCast(input.tools) }, .{ .ascii = false }) else "",
        .decide = if (input.prompt == null and !background) prompt_mod.decideOf(srv, rendered.ids) else null,
        .images = rendered.images,
        .cache_key = rendered.cache_key,
    };
    try srv.checkFeatures(cx, f, input.tools.len > 0, thinking, rendered.ids, input.tools, &request);
    if (thinking) {
        const budget_field = f.get("thinking_budget");
        const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else srv.config.thinking_budget;
        if (budget > 0 and !srv.info.thinking_budget)
            return cx.refuse("thinking_budget is not supported by this native engine");
        if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
        if ((budget > 0 or request.loop_guard) and srv.think_close.len > 0) {
            if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
            request.think_close = srv.think_close;
            request.think_end = srv.think_close_end;
        }
    }
    if (input.logprobs) |count| request.logprobs = try @import("logprobs.zig").admit(srv, cx, thinking, request.think_budget, count);
    request.prompt_logprobs = input.prompt_logprobs;
    return .{ .input = input, .request = request, .prompt_len = rendered.ids.len, .received = received, .thinking = thinking, .effort = effort, .sampled = sampling != null, .drafts = drafts, .stops = stops_opt, .preparing = preparing, .images_omitted = rendered.images_omitted, .family_images = rendered.family_images };
}

/// Submit a prepared request and collect its reply; ``sink`` hears the stream (null: not streamed).
pub fn generate(srv: *Server, cx: *Cx, prepared: Prepared, sink: ?Sink, gone: anytype) Failure!Reply {
    const a = cx.a;
    const io = srv.io;
    var request = prepared.request;
    const input = prepared.input;
    const received = prepared.received;
    const thinking = prepared.thinking;
    const effort = prepared.effort;
    const stops_opt = prepared.stops;
    var preparing = prepared.preparing;
    defer release(srv, preparing);
    defer if (prepared.family_images) |im| im.release(); // after the engine's finished event (the loop below waits for it)
    if (request.background) {
        while (true) {
            const now = nowNs(io);
            const waiting = now < received + 150 * std.time.ns_per_ms or (srv.preparing.load(.acquire) > 0 and now < received + 2 * std.time.ns_per_s);
            if (!waiting) break;
            if (gone.check()) return error.Cancelled;
            std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
        }
    }
    if (gone.check()) return error.Cancelled;
    if (!srv.info.text_generation) return cx.refuse("this model does not support text generation");
    if (srv.info.decides) return cx.refuse("this model makes decisions and does not write text: use POST /v1/systemone");
    var stop_hook: StopHook = .{ .srv = srv, .stops = .{ .strings = stops_opt.strings } };
    // a family matches stop strings in the answer (``push``); the generic path in the raw text on the engine's thread
    if (stops_opt.strings.len > 0 and srv.family == null) request.stop = .{ .ctx = &stop_hook, .check = StopHook.check };
    var box: Mailbox = .{ .io = io, .gpa = srv.gpa };
    defer box.deinit();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin(); // the GPU is busy: the idle ticker holds its commits
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => cx.fail(.capacity, "the engine is busy; retry shortly", .{}),
        error.Closed => cx.fail(.capacity, "the engine is unavailable; retry after it restarts", .{}),
        error.InvalidSpans => cx.fail(.request, "the prompt span layout is invalid", .{}),
        error.Stale => cx.fail(.server, "{s}", .{api.stale_message}),
    };
    release(srv, preparing); // a background request waits only while a foreground one prepares
    preparing = false;
    var gen: Generation = .{ .srv = srv, .a = a, .box = &box, .id = id, .reply_id = input.id, .sink = sink, .thinking = thinking or reply_text.isChannel(srv.markers), .stops = .{ .strings = stops_opt.strings }, .ignore_eos = stops_opt.ignore_eos, .max_tokens = request.max_tokens, .tools = input.tools, .lp = if (request.logprobs != null) .{} else null };
    defer srv.noteRequest(prepared.prompt_len, box.cached orelse 0, gen.collected.items.len, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, gen.first_ns, gen.last_ns, box.stats.prefill_seconds);
    errdefer if (!gen.engine_done) gen.cancel(); // the engine writes to the mailbox until it says finished
    const result: Failure!Reply = blk: {
        if (sink) |s| if (s.begin) |begin| begin(s.ctx, prepared.prompt_len) catch break :blk error.Cancelled;
        if (srv.family) |fam| {
            gen.reader = fam.reader(a, thinking, input.tools, input.single_call) catch |e| break :blk e;
        } else if (sink != null and input.tools.len > 0) gen.calls = tool_stream.Streamer.init(a, input.tools) catch |e| break :blk e;
        gen.loop(gone) catch |e| break :blk e;
        break :blk gen.finish(cx, prepared.prompt_len, received, thinking, effort, prepared.sampled, prepared.drafts, prepared.images_omitted);
    };
    if (result) |_| {} else |e| logEnded(input.id, e, cx.message, prepared.prompt_len, gen.collected.items.len, seconds(nowNs(io) - received));
    return result;
}

/// The ``ended`` line of a submitted reply that ends without one: its client left, or it failed.
fn logEnded(id: []const u8, e: Failure, message: []const u8, prompt: usize, tokens: usize, after: f64) void {
    const why: ?[]const u8 = switch (e) {
        error.Cancelled => null,
        error.Refused => message, // once submitted, only the engine's own failure refuses
        else => @errorName(e),
    };
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, why, prompt, tokens, after)});
}

/// The reply to ``input``; ``sink`` hears the stream (null: not streamed). ``gone`` says the client left.
pub fn run(srv: *Server, cx: *Cx, input: Input, sink: ?Sink, gone: anytype) Failure!Reply {
    if (srv.config.compact_at == null) return generate(srv, cx, try prepare(srv, cx, input, gone), sink, gone);
    return @import("compact.zig").run(srv, cx, input, sink, gone);
}

/// Give back a foreground request's preparing count: at its submit, or when it is never generated.
pub fn release(srv: *Server, preparing: bool) void {
    if (preparing) _ = srv.preparing.fetchSub(1, .acq_rel);
}

/// The request's stop strings as the engine checks them after each token: the newest tokens' text holds one.
const StopHook = struct {
    srv: *Server,
    stops: reply_text.Stops,

    fn check(ctx: *anyopaque, emitted: []const u32) bool {
        const h: *StopHook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        const tail = emitted[emitted.len -| h.stops.tail()..];
        const text = h.srv.text.decode(arena.allocator(), tail) catch return false;
        for (h.stops.strings) |stop| if (std.mem.indexOf(u8, text, stop) != null) return true;
        return false;
    }
};

/// Close a tool call left open at the model end token, except when another stop condition ended the reply.
pub fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
    const t = g.collected.items;
    if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
    return tool_parse.closeCall(g.a, text, g.tools);
}

/// End the request and wait for the engine; unfinished requests count as disconnects.
pub fn cancel(g: *Generation) void {
    if (!g.engine_done) g.srv.engine.cancel(g.id);
    g.drain();
    if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
}

pub fn decodeText(t: anytype, a: Allocator, tokens: []const u32) ![]u8 {
    return t.decode(a, tokens);
}

/// ``{key: text}``: a streamed delta (role, content or reasoning_content).
pub fn deltaOf(a: Allocator, key: []const u8, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, key, .{ .string = text });
    return .{ .object = o };
}
