//! One reply, as ``ChatApp.chat`` makes it: render, submit to the engine, stream text, reasoning and calls.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields_mod = @import("fields.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const prompt_mod = @import("prompt.zig");
const images = @import("images.zig");
const log = @import("log.zig");
const cached_tokens = @import("cached_tokens.zig");
const ids = @import("ids.zig");
const clock = @import("clock.zig");
const wait = @import("wait_reason.zig");
const server_mod = @import("server.zig");
const Server = server_mod.Server;
const chunk_plan = @import("chunk_plan.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const StopHook = @import("stop_hook.zig").StopHook;
const engine_refusal = @import("engine_refusal.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };
/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    /// The client itself asked for thinking off (no server, template or route default counts): it must be honored.
    explicit_off: bool = false,
    trace: ?*@import("request_trace.zig").Trace = null, // the request's journal records; null when the journal is off
    /// The server's own request (a compaction summary, a lesson's wordings): it never takes --instruct-sampling's set.
    own: bool = false,
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
    /// POST /v1/requests/{id}/cancel's ask for the client's request (cancel_ids.zig); a server-made request has none.
    cancel: ?*const std.atomic.Value(bool) = null,
    /// ``logprobs: true`` with this many ``top_logprobs``; null: not asked.
    logprobs: ?u8 = null,
    /// ``/v1/completions``' ``prompt_logprobs``: rows for the prompt with this many alternatives; null: not asked.
    prompt_logprobs: ?u8 = null,
    /// The client's own name for the grammar field (text.format, output_config.format); null on chat and completions.
    format: ?[]const u8 = null,
    /// The encoded image bytes the request may send; chat's limits unless a route (OCR) sets its own.
    image_bytes: images.ByteLimits = .{},
    /// With a known window, the reply may fill all the room the prompt leaves; max_tokens otherwise.
    fill_room: bool = false,
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
    native_tool_format: bool = false,
    native_calls: ?[]Value = null,
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
    /// ``choices[0].logprobs`` (prose, reasoning and call entries) for a request that asked; null otherwise.
    logprobs: ?Value = null,
    /// The tokens and target rows ``logprobs`` are projected from, once the route's tool policy is final.
    probability_rows: ?@import("logprob_parts.zig").Capture = null,
    /// ``choices[0].prompt_logprobs`` (vLLM's form) for a request that asked; null otherwise.
    prompt_logprobs: ?Value = null,
    loop_stop: ?u32 = null, // the loop period when --loop-guard-stop ended the reply in its reasoning
};

pub const Failure = error{ Refused, Cancelled, Failed, OutOfMemory };

const nowNs = clock.nowNs;
const seconds = clock.seconds;

test "Server recovery: token staging allocation failure ends the loop without consuming tokens" {
    for (0..2) |fail_index| {
        var m: Mailbox = .{ .io = std.testing.io, .gpa = std.testing.allocator, .finished = true };
        defer m.deinit();
        try m.tokens.append(std.testing.allocator, 42);
        try m.chunks.append(std.testing.allocator, 1);
        try m.rows.append(std.testing.allocator, .{ .token = 42, .logprob = -1 });
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        var g: Generation = undefined;
        g.a = failing.allocator();
        g.box = &m;
        g.chunk_index = 0;
        g.consumed = 0;
        g.lp = .{};
        const Gone = struct {
            pub fn check(_: @This()) bool {
                return false;
            }
        };
        try std.testing.expectError(error.OutOfMemory, g.loop(Gone{}));
        try std.testing.expectEqual(@as(usize, 0), g.consumed);
        try std.testing.expectEqual(@as(usize, 0), g.chunk_index);
    }
}

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
    reasoning: bool, // thinking on a model with Info.reasoning: the reply splits off its reasoning block
    effort: ?[]const u8,
    sampled: bool, // a sampling config, which the reply labels "exact"; greedy otherwise
    drafts: bool,
    token_ids: bool,
    stops: fields_mod.Stops,
    preparing: bool,
    frames: []const ?api.Size = &.{}, // each image part's tower input size, in part order (prompt.Rendered.frames)
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
        (priority != null and priority.? == .string and std.mem.eql(u8, priority.?.string, "background")) or
        (priority == null and @import("server.zig").namedBackground(srv.config, f));
    const preparing = !background;
    if (preparing) _ = srv.preparing.fetchAdd(1, .acq_rel);
    errdefer srv.release(preparing, null);
    if (gone.check()) return error.Cancelled;
    const thinking = input.prompt == null and (flag(f, "enable_thinking") orelse srv.config.enable_thinking);
    const instruct = input.prompt == null and !thinking and !input.own; // a client's chat with thinking off may take --instruct-sampling's set
    const reasoning = thinking and srv.info.capability(.reasoning) and (srv.text.vtable.format != .kimi or input.prompt == null); // the prompt renders `thinking`; the reply reads `reasoning`
    const asked = if (f.get("reasoning_effort")) |e| (if (e == .string) e.string else null) else null;
    const effort = srv.effortFor(asked);
    cx.named_effort = if (input.prompt == null) fields_mod.namedEffort(asked, srv.effort_levels) else null; // a raw prompt has no template
    cx.clear_thinking = flag(f, "clear_thinking");
    cx.template_fields = f;
    const rendered = try prompt_mod.prepare(srv, cx, input, thinking, effort);
    if (gone.check()) return error.Cancelled;
    if (rendered.ids.len == 0) return cx.refuse("rendered prompt is empty");
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0) {
        const room = window - n;
        if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.capability(.context_fitted)) ", the most this server's memory budget fits" else "", n });
        if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
        limit = if (input.fill_room) room else @min(limit, room);
    }
    const system_len: usize = if (input.prompt != null) 0 else prompt_mod.systemPrefixLen(srv, cx, input.messages, input.tools, rendered.ids, thinking, effort);
    var shared: std.ArrayList(u32) = .empty;
    if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
        if (cut >= 512) try shared.append(a, @intCast(cut));
    };
    const sampling = try srv.resolveSampling(cx, f, input.temperature, rendered.ids, instruct);
    const draft_field = f.get("draft");
    const drafts = srv.config.use_drafts and !(draft_field != null and draft_field.? == .bool and !draft_field.?.bool);
    var request: api.Request = .{
        .prompt = rendered.ids,
        .max_tokens = @intCast(@min(limit, std.math.maxInt(u32))),
        .sampling = sampling,
        .penalties = try srv.resolvePenalties(cx, f, instruct),
        .eos = if (stops_opt.ignore_eos) &.{} else srv.eos,
        .drafts = drafts,
        .background = background,
        .history_len = @intCast(rendered.history_len),
        .rewind_len = @intCast(rendered.rewind_len),
        .user_end = @intCast(rendered.user_end),
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
    try srv.checkFeatures(cx, f, input.tools.len > 0, reasoning, rendered.ids, input.tools, &request);
    server_mod.nameStructure(input.format, &request);
    errdefer srv.release(false, request);
    if (reasoning) {
        const budget_field = f.get("thinking_budget");
        const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else srv.config.thinking_budget;
        if (budget > 0 and !srv.info.capability(.thinking_budget))
            return cx.refuse("thinking_budget is not supported by this native engine");
        if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
        request.loop_stop = request.loop_guard and srv.config.loop_stop;
        if ((budget > 0 or request.loop_guard) and srv.think_close.len > 0) {
            if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
            request.think_close = srv.think_close;
            request.think_end = srv.think_close_end;
        }
        // The answer quota is separate only on engines that enforce it.
        if (srv.info.capability(.answer_limit)) @import("thinking_reserve.zig").apply(&request, srv.config.thinking_reserve, if (window > 0) window - n else std.math.maxInt(u32), srv.think_close_end);
    }
    request.logprobs = input.logprobs;
    request.prompt_logprobs = input.prompt_logprobs;
    if (try engine_refusal.of(srv.engine, cx, &request)) |message| return cx.refuse(message);
    if (input.trace) |trace| trace.prepared(&request, thinking, effort);
    return .{ .input = input, .request = request, .prompt_len = rendered.ids.len, .received = received, .thinking = thinking, .reasoning = reasoning, .effort = effort, .sampled = sampling != null, .drafts = drafts, .token_ids = flag(f, "return_token_ids") orelse false, .stops = stops_opt, .preparing = preparing, .frames = rendered.frames };
}

/// Submit a prepared request and collect its reply; ``sink`` hears the stream (null: not streamed).
pub fn generate(srv: *Server, cx: *Cx, prepared: Prepared, sink: ?Sink, gone: anytype) Failure!Reply {
    const a = cx.a;
    const io = srv.io;
    var request = prepared.request;
    const input = prepared.input;
    const received = prepared.received;
    const effort = prepared.effort;
    const stops_opt = prepared.stops;
    var preparing = prepared.preparing;
    defer srv.release(preparing, request); // after the engine's finished: the errdefer below drains first
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
    if (!srv.info.capability(.text_generation)) return cx.refuse("this model does not support text generation");
    if (srv.info.capability(.decides)) return cx.refuse("this model makes decisions and does not write text: use POST /v1/systemone");
    var stop_hook: StopHook = .{ .srv = srv, .stops = .{ .strings = stops_opt.strings } };
    if (stops_opt.strings.len > 0) request.stop = .{ .ctx = &stop_hook, .check = StopHook.check };
    var box: Mailbox = .{ .io = io, .gpa = srv.gpa };
    defer box.deinit();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin(); // the GPU is busy: the idle ticker holds its commits
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => cx.wait(wait.engine_busy, "the engine is busy; retry shortly", .{}),
        error.Closed => cx.wait(wait.engine_unavailable, "the engine is unavailable; retry after it restarts", .{}),
        error.InvalidSpans => cx.fail(.request, "the prompt span layout is invalid", .{}),
        error.InvalidRequest => cx.fail(.request, "the request does not match the loaded model", .{}),
        error.Stale => cx.fail(.server, "{s}", .{api.stale_message}),
        error.MemoryPressure => cx.wait(wait.memory_pressure, "{s}", .{api.pressure_message}),
    };
    srv.release(preparing, null); // a background request waits only while a foreground one prepares
    preparing = false;
    var gen: Generation = .{ .srv = srv, .a = a, .box = &box, .id = id, .asked = input.cancel, .reply_id = input.id, .seed = if (request.sampling) |v| v.seed else null, .sink = sink, .thinking = prepared.reasoning or (srv.text.vtable.format != .kimi and srv.info.capability(.reasoning) and reply_text.isChannel(srv.markers)), .chat = input.prompt == null, .stops = .{ .strings = stops_opt.strings }, .ignore_eos = stops_opt.ignore_eos, .max_tokens = request.max_tokens, .tools = input.tools, .token_ids = prepared.token_ids, .lp = if (request.logprobs != null) .{} else null, .loop_stop = request.loop_stop, .single = flag(input.fields, "tool_call_single") orelse false };
    defer srv.noteRequest(prepared.prompt_len, cached_tokens.within(box.cached, prepared.prompt_len), gen.collected.items.len, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, gen.first_ns, gen.last_ns, box.stats.prefill_seconds, if (box.stats.prefill_seconds != null) box.stats.queued_seconds else null);
    errdefer if (!gen.engine_done) gen.cancel(); // the engine writes to the mailbox until it says finished
    const result: Failure!Reply = blk: {
        if (sink) |s| if (s.begin) |begin| begin(s.ctx, prepared.prompt_len) catch break :blk error.Cancelled;
        if (sink != null and input.tools.len > 0 and srv.text.vtable.format != .kimi) gen.calls = tool_stream.Streamer.init(a, input.tools) catch |e| break :blk e;
        gen.loop(gone) catch |e| break :blk e;
        break :blk gen.finish(cx, prepared.prompt_len, received, prepared.thinking, prepared.reasoning, effort, prepared.sampled, prepared.drafts);
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

/// The token loop and the text it streams.
const Generation = @import("chat_generation.zig").Generation;

pub const deltaOf = reply_text.deltaOf;
