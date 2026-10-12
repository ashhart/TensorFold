//! A submitted reply's collection: the engine's events in, streamed deltas and the finished ``Reply`` out.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Server = @import("server.zig").Server;
const family_mod = @import("family.zig");
const chunk_plan = @import("chunk_plan.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const chat = @import("chat.zig");
const Sink = chat.Sink;
const Reply = chat.Reply;
const Failure = chat.Failure;
const Mailbox = chat.Mailbox;
const decodeText = chat.decodeText;
const deltaOf = chat.deltaOf;
const nowNs = clock.nowNs;
const seconds = clock.seconds;

/// The text a stream has sent: Python's ``streamed = visible``, grown by its delta while it only grows.
pub const Shown = struct {
    text: std.ArrayList(u8) = .empty,
    chars: usize = 0,
    at: ?[*]const u8 = null, // where the last shown text lay; the decode buffer only grows, so that prefix holds
    extends: bool = false,

    /// Python's ``now[len(shown):]``; ``stable``: ``now`` lies in the append-only decode buffer.
    fn after(s: *Shown, now: []const u8, stable: bool) []const u8 {
        const sent = s.text.items;
        s.extends = (stable and s.at == now.ptr and now.len >= sent.len) or std.mem.startsWith(u8, now, sent);
        if (!s.extends) return reply_text.afterChars(now, s.chars);
        if (stable) s.at = now.ptr; // the buffer moved but kept its bytes: the next check is free again
        return now[sent.len..];
    }

    fn set(s: *Shown, a: Allocator, now: []const u8, delta: []const u8) Allocator.Error!void {
        if (s.extends) {
            try s.text.appendSlice(a, delta);
            s.chars += reply_text.charCount(delta);
        } else {
            s.text = .empty;
            try s.text.appendSlice(a, now);
            s.chars = reply_text.charCount(now);
        }
        s.at = now.ptr;
    }
};

/// The token loop and the text it streams.
pub const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    sink: ?Sink,
    thinking: bool,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: Shown = .{},
    streaming_done: bool = false,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    /// when the stream last carried bytes (a keepalive goes out after a silence)
    wrote_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    logprobs: bool = false, // the request asked for logprobs: the mailbox holds a row a token
    // a family's reply: its reader, the answer so far, how much of it streamed, and a stop string's hit
    reader: ?family_mod.Reader = null,
    answer: std.ArrayList(u8) = .empty,
    answer_sent: usize = 0,
    deltas: std.ArrayList(Value) = .empty,
    stop_hit: bool = false,
    stop_at: usize = 0,

    pub fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    const closeCall = chat.closeCall; // in chat.zig with its doc

    /// The engine's stop check, here: the newest tokens' text holds a stop string.
    fn stopHit(g: *Generation) Allocator.Error!bool {
        if (g.stops.strings.len == 0) return false;
        const t = g.collected.items;
        const tail = t[t.len -| g.stops.tail()..];
        const text = try g.srv.text.decode(g.a, tail);
        for (g.stops.strings) |s| if (std.mem.indexOf(u8, text, s) != null) return true;
        return false;
    }

    /// Commits a round's tokens as ``LaneStream.commit`` would, then streams what they add.
    fn commit(g: *Generation, chunk: []const u32) Failure!void {
        if (g.reader) |r| return g.commitFamily(r, chunk);
        var landed: usize = 0;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
            } else if (try g.stopHit()) {
                g.reason = "stop";
                g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        const sink = g.sink orelse return;
        if (g.streaming_done) return;
        var fresh: std.ArrayList(u32) = .empty;
        for (chunk[0..landed]) |t| {
            if (g.eos(t)) {
                g.streaming_done = true;
                break;
            }
            try fresh.append(g.a, t);
        }
        const all = try g.visible.extend(g.a, g.srv.text, decodeText, fresh.items);
        const text = g.stops.visible(all, true);
        if (g.visible.pending()) return; // a character still split across tokens
        var answer = text;
        if (g.thinking) {
            const split = try reply_text.splitThinking(g.a, text, false, g.srv.markers);
            const piece = g.streamed_reasoning.after(split.reasoning, true);
            if (piece.len > 0) {
                try g.streamed_reasoning.set(g.a, split.reasoning, piece);
                try g.emit(sink, try deltaOf(g.a, "reasoning_content", piece));
            }
            answer = split.answer;
        }
        const shown = if (g.calls != null) try reply_text.hideInto(g.a, &g.hidden, answer, false) else answer;
        const vis = reply_text.heldBack(reply_text.streamingVisible(shown), g.calls != null);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) {
            try g.streamed.set(g.a, vis, delta);
            try g.emit(sink, .{ .string = delta });
        }
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.wrote_ns = nowNs(g.srv.io);
    }

    pub const cancel = chat.cancel; // in chat.zig with its doc

    /// Waits for the engine's own end, so the request it holds may be freed.
    pub fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    pub fn loop(g: *Generation, gone: anytype) Failure!void {
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                g.chunk_index += 1;
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch null;
                g.consumed = end;
            } else if (m.finished) {
                ended = true;
            } else {
                m.cond.waitTimeout(m.io, &m.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
            }
            m.mutex.unlock(m.io);
            if (ended) {
                g.engine_done = true;
                return;
            }
            if (gone.check()) {
                g.cancel();
                return error.Cancelled;
            }
            if (chunk == null and @hasDecl(@TypeOf(gone), "beat")) {
                // queueing and a long prefill send nothing: a client that times out a silent stream would leave
                const now = nowNs(g.srv.io);
                const since = g.wrote_ns orelse now;
                if (g.wrote_ns == null) g.wrote_ns = now;
                if (gone.beat(now - since) catch {
                    g.cancel();
                    return error.Cancelled;
                }) g.wrote_ns = now;
            }
            const tokens = chunk orelse continue;
            if (g.reason == null) try g.commit(tokens);
        }
    }

    /// A family's round: committed tokens through the family's reader, its deltas pushed (an EOS id ends, unshown).
    fn commitFamily(g: *Generation, r: family_mod.Reader, chunk: []const u32) Failure!void {
        var landed: usize = 0;
        const start = g.collected.items.len;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        for (g.collected.items[start..]) |t| if (!g.isEos(t)) try r.push(&.{t});
        if (r.pending()) return; // a character still split across tokens
        g.deltas.clearRetainingCapacity();
        try r.feed(false, &g.deltas);
        try g.pushFamily(g.deltas.items, false);
    }

    /// Any end-of-sentence id, ``ignore_eos`` or not: never text.
    fn isEos(g: *const Generation, t: u32) bool {
        return std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// ``push``: reasoning and calls go out as they come; the answer is held where a stop may begin, cut where it does.
    fn pushFamily(g: *Generation, deltas: []const Value, finished: bool) Failure!void {
        for (deltas) |d| switch (d) {
            .string => |t| try g.answer.appendSlice(g.a, t),
            else => if (g.sink) |sink| if (!g.streaming_done or finished) try g.emit(sink, d),
        };
        var full = g.answer.items;
        const stops = g.stops.strings;
        if (stops.len > 0 and !g.stop_hit) {
            var longest: usize = 0;
            for (stops) |st| longest = @max(longest, st.len);
            const from = (g.answer_sent + 1) -| longest;
            var cut: ?usize = null;
            for (stops) |st| if (std.mem.indexOfPos(u8, full, @min(from, full.len), st)) |at| {
                if (cut == null or at < cut.?) cut = at;
            };
            if (cut) |c| {
                g.answer.shrinkRetainingCapacity(c);
                full = g.answer.items;
                g.stop_hit = true;
                g.stop_at = g.collected.items.len;
                if (g.reason == null) {
                    g.reason = "stop";
                    g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
                }
            }
        }
        var upto = full.len;
        if (!finished and !g.stop_hit) {
            var hold: usize = 0;
            for (stops) |st| {
                var k = @min(st.len - 1, full.len);
                while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, full, st[0..k])) break;
                hold = @max(hold, k);
            }
            upto -= hold;
        }
        if (upto > g.answer_sent) {
            if (g.sink) |sink| try g.emit(sink, .{ .string = full[g.answer_sent..upto] });
            g.answer_sent = upto;
        }
    }

    /// A family's finished reply: the stream's last deltas, then the parts (``app._run``'s end).
    fn finishFamily(g: *Generation, r: family_mod.Reader) Failure!struct { content: []const u8, reasoning: ?[]const u8, calls: []Value, reason: []const u8, used: usize } {
        const streaming = g.sink != null;
        g.deltas.clearRetainingCapacity();
        try r.feed(true, &g.deltas);
        try g.pushFamily(g.deltas.items, true);
        const parsed = try r.parse();
        var content: []const u8 = undefined;
        var calls: []Value = undefined;
        if (streaming) {
            calls = try r.calls();
            content = g.answer.items[0..g.answer_sent];
        } else {
            calls = parsed.calls;
            content = parsed.content;
            for (g.stops.strings) |st| if (std.mem.indexOf(u8, content, st)) |at| {
                content = content[0..at];
                if (!g.stop_hit) {
                    g.stop_hit = true;
                    g.stop_at = g.collected.items.len;
                }
            };
        }
        const t = g.collected.items;
        const reason: []const u8 = if (calls.len > 0) "tool_calls" else if (g.stop_hit or (!g.ignore_eos and t.len > 0 and g.isEos(t[t.len - 1]))) "stop" else "length";
        if (g.srv.family) |fam| fam.remember(parsed.reasoning, content, calls);
        // tools offered, no call, but tool markup: log up to 300 bytes of what the parser saw (UTF-8 cut, escaped)
        if (calls.len == 0 and g.tools.len > 0) if (r.markup() catch null) |m|
            std.log.warn("tool markup with no call: request {s}, finish {s}: {s}; reply bytes {d}..: {f}", .{ g.reply_id, reason, m.reason, m.at, std.zig.fmtString(m.window) });
        return .{ .content = content, .reasoning = if (parsed.reasoning.len > 0) parsed.reasoning else null, .calls = calls, .reason = reason, .used = if (g.stop_hit and g.stop_at > 0) g.stop_at else t.len };
    }

    pub fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, thinking: bool, effort: ?[]const u8, exact: bool, drafts: bool, images_omitted: u32) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done) g.drain();
        const reason_generic: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .cancelled => "cancelled",
            .failed => return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed"),
        };
        var reason = reason_generic;
        var raw: []const u8 = "";
        var content: []const u8 = undefined;
        var reasoning: ?[]const u8 = null;
        var calls: ?[]Value = null;
        var used = g.collected.items.len;
        var logprobs: ?Value = null;
        if (g.reader) |r| {
            const f = try g.finishFamily(r);
            content = f.content;
            reasoning = f.reasoning;
            calls = f.calls;
            used = f.used;
            if (!(m.reason == .cancelled and g.reason == null)) reason = f.reason;
        } else {
            const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
            raw = try g.srv.text.decode(a, content_tokens);
            if (g.logprobs) logprobs = try @import("logprobs.zig").value(g.srv, cx, a, g.thinking, content_tokens, m.rows.items);
            const visible = g.stops.visible(raw, false);
            // closed before the think split, so a call ending an unclosed think block is the answer
            const close = try g.closeCall(visible);
            const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
            content = text;
            if (g.thinking) {
                const split = try reply_text.splitThinking(a, text, true, g.srv.markers);
                const r = reply_text.pyStrip(split.reasoning);
                reasoning = if (r.len > 0) r else null;
                content = split.answer;
            } else {
                const h = reply_text.parseHarmony(text);
                content = h.content;
                reasoning = h.reasoning;
            }
            if (g.sink) |sink| {
                const thought = g.streamed_reasoning.text.items;
                if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                    try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
                const shown = if (g.calls != null) try reply_text.finishedProse(a, content) else content;
                const sent = g.streamed.text.items;
                if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) try g.emit(sink, .{ .string = shown[sent.len..] });
                if (g.calls) |*c| if (close.len > 0) {
                    // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                    var out: std.ArrayList(Value) = .empty;
                    try c.feed(if (g.thinking) content else text, &out);
                    for (out.items) |d| try g.emit(sink, d);
                };
            }
        }
        const finished_ns = nowNs(g.srv.io);
        const prefilled = m.prefilled_ns;
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, decode_s) }); // the reply's decode seconds, as tokens_per_second divides them
        try runtime.put(a, "prefill_seconds", if (m.stats.prefill_seconds) |p| .{ .float = p } else .null);
        const widths = try a.alloc(Value, m.widths.len);
        for (m.widths, widths) |w, *slot| slot.* = try json.intValue(a, w);
        try runtime.put(a, "prefill_widths", .{ .array = widths });
        const raised = try a.alloc(Value, m.raised.len);
        for (m.raised, raised) |r, *slot| slot.* = .{ .bool = r };
        try runtime.put(a, "prefill_raised", .{ .array = raised });
        try runtime.put(a, "time_to_first_token", if (g.first_ns) |t| .{ .float = seconds(t - received) } else .null);
        try runtime.put(a, "sampling", .{ .string = if (exact) "exact" else "greedy" });
        try runtime.put(a, "drafts", .{ .bool = drafts });
        const sha = @import("tokens.zig").tokenSha(g.collected.items);
        try runtime.put(a, "token_sha", .{ .string = try a.dupe(u8, &sha) });
        if (images_omitted > 0) try runtime.put(a, "images_omitted", try json.intValue(a, images_omitted));
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            try runtime.put(a, "loop", .{ .object = loop_field });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        if (m.telemetry.len > 0) if ((try json.parse(a, m.telemetry)) == .ok) try spec.put(a, "proposer", (try json.parse(a, m.telemetry)).ok);
        const think_end: ?u32 = if (thinking) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = g.calls != null and g.calls.?.streamed,
            .finish_reason = reason,
            .prompt_tokens = prompt_len,
            .cached_tokens = m.cached orelse 0,
            .completion_tokens = used,
            .reasoning_tokens = reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .thinking = thinking,
            .effort = effort,
            .logprobs = logprobs,
            .calls = calls,
        };
        if (std.mem.eql(u8, reason, "length") and thinking and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted });
        return reply;
    }
};
