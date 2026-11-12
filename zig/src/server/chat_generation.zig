//! One admitted reply: token ownership, streaming, completion and native-family parsing.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const recipient_reply = @import("recipient_reply.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const tool_lfm2 = @import("tool_lfm2.zig");
const log = @import("log.zig");
const cached_tokens = @import("cached_tokens.zig");
const clock = @import("clock.zig");
const Server = @import("server.zig").Server;
const Mailbox = @import("mailbox.zig").Mailbox;
const Cx = @import("errors.zig").Cx;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Failure = chat.Failure;
const Reply = chat.Reply;
const Sink = chat.Sink;
const nowNs = clock.nowNs;
const seconds = clock.seconds;
const deltaOf = reply_text.deltaOf;

pub const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    seed: ?u64 = null,
    sink: ?Sink,
    thinking: bool,
    chat: bool = true,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    single: bool, // parallel_tool_calls: false: a cut call is no call, so its prose ends as the whole reply's does
    token_ids: bool, // "return_token_ids": the response's tensorfold block carries the reply's token ids
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    recipient: ?recipient_reply.Reading = null, // a chat whose reply is messages to recipients (Muse Spark's)
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    kimi: ?@import("kimi_reply.zig").Stream = null,
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: reply_text.Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: reply_text.Shown = .{},
    streaming_done: bool = false,
    wrote_ns: ?i96 = null,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    lp: ?@import("logprobs.zig").Stream = null, // the request asked for logprobs: its rows, copied with each round
    loop_stop: bool = false, // --loop-guard-stop: the engine ends the reply where the loop guard fires

    fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// Close a tool call left open at the model end token, except when another stop condition ended the reply.
    fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
        const t = g.collected.items;
        if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
        return tool_parse.closeCall(g.a, text, g.tools);
    }

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
        const all = try g.visible.extend(g.a, g.srv.text, reply_text.decodeText, fresh.items);
        if (g.visible.pending()) return; // a character still split across tokens
        if (g.srv.text.vtable.format == .kimi) return @import("chat_kimi.zig").stream(g, sink);
        const raw_visible = g.stops.visible(all, true);
        const text = try @import("tool_minimax.zig").dsml(g.a, raw_visible, g.tools); // MiniMax-M3's calls as DSML
        var answer = text;
        if (g.thinking or g.recipient != null) {
            const split = if (g.recipient) |*r| try r.split(g.a, text, .streaming) else blk: {
                const original_close = try g.visible.closeOf(g.a, reply_text.decodeText, g.srv.text, g.srv.close_id);
                const close = try @import("tool_minimax.zig").closeAfterRewrite(g.a, raw_visible, g.tools, original_close);
                break :blk try reply_text.splitThinkingAt(g.a, text, false, g.srv.markers, close);
            };
            const piece = g.streamed_reasoning.after(reply_text.pyStrip(split.reasoning), true); // the whole reply strips it
            if (piece.len > 0) {
                try g.streamed_reasoning.set(g.a, reply_text.pyStrip(split.reasoning), piece);
                try g.emit(sink, try deltaOf(g.a, "reasoning_content", piece));
            }
            answer = split.answer;
        }
        answer = try tool_lfm2.dsml(g.a, answer, g.tools, false); // the answer only, so a call the reasoning mentions is not made
        const shown = if (g.calls != null) try reply_text.hideInto(g.a, &g.hidden, answer, false, true, reply_text.openersOf(g.srv.markers)) else answer;
        const vis = reply_text.heldBack(if (g.srv.info.capability(.reasoning)) reply_text.streamingVisible(shown) else shown, g.calls != null);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) {
            try g.streamed.set(g.a, vis, delta);
            try g.emit(sink, if (g.lp) |*l| (try l.delta(g.srv, g.a, g.thinking, g.ignore_eos, g.collected.items, delta, g.streamed.text.items.len, false)).? else .{ .string = delta });
        }
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    pub fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.wrote_ns = nowNs(g.srv.io);
    }

    /// End the request and wait for the engine; unfinished requests count as disconnects.
    pub fn cancel(g: *Generation) void {
        if (!g.engine_done) g.srv.engine.cancel(g.id);
        g.drain();
        if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
    }

    /// Waits for the engine's own end, so the request it holds may be freed.
    fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    pub fn loop(g: *Generation, gone: anytype) Failure!void {
        const Gone = switch (@typeInfo(@TypeOf(gone))) {
            .pointer => |p| p.child,
            else => @TypeOf(gone),
        };
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch |e| {
                    m.mutex.unlock(m.io);
                    return e;
                };
                if (g.lp) |*l| l.rows.appendSlice(g.a, m.rows.items[g.consumed..end]) catch |e| {
                    g.a.free(chunk.?);
                    m.mutex.unlock(m.io);
                    return e;
                };
                g.chunk_index += 1;
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
            if (chunk == null and @hasDecl(Gone, "beat")) {
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

    pub fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, thinking: bool, reasons: bool, effort: ?[]const u8, exact: bool, drafts: bool) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done) g.drain();
        const ended: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .silence => return cx.other("the text engine returned a speech-only finish reason"),
            .cancelled => "cancelled",
            .failed => return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed"),
        };
        const stopped: ?u32 = if (g.loop_stop) m.stats.loop_period else null;
        const reason = if (stopped != null) @import("loop_stop.zig").finish_reason else ended;
        const finished_ns = nowNs(g.srv.io);
        const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
        const raw = try g.srv.text.decode(a, content_tokens);
        const native = if (g.srv.text.vtable.format == .kimi) try @import("chat_kimi.zig").read(g, content_tokens, true) else null;
        const raw_visible = g.stops.visible(raw, false);
        const visible = if (native) |n| n.content else try @import("tool_minimax.zig").dsml(a, raw_visible, g.tools);
        // closed before the think split, so a call ending an unclosed think block is the answer
        const close = if (native != null) "" else try g.closeCall(visible);
        const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
        var content: []const u8 = text;
        var reasoning: ?[]const u8 = null;
        if (native) |n| {
            const r = reply_text.pyStrip(n.reasoning);
            reasoning = if (r.len > 0) r else null;
        } else if (g.recipient) |*rr| {
            const split = try rr.split(a, text, if (std.mem.eql(u8, reason, "length")) .cut else if (g.stops.matched(raw) != null) .stopped else .ended);
            const r = reply_text.pyStrip(split.reasoning);
            reasoning = if (r.len > 0) r else null;
            content = split.answer;
        } else if (g.thinking) {
            const original_close = try reply_text.closeWhere(a, reply_text.decodeText, g.srv.text, content_tokens, g.srv.close_id);
            const close_at = try @import("tool_minimax.zig").closeAfterRewrite(a, raw_visible, g.tools, original_close);
            const split = try reply_text.splitThinkingAt(a, text, true, g.srv.markers, close_at);
            const r = reply_text.pyStrip(split.reasoning);
            reasoning = if (r.len > 0) r else null;
            content = split.answer;
        } else if (g.srv.info.capability(.reasoning)) {
            const h = reply_text.parseHarmony(text);
            content = h.content;
            reasoning = h.reasoning;
        }
        if (native == null) content = try tool_lfm2.dsml(a, content, g.tools, true);
        const read = g.thinking or g.recipient != null; // the reply's split text is its content
        if (g.sink) |sink| {
            const thought = g.streamed_reasoning.text.items;
            if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
            const shown = if (g.calls != null) try tool_stream.endProse(a, content, if (read) content else text, g.tools, reason, g.single) else content;
            const sent = g.streamed.text.items;
            const rest = if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) shown[sent.len..] else "";
            if (g.lp) |*l| (if (try l.delta(g.srv, a, g.thinking, g.ignore_eos, g.collected.items, rest, shown.len, true)) |d| try g.emit(sink, d)) else if (rest.len > 0) try g.emit(sink, .{ .string = rest });
            if (g.calls) |*c| if (close.len > 0 or std.mem.eql(u8, reason, "length") or g.recipient != null) {
                // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                var out: std.ArrayList(Value) = .empty;
                try c.finishCut(if (read) content else text, &out); // a cut value sends its held bytes
                for (out.items) |d| try g.emit(sink, d);
            };
        }
        const prefilled = m.prefilled_ns;
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, decode_s) }); // the reply's decode seconds, as tokens_per_second divides them
        try @import("grammar_metrics.zig").put(a, runtime, m.stats.grammar);
        if (m.stats.gpu_seconds) |gpu| try runtime.put(a, "gpu_seconds", .{ .float = gpu }); // TF_Q27_PROFILE only
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
        const sha256 = @import("tokens.zig").tokenSha256(g.collected.items);
        try runtime.put(a, "token_sha256", .{ .string = try a.dupe(u8, &sha256) });
        // "return_token_ids": the reply's tokens, the same slice the shas above hash, so ids and sha cannot disagree
        if (g.token_ids) {
            const reply_ids = try a.alloc(Value, g.collected.items.len);
            for (g.collected.items, reply_ids) |t, *slot| slot.* = try json.intValue(a, t);
            try runtime.put(a, "token_ids", .{ .array = reply_ids });
        }
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            if (stopped != null) try loop_field.put(a, "stopped", .{ .bool = true });
            try runtime.put(a, "loop", .{ .object = loop_field });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        const counters = try a.dupe(u8, m.telemetry); // parsed strings point into it: it outlives the box
        if (counters.len > 0) switch (try json.parse(a, counters)) {
            .ok => |v| try spec.put(a, "proposer", v),
            .err => {},
        };
        const think_end: ?u32 = if (reasons) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = g.calls != null and g.calls.?.streamed,
            .native_tool_format = native != null,
            .native_calls = if (native) |n| n.calls else null,
            .finish_reason = reason,
            .prompt_tokens = prompt_len,
            .cached_tokens = cached_tokens.within(m.cached, prompt_len),
            .completion_tokens = g.collected.items.len,
            .reasoning_tokens = if (native) |n| n.reasoning_tokens else reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .thinking = thinking,
            .effort = effort,
            .logprobs = if (g.lp) |l| try @import("logprobs.zig").value(g.srv, cx, a, g.thinking, content_tokens, l.rows.items) else null,
            .prompt_logprobs = if (m.prompt_rows) |r| try @import("logprobs.zig").promptValue(g.srv, a, r) else null,
            .loop_stop = stopped,
        };
        if (std.mem.eql(u8, reason, "length") and reasons and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        const done_ns = nowNs(g.srv.io);
        const ttft_s = if (g.first_ns) |at| @max(0, seconds(at - received)) else 0;
        const decode_log_s = if (g.first_ns) |at| @max(0, seconds(done_ns - at)) else 0;
        var seed_text: [32]u8 = undefined;
        const seed = if (g.seed) |v| std.fmt.bufPrint(&seed_text, " seed={d}", .{v}) catch unreachable else "";
        const rate = if (decode_log_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_log_s else 0;
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d} ttft={d:.3} decode={d:.3} elapsed={d:.3} queued={d:.3} tok/s={d:.1}{s}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted, ttft_s, decode_log_s, @max(0, seconds(done_ns - received)), s.queued_seconds, rate, seed });
        return reply;
    }
};
