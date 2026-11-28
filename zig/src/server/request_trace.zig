//! One request's journal records: its start before any work, its resolved settings, then its reply in one finish.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const journal_mod = @import("request_journal.zig");
const request_log = @import("request_log.zig");
const Journal = journal_mod.Journal;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The reply bytes a finish keeps for replay; a longer reply keeps its size, SHA-256 and token hash.
pub const reply_limit: usize = 4 << 20;
/// The reply text and reasoning one reply record keeps together; a longer reply is recorded truncated.
pub const text_limit: usize = 256 << 10;
pub const quarantine_message = "this request was in flight when the server crashed and is quarantined; its operator can clear it";

pub const Trace = struct {
    journal: ?*Journal,
    a: Allocator,
    id: []const u8,
    parent: openai.Out,
    status: u16 = 0,
    stream: bool = false,
    done: bool = false,
    failed: bool = false,
    started: bool = false,
    quarantined: bool = false,
    whole: bool = true, // every reply byte is kept
    size: usize = 0,
    sha: Sha256 = .init(.{}),
    kept: std.ArrayList(u8) = .empty,
    token_sha256: ?[]const u8 = null,
    timing: ?Timing = null,
    /// TENSORFOLD_REQUEST_LOG_REPLIES: where this request's one reply record is appended; null: nowhere.
    reply_log: ?[]const u8 = null,
    text: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    have_reasoning: bool = false,
    finish_reason: ?[]const u8 = null,
    cut: bool = false,
    error_message: ?[]const u8 = null,
    prompt_tokens: ?i64 = null,
    cached_tokens: ?i64 = null,
    completion_tokens: ?i64 = null,

    /// The finish's measures: several generations sum their counts and keep the last one's timings.
    pub const Timing = struct { queue_ms: f64, ttft_ms: ?f64, decode_tps: f64, prompt: usize, cached: usize, completion: usize };

    /// A generation's measures, as the done log line reports them; nothing is measured here.
    pub fn timed(t: *Trace, r: anytype) void {
        if (t.journal == null and t.reply_log == null) return;
        const before: Timing = t.timing orelse .{ .queue_ms = 0, .ttft_ms = null, .decode_tps = 0, .prompt = 0, .cached = 0, .completion = 0 };
        t.timing = .{
            .queue_ms = r.queued_s * 1000,
            .ttft_ms = if (r.ttft_s) |s| s * 1000 else null,
            .decode_tps = r.decode_tps,
            .prompt = before.prompt + r.prompt_tokens,
            .cached = before.cached + r.cached_tokens,
            .completion = before.completion + r.completion_tokens,
        };
    }

    /// Writes the start before any work; true when this key's same request is quarantined, which the caller refuses.
    pub fn begin(t: *Trace, body: Value, chat: bool, key: []const u8) bool {
        const j = t.journal orelse return false;
        const fp = journal_mod.fingerprint(key, chat, body);
        t.started = true;
        t.quarantined = j.quarantined(fp);
        t.start(j, body, chat, fp) catch {};
        return t.quarantined;
    }

    fn start(t: *Trace, j: *Journal, body: Value, chat: bool, fp: journal_mod.Fingerprint) Allocator.Error!void {
        const a = t.a;
        const o = try j.record(a, "start", t.id);
        try o.put(a, "path", .{ .string = if (chat) "/v1/chat/completions" else "/v1/completions" });
        const hex = std.fmt.bytesToHex(fp, .lower);
        try o.put(a, "fingerprint", .{ .string = try a.dupe(u8, &hex) });
        try o.put(a, "seed", body.get("seed") orelse .null);
        const redacted = try @import("request_log.zig").redact(a, body);
        var buf: [1024]u8 = undefined;
        var counted: std.Io.Writer.Discarding = .init(&buf);
        json.write(&counted.writer, redacted, .{ .compact = true }) catch {};
        // a body too large for one record keeps its size; replay then needs --body
        if (counted.fullCount() + 4096 <= journal_mod.record_limit) try o.put(a, "body", redacted) else try o.put(a, "body_bytes", try json.intValue(a, counted.fullCount()));
        j.append(a, .{ .object = o });
    }

    /// The settings the server resolved before the engine saw the request; its seed is what a replay needs.
    pub fn prepared(t: *Trace, request: *const api.Request, thinking: bool, effort: ?[]const u8) void {
        const j = t.journal orelse return;
        t.settings(j, request, thinking, effort) catch {};
    }

    fn settings(t: *Trace, j: *Journal, request: *const api.Request, thinking: bool, effort: ?[]const u8) Allocator.Error!void {
        const a = t.a;
        const o = try j.record(a, "prepared", t.id);
        if (request.sampling) |s| {
            try o.put(a, "seed", try json.intValue(a, s.seed));
            try o.put(a, "temperature", .{ .float = s.temperature });
            try o.put(a, "top_k", try json.intValue(a, s.top_k));
            try o.put(a, "top_p", .{ .float = s.top_p });
            try o.put(a, "min_p", .{ .float = s.min_p });
        } else try o.put(a, "seed", .null);
        try o.put(a, "max_tokens", try json.intValue(a, request.max_tokens));
        try o.put(a, "drafts", .{ .bool = request.drafts });
        try o.put(a, "thinking", .{ .bool = thinking });
        try o.put(a, "reasoning_effort", if (effort) |s| .{ .string = s } else .null);
        try o.put(a, "prompt_tokens", try json.intValue(a, request.prompt.len));
        const sha = @import("tokens.zig").tokenSha256(request.prompt);
        try o.put(a, "prompt_sha256", .{ .string = try a.dupe(u8, &sha) });
        j.append(a, .{ .object = o });
    }

    /// The finish: status, outcome and the reply as the chat route produced it, written once the reply has gone.
    pub fn finish(t: *Trace) void {
        if (t.journal) |j| if (t.started) t.end(j) catch {};
        if (t.reply_log) |path| if (t.status != 0) t.writeReply(path);
    }

    /// The one reply record this request appends; a refusal keeps its message only.
    fn writeReply(t: *Trace, path: []const u8) void {
        const m = t.timing;
        const prompt: ?i64 = if (m) |x| @as(i64, @intCast(x.prompt)) else t.prompt_tokens;
        const cached: ?i64 = if (m) |x| @as(i64, @intCast(x.cached)) else t.cached_tokens;
        const completion: ?i64 = if (m) |x| @as(i64, @intCast(x.completion)) else t.completion_tokens;
        @import("request_log.zig").appendReply(t.a, path, .{
            .id = t.id,
            .status = t.status,
            .stream = t.stream,
            .finish_reason = t.finish_reason,
            .text = t.text.items,
            .reasoning = if (t.have_reasoning) t.reasoning.items else null,
            .prompt_tokens = prompt,
            .cached_tokens = cached,
            .completion_tokens = completion,
            .queue_ms = if (m) |x| x.queue_ms else null,
            .ttft_ms = if (m) |x| x.ttft_ms else null,
            .decode_tps = if (m) |x| x.decode_tps else null,
            .truncated = t.cut,
            .error_message = t.error_message,
        }, @import("request_log.zig").file_limit);
    }

    fn end(t: *Trace, j: *Journal) Allocator.Error!void {
        const a = t.a;
        const o = try j.record(a, "finish", t.id);
        try o.put(a, "status", try json.intValue(a, t.status));
        try o.put(a, "stream", .{ .bool = t.stream });
        const outcome = if (t.quarantined) "quarantined" else if (t.failed) "failed" else if (!t.done) "cancelled" else if (t.status >= 400) "refused" else "completed";
        try o.put(a, "outcome", .{ .string = outcome });
        try o.put(a, "reply_bytes", try json.intValue(a, t.size));
        const hex = std.fmt.bytesToHex(t.sha.finalResult(), .lower);
        try o.put(a, "reply_sha256", .{ .string = try a.dupe(u8, &hex) });
        try o.put(a, "token_sha256", if (t.token_sha256) |s| .{ .string = s } else .null);
        try putTiming(a, o, t.timing);
        if (t.whole) {
            const codec = std.base64.standard.Encoder;
            const encoded = try a.alloc(u8, codec.calcSize(t.kept.items.len));
            try o.put(a, "reply_base64", .{ .string = codec.encode(encoded, t.kept.items) });
        }
        j.append(a, .{ .object = o });
    }

    fn putTiming(a: Allocator, o: *json.Object, timing: ?Timing) Allocator.Error!void {
        const m = timing orelse {
            for ([_][]const u8{ "queue_ms", "ttft_ms", "decode_tps", "prompt_tokens", "cached_tokens", "completion_tokens" }) |k| try o.put(a, k, .null);
            return;
        };
        try o.put(a, "queue_ms", .{ .float = round1(m.queue_ms) });
        try o.put(a, "ttft_ms", if (m.ttft_ms) |v| .{ .float = round1(v) } else .null);
        try o.put(a, "decode_tps", .{ .float = round1(m.decode_tps) });
        try o.put(a, "prompt_tokens", try json.intValue(a, m.prompt));
        try o.put(a, "cached_tokens", try json.intValue(a, m.cached));
        try o.put(a, "completion_tokens", try json.intValue(a, m.completion));
    }

    fn round1(v: f64) f64 {
        return @round(v * 10) / 10;
    }

    /// The quarantine refusal, a 400 a client should not retry, sent and journaled like any other reply.
    pub fn refuse(t: *Trace, id: []const u8) void {
        const cx: @import("errors.zig").Cx = .{ .a = t.a, .message = quarantine_message, .code = "request_quarantined" };
        var buf: [256]u8 = undefined;
        @import("log.zig").line("{s}", .{@import("log.zig").refused(&buf, id, cx.message)});
        const body = openai.errorBody(t.a, &cx, null) catch return;
        reply(t, cx.status(), openai.wrapError(t.a, body) catch return);
    }

    /// The parent itself when neither the journal nor the reply log is on, so a request runs as it would without one.
    pub fn out(t: *Trace) openai.Out {
        if (t.journal == null and t.reply_log == null) return t.parent;
        // Keep each vtable comptime: a runtime field in a returned anonymous table would dangle.
        const vt: *const openai.Out.VTable = if (t.parent.vt.progress != null)
            &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive, .progress = progress }
        else
            &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive };
        return .{ .ctx = t, .vt = vt };
    }

    fn keep(t: *Trace, bytes: []const u8) void {
        if (t.journal == null) return;
        t.sha.update(bytes);
        t.size += bytes.len;
        if (!t.whole) return;
        if (t.kept.items.len + bytes.len > reply_limit) return t.drop();
        t.kept.appendSlice(t.a, bytes) catch t.drop();
    }

    fn keepJson(t: *Trace, value: Value) void {
        if (t.journal == null) return;
        var scratch: std.heap.ArenaAllocator = .init(t.journal.?.gpa);
        defer scratch.deinit();
        t.keep(json.stringify(scratch.allocator(), value, .{}) catch return t.drop());
        if (value.get("tensorfold")) |runtime| if (runtime.strField("token_sha256")) |s| {
            t.token_sha256 = t.a.dupe(u8, s) catch null;
        };
    }

    fn drop(t: *Trace) void {
        t.whole = false;
        t.kept.clearAndFree(t.a);
    }

    fn open(ctx: *anyopaque, n: usize) error{Closed}!void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        t.stream = true;
        t.status = 200;
        return t.parent.vt.open(t.parent.ctx, n);
    }

    fn progress(ctx: *anyopaque, p: api.Progress) error{Closed}!void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        return t.parent.vt.progress.?(t.parent.ctx, p);
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        if (payload) |p| {
            if (p.has("error")) t.failed = true;
            t.keep("data: ");
            t.keepJson(p);
            t.keep("\n\n");
            t.capture(p, "delta");
        } else t.keep("data: [DONE]\n\n");
        try t.parent.vt.event(t.parent.ctx, payload);
        if (payload == null) t.done = true;
    }

    /// What a reply payload carries for the reply record: the first choice's text, reasoning and finish reason.
    fn capture(t: *Trace, p: Value, part: []const u8) void {
        if (t.reply_log == null) return;
        if (p.get("error")) |e| if (e.get("message")) |m| {
            if (m == .string) {
                const kept = request_log.cutTo(m.string, request_log.error_limit);
                if (t.a.dupe(u8, kept) catch null) |s| t.error_message = s;
            }
        };
        if (p.get("choices")) |choices| if (choices == .array) for (choices.array) |choice| {
            // one record per request: the first reply's text, never two choices interleaved
            if (choice.get("index")) |i| if (i.int64() orelse 0 != 0) continue;
            if (choice.get(part)) |body| {
                if (body.strField("content")) |s| t.add(&t.text, s);
                if (body.strField("reasoning_content") orelse body.strField("reasoning")) |s| {
                    t.have_reasoning = true;
                    t.add(&t.reasoning, s);
                }
            }
            // a legacy completion's text is its choice's, not a delta's or a message's
            if (choice.strField("text")) |s| t.add(&t.text, s);
            if (choice.strField("finish_reason")) |f| {
                if (t.a.dupe(u8, f) catch null) |s| t.finish_reason = s;
            }
        };
        if (p.get("usage")) |u| t.usage(u);
    }

    /// Appends one piece while text and reasoning are under ``text_limit`` together; past it, the record says cut.
    fn add(t: *Trace, list: *std.ArrayList(u8), bytes: []const u8) void {
        if (t.cut or bytes.len == 0) return;
        if (t.text.items.len + t.reasoning.items.len + bytes.len > text_limit) {
            const used = t.text.items.len + t.reasoning.items.len;
            list.appendSlice(t.a, request_log.cutTo(bytes, text_limit - used)) catch {};
            t.cut = true;
            return;
        }
        list.appendSlice(t.a, bytes) catch {
            t.cut = true;
        };
    }

    /// The usage a payload reports, kept for when no generation timed this request.
    fn usage(t: *Trace, u: Value) void {
        if (u.get("prompt_tokens")) |n| if (n.int64()) |v| {
            t.prompt_tokens = v;
        };
        const cached = if (u.get("prompt_tokens_details")) |d| d.get("cached_tokens") else u.get("cached_tokens");
        if (cached) |n| if (n.int64()) |v| {
            t.cached_tokens = v;
        };
        if (u.get("completion_tokens")) |n| if (n.int64()) |v| {
            t.completion_tokens = v;
        };
    }

    fn reply(ctx: *anyopaque, status: u16, value: Value) void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        t.status = status;
        if (status >= 500) t.failed = true;
        t.keepJson(value);
        t.capture(value, "message");
        t.parent.vt.reply(t.parent.ctx, status, value);
        t.done = true;
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        const send = t.parent.vt.keepalive orelse return;
        t.keep(": keepalive\n\n");
        return send(t.parent.ctx);
    }
};
