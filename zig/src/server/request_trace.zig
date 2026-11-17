//! One request's journal records: its start before any work, its resolved settings, then its reply in one finish.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const journal_mod = @import("request_journal.zig");
const Journal = journal_mod.Journal;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The reply bytes a finish keeps for replay; a longer reply keeps its size, SHA-256 and token hash.
pub const reply_limit: usize = 4 << 20;
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
        const j = t.journal orelse return;
        if (!t.started) return;
        t.end(j) catch {};
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
        if (t.whole) {
            const codec = std.base64.standard.Encoder;
            const encoded = try a.alloc(u8, codec.calcSize(t.kept.items.len));
            try o.put(a, "reply_base64", .{ .string = codec.encode(encoded, t.kept.items) });
        }
        j.append(a, .{ .object = o });
    }

    /// The quarantine refusal, a 400 a client should not retry, sent and journaled like any other reply.
    pub fn refuse(t: *Trace, id: []const u8) void {
        const cx: @import("errors.zig").Cx = .{ .a = t.a, .message = quarantine_message, .code = "request_quarantined" };
        var buf: [256]u8 = undefined;
        @import("log.zig").line("{s}", .{@import("log.zig").refused(&buf, id, cx.message)});
        const body = openai.errorBody(t.a, &cx, null) catch return;
        reply(t, cx.status(), openai.wrapError(t.a, body) catch return);
    }

    /// The parent itself when the journal is off, so a request runs exactly as it would without one.
    pub fn out(t: *Trace) openai.Out {
        if (t.journal == null) return t.parent;
        return .{ .ctx = t, .vt = &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive } };
    }

    fn keep(t: *Trace, bytes: []const u8) void {
        t.sha.update(bytes);
        t.size += bytes.len;
        if (!t.whole) return;
        if (t.kept.items.len + bytes.len > reply_limit) return t.drop();
        t.kept.appendSlice(t.a, bytes) catch t.drop();
    }

    fn keepJson(t: *Trace, value: Value) void {
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

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        if (payload) |p| {
            if (p.has("error")) t.failed = true;
            t.keep("data: ");
            t.keepJson(p);
            t.keep("\n\n");
        } else t.keep("data: [DONE]\n\n");
        try t.parent.vt.event(t.parent.ctx, payload);
        if (payload == null) t.done = true;
    }

    fn reply(ctx: *anyopaque, status: u16, value: Value) void {
        const t: *Trace = @ptrCast(@alignCast(ctx));
        t.status = status;
        if (status >= 500) t.failed = true;
        t.keepJson(value);
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
