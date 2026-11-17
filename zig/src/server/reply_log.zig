//! Builds and writes a request's reply line after its last byte reaches the client.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const errors = @import("errors.zig");
const request_log = @import("request_log.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Reply-line state, written by runLogged's defer after the last byte is sent (streams end at [DONE]).
pub const ReplyLog = struct {
    a: Allocator,
    srv: *Server,
    enabled: bool,
    id: []const u8,
    route: []const u8,
    model: []const u8 = "",
    status: i64 = 0, // 0: the client left before a reply was delivered
    text: ?[]const u8 = null,
    reasoning: ?[]const u8 = null,
    calls: ?[]Value = null,
    finish_reason: ?[]const u8 = null,
    error_msg: ?[]const u8 = null,
    usage: ?struct { prompt: usize, cached: usize, completion: usize } = null,
    queued: ?f64 = null,
    ttft: ?f64 = null,
    decode_tps: ?f64 = null,
    total: ?f64 = null,

    /// A finished reply: its text, calls, usage and engine timings, captured once generation ends.
    pub fn done(s: *ReplyLog, reply: *const chat.Reply, calls: ?[]Value) void {
        s.status = 200;
        s.text = reply.content;
        s.reasoning = reply.reasoning;
        s.calls = calls;
        s.finish_reason = reply.finish_reason;
        s.usage = .{ .prompt = reply.prompt_tokens, .cached = reply.cached_tokens, .completion = reply.completion_tokens };
        s.queued = reply.queued_seconds;
        s.ttft = reply.ttft_seconds;
        s.decode_tps = reply.decode_tps;
        s.total = reply.total_seconds;
    }

    /// A refusal or failure before a complete reply: its status and message, with no usage.
    pub fn fail(s: *ReplyLog, status: u16, message: ?[]const u8) void {
        s.status = status;
        s.error_msg = message;
        s.finish_reason = "error";
    }

    pub fn refused(s: *ReplyLog, cx: *const Cx) void {
        s.fail(cx.status(), cx.message);
    }

    pub fn cancelled(s: *ReplyLog) void {
        if (s.status == 0) s.finish_reason = "cancelled";
    }

    pub fn write(s: *ReplyLog) void {
        if (!s.enabled) return;
        const path = s.srv.config.request_log orelse return;
        var line: request_log.Reply = .{
            .id = s.id,
            .route = s.route,
            .model = s.model,
            .status = s.status,
            .text = s.text,
            .reasoning = s.reasoning,
            .tool_calls = s.calls,
            .finish_reason = if (s.finish_reason) |f| f else if (s.status == 0) "cancelled" else null,
            .err = s.error_msg,
            .queued = s.queued,
            .ttft = s.ttft,
            .decode_tps = s.decode_tps,
            .total = s.total,
        };
        if (s.usage) |u| {
            line.prompt_tokens = u.prompt;
            line.cached_tokens = u.cached;
            line.completion_tokens = u.completion;
            line.has_usage = true;
        }
        request_log.appendReply(s.a, path, line);
    }
};
