//! A chat body checked and rendered as its run would be, never submitted: a Responses warmup's prompt and refusals.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const chat = @import("chat.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const PrepareResult = union(enum) {
    /// Rendered and checked: the caller reads ``prompt_len`` and must hand it to ``srv.release``.
    prepared: chat.Prepared,
    /// The whole error reply ``runNamed`` would send before its run.
    refused: struct { status: u16, payload: Value },
    /// The client left while it was prepared.
    cancelled,
};

/// ``plan`` then ``chat.prepare``: images, template fields and the context window, checked as a real turn checks them.
pub fn prepare(srv: *Server, a: Allocator, gone: openai.Gone, raw: Value, format: ?[]const u8, client_thinking: bool) PrepareResult {
    const id = ids.make(a, "chatcmpl-", 32) catch return outOfMemory(a);
    var cx: Cx = .{ .a = a };
    var p = openai.plan(srv, &cx, true, raw, format, client_thinking) catch |e| {
        if (e == error.OutOfMemory) cx.message = "out of memory";
        return refusal(a, id, &cx, cx.status());
    };
    p.input.id = id;
    const prepared = chat.prepare(srv, &cx, p.input, gone) catch |e| switch (e) {
        error.Cancelled => return .cancelled,
        error.OutOfMemory => return outOfMemory(a),
        error.Failed => return refusal(a, id, &.{ .a = a, .kind = .server, .message = "the reply failed" }, 500),
        error.Refused => {
            if (cx.kind == .other) cx.kind = .server;
            return refusal(a, id, &cx, if (cx.kind == .server) 500 else cx.status());
        },
    };
    return .{ .prepared = prepared };
}

fn outOfMemory(a: Allocator) PrepareResult {
    return refusal(a, "", &.{ .a = a, .kind = .server, .message = "out of memory" }, 500);
}

/// The chat route's error reply for ``cx``, logged as its refusal when it is the client's.
fn refusal(a: Allocator, id: []const u8, cx: *const Cx, status: u16) PrepareResult {
    if (cx.kind != .server and id.len > 0) {
        var buf: [1400]u8 = undefined;
        log.line("{s}", .{log.refused(&buf, id, cx.message)});
    }
    const body = openai.errorBody(a, cx, "messages") catch return .{ .refused = .{ .status = 500, .payload = .null } };
    return .{ .refused = .{ .status = status, .payload = openai.wrapError(a, body) catch .null } };
}
