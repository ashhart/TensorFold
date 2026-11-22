//! A POST /v1/responses body as both of its routes read it: held to its key's scope and translated, or refused.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const responses = @import("responses.zig");
const translate_mod = @import("responses_translate.zig");
const store_mod = @import("responses_store.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Request = struct { body: Value, t: translate_mod.Request };

/// A POST body read, held to its key's scope and translated as the chat request it runs as; null once refused.
pub fn request(srv: *Server, conn: *Conn, a: Allocator, cx: *Cx) ?Request {
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return null) {
        .ok => |b| b,
        .refused => |m| return untranslated(conn, a, m),
    };
    const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return null) {
        .ok => |v| v,
        .err => return untranslated(conn, a, "the request body is not JSON"),
    };
    // a restricted key is stateless here: checked before any stored conversation is read
    if (@import("auth_scope.zig").responseDenial(conn.key_scopes, body)) |why| {
        @import("auth_scope_reply.zig").deny(conn, false, why);
        return null;
    }
    if (body.get("previous_response_id")) |p| if (p == .string and srv.background_jobs.running(srv.io, p.string, conn.key_digest)) {
        responses.refuse(conn, a, "previous_response_id names a background response still in progress: poll it until it ends", 400, "previous_response_id");
        return null;
    };
    const history: store_mod.History = .{ .store = &srv.store, .io = srv.io, .owner = conn.key_digest };
    const t = translate_mod.translate(cx, body, history) catch return untranslated(conn, a, cx.message);
    return .{ .body = body, .t = t };
}

/// The 400 for a body that never became a chat request.
fn untranslated(conn: *Conn, a: Allocator, message: []const u8) ?Request {
    responses.refuse(conn, a, message, 400, null);
    return null;
}

/// Logprobs on a model without them: rows only an include asked for are dropped, an explicit ask is refused.
pub fn unofferedLogprobs(srv: *Server, a: Allocator, t: translate_mod.Request) !?Value {
    if (@import("logprobs.zig").offered(srv, a) catch false) return null;
    if (t.rows_if_offered) {
        _ = t.chat.object.orderedRemove("logprobs");
        return null;
    }
    if (!json.truthyField(t.chat, "logprobs")) return null;
    return try responses.refusal(a, "logprobs are not supported by this model or backend", null);
}
