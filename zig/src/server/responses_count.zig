//! POST /v1/responses/input_tokens: the prompt POST /v1/responses would prefill for the same body, none generated.
const std = @import("std");
const json = @import("json");
const responses = @import("responses.zig");
const prepare = @import("openai_prepare.zig").prepare;
const compact = @import("compact.zig");
const cortheon = @import("cortheon/cortheon.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const Allocator = std.mem.Allocator;

/// The route's paths: the Responses collection's own `input_tokens`, with or without `/v1`.
pub fn route(path: []const u8) bool {
    return std.mem.eql(u8, path, "/v1/responses/input_tokens") or std.mem.eql(u8, path, "/responses/input_tokens");
}

/// The count, or the refusal POST /v1/responses would send for the same body before its run.
pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: Cx = .{ .a = a };
    const r = responses.request(srv, conn, a, &cx) orelse return;
    if (responses.unofferedLogprobs(srv, a, r.t) catch return) |payload| return routes.sendValue(conn, a, 400, payload);
    switch (prepare(srv, a, .{ .conn = conn }, r.t.chat, r.t.format, true, true)) {
        .cancelled => {},
        .refused => |x| {
            // a conversation past the window is compacted by the run, not refused
            if (x.kind == .context_length and compacts(srv)) return uncounted(conn, a, compaction);
            routes.sendValue(conn, a, x.status, x.payload);
        },
        .prepared => |p| {
            srv.release(p.preparing, p.request);
            if (compacts(srv) and p.input.messages == .array and compact.over(p.prompt_len, compact.wanted(srv, p.input), srv.info.context_window, srv.config.compact_at.?))
                return uncounted(conn, a, compaction);
            if (cortheon.takes(srv, a, r.t.chat, true, &p.input)) return uncounted(conn, a, "--cortheon runs it as a Cortheon turn, whose prompt adds its program, tools and research");
            const o = json.newObject(a) catch return;
            o.put(a, "object", .{ .string = "response.input_tokens" }) catch return;
            o.put(a, "input_tokens", json.intValue(a, p.prompt_len) catch return) catch return;
            routes.sendValue(conn, a, 200, .{ .object = o });
        },
    }
}

/// Whether --compact-at may rewrite a Responses conversation before its run, as compact.prepare decides it.
fn compacts(srv: *const Server) bool {
    return srv.config.compact_at != null and srv.info.context_window > 0;
}

const compaction = "--compact-at compacts its conversation first, with a note the model writes";

/// A prompt the run itself rewrites, with text the model or the web has not given yet, has no count before that run.
fn uncounted(conn: *Conn, a: Allocator, why: []const u8) void {
    const err = json.newObject(a) catch return;
    const message = std.fmt.allocPrint(a, "input_tokens cannot count this request: {s}", .{why}) catch return;
    err.put(a, "message", .{ .string = message }) catch return;
    err.put(a, "type", .{ .string = "invalid_request_error" }) catch return;
    err.put(a, "param", .null) catch return;
    err.put(a, "code", .null) catch return;
    const o = json.newObject(a) catch return;
    o.put(a, "error", .{ .object = err }) catch return;
    routes.sendValue(conn, a, 400, .{ .object = o });
}
