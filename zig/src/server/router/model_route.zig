//! A model route: read the body, pick the child by its ``model``, start it if need be, and relay its reply.
const std = @import("std");
const json = @import("json");
const http_body = @import("../http_body.zig");
const log = @import("../log.zig");
const catalog = @import("catalog.zig");
const pool_mod = @import("pool.zig");
const relay = @import("relay.zig");
const replies = @import("replies.zig");
const front = @import("front.zig");
const Conn = @import("../http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// Seconds a refused client is told to wait while the router stops, as the server tells it while it drains.
const retry_after_s = 30;

pub fn serve(f: *front.Front, conn: *Conn, a: Allocator, route: front.ModelRoute) void {
    const shape = route.shape;
    const body = switch (http_body.read(conn, a, http_body.limit) catch return replies.send(conn, a, shape, .{ .status = 500, .kind = "server_error", .message = "out of memory" })) {
        .ok => |b| b,
        .refused => |message| return replies.send(conn, a, shape, .{ .status = 400, .message = message }),
    };
    const value = switch (json.parse(a, if (body.len == 0) "{}" else body) catch return replies.send(conn, a, shape, .{ .status = 500, .kind = "server_error", .message = "out of memory" })) {
        .ok => |v| v,
        .err => |message| return replies.send(conn, a, shape, .{ .status = 400, .message = message }),
    };
    const field = (if (value == .object) value.get("model") else null) orelse
        return replies.send(conn, a, shape, .{ .status = 400, .message = "model is required: the router picks a model by the request's model field; GET /v1/models lists them", .param = "model", .code = "model_required" });
    if (field != .string) return replies.send(conn, a, shape, .{ .status = 400, .message = "model must be a string", .param = "model" });
    const i = catalog.find(f.entries, field.string) orelse return replies.send(conn, a, shape, replies.unknownModel(a, field.string));
    // the child's own key has every scope: a restricted key's Responses request must be stateless, as on a server
    if (route.responses) if (@import("../auth_scope.zig").responseDenial(conn.key_scopes, value)) |why| return @import("../auth_scope_reply.zig").deny(conn, false, why);
    if (f.draining.load(.acquire)) return replies.send(conn, a, shape, refusal(a, f, i, .stopping));
    const lease = switch (f.pool.acquire(a, i)) {
        .lease => |l| l,
        .refused => |r| return replies.send(conn, a, shape, refusal(a, f, i, r)),
    };
    defer f.pool.release(lease);
    const id = f.entries[i].id;
    const result = relay.forward(conn, a, lease.port, &lease.key, body, f.limits);
    switch (result.outcome) {
        .done => replies.relayed(conn, result.status, id),
        .no_reply => {
            const exited = f.pool.lost(lease);
            const message = if (exited)
                std.fmt.allocPrint(a, "the server for model '{s}' exited while answering; the next request starts it again", .{id}) catch "the model's server exited"
            else
                std.fmt.allocPrint(a, "the server for model '{s}' closed the connection without a reply", .{id}) catch "the model's server sent no reply";
            replies.send(conn, a, shape, .{ .status = 502, .kind = "server_error", .message = message, .code = "model_server_failed" });
        },
        .cut => {
            conn.close = true;
            const exited = f.pool.lost(lease);
            log.line("{s} \"{s}\" {d} - model={s}: reply cut short ({s})", .{ conn.peer, conn.requestline, result.status, id, if (exited) "its server exited" else "its server stopped sending" });
        },
        .client_gone => conn.close = true,
    }
}

/// The 503 a request gets when the pool gives it no child.
fn refusal(a: Allocator, f: *const front.Front, i: usize, r: pool_mod.Refusal) replies.Refusal {
    const id = f.entries[i].id;
    const gib = 1 << 30;
    return switch (r) {
        .stopping => .{ .status = 503, .kind = "server_error", .message = "server restarting: retry shortly", .code = "server_restarting", .retry_after_s = retry_after_s },
        .busy => .{
            .status = 503,
            .kind = "server_error",
            .message = std.fmt.allocPrint(a, "every loaded model is answering a request, and --max-loaded {d} leaves no room to load '{s}'; retry shortly", .{ f.pool.options.max_loaded, id }) catch "every loaded model is busy",
            .code = "models_busy",
            .retry_after_s = 10,
        },
        .too_large => |t| .{
            .status = 503,
            .kind = "server_error",
            .message = std.fmt.allocPrint(a, "model '{s}' needs about {d:.1} GiB loaded and this machine has {d:.1} GiB, so it cannot load here", .{ id, @as(f64, @floatFromInt(t.need)) / gib, @as(f64, @floatFromInt(t.have)) / gib }) catch "the model cannot fit in this machine's memory",
            .code = "model_too_large",
        },
        .failed => |why| .{ .status = 503, .kind = "server_error", .message = std.fmt.allocPrint(a, "model '{s}' could not start: {s}", .{ id, why }) catch "the model could not start", .code = "model_load_failed" },
        .timeout => |s| .{ .status = 503, .kind = "server_error", .message = std.fmt.allocPrint(a, "model '{s}' was not ready within {d} s", .{ id, s }) catch "the model was not ready in time", .code = "model_load_timeout" },
    };
}
