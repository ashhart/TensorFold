//! Embedding requests use the model worker and keep caller-owned vectors alive through terminal completion.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const request = @import("embedding_request.zig");
const response = @import("embedding_response.zig");
const openai = @import("openai.zig");
const http_body = @import("http_body.zig");
const Conn = @import("http_conn.zig").Conn;
const Server = @import("server.zig").Server;
pub const Failure = errors.Refused || error{Cancelled};

const Completion = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    reason: ?api.Reason = null,
    unexpected_tokens: bool = false,

    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const c: *Completion = @ptrCast(@alignCast(ptr));
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        switch (value.*) {
            .tokens => |tokens| c.unexpected_tokens = c.unexpected_tokens or tokens.len != 0,
            .finished => |f| c.reason = f.reason,
            else => {},
        }
    }
};

pub fn reply(srv: *Server, cx: *errors.Cx, body: json.Value, gone: anytype) Failure!json.Value {
    const cap = srv.info.embeddings orelse return cx.refuse("this model does not produce embeddings");
    if (srv.info.text_generation) return cx.fail(.server, "invalid embedding model capabilities", .{});
    if (body.field("model")) |model| {
        if (model != .string) return cx.refuse("model must be a string");
        var found = false;
        for (srv.config.model_ids) |id| found = found or std.mem.eql(u8, id, model.string);
        if (!found) return cx.refuse("model is not served here");
    }
    const parsed = try request.parse(cx, srv.text, cap, srv.info.context_window, body);
    const count = std.math.mul(usize, parsed.ends.len, cap.dimensions) catch return cx.refuse("embedding output size overflows");
    const output = try cx.a.alloc(f32, count);
    @memset(output, std.math.nan(f32));
    const ask: api.Request = .{ .prompt = parsed.tokens, .max_tokens = 0, .drafts = false, .untaught = true, .embedding = .{ .ends = parsed.ends, .dimensions = cap.dimensions, .output = output } };
    var done: Completion = .{ .io = srv.io };
    const id = srv.next_id.fetchAdd(1, .monotonic);
    if (gone.check()) return error.Cancelled;
    srv.engine.submit(id, &ask, .{ .ctx = &done, .event = Completion.event }) catch |err| return switch (err) {
        error.InvalidRequest, error.InvalidSpans => cx.refuse("the embedding request does not match the model"),
        error.Busy, error.Closed, error.Stale => cx.fail(.capacity, "the embedding engine is unavailable", .{}),
        error.MemoryPressure => cx.fail(.capacity, "{s}", .{api.pressure_message}),
    };
    var cancelling = false;
    while (true) {
        done.mutex.lockUncancelable(srv.io);
        const reason = done.reason;
        const bad = done.unexpected_tokens;
        done.mutex.unlock(srv.io);
        if (reason) |why| {
            if (cancelling or why == .cancelled) return error.Cancelled;
            if (why != .length or bad) return cx.fail(.server, "embedding inference failed", .{});
            break;
        }
        if (!cancelling and gone.check()) {
            srv.engine.cancel(id);
            cancelling = true;
        }
        std.Io.sleep(srv.io, .fromMilliseconds(1), .awake) catch {};
    }
    return response.encode(cx, srv.config.served_name, parsed, output);
}

pub fn post(srv: *Server, conn: *Conn, a: std.mem.Allocator) void {
    var cx: errors.Cx = .{ .a = a };
    const raw = http_body.readJson(conn, &cx) catch |err| {
        if (err == error.OutOfMemory) {
            cx.kind = .server;
            cx.message = "out of memory";
        }
        return refused(conn, &cx);
    };
    const Gone = struct {
        conn: *Conn,
        fn check(g: @This()) bool {
            return g.conn.peerGone();
        }
    };
    const value = reply(srv, &cx, raw, Gone{ .conn = conn }) catch |err| {
        if (err == error.Cancelled) {
            conn.close = true;
            return;
        }
        if (err == error.OutOfMemory) {
            cx.kind = .server;
            cx.message = "out of memory";
        }
        return refused(conn, &cx);
    };
    @import("routes.zig").sendValue(conn, a, 200, value);
}

fn refused(conn: *Conn, cx: *errors.Cx) void {
    const body = openai.errorBody(cx.a, cx, null) catch return;
    const wrapped = openai.wrapError(cx.a, body) catch return;
    @import("routes.zig").sendValue(conn, cx.a, cx.status(), wrapped);
}
