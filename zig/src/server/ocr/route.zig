//! POST /v1/ocr: the body read once, then Mistral's handler (docs/api.md OCR).
const Allocator = @import("std").mem.Allocator;
const http_body = @import("../http_body.zig");
const openai = @import("../openai.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const mistral = @import("mistral.zig");
const mistral_reply = @import("mistral_reply.zig");

/// The handler for POST /v1/ocr.
pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    const body = switch (http_body.read(conn, a, http_body.limit) catch {
        conn.close = true;
        return mistral_reply.sendFault(conn, a, .{ .class = .internal, .message = "out of memory" });
    }) {
        .ok => |b| b,
        .refused => |message| return mistral.refuseBody(conn, a, message),
    };
    const gone: openai.Gone = .{ .conn = conn };
    mistral.post(srv, conn, a, body, gone);
}
