//! /v1/imprint: how much of a conversation the server resumes and from where, a disk read into RAM now, and forgetting.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const chat = @import("chat.zig");
const log = @import("log.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Ask = api.ImprintAsk;

/// An answer comes from the engine thread between rounds; a long prompt pass can hold it this long.
const wait_ms = 10 * 60 * 1000;

const Never = struct {
    pub fn check(_: Never) bool {
        return false;
    }
};

/// POST /v1/imprint/resume and /v1/imprint/load: a chat body rendered as its request would be, asked of the engine.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, op: api.ImprintOp) void {
    var cx: errors.Cx = .{ .a = a };
    const body = http_body.readJson(conn, &cx) catch return refuse(conn, a, &cx);
    const input = openai.chatInput(srv, &cx, body) catch return refuse(conn, a, &cx);
    const p = chat.prepare(srv, &cx, input, Never{}) catch return refuse(conn, a, &cx);
    defer chat.release(srv, p.preparing);
    const ask = make(srv, op, p.request) catch return conn.sendError(a, 500, "out of memory", null) catch {};
    answer(srv, conn, a, ask);
}

/// DELETE /v1/imprint: this model's conversation states on disk forgotten (RAM keeps what it holds).
pub fn forget(srv: *Server, conn: *Conn, a: Allocator) void {
    const ask = make(srv, .forget, .{ .prompt = &.{}, .max_tokens = 0 }) catch return conn.sendError(a, 500, "out of memory", null) catch {};
    answer(srv, conn, a, ask);
}

/// The question in the server's own memory: if the wait runs out, the engine may still write into it (it then leaks).
fn make(srv: *Server, op: api.ImprintOp, r: api.Request) !*Ask {
    const ask = try srv.gpa.create(Ask);
    ask.* = .{ .op = op };
    errdefer free(srv, ask);
    ask.prompt = try srv.gpa.dupe(u32, r.prompt);
    ask.starts = try srv.gpa.dupe(u32, r.chunks);
    ask.spans = try srv.gpa.dupe([2]u32, r.decode_spans);
    return ask;
}

fn free(srv: *Server, ask: *Ask) void {
    srv.gpa.free(ask.prompt);
    srv.gpa.free(ask.starts);
    srv.gpa.free(ask.spans);
    srv.gpa.destroy(ask);
}

fn answer(srv: *Server, conn: *Conn, a: Allocator, ask: *Ask) void {
    if (!srv.engine.imprintAsk(ask)) {
        free(srv, ask);
        return conn.sendError(a, 501, "this engine keeps no prompt states it can report on", null) catch {};
    }
    if (!ask.wait(srv.io, wait_ms)) {
        log.line("imprint: the engine did not answer in {d} s", .{wait_ms / 1000});
        return conn.sendError(a, 503, "the engine did not answer in time", null) catch {};
    }
    defer free(srv, ask);
    const out = reply(a, ask) catch return conn.sendError(a, 500, "out of memory", null) catch {};
    routes.sendValue(conn, a, 200, out);
}

/// The answer as JSON: where the next request would resume, and what a load or a forget did.
fn reply(a: Allocator, ask: *const Ask) !Value {
    const o = try json.newObject(a);
    const x = ask.out;
    try o.put(a, "object", .{ .string = switch (ask.op) {
        .where => "imprint.resume",
        .load => "imprint.load",
        .forget => "imprint.forget",
    } });
    try o.put(a, "disk", .{ .bool = ask.disk });
    if (ask.op == .forget) {
        try o.put(a, "freed_bytes", try json.intValue(a, ask.freed));
        return .{ .object = o };
    }
    try o.put(a, "tokens", try json.intValue(a, x.tokens));
    try o.put(a, "resume_from", .{ .string = @tagName(x.from) });
    try o.put(a, "ram_tokens", try json.intValue(a, x.ram));
    try o.put(a, "disk_tokens", try json.intValue(a, x.disk));
    try o.put(a, "disk_bytes", try json.intValue(a, x.disk_bytes));
    try o.put(a, "wait_seconds", .{ .float = x.seconds });
    if (ask.op == .load) {
        try o.put(a, "loaded_tokens", try json.intValue(a, ask.loaded));
        try o.put(a, "load_seconds", .{ .float = ask.load_seconds });
    }
    return .{ .object = o };
}

fn refuse(conn: *Conn, a: Allocator, cx: *const errors.Cx) void {
    const body = openai.errorBody(a, cx, "messages") catch return;
    routes.sendValue(conn, a, cx.status(), openai.wrapError(a, body) catch return);
}
