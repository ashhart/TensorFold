//! Durable image job submission, inspection, cancellation, deletion and immutable content routes.
const std = @import("std");
const media = @import("media_job.zig");
const store = @import("media_job_store.zig");
const images = @import("images_generations.zig");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const Allocator = std.mem.Allocator;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;

const prefix = "/v1/images/jobs/";
pub const collection = "/v1/images/jobs";
const ContentPath = struct { id: []const u8, index: usize };

pub fn itemPath(path: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, path, prefix)) return null;
    const id = path[prefix.len..];
    if (!store.validId(id)) return null;
    return id;
}

pub fn cancelPath(path: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, path, prefix) or !std.mem.endsWith(u8, path, "/cancel")) return null;
    const id = path[prefix.len .. path.len - "/cancel".len];
    if (!store.validId(id)) return null;
    return id;
}

pub fn contentPath(path: []const u8) ?ContentPath {
    if (!std.mem.startsWith(u8, path, prefix)) return null;
    const rest = path[prefix.len..];
    const marker = "/content/";
    const at = std.mem.indexOf(u8, rest, marker) orelse return null;
    const id = rest[0..at];
    if (!store.validId(id)) return null;
    const text = rest[at + marker.len ..];
    if (text.len == 0) return null;
    const index = std.fmt.parseInt(usize, text, 10) catch return null;
    return .{ .id = id, .index = index };
}

fn sendError(conn: *Conn, a: Allocator, status: u16, message: []const u8) void {
    const quoted = json.quote(a, message, .{}) catch return;
    const body = std.fmt.allocPrint(a, "{{\"error\":{{\"message\":{s}}}}}", .{quoted}) catch return;
    conn.sendJson(status, body[0..body.len]);
}

fn enabled(srv: *Server) bool {
    return srv.media_jobs.enabled and srv.info.image != null;
}

fn unavailable(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.media_jobs.enabled) return routes.unknown(conn, a);
    sendError(conn, a, 400, "durable image jobs require an image model");
}

fn submissionProblem(conn: *Conn, a: Allocator, e: anyerror) void {
    const status: u16 = switch (e) {
        error.JobLimit, error.OwnerLimit, error.RequestByteLimit => 429,
        error.InvalidRequest, error.RequestTooLarge => 400,
        error.Closed => 503,
        else => 500,
    };
    const message = switch (e) {
        error.JobLimit => "the durable media job quota is full",
        error.OwnerLimit => "this principal's durable media job quota is full",
        error.RequestByteLimit => "the durable media request byte quota is full",
        error.RequestTooLarge => "the durable media request is too large",
        error.InvalidRequest => "the durable media request is empty",
        error.Closed => "the durable media service is shutting down",
        else => "the durable media job could not be persisted",
    };
    sendError(conn, a, status, message);
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator, id: ?[]const u8) void {
    if (!enabled(srv)) {
        if (id == null) _ = http_body.read(conn, a, http_body.limit) catch {};
        return unavailable(srv, conn, a);
    }
    if (id) |job_id| {
        const status = srv.media_jobs.registry.cancel(job_id, conn.key_digest) catch |e| return sendError(conn, a, 500, @errorName(e));
        if (status == null) return sendError(conn, a, 404, "media job not found");
        return inspect(srv, conn, a, job_id);
    }
    const raw = switch (http_body.read(conn, a, (media.Limits{}).max_request_bytes) catch return sendError(conn, a, 500, "out of memory")) {
        .ok => |bytes| bytes,
        .refused => |message| return sendError(conn, a, 400, message),
    };
    const parsed = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return sendError(conn, a, 500, "out of memory")) {
        .ok => |value| value,
        .err => |message| return sendError(conn, a, 400, message),
    };
    var cx: errors.Cx = .{ .a = a };
    const plan = images.prepare(srv, &cx, srv.info.image.?, parsed) catch {
        sendError(conn, a, cx.status(), cx.message);
        return;
    };
    if (plan.stream) return sendError(conn, a, 400, "durable image jobs do not accept stream or preview fields");
    const submission = srv.media_jobs.registry.submit(a, conn.key_digest, .image, raw) catch |e| return submissionProblem(conn, a, e);
    _ = submission.serial;
    inspectStatus(srv, conn, a, submission.id, 202);
}

fn snapshotValue(a: Allocator, snapshot: *const media.Snapshot) !json.Value {
    const out = try json.newObject(a);
    try out.put(a, "id", .{ .string = snapshot.id });
    try out.put(a, "kind", .{ .string = @tagName(snapshot.kind) });
    try out.put(a, "created", try json.intValue(a, snapshot.created));
    try out.put(a, "status", .{ .string = @tagName(snapshot.state) });
    try out.put(a, "cancel_requested", .{ .bool = snapshot.cancel_requested });
    if (snapshot.message) |message| try out.put(a, "error", .{ .string = message });
    const artifacts = try a.alloc(json.Value, snapshot.artifacts.len);
    for (snapshot.artifacts, artifacts) |artifact, *value| {
        const item = try json.newObject(a);
        try item.put(a, "index", try json.intValue(a, artifact.index));
        try item.put(a, "media_type", .{ .string = artifact.media_type });
        try item.put(a, "bytes", try json.intValue(a, artifact.bytes));
        const digest = std.fmt.bytesToHex(artifact.digest, .lower);
        try item.put(a, "sha256", .{ .string = try a.dupe(u8, &digest) });
        try item.put(a, "url", .{ .string = try std.fmt.allocPrint(a, "{s}{s}/content/{d}", .{ prefix, snapshot.id, artifact.index }) });
        value.* = .{ .object = item };
    }
    try out.put(a, "artifacts", .{ .array = artifacts });
    return .{ .object = out };
}

fn inspect(srv: *Server, conn: *Conn, a: Allocator, id: []const u8) void {
    inspectStatus(srv, conn, a, id, 200);
}

fn inspectStatus(srv: *Server, conn: *Conn, a: Allocator, id: []const u8, status: u16) void {
    var snapshot = srv.media_jobs.registry.inspect(a, id, conn.key_digest) catch |e| return sendError(conn, a, 500, @errorName(e));
    if (snapshot == null) return sendError(conn, a, 404, "media job not found");
    defer snapshot.?.deinit(a);
    const value = snapshotValue(a, &snapshot.?) catch return;
    routes.sendValue(conn, a, status, value);
}

pub fn get(srv: *Server, conn: *Conn, a: Allocator, content: ?ContentPath, id: ?[]const u8) void {
    if (!enabled(srv)) return unavailable(srv, conn, a);
    if (id) |job_id| return inspect(srv, conn, a, job_id);
    const asked = content.?;
    var snapshot = srv.media_jobs.registry.inspect(a, asked.id, conn.key_digest) catch |e| return sendError(conn, a, 500, @errorName(e));
    if (snapshot == null) return sendError(conn, a, 404, "media job not found");
    defer snapshot.?.deinit(a);
    if (snapshot.?.state != .succeeded) return sendError(conn, a, 409, "media job content is not ready");
    var result = srv.media_jobs.registry.result(a, asked.id, conn.key_digest, asked.index) catch |e| return sendError(conn, a, 500, @errorName(e));
    if (result == null) return sendError(conn, a, 404, "media artifact not found");
    defer result.?.deinit(a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", result.?.media_type) catch return;
    var length: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{result.?.bytes.len}) catch return) catch return;
    const digest = std.fmt.bytesToHex(result.?.digest, .lower);
    conn.addHeader("X-TensorFold-SHA256", &digest) catch return;
    conn.finish(result.?.bytes) catch {};
}

pub fn remove(srv: *Server, conn: *Conn, a: Allocator, id: []const u8) void {
    if (!enabled(srv)) return unavailable(srv, conn, a);
    const removed = srv.media_jobs.registry.remove(id, conn.key_digest) catch |e| switch (e) {
        error.JobActive => return sendError(conn, a, 409, "an active media job must be cancelled before deletion"),
        else => return sendError(conn, a, 500, @errorName(e)),
    };
    if (!removed) return sendError(conn, a, 404, "media job not found");
    conn.startResponse(204, null) catch return;
    conn.addHeader("Content-Length", "0") catch return;
    conn.finish("") catch {};
}
