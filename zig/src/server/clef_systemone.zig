//! /v1/systemone on a readout engine: Clef's record encoded, its joint schema head's logits, the reference's answers.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const readout = @import("lanes").readout;
const enc = @import("clef_encode.zig");
const answer = @import("clef_answer.zig");
const http_body = @import("http_body.zig");
const errors = @import("errors.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// The release's longest record, and the bounds the engine plans its readouts for (the design's).
pub const max_length = 16384;
pub const max_questions = 64;
pub const max_group = 256;
pub const max_options = 1024;
pub const score_levels = [2]usize{ 2, 10 };

/// What a request gets back: the answers (200), a validation error list (422), or a refusal (400, 500, 503).
pub const Reply = struct { status: u16, body: Value };

/// A FastAPI validation error: where (`loc`), what (`msg`) and its pydantic type.
const Issue = struct { loc: []const []const u8, msg: []const u8, kind: []const u8 };

const Check = struct {
    a: Allocator,
    issues: std.ArrayList(Issue) = .empty,

    fn add(c: *Check, loc: []const []const u8, msg: []const u8, kind: []const u8) !void {
        try c.issues.append(c.a, .{ .loc = try c.a.dupe([]const u8, loc), .msg = msg, .kind = kind });
    }
    fn missing(c: *Check, loc: []const []const u8) !void {
        try c.add(loc, "Field required", "missing");
    }
};

/// 422's checks: every field the request needs is there with its JSON kind; values are judged after (400).
fn shapes(c: *Check, body: Value) !void {
    if (body != .object) return c.add(&.{"body"}, "Input should be a valid dictionary or object to extract fields from", "dict_type");
    if (body.get("model")) |m| {
        if (m != .string) try c.add(&.{ "body", "model" }, "Input should be a valid string", "string_type");
    } else try c.missing(&.{ "body", "model" });
    if (!body.has("state")) try c.missing(&.{ "body", "state" });
    if (body.get("raw")) |r| if (r != .bool) try c.add(&.{ "body", "raw" }, "Input should be a valid boolean", "bool_type");
    const qs = body.get("questions") orelse return c.missing(&.{ "body", "questions" });
    if (qs != .object) return c.add(&.{ "body", "questions" }, "Input should be a valid dictionary", "dict_type");
    for (qs.object.keys(), qs.object.values()) |id, q| {
        if (q != .object) {
            try c.add(&.{ "body", "questions", id }, "Input should be a valid dictionary", "dict_type");
            continue;
        }
        const t = q.get("type") orelse {
            try c.missing(&.{ "body", "questions", id, "type" });
            continue;
        };
        if (t != .string) try c.add(&.{ "body", "questions", id, "type" }, "Input should be a valid string", "string_type");
        const crit = q.get("criteria") orelse continue;
        const kind = if (t == .string) std.meta.stringToEnum(enc.Kind, t.string) else null;
        const loc: []const []const u8 = &.{ "body", "questions", id, "criteria" };
        switch (kind orelse continue) {
            .choice => if (crit != .object) try c.add(loc, "Input should be a valid dictionary", "dict_type"),
            .score => if (crit != .array) try c.add(loc, "Input should be a valid list", "list_type"),
            // the reference's `criteria.update(question.get("criteria") or {})`: falsy values change nothing
            .noul => if (crit != .object and crit.truthy()) try c.add(loc, "Input should be a valid dictionary", "dict_type"),
        }
    }
}

fn detail(a: Allocator, issues: []const Issue) !Value {
    const list = try a.alloc(Value, issues.len);
    for (issues, list) |i, *v| {
        const o = try json.newObject(a);
        const loc = try a.alloc(Value, i.loc.len);
        for (i.loc, loc) |part, *l| l.* = .{ .string = part };
        try o.put(a, "type", .{ .string = i.kind });
        try o.put(a, "loc", .{ .array = loc });
        try o.put(a, "msg", .{ .string = i.msg });
        v.* = .{ .object = o };
    }
    const out = try json.newObject(a);
    try out.put(a, "detail", .{ .array = list });
    return .{ .object = out };
}

fn refusal(a: Allocator, status: u16, comptime fmt: []const u8, args: anytype) !Reply {
    var cx: errors.Cx = .{ .a = a };
    _ = cx.fail(if (status == 500) .server else if (status == 503) .capacity else .request, fmt, args) catch {};
    return .{ .status = status, .body = try openai.wrapError(a, try openai.errorBody(a, &cx, null)) };
}

/// 400's checks on a well-shaped request: known types, non-empty criteria, the engine's bounds, no media.
fn values(a: Allocator, srv: *const Server, body: Value) !?Reply {
    for ([_][]const u8{ "images", "videos" }) |k| if (body.get(k)) |v| if (v.truthy())
        return try refusal(a, 400, "{s}: this engine reads text records only", .{k});
    const cap = srv.info.readouts.?;
    const qs = body.get("questions").?.object;
    if (qs.count() == 0) return try refusal(a, 400, "at least one question is required", .{});
    if (qs.count() > @min(max_questions, cap.max_questions)) return try refusal(a, 400, "at most {d} questions", .{@min(max_questions, cap.max_questions)});
    var total: usize = 0;
    for (qs.keys(), qs.values()) |id, q| {
        const kind = std.meta.stringToEnum(enc.Kind, q.get("type").?.string) orelse
            return try refusal(a, 400, "{s}: type must be noul, choice, or score", .{id});
        const crit = q.get("criteria") orelse Value.null;
        if (kind != .noul and !crit.truthy()) return try refusal(a, 400, "{s}: criteria must not be empty", .{id});
        const n: usize = switch (kind) {
            .noul => 2,
            .choice => crit.object.count(),
            .score => crit.array.len,
        };
        if (kind == .score and (n < score_levels[0] or n > score_levels[1])) return try refusal(a, 400, "{s}: a score has {d} to {d} levels", .{ id, score_levels[0], score_levels[1] });
        if (n > max_group) return try refusal(a, 400, "{s}: at most {d} options", .{ id, max_group });
        total += n;
    }
    if (total > @min(max_options, cap.max_options)) return try refusal(a, 400, "at most {d} options in all", .{@min(max_options, cap.max_options)});
    return null;
}

/// The reply to a System One body on a readout engine; `gone` says when the client left (the request is cancelled).
pub fn reply(srv: *Server, a: Allocator, body: Value, gone: anytype) !?Reply {
    var check: Check = .{ .a = a };
    try shapes(&check, body);
    if (check.issues.items.len > 0) return .{ .status = 422, .body = try detail(a, check.issues.items) };
    if (try values(a, srv, body)) |r| return r;
    const window: u32 = @intCast(@min(max_length, srv.info.context_window, srv.info.readouts.?.max_rows));
    const fields = enc.fields(a, body.get("questions").?) catch |e| return if (e == error.OutOfMemory) e else try refusal(a, 400, "the questions cannot be encoded", .{});
    const rec = enc.encode(a, srv.text, body.get("state").?, fields, window) catch |e| return switch (e) {
        error.OutOfMemory => e,
        error.SchemaTooLong => try refusal(a, 400, "the schema needs more than {d} tokens before the state", .{window}),
        error.Template => try refusal(a, 400, "the record cannot be tokenized", .{}),
    };
    var spans: std.ArrayList([2]u32) = .empty;
    var owner: std.ArrayList(u16) = .empty;
    const qspans = try a.alloc([2]u32, rec.questions.len);
    const types = try a.alloc(u8, rec.questions.len);
    for (rec.questions, qspans, types, 0..) |q, *s, *t, k| {
        s.* = q.span;
        t.* = @backingInt(q.kind);
        try spans.appendSlice(a, q.options);
        try owner.appendNTimes(a, @intCast(k), q.options.len);
    }
    const logits = try a.alloc(u16, spans.items.len);
    const spec: readout.Spec = .{ .question_spans = qspans, .question_types = types, .option_spans = spans.items, .option_question = owner.items, .logits = logits };
    const ask: api.Request = .{ .prompt = rec.ids, .max_tokens = 0, .drafts = false, .untaught = true, .readout = &spec };
    const id = srv.next_id.fetchAdd(1, .monotonic);
    if (gone.check()) return null;
    const ended = @import("prompt_only_wait.zig").run(srv.engine, srv.io, id, &ask, gone) catch |err| return switch (err) {
        error.InvalidRequest, error.InvalidSpans => try refusal(a, 400, "the record does not fit the engine's readouts", .{}),
        error.Busy, error.Closed, error.Stale => try refusal(a, 503, "the readout engine is unavailable", .{}),
        error.MemoryPressure => try refusal(a, 503, "{s}", .{api.pressure_message}),
    };
    switch (ended) {
        .cancelled => return null,
        .failed => return try refusal(a, 500, "the joint schema head failed", .{}),
        .done => {},
    }
    const raw = if (body.get("raw")) |r| r.bool else false;
    const value = answer.body(a, body.get("model").?, fields, rec, logits, raw) catch |e| return switch (e) {
        error.OutOfMemory => e,
        error.NonFinite => try refusal(a, 500, "the joint schema head gave a non-finite logit", .{}),
    };
    return .{ .status = 200, .body = value };
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: errors.Cx = .{ .a = a };
    const raw = http_body.readJson(conn, &cx) catch {
        const err = openai.errorBody(a, &cx, null) catch return;
        return routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
    };
    const Gone = struct {
        conn: *Conn,
        pub fn check(g: @This()) bool {
            return g.conn.peerGone();
        }
    };
    const r = (reply(srv, a, raw, Gone{ .conn = conn }) catch return conn.sendJson(500, "{\"error\": {\"message\": \"out of memory\"}}")) orelse {
        conn.close = true;
        return;
    };
    routes.sendValue(conn, a, r.status, r.body);
}
