//! One message on a Responses socket: ``response.create`` through the POST route's gates and turn, steer, or refusal.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const log = @import("log.zig");
const openai = @import("openai.zig");
const responses = @import("responses.zig");
const translate_mod = @import("responses_translate.zig");
const store_mod = @import("responses_store.zig");
const request_gate = @import("request_gate.zig");
const admission = @import("request_admission.zig");
const auth_scope = @import("auth_scope.zig");
const scope_reply = @import("auth_scope_reply.zig");
const wait = @import("wait_reason.zig");
const read = @import("responses_ws_read.zig");
const events = @import("responses_ws_events.zig");
const ws = @import("responses_ws.zig");
const Session = ws.Session;
const Problem = events.Problem;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Handles one whole client message in its own arena, with the session answering the run's cancellation check.
pub fn message(s: *Session, text: []const u8) void {
    var arena = std.heap.ArenaAllocator.init(s.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    s.conn.gone_hook = .{ .ctx = s, .call = read.pollGone };
    defer s.conn.gone_hook = null;
    const parsed = json.parse(a, text) catch return s.closeWith(1011, "out of memory");
    const body: Value = if (parsed == .ok and parsed.ok == .object) parsed.ok else .null;
    const kind = body.strField("type") orelse "";
    if (std.mem.eql(u8, kind, "response.create")) return create(s, a, body);
    if (std.mem.eql(u8, kind, "response.steer")) return events.steerFailed(s, a, body);
    events.connectionError(s, .{
        .status = 400,
        .kind = "invalid_request_error",
        .code = "invalid_event",
        .message = if (body == .null) "a client event must be a JSON object with a type" else "unknown event type: send response.create",
        .param = if (body == .null) null else "type",
    });
}

/// How a create ended: the HTTP status the request gets (0: the client left before any event) and what the cache keeps.
const Done = struct {
    status: u16,
    final: ?Value = null,
    added: []const Value = &.{},
};

fn create(s: *Session, a: Allocator, body: Value) void {
    const started = s.now();
    const lane = laneOf(s, body) orelse return count(s, 400, null, null, started);
    var history: SocketHistory = .{ .session = s, .store_flag = storeFlag(body) };
    const done = respond(s, a, body, lane, &history);
    const final = done.final;
    const status = if (final) |f| f.strField("status") orelse "" else "";
    const ended = for ([_][]const u8{ "completed", "incomplete", "cancelled" }) |k| { // a cut reply is kept as written
        if (std.mem.eql(u8, status, k)) break true;
    } else false;
    const kept = done.status == 200 and ended;
    if (kept) remember(s, a, lane, final.?, history.earlier, done.added) else if (done.status != 0) forget(s, body, lane);
    count(s, done.status, if (final) |f| f.strField("id") else null, lane, started);
}

/// A finished response becomes its lane's cached entry: the earlier messages, the request's, then its output's.
fn remember(s: *Session, a: Allocator, lane: []const u8, final: Value, earlier: []const Value, added: []const Value) void {
    var cx: Cx = .{ .a = a };
    const output = final.get("output") orelse return;
    if (output != .array) return;
    const from_output = translate_mod.messages(&cx, output.array) catch return;
    const all = std.mem.concat(a, Value, &.{ earlier, added, from_output }) catch return;
    _ = s.cache.put(lane, final.strField("id") orelse return, all) catch {};
}

/// A failed continuation on its parent's own lane drops that parent; a fork from another lane keeps it.
fn forget(s: *Session, body: Value, lane: []const u8) void {
    const parent = body.strField("previous_response_id") orelse return;
    const entry = s.cache.find(parent) orelse return;
    if (std.mem.eql(u8, entry.lane, lane)) s.cache.evict(parent);
}

/// The request's lane ("" by default), or null after its error event.
fn laneOf(s: *Session, body: Value) ?[]const u8 {
    const raw = body.get("stream_id") orelse return "";
    if (raw == .null) return "";
    if (raw != .string or !validLane(raw.string)) {
        events.requestError(s, null, .{ .status = 400, .kind = "invalid_request_error", .code = "invalid_stream_id", .message = "stream_id must be 1 to 256 letters, digits, '_', '.' or '-'", .param = "stream_id" });
        return null;
    }
    if (s.lanes.contains(raw.string)) return raw.string;
    if (s.lanes.count() >= ws.max_lanes) {
        events.requestError(s, raw.string, .{ .status = 400, .kind = "invalid_request_error", .code = "websocket_stream_limit_reached", .message = "this connection already has 32 named streams: reuse one or open a new connection", .param = "stream_id" });
        return null;
    }
    const owned = s.gpa.dupe(u8, raw.string) catch return null;
    s.lanes.put(s.gpa, owned, {}) catch {
        s.gpa.free(owned);
        return null;
    };
    return owned;
}

/// ``^[A-Za-z0-9_.-]{1,256}$``.
fn validLane(name: []const u8) bool {
    if (name.len == 0 or name.len > 256) return false;
    for (name) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-')) return false;
    return true;
}

/// ``store`` as the translation reads it: true unless it is ``false``.
fn storeFlag(body: Value) bool {
    const field = body.get("store") orelse return true;
    return !(field == .bool and !field.bool);
}

fn refuse(s: *Session, stream_id: ?[]const u8, p: Problem) Done {
    events.requestError(s, stream_id, p);
    return .{ .status = p.status };
}

/// The POST route's order: the key, the gate, the body's scope, the history, then the turn or the warmup.
fn respond(s: *Session, a: Allocator, body: Value, lane: []const u8, history: *SocketHistory) Done {
    const stream_id: ?[]const u8 = if (lane.len == 0) null else lane;
    if (!revalidate(s, a)) return .{ .status = 401 };
    if (!auth_scope.allowsUpgrade(s.conn.key_scopes)) return refuse(s, stream_id, Problem.fromBody(a, 403, scope_reply.text(false, .route)));
    var entered = request_gate.enter(s.srv, a, s.conn.key_digest, .responses, .{ .text = true, .embedding = false, .rerank = false, .systemone = false });
    defer entered.held.release();
    if (entered.refusal) |r| return refuse(s, stream_id, gateProblem(a, r));
    if (bodyDenial(s, s.conn.key_scopes, body)) |why| return refuse(s, stream_id, Problem.fromBody(a, 403, scope_reply.text(false, why)));
    if (json.truthyField(body, "background")) return refuse(s, stream_id, .{ .status = 400, .kind = "invalid_request_error", .message = "background: true is not supported on a socket: POST /v1/responses over HTTP, then poll GET /v1/responses/{id}", .param = "background" });
    const generate = body.get("generate") orelse Value{ .bool = true };
    if (generate != .bool) return refuse(s, stream_id, .{ .status = 400, .kind = "invalid_request_error", .message = "generate must be a boolean", .param = "generate" });
    for ([_][]const u8{ "type", "stream_id", "generate" }) |k| _ = body.object.orderedRemove(k);
    var cx: Cx = .{ .a = a };
    var t = translate_mod.translate(&cx, body, history) catch |e| {
        if (e == error.OutOfMemory) cx.message = "out of memory";
        return refuse(s, stream_id, .{ .status = 400, .kind = "invalid_request_error", .code = cx.code, .message = cx.message, .param = cx.param });
    };
    // every socket response streams: the request's own ``stream`` is ignored
    t.chat.object.put(a, "stream", .{ .bool = true }) catch return refuse(s, stream_id, .{ .status = 500, .kind = "server_error", .message = "out of memory" });
    t.stream = true;
    var out: events.Lane = .{ .session = s, .stream_id = stream_id };
    const opts: responses.TurnOptions = .{ .whole_history = if (history.from_cache) history.earlier else null };
    if (!generate.bool) return warm(s, a, &cx, body, t, &out, opts);
    return switch (responses.turn(s.srv, a, s.conn, &cx, body, t, out.sink(), opts)) {
        .streamed => |final| .{ .status = if (final == null and s.gone and !out.sent) 0 else 200, .final = final, .added = t.added },
        .refused => |r| refuse(s, stream_id, Problem.fromPayload(r.status, r.payload)),
        // unreachable with ``stream`` forced on: nothing was sent, so nothing is kept
        .whole => .{ .status = 500 },
    };
}

/// ``generate: false``: the turn's own checks and prompt count, then three events and no engine run.
fn warm(s: *Session, a: Allocator, cx: *Cx, body: Value, t: translate_mod.Request, out: *events.Lane, opts: responses.TurnOptions) Done {
    const gone: openai.Gone = .{ .conn = s.conn };
    switch (@import("openai_prepare.zig").prepare(s.srv, a, gone, t.chat, t.format, true)) {
        .cancelled => return .{ .status = 0 },
        .refused => |r| return refuse(s, out.stream_id, Problem.fromPayload(r.status, r.payload)),
        .prepared => |p| {
            s.srv.release(p.preparing, p.request);
            const final = responses.warmup(s.srv, a, cx, body, t, out.sink(), p.prompt_len, opts);
            return .{ .status = if (final == null and s.gone and !out.sent) 0 else 200, .final = final, .added = t.added };
        },
    }
}

/// The gate's refusal with the status and code the POST route sends for it.
fn gateProblem(a: Allocator, r: request_gate.Refusal) Problem {
    return switch (r) {
        .draining => .{ .status = 503, .kind = "server_error", .code = wait.restarting.code, .message = wait.restarting.message },
        .overloaded => |o| .{ .status = 429, .kind = "rate_limit_error", .code = admission.code(o.reason), .message = admission.message(o.reason) },
        .model_kind => |text| Problem.fromBody(a, 400, text),
        .waiting => |w| blk: {
            log.line("idle reload failed: {s}", .{w.problem});
            break :blk .{ .status = 503, .kind = "server_error", .code = w.wait.code, .message = w.wait.message };
        },
    };
}

/// A restricted key sends ``store: false``; its ``previous_response_id`` must be in this socket's cache.
fn bodyDenial(s: *const Session, scopes: auth_scope.Scopes, body: Value) ?auth_scope.Denial {
    if (scopes.admin) return null;
    if (body.get("previous_response_id")) |parent| if (parent != .null) {
        if (parent != .string or s.cache.find(parent.string) == null) return .response_history;
    };
    const store = body.get("store") orelse return .response_storage;
    if (store != .bool or store.bool) return .response_storage;
    return null;
}

/// The upgrade's key looked up again: gone closes the socket (401, 1008); changed scopes apply from now on.
fn revalidate(s: *Session, a: Allocator) bool {
    const p = if (s.principal) |*held| held else return true;
    const keys = s.srv.keys orelse return true;
    if (!keys.enabled()) return true;
    const current = keys.byDigest(a, p.digest) orelse {
        events.connectionError(s, .{ .status = 401, .kind = "invalid_request_error", .code = "invalid_api_key", .message = "A valid API key is required" });
        s.closeWith(1008, "API key removed");
        return false;
    };
    if (!std.mem.eql(u8, current.label, p.label)) if (s.gpa.dupe(u8, current.label)) |owned| {
        s.gpa.free(p.label);
        p.label = owned;
    } else |_| {};
    p.scopes = current.scopes;
    s.conn.key_label = p.label;
    s.conn.key_scopes = p.scopes;
    return true;
}

/// The key-labelled request counter and one log line per create, as an HTTP request gets.
fn count(s: *Session, status: u16, id: ?[]const u8, lane: ?[]const u8, started: i64) void {
    if (status == 0) return;
    if (s.conn.auth_enabled) s.srv.metrics.httpRequest(s.srv.io, s.conn.key_label orelse "unauthenticated", status);
    const shown = if (lane) |l| (if (l.len == 0) "default" else l) else "-";
    log.line("{s} \"WS response.create\" {d} id={s} lane={s} {d} ms{s}", .{ s.conn.peer, status, id orelse "-", shown, s.now() - started, log.keySuffix(s.conn) });
}

/// ``previous_response_id`` read from the connection cache first, then the server store only with ``store: true``.
const SocketHistory = struct {
    session: *Session,
    store_flag: bool,
    /// The earlier messages returned, in the turn's arena.
    earlier: []Value = &.{},
    from_cache: bool = false,

    pub fn conversation(h: *SocketHistory, cx: *Cx, rid: Value) errors.Refused![]Value {
        if (rid != .string) return cx.refuse("previous_response_id must be a string");
        if (h.session.cache.find(rid.string)) |e| {
            const copy = try cx.a.alloc(Value, e.messages.len);
            for (e.messages, copy) |m, *slot| slot.* = try json.deepCopy(cx.a, m);
            h.earlier = copy;
            h.from_cache = true;
            return copy;
        }
        if (h.store_flag) {
            const stored: store_mod.History = .{ .store = &h.session.srv.store, .io = h.session.srv.io };
            if (stored.lookup(cx, rid.string)) |messages| {
                h.earlier = messages;
                return messages;
            }
        }
        cx.code = "previous_response_not_found";
        cx.param = "previous_response_id";
        return cx.fail(.request, "Previous response with id '{s}' not found.", .{rid.string});
    }
};

test "a lane name is 1 to 256 of letters, digits and _.-" {
    const most: [256]u8 = @splat('x');
    const over: [257]u8 = @splat('x');
    try std.testing.expect(validLane("main") and validLane("a.b_c-9") and validLane(&most));
    try std.testing.expect(!validLane("") and !validLane(&over) and !validLane("a b") and !validLane("é"));
}
