//! The `compat` wire (``Config.wire``): replies, errors, /health, /metrics, /v1/models and the request log.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const compat_log = @import("compat_log.zig");
pub const RequestLog = compat_log.RequestLog;

pub const Wire = enum { tensorfold, compat };
/// /health's ``streams``: a batch engine's slots decoding, prefilling, and in all.
pub const Streams = struct { decoding: u32, prefilling: u32, max: u32 };
pub const Mode = enum { basic, strict };

/// The reply id's random hex digits (``uuid4().hex[:24]``).
pub const id_digits = 24;

pub const LogSettings = struct {
    path: []const u8,
    max_bytes: u64 = 64 << 20,
    keep: u32 = 3,
    window: u32 = 64,
    salt: []const u8 = "",
};

pub const Settings = struct {
    disconnect: bool = true,
    health: Mode = .basic,
    stall_s: f64 = 0,
    stall_prefill_tps: f64 = 200,
    log: ?LogSettings = null,
    /// TF_DRAIN_S: seconds requests in progress may finish after a stop signal, on every wire (serve.zig)
    drain_s: f64 = 20,
    /// TF_SSE_KEEPALIVE_S: a silent stream's keepalive period before its first token, on every wire (0: none)
    keepalive_s: f64 = 15,

    /// The knobs from ``env`` (``TF_COMPAT_X``); ``problem`` names a bad one.
    pub fn fromEnv(env: ?*const std.process.Environ.Map, problem: *[]const u8) error{Invalid}!Settings {
        var s: Settings = .{};
        const m = env orelse return s;
        if (knob(m, "DISCONNECT")) |v| {
            const t = trimmed(v, "1");
            if (!std.mem.eql(u8, t, "0") and !std.mem.eql(u8, t, "1")) return bad(problem, "TF_COMPAT_DISCONNECT: expected 0 or 1");
            s.disconnect = std.mem.eql(u8, t, "1");
        }
        if (knob(m, "HEALTH")) |v| {
            const t = trimmed(v, "basic");
            s.health = if (std.ascii.eqlIgnoreCase(t, "basic")) .basic else if (std.ascii.eqlIgnoreCase(t, "strict")) .strict else return bad(problem, "TF_COMPAT_HEALTH: expected one of basic, strict");
        }
        if (knob(m, "STALL_S")) |v| s.stall_s = try nonNegative(v, 0, problem, "TF_COMPAT_STALL_S: must be a number >= 0");
        if (m.get("TF_COMPAT_STALL_PREFILL_TPS")) |v| s.stall_prefill_tps = try nonNegative(v, 200, problem, "TF_COMPAT_STALL_PREFILL_TPS: must be a number >= 0");
        if (m.get("TF_DRAIN_S")) |v| s.drain_s = try nonNegative(v, 20, problem, "TF_DRAIN_S: seconds, a number >= 0");
        if (m.get("TF_SSE_KEEPALIVE_S")) |v| s.keepalive_s = try nonNegative(v, 15, problem, "TF_SSE_KEEPALIVE_S: seconds, a number >= 0");
        if (knob(m, "REQUEST_LOG")) |v| {
            const path = std.mem.trim(u8, v, " \t");
            if (path.len > 0 and !std.mem.eql(u8, path, "0")) {
                var l: LogSettings = .{ .path = path };
                if (m.get("TF_COMPAT_REQUEST_LOG_MB")) |x| {
                    const mb = std.fmt.parseFloat(f64, trimmed(x, "64")) catch return bad(problem, "TF_COMPAT_REQUEST_LOG_MB: a number > 0");
                    if (!(mb > 0)) return bad(problem, "TF_COMPAT_REQUEST_LOG_MB must be > 0");
                    l.max_bytes = @intFromFloat(mb * (1 << 20));
                }
                if (m.get("TF_COMPAT_REQUEST_LOG_KEEP")) |x| l.keep = std.fmt.parseInt(u32, trimmed(x, "3"), 10) catch return bad(problem, "TF_COMPAT_REQUEST_LOG_KEEP: an integer >= 0");
                if (m.get("TF_COMPAT_REQUEST_LOG_PROMPTS")) |x| l.window = std.fmt.parseInt(u32, trimmed(x, "64"), 10) catch return bad(problem, "TF_COMPAT_REQUEST_LOG_PROMPTS: an integer >= 0");
                if (m.get("TF_COMPAT_REQUEST_LOG_SALT")) |x| l.salt = x;
                s.log = l;
            }
        }
        return s;
    }

    fn knob(m: *const std.process.Environ.Map, comptime name: []const u8) ?[]const u8 {
        return m.get("TF_COMPAT_" ++ name);
    }

    fn trimmed(v: []const u8, default: []const u8) []const u8 {
        const t = std.mem.trim(u8, v, " \t");
        return if (t.len == 0) default else t;
    }

    fn bad(problem: *[]const u8, message: []const u8) error{Invalid} {
        problem.* = message;
        return error.Invalid;
    }

    fn nonNegative(v: []const u8, default: f64, problem: *[]const u8, message: []const u8) error{Invalid}!f64 {
        const t = std.mem.trim(u8, v, " \t");
        if (t.len == 0) return default;
        const x = std.fmt.parseFloat(f64, t) catch return bad(problem, message);
        if (!(x >= 0)) return bad(problem, message);
        return x;
    }
};

/// Python's ``round(x, n)`` (correctly rounded through the decimal text).
pub fn round(x: f64, comptime n: u8) f64 {
    var buf: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "{d:." ++ std.fmt.comptimePrint("{d}", .{n}) ++ "}", .{x}) catch return x;
    return std.fmt.parseFloat(f64, text) catch x;
}

/// A JSON number as Python's ``dict`` held it: an int when integral and ``integral``, else a float.
fn number(a: Allocator, x: f64, integral: bool) Allocator.Error!Value {
    if (integral and x == @trunc(x) and @abs(x) < 9.0e15) return json.intValue(a, @as(i64, @intFromFloat(x)));
    return .{ .float = x };
}

/// What one request's engine run reports, in the wire's key order.
pub const Outcome = struct {
    prompt: usize,
    cached: u32 = 0,
    ttft_s: ?f64 = null,
    cancelled: bool = false,
    /// tokens the engine delivered (an EOS and tokens past a stop string included)
    completion: usize = 0,
    prefill_s: ?f64 = null,
    decode_s: ?f64 = null,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,

    /// The reply's ``tensorfold`` object.
    pub fn value(o: Outcome, a: Allocator) Allocator.Error!*json.Object {
        const s = try json.newObject(a);
        try s.put(a, "prompt", try json.intValue(a, o.prompt));
        try s.put(a, "cached", try json.intValue(a, o.cached));
        if (o.ttft_s) |t| try s.put(a, "ttft_s", .{ .float = round(t, 4) });
        try s.put(a, "finish", .{ .string = if (o.cancelled) "cancelled" else "done" });
        try s.put(a, "completion", try json.intValue(a, o.completion));
        if (o.prefill_s) |p| try s.put(a, "prefill_s", .{ .float = round(p, 4) });
        if (o.decode_s) |d| try s.put(a, "decode_s", .{ .float = round(d, 4) });
        try s.put(a, "rounds", try json.intValue(a, o.rounds));
        try s.put(a, "drafted", try json.intValue(a, o.drafted));
        try s.put(a, "accepted", try json.intValue(a, o.accepted));
        return s;
    }
};

// -- /health and /metrics -------------------------------------------------------------------------------------------

/// Health: requests in flight, the first fatal engine error, counters and the engine's live totals.
pub const Health = struct {
    gpa: Allocator,
    io: std.Io,
    mode: Mode,
    stall_s: f64,
    prefill_tps: f64,
    started: i96,
    mutex: std.Io.Mutex = .init,
    fatal: ?[]u8 = null,
    inflight: std.AutoArrayHashMapUnmanaged(u64, Entry) = .empty,
    next_id: u64 = 0,
    last_done: ?i96 = null,
    c: Counters = .{},
    t: Totals = .{},

    const Entry = struct { prompt: f64, start: i96, last: i96, tokens: f64 = 0 };
    const Counters = struct { requests: f64 = 0, errors: f64 = 0, rejected: f64 = 0, prompt_tokens: f64 = 0, completion_tokens: f64 = 0, cached_tokens: f64 = 0, decode_rounds: f64 = 0, decode_seconds: f64 = 0, prefill_seconds: f64 = 0 };
    const Totals = struct { requests_total: f64 = 0, prompt_tokens_total: f64 = 0, completion_tokens_total: f64 = 0, prefill_seconds_total: f64 = 0, decode_seconds_total: f64 = 0, cached_tokens_total: f64 = 0, rounds_total: f64 = 0, drafted_total: f64 = 0, accepted_total: f64 = 0 };

    /// How a run ended: its stats, or an error (``value_error``: the request's own fault, not the engine's).
    pub const End = union(enum) { ok: Outcome, failed: struct { value_error: bool, message: []const u8 } };

    pub fn init(gpa: Allocator, io: std.Io, s: Settings) Health {
        return .{ .gpa = gpa, .io = io, .mode = s.health, .stall_s = s.stall_s, .prefill_tps = s.stall_prefill_tps, .started = now(io) };
    }

    pub fn deinit(h: *Health) void {
        if (h.fatal) |f| h.gpa.free(f);
        h.inflight.deinit(h.gpa);
    }

    fn now(io: std.Io) i96 {
        return std.Io.Clock.awake.now(io).toNanoseconds();
    }

    fn secs(ns: i96) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e9;
    }

    pub fn begin(h: *Health, prompt: usize) u64 {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const id = h.next_id;
        h.next_id += 1;
        const t = now(h.io);
        h.inflight.put(h.gpa, id, .{ .prompt = @floatFromInt(prompt), .start = t, .last = t }) catch {};
        return id;
    }

    pub fn progress(h: *Health, id: u64, n: usize) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const e = h.inflight.getPtr(id) orelse return;
        e.last = now(h.io);
        e.tokens += @floatFromInt(n);
    }

    pub fn end(h: *Health, id: u64, how: End) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const entry = h.inflight.fetchSwapRemove(id);
        h.last_done = now(h.io);
        const t = &h.t;
        t.requests_total += 1;
        if (entry) |e| {
            t.prompt_tokens_total += e.value.prompt;
            t.completion_tokens_total += e.value.tokens;
        }
        h.c.requests += 1;
        switch (how) {
            .failed => |f| {
                h.c.errors += 1;
                if (!f.value_error and h.fatal == null) h.fatal = h.gpa.dupe(u8, f.message[0..@min(f.message.len, 500)]) catch null;
            },
            .ok => |o| {
                const cached: f64 = @floatFromInt(o.cached);
                t.prefill_seconds_total += o.prefill_s orelse 0;
                t.decode_seconds_total += o.decode_s orelse 0;
                t.cached_tokens_total += cached;
                t.rounds_total += @floatFromInt(o.rounds);
                t.drafted_total += @floatFromInt(o.drafted);
                t.accepted_total += @floatFromInt(o.accepted);
                h.c.prompt_tokens += if (entry) |e| e.value.prompt else 0;
                h.c.completion_tokens += @floatFromInt(o.completion);
                h.c.cached_tokens += cached;
                h.c.decode_rounds += @floatFromInt(o.rounds);
                h.c.decode_seconds += o.decode_s orelse 0;
                h.c.prefill_seconds += o.prefill_s orelse 0;
            },
        }
    }

    /// Why a new completion must not start (strict mode after a fatal error), or null.
    pub fn reject(h: *Health, a: Allocator) ?[]const u8 {
        if (h.mode != .strict) return null;
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const f = h.fatal orelse return null;
        h.c.rejected += 1;
        return std.fmt.allocPrint(a, "the engine failed earlier ({s}); restart both ranks", .{f}) catch "the engine failed earlier; restart both ranks";
    }

    const Stalled = struct { prompt: f64, quiet_s: f64, allowed_s: f64 };

    fn stalled(h: *Health, a: Allocator, t: i96) Allocator.Error![]Stalled {
        var out: std.ArrayList(Stalled) = .empty;
        if (h.stall_s <= 0) return out.items;
        for (h.inflight.values()) |e| {
            var allow = h.stall_s;
            if (e.tokens == 0 and h.prefill_tps > 0) allow += e.prompt / h.prefill_tps;
            const quiet = secs(t - e.last);
            if (quiet > allow) try out.append(a, .{ .prompt = e.prompt, .quiet_s = round(quiet, 1), .allowed_s = round(allow, 1) });
        }
        return out.items;
    }

    /// ``GET /health``: the status code and body. ``streams``: the engine's slots (null: not a batch engine).
    pub fn status(h: *Health, a: Allocator, streams: ?Streams, context: u32) Allocator.Error!struct { code: u16, body: Value } {
        const o = try json.newObject(a);
        h.mutex.lockUncancelable(h.io);
        const t = now(h.io);
        const st = try h.stalled(a, t);
        const ok = h.fatal == null and st.len == 0;
        var oldest: f64 = 0;
        var running_tokens: f64 = 0;
        for (h.inflight.values()) |e| {
            oldest = @max(oldest, secs(t - e.start));
            running_tokens += e.tokens;
        }
        const running = h.inflight.count();
        try o.put(a, "ok", .{ .bool = ok });
        try o.put(a, "mode", .{ .string = @tagName(h.mode) });
        try o.put(a, "uptime_s", .{ .float = round(secs(t - h.started), 1) });
        try o.put(a, "inflight", try json.intValue(a, running));
        try o.put(a, "oldest_s", .{ .float = round(oldest, 1) });
        try o.put(a, "idle_s", if (h.last_done != null and running == 0) .{ .float = round(secs(t - h.last_done.?), 1) } else .null);
        try o.put(a, "requests", try number(a, h.c.requests, true));
        try o.put(a, "errors", try number(a, h.c.errors, true));
        if (h.fatal) |f| try o.put(a, "fatal", .{ .string = try a.dupe(u8, f) });
        if (st.len > 0) {
            const list = try a.alloc(Value, st.len);
            for (st, list) |s, *slot| {
                const x = try json.newObject(a);
                try x.put(a, "prompt", .{ .float = s.prompt });
                try x.put(a, "quiet_s", .{ .float = s.quiet_s });
                try x.put(a, "allowed_s", .{ .float = s.allowed_s });
                slot.* = .{ .object = x };
            }
            try o.put(a, "stalled", .{ .array = list });
        }
        try o.put(a, "backend", .{ .string = "tensorfold" });
        try o.put(a, "busy", .{ .bool = running > 0 });
        try o.put(a, "requests_running", try json.intValue(a, running));
        const tt = h.t;
        h.mutex.unlock(h.io);
        inline for (@typeInfo(Totals).@"struct".field_names) |name| {
            var x: f64 = @field(tt, name);
            if (comptime std.mem.eql(u8, name, "completion_tokens_total")) x += running_tokens;
            try o.put(a, name, try number(a, if (x == @trunc(x)) x else round(x, 6), true));
        }
        if (streams) |s| {
            const x = try json.newObject(a);
            try x.put(a, "decoding", try json.intValue(a, s.decoding));
            try x.put(a, "prefilling", try json.intValue(a, s.prefilling));
            try x.put(a, "max", try json.intValue(a, s.max));
            try o.put(a, "streams", .{ .object = x });
        }
        if (context > 0) try o.put(a, "context_length", try json.intValue(a, context));
        return .{ .code = if (h.mode == .strict and !ok) 503 else 200, .body = .{ .object = o } };
    }

    /// ``GET /metrics``: Prometheus text, one ``model`` label.
    pub fn metrics(h: *Health, w: *std.Io.Writer, served: []const u8) std.Io.Writer.Error!void {
        h.mutex.lockUncancelable(h.io);
        const t = now(h.io);
        const c = h.c;
        const inflight = h.inflight.count();
        var stalled_n: usize = 0;
        if (h.stall_s > 0) {
            var scratch: [4096]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&scratch);
            stalled_n = if (h.stalled(fba.allocator(), t)) |st| st.len else |_| 0;
        }
        const fatal: f64 = if (h.fatal != null) 1 else 0;
        const uptime = secs(t - h.started);
        h.mutex.unlock(h.io);
        const Row = struct { []const u8, []const u8, []const u8, f64 };
        const rows = [_]Row{
            .{ "tensorfold_requests_total", "counter", "completions finished (errors included)", c.requests },
            .{ "tensorfold_request_errors_total", "counter", "completions that raised", c.errors },
            .{ "tensorfold_requests_rejected_total", "counter", "completions refused after a fatal error", c.rejected },
            .{ "tensorfold_prompt_tokens_total", "counter", "prompt tokens of finished completions", c.prompt_tokens },
            .{ "tensorfold_cached_tokens_total", "counter", "prompt tokens resumed instead of prefilled", c.cached_tokens },
            .{ "tensorfold_completion_tokens_total", "counter", "generated tokens", c.completion_tokens },
            .{ "tensorfold_decode_rounds_total", "counter", "decode rounds (tokens / rounds: draft acceptance)", c.decode_rounds },
            .{ "tensorfold_decode_seconds_total", "counter", "seconds decoding", c.decode_seconds },
            .{ "tensorfold_prefill_seconds_total", "counter", "seconds prefilling", c.prefill_seconds },
            .{ "tensorfold_requests_inflight", "gauge", "completions running now", @floatFromInt(inflight) },
            .{ "tensorfold_requests_stalled", "gauge", "running completions past TF_COMPAT_STALL_S", @floatFromInt(stalled_n) },
            .{ "tensorfold_engine_fatal", "gauge", "1 once the engine raised (restart both ranks)", fatal },
            .{ "tensorfold_uptime_seconds", "gauge", "seconds since the server started", uptime },
        };
        for (rows) |r| {
            try w.print("# HELP {s} {s}\n# TYPE {s} {s}\n{s}{{model=\"", .{ r[0], r[2], r[0], r[1], r[0] });
            for (served) |ch| switch (ch) {
                '\\' => try w.writeAll("\\\\"),
                '"' => try w.writeAll("\\\""),
                else => try w.writeByte(ch),
            };
            if (r[3] == @trunc(r[3]) and @abs(r[3]) < 9.0e15) try w.print("\"}} {d}\n", .{@as(i64, @intFromFloat(r[3]))}) else try w.print("\"}} {d:.6}\n", .{r[3]});
        }
    }
};

// -- replies --------------------------------------------------------------------------------------------------------

/// ``context_problem``'s message: OpenAI's and vLLM's words, then how to raise the limit.
pub fn contextMessage(a: Allocator, prompt: usize, asked: ?u64, limit: u32, chat: bool) Allocator.Error![]const u8 {
    const where = if (chat) "messages" else "prompt";
    const need = prompt + (asked orelse 1);
    const what = if (asked) |n|
        try std.fmt.allocPrint(a, "However, you requested {d} tokens ({d} in the {s}, {d} in the completion). Please reduce the length of the {s} or completion.", .{ need, prompt, where, n, where })
    else
        try std.fmt.allocPrint(a, "However, your {s} resulted in {d} tokens. Please reduce the length of the {s}.", .{ where, prompt, where });
    return std.fmt.allocPrint(a, "This model's maximum context length is {d} tokens. {s} (This TensorFold server was started with --context {d} (CONTEXT); restart both ranks with CONTEXT={d} or more to serve it.)", .{ limit, what, limit, need });
}

/// ``error_body``: ``{"error": {"message", "type", "code"?, "param"?}}``.
pub fn errorBody(a: Allocator, message: []const u8, kind: []const u8, code: ?[]const u8, param: ?[]const u8) Allocator.Error!Value {
    const e = try json.newObject(a);
    try e.put(a, "message", .{ .string = message });
    try e.put(a, "type", .{ .string = kind });
    if (code) |c| try e.put(a, "code", .{ .string = c });
    if (param) |p| try e.put(a, "param", .{ .string = p });
    const o = try json.newObject(a);
    try o.put(a, "error", .{ .object = e });
    return .{ .object = o };
}

/// ``GET /v1/models``: the served name, then each alias, with the context limit as vLLM names it.
pub fn models(a: Allocator, ids: []const []const u8, served: []const u8, created: i64, limit: u32) Allocator.Error!Value {
    const data = try a.alloc(Value, ids.len);
    for (ids, data) |id, *slot| {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = id });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        if (limit > 0) {
            try m.put(a, "created", try json.intValue(a, created));
            try m.put(a, "root", .{ .string = served });
            try m.put(a, "max_model_len", try json.intValue(a, limit));
            try m.put(a, "context_length", try json.intValue(a, limit));
        }
        slot.* = .{ .object = m };
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    return .{ .object = o };
}

/// The sampling and length fields read strictly (``20.0`` and ``"20"`` are integers, ``20.5`` and booleans not).
pub fn numbersProblem(a: Allocator, body: Value) Allocator.Error!?[]const u8 {
    const names = [_]struct { []const u8, bool }{ .{ "temperature", false }, .{ "top_p", false }, .{ "top_k", true }, .{ "seed", true }, .{ "max_tokens", true }, .{ "max_completion_tokens", true } };
    for (names) |n| {
        const v = body.field(n[0]) orelse continue;
        if (!numberOk(v, n[1])) return try std.fmt.allocPrint(a, "{s} must be {s} or null", .{ n[0], if (n[1]) "an integer" else "a finite number" });
    }
    return null;
}

fn numberOk(v: Value, integer: bool) bool {
    switch (v) {
        .int => return true,
        .float => |f| return if (integer) f == @trunc(f) and std.math.isFinite(f) else std.math.isFinite(f),
        .string => |s| {
            const t = std.mem.trim(u8, s, " \t\r\n\x0b\x0c");
            if (integer) {
                const digits = std.mem.trimStart(u8, t, "+-");
                if (digits.len == 0 or digits.len + 1 < t.len) return false;
                for (digits, 0..) |ch, i| if (!(std.ascii.isDigit(ch) or (ch == '_' and i > 0 and i + 1 < digits.len and digits[i - 1] != '_'))) return false;
                return true;
            }
            const f = std.fmt.parseFloat(f64, t) catch return false;
            return std.math.isFinite(f);
        },
        else => return false,
    }
}

/// ``parse_stop``: a string, a list of strings, or null.
pub fn stopProblem(body: Value) ?[]const u8 {
    const v = body.get("stop") orelse return null;
    switch (v) {
        .null, .string => return null,
        .array => |items| {
            for (items) |x| if (x != .string) return "stop must be a string or a list of strings";
            return null;
        },
        else => return "stop must be a string or a list of strings",
    }
}

/// ``token_ids_problem``: a ``/v1/completions`` token-ID prompt that is not one flat list of in-vocabulary ints.
pub fn tokenIdsProblem(a: Allocator, prompt: []const Value, vocab: u32) Allocator.Error!?[]const u8 {
    if (prompt.len == 0) return "prompt must not be empty";
    var lists = true;
    var strings = true;
    for (prompt) |p| {
        lists = lists and p == .array;
        strings = strings and p == .string;
    }
    if (lists or strings) return "one prompt a request: send a string or one list of token ids";
    for (prompt) |p| if (p != .int) return "a token-ID prompt must be a list of integers";
    for (prompt) |p| {
        const n = p.int64() orelse -1;
        if (n < 0 or n >= vocab) return try std.fmt.allocPrint(a, "token id {s} is outside the vocabulary (0 to {d})", .{ p.int, @as(i64, vocab) - 1 });
    }
    return null;
}

test "contextMessage is the Python server's" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("This model's maximum context length is 512 tokens. However, you requested 585 tokens (73 in the messages, 512 in the completion). Please reduce the length of the messages or completion. (This TensorFold server was started with --context 512 (CONTEXT); restart both ranks with CONTEXT=585 or more to serve it.)", try contextMessage(arena.allocator(), 73, 512, 512, true));
    try std.testing.expectEqualStrings("This model's maximum context length is 10 tokens. However, your prompt resulted in 12 tokens. Please reduce the length of the prompt. (This TensorFold server was started with --context 10 (CONTEXT); restart both ranks with CONTEXT=13 or more to serve it.)", try contextMessage(arena.allocator(), 12, null, 10, false));
}

test {
    _ = @import("compat_log.zig");
}
