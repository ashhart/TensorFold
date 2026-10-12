//! One streamed request per attempt with a deadline and Ctrl-C, and the warm-up and repeat order of a run.
const std = @import("std");
const json = @import("json.zig");
const suite = @import("benchmark_suite.zig");
const sse = @import("benchmark_sse.zig");
const stop = @import("benchmark_stop.zig");
const Allocator = std.mem.Allocator;

pub const user_agent = "TensorFold-benchmark/1";
/// A 512 KiB line plus its line break fits, so the line limits are checked here, not by the reader.
const read_buffer = 576 << 10;

pub const Target = struct { base: []const u8, served: []const u8 };
pub const Opts = struct { tokens: u32, reps: u32, temperatures: []const f64, serial: bool, timeout_s: f64 };
/// ``call`` after each attempt; ``said`` after one the server refused, with its reason for the terminal only.
pub const Progress = struct { ctx: *anyopaque, call: *const fn (*anyopaque, *const suite.Sample) void, said: ?*const fn (*anyopaque, *const suite.Sample, []const u8) void = null };

/// An http(s) URL split as ``urlsplit`` reads it, with no credentials, query or fragment.
pub const Url = struct { scheme: []const u8, authority: []const u8, host: []const u8, path: []const u8 };

pub fn splitUrl(text: []const u8) ?Url {
    const uri = std.Uri.parse(text) catch return null;
    const http = std.ascii.eqlIgnoreCase(uri.scheme, "http") or std.ascii.eqlIgnoreCase(uri.scheme, "https");
    if (!http or uri.user != null or uri.password != null) return null;
    const host = (uri.host orelse return null).percent_encoded;
    if (host.len == 0) return null;
    if (uri.query) |q| if (!q.isEmpty()) return null;
    if (uri.fragment) |f| if (!f.isEmpty()) return null;
    const after = text[uri.scheme.len + 3 ..];
    const end = std.mem.indexOfAny(u8, after, "/?#") orelse after.len;
    const path = uri.path.percent_encoded;
    return .{ .scheme = uri.scheme, .authority = after[0..end], .host = std.mem.trim(u8, host, "[]"), .path = path };
}

/// The route for ``base``: ``/v1`` added unless the path ends with it, then the chat or completion route.
pub fn endpoint(a: Allocator, base: []const u8, chat: bool, problem: *[]const u8) !?[]const u8 {
    const u = splitUrl(base) orelse {
        problem.* = "Server must be an http or https URL without credentials, query or fragment";
        return null;
    };
    const path = std.mem.trimEnd(u8, u.path, "/");
    const v1 = if (std.mem.endsWith(u8, path, "/v1")) "" else "/v1";
    const scheme = try std.ascii.allocLowerString(a, u.scheme);
    return try std.fmt.allocPrint(a, "{s}://{s}{s}{s}{s}", .{ scheme, u.authority, path, v1, if (chat) "/chat/completions" else "/completions" });
}

/// The request body in its fixed key order, written with ``json.dumps`` defaults.
pub fn body(a: Allocator, served: []const u8, f: suite.Fixture, tokens: u32, t: f64, seed: u32, serial: bool) ![]u8 {
    const o = try json.newObject(a);
    try o.put(a, "model", .{ .string = served });
    try o.put(a, "max_tokens", try json.intValue(a, tokens));
    try o.put(a, "temperature", .{ .float = t });
    try o.put(a, "seed", try json.intValue(a, seed));
    try o.put(a, "stream", .{ .bool = true });
    const so = try json.newObject(a);
    try so.put(a, "include_usage", .{ .bool = true });
    try o.put(a, "stream_options", .{ .object = so });
    if (t > 0) {
        try o.put(a, "top_k", try json.intValue(a, 20));
        try o.put(a, "top_p", .{ .float = 0.95 });
    }
    if (serial) try o.put(a, "draft", .{ .bool = false });
    if (f.chat) {
        const m = try json.newObject(a);
        try m.put(a, "role", .{ .string = "user" });
        try m.put(a, "content", .{ .string = f.prompt });
        const messages = try a.alloc(json.Value, 1);
        messages[0] = .{ .object = m };
        try o.put(a, "messages", .{ .array = messages });
        const kw = try json.newObject(a);
        try kw.put(a, "enable_thinking", .{ .bool = false });
        try o.put(a, "chat_template_kwargs", .{ .object = kw });
    } else try o.put(a, "prompt", .{ .string = f.prompt });
    return json.stringify(a, .{ .object = o }, .{});
}

pub fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn deadlineAfter(io: std.Io, seconds: f64) i96 {
    return now(io) + nanoseconds(seconds);
}

/// Seconds as nanoseconds through i64, so no 128-bit float conversion is linked.
pub fn nanoseconds(seconds: f64) i96 {
    return @as(i64, @intFromFloat(@min(@max(seconds, 0), 1e7) * 1e9));
}

pub const Ended = enum { finished, deadline, stopped, unavailable };

/// Runs ``f(ctx)`` as a task polled every 10 ms; a deadline, stop or canceled caller ends it; ``unavailable``: none.
pub fn bounded(io: std.Io, deadline: ?i96, comptime f: anytype, ctx: anytype) Ended {
    const Wrap = struct {
        fn call(c: @TypeOf(ctx)) void {
            defer c.done.store(true, .release);
            f(c);
        }
    };
    var future = io.concurrent(Wrap.call, .{ctx}) catch return .unavailable;
    while (true) {
        if (ctx.done.load(.acquire)) {
            future.await(io);
            return .finished;
        }
        if (stop.requested() != null) {
            future.cancel(io);
            return .stopped;
        }
        if (deadline) |d| if (now(io) >= d) {
            future.cancel(io);
            return .deadline;
        };
        io.sleep(.fromMilliseconds(10), .awake) catch {
            future.cancel(io);
            return .stopped;
        };
    }
}

const Exchange = struct {
    a: Allocator,
    io: std.Io,
    client: *std.http.Client,
    uri: std.Uri,
    body: []u8,
    tokens: u64,
    stream: sse.Stream,
    sample: *suite.Sample,
    end: ?i96 = null,
    failure: ?suite.ErrorCode = null,
    status: u16 = 0,
    refused: []const u8 = "",
    decided: bool = false,
    completed: bool = false,
    done: std.atomic.Value(bool) = .init(false),

    fn run(x: *Exchange) void {
        var req = x.client.request(.POST, x.uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .content_type = .{ .override = "application/json" }, .accept_encoding = .{ .override = "identity" }, .user_agent = .{ .override = user_agent } },
            .extra_headers = &.{.{ .name = "accept", .value = "text/event-stream" }},
        }) catch {
            x.failure = .connection_error;
            return;
        };
        defer {
            req.deinit();
            x.end = x.end orelse now(x.io);
        }
        req.sendBodyComplete(x.body) catch {
            x.failure = .connection_error;
            return;
        };
        var resp = req.receiveHead(&.{}) catch |e| {
            x.failure = if (e == error.HttpHeadersInvalid or e == error.HttpHeadersOversize) .truncated_stream else .connection_error;
            return;
        };
        if (resp.head.status.class() != .success) {
            x.failure = .http_error;
            x.status = @backingInt(resp.head.status);
            x.end = now(x.io);
            x.decided = true;
            x.refused = errorBody(x.a, x.io, &resp);
            return;
        }
        if (!eventStream(resp.head.content_type)) {
            x.failure = .invalid_stream;
            return;
        }
        var arrived = now(x.io);
        const transfer = x.a.alloc(u8, read_buffer) catch {
            x.failure = .invalid_stream;
            return;
        };
        const r = resp.reader(transfer);
        while (true) {
            const had_line = std.mem.indexOfScalar(u8, r.buffered(), '\n') != null;
            const raw = r.takeDelimiterInclusive('\n') catch |e| switch (e) {
                error.EndOfStream => {
                    if (r.buffered().len > sse.max_event) x.failure = .invalid_stream;
                    break;
                },
                error.StreamTooLong => {
                    x.failure = .invalid_stream;
                    return;
                },
                error.ReadFailed => {
                    x.failure = if (resp.bodyErr() != null) .truncated_stream else .connection_error;
                    return;
                },
            };
            if (!had_line) arrived = now(x.io);
            const at_done = x.stream.line(raw, arrived) catch |f| {
                x.failure = sse.code(f);
                return;
            };
            if (at_done) break;
        }
        if (x.failure != null) return;
        x.completed = true;
        x.stream.finish(x.tokens, x.sample) catch |f| {
            x.failure = sse.code(f);
        };
    }
};

/// A refusal's body, up to 16 KiB, read for at most 100 ms past its head, after its sample's time and code are set.
fn errorBody(a: Allocator, io: std.Io, resp: *const std.http.Client.Response) []const u8 {
    if (resp.head.transfer_encoding != .none) return "";
    var wait: BodyWait = .{ .in = resp.request.reader.in, .want = @min(resp.head.content_length orelse 16 << 10, 16 << 10) };
    if (wait.in.buffered().len < wait.want) _ = bounded(io, deadlineAfter(io, 0.1), BodyWait.run, &wait);
    const got = wait.in.buffered();
    return a.dupe(u8, got[0..@min(got.len, wait.want)]) catch "";
}

const BodyWait = struct {
    in: *std.Io.Reader,
    want: usize,
    done: std.atomic.Value(bool) = .init(false),

    fn run(w: *BodyWait) void {
        while (w.in.buffered().len < w.want) w.in.fillMore() catch return;
    }
};

fn eventStream(content_type: ?[]const u8) bool {
    const t = content_type orelse return false;
    const media = std.mem.trim(u8, t[0 .. std.mem.indexOfScalar(u8, t, ';') orelse t.len], " \t");
    return std.ascii.eqlIgnoreCase(media, "text/event-stream");
}

/// One request as one sample: failures keep a fixed code, never server text; ``said`` gets a refusal's reason.
pub fn attempt(a: Allocator, io: std.Io, client: *std.http.Client, t: Target, o: Opts, f: suite.Fixture, temperature: f64, seed: u32, repeat: i32, said: ?*[]const u8) suite.Sample {
    var sample: suite.Sample = .{ .fixture_id = f.id, .temperature = temperature, .repeat = repeat, .seed = seed };
    if (said) |out| out.* = "";
    if (stop.requested() != null) {
        sample.error_code = .cancelled;
        sample.end_to_end_seconds = 0;
        return sample;
    }
    var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
    defer arena.deinit();
    const ta = arena.allocator();
    var x: Exchange = .{ .a = ta, .io = io, .client = client, .uri = undefined, .body = undefined, .tokens = o.tokens, .stream = undefined, .sample = &sample };
    const start = now(io);
    const ready = blk: {
        var problem: []const u8 = "";
        const url = (endpoint(ta, t.base, f.chat, &problem) catch break :blk false) orelse break :blk false;
        x.uri = std.Uri.parse(url) catch break :blk false;
        x.body = body(ta, t.served, f, o.tokens, temperature, seed, o.serial) catch break :blk false;
        x.stream = sse.Stream.init(ta) catch break :blk false;
        break :blk true;
    };
    if (!ready) {
        sample.error_code = .connection_error;
        sample.end_to_end_seconds = 0;
        return sample;
    }
    const ended = bounded(io, start + nanoseconds(o.timeout_s), Exchange.run, &x);
    x.stream.timings(start, x.end orelse now(io), &sample);
    sample.token_sha = if (sample.token_sha) |s| a.dupe(u8, s) catch null else null;
    const failure: ?suite.ErrorCode = if (x.completed) x.failure else if (ended == .stopped or stop.requested() != null) .cancelled else if (ended == .deadline and !x.decided) .timeout else x.failure orelse .connection_error;
    if (failure) |c| {
        sample.status = .@"error";
        sample.error_code = c;
    }
    const refused = if (failure) |c| c == .http_error else false;
    if (said != null and refused) said.?.* = a.dupe(u8, sse.refusal(ta, x.status, x.refused) catch "") catch "";
    return sample;
}

/// The run order: per temperature and fixture, a warm-up kept only when it failed, then the repeats.
pub fn run(a: Allocator, io: std.Io, t: Target, o: Opts, progress: ?Progress) ![]suite.Sample {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io, .read_buffer_size = read_buffer };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    var samples: std.ArrayList(suite.Sample) = .empty;
    var said: []const u8 = "";
    for (o.temperatures) |temp| for (suite.fixtures) |f| {
        const warm = attempt(a, io, &client, t, o, f, temp, suite.seed_base, -1, &said);
        tell(progress, &warm, said);
        if (warm.status == .@"error") {
            try samples.append(a, warm);
            if (warm.error_code == .cancelled) return samples.items;
        }
        for (0..o.reps) |r| {
            const s = attempt(a, io, &client, t, o, f, temp, suite.seed_base + @as(u32, @intCast(r)), @intCast(r), &said);
            try samples.append(a, s);
            tell(progress, &s, said);
            if (s.error_code == .cancelled) return samples.items;
        }
    };
    return samples.items;
}

fn tell(progress: ?Progress, s: *const suite.Sample, said: []const u8) void {
    const p = progress orelse return;
    p.call(p.ctx, s);
    if (said.len > 0) if (p.said) |f| f(p.ctx, s, said);
}

/// A whole small exchange for the JSON routes: readiness probes, the receiver and the public-checkpoint check.
pub const Fetch = struct {
    method: std.http.Method = .GET,
    url: []const u8,
    body: ?[]const u8 = null,
    token: ?[]const u8 = null,
    limit: usize,
    timeout_s: f64,
};

pub const Fetched = struct {
    status: u16 = 0,
    json_type: bool = false,
    body: []const u8 = "",
    failed: enum { none, network, too_large, stopped, deadline } = .none,
};

const FetchTask = struct {
    a: Allocator,
    client: *std.http.Client,
    f: Fetch,
    out: Fetched = .{},
    done: std.atomic.Value(bool) = .init(false),

    fn run(x: *FetchTask) void {
        x.out.failed = .network;
        const uri = std.Uri.parse(x.f.url) catch return;
        var auth_buf: [300]u8 = undefined;
        const auth: std.http.Client.Request.Headers.Value = if (x.f.token) |tok| .{ .override = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{tok}) catch return } else .omit;
        var req = x.client.request(x.f.method, uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .authorization = auth, .accept_encoding = .{ .override = "identity" }, .user_agent = .{ .override = user_agent }, .content_type = if (x.f.body != null) .{ .override = "application/json" } else .default },
            .extra_headers = &.{.{ .name = "accept", .value = "application/json" }},
        }) catch return;
        defer req.deinit();
        if (std.ascii.eqlIgnoreCase(uri.scheme, "https") and !tlsDirect(req.connection)) return;
        if (x.f.body) |b| req.sendBodyComplete(@constCast(b)) catch return else req.sendBodiless() catch return;
        var resp = req.receiveHead(&.{}) catch return;
        x.out.status = @backingInt(resp.head.status);
        x.out.json_type = if (resp.head.content_type) |ct| std.mem.indexOf(u8, ct, "application/json") != null else false;
        if (resp.head.status.class() != .success) {
            x.out.failed = .none;
            return;
        }
        var transfer: [16 << 10]u8 = undefined;
        const r = resp.reader(&transfer);
        x.out.body = r.allocRemaining(x.a, .limited(x.f.limit)) catch |e| {
            if (e == error.StreamTooLong) x.out.failed = .too_large;
            return;
        };
        x.out.failed = .none;
    }
};

/// An https exchange is sent only over TLS to the URL's own host, never in clear text to a proxy.
fn tlsDirect(connection: ?*std.http.Client.Connection) bool {
    const c = connection orelse return false;
    return c.protocol == .tls and !c.proxied;
}

/// One ``Fetch`` under its deadline; a body over ``limit`` bytes is ``too_large``.
pub fn fetch(a: Allocator, io: std.Io, client: *std.http.Client, f: Fetch) Fetched {
    var task: FetchTask = .{ .a = a, .client = client, .f = f };
    return switch (bounded(io, deadlineAfter(io, f.timeout_s), FetchTask.run, &task)) {
        .finished => task.out,
        .stopped => .{ .failed = .stopped },
        .deadline => .{ .failed = .deadline },
        .unavailable => .{ .failed = .network },
    };
}
