//! One chat turn over HTTP: the request body, the streamed reply printed as it arrives, and how the turn ended.
const std = @import("std");
const json = @import("json");
const stream = @import("benchmark_stream.zig");
const sse = @import("benchmark_sse.zig");
const reply = @import("chat_cli_reply.zig");
const History = @import("chat_cli_history.zig").History;
const Allocator = std.mem.Allocator;

pub const user_agent = "TensorFold-chat/1";
/// One read's room; the reply checks its own line limit as lines arrive.
const read_buffer = 64 << 10;

/// What every request of a chat sends besides its messages.
pub const Settings = struct { model: []const u8, temperature: ?f64 = null, max_tokens: ?i64 = null, no_thinking: bool = false };

/// How a reply is shown: reasoning dimmed or hidden, and on a terminal its styles and only printable text.
pub const Show = struct { thinking: bool = false, terminal: bool = false };

/// A finished reply: its answer, whether it stopped at its token limit, and whether reasoning came.
pub const Finished = struct { content: []const u8, cut: bool, reasoned: bool };

/// Refused, failed or never sent: why, whether the server could not be reached, and a refusal's HTTP status.
pub const Failure = struct { reason: []const u8, connect: bool = false, status: u16 = 0 };

pub const Ended = union(enum) {
    finished: Finished,
    /// A stop signal ended it; the answer printed so far.
    stopped: []const u8,
    failed: Failure,
};

/// The streamed request: the model, the messages, then the settings the chat was given.
pub fn body(a: Allocator, s: Settings, history: *const History, user: []const u8) ![]u8 {
    const o = try json.newObject(a);
    try o.put(a, "model", .{ .string = s.model });
    try o.put(a, "messages", .{ .array = try history.messages(a, user) });
    try o.put(a, "stream", .{ .bool = true });
    if (s.max_tokens) |n| try o.put(a, "max_tokens", try json.intValue(a, n));
    if (s.temperature) |t| try o.put(a, "temperature", .{ .float = t });
    if (s.no_thinking) {
        const kw = try json.newObject(a);
        try kw.put(a, "enable_thinking", .{ .bool = false });
        try o.put(a, "chat_template_kwargs", .{ .object = kw });
    }
    return json.stringify(a, .{ .object = o }, .{});
}

/// One streamed turn to ``url``, the chat route, printed to ``out`` as it arrives; a stop signal ends it ``stopped``.
pub fn run(a: Allocator, io: std.Io, url: []const u8, key: ?[]const u8, request: []u8, out: *std.Io.Writer, show: Show) !Ended {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    var printer: Printer = .{ .io = io, .out = out, .show = show };
    var r: reply.Reply = .{ .a = a, .sink = printer.sink() };
    var auth_buf: [300]u8 = undefined;
    const bearer = if (key) |k| stream.rawHeader(&auth_buf, k) orelse return .{ .failed = .{
        .reason = "API key cannot be encoded as Latin-1 header text",
    } } else null;
    var x: Exchange = .{ .a = a, .io = io, .client = &client, .uri = try std.Uri.parse(url), .bearer = bearer, .body = request, .reply = &r };
    const ended = stream.bounded(io, null, Exchange.run, &x);
    printer.end();
    switch (ended) {
        .stopped, .deadline => return .{ .stopped = r.content.items },
        .unavailable => return .{ .failed = .{ .reason = "could not start the request" } },
        .finished => {},
    }
    if (x.failure) |f| return .{ .failed = .{ .reason = try x.reason(f, url), .connect = f == .connect, .status = x.status } };
    if (r.failure) |why| return .{ .failed = .{ .reason = why } };
    if (!r.complete()) return .{ .failed = .{ .reason = "the server closed the stream before the reply finished" } };
    return .{ .finished = .{ .content = r.content.items, .cut = std.mem.eql(u8, r.finish orelse "", "length"), .reasoned = r.reasoned } };
}

const Failed = enum { connect, lost, http, not_events, stream, output };

/// The request as its own task, so a stop signal ends the wait and closes the stream.
const Exchange = struct {
    a: Allocator,
    io: std.Io,
    client: *std.http.Client,
    uri: std.Uri,
    /// ``Bearer KEY``, when the server wants a key.
    bearer: ?[]const u8,
    body: []u8,
    reply: *reply.Reply,
    status: u16 = 0,
    refused: []const u8 = "",
    content_type: []const u8 = "",
    failure: ?Failed = null,
    done: std.atomic.Value(bool) = .init(false),

    fn run(x: *Exchange) void {
        const auth: std.http.Client.Request.Headers.Value = if (x.bearer) |b| .{ .override = b } else .omit;
        var req = x.client.request(.POST, x.uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .authorization = auth, .content_type = .{ .override = "application/json" }, .accept_encoding = .{ .override = "identity" }, .user_agent = .{ .override = user_agent } },
            .extra_headers = &.{.{ .name = "accept", .value = "text/event-stream" }},
        }) catch {
            x.failure = .connect;
            return;
        };
        defer req.deinit();
        req.sendBodyComplete(x.body) catch {
            x.failure = .connect;
            return;
        };
        var resp = req.receiveHead(&.{}) catch {
            x.failure = .connect;
            return;
        };
        if (resp.head.status.class() != .success) {
            x.failure = .http;
            x.status = @backingInt(resp.head.status);
            x.refused = stream.errorBody(x.a, x.io, &resp);
            return;
        }
        if (!stream.eventStream(resp.head.content_type)) {
            x.failure = .not_events;
            x.content_type = x.a.dupe(u8, resp.head.content_type orelse "none") catch "";
            return;
        }
        const transfer = x.a.alloc(u8, read_buffer) catch {
            x.failure = .stream;
            return;
        };
        const r = resp.reader(transfer);
        while (!x.reply.done) {
            const got = r.peekGreedy(1) catch |e| switch (e) {
                error.EndOfStream => return,
                error.ReadFailed => {
                    x.failure = .lost;
                    return;
                },
            };
            x.reply.feed(got) catch |e| {
                x.failure = if (e == error.WriteFailed) .output else .stream;
                return;
            };
            r.toss(got.len);
        }
    }

    fn reason(x: *const Exchange, f: Failed, url: []const u8) ![]const u8 {
        return switch (f) {
            .connect => try std.fmt.allocPrint(x.a, "could not reach the server at {s}", .{url}),
            .lost => "the connection to the server broke during the reply",
            .http => try sse.refusal(x.a, x.status, x.refused),
            .not_events => try std.fmt.allocPrint(x.a, "the server answered with {s}, not an event stream", .{try sse.clean(x.a, x.content_type, sse.line_limit)}),
            .stream => "the server sent a malformed event stream",
            .output => "could not write the reply",
        };
    }
};

const dim = "\x1b[2m";
const plain = "\x1b[0m";
const clear_line = "\r\x1b[K";

/// Pieces to ``out`` as they arrive, flushed at once, styled as ``show`` asks.
const Printer = struct {
    io: std.Io,
    out: *std.Io.Writer,
    show: Show,
    /// Shown reasoning is open: dimmed on a terminal, and a blank line comes before the answer.
    thinking: bool = false,
    /// The "thinking..." hint for hidden reasoning is on the line.
    hinted: bool = false,
    answered: bool = false,
    /// The last byte written was not a newline.
    open_line: bool = false,

    fn sink(p: *Printer) reply.Sink {
        return .{ .ctx = p, .piece = piece };
    }

    fn piece(ctx: *anyopaque, kind: reply.Kind, text: []const u8) error{WriteFailed}!void {
        const p: *Printer = @ptrCast(@alignCast(ctx));
        // a stop waits for this write, so a stopped reply never leaves a style or a character half written
        const old = p.io.swapCancelProtection(.blocked);
        defer _ = p.io.swapCancelProtection(old);
        try p.write(kind, text);
        try p.out.flush();
    }

    fn write(p: *Printer, kind: reply.Kind, text: []const u8) error{WriteFailed}!void {
        if (kind == .reasoning and !p.show.thinking) {
            if (!p.show.terminal or p.hinted or p.answered) return;
            try p.out.writeAll(dim ++ "thinking..." ++ plain);
            p.hinted = true;
            return;
        }
        if (p.hinted) try p.out.writeAll(clear_line);
        p.hinted = false;
        if (kind == .reasoning and !p.thinking) {
            if (p.show.terminal) try p.out.writeAll(dim);
            p.thinking = true;
        }
        if (kind == .content and p.thinking) {
            if (p.show.terminal) try p.out.writeAll(plain);
            try p.out.writeAll(if (p.open_line) "\n\n" else "\n");
            p.thinking = false;
            p.open_line = false;
        }
        if (kind == .content) p.answered = true;
        if (text.len == 0) return;
        if (p.show.terminal) try printable(p.out, text) else try p.out.writeAll(text);
        p.open_line = text[text.len - 1] != '\n';
    }

    /// After the reply: the hint gone, styles off and the cursor at the start of a line.
    fn end(p: *Printer) void {
        if (p.hinted) p.out.writeAll(clear_line) catch {};
        if (p.thinking and p.show.terminal) p.out.writeAll(plain) catch {};
        if (p.open_line) p.out.writeAll("\n") catch {};
        p.out.flush() catch {};
    }
};

/// ``text`` without control characters besides newline and tab, so a reply cannot move the cursor or restyle.
fn printable(out: *std.Io.Writer, text: []const u8) error{WriteFailed}!void {
    var start: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const c = text[i];
        const c1 = c == 0xc2 and i + 1 < text.len and text[i + 1] >= 0x80 and text[i + 1] < 0xa0;
        const drop: usize = if (c1) 2 else if ((c < 0x20 and c != '\n' and c != '\t') or c == 0x7f) 1 else 0;
        if (drop == 0) {
            i += 1;
            continue;
        }
        try out.writeAll(text[start..i]);
        i += drop;
        start = i;
    }
    try out.writeAll(text[start..]);
}

test "chat CLI: the request body sends the history, stream and only the settings given" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: History = .{ .a = a };
    try h.setSystem("Be brief.");
    try h.keep("one", "1");
    try std.testing.expectEqualStrings("{\"model\": \"m\", \"messages\": [{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"one\"}, " ++
        "{\"role\": \"assistant\", \"content\": \"1\"}, {\"role\": \"user\", \"content\": \"two\"}], \"stream\": true}", try body(a, .{ .model = "m" }, &h, "two"));
    h.clear();
    try std.testing.expectEqualStrings("{\"model\": \"m\", \"messages\": [{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"q\"}], " ++
        "\"stream\": true, \"max_tokens\": 64, \"temperature\": 0.0, \"chat_template_kwargs\": {\"enable_thinking\": false}}", try body(a, .{ .model = "m", .temperature = 0, .max_tokens = 64, .no_thinking = true }, &h, "q"));
}

test "chat CLI: on a terminal hidden reasoning shows a hint, shown reasoning is dimmed, and control characters are dropped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const cases = [_]struct { Show, []const u8 }{
        .{ .{}, "Hi \x1b[31mthere\x07\r\xc2\x9b\n" },
        .{ .{ .terminal = true }, dim ++ "thinking..." ++ plain ++ clear_line ++ "Hi [31mthere\n" },
        .{ .{ .thinking = true }, "Let me see.\n\nHi \x1b[31mthere\x07\r\xc2\x9b\n" },
        .{ .{ .thinking = true, .terminal = true }, dim ++ "Let me see." ++ plain ++ "\n\nHi [31mthere\n" },
    };
    for (cases) |c| {
        var out: std.Io.Writer.Allocating = .init(a);
        var p: Printer = .{ .io = io, .out = &out.writer, .show = c[0] };
        try Printer.piece(&p, .reasoning, "Let me ");
        try Printer.piece(&p, .reasoning, "see.");
        try Printer.piece(&p, .content, "Hi \x1b[31mthere\x07\r\xc2\x9b");
        if (!c[0].thinking) try Printer.piece(&p, .reasoning, "reasoning after the answer is never hinted");
        p.end();
        try std.testing.expectEqualStrings(c[1], out.written());
    }
    var out: std.Io.Writer.Allocating = .init(a);
    var stopped: Printer = .{ .io = io, .out = &out.writer, .show = .{ .thinking = true, .terminal = true } };
    try Printer.piece(&stopped, .reasoning, "cut off");
    stopped.end();
    try std.testing.expectEqualStrings(dim ++ "cut off" ++ plain ++ "\n", out.written());
}
