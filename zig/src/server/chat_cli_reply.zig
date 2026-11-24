//! One streamed chat reply as its bytes arrive, however reads cut them: answer and reasoning pieces, finish, error.
const std = @import("std");
const json = @import("json");
const sse = @import("benchmark_sse.zig");
const Allocator = std.mem.Allocator;

pub const Kind = enum { content, reasoning };

/// Where each piece goes as it arrives; a failed write stops the reply.
pub const Sink = struct { ctx: *anyopaque, piece: *const fn (*anyopaque, Kind, []const u8) error{WriteFailed}!void };

pub const Failure = error{ InvalidStream, WriteFailed, OutOfMemory };

/// The bytes one line of a server's error keeps on a terminal.
const error_limit = 2000;

pub const Reply = struct {
    a: Allocator,
    sink: Sink,
    secrets: []const []const u8 = &.{},
    answer_stream: Stream = .{},
    reason_stream: Stream = .{},
    pending: std.ArrayList(u8) = .empty,
    frames: sse.Frames = .{},
    /// The answer so far: what a kept turn holds.
    content: std.ArrayList(u8) = .empty,
    /// Reasoning arrived, shown or not.
    reasoned: bool = false,
    finish: ?[]const u8 = null,
    /// An error event's message, cleaned for a terminal.
    failure: ?[]const u8 = null,
    done: bool = false,

    /// Redact a final ambiguous credential prefix after the stream stops.
    pub fn end(r: *Reply) Failure!void {
        try r.flush(.reasoning);
        try r.flush(.content);
    }

    fn redacted(r: *Reply, kind: Kind, text: []const u8) !?[]const u8 {
        const state = streamFor(r, kind);
        const whole = if (state.pending.len > 0) try std.mem.concat(r.a, u8, &.{ state.pending, text }) else text;
        const cleaned = try sse.sanitizeStream(r.a, whole);
        const got = try sse.redactPiece(r.a, cleaned.text, r.secrets);
        state.pending = if (got.rest.len == 0 and cleaned.rest.len == 0) "" else try std.mem.concat(r.a, u8, &.{ got.rest, cleaned.rest });
        const candidate = if (state.transformed.len > 0) try std.mem.concat(r.a, u8, &.{ state.transformed, got.text }) else got.text;
        const checked = try sse.containedPiece(r.a, candidate, r.secrets);
        state.transformed = checked.rest;
        return checked.emit;
    }

    fn flush(r: *Reply, kind: Kind) Failure!void {
        const state = streamFor(r, kind);
        // Unresolved source or transformed suffixes are consumed, never replaced by an unchecked marker.
        state.pending = "";
        state.transformed = "";
    }

    /// Bytes as they were read: whole lines are handled now, a cut line waits for the rest.
    pub fn feed(r: *Reply, bytes: []const u8) Failure!void {
        var rest = bytes;
        while (!r.done) {
            const nl = std.mem.indexOfScalar(u8, rest, '\n') orelse {
                if (r.pending.items.len + rest.len > sse.max_event + 2) return error.InvalidStream;
                try r.pending.appendSlice(r.a, rest);
                return;
            };
            const whole = rest[0 .. nl + 1];
            rest = rest[nl + 1 ..];
            if (r.pending.items.len == 0) {
                try r.line(whole);
                continue;
            }
            try r.pending.appendSlice(r.a, whole);
            defer r.pending.clearRetainingCapacity();
            try r.line(r.pending.items);
        }
    }

    /// The reply ended as a reply should: ``[DONE]``, or a finish reason before the stream closed.
    pub fn complete(r: *const Reply) bool {
        return r.failure == null and (r.done or r.finish != null);
    }

    fn line(r: *Reply, with_newline: []const u8) Failure!void {
        const data = try r.frames.line(r.a, with_newline) orelse return;
        try r.event(data);
    }

    fn event(r: *Reply, data: []const u8) Failure!void {
        if (std.mem.eql(u8, std.mem.trim(u8, data, " \t\r\n"), "[DONE]")) {
            r.done = true;
            return;
        }
        const chunk = switch (try json.parseText(r.a, data)) {
            .ok => |v| v,
            .err => return error.InvalidStream,
        };
        if (chunk != .object) return error.InvalidStream;
        if (chunk.field("error")) |e| {
            try r.flush(.reasoning);
            try r.flush(.content);
            r.failure = try errorText(r.a, e, r.secrets);
            return;
        }
        const choices = chunk.get("choices") orelse return; // a usage event carries none
        if (choices != .array or choices.array.len == 0) return;
        const choice = choices.array[0];
        if (choice.get("delta")) |delta| {
            const thought = delta.strField("reasoning_content") orelse delta.strField("reasoning");
            if (thought) |t| if (t.len > 0) {
                r.reasoned = true;
                if (try r.redacted(.reasoning, t)) |safe| if (safe.len > 0) try r.sink.piece(r.sink.ctx, .reasoning, safe);
            };
            if (delta.strField("content")) |c| if (c.len > 0) {
                if (try r.redacted(.content, c)) |safe| {
                    try r.content.appendSlice(r.a, safe);
                    if (safe.len > 0) try r.sink.piece(r.sink.ctx, .content, safe);
                }
            };
        }
        if (choice.field("finish_reason")) |f| if (f == .string) {
            r.finish = f.string;
        };
    }
};

const Stream = struct { pending: []const u8 = "", transformed: []const u8 = "" };

fn streamFor(r: *Reply, kind: Kind) *Stream {
    return if (kind == .content) &r.answer_stream else &r.reason_stream;
}

/// An error event's ``message`` sanitized before matching and contained after every transformation.
fn errorText(a: Allocator, e: json.Value, secrets: []const []const u8) ![]const u8 {
    const text = e.strField("message") orelse e.str() orelse "the server reported an error without a message";
    return (try sse.cleanContained(a, text, secrets, error_limit)) orelse "";
}

const Seen = struct {
    a: Allocator,
    text: std.ArrayList(u8) = .empty,

    fn piece(ctx: *anyopaque, kind: Kind, text: []const u8) error{WriteFailed}!void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.text.appendSlice(s.a, if (kind == .reasoning) "<r>" else "<c>") catch return error.WriteFailed;
        s.text.appendSlice(s.a, text) catch return error.WriteFailed;
    }

    fn sink(s: *Seen) Sink {
        return .{ .ctx = s, .piece = piece };
    }
};

const content_event = "data: {\"id\": \"x\", \"choices\": [{\"index\": 0, \"delta\": {\"content\": \"Hel\"}, \"finish_reason\": null}]}\n\n";
const stream_bytes = ": keepalive\n\n" ++
    "data: {\"choices\": [{\"index\": 0, \"delta\": {\"role\": \"assistant\", \"reasoning_content\": \"The user greets.\"}}]}\n\n" ++
    content_event ++
    "data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"lo \\u00e9\\n\"}}]}\r\n\r\n" ++
    "data: {\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}], \"tensorfold\": {\"tokens_per_second\": 1.0}}\n\n" ++
    "data: {\"choices\": [], \"usage\": {\"prompt_tokens\": 3, \"completion_tokens\": 4}}\n\n" ++
    "data: [DONE]\n\n" ++
    "data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"after the end\"}}]}\n\n";

/// ``stream_bytes`` fed in reads of ``step`` bytes after a first read of ``cut``, checked as one whole read.
fn playCut(cut: usize, step: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: Seen = .{ .a = a };
    var r: Reply = .{ .a = a, .sink = seen.sink() };
    try r.feed(stream_bytes[0..cut]);
    var at = cut;
    while (at < stream_bytes.len) : (at = @min(at + step, stream_bytes.len)) try r.feed(stream_bytes[at..@min(at + step, stream_bytes.len)]);
    try std.testing.expectEqualStrings("<r>The user greets.<c>Hel<c>lo \xc3\xa9\n", seen.text.items);
    try std.testing.expectEqualStrings("Hello \xc3\xa9\n", r.content.items);
    try std.testing.expectEqualStrings("stop", r.finish.?);
    try std.testing.expect(r.done and r.complete() and r.failure == null and r.reasoned);
}

test "chat CLI: a stream's reasoning and answer pieces, finish and [DONE], with reads cut at every byte" {
    for (0..stream_bytes.len + 1) |cut| try playCut(cut, stream_bytes.len);
    try playCut(0, 1);
    try playCut(3, 7);
}

test "chat CLI: an error event's message is kept, cleaned for a terminal, and the reply is not complete" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: Seen = .{ .a = a };
    var r: Reply = .{ .a = a, .sink = seen.sink() };
    try r.feed(content_event ++ "data: {\"error\": {\"message\": \"This server's maximum context length is 8 tokens\\u001b[2J\", \"type\": \"invalid_request_error\"}}\n\ndata: [DONE]\n\n");
    try std.testing.expectEqualStrings("This server's maximum context length is 8 tokens", r.failure.?);
    try std.testing.expect(r.done and !r.complete());
    var plain: Reply = .{ .a = a, .sink = seen.sink() };
    try plain.feed("data: {\"error\": \"overloaded\"}\n\n");
    try std.testing.expectEqualStrings("overloaded", plain.failure.?);
    var bare: Reply = .{ .a = a, .sink = seen.sink() };
    try bare.feed("data: {\"error\": {\"code\": 500}}\n\n");
    try std.testing.expectEqualStrings("the server reported an error without a message", bare.failure.?);
}

test "chat CLI: a stream that ends early, an event that is not JSON and an endless line are told apart" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: Seen = .{ .a = a };
    var cut: Reply = .{ .a = a, .sink = seen.sink() };
    try cut.feed(content_event);
    try std.testing.expect(!cut.complete());
    var finished: Reply = .{ .a = a, .sink = seen.sink() };
    try finished.feed("data: {\"choices\": [{\"index\": 0, \"delta\": {\"reasoning\": \"r\"}, \"finish_reason\": \"length\"}]}\n\n");
    try std.testing.expect(finished.complete() and !finished.done and finished.reasoned and finished.content.items.len == 0);
    try std.testing.expectEqualStrings("length", finished.finish.?);
    var broken: Reply = .{ .a = a, .sink = seen.sink() };
    try std.testing.expectError(error.InvalidStream, broken.feed("data: {\"choices\": [\n\n"));
    var long: Reply = .{ .a = a, .sink = seen.sink() };
    const filler = try a.alloc(u8, sse.max_event);
    @memset(filler, 'x');
    try long.feed("data: ");
    try std.testing.expectError(error.InvalidStream, long.feed(filler));
}
