//! TENSORFOLD_REQUEST_LOG_REPLIES: one reply record per finished reply, beside its request, never per chunk.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const openai = @import("openai.zig");
const request_log = @import("request_log.zig");
const request_trace = @import("request_trace.zig");
const Value = json.Value;
const Trace = request_trace.Trace;
const t = std.testing;

fn noOpen(ctx: *anyopaque, n: usize) error{Closed}!void {
    _ = ctx;
    _ = n;
}

fn noEvent(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
    _ = ctx;
    _ = payload;
}

fn noReply(ctx: *anyopaque, status: u16, payload: Value) void {
    _ = ctx;
    _ = status;
    _ = payload;
}

/// A client that records nothing: only the file shows what the trace logged.
fn sink(ctx: *anyopaque) openai.Out {
    return .{ .ctx = ctx, .vt = &.{ .open = noOpen, .event = noEvent, .reply = noReply } };
}

/// A trace over a sink, its reply log on (``path``) or off (null).
fn traced(trace: *Trace, a: std.mem.Allocator, ctx: *anyopaque, path: ?[]const u8) void {
    trace.* = .{ .journal = null, .a = a, .id = "chatcmpl-abc", .parent = sink(ctx), .reply_log = path };
}

fn tmpPath(a: std.mem.Allocator, tmp: anytype, name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
}

/// The one nonempty line ``path`` holds, parsed.
fn only(a: std.mem.Allocator, path: []const u8) !Value {
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 24));
    var lines = std.mem.splitScalar(u8, text, '\n');
    const line = lines.next() orelse return error.MissingRecord;
    if (lines.next()) |next| if (next.len != 0) return error.ExtraRecord;
    return (try json.parse(a, line)).ok;
}

/// Every nonempty line ``path`` holds, parsed in order.
fn records(a: std.mem.Allocator, path: []const u8) ![]Value {
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 24));
    var out: std.ArrayList(Value) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const parsed = json.parse(a, line) catch return error.BadRecord;
        if (parsed != .ok) return error.BadRecord;
        try out.append(a, parsed.ok);
    }
    return out.items;
}

/// ``path``'s size in bytes; the file must exist.
fn sizeOf(path: []const u8) !u64 {
    return (try std.Io.Dir.cwd().statFile(t.io, path, .{})).size;
}

test "request log replies: off, a whole reply through the trace writes no file" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), null);
    const out = trace.out();
    try t.expect(out.ctx == @as(*anyopaque, @ptrCast(&dummy)));
    out.vt.reply(out.ctx, 200, (try json.parse(a, "{\"choices\":[{\"message\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}")).ok);
    trace.finish();
}

test "request log replies: a whole JSON reply records its id, outcome, usage and timings" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "whole.jsonl");
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), path);
    trace.timed(.{ .queued_s = 0.0012, .ttft_s = @as(?f64, 0.0034), .decode_tps = 56.7, .prompt_tokens = 7, .cached_tokens = 2, .completion_tokens = 3 });
    const whole = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"hello\",\"reasoning_content\":\"because\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}";
    const out = trace.out();
    out.vt.reply(out.ctx, 200, (try json.parse(a, whole)).ok);
    trace.finish();
    const r = try only(a, path);
    try t.expectEqualStrings("reply", r.strField("type").?);
    try t.expectEqualStrings("chatcmpl-abc", r.strField("id").?);
    try t.expectEqual(@as(i64, 200), r.get("status").?.int64().?);
    try t.expect(!r.get("stream").?.bool);
    try t.expectEqualStrings("stop", r.strField("finish_reason").?);
    try t.expectEqualStrings("hello", r.strField("text").?);
    try t.expectEqualStrings("because", r.strField("reasoning").?);
    const usage = r.get("usage").?;
    try t.expectEqual(@as(i64, 7), usage.get("prompt_tokens").?.int64().?);
    try t.expectEqual(@as(i64, 2), usage.get("cached_tokens").?.int64().?);
    try t.expectEqual(@as(i64, 3), usage.get("completion_tokens").?.int64().?);
    const timings = r.get("timings").?;
    try t.expectApproxEqAbs(@as(f64, 1.2), timings.get("queue_ms").?.float, 0.001);
    try t.expectApproxEqAbs(@as(f64, 3.4), timings.get("ttft_ms").?.float, 0.001);
    try t.expectApproxEqAbs(@as(f64, 56.7), timings.get("decode_tps").?.float, 0.001);
    try t.expect(!r.get("truncated").?.bool);
}

test "request log replies: an SSE reply writes exactly one line, after every delta" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "sse.jsonl");
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), path);
    const out = trace.out();
    try out.vt.open(out.ctx, 3);
    try out.vt.event(out.ctx, (try json.parse(a, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"he\"},\"finish_reason\":null}]}")).ok);
    try out.vt.event(out.ctx, (try json.parse(a, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\",\"reasoning_content\":\"think\"},\"finish_reason\":null}]}")).ok);
    const last = "{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":1},\"completion_tokens\":2}}";
    try out.vt.event(out.ctx, (try json.parse(a, last)).ok);
    try out.vt.event(out.ctx, null);
    trace.finish();
    const r = try only(a, path);
    try t.expectEqualStrings("reply", r.strField("type").?);
    try t.expect(r.get("stream").?.bool);
    try t.expectEqualStrings("stop", r.strField("finish_reason").?);
    try t.expectEqualStrings("hello", r.strField("text").?);
    try t.expectEqualStrings("think", r.strField("reasoning").?);
    const usage = r.get("usage").?;
    try t.expectEqual(@as(i64, 5), usage.get("prompt_tokens").?.int64().?);
    try t.expectEqual(@as(i64, 1), usage.get("cached_tokens").?.int64().?);
    try t.expectEqual(@as(i64, 2), usage.get("completion_tokens").?.int64().?);
}

test "request log replies: a legacy completion's choices[].text is recorded, whole and streamed" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var dummy: u8 = 0;
    const whole_path = try tmpPath(a, tmp, "completion.jsonl");
    var whole_trace: Trace = undefined;
    traced(&whole_trace, a, @ptrCast(&dummy), whole_path);
    const whole = "{\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}";
    const out = whole_trace.out();
    out.vt.reply(out.ctx, 200, (try json.parse(a, whole)).ok);
    whole_trace.finish();
    const r = try only(a, whole_path);
    try t.expectEqualStrings("hi", r.strField("text").?);
    try t.expectEqualStrings("stop", r.strField("finish_reason").?);

    const stream_path = try tmpPath(a, tmp, "completion_stream.jsonl");
    var stream_trace: Trace = undefined;
    traced(&stream_trace, a, @ptrCast(&dummy), stream_path);
    const streamed = stream_trace.out();
    try streamed.vt.open(streamed.ctx, 0);
    try streamed.vt.event(streamed.ctx, (try json.parse(a, "{\"choices\":[{\"text\":\"hi\",\"index\":0}]}")).ok);
    try streamed.vt.event(streamed.ctx, null);
    stream_trace.finish();
    const s = try only(a, stream_path);
    try t.expectEqualStrings("hi", s.strField("text").?);
    try t.expect(s.get("stream").?.bool);
}

test "request log replies: a 400 refusal records its status and message, no text" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "refused.jsonl");
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), path);
    const out = trace.out();
    out.vt.reply(out.ctx, 400, (try json.parse(a, "{\"error\":{\"message\":\"model does not exist\",\"type\":\"invalid_request_error\"}}")).ok);
    trace.finish();
    const r = try only(a, path);
    try t.expectEqualStrings("reply", r.strField("type").?);
    try t.expectEqual(@as(i64, 400), r.get("status").?.int64().?);
    try t.expectEqualStrings("model does not exist", r.strField("error").?);
    try t.expect(r.get("text") == null and r.get("usage") == null and r.get("timings") == null);
}

test "request log replies: text past 256 KiB is recorded truncated" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "truncated.jsonl");
    const big = try a.alloc(u8, 200 << 10);
    @memset(big, 'a');
    const text = try std.fmt.allocPrint(a, "{{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{s}\"}},\"finish_reason\":null}}]}}", .{big});
    const chunk = (try json.parse(a, text)).ok;
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), path);
    const out = trace.out();
    try out.vt.open(out.ctx, 0);
    try out.vt.event(out.ctx, chunk);
    try out.vt.event(out.ctx, chunk);
    try out.vt.event(out.ctx, null);
    trace.finish();
    const r = try only(a, path);
    try t.expect(r.get("truncated").?.bool);
    try t.expectEqual(@as(usize, request_trace.text_limit), r.strField("text").?.len);
}

test "request log replies: text truncation keeps a UTF-8 boundary through the trace" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "truncated_utf8.jsonl");
    // "a", then "\u{00e9}" (0xc3 0xa9) pairs, then "b": the byte at text_limit is a continuation byte
    const text = try a.alloc(u8, request_trace.text_limit + 2);
    text[0] = 'a';
    text[text.len - 1] = 'b';
    var i: usize = 1;
    while (i + 1 <= request_trace.text_limit) : (i += 2) {
        text[i] = 0xc3;
        text[i + 1] = 0xa9;
    }
    const delta = try json.newObject(a);
    try delta.put(a, "content", .{ .string = text });
    const choice = try json.newObject(a);
    try choice.put(a, "index", try json.intValue(a, 0));
    try choice.put(a, "delta", .{ .object = delta });
    const payload = try json.newObject(a);
    try payload.put(a, "choices", .{ .array = try a.dupe(Value, &.{.{ .object = choice }}) });
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), path);
    const out = trace.out();
    try out.vt.open(out.ctx, 0);
    try out.vt.event(out.ctx, .{ .object = payload });
    try out.vt.event(out.ctx, null);
    trace.finish();
    const r = try only(a, path);
    try t.expect(r.get("truncated").?.bool);
    const kept = r.strField("text").?;
    try t.expect(kept.len <= request_trace.text_limit);
    try t.expect(std.unicode.utf8ValidateSlice(kept));
    try t.expect(std.mem.indexOf(u8, kept, "\u{fffd}") == null);
}

test "request log replies: the request line keeps its id and the redacted body" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "requests.jsonl");
    const url = try json.newObject(a);
    try url.put(a, "url", .{ .string = "https://example.test/private.png" });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "image_url" });
    try part.put(a, "image_url", .{ .object = url });
    const body = try json.newObject(a);
    const file = (try json.parse(a, "{\"type\": \"file\", \"file\": {\"filename\": \"a.pdf\", \"file_data\": \"JVBERi1zZWNyZXQ=\"}}")).ok;
    try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{ .{ .object = part }, file }) });
    try body.put(a, "cache_salt", .{ .string = "tenant-secret-salt" });
    request_log.appendRequest(a, path, "chatcmpl-abc", .{ .object = body }, request_log.file_limit);
    const r = try only(a, path);
    try t.expectEqualStrings("request", r.strField("type").?);
    try t.expectEqualStrings("chatcmpl-abc", r.strField("id").?);
    const text = try json.stringify(a, r.get("body").?, .{});
    try t.expect(std.mem.indexOf(u8, text, "<redacted>") != null);
    try t.expect(std.mem.indexOf(u8, text, "example.test") == null);
    try t.expect(std.mem.indexOf(u8, text, "tenant-secret-salt") == null and std.mem.indexOf(u8, text, "cache_salt") != null);
    try t.expect(std.mem.indexOf(u8, text, "JVBERi1zZWNyZXQ=") == null and std.mem.indexOf(u8, text, "a.pdf") != null);
}

test "request log replies: a background priority writes no request line" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "background.jsonl");
    const body = (try json.parse(a, "{\"priority\":\"background\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")).ok;
    request_log.appendRequest(a, path, "chatcmpl-abc", body, request_log.file_limit);
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(t.io, path, .{}));
}

test "request log replies: only a background priority is background work" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expect(request_log.isBackground((try json.parse(a, "{\"priority\":\"background\"}")).ok));
    try t.expect(!request_log.isBackground((try json.parse(a, "{\"priority\":\"normal\"}")).ok));
    try t.expect(!request_log.isBackground((try json.parse(a, "{\"priority\":1}")).ok));
    try t.expect(!request_log.isBackground((try json.parse(a, "{\"model\":\"m\"}")).ok));
}

test "request log replies: a record past the record limit is refused and the file stays whole" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "record_limit.jsonl");
    const body = (try json.parse(a, "{\"model\":\"m\"}")).ok;
    const huge = try a.alloc(u8, request_log.record_limit + 1024);
    @memset(huge, 'r');
    const oversized = try json.newObject(a);
    try oversized.put(a, "text", .{ .string = huge });
    request_log.appendRequest(a, path, "chatcmpl-before", body, request_log.file_limit);
    request_log.appendRequest(a, path, "chatcmpl-huge", .{ .object = oversized }, request_log.file_limit);
    request_log.appendRequest(a, path, "chatcmpl-after", body, request_log.file_limit);
    const got = try records(a, path);
    try t.expectEqual(@as(usize, 2), got.len);
    try t.expectEqualStrings("chatcmpl-before", got[0].strField("id").?);
    try t.expectEqualStrings("chatcmpl-after", got[1].strField("id").?);
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 16));
    try t.expectEqual(@as(u8, '\n'), text[text.len - 1]);
    try t.expect(std.mem.indexOf(u8, text, "chatcmpl-huge") == null);
    // a refused first record leaves no file behind: the bound is checked before the open
    const fresh = try tmpPath(a, tmp, "refused_first.jsonl");
    request_log.appendRequest(a, fresh, "chatcmpl-huge", .{ .object = oversized }, request_log.file_limit);
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(t.io, fresh, .{}));
}

test "request log replies: the file rotates at its size limit and keeps the newest records" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "rotation.jsonl");
    const older = try std.fmt.allocPrint(a, "{s}.1", .{path});
    const limit: u64 = 256;
    for (0..12) |n| {
        const body = try json.newObject(a);
        try body.put(a, "n", try json.intValue(a, @as(i64, @intCast(n))));
        try body.put(a, "pad", .{ .string = "pad-pad-pad-pad" });
        request_log.append(a, path, .{ .object = body }, limit);
    }
    const current = try records(a, path);
    const previous = try records(a, older);
    try t.expect(current.len > 0 and previous.len > 0);
    try t.expectEqual(@as(i64, 11), current[current.len - 1].get("n").?.int64().?);
    try t.expectEqual(@as(i64, 0), previous[0].get("n").?.int64().?);
    try t.expect((try sizeOf(path)) <= limit);
    try t.expect((try sizeOf(older)) <= limit);
    try t.expect((try sizeOf(path)) + (try sizeOf(older)) <= @as(u64, request_log.files) * limit);
}

test "request log replies: an oversized archive is cut to the limit" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "oversized_archive.jsonl");
    const older = try std.fmt.allocPrint(a, "{s}.1", .{path});
    const body = (try json.parse(a, "{\"model\":\"fixture\"}")).ok;
    for (0..40) |_| request_log.appendRequest(a, path, "chatcmpl-abc", body, 1024);
    try t.expect((try sizeOf(path)) > 512);
    request_log.appendRequest(a, path, "chatcmpl-newest", body, 256);
    const current = try records(a, path);
    try t.expectEqual(@as(usize, 1), current.len);
    try t.expect((try sizeOf(older)) <= 256);
    try t.expect((try sizeOf(path)) + (try sizeOf(older)) <= 2 * 256);
    // the cut lands on a record boundary: every archived line parses, is a whole request, and the archive ends on '\n'
    const archived = try records(a, older);
    try t.expect(archived.len > 0);
    for (archived) |r| {
        try t.expectEqualStrings("request", r.strField("type") orelse return error.BadRecord);
        try t.expectEqualStrings("chatcmpl-abc", r.strField("id") orelse return error.BadRecord);
    }
    const archived_text = try std.Io.Dir.cwd().readFileAlloc(t.io, older, a, .limited(1 << 16));
    try t.expectEqual(@as(u8, '\n'), archived_text[archived_text.len - 1]);
    // the active file holds the one record written after the cut, so archive then active reads oldest to newest
    try t.expectEqualStrings("chatcmpl-newest", current[0].strField("id") orelse return error.BadRecord);
    // the cut is maximal: whole records only, and one more record would pass the limit
    const archived_size = try sizeOf(older);
    const first_line = std.mem.indexOfScalar(u8, archived_text, '\n') orelse return error.BadRecord;
    const record = @as(u64, @intCast(first_line + 1)); // one whole archived record, its newline included
    try t.expect(archived_size % record == 0);
    try t.expect(archived_size + record > 256);
    try t.expectEqual(archived_size / record, @as(u64, @intCast(archived.len)));
}

test "request log replies: rotation keeps whole records under concurrent writers" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    const log_level = t.log_level;
    t.log_level = .err;
    defer t.log_level = log_level;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(t.io, a);
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}/concurrent.jsonl", .{ cwd, tmp.sub_path });
    const older = try std.fmt.allocPrint(a, "{s}.1", .{path});
    const limit: u64 = 512;
    const threads = 8;
    const lines = 8;
    const Worker = struct {
        fn run(p: []const u8, lim: u64, k: usize) void {
            var local: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
            defer local.deinit();
            for (0..lines) |i| {
                const body = json.newObject(local.allocator()) catch return;
                body.put(local.allocator(), "w", json.intValue(local.allocator(), k) catch return) catch return;
                body.put(local.allocator(), "i", json.intValue(local.allocator(), i) catch return) catch return;
                body.put(local.allocator(), "pad", .{ .string = "pad-pad-pad-pad-pad-pad-pad-pad-pad-pad-pad-pad-pad-pad" }) catch return;
                const id = std.fmt.allocPrint(local.allocator(), "r{d}-{d}", .{ k, i }) catch return;
                request_log.appendRequest(local.allocator(), p, id, .{ .object = body }, lim);
            }
        }
    };
    var workers: [threads]std.Thread = undefined;
    for (&workers, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, Worker.run, .{ path, limit, k });
    for (workers) |w| w.join();
    // rotation may retire the oldest records; every line that survives is whole and parses
    var total: u64 = 0;
    for ([_][]const u8{ path, older }) |file| {
        const size = try sizeOf(file);
        try t.expect(size > 0 and size <= limit);
        total += size;
        const text = try std.Io.Dir.cwd().readFileAlloc(t.io, file, a, .limited(@intCast(limit)));
        try t.expectEqual(@as(u8, '\n'), text[text.len - 1]);
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            const parsed = json.parse(a, line) catch return error.BadRecord;
            if (parsed != .ok) return error.BadRecord;
            try t.expectEqualStrings("request", parsed.ok.strField("type") orelse return error.BadRecord);
        }
    }
    try t.expect(total <= 2 * limit);
}

test "request log replies: a refusal's message is kept to the error limit" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const msg = try a.alloc(u8, 4 * request_log.error_limit);
    @memset(msg, 'e');
    const refused_path = try tmpPath(a, tmp, "refused_error.jsonl");
    request_log.appendReply(a, refused_path, .{ .id = "chatcmpl-abc", .status = 400, .stream = false, .error_message = msg }, request_log.file_limit);
    const r = try only(a, refused_path);
    try t.expectEqual(@as(usize, request_log.error_limit), r.strField("error").?.len);
    // a 200 that still carries an error message is cut the same way
    const errored_path = try tmpPath(a, tmp, "error.jsonl");
    request_log.appendReply(a, errored_path, .{ .id = "chatcmpl-def", .status = 200, .stream = false, .error_message = msg }, request_log.file_limit);
    const e = try only(a, errored_path);
    try t.expectEqual(@as(usize, request_log.error_limit), e.strField("error").?.len);
}

test "request log replies: a refusal's message is cut on a UTF-8 boundary" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "refused_utf8.jsonl");
    // "a", then "\u{00e9}" (0xc3 0xa9) pairs, then "b": the byte at error_limit is a continuation byte
    const msg = try a.alloc(u8, request_log.error_limit + 2);
    @memset(msg, 'b');
    msg[0] = 'a';
    var i: usize = 1;
    while (i + 1 < request_log.error_limit + 2) : (i += 2) {
        msg[i] = 0xc3;
        msg[i + 1] = 0xa9;
    }
    request_log.appendReply(a, path, .{ .id = "chatcmpl-abc", .status = 400, .stream = false, .error_message = msg }, request_log.file_limit);
    const got = (try only(a, path)).strField("error").?;
    try t.expect(std.unicode.utf8ValidateSlice(got));
    try t.expect(got.len <= request_log.error_limit);
}

test "request log replies: a refusal message through the trace is cut on a UTF-8 boundary" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "trace_utf8.jsonl");
    // "a", then 2048 "\u{00e9}" pairs, then "b": the byte at error_limit is a continuation byte
    const msg = try a.alloc(u8, request_log.error_limit + 2);
    msg[0] = 'a';
    msg[request_log.error_limit + 1] = 'b';
    var i: usize = 1;
    while (i + 1 <= request_log.error_limit) : (i += 2) {
        msg[i] = 0xc3;
        msg[i + 1] = 0xa9;
    }
    const error_obj = try json.newObject(a);
    try error_obj.put(a, "message", .{ .string = msg });
    const value = try json.newObject(a);
    try value.put(a, "error", .{ .object = error_obj });
    var dummy: u8 = 0;
    var trace: Trace = undefined;
    traced(&trace, a, @ptrCast(&dummy), path);
    const out = trace.out();
    out.vt.reply(out.ctx, 400, .{ .object = value });
    trace.finish();
    const got = (try only(a, path)).strField("error").?;
    try t.expect(std.unicode.utf8ValidateSlice(got));
    try t.expect(got.len <= request_log.error_limit);
    try t.expect(std.mem.indexOf(u8, got, "\u{fffd}") == null);
}

test "request log replies: a lock file others can read gets no records" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the reply log stays off on Windows
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "lock.jsonl");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "lock.jsonl.lock", .data = "" });
    try tmp.dir.setFilePermissions(t.io, "lock.jsonl.lock", .fromMode(0o644), .{}); // set after creating: a umask clears bits
    const body = (try json.parse(a, "{\"model\":\"m\"}")).ok;
    request_log.appendRequest(a, path, "chatcmpl-abc", body, request_log.file_limit);
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(t.io, path, .{}));
}

test "request log replies: Windows refuses the file rather than accept inherited access" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpPath(a, tmp, "windows.jsonl");
    const body = (try json.parse(a, "{\"model\":\"m\"}")).ok;
    request_log.appendRequest(a, path, "chatcmpl-abc", body, request_log.file_limit);
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(t.io, path, .{}));
}
