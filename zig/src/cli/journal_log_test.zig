//! `tensorfold log` over a fixture journal: rotation, a torn last line, in-flight, quarantined and too-large replies.
const std = @import("std");
const log = @import("journal_log.zig");
const t = std.testing;
const ab64: [64]u8 = @splat('a');
const cd64: [64]u8 = @splat('c');
const ef64: [64]u8 = @splat('e');
const Allocator = std.mem.Allocator;

/// A reply's bytes as the journal keeps them: base64, standard alphabet.
fn b64(a: Allocator, bytes: []const u8) ![]const u8 {
    const codec = std.base64.standard.Encoder;
    return codec.encode(try a.alloc(u8, codec.calcSize(bytes.len)), bytes);
}

const whole_reply = "{\"id\":\"chatcmpl-a\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"Paris.\"," ++
    "\"reasoning_content\":\"The capital.\"},\"finish_reason\":\"stop\"}]}";
const stream_reply = "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Look \"}}]}\n\n" ++
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"It is \"}}]}\n\n" ++
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"get\",\"arguments\":\"{\\\"q\\\"\"}}]}}]}\n\n" ++
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"sunny.\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]}," ++
    "\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

fn start(a: Allocator, id: []const u8, at: i64, body: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"version\":1,\"kind\":\"start\",\"id\":\"{s}\",\"at\":{d},\"path\":\"/v1/chat/completions\",\"fingerprint\":\"{s}\",\"seed\":null,\"body\":{s}}}\n", .{ id, at, &ab64, body });
}

fn finish(a: Allocator, id: []const u8, at: i64, stream: bool, reply: ?[]const u8, bytes: usize, done: []const u8) ![]const u8 {
    const kept = if (reply) |r| try std.fmt.allocPrint(a, ",\"reply_base64\":\"{s}\"", .{try b64(a, r)}) else "";
    return std.fmt.allocPrint(a, "{{\"version\":1,\"kind\":\"finish\",\"id\":\"{s}\",\"at\":{d},\"status\":200,\"stream\":{s},\"outcome\":\"completed\",\"reply_bytes\":{d},\"reply_sha256\":\"{s}\",\"token_sha256\":\"{s}\",{s}{s}}}\n", .{ id, at, if (stream) "true" else "false", bytes, &cd64, &ef64, done, kept });
}

const timings = "\"queue_ms\":1.5,\"ttft_ms\":35.2,\"decode_tps\":41,\"prompt_tokens\":12,\"cached_tokens\":4,\"completion_tokens\":7";
const ask = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"stream\":false}";
const ask_stream = "{\"model\":\"m2\",\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Weather?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"<redacted>\"}}]}],\"stream\":true}";

/// PATH.1: a reply and a crash's open start; PATH: its quarantine, a stream, in flight, too large, a torn line.
fn fixture(a: Allocator, tmp: t.TmpDir) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    const older = try std.mem.concat(a, u8, &.{
        try start(a, "chatcmpl-a", 1795136000, ask),
        "{\"version\":1,\"kind\":\"prepared\",\"id\":\"chatcmpl-a\",\"at\":1795136000,\"seed\":5,\"prompt_tokens\":12}\n",
        try finish(a, "chatcmpl-a", 1795136001, false, whole_reply, whole_reply.len, timings),
        try start(a, "chatcmpl-q", 1795136005, ask),
    });
    const newer = try std.mem.concat(a, u8, &.{
        "{\"version\":1,\"kind\":\"quarantine\",\"id\":\"chatcmpl-q\",\"at\":1795136009,\"fingerprint\":\"" ++ ab64 ++ "\"}\n",
        "{\"version\":1,\"kind\":\"stop\",\"at\":1795136009}\n",
        try start(a, "chatcmpl-b", 1795136010, ask_stream),
        try finish(a, "chatcmpl-b", 1795136011, true, stream_reply, stream_reply.len, timings),
        try start(a, "chatcmpl-c", 1795136020, ask),
        try start(a, "chatcmpl-d", 1795136030, ask),
        try finish(a, "chatcmpl-d", 1795136031, false, null, 9_000_000, "\"queue_ms\":0,\"ttft_ms\":null,\"decode_tps\":0,\"prompt_tokens\":12,\"cached_tokens\":0,\"completion_tokens\":2048"),
        "{\"version\":1,\"kind\":\"start\",\"id\":\"chatcmpl-torn\",\"at\":179",
    });
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.1", .{path}), .data = older });
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = newer });
    return path;
}

fn runLog(a: Allocator, o: log.Options) !struct { u8, []const u8, []const u8 } {
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const code = try log.runWith(a, t.io, &out.writer, &err.writer, null, o);
    return .{ code, out.written(), err.written() };
}

test "log: arguments name the command, the request and the filters, and refuse what they don't take" {
    const l = log.parse(&.{}).?;
    try t.expect(l.command == .list and l.path == null and !l.json);
    const s = log.parse(&.{ "show", "chatcmpl-a", "j.jsonl", "--json" }).?;
    try t.expect(s.command == .show and s.json);
    try t.expectEqualStrings("chatcmpl-a", s.id.?);
    try t.expectEqualStrings("j.jsonl", s.path.?);
    const f = log.parse(&.{ "list", "--last", "3", "--route", "chat", "--model", "m" }).?;
    try t.expect(f.last.? == 3);
    try t.expectEqualStrings("m", f.model.?);
    try t.expect(log.parse(&.{"show"}) == null and log.parse(&.{ "list", "--last", "x" }) == null);
    try t.expect(log.parse(&.{ "follow", "--last", "2" }) == null and log.parse(&.{ "a", "b" }) == null);
}

test "log: list reads PATH.1 then PATH, in flight and quarantined say so, and a torn last line is left out" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(a, tmp);
    const code, const out, const err = try runLog(a, .{ .path = path });
    try t.expectEqual(@as(u8, 0), code);
    try t.expectEqualStrings("", err);
    try t.expectEqualStrings(
        \\2026-11-20 00:53:20  chatcmpl-a  chat  m  200 completed  prompt 12  out 7  ttft 35.2 ms  41.0 tok/s
        \\2026-11-20 00:53:25  chatcmpl-q  chat  m  quarantined (in flight at a crash)  prompt -  out -  ttft -  - tok/s
        \\2026-11-20 00:53:30  chatcmpl-b  chat  m2  200 completed  prompt 12  out 7  ttft 35.2 ms  41.0 tok/s
        \\2026-11-20 00:53:40  chatcmpl-c  chat  m  in flight  prompt -  out -  ttft -  - tok/s
        \\2026-11-20 00:53:50  chatcmpl-d  chat  m  200 completed  prompt 12  out 2048  ttft -  0.0 tok/s
        \\
    , out);
    _, const last, _ = try runLog(a, .{ .path = path, .last = 2, .model = "m" });
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, last, "\n"));
    try t.expect(std.mem.indexOf(u8, last, "chatcmpl-c") != null and std.mem.indexOf(u8, last, "chatcmpl-d") != null);
    _, const routed, _ = try runLog(a, .{ .path = path, .route = "completions" });
    try t.expectEqualStrings("", routed);
}

test "log: show prints the request as journaled and the reply decoded, whole, streamed and too large" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(a, tmp);
    _, const whole, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-a", .path = path });
    try t.expectEqualStrings(
        \\id        chatcmpl-a
        \\started   2026-11-20 00:53:20 UTC
        \\route     /v1/chat/completions
        \\model     m
        \\status    200 completed
        \\tokens    prompt 12, cached 4, completion 7
        \\timings   queue 1.5 ms, first token 35.2 ms, decode 41.0 tok/s
        \\
        \\request
        \\  [user] Capital of France?
        \\
        \\reply
        \\  reasoning: The capital.
        \\  text: Paris.
        \\  finish reason: stop
        \\
    , whole);
    _, const streamed, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-b", .path = path });
    for ([_][]const u8{ "status    200 completed (stream)", "  [system] Be brief.", "  [user] Weather? {\"type\":\"image_url\",\"image_url\":{\"url\":\"<redacted>\"}}", "  reasoning: Look ", "  text: It is sunny.", "  tool call: get({\"q\":1})", "  finish reason: tool_calls" }) |want| {
        if (std.mem.indexOf(u8, streamed, want) == null) {
            std.debug.print("missing {s} in:\n{s}\n", .{ want, streamed });
            return error.TestUnexpectedResult;
        }
    }
    _, const large, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-d", .path = path });
    try t.expect(std.mem.indexOf(u8, large, "  not kept: 9000000 bytes, sha256 " ++ cd64) != null);
    _, const flying, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-c", .path = path });
    try t.expect(std.mem.indexOf(u8, flying, "status    in flight (no finish yet)") != null and std.mem.indexOf(u8, flying, "  (none yet)") != null);
    _, const crashed, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-q", .path = path });
    try t.expect(std.mem.indexOf(u8, crashed, "status    quarantined: it was in flight when the server crashed") != null);
    const code, _, const missing = try runLog(a, .{ .command = .show, .id = "chatcmpl-torn", .path = path });
    try t.expectEqual(@as(u8, 1), code);
    try t.expect(std.mem.indexOf(u8, missing, "no request chatcmpl-torn") != null);
}

test "log: --json prints each request's records as one parseable object a line" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(a, tmp);
    _, const out, _ = try runLog(a, .{ .path = path, .json = true });
    var it = std.mem.splitScalar(u8, std.mem.trimEnd(u8, out, "\n"), '\n');
    var ids: std.ArrayList([]const u8) = .empty;
    while (it.next()) |line| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{});
        try ids.append(a, v.object.get("id").?.string);
        if (std.mem.eql(u8, ids.items[ids.items.len - 1], "chatcmpl-a")) {
            try t.expectEqual(@as(i64, 7), v.object.get("finish").?.object.get("completion_tokens").?.integer);
            try t.expectEqual(@as(i64, 5), v.object.get("prepared").?.object.get("seed").?.integer);
        }
        if (std.mem.eql(u8, ids.items[ids.items.len - 1], "chatcmpl-q")) try t.expect(v.object.get("quarantined").?.bool);
    }
    try t.expectEqual(@as(usize, 5), ids.items.len);
}

test "log: no path and no TENSORFOLD_REQUEST_JOURNAL is refused, and a missing journal says so" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const code, _, const err = try runLog(a, .{});
    try t.expectEqual(@as(u8, 2), code);
    try t.expect(std.mem.indexOf(u8, err, "set TENSORFOLD_REQUEST_JOURNAL") != null);
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("TENSORFOLD_REQUEST_JOURNAL", ".zig-cache/tmp/no-such-journal.jsonl");
    var out: std.Io.Writer.Allocating = .init(a);
    var e: std.Io.Writer.Allocating = .init(a);
    try t.expectEqual(@as(u8, 1), try log.runWith(a, t.io, &out.writer, &e.writer, &env, .{}));
    try t.expect(std.mem.indexOf(u8, e.written(), "no journal at .zig-cache/tmp/no-such-journal.jsonl") != null);
}

/// A server's appends while `follow` runs: the in-flight request finishes, the file rotates, a new request lands.
const Writer = struct {
    path: []const u8,
    fn run(w: *const Writer) void {
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator); // its own: the reader keeps its arena
        defer arena.deinit();
        w.go(arena.allocator()) catch |e| std.debug.print("fixture writer: {t}\n", .{e});
    }
    fn go(w: *const Writer, a: Allocator) !void {
        const io = t.io;
        std.Io.sleep(io, .fromMilliseconds(120), .awake) catch {};
        const f = try std.Io.Dir.cwd().openFile(io, w.path, .{ .mode = .read_write });
        const end = (try f.stat(io)).size;
        const done = try finish(a, "chatcmpl-c", 1795136040, false, whole_reply, whole_reply.len, timings);
        try f.writePositionalAll(io, done[0 .. done.len / 2], end); // half a line first: follow waits for the rest
        std.Io.sleep(io, .fromMilliseconds(80), .awake) catch {};
        try f.writePositionalAll(io, done[done.len / 2 ..], end + done.len / 2);
        f.close(io);
        std.Io.sleep(io, .fromMilliseconds(120), .awake) catch {};
        const dir = std.Io.Dir.cwd();
        try dir.rename(try std.fmt.allocPrint(a, "{s}.1", .{w.path}), dir, try std.fmt.allocPrint(a, "{s}.2", .{w.path}), io);
        try dir.rename(w.path, dir, try std.fmt.allocPrint(a, "{s}.1", .{w.path}), io);
        try dir.writeFile(io, .{ .sub_path = w.path, .data = try std.mem.concat(a, u8, &.{ try start(a, "chatcmpl-e", 1795136050, ask), try finish(a, "chatcmpl-e", 1795136051, false, whole_reply, whole_reply.len, timings) }) });
    }
};

test "log: follow prints each newly finished request once its line is whole, across a rotation" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(a, tmp);
    // the torn line is a crash's leftover; a live server's file ends on a whole line
    const cut = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = cut[0 .. std.mem.lastIndexOfScalar(u8, cut, '\n').? + 1] });
    const w: Writer = .{ .path = path };
    const thread = try std.Thread.spawn(.{}, Writer.run, .{&w});
    _, const out, _ = try runLog(a, .{ .command = .follow, .path = path, .poll_ms = 20, .follow_limit = 2 });
    thread.join();
    try t.expectEqualStrings(
        \\2026-11-20 00:53:40  chatcmpl-c  chat  m  200 completed  prompt 12  out 7  ttft 35.2 ms  41.0 tok/s
        \\2026-11-20 00:54:10  chatcmpl-e  chat  m  200 completed  prompt 12  out 7  ttft 35.2 ms  41.0 tok/s
        \\
    , out);
}
