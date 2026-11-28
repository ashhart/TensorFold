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

/// The path the reader would take; `null` for the name it must refuse to give. No unwrap, so a miss fails as a test.
fn pathIs(o: log.Options, env: ?*const std.process.Environ.Map, want: ?[]const u8) !void {
    const got = log.journalPath(o, env) orelse return t.expect(want == null);
    const w = want orelse return error.TestUnexpectedResult;
    return t.expectEqualStrings(w, got);
}

test "log: the request log's own path answers when no journal is set, and the journal wins over it" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var only_log: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try only_log.put("TENSORFOLD_REQUEST_LOG", "requests.jsonl");
    try pathIs(.{}, &only_log, "requests.jsonl"); // the fallback the writer's own variable names
    try pathIs(.{}, null, null); // no environment at all is still refused
    var both: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try both.put("TENSORFOLD_REQUEST_JOURNAL", "journal.jsonl");
    try both.put("TENSORFOLD_REQUEST_LOG", "requests.jsonl");
    try pathIs(.{}, &both, "journal.jsonl");
    var blank: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try blank.put("TENSORFOLD_REQUEST_JOURNAL", "");
    try blank.put("TENSORFOLD_REQUEST_LOG", "requests.jsonl");
    try pathIs(.{}, &blank, "requests.jsonl"); // an empty journal still falls through
    var none: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try pathIs(.{ .path = "given.jsonl" }, &none, "given.jsonl"); // the argument wins
    // and the refusal names only the files the source that was chosen keeps
    var j: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try j.put("TENSORFOLD_REQUEST_JOURNAL", ".zig-cache/tmp/no-such-journal.jsonl");
    var r: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try r.put("TENSORFOLD_REQUEST_LOG", ".zig-cache/tmp/no-such-log.jsonl");
    inline for ([_]struct { e: std.process.Environ.Map, want: []const u8 }{
        .{ .e = j, .want = "tensorfold log: no journal at .zig-cache/tmp/no-such-journal.jsonl (nor .zig-cache/tmp/no-such-journal.jsonl.1 to .zig-cache/tmp/no-such-journal.jsonl.3)\n" },
        .{ .e = r, .want = "tensorfold log: no journal at .zig-cache/tmp/no-such-log.jsonl (nor .zig-cache/tmp/no-such-log.jsonl.1)\n" },
    }) |c| {
        var o: std.Io.Writer.Allocating = .init(a);
        var e: std.Io.Writer.Allocating = .init(a);
        try t.expectEqual(@as(u8, 1), try log.runWith(a, t.io, &o.writer, &e.writer, &c.e, .{}));
        try t.expectEqualStrings(c.want, e.written());
    }
}

/// A server's appends while `follow` runs: two writes to the followed file, then the rotation and its new file.
const Lines = *const fn (a: Allocator) []const u8;

const Writer = struct {
    path: []const u8,
    first: Lines = &journalHalf,
    second: Lines = &journalRest,
    after: Lines = &journalPair,
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
        const one = w.first(a);
        try f.writePositionalAll(io, one, end); // a line split in two writes: `follow` waits for the rest of it
        std.Io.sleep(io, .fromMilliseconds(80), .awake) catch {};
        try f.writePositionalAll(io, w.second(a), end + one.len);
        f.close(io);
        std.Io.sleep(io, .fromMilliseconds(120), .awake) catch {};
        const dir = std.Io.Dir.cwd();
        // only the journal has an archive to move up; a request-log file has just `PATH`
        dir.rename(try std.fmt.allocPrint(a, "{s}.1", .{w.path}), dir, try std.fmt.allocPrint(a, "{s}.2", .{w.path}), io) catch {};
        try dir.rename(w.path, dir, try std.fmt.allocPrint(a, "{s}.1", .{w.path}), io);
        try dir.writeFile(io, .{ .sub_path = w.path, .data = w.after(a) });
    }
};

/// The journal's own appends: `chatcmpl-c`'s finish in two writes, then a whole pair past the rotation.
fn journalHalf(a: Allocator) []const u8 {
    const done = finish(a, "chatcmpl-c", 1795136040, false, whole_reply, whole_reply.len, timings) catch return "";
    return done[0 .. done.len / 2];
}

fn journalRest(a: Allocator) []const u8 {
    const done = finish(a, "chatcmpl-c", 1795136040, false, whole_reply, whole_reply.len, timings) catch return "";
    return done[done.len / 2 ..];
}

fn journalPair(a: Allocator) []const u8 {
    const s = start(a, "chatcmpl-e", 1795136050, ask) catch return "";
    const f = finish(a, "chatcmpl-e", 1795136051, false, whole_reply, whole_reply.len, timings) catch return "";
    return std.mem.concat(a, u8, &.{ s, f }) catch return "";
}

/// The request log's, one pair a write, both records whole together.
fn logPair9(a: Allocator) []const u8 {
    return pair(a, "chatcmpl-9", "m9");
}

fn logPair10(a: Allocator) []const u8 {
    return pair(a, "chatcmpl-10", "m10");
}

fn nothing(a: Allocator) []const u8 {
    _ = a;
    return "";
}

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

/// The null stamp the request log's records get: its writer keeps no time.
const no_stamp = "-------------------";

/// A request-log file with replies on: two pairs, then a refused one and a lone reply.
fn repliesOn(a: Allocator, tmp: t.TmpDir) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    const data = try std.mem.concat(a, u8, &.{
        "{\"type\":\"request\",\"id\":\"chatcmpl-1\",\"body\":{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"stream\":false}}\n",
        "{\"type\":\"reply\",\"id\":\"chatcmpl-1\",\"status\":200,\"stream\":false,\"finish_reason\":\"stop\",\"text\":\"Paris.\",\"reasoning\":\"The capital.\",\"usage\":{\"prompt_tokens\":12,\"cached_tokens\":4,\"completion_tokens\":7},\"timings\":{\"queue_ms\":1.5,\"ttft_ms\":35.2,\"decode_tps\":41},\"truncated\":false}\n",
        "{\"type\":\"request\",\"id\":\"chatcmpl-2\",\"body\":{\"model\":\"m2\",\"prompt\":\"Once upon\",\"stream\":false}}\n",
        "{\"type\":\"reply\",\"id\":\"chatcmpl-2\",\"status\":400,\"error\":\"max_tokens must be positive\"}\n",
        "{\"type\":\"request\",\"id\":\"chatcmpl-3\",\"body\":{\"model\":\"m3\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}],\"stream\":true}}\n",
        "{\"type\":\"reply\",\"id\":\"chatcmpl-3\",\"status\":200,\"stream\":true,\"finish_reason\":\"tool_calls\",\"text\":\"It is sunny.\",\"reasoning\":null,\"usage\":{\"prompt_tokens\":9,\"cached_tokens\":0,\"completion_tokens\":3},\"timings\":{\"queue_ms\":0.4,\"ttft_ms\":8.1,\"decode_tps\":52.5},\"truncated\":true}\n",
        "{\"type\":\"reply\",\"id\":\"chatcmpl-4\",\"status\":200,\"stream\":false,\"finish_reason\":\"stop\",\"text\":\"Four.\",\"usage\":{\"prompt_tokens\":1,\"cached_tokens\":0,\"completion_tokens\":1},\"timings\":{\"queue_ms\":0.1,\"ttft_ms\":2.5,\"decode_tps\":20},\"truncated\":false}\n",
    });
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = data });
    return path;
}

test "log: the request log's own records list with the reply beside each request" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try repliesOn(a, tmp);
    const code, const out, const err = try runLog(a, .{ .path = path });
    try t.expectEqual(@as(u8, 0), code);
    try t.expectEqualStrings("", err);
    try t.expectEqualStrings(
        \\-------------------  chatcmpl-1  chat  m  200 finished  prompt 12  out 7  ttft 35.2 ms  41.0 tok/s
        \\-------------------  chatcmpl-2  completions  m2  400 finished  prompt -  out -  ttft -  - tok/s
        \\-------------------  chatcmpl-3  chat  m3  200 finished  prompt 9  out 3  ttft 8.1 ms  52.5 tok/s
        \\-------------------  chatcmpl-4  -  -  200 finished  prompt 1  out 1  ttft 2.5 ms  20.0 tok/s
        \\
    , out);
    // the filters see the derived route and the body's model, so a completion alone answers
    _, const routed, _ = try runLog(a, .{ .path = path, .route = "completions" });
    try t.expectEqualStrings(no_stamp ++ "  chatcmpl-2  completions  m2  400 finished  prompt -  out -  ttft -  - tok/s\n", routed);
    _, const last, _ = try runLog(a, .{ .path = path, .last = 1, .model = "m3" });
    try t.expectEqualStrings(no_stamp ++ "  chatcmpl-3  chat  m3  200 finished  prompt 9  out 3  ttft 8.1 ms  52.5 tok/s\n", last);
}

test "log: show of a request-log record prints its body and the reply the record itself holds" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try repliesOn(a, tmp);
    _, const whole, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-1", .path = path });
    try t.expectEqualStrings(
        \\id        chatcmpl-1
        \\started   ------------------- UTC
        \\route     chat
        \\model     m
        \\status    200 finished
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
    _, const refused, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-2", .path = path });
    try t.expectEqualStrings(
        \\id        chatcmpl-2
        \\started   ------------------- UTC
        \\route     completions
        \\model     m2
        \\status    400 finished
        \\tokens    prompt -, cached -, completion -
        \\timings   queue -, first token -, decode -
        \\
        \\request
        \\  [prompt] Once upon
        \\
        \\reply
        \\  error: max_tokens must be positive
        \\
    , refused);
    _, const streamed, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-3", .path = path });
    for ([_][]const u8{ "status    200 finished (stream)", "  text: It is sunny.", "  finish reason: tool_calls", "  (its text was cut at the record's limit)" }) |want| {
        if (std.mem.indexOf(u8, streamed, want) == null) {
            std.debug.print("missing {s} in:\n{s}\n", .{ want, streamed });
            return error.TestUnexpectedResult;
        }
    }
    _, const lone, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-4", .path = path });
    try t.expect(std.mem.indexOf(u8, lone, "  (its start record is not in the kept files)") != null);
}

test "log: --json over the request log lifts each reply's nested measures beside its own fields" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try repliesOn(a, tmp);
    _, const out, _ = try runLog(a, .{ .path = path, .json = true });
    var it = std.mem.splitScalar(u8, std.mem.trimEnd(u8, out, "\n"), '\n');
    var n: usize = 0;
    while (it.next()) |line| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{});
        n += 1;
        if (n != 1) continue;
        try t.expectEqualStrings("chatcmpl-1", v.object.get("id").?.string);
        const f = v.object.get("finish").?;
        try t.expectEqual(@as(i64, 7), f.object.get("completion_tokens").?.integer);
        try t.expectEqual(@as(i64, 12), f.object.get("prompt_tokens").?.integer);
        try t.expectEqualStrings("stop", f.object.get("finish_reason").?.string);
        try t.expect(f.object.get("usage").?.object.get("cached_tokens").?.integer == 4); // its own nesting is kept too
        try t.expect(v.object.get("start").?.object.get("body").?.object.get("model").?.string.len == 1);
    }
    try t.expectEqual(@as(usize, 4), n);
}

/// A request log with replies off, including bodies with client-supplied fields that resemble log metadata.
fn repliesOff(a: Allocator, tmp: t.TmpDir) ![]const u8 {
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data =
        \\{"id":"shared","type":"reply","status":400,"error":"client field","model":"m","messages":[{"role":"user","content":"Capital of France?"}]}
        \\{"id":"shared","type":"request","body":{"prompt":"extension"},"model":"m2","prompt":"Once upon"}
        \\{"kind":"finish","version":1,"at":1,"id":"3","model":"m","messages":[{"role":"user","content":[{"type":"text","text":"Weather?"},{"type":"image_url","image_url":{"url":"<redacted>"}}]}],"stream":true}
        \\{"model":"m","kind":"stop","type":"reply"}
        \\{oops}
        \\
    });
    return path;
}

test "log: with replies off each body is numbered by its order, and its shape names the route" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try repliesOff(a, tmp);
    const code, const out, const err = try runLog(a, .{ .path = path });
    try t.expectEqual(@as(u8, 0), code);
    try t.expect(std.mem.startsWith(u8, err, "tensorfold log: 1 unreadable record"));
    try t.expectEqualStrings(
        \\-------------------  1  chat  m  in flight  prompt -  out -  ttft -  - tok/s
        \\-------------------  2  completions  m2  in flight  prompt -  out -  ttft -  - tok/s
        \\-------------------  3  chat  m  in flight  prompt -  out -  ttft -  - tok/s
        \\-------------------  4  -  m  in flight  prompt -  out -  ttft -  - tok/s
        \\
    , out);
    _, const second, _ = try runLog(a, .{ .command = .show, .id = "2", .path = path });
    try t.expect(std.mem.indexOf(u8, second, "  [prompt] Once upon\n") != null);
    _, const one, _ = try runLog(a, .{ .command = .show, .id = "3", .path = path });
    try t.expect(std.mem.indexOf(u8, one, "  [user] Weather? {\"type\":\"image_url\",\"image_url\":{\"url\":\"<redacted>\"}}") != null);
    _, const plain, _ = try runLog(a, .{ .command = .show, .id = "4", .path = path });
    try t.expect(std.mem.indexOf(u8, plain, "  {\"model\":\"m\",\"kind\":\"stop\",\"type\":\"reply\"}\n") != null and std.mem.indexOf(u8, plain, "  (none yet)") != null);
    _, const routed, _ = try runLog(a, .{ .path = path, .route = "completions" });
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, routed, "\n"));
    try t.expect(std.mem.indexOf(u8, routed, "  2  completions  m2  ") != null);
}

/// One request record and its reply record, as `TENSORFOLD_REQUEST_LOG_REPLIES=1` writes them.
fn pair(a: Allocator, id: []const u8, model: []const u8) []const u8 {
    var out: std.ArrayList(u8) = .empty;
    out.print(a, "{{\"type\":\"request\",\"id\":\"{s}\",\"body\":{{\"model\":\"{s}\"}}}}\n", .{ id, model }) catch return "";
    out.print(a, "{{\"type\":\"reply\",\"id\":\"{s}\",\"status\":200,\"stream\":false,\"finish_reason\":\"stop\",\"text\":\"Nine.\",\"usage\":{{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2}},\"timings\":{{\"queue_ms\":0.2,\"ttft_ms\":null,\"decode_tps\":30}},\"truncated\":false}}\n", .{id}) catch return "";
    return out.items;
}

/// A request-log record whose id is JSON `null`: the route made no reply id, yet the line keeps a number of its own.
fn nullId(a: Allocator, tmp: t.TmpDir) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    const data = try std.mem.concat(a, u8, &.{
        "{\"type\":\"request\",\"id\":null,\"body\":{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"stream\":false}}\n",
        "{\"type\":\"reply\",\"id\":\"chatcmpl-n\",\"status\":200,\"stream\":false,\"finish_reason\":\"stop\",\"text\":\"Paris.\",\"usage\":{\"prompt_tokens\":12,\"cached_tokens\":4,\"completion_tokens\":7},\"timings\":{\"queue_ms\":1.5,\"ttft_ms\":35.2,\"decode_tps\":41},\"truncated\":false}\n",
    });
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = data });
    return path;
}

test "log: a request-log record whose id is null still lists under a number that show takes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try nullId(a, tmp);
    const code, const out, const err = try runLog(a, .{ .path = path });
    try t.expectEqual(@as(u8, 0), code);
    try t.expectEqualStrings("", err);
    try t.expectEqualStrings(
        \\-------------------  1  chat  m  in flight  prompt -  out -  ttft -  - tok/s
        \\-------------------  chatcmpl-n  -  -  200 finished  prompt 12  out 7  ttft 35.2 ms  41.0 tok/s
        \\
    , out);
    _, const one, _ = try runLog(a, .{ .command = .show, .id = "1", .path = path });
    try t.expect(std.mem.indexOf(u8, one, "id        1\n") != null and std.mem.indexOf(u8, one, "  [user] Capital of France?") != null);
}

/// A typed record whose name is none of the six the reader knows: still the request, and not a finished one.
const unknown_line = "{\"type\":\"answer\",\"id\":\"chatcmpl-u\",\"body\":{\"model\":\"mu\",\"messages\":[{\"role\":\"user\",\"content\":\"Unknown.\"}],\"stream\":false}}\n";

fn unknownPair9(a: Allocator) []const u8 {
    return std.mem.concat(a, u8, &.{ unknown_line, pair(a, "chatcmpl-9", "m9") }) catch return "";
}

test "log: a typed record the reader does not name still reads as the request" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = try std.mem.concat(a, u8, &.{ pair(a, "chatcmpl-1", "m"), unknown_line }) });
    const code, const out, const err = try runLog(a, .{ .path = path });
    try t.expectEqual(@as(u8, 0), code);
    try t.expectEqualStrings("", err);
    try t.expectEqualStrings(
        \\-------------------  chatcmpl-1  -  m  200 finished  prompt 3  out 2  ttft -  30.0 tok/s
        \\-------------------  chatcmpl-u  chat  mu  in flight  prompt -  out -  ttft -  - tok/s
        \\
    , out);
    _, const shown, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-u", .path = path });
    try t.expect(std.mem.indexOf(u8, shown, "  [user] Unknown.") != null and std.mem.indexOf(u8, shown, "  (none yet)") != null);
    // the unknown name does not finish a request: only a `reply` does, so `follow` prints just the pair
    const w: Writer = .{ .path = path, .first = &unknownPair9, .second = &nothing, .after = &nothing };
    const thread = try std.Thread.spawn(.{}, Writer.run, .{&w});
    _, const flown, _ = try runLog(a, .{ .command = .follow, .path = path, .poll_ms = 20, .follow_limit = 1 });
    thread.join();
    try t.expectEqualStrings(no_stamp ++ "  chatcmpl-9  -  m9  200 finished  prompt 3  out 2  ttft -  30.0 tok/s\n", flown);
}

/// One batch: `chatcmpl-9`'s whole request line and then half of its reply line; the rest lands in the next read.
fn tornBatch(a: Allocator) []const u8 {
    const both = pair(a, "chatcmpl-9", "m9");
    const nl = std.mem.indexOfScalar(u8, both, '\n') orelse return both;
    return both[0 .. nl + 1 + (both.len - nl - 1) / 2];
}

fn tornRest(a: Allocator) []const u8 {
    const both = pair(a, "chatcmpl-9", "m9");
    const nl = std.mem.indexOfScalar(u8, both, '\n') orelse return "";
    return both[nl + 1 + (both.len - nl - 1) / 2 ..];
}

test "log: follow reads a whole line and the torn tail beside it from one batch" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = pair(a, "chatcmpl-1", "m") });
    // the request record comes from the whole line of that batch, its reply from the tail the next read completes
    const w: Writer = .{ .path = path, .first = &tornBatch, .second = &tornRest, .after = &nothing };
    const thread = try std.Thread.spawn(.{}, Writer.run, .{&w});
    _, const out, _ = try runLog(a, .{ .command = .follow, .path = path, .poll_ms = 20, .follow_limit = 1 });
    thread.join();
    try t.expectEqualStrings(no_stamp ++ "  chatcmpl-9  -  m9  200 finished  prompt 3  out 2  ttft -  30.0 tok/s\n", out);
}

test "log: follow over the request log prints each pair once its reply lands, across a rotation" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = pair(a, "chatcmpl-1", "m") });
    const w: Writer = .{ .path = path, .first = &logPair9, .second = &nothing, .after = &logPair10 };
    const thread = try std.Thread.spawn(.{}, Writer.run, .{&w});
    _, const out, _ = try runLog(a, .{ .command = .follow, .path = path, .poll_ms = 20, .follow_limit = 2 });
    thread.join();
    try t.expectEqualStrings(
        \\-------------------  chatcmpl-9  -  m9  200 finished  prompt 3  out 2  ttft -  30.0 tok/s
        \\-------------------  chatcmpl-10  -  m10  200 finished  prompt 3  out 2  ttft -  30.0 tok/s
        \\
    , out);
}
