//! The request journal on a fake engine: records, replay by the resolved seed, crash quarantine, rotation, failures.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const json = @import("json");
const platform = @import("sys");
const journal_mod = @import("request_journal.zig");
const Journal = journal_mod.Journal;
const Trace = @import("request_trace.zig").Trace;
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const TestText = @import("stream_preflight_test.zig").TestText;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// Replies with the decimal digits of the seed it was given, so a replay without that seed reads differently.
const Fake = struct {
    submitted: usize = 0,
    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.submitted += 1;
        var buf: [24]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "{d}", .{if (r.sampling) |s| s.seed else 7}) catch unreachable;
        var ids: [24]u32 = undefined;
        for (text, 0..) |b, i| ids[i] = b;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..text.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The bytes a client would read: each payload serialized as the socket writers do.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    status: u16 = 0,
    last: Value = .null,
    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.status = 200;
    }
    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        const p = payload orelse return w.bytes.appendSlice(w.a, "data: [DONE]\n\n") catch error.Closed;
        w.last = p;
        const text = json.stringify(w.a, p, .{}) catch return error.Closed;
        w.bytes.appendSlice(w.a, std.mem.concat(w.a, u8, &.{ "data: ", text, "\n\n" }) catch return error.Closed) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, code: u16, payload: Value) void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.status = code;
        w.last = payload;
        w.bytes.appendSlice(w.a, json.stringify(w.a, payload, .{}) catch return) catch {};
    }
    fn tokens(w: *const Wire) []const u8 {
        return w.last.get("tensorfold").?.strField("token_sha256").?;
    }
};

const hello = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":1,\"max_tokens\":8";

fn serve(fake: *Fake, text: *TestText, journal: ?*Journal) !*Server {
    return Server.init(t.allocator, t.io, fake.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .journal = journal }, null);
}

fn ask(srv: *Server, a: Allocator, body: []const u8, key: ?[]const u8) !Wire {
    var wire: Wire = .{ .a = a };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .key_label = key };
    openai.run(srv, a, wire.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return wire;
}

fn records(a: Allocator, path: []const u8) ![]Value {
    const raw = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 26));
    var found: std.ArrayList(Value) = .empty;
    var it = std.mem.splitScalar(u8, raw, '\n');
    while (it.next()) |line| if (line.len > 0) try found.append(a, (try json.parse(a, line)).ok);
    return found.items;
}

fn kinds(a: Allocator, path: []const u8, kind: []const u8) ![]Value {
    var found: std.ArrayList(Value) = .empty;
    for (try records(a, path)) |r| if (std.mem.eql(u8, r.strField("kind").?, kind)) try found.append(a, r);
    return found.items;
}

fn journalPath(a: Allocator, tmp: t.TmpDir, name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
}

/// A request that began and never finished: the process died with it in flight.
fn inFlight(j: *Journal, a: Allocator, body: []const u8, key: []const u8) !void {
    var wire: Wire = .{ .a = a };
    var trace: Trace = .{ .journal = j, .a = a, .id = try std.fmt.allocPrint(a, "chatcmpl-{d}", .{j.bytes}), .parent = wire.out() };
    try t.expect(!trace.begin((try json.parse(a, body)).ok, true, key));
}

test "request journal: whole and streamed replies keep their start, resolved seed and exact reply bytes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const j = try Journal.open(t.allocator, t.io, path);
    defer j.close();
    var fake: Fake = .{};
    var text: TestText = .{};
    const srv = try serve(&fake, &text, j);
    defer srv.deinit();
    inline for (.{ "}", ",\"stream\":true}" }) |tail| {
        const wire = try ask(srv, a, hello ++ ",\"seed\":1234" ++ tail, null);
        try t.expectEqual(@as(u16, 200), wire.status);
        const finishes = try kinds(a, path, "finish");
        const finish = finishes[finishes.len - 1];
        const id = finish.strField("id").?;
        try t.expectEqualStrings("completed", finish.strField("outcome").?);
        try t.expectEqual(tail.len > 1, finish.get("stream").?.bool);
        const encoded = finish.strField("reply_base64").?;
        const decoder = std.base64.standard.Decoder;
        const bytes = try a.alloc(u8, try decoder.calcSizeForSlice(encoded));
        try decoder.decode(bytes, encoded);
        try t.expectEqualStrings(wire.bytes.items, bytes);
        try t.expectEqualStrings(wire.tokens(), finish.strField("token_sha256").?);
        for (try records(a, path)) |r| if (std.mem.eql(u8, r.strField("id") orelse "", id)) {
            const kind = r.strField("kind").?;
            if (std.mem.eql(u8, kind, "prepared")) try t.expectEqual(@as(i64, 1234), r.get("seed").?.int64().?);
            if (std.mem.eql(u8, kind, "start")) try t.expectEqual(@as(usize, 64), r.strField("fingerprint").?.len);
        };
    }
    try t.expectEqual(@as(usize, 2), (try kinds(a, path, "start")).len);
    try t.expectEqual(@as(usize, 2), (try kinds(a, path, "prepared")).len);
}

test "request journal: an unseeded request's drawn seed replays the same tokens, and a replay without it differs" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    var fake: Fake = .{};
    var text: TestText = .{};
    const first = blk: {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        const srv = try serve(&fake, &text, j);
        defer srv.deinit();
        break :blk try ask(srv, a, hello ++ "}", null);
    };
    const start = (try kinds(a, path, "start"))[0];
    const seed = (try kinds(a, path, "prepared"))[0].get("seed").?;
    try t.expect(start.get("seed").? == .null);
    const body = start.get("body").?;
    try body.object.put(a, "seed", seed);
    const replay = try json.stringify(a, body, .{});
    const srv = try serve(&fake, &text, null);
    defer srv.deinit();
    const again = try ask(srv, a, replay, null);
    try t.expectEqualStrings(first.tokens(), again.tokens());
    const content = first.last.get("choices").?.array[0].get("message").?.strField("content").?;
    try t.expectEqualStrings(content, again.last.get("choices").?.array[0].get("message").?.strField("content").?);
    _ = body.object.orderedRemove("seed");
    const fresh = try ask(srv, a, try json.stringify(a, body, .{}), null);
    try t.expect(!std.mem.eql(u8, first.tokens(), fresh.tokens())); // a fresh seed: one chance in about 2^63 to match
}

test "request journal: image and PDF data stay out, and the journal and its lock are owner-only" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const j = try Journal.open(t.allocator, t.io, path);
    defer j.close();
    var fake: Fake = .{};
    var text: TestText = .{};
    const srv = try serve(&fake, &text, j);
    defer srv.deinit();
    const parts = "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.test/private.png\"}},{\"type\":\"file\",\"file\":{\"filename\":\"a.pdf\",\"file_data\":\"SECRET-PDF\"}}]";
    _ = try ask(srv, a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":" ++ parts ++ "}]}", "alice");
    const raw = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    try t.expect(std.mem.indexOf(u8, raw, "private.png") == null and std.mem.indexOf(u8, raw, "SECRET-PDF") == null);
    try t.expect(std.mem.indexOf(u8, raw, "a.pdf") != null and std.mem.indexOf(u8, raw, "<redacted>") != null);
    try t.expect(std.mem.indexOf(u8, raw, "alice") == null); // the key's label is in the fingerprint, not the file
    if (builtin.os.tag == .windows) return; // NTFS keeps access in ACLs, not mode bits
    for ([_][]const u8{ "requests.jsonl", "requests.jsonl.lock" }) |name| {
        const st = try tmp.dir.statFile(t.io, name, .{});
        try t.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
    }
}

test "request journal: a request in flight at two crashes is refused after restart, for its own key only" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const crashed = hello ++ ",\"seed\":9}";
    var fake: Fake = .{};
    var text: TestText = .{};
    for (0..2) |crash| {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close(); // a close without a stop: the process died
        try t.expectEqual(crash, (try kinds(a, path, "quarantine")).len);
        const srv = try serve(&fake, &text, j);
        defer srv.deinit();
        try t.expectEqual(@as(u16, 200), (try ask(srv, a, crashed, "alice")).status); // one crash only flags it
        try inFlight(j, a, crashed, "alice");
    }
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try t.expectEqual(@as(usize, 2), (try kinds(a, path, "quarantine")).len);
        const srv = try serve(&fake, &text, j);
        defer srv.deinit();
        const before = fake.submitted;
        const refused = try ask(srv, a, crashed, "alice");
        try t.expectEqual(@as(u16, 400), refused.status);
        try t.expectEqualStrings("request_quarantined", refused.last.get("error").?.strField("code").?);
        try t.expectEqual(before, fake.submitted); // refused before the engine saw it
        try t.expectEqual(@as(u16, 200), (try ask(srv, a, crashed, "bob")).status);
        try t.expectEqual(@as(u16, 200), (try ask(srv, a, hello ++ ",\"seed\":10}", "alice")).status);
        try t.expectEqual(before + 2, fake.submitted);
        const finishes = try kinds(a, path, "finish");
        try t.expectEqualStrings("quarantined", finishes[finishes.len - 3].strField("outcome").?);
        j.stopped();
    }
    const j = try Journal.open(t.allocator, t.io, path);
    defer j.close();
    try t.expectEqual(@as(usize, 2), (try kinds(a, path, "quarantine")).len); // nothing in flight, nothing new
    try t.expect(j.quarantined(journal_mod.fingerprint("alice", true, (try json.parse(a, crashed)).ok)));
}

test "request journal: of several requests in flight at one crash, only one in flight again at a second is refused" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const bodies = [_][]const u8{ hello ++ ",\"seed\":1}", hello ++ ",\"seed\":2}", hello ++ ",\"seed\":3}" };
    const fp = struct {
        fn of(arena_a: Allocator, body: []const u8) !journal_mod.Fingerprint {
            return journal_mod.fingerprint("", true, (try json.parse(arena_a, body)).ok);
        }
    }.of;
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 2 } }) |pair| {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        for (bodies) |body| try t.expect(!j.quarantined(try fp(a, body)));
        for (pair) |k| try inFlight(j, a, bodies[k], "");
    }
    const j = try Journal.open(t.allocator, t.io, path);
    defer j.close();
    try t.expect(j.quarantined(try fp(a, bodies[0])));
    try t.expect(!j.quarantined(try fp(a, bodies[1])) and !j.quarantined(try fp(a, bodies[2])));
    try t.expectEqual(@as(usize, 4), (try kinds(a, path, "quarantine")).len);
}

test "request journal: a clean stop quarantines nothing, a clear record lifts one, and an old one lapses" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const body = hello ++ "}";
    const fp = journal_mod.fingerprint("", true, (try json.parse(a, body)).ok);
    const hex = std.fmt.bytesToHex(fp, .lower);
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try inFlight(j, a, body, "");
        j.stopped();
    }
    for (0..2) |_| {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try inFlight(j, a, body, "");
    }
    try t.expectEqual(@as(usize, 1), (try kinds(a, path, "quarantine")).len); // the stopped one was not a crash
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try t.expect(j.quarantined(fp));
        const line = try std.fmt.allocPrint(a, "{{\"version\":1,\"kind\":\"clear\",\"fingerprint\":\"{s}\"}}\n", .{hex});
        try t.expectEqual(@as(isize, @intCast(line.len)), platform.c.write(j.fd, line.ptr, line.len));
    }
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try t.expect(!j.quarantined(fp));
        const old = std.Io.Clock.real.now(t.io).toSeconds() - journal_mod.quarantine_s - 1;
        for (0..2) |_| {
            const line = try std.fmt.allocPrint(a, "{{\"version\":1,\"kind\":\"quarantine\",\"id\":\"x\",\"fingerprint\":\"{s}\",\"at\":{d}}}\n", .{ hex, old });
            try t.expectEqual(@as(isize, @intCast(line.len)), platform.c.write(j.fd, line.ptr, line.len));
        }
    }
    const j = try Journal.open(t.allocator, t.io, path);
    defer j.close();
    try t.expect(!j.quarantined(fp));
}

test "request journal: a torn last line is cut, an unreadable one passed over, and a second server refused" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try t.expectError(error.JournalBusy, Journal.open(t.allocator, t.io, path));
        try t.expectEqual(@as(isize, 4), platform.c.write(j.fd, "bad\n", 4));
        try t.expectEqual(@as(isize, 8), platform.c.write(j.fd, "{\"kind\":", 8));
    }
    const j = try Journal.open(t.allocator, t.io, path);
    j.stopped();
    j.close();
    const raw = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    try t.expect(std.mem.startsWith(u8, raw, "bad\n{\"version\":1,\"kind\":\"stop\""));
    if (builtin.os.tag == .windows) return; // refusal by mode bits and links; NTFS keeps access in ACLs
    try tmp.dir.writeFile(t.io, .{ .sub_path = "shared.jsonl", .data = "" });
    try tmp.dir.setFilePermissions(t.io, "shared.jsonl", .fromMode(0o644), .{});
    try t.expectError(error.JournalPermissions, Journal.open(t.allocator, t.io, try journalPath(a, tmp, "shared.jsonl")));
    try tmp.dir.symLink(t.io, "requests.jsonl", "link.jsonl", .{});
    try t.expectError(error.JournalOpen, Journal.open(t.allocator, t.io, try journalPath(a, tmp, "link.jsonl")));
}

test "request journal: files rotate at their cap, four are kept, and a start in a rotated file is still flagged" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const body = hello ++ "}";
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        j.limit = 2048;
        try inFlight(j, a, body, "");
        for (0..40) |_| j.append(a, .{ .object = try j.record(a, "note", "filler") }); // records recovery ignores
    }
    try t.expectEqual(@as(usize, 1), (try kinds(a, try journalPath(a, tmp, "requests.jsonl.1"), "start")).len);
    {
        const j = try Journal.open(t.allocator, t.io, path);
        defer j.close();
        try t.expectEqual(@as(usize, 1), (try kinds(a, path, "quarantine")).len);
        j.limit = 2048;
        for (0..400) |_| j.append(a, .{ .object = try j.record(a, "note", "filler") });
        try t.expectEqual(@as(u64, 0), j.lost.load(.monotonic));
    }
    for ([_][]const u8{ "requests.jsonl", "requests.jsonl.1", "requests.jsonl.2", "requests.jsonl.3" }) |name| {
        try t.expect((try tmp.dir.statFile(t.io, name, .{})).size <= 2048);
    }
    try t.expectError(error.FileNotFound, tmp.dir.statFile(t.io, "requests.jsonl.4", .{}));
}

test "request journal: a write that fails loses its record and the request still completes the same" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try journalPath(a, tmp, "requests.jsonl");
    const j = try Journal.open(t.allocator, t.io, path);
    defer j.close();
    const writable = j.fd;
    j.fd = platform.c.open(try a.dupeSentinel(u8, path, 0), .{ .ACCMODE = .RDONLY }); // every write now fails, as on a full disk
    defer {
        _ = platform.c.close(j.fd);
        j.fd = writable;
    }
    var fake: Fake = .{};
    var text: TestText = .{};
    const srv = try serve(&fake, &text, j);
    defer srv.deinit();
    const wire = try ask(srv, a, hello ++ ",\"seed\":5}", null);
    try t.expectEqual(@as(u16, 200), wire.status);
    try t.expectEqualStrings("5", wire.last.get("choices").?.array[0].get("message").?.strField("content").?);
    try t.expect(j.lost.load(.monotonic) >= 3);
}

test "request journal: off, a request's output path is the caller's and its tokens are a journaled one's" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var wire: Wire = .{ .a = a };
    const parent = wire.out();
    var off: Trace = .{ .journal = null, .a = a, .id = "x", .parent = parent };
    try t.expect(off.out().vt == parent.vt and off.out().ctx == parent.ctx);
    try t.expect(!off.begin((try json.parse(a, hello ++ "}")).ok, true, ""));
    var fake: Fake = .{};
    var text: TestText = .{};
    const j = try Journal.open(t.allocator, t.io, try journalPath(a, tmp, "requests.jsonl"));
    defer j.close();
    var replies: [2]Wire = undefined;
    for ([_]?*Journal{ null, j }, &replies) |journal, *reply| {
        const srv = try serve(&fake, &text, journal);
        defer srv.deinit();
        reply.* = try ask(srv, a, hello ++ ",\"seed\":77,\"stream\":true}", null);
    }
    try t.expectEqualStrings(replies[0].tokens(), replies[1].tokens());
    try t.expect(json.equal(replies[0].last.get("choices").?, replies[1].last.get("choices").?));
}
