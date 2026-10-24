//! --notes in a chat request: LLWZ Python's answers for a byte-tokenized store, the fields, and the block put first.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const errors = @import("errors.zig");
const notes_route = @import("notes_route.zig");
const retrieve = @import("notes/retrieve.zig");
const Storage = @import("notes/storage.zig").Storage;
const past = @import("notes/past.zig");
const index = @import("notes/index.zig");
const lookup = @import("notes/lookup.zig");
const Conn = @import("http_conn.zig").Conn;
const testing = std.testing;

/// LLWZ's L6-seed1 teach store under a byte tokenizer, and retrieve()'s answers (tools/notes/fixture.py writes both).
pub const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");
const cases_json = @embedFile("notes/testdata/l6_bytes_cases.json");

/// Token ids are UTF-8 bytes, as the fixture's tokenizer; a prompt renders as "[role]content" lines.
pub const Bytes = @import("notes_storage_bytes_test.zig").Bytes;

/// Keeps the prompt the server submits, then answers "ok".
const Capture = struct {
    prompt: []u32 = &.{},

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        testing.allocator.free(e.prompt);
        e.prompt = testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for ("ok") |byte| sink.event(sink.ctx, id, &.{ .tokens = &.{@as(u32, byte)} });
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

pub const Reply = struct { status: u16, body: []u8 };

/// One HTTP request to `srv` over a socket pair, as a client sends it; the reply's status and body.
pub fn exchange(srv: *server.Server, method: []const u8, path: []const u8, body: []const u8) !Reply {
    var pair: [2]platform.posix.socket_t = undefined;
    if (platform.windows) try platform.posix.socketpair(&pair) else if (platform.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = platform.posix.system.close(pair[0]);
    defer _ = platform.posix.system.close(pair[1]);
    const head = try std.fmt.allocPrint(testing.allocator, "{s} {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ method, path, body.len });
    defer testing.allocator.free(head);
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
    var conn = try Conn.init(testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    const raw = try readReady(testing.allocator, pair[0]);
    defer testing.allocator.free(raw);
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const code = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    return .{ .status = code, .body = try testing.allocator.dupe(u8, raw[split + 4 ..]) };
}

fn writeAll(fd: platform.posix.socket_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = platform.posix.system.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn readReady(a: std.mem.Allocator, fd: platform.posix.socket_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]platform.posix.pollfd{.{ .fd = fd, .events = platform.posix.POLL.IN, .revents = 0 }};
        if (try platform.posix.poll(&pollfd, 200) == 0) break;
        const n = try platform.posix.read(fd, &tmp);
        if (n == 0) break;
        try out.appendSlice(a, tmp[0..n]);
    }
    return out.toOwnedSlice(a);
}

fn openStore(text: model_text.Text, bytes: []const u8, kind: index.Kind) !*Storage {
    var why: []const u8 = "";
    return Storage.loadKind(testing.allocator, bytes, text, null, kind, &why);
}

/// How many fixture cases the store answers otherwise than Python's retrieve(); `answered`: cases with a block.
pub fn differing(store: *Storage, text: model_text.Text, answered: *usize) !usize {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = (try json.parse(a, cases_json)).ok;
    var bad: usize = 0;
    for (cases.array) |c| {
        const q = c.strField("question").?;
        const given = c.field("past");
        try testing.expectEqual(c.get("past_read").?.bool, if (given) |g| g.bool else past.isPast(q));
        var ex: std.ArrayList(u32) = .empty;
        for (c.get("exclude").?.array) |x| try ex.append(a, @intCast(x.int64().?));
        const cap: u32 = if (c.field("cap")) |x| @intCast(x.int64().?) else store.model.settings.cap;
        const qids = try text.encode(a, q, false);
        const got = try retrieve.retrieve(store, testing.io, a, text, qids, c.get("past_read").?.bool, cap, ex.items, .{});
        var same = got.tokens == @as(usize, @intCast(c.get("tokens").?.int64().?)) and got.candidates == c.get("candidates").?.int64().?;
        const ids = c.get("ids").?.array;
        same = same and ids.len == got.ids.len;
        if (same) for (ids, got.ids) |want, have| {
            same = same and want.int64().? == have;
        };
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(got.block, &sha, .{});
        const hex = std.fmt.bytesToHex(sha, .lower);
        if (c.field("block_sha256")) |h| {
            same = same and std.mem.eql(u8, h.string, &hex);
        } else same = same and got.block.len == 0;
        if (c.field("block")) |b| same = same and std.mem.eql(u8, b.string, got.block);
        if (got.block.len > 0) answered.* += 1;
        if (!same) bad += 1;
    }
    return bad;
}

test "the server's lookup gives LLWZ Python's ids, tokens, candidates and block for every fixture case" {
    var text: Bytes = .{};
    for ([_]index.Kind{ .postings, .sa }) |kind| {
        const store = try openStore(text.text(), snapshot, kind);
        defer store.deinit();
        var answered: usize = 0;
        try testing.expectEqual(@as(usize, 0), try differing(store, text.text(), &answered));
        try testing.expect(answered >= 60); // most cases retrieve notes, so the comparison is not of empty blocks
    }
}

test "negative control: the same store with its hop off answers otherwise than Python" {
    var text: Bytes = .{};
    const broken = try std.mem.replaceOwned(u8, testing.allocator, snapshot, "\"hop\":\"subject\"", "\"hop\":\"none\"");
    defer testing.allocator.free(broken);
    try testing.expect(!std.mem.eql(u8, broken, snapshot));
    for ([_]index.Kind{ .postings, .sa }) |kind| {
        const store = try openStore(text.text(), broken, kind);
        defer store.deinit();
        var answered: usize = 0;
        try testing.expect(try differing(store, text.text(), &answered) > 0);
    }
}

test "the notes fields a chat request may send" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { body: []const u8, on: bool, refusal: []const u8 = "" };
    const cases = [_]Case{
        .{ .body = "{}", .on = false },
        .{ .body = "{\"notes\": false}", .on = false },
        .{ .body = "{\"notes\": null}", .on = false },
        .{ .body = "{\"notes\": true}", .on = true },
        .{ .body = "{\"notes_exclude\": [3, 7]}", .on = true },
        .{ .body = "{\"notes\": true, \"notes_past\": true, \"notes_cap\": 128, \"notes_echo\": true}", .on = true },
        .{ .body = "{\"notes\": \"yes\"}", .on = false, .refusal = "notes must be a boolean" },
        .{ .body = "{\"notes\": 1}", .on = false, .refusal = "notes must be a boolean" },
        .{ .body = "{\"notes_exclude\": 3}", .on = false, .refusal = "notes_exclude must be a list of note ids" },
        .{ .body = "{\"notes_exclude\": [-1]}", .on = false, .refusal = "notes_exclude must be a list of note ids" },
        .{ .body = "{\"notes_exclude\": [true]}", .on = false, .refusal = "notes_exclude must be a list of note ids" },
        .{ .body = "{\"notes\": true, \"notes_cap\": 0}", .on = false, .refusal = "notes_cap must be a whole number of tokens, 1 or more" },
        .{ .body = "{\"notes\": true, \"notes_past\": 1}", .on = false, .refusal = "notes_past must be a boolean" },
        .{ .body = "{\"notes\": false, \"notes_exclude\": [1]}", .on = false, .refusal = "notes is false but notes_exclude names notes to leave out" },
        .{ .body = "{\"notes_cap\": 64}", .on = false, .refusal = "notes_past, notes_cap and notes_echo need \"notes\": true" },
        .{ .body = "{\"notes\": true, \"notes_language\": \"de\"}", .on = true },
        .{ .body = "{\"notes\": true, \"notes_language\": \"sv\"}", .on = false, .refusal = "notes_language must be en, de or fi" },
        .{ .body = "{\"notes\": true, \"notes_language\": 1}", .on = false, .refusal = "notes_language must be a string" },
        .{ .body = "{\"notes_language\": \"de\"}", .on = false, .refusal = "notes_language needs \"notes\": true" },
    };
    for (cases) |c| {
        var cx: errors.Cx = .{ .a = a };
        const body = (try json.parse(a, c.body)).ok;
        if (c.refusal.len > 0) {
            try testing.expectError(error.Refused, notes_route.asked(&cx, body));
            try testing.expectEqualStrings(c.refusal, cx.message);
            continue;
        }
        const got = try notes_route.asked(&cx, body);
        try testing.expectEqual(c.on, got != null);
    }
    var cx: errors.Cx = .{ .a = a };
    const full = (try notes_route.asked(&cx, (try json.parse(a, "{\"notes_exclude\": [3, 7], \"notes_past\": false, \"notes_cap\": 99}")).ok)).?;
    try testing.expectEqualSlices(u32, &.{ 3, 7 }, full.exclude);
    try testing.expectEqual(@as(?bool, false), full.past);
    try testing.expectEqual(@as(?u32, 99), full.cap);
    try testing.expect(full.lang == null);
    const de = (try notes_route.asked(&cx, (try json.parse(a, "{\"notes\": true, \"notes_language\": \"de\"}")).ok)).?;
    try testing.expectEqual(@import("notes/lang.zig").Lang.de, de.lang.?);
}

/// The prompt a chat body renders to, and its reply's parsed body.
fn ask(srv: *server.Server, backend: *Capture, a: std.mem.Allocator, body: []const u8) !struct { prompt: []u32, reply: json.Value, status: u16 } {
    const got = try exchange(srv, "POST", "/v1/chat/completions", body);
    defer testing.allocator.free(got.body);
    const reply = (try json.parse(a, try a.dupe(u8, got.body))).ok;
    return .{ .prompt = try a.dupe(u32, backend.prompt), .reply = reply, .status = got.status };
}

fn chatBody(a: std.mem.Allocator, msgs: []const [2][]const u8, extra: []const u8) ![]const u8 {
    var list: std.ArrayList(json.Value) = .empty;
    for (msgs) |m| {
        const o = try json.newObject(a);
        try o.put(a, "role", .{ .string = m[0] });
        try o.put(a, "content", .{ .string = m[1] });
        try list.append(a, .{ .object = o });
    }
    const text = try json.stringify(a, .{ .array = list.items }, .{});
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":4,\"messages\":{s}{s}}}", .{ text, extra });
}

test "a chat request asking for notes gets the block as its first system message, as if the client had sent it" {
    var text: Bytes = .{};
    var backend: Capture = .{};
    defer testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(testing.allocator, testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q = "In which city is my home?";
    const plain = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", q }}, ""));
    // without --notes the fields change nothing: the same prompt, and no notes in the reply
    const ignored = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", q }}, ",\"notes\":true"));
    try testing.expectEqualSlices(u32, plain.prompt, ignored.prompt);
    try testing.expect(ignored.reply.get("notes") == null);

    srv.notes = try openStore(text.text(), snapshot, .postings);
    const engine = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", q }}, ",\"notes\":true,\"notes_echo\":true"));
    try testing.expectEqual(@as(u16, 200), engine.status);
    const notes = engine.reply.get("notes").?;
    const block = notes.strField("block").?;
    try testing.expect(std.mem.startsWith(u8, block, "Notes from memory:"));
    try testing.expect(notes.get("ids").?.array.len > 0);
    try testing.expectEqual(@as(i64, @intCast(block.len)), notes.get("tokens").?.int64().?); // bytes are its tokens here
    const client = try ask(srv, &backend, a, try chatBody(a, &.{ .{ "system", block }, .{ "user", q } }, ""));
    try testing.expectEqualSlices(u32, client.prompt, engine.prompt);
    try testing.expect(!std.mem.eql(u32, plain.prompt, engine.prompt));
    try testing.expectEqual(plain.prompt.len + block.len + "[system]\n".len, engine.prompt.len);

    // the client's own system message follows the block, as two leading system messages merge
    const own = try ask(srv, &backend, a, try chatBody(a, &.{ .{ "system", "Be brief." }, .{ "user", q } }, ",\"notes\":true"));
    const both = try ask(srv, &backend, a, try chatBody(a, &.{ .{ "system", block }, .{ "system", "Be brief." }, .{ "user", q } }, ""));
    try testing.expectEqualSlices(u32, both.prompt, own.prompt);

    // the last user message is the question; nothing retrieved puts nothing in
    const none = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", "?!?" }}, ",\"notes\":true"));
    const none_plain = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", "?!?" }}, ""));
    try testing.expectEqualSlices(u32, none_plain.prompt, none.prompt);
    try testing.expectEqual(@as(usize, 0), none.reply.get("notes").?.get("ids").?.array.len);
    try testing.expect(none.reply.get("notes").?.get("block_sha256").? == .null);

    // excluding every note it gave leaves those out
    var ex: std.ArrayList(u8) = .empty;
    for (notes.get("ids").?.array, 0..) |id, k| try ex.print(a, "{s}{s}", .{ if (k > 0) "," else "", id.int });
    const left = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", q }}, try std.fmt.allocPrint(a, ",\"notes_exclude\":[{s}]", .{ex.items})));
    for (left.reply.get("notes").?.get("ids").?.array) |id| for (notes.get("ids").?.array) |gone| try testing.expect(!std.mem.eql(u8, id.int, gone.int));

    const bad = try ask(srv, &backend, a, try chatBody(a, &.{.{ "user", q }}, ",\"notes\":\"yes\""));
    try testing.expectEqual(@as(u16, 400), bad.status);
    try testing.expectEqualStrings("notes must be a boolean", bad.reply.get("error").?.strField("message").?);
}

fn hasId(reply: json.Value, id: i64) bool {
    for (reply.get("notes").?.get("ids").?.array) |x| if (x.int64().? == id) return true;
    return false;
}

test "a note written over /v1/notes is looked up by the next chat request, and an erased one is not" {
    var text: Bytes = .{};
    var backend: Capture = .{};
    defer testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(testing.allocator, testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var why: []const u8 = "";
    for ([_]index.Kind{ .postings, .sa }) |kind| {
        const path = try std.fs.path.join(a, &.{ dir, @tagName(kind) });
        srv.notes = try Storage.loadKind(testing.allocator, snapshot, text.text(), path, kind, &why);
        defer {
            srv.notes.?.deinit();
            srv.notes = null;
        }
        try srv.notes.?.beginRun(a, .{});
        const q = try chatBody(a, &.{.{ "user", "What does my zebra eat?" }}, ",\"notes\":true");
        const before = try ask(srv, &backend, a, q); // the kept scratch is made for the store's notes
        const added = try exchange(srv, "POST", "/v1/notes", "{\"text\":\"My zebra eats hay.\"}");
        defer testing.allocator.free(added.body);
        try testing.expectEqual(@as(u16, 201), added.status);
        const id = (try json.parse(a, try a.dupe(u8, added.body))).ok.get("id").?.int64().?;
        try testing.expect(!hasId(before.reply, id));
        try testing.expect(hasId((try ask(srv, &backend, a, q)).reply, id)); // the scratch grows with the index
        const erased = try exchange(srv, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), "");
        defer testing.allocator.free(erased.body);
        try testing.expectEqual(@as(u16, 200), erased.status);
        try testing.expect(srv.notes.?.scratch == null); // an erase drops the scratch that may hold the note's text
        const after = try ask(srv, &backend, a, q);
        try testing.expect(!hasId(after.reply, id));
        try testing.expectEqualSlices(i64, try idList(a, before.reply), try idList(a, after.reply));
    }
}

fn idList(a: std.mem.Allocator, reply: json.Value) ![]i64 {
    var out: std.ArrayList(i64) = .empty;
    for (reply.get("notes").?.get("ids").?.array) |x| try out.append(a, x.int64().?);
    return out.items;
}

fn scratchInit(gpa: std.mem.Allocator, idx: *const index.Index) !void {
    var sc = try lookup.Scratch.init(gpa, idx);
    sc.deinit();
}

fn indexLoad(gpa: std.mem.Allocator, bytes: []const u8) !void {
    // Disable address-dependent in-place growth so the failure sweep visits the same allocations every attempt.
    var child = gpa;
    const Stable = struct {
        fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
            const a: *std.mem.Allocator = @ptrCast(@alignCast(ctx));
            return a.rawAlloc(len, alignment, ret);
        }
        fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret: usize) void {
            const a: *std.mem.Allocator = @ptrCast(@alignCast(ctx));
            a.rawFree(memory, alignment, ret);
        }
    };
    const stable: std.mem.Allocator = .{ .ptr = &child, .vtable = &.{ .alloc = Stable.alloc, .free = Stable.free, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap } };
    const idx = try index.Index.load(stable, bytes);
    idx.deinit();
}

test "a lookup's scratch and the index each free what they took when any one of their allocations fails" {
    const idx = try index.Index.load(testing.allocator, snapshot);
    defer idx.deinit();
    try testing.checkAllAllocationFailures(testing.allocator, scratchInit, .{idx});
    try testing.checkAllAllocationFailures(testing.allocator, indexLoad, .{snapshot});
}

/// A store loaded with `gpa` answers one question twice (a new scratch, then the kept one) and is freed.
fn storeRetrieve(gpa: std.mem.Allocator, text: model_text.Text, qids: []const u32) !void {
    var why: []const u8 = "";
    const store = try Storage.loadKind(gpa, snapshot, text, null, .postings, &why);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator); // the request's own memory, freed with the request
    defer arena.deinit();
    for (0..2) |_| _ = try retrieve.retrieve(store, testing.io, arena.allocator(), text, qids, false, store.model.settings.cap, &.{ 7, 11 }, .{});
}

test "the server's store leaks nothing when any one of its long-lived allocations fails" {
    // by hand, as the store frees what it took on each failure: OutOfMemory or success, and all memory back
    var text: Bytes = .{};
    const qids = try text.text().encode(testing.allocator, "Where does my sister live?", false);
    defer testing.allocator.free(qids);
    var points: usize = 0;
    while (true) : (points += 1) {
        var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = points });
        if (storeRetrieve(failing.allocator(), text.text(), qids)) |_| {} else |e| try testing.expectEqual(error.OutOfMemory, e);
        try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        if (!failing.has_induced_failure) break;
    }
    try testing.expect(points >= 8); // the load's and the scratch's: all were failed in turn
}
