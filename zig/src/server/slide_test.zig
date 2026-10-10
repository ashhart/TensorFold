//! Host checks for the learn flow's take-out. The fake learner counts the undos a lesson that did not come back sends.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;
const slide = @import("slide.zig");
const server_mod = @import("server.zig");
const Allocator = std.mem.Allocator;

const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Answers every learn request at once, as an undo ends (`done`, no report), or refuses it as a closed engine does.
const Learner = struct {
    undos: usize = 0,
    others: usize = 0,
    refuse: bool = false,
    refuse_undo: bool = false,

    fn engine(l: *Learner) api.Engine {
        return .{ .ctx = l, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .name = "fake" };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn learn(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const l: *Learner = @ptrCast(@alignCast(ctx));
        if (request.undo) l.undos += 1 else l.others += 1;
        if (l.refuse or (l.refuse_undo and request.undo)) return error.Closed;
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};

fn serve(text: *ByteText, learner: *Learner) !*server_mod.Server {
    return server_mod.Server.init(std.testing.allocator, std.testing.io, learner.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
}

test "a lesson whose facts did not all come back is taken out with one undo, and says why" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var learner: Learner = .{};
    const srv = try serve(&text, &learner);
    defer srv.deinit();
    var missed = [_]bool{false};
    var out: slide.Rounds = .{ .recalled = &missed, .steps = 300 };
    try std.testing.expect(slide.takeOut(srv, a, &out));
    try std.testing.expectEqual(@as(usize, 1), learner.undos);
    try std.testing.expectEqualStrings("taken out: it did not come back on its held-out questions", out.why.?);
    var leaked = [_]bool{false};
    var round: slide.Rounds = .{ .recalled = &leaked, .steps = 40, .why = "in round 2, a fact leaked into an answer about something else" };
    try std.testing.expect(slide.takeOut(srv, a, &round));
    try std.testing.expectEqual(@as(usize, 2), learner.undos);
    try std.testing.expectEqualStrings("taken out: in round 2, a fact leaked into an answer about something else", round.why.?);
    var half = [_]bool{ true, false };
    var pair: slide.Rounds = .{ .recalled = &half, .steps = 120 };
    try std.testing.expect(slide.takeOut(srv, a, &pair));
    try std.testing.expect(learner.undos == 3 and !half[0] and !half[1]);
    try std.testing.expectEqual(@as(usize, 0), learner.others);
}

test "a lesson that finished no round sends no undo, which would reach into the lesson before" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var learner: Learner = .{};
    const srv = try serve(&text, &learner);
    defer srv.deinit();
    var refused = [_]bool{false};
    var early: slide.Rounds = .{ .recalled = &refused, .why = "LearnedChangeFull" };
    try std.testing.expect(slide.takeOut(srv, a, &early));
    try std.testing.expectEqualStrings("LearnedChangeFull", early.why.?);
    var silent = [_]bool{false};
    var none: slide.Rounds = .{ .recalled = &silent };
    try std.testing.expect(slide.takeOut(srv, a, &none));
    try std.testing.expectEqualStrings("it did not come back on its held-out questions", none.why.?);
    try std.testing.expectEqual(@as(usize, 0), learner.undos);
}

test "a lesson is back only when every fact came back" {
    try std.testing.expect(slide.all(&.{true}));
    try std.testing.expect(slide.all(&.{ true, true }));
    try std.testing.expect(!slide.all(&.{false}));
    try std.testing.expect(!slide.all(&.{ true, false }));
    try std.testing.expect(!slide.all(&.{})); // facts that could not be counted did not come back
}

test "an undo the engine refuses is reported, so the request saves nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var learner: Learner = .{ .refuse = true };
    const srv = try serve(&text, &learner);
    defer srv.deinit();
    var missed = [_]bool{false};
    var out: slide.Rounds = .{ .recalled = &missed, .steps = 300 };
    try std.testing.expect(!slide.takeOut(srv, a, &out));
    try std.testing.expectEqual(@as(usize, 1), learner.undos);
    const said = "it did not come back on its held-out questions; it could not be taken out (Closed), so nothing of this request is saved: restart the server";
    try std.testing.expectEqualStrings(said, out.why.?);
}

test "a commit or mining round that is refused or ends without a result leaves no fact recalled" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var refusing: Learner = .{ .refuse = true };
    const closed = try serve(&text, &refusing);
    defer closed.deinit();
    var back = [_]bool{true};
    var out: slide.Rounds = .{ .recalled = &back, .steps = 80 };
    try std.testing.expect(!slide.step(closed, a, &.{ .commit = true }, &out));
    try std.testing.expect(!back[0] and !slide.all(&back));
    try std.testing.expectEqualStrings("Closed", out.why.?);
    var silent: Learner = .{};
    const quiet = try serve(&text, &silent);
    defer quiet.deinit();
    var mined = [_]bool{ true, true };
    var more: slide.Rounds = .{ .recalled = &mined, .steps = 140 };
    try std.testing.expect(!slide.step(quiet, a, &.{ .more = true }, &more));
    try std.testing.expect(!mined[0] and !mined[1]);
    try std.testing.expectEqualStrings("the learner ended the round without a result", more.why.?);
    try std.testing.expectEqual(@as(usize, 80), out.steps);
}

test "after an undo that did not run, the next learn request is refused and nothing saves" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var learner: Learner = .{ .refuse_undo = true };
    const srv = try serve(&text, &learner);
    defer srv.deinit();
    try std.testing.expect(!srv.teacher.stuck);
    var missed = [_]bool{false};
    var out: slide.Rounds = .{ .recalled = &missed, .steps = 300 };
    try std.testing.expect(!slide.takeOut(srv, a, &out));
    try std.testing.expect(srv.teacher.stuck);
    const before = learner.others;
    const reply = try exchange(srv, "/v1/slide/learn", "{\"text\": \"My cat is called Pipsa.\"}");
    defer std.testing.allocator.free(reply.body);
    try std.testing.expectEqual(@as(u16, 503), reply.status);
    try std.testing.expect(std.mem.indexOf(u8, reply.body, "restart the server") != null);
    try std.testing.expectEqual(before + 1, learner.others); // the probe only: no lesson and no save reached it
}

const Reply = struct { status: u16, body: []u8 };

/// One request through the server's routes over a socket pair, its reply read back.
fn exchange(srv: *server_mod.Server, path: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(std.testing.allocator, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ path, body.len });
    defer std.testing.allocator.free(head);
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, a);
    const raw = try readReady(std.testing.allocator, pair[0]);
    defer std.testing.allocator.free(raw);
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const status = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    return .{ .status = status, .body = try std.testing.allocator.dupe(u8, raw[split + 4 ..]) };
}

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn readReady(a: Allocator, fd: std.c.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&pollfd, 200) == 0) break;
        const n = try std.posix.read(fd, &tmp);
        if (n == 0) break;
        try out.appendSlice(a, tmp[0..n]);
    }
    return out.toOwnedSlice(a);
}
