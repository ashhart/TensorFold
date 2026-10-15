//! DELETE /v1/slide/fact/{id}: one fact's neurons zeroed, every fact it leaves asked again, then written or put back.
const std = @import("std");
const json = @import("json.zig");
const sse = @import("sse.zig");
const http_body = @import("http_body.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const slide = @import("slide.zig");
const slide_graph = @import("slide_graph.zig");
const slide_teacher = @import("slide_teacher.zig");
const recheck = @import("slide_recheck.zig");
const Gone = @import("openai.zig").Gone;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const Fact = slide_graph.Fact;

const busy = "a learn or delete request is running: send yours when it ends";

/// A refused delete: its HTTP status and why.
pub const Refused = struct { code: u16, message: []const u8 };

/// Once its fact's neurons are zeroed a delete ends written or put back, whether its client stays or not.
const Stays = struct {
    pub fn check(_: Stays) bool {
        return false;
    }
};

pub fn route(srv: *Server, conn: *Conn, a: Allocator, raw_id: []const u8) void {
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |message| return slide.refuse(conn, a, 400, message),
    };
    const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
        .ok => |v| v,
        .err => |message| return slide.refuse(conn, a, 400, message),
    };
    const reason = body.strField("reason") orelse "deleted on request";
    const id = std.fmt.parseInt(u32, raw_id, 10) catch return slide.refuse(conn, a, 400, "a fact's id is a whole number, as GET /v1/slide/graph lists it");
    if (admit(srv, a)) |r| return slide.refuse(conn, a, r.code, r.message);
    defer srv.teacher.learning.unlock(srv.io);
    const facts = srv.slide.snapshot(a) catch return slide.refuse(conn, a, 500, "out of memory");
    const at = find(facts, id) orelse return slide.refuse(conn, a, 404, std.fmt.allocPrint(a, "no fact {d} in the graph", .{id}) catch "no such fact");
    sse.open(conn) catch return;
    var out: slide.Out = .{ .conn = conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    out.event("fact", .{ .id = id, .text = facts[at].text, .state = @tagName(facts[at].state) });
    if (nothing(a, facts[at])) |why| {
        out.event("forgot", .{ .id = id, .deleted = false, .state = @tagName(facts[at].state), .message = why });
    } else delete(srv, &cx, &out, facts, at, reason, Gone{ .conn = conn }) catch |e| return out.fail(&cx, e);
    sse.done(conn) catch {};
}

/// Why a delete cannot run now, else null with the learning lock taken: one learn or delete at a time, under --slide.
pub fn admit(srv: *Server, a: Allocator) ?Refused {
    if (!srv.teacher.learning.tryLock()) return .{ .code = 503, .message = busy };
    switch (slide.run(srv, a, &.{}, 0)) {
        .refused => |e| {
            srv.teacher.learning.unlock(srv.io);
            return switch (e) {
                error.Unsupported => .{ .code = 501, .message = "this engine does not learn: serve its model with --slide" },
                error.Busy, error.Closed => .{ .code = 503, .message = "the engine cannot take a delete request now" },
            };
        },
        else => return null,
    }
}

fn find(facts: []const Fact, id: u32) ?usize {
    for (facts, 0..) |f, i| if (f.id == id) return i;
    return null;
}

/// Why fact `f` has nothing to delete (its neurons zero already, or none), or null.
pub fn nothing(a: Allocator, f: Fact) ?[]const u8 {
    const why: []const u8 = switch (f.state) {
        .replaced => "was replaced by a newer fact, so its neurons are zero already",
        .deleted => "was deleted already",
        .missed => "was not learned, so it holds no neurons",
        else => if (f.neurons.len == 0) "holds no neurons" else return null,
    };
    return std.fmt.allocPrint(a, "nothing to delete: fact {d} {s}", .{ f.id, why }) catch "nothing to delete";
}

/// Fact `at` zeroed, every learned fact it leaves asked again, then the change written or put back whole.
pub fn delete(srv: *Server, cx: *errors.Cx, out: *slide.Out, facts: []const Fact, at: usize, reason: []const u8, gone: anytype) !void {
    const a = cx.a;
    const id = facts[at].id;
    const began = slide.now(srv);
    // the deleted fact first, then each learned fact it leaves
    var order: std.ArrayList(Fact) = .empty;
    try order.append(a, facts[at]);
    for (facts, 0..) |f, i| if (i != at and f.state == .learned and f.neurons.len > 0) try order.append(a, f);
    const kept = order.items;
    const asks = try recheck.plan(a, kept, try slide_teacher.thinks(srv, cx, &srv.teacher));
    out.event("checking", .{ .id = id, .when = "before", .facts = @as(u32, @intCast(kept.len - 1)), .questions = @as(u32, @intCast(asks.len)) });
    const said = try recheck.ask(srv, cx, asks, false, gone);
    const was = try recheck.brings(a, kept, asks, said, null);
    const state = facts[at].state;
    const zeroed: u32 = switch (slide.run(srv, a, &.{ .forget = facts[at].neurons }, 0)) {
        .learned => |l| l.steps,
        .failed => |message| {
            _ = putBack(srv, a);
            return ended(out, id, state, message);
        },
        .refused => return ended(out, id, state, "the engine could not take the delete"),
        .none => return ended(out, id, state, "this learner keeps its lessons outside a pocket of neurons, so it cannot delete one"),
    };
    if (zeroed == 0) {
        if (state == .deleting) return settle(srv, a, out, facts[at], reason);
        return ended(out, id, state, try std.fmt.allocPrint(a, "nothing to delete: fact {d}'s neurons hold nothing in the pocket", .{id}));
    }
    // from here on it ends written or put back: an error before the write puts it back
    var written = false;
    errdefer if (!written) {
        _ = putBack(srv, a);
    };
    out.event("zeroed", .{ .id = id, .neurons = zeroed });
    log.line("slide: fact {d}'s {d} neurons are silenced; {d} questions of {d} facts asked again", .{ id, zeroed, asks.len, kept.len });
    out.event("checking", .{ .id = id, .when = "after", .facts = @as(u32, @intCast(kept.len - 1)), .questions = @as(u32, @intCast(asks.len)) });
    const after = check(srv, cx, kept, asks, said, was) catch |e| {
        const back = putBack(srv, a);
        return ended(out, id, state, try std.fmt.allocPrint(a, "{s}: the checks after the cut failed ({s}), nothing was written", .{ if (back) "put back" else "not put back (restart the server)", slide.words(cx, e) }));
    };
    for (kept, 0..) |f, i| out.event("checked", .{ .id = f.id, .deleted = i == 0, .before = recheck.recalled(asks, was, i), .after = recheck.recalled(asks, after.got, i) });
    if (after.miss) |m| {
        const why = try describe(a, kept, asks, after.said, m);
        const back = putBack(srv, a);
        log.line("slide: the delete of fact {d} is {s}: {s}", .{ id, if (back) "put back" else "not put back", why });
        srv.slide.note(id, try std.fmt.allocPrint(a, "a delete was put back: {s}", .{why}));
        return ended(out, id, state, try std.fmt.allocPrint(a, "{s}, nothing was written: {s}", .{ if (back) "put back as it was" else "not put back (restart the server; the folder is as it was)", why }));
    }
    // the graph file says `deleting` before any shard is written, so it never claims more than is on disk
    srv.slide.record(id, .deleting) catch |e| {
        const back = putBack(srv, a);
        return ended(out, id, state, try std.fmt.allocPrint(a, "the graph file could not be written ({s}), so the delete was {s} and nothing was written", .{ @errorName(e), if (back) "put back" else "not put back (restart the server)" }));
    };
    if (slide.saveAll(srv, a)) |failed| return unsaved(srv, a, out, id, failed);
    written = true;
    srv.teacher.forgetFact(srv.io, try questionsOf(a, facts[at]));
    srv.slide.remove(id, slide.now(srv), reason) catch |e| return ended(out, id, .deleting, try std.fmt.allocPrint(a, "the shards are written without the fact, but the graph file could not be ({s}): it still marks the fact deleting; send the delete again to settle it", .{@errorName(e)}));
    log.line("slide: fact {d} is deleted in {d:.1} s: {d} neurons silenced and written, {d} other facts checked", .{ id, slide.seconds(srv, began), zeroed, kept.len - 1 });
    out.event("forgot", .{ .id = id, .deleted = true, .state = "deleted", .neurons = zeroed, .message = try outcome(a, facts, id, zeroed, kept.len - 1) });
}

/// The last event when a delete did not complete: `state` is the fact's state as the graph file holds it.
fn ended(out: *slide.Out, id: u32, state: slide_graph.State, message: []const u8) void {
    out.event("forgot", .{ .id = id, .deleted = false, .state = @tagName(state), .message = message });
}

/// Shards that could not be written: the delete put back and the folder written again as it was, the graph after it.
fn unsaved(srv: *Server, a: Allocator, out: *slide.Out, id: u32, failed: []const u8) !void {
    const back = putBack(srv, a);
    const again = if (back) slide.saveAll(srv, a) else failed;
    if (again != null) return ended(out, id, .deleting, try std.fmt.allocPrint(a, "the shards could not be written ({s}), and {s}: the folder may hold part of the delete, and the graph file marks the fact deleting; send the delete again once the folder can be written", .{ failed, if (back) "writing them back as they were failed too" else "the delete could not be put back" }));
    srv.slide.record(id, .learned) catch |e| return ended(out, id, .deleting, try std.fmt.allocPrint(a, "the shards could not be written ({s}), so the delete was put back and the folder written as it was, but the graph file could not be ({s}) and still marks the fact deleting; send the delete again to settle it", .{ failed, @errorName(e) }));
    return ended(out, id, .learned, try std.fmt.allocPrint(a, "the shards could not be written ({s}), so the delete was put back and the folder written as it was: the folder and the graph file are as they were", .{failed}));
}

/// A fact left `deleting` whose neurons are silent already: an earlier delete reached the shards, so it is recorded.
fn settle(srv: *Server, a: Allocator, out: *slide.Out, f: Fact, reason: []const u8) !void {
    srv.teacher.forgetFact(srv.io, try questionsOf(a, f));
    srv.slide.remove(f.id, slide.now(srv), reason) catch |e| return ended(out, f.id, .deleting, try std.fmt.allocPrint(a, "its neurons are silent, so an earlier delete reached the shards, but the graph file could not be written ({s}) and still marks it deleting", .{@errorName(e)}));
    log.line("slide: fact {d}'s earlier delete had reached the shards; the graph file now records it", .{f.id});
    out.event("forgot", .{ .id = f.id, .deleted = true, .state = "deleted", .neurons = @as(u32, 0), .message = "deleted: an earlier delete had reached the shards (its neurons were silent already), and the graph file now records it" });
}

/// The questions a fact's checks asked, which the teacher's replayed answers are dropped by.
fn questionsOf(a: Allocator, f: Fact) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (f.answers) |x| try out.append(a, x.question);
    try out.appendSlice(a, f.questions);
    return out.items;
}

/// The checks after the cut: what each ask says now, whether it brings its fact back, and the first thing wrong.
const After = struct { said: []const []const u8, got: []const bool, miss: ?recheck.Miss };

fn check(srv: *Server, cx: *errors.Cx, kept: []const Fact, asks: []const recheck.Ask, before: []const []const u8, was: []const bool) !After {
    const a = cx.a;
    const stays: Stays = .{};
    const said = try recheck.ask(srv, cx, asks, false, stays);
    const got = try recheck.brings(a, kept, asks, said, before);
    // the deleted fact's questions that still give it, asked of the model as shipped: its own guess is no trace
    var still: std.ArrayList(recheck.Ask) = .empty;
    var where: std.ArrayList(usize) = .empty;
    for (asks, got, 0..) |x, g, i| if (x.fact == 0 and g) {
        try still.append(a, x);
        try where.append(a, i);
    };
    const shipped = try a.alloc(bool, asks.len);
    @memset(shipped, false);
    if (still.items.len > 0) {
        const base = try recheck.brings(a, kept, still.items, try recheck.ask(srv, cx, still.items, true, stays), null);
        for (where.items, base) |i, b| shipped[i] = b;
    }
    var miss = recheck.judge(asks, was, got, shipped, kept.len);
    // a test build's failing check (-Dslide-test-hooks with TF_SLIDE_FORGET_FAIL): release builds never set it
    if (miss == null and srv.config.slide_forget_fail != 0) for (kept[1..], 1..) |f, i| if (f.id == srv.config.slide_forget_fail) {
        miss = .{ .fact = i, .why = "a test build's TF_SLIDE_FORGET_FAIL fails its check" };
    };
    return .{ .said = said, .got = got, .miss = miss };
}

/// The last delete put back bit for bit; whether the engine did.
fn putBack(srv: *Server, a: Allocator) bool {
    return switch (slide.run(srv, a, &.{ .unforget = true }, 0)) {
        .learned => true,
        else => false,
    };
}

/// What went wrong, in words: whose fact, which question and how it was asked, and what the model said.
pub fn describe(a: Allocator, kept: []const Fact, asks: []const recheck.Ask, said: []const []const u8, m: recheck.Miss) ![]const u8 {
    const f = kept[m.fact];
    const who = if (m.fact == 0) try std.fmt.allocPrint(a, "the deleted fact (\"{s}\")", .{f.text}) else try std.fmt.allocPrint(a, "fact {d} (\"{s}\")", .{ f.id, f.text });
    const i = m.ask orelse return std.fmt.allocPrint(a, "{s}: {s}", .{ who, m.why });
    // at most 240 bytes of the reply, cut where a character begins
    var n = @min(said[i].len, 240);
    while (n > 0 and n < said[i].len and said[i][n] & 0xc0 == 0x80) n -= 1;
    const reply = said[i][0..n];
    return std.fmt.allocPrint(a, "{s}, asked \"{s}\" with thinking {s}: {s}; it said: {s}", .{ who, asks[i].question, if (asks[i].thinking) "on" else "off", m.why, reply });
}

/// The kept delete in words, and the facts it once replaced, which it leaves replaced.
pub fn outcome(a: Allocator, facts: []const Fact, id: u32, zeroed: u32, left: usize) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.print(a, "deleted: the input weights of its {d} neurons zeroed and written", .{zeroed});
    if (left == 0) try text.appendSlice(a, ", with no other learned fact to check") else try text.print(a, ", and the {d} learned fact{s} it leaves still come{s} back", .{ left, if (left == 1) "" else "s", if (left == 1) "s" else "" });
    const mark = try std.fmt.allocPrint(a, "replaced by fact {d}", .{id});
    for (facts) |f| if (f.state == .replaced and std.mem.eql(u8, f.message, mark)) try text.print(a, "; fact {d} (\"{s}\"), which it replaced, stays replaced", .{ f.id, f.text });
    return text.items;
}
