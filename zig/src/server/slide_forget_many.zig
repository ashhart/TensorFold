//! Explicit fact IDs selected for one delete; graph links never add another fact to the transaction.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const graph = @import("slide_graph.zig");
const forget = @import("slide_forget.zig");
const slide = @import("slide.zig");
const sse = @import("sse.zig");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const A = std.mem.Allocator;

pub const Batch = struct {
    ids: []const u32,
    states: []const graph.State,
    facts: []const graph.Fact,
    neurons: []const api.Neuron,
    questions: []const []const u8,
};

pub fn indices(a: A, body: json.Value, facts: []const graph.Fact) ![]const usize {
    const raw = (body.field("ids") orelse return error.MissingIds).items() orelse return error.BadIds;
    if (raw.len == 0 or raw.len > 256) return error.BadIds;
    const ats = try a.alloc(usize, raw.len);
    for (raw, ats, 0..) |v, *at, i| {
        const id = v.int64() orelse return error.BadIds;
        if (id <= 0 or id > std.math.maxInt(u32)) return error.BadIds;
        at.* = for (facts, 0..) |f, j| {
            if (f.id == id) break j;
        } else return error.NoSuchFact;
        if (std.mem.indexOfScalar(usize, ats[0..i], at.*) != null) return error.DuplicateFact;
        if (forget.nothing(a, facts[at.*]) != null) return error.FactNotLearned;
    }
    return ats;
}

pub fn plan(a: A, facts: []const graph.Fact, ats: []const usize) !Batch {
    if (ats.len == 0 or ats.len > 256) return error.BadIds;
    const ids = try a.alloc(u32, ats.len);
    const states = try a.alloc(graph.State, ats.len);
    var order: std.ArrayList(graph.Fact) = .empty;
    var neurons: std.ArrayList(api.Neuron) = .empty;
    var questions: std.ArrayList([]const u8) = .empty;
    for (ats, 0..) |at, i| {
        if (at >= facts.len) return error.NoSuchFact;
        if (std.mem.indexOfScalar(usize, ats[0..i], at) != null) return error.DuplicateFact;
        const f = facts[at];
        ids[i] = f.id;
        states[i] = f.state;
        try order.append(a, f);
        adding: for (f.neurons) |x| {
            for (neurons.items) |y| if (x.layer == y.layer and x.neuron == y.neuron) continue :adding;
            try neurons.append(a, x);
        }
        for (f.answers) |x| try questions.append(a, x.question);
        try questions.appendSlice(a, f.questions);
    }
    for (facts, 0..) |f, i| if (std.mem.indexOfScalar(usize, ats, i) == null and f.state == .learned and f.neurons.len > 0) try order.append(a, f);
    return .{ .ids = ids, .states = states, .facts = order.items, .neurons = neurons.items, .questions = questions.items };
}

pub fn route(srv: *Server, conn: *Conn, a: A) void {
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |message| return slide.refuse(conn, a, 400, message),
    };
    const body = switch (json.parse(a, raw) catch return) {
        .ok => |v| v,
        .err => |message| return slide.refuse(conn, a, 400, message),
    };
    if (forget.admit(srv, a)) |r| return slide.refuse(conn, a, r.code, r.message);
    defer srv.teacher.learning.unlock(srv.io);
    const facts = srv.slide.snapshot(a) catch return slide.refuse(conn, a, 500, "out of memory");
    const ats = indices(a, body, facts) catch |e| return slide.refuse(conn, a, switch (e) {
        error.NoSuchFact => 404,
        error.FactNotLearned => 409,
        error.OutOfMemory => 500,
        else => 400,
    }, "ids must list 1 to 256 distinct learned fact IDs with neurons; every ID must exist and be deletable");
    if (!slide.keepOriginals(srv, conn, a)) return;
    sse.open(conn) catch return;
    var out: slide.Out = .{ .conn = conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    for (ats) |at| out.event("fact", .{ .id = facts[at].id, .text = facts[at].text, .state = @tagName(facts[at].state) });
    forget.deleteMany(srv, &cx, &out, facts, ats, body.strField("reason") orelse "deleted on request", @import("openai.zig").Gone{ .conn = conn }) catch |e| return out.fail(&cx, e);
    sse.done(conn) catch {};
}
