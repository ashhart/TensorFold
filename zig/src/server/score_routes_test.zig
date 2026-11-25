//! Compatibility score/rerank requests through the real server dispatcher and existing fake score hook.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const shapes = @import("score_routes.zig");
const legacy = @import("rerank.zig");
const fixtures = @import("rerank_test.zig");
const Server = @import("server.zig").Server;
const t = std.testing;

fn ask(srv: *Server, a: std.mem.Allocator, path: []const u8, body: []const u8) !json.Value {
    var cx: errors.Cx = .{ .a = a };
    return shapes.reply(srv, &cx, path, (try json.parse(a, body)).ok, fixtures.Present{});
}
fn index(row: json.Value) i64 {
    return row.get("index").?.int64().?;
}
fn value(v: json.Value) f64 {
    return if (v == .float) v.float else @floatFromInt(v.int64().?);
}

const Fixture = struct {
    text: fixtures.Text = .{},
    scorer: fixtures.Scorer = .{ .gpa = t.allocator, .table = &fixtures.table },
    pub fn open(f: *Fixture) !*Server {
        return fixtures.serve(&f.text, &f.scorer);
    }
    pub fn deinit(f: *Fixture) void {
        f.scorer.deinit();
    }
};

test "score shapes: pair scores keep request order, broadcast both sides and report usage" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "/score", "{\"model\":\"m\",\"text_1\":\"q\",\"text_2\":[\"low\",\"high\",\"mid\"]}");
    const rows = out.get("data").?.array;
    try t.expectEqual(@as(usize, 3), rows.len);
    for (rows, 0..) |row, i| {
        try t.expectEqual(@as(i64, @intCast(i)), index(row));
        try t.expectEqualStrings("score", row.get("object").?.string);
    }
    try t.expect(value(rows[0].get("score").?) < value(rows[1].get("score").?));
    try t.expectEqualStrings("list", out.get("object").?.string);
    try t.expectEqualStrings("m", out.get("model").?.string);
    try t.expect(out.get("created").?.int64().? > 0);
    try t.expect(out.get("usage").?.get("total_tokens").?.int64().? > 0);
    try t.expectEqual(@as(i64, 0), out.get("usage").?.get("completion_tokens").?.int64().?);
    const paired = try ask(srv, a, "/v1/score", "{\"text_1\":[\"q1\",\"q2\"],\"text_2\":[\"high\",\"low\"]}");
    try t.expectEqual(@as(usize, 2), paired.get("data").?.array.len);
    try t.expect(std.mem.indexOf(u8, f.scorer.prompts.items[3], "|Q:q1|D:high") != null);
    const broadcast = try ask(srv, a, "/score", "{\"text_1\":[\"q1\",\"q2\"],\"text_2\":\"mid\"}");
    try t.expectEqual(@as(usize, 2), broadcast.get("data").?.array.len);
}

test "score shapes: every shape returns the legacy route's scores, order and label logits exactly" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const docs = "[\"mid\",\"low\",\"tie\",\"high\"]";
    const body = try std.fmt.allocPrint(a, "{{\"query\":\"q\",\"documents\":{s},\"return_logits\":true}}", .{docs});
    const old = (try legacy.reply(srv, &cx, (try json.parse(a, body)).ok, fixtures.Present{})).get("results").?.array;
    const tei = (try ask(srv, a, "/rerank", try std.fmt.allocPrint(a, "{{\"query\":\"q\",\"texts\":{s}}}", .{docs}))).array;
    const raw = (try ask(srv, a, "/rerank", try std.fmt.allocPrint(a, "{{\"query\":\"q\",\"texts\":{s},\"raw_scores\":true}}", .{docs}))).array;
    const cohere = (try ask(srv, a, "/v2/rerank", try std.fmt.allocPrint(a, "{{\"query\":\"q\",\"documents\":{s}}}", .{docs}))).get("results").?.array;
    const one = (try ask(srv, a, "/score", try std.fmt.allocPrint(a, "{{\"text_1\":\"q\",\"text_2\":{s}}}", .{docs}))).get("data").?.array;
    const pairs = (try ask(srv, a, "/v1/score", try std.fmt.allocPrint(a, "{{\"text_1\":[\"q\",\"q\",\"q\",\"q\"],\"text_2\":{s}}}", .{docs}))).get("data").?.array;
    for ([_][]const json.Value{ tei, raw, cohere }) |rows| try t.expectEqual(old.len, rows.len);
    for (old, tei, raw, cohere) |want, x, r, c| {
        const i: usize = @intCast(index(want));
        const score = want.get("relevance_score").?.float;
        const meta = want.get("meta_info").?;
        try t.expectEqual(index(want), index(x));
        try t.expectEqual(index(want), index(r));
        try t.expectEqual(index(want), index(c));
        try t.expectEqual(score, value(x.get("score").?));
        try t.expectEqual(score, value(c.get("relevance_score").?));
        try t.expectEqual(meta.get("logit_yes").?.float - meta.get("logit_no").?.float, value(r.get("score").?));
        try t.expectEqual(score, value(one[i].get("score").?));
        try t.expectEqual(score, value(pairs[i].get("score").?));
    }
}

test "score shapes: TEI ranks stably, applies top_n and returns text only when requested" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "/rerank", "{\"query\":\"q\",\"texts\":[\"mid\",\"high\",\"tie\",\"low\"],\"top_n\":3,\"return_text\":true}");
    try t.expect(out == .array);
    try t.expectEqual(@as(usize, 3), out.array.len);
    try t.expectEqual(@as(i64, 1), index(out.array[0]));
    try t.expectEqual(@as(i64, 0), index(out.array[1]));
    try t.expectEqual(@as(i64, 2), index(out.array[2]));
    try t.expectEqualStrings("high", out.array[0].get("text").?.string);
    try t.expect(out.array[0].get("relevance_score") == null);
    const plain = try ask(srv, a, "/v1/rerank", "{\"query\":\"q\",\"texts\":[\"low\",\"high\"],\"top_n\":9}");
    try t.expectEqual(@as(usize, 2), plain.array.len);
    try t.expect(plain.array[0].get("text") == null);
    const alias = try ask(srv, a, "/rerank", "{\"query\":\"q\",\"texts\":[\"high\"],\"return_documents\":true}");
    try t.expectEqualStrings("high", alias.array[0].get("text").?.string);
    const past = try ask(srv, a, "/rerank", "{\"query\":\"q\",\"texts\":[\"low\",\"high\"],\"top_n\":92233720368547758080}");
    try t.expectEqual(@as(usize, 2), past.array.len);
    const nulls = try ask(srv, a, "/rerank", "{\"query\":\"q\",\"texts\":[\"high\"],\"truncate\":null,\"truncation_direction\":null,\"return_text\":null}");
    try t.expect(nulls.array[0].get("text") == null);
}

test "score shapes: raw TEI scores sort before top_n even when sigmoid scores saturate" {
    const saturated = .{ .{ "x", 1000.0, 0.0 }, .{ "y", 2000.0, 0.0 } };
    var f: Fixture = .{};
    f.scorer.table = &saturated;
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const out = try ask(srv, arena.allocator(), "/rerank", "{\"query\":\"q\",\"texts\":[\"x\",\"y\"],\"raw_scores\":true,\"top_n\":1}");
    try t.expectEqual(@as(i64, 1), index(out.array[0]));
    try t.expectEqual(@as(f64, 2000), value(out.array[0].get("score").?));
}

test "score shapes: Cohere wraps string documents and preserves supplied document objects" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "/v2/rerank", "{\"query\":\"q\",\"documents\":[\"low\",{\"text\":\"high\",\"title\":\"kept\"},\"mid\"],\"top_n\":2,\"return_documents\":true}");
    const rows = out.get("results").?.array;
    try t.expectEqual(@as(usize, 2), rows.len);
    try t.expectEqual(@as(i64, 1), index(rows[0]));
    try t.expectEqualStrings("kept", rows[0].get("document").?.get("title").?.string);
    try t.expectEqualStrings("mid", rows[1].get("document").?.get("text").?.string);
    try t.expect(std.mem.startsWith(u8, out.get("id").?.string, "rerank-"));
    try t.expectEqualStrings("2", out.get("meta").?.get("api_version").?.get("version").?.string);
    const omitted = try ask(srv, a, "/v2/rerank", "{\"query\":\"q\",\"documents\":[\"high\"]}");
    try t.expect(omitted.get("results").?.array[0].get("document") == null);
}

test "score shapes: malformed fields and all score-pair bounds refuse before inference" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { path: []const u8, body: []const u8, message: []const u8 }{
        .{ .path = "/score", .body = "{\"text_1\":[\"q\",\"r\"],\"text_2\":[\"low\",\"mid\",\"high\"]}", .message = "equal lengths" },
        .{ .path = "/score", .body = "{\"text_1\":\"q\",\"text_2\":[\"low\",3]}", .message = "text_2" },
        .{ .path = "/score", .body = "{\"text_1\":\"q\",\"text_2\":\"high\",\"top_n\":1}", .message = "top_n" },
        .{ .path = "/score", .body = "{\"model\":\"other\",\"text_1\":\"q\",\"text_2\":\"high\"}", .message = "not served" },
        .{ .path = "/rerank", .body = "{\"query\":\"q\",\"texts\":[]}", .message = "texts" },
        .{ .path = "/rerank", .body = "{\"query\":\"q\",\"texts\":[\"high\"],\"top_n\":0}", .message = "top_n" },
        .{ .path = "/rerank", .body = "{\"query\":\"q\",\"texts\":[\"high\"],\"return_text\":1}", .message = "return_text" },
        .{ .path = "/rerank", .body = "{\"query\":\"q\",\"texts\":[\"high\"],\"truncation_direction\":\"Left\"}", .message = "Right" },
        .{ .path = "/v2/rerank", .body = "{\"query\":\"q\",\"documents\":[{}]}", .message = "text" },
        .{ .path = "/v2/rerank", .body = "{\"query\":\"q\",\"documents\":[\"high\"],\"return_documents\":\"yes\"}", .message = "return_documents" },
        .{ .path = "/v2/rerank", .body = "{\"query\":\"q\",\"documents\":[\"high\"],\"max_tokens_per_doc\":32}", .message = "max_tokens_per_doc" },
    };
    for (cases) |c| {
        var cx: errors.Cx = .{ .a = a };
        try t.expectError(error.Refused, shapes.reply(srv, &cx, c.path, (try json.parse(a, c.body)).ok, fixtures.Present{}));
        try t.expectEqual(@as(u16, 400), cx.status());
        try t.expect(std.mem.indexOf(u8, cx.message, c.message) != null);
        try t.expectEqual(@as(usize, 0), f.scorer.prompts.items.len);
    }
    srv.info.context_window = 32;
    const long = try a.alloc(u8, 100);
    @memset(long, 'x');
    const body = try std.fmt.allocPrint(a, "{{\"text_1\":\"q\",\"text_2\":[\"low\",\"{s}\"]}}", .{long});
    var cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, shapes.reply(srv, &cx, "/score", (try json.parse(a, body)).ok, fixtures.Present{}));
    try t.expectEqual(@as(usize, 0), f.scorer.prompts.items.len);
    const pairs = try std.fmt.allocPrint(a, "{{\"text_1\":[\"q\",\"r\"],\"text_2\":[\"low\",\"{s}\"]}}", .{long});
    try t.expectError(error.Refused, shapes.reply(srv, &cx, "/score", (try json.parse(a, pairs)).ok, fixtures.Present{}));
    try t.expectEqual(@as(usize, 0), f.scorer.prompts.items.len);
}

test "score shapes: TEI truncation is explicit and legacy responses stay byte-identical" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a, "{\"query\":\"q\",\"documents\":[\"low\",\"high\"]}")).ok;
    var cx: errors.Cx = .{ .a = a };
    const old = try legacy.reply(srv, &cx, body, fixtures.Present{});
    const translated = try shapes.reply(srv, &cx, "/v1/rerank", body, fixtures.Present{});
    try t.expectEqualStrings(try legacy.encode(a, old), try legacy.encode(a, translated));
    const long = try a.alloc(u8, 100);
    @memset(long, 'x');
    const raw = try std.fmt.allocPrint(a, "{{\"query\":\"q\",\"texts\":[\"{s}\"],\"max_length\":40}}", .{long});
    try t.expectError(error.Refused, shapes.reply(srv, &cx, "/rerank", (try json.parse(a, raw)).ok, fixtures.Present{}));
    try t.expect(std.mem.indexOf(u8, cx.message, "truncation is disabled") != null);
    const permitted = try std.fmt.allocPrint(a, "{{\"query\":\"q\",\"texts\":[\"{s}\"],\"max_length\":40,\"truncate\":true}}", .{long});
    try t.expectEqual(@as(usize, 1), (try ask(srv, a, "/rerank", permitted)).array.len);
}

test "score shapes: dispatcher and key scopes include all aliases and keep PUT limited" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    const cases = [_][2][]const u8{
        .{ "/score", "{\"text_1\":\"q\",\"text_2\":\"high\"}" },
        .{ "/v1/score", "{\"text_1\":\"q\",\"text_2\":\"high\"}" },
        .{ "/rerank", "{\"query\":\"q\",\"texts\":[\"high\"]}" },
        .{ "/v2/rerank", "{\"query\":\"q\",\"documents\":[\"high\"]}" },
    };
    const scopes = @import("auth_scope.zig");
    for (cases) |c| {
        var wire: fixtures.Wire = .{};
        defer wire.bytes.deinit(t.allocator);
        try fixtures.dispatch(srv, c[0], c[1], &wire);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "200 OK") != null);
        try t.expect(scopes.allows(.{ .inference = true }, "POST", c[0]));
        try t.expect(!scopes.allows(.{ .read = true }, "POST", c[0]));
    }
    try t.expect(!scopes.allows(.{ .inference = true }, "PUT", "/score"));
    try t.expect(!scopes.allows(.{ .inference = true }, "PUT", "/v2/rerank"));
    for ([_][]const u8{ "/score", "/v1/score", "/v2/rerank" }) |path| {
        var put: fixtures.Wire = .{};
        defer put.bytes.deinit(t.allocator);
        try fixtures.dispatchAs(srv, "PUT", path, cases[0][1], &put);
        try t.expect(std.mem.indexOf(u8, put.bytes.items, "501") != null);
    }
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const body = (try json.parse(cx.a, cases[0][1])).ok;
    srv.info.rerank = null;
    try t.expectError(error.Refused, shapes.reply(srv, &cx, "/score", body, fixtures.Present{}));
    try t.expectEqualStrings("this model is not a reranker", cx.message);
    for (cases) |c| {
        var wire: fixtures.Wire = .{};
        defer wire.bytes.deinit(t.allocator);
        try fixtures.dispatch(srv, c[0], c[1], &wire);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "400 Bad Request") != null);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "this model is not a reranker") != null);
    }
    srv.info.rerank = .{ .yes = 7, .no = 9 };
    f.scorer.hook = false;
    srv.engine = f.scorer.engine();
    for (cases) |c| {
        try t.expectError(error.Refused, shapes.reply(srv, &cx, c[0], (try json.parse(cx.a, c[1])).ok, fixtures.Present{}));
        try t.expectEqualStrings("this engine does not score labels", cx.message);
    }
}

test "score shapes: a departed client cancels every new shape before scoring" {
    var f: Fixture = .{};
    defer f.deinit();
    const srv = try f.open();
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const Gone = struct {
        pub fn check(_: @This()) bool {
            return true;
        }
    };
    const cases = [_][2][]const u8{
        .{ "/score", "{\"text_1\":\"q\",\"text_2\":\"high\"}" },
        .{ "/rerank", "{\"query\":\"q\",\"texts\":[\"high\"]}" },
        .{ "/v2/rerank", "{\"query\":\"q\",\"documents\":[\"high\"]}" },
    };
    for (cases) |c| try t.expectError(error.Cancelled, shapes.reply(srv, &cx, c[0], (try json.parse(cx.a, c[1])).ok, Gone{}));
    try t.expectEqual(@as(usize, 0), f.scorer.prompts.items.len);
}
