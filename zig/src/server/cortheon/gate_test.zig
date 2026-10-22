//! Research refusals and cancellation visit production request paths, with exact output and ownership counts.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const fixture = @import("cortheon_test.zig");
const GateEngine = @import("gate_engine.zig").Engine;
const openai = @import("../openai.zig");
const Conn = @import("../http_conn.zig").Conn;
const t = std.testing;
pub const question = "What is the latest release of Zig?";
pub const planner = "Zig latest release notes";

fn ask(r: *fixture.Rig, a: std.mem.Allocator, conn: *Conn, got: *fixture.Got, request_body: []const u8) !void {
    openai.run(r.srv, a, got.out(), .{ .conn = conn }, true, (try json.parse(a, request_body)).ok);
}
fn body(a: std.mem.Allocator, streaming: bool, research: bool) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"stream\":{s},\"max_tokens\":1,\"cortheon\":{s},\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]}}", .{ if (streaming) "true" else "false", if (research) "true" else "false", if (research) question else "x" });
}
fn clean(r: *fixture.Rig) !void {
    try t.expectEqual(@as(i64, 0), r.srv.preparing.load(.acquire));
    try t.expectEqual(@as(i64, 0), r.minimum.load(.acquire));
    try t.expect(r.observations.load(.acquire) > 0);
}
fn next(r: *fixture.Rig, a: std.mem.Allocator) !void {
    r.srv.config.compact_at = null;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: fixture.Got = .{ .a = a };
    try ask(r, a, &conn, &got, "{\"cortheon\":false,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"ordinary\"}]}");
    try t.expectEqual(@as(?u16, 200), got.status);
    try t.expectEqual(@as(usize, 1), got.replies);
    try clean(r);
}

test "context gate: late refusals at both windows terminate once and the same server serves next" {
    for ([_]u32{ 2000, 2400 }) |window| for ([_]bool{ false, true }) |streaming| {
        var r: fixture.Rig = .{ .eng = .{ .script = &.{planner} } };
        try r.start(true);
        defer r.stop();
        r.srv.info.context_window = window;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var got: fixture.Got = .{ .a = a };
        try ask(&r, a, &conn, &got, try body(a, streaming, true));
        try t.expect(r.web.searches > 0);
        try t.expectEqual(@as(usize, 1), r.eng.prompts.items.len);
        if (streaming) {
            try t.expect(got.opened);
            try t.expectEqual(@as(usize, 1), got.opens);
            try t.expectEqual(@as(usize, 1), got.errors);
            try t.expectEqual(@as(usize, 1), got.terminals);
            try t.expectEqual(@as(usize, 0), got.replies);
            try t.expectEqual(@as(usize, 0), got.after_terminal);
        } else {
            try t.expectEqual(@as(?u16, 400), got.status);
            try t.expectEqual(@as(usize, 1), got.replies);
            try t.expectEqual(@as(usize, 0), got.opens);
        }
        try clean(&r);
        try next(&r, a);
    };
}

test "context gate: pre-prepare and compact-prepare refusals retain stream and whole contracts" {
    for ([_]bool{ false, true }) |compacting| for ([_]bool{ false, true }) |streaming| {
        var r: fixture.Rig = .{};
        try r.start(false);
        defer r.stop();
        r.srv.info.context_window = 8;
        if (compacting) r.srv.config.compact_at = .{ .fraction = 0.8 };
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var got: fixture.Got = .{ .a = a };
        try ask(&r, a, &conn, &got, try body(a, streaming, false));
        try t.expectEqual(@as(usize, 0), r.eng.prompts.items.len);
        if (streaming) {
            try t.expectEqual(@as(usize, 1), got.opens);
            try t.expectEqual(@as(usize, 1), got.errors);
            try t.expectEqual(@as(usize, 1), got.terminals);
            try t.expectEqual(@as(usize, 0), got.replies);
        } else try t.expectEqual(@as(?u16, 400), got.status);
        try clean(&r);
        r.srv.info.context_window = 4096;
        try next(&r, a);
    };
}

test "context gate: non-context failures keep HTTP 400 500 503 and never open SSE" {
    for ([_]u32{ 2000, 2400 }) |window| for ([_]api.SubmitError{ error.InvalidRequest, error.Stale, error.Busy }) |reason| for ([_]bool{ false, true }) |streaming| {
        var r: fixture.Rig = .{ .eng = .{ .refuse_at = 0, .refusal = reason } };
        try r.start(false);
        r.srv.info.context_window = window;
        defer r.stop();
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var got: fixture.Got = .{ .a = a };
        try ask(&r, a, &conn, &got, try body(a, streaming, false));
        try t.expectEqual(@as(?u16, if (reason == error.InvalidRequest) 400 else if (reason == error.Stale) 500 else 503), got.status);
        try t.expectEqual(@as(usize, 0), got.opens);
        try clean(&r);
        r.eng.refuse_at = null;
        try next(&r, a);
    };
}

fn disconnect(ctx: *anyopaque) void {
    const conn: *Conn = @ptrCast(@alignCast(ctx));
    conn.broken = true;
}

test "context gate: search cancellation never submits a final request or invents an extra cancel" {
    for ([_]u32{ 2000, 2400 }) |window| {
        var r: fixture.Rig = .{ .eng = .{ .script = &.{planner} } };
        try r.start(true);
        defer r.stop();
        r.srv.info.context_window = window;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        r.web.on_search = .{ .ctx = &conn, .call = disconnect };
        var engine: GateEngine = .{ .original = &r.eng };
        defer engine.join();
        r.srv.engine = engine.engine();
        var got: fixture.Got = .{ .a = a };
        try ask(&r, a, &conn, &got, try body(a, true, true));
        try t.expectEqual(@as(usize, 1), engine.submissions);
        try t.expectEqual(@as(usize, 0), engine.cancellations);
        try t.expectEqual(@as(usize, 1), r.web.searches);
        try t.expectEqual(@as(usize, 0), got.opens);
        try t.expectEqual(@as(usize, 0), got.replies);
        try clean(&r);
        r.web.on_search = null;
        try next(&r, a);
    }
}

test "context gate: output Closed cancels one admitted research request and drains the sink" {
    for ([_]bool{ false, true }) |event_closed| {
        var r: fixture.Rig = .{ .eng = .{ .script = &.{planner} } };
        try r.start(true);
        defer r.stop();
        r.srv.info.context_window = 32768;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var engine: GateEngine = .{ .original = &r.eng, .at = 1 };
        defer engine.join();
        r.srv.engine = engine.engine();
        var got: fixture.Got = .{ .a = a, .close_open = !event_closed, .close_event = event_closed };
        try ask(&r, a, &conn, &got, try body(a, true, true));
        engine.join();
        try t.expectEqual(@as(usize, 2), engine.submissions);
        try t.expectEqual(@as(usize, 1), engine.cancellations);
        try t.expect(!engine.active.load(.acquire) and engine.held == null);
        try t.expectEqual(@as(usize, 1), got.opens);
        try t.expectEqual(@as(usize, 0), got.replies);
        try clean(&r);
        engine.at = null;
        try next(&r, a);
    }
}

test "context gate: cancellation during compaction drains one submitted note and releases preparing" {
    var r: fixture.Rig = .{};
    try r.start(false);
    defer r.stop();
    r.srv.info.context_window = 4096;
    r.srv.config.compact_at = .{ .fraction = 0.25 };
    r.srv.config.compact_keep = 1;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const long = try a.alloc(u8, 2800);
    @memset(long, 'x');
    const raw = try std.fmt.allocPrint(a, "{{\"stream\":true,\"cortheon\":false,\"max_tokens\":32,\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}},{{\"role\":\"assistant\",\"content\":\"old\"}},{{\"role\":\"user\",\"content\":\"next\"}}]}}", .{long});
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var engine: GateEngine = .{ .original = &r.eng, .at = 0, .disconnect = &conn };
    defer engine.join();
    r.srv.engine = engine.engine();
    var got: fixture.Got = .{ .a = a };
    try ask(&r, a, &conn, &got, raw);
    engine.join();
    try t.expectEqual(@as(usize, 1), engine.submissions);
    try t.expectEqual(@as(usize, 1), engine.cancellations);
    try t.expect(engine.held == null and !engine.active.load(.acquire));
    try t.expectEqual(@as(usize, 0), got.opens);
    try clean(&r);
    engine.at = null;
    try next(&r, a);
}

test "context gate: an already-open engine failure emits one error and terminal without success" {
    var r: fixture.Rig = .{};
    try r.start(false);
    defer r.stop();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var engine: GateEngine = .{ .original = &r.eng, .at = 0, .fail = true };
    defer engine.join();
    r.srv.engine = engine.engine();
    var got: fixture.Got = .{ .a = a };
    try ask(&r, a, &conn, &got, try body(a, true, false));
    try t.expectEqual(@as(usize, 1), got.opens);
    try t.expectEqual(@as(usize, 1), got.errors);
    try t.expectEqual(@as(usize, 1), got.terminals);
    try t.expectEqual(@as(usize, 0), got.after_terminal);
    try t.expectEqual(@as(usize, 0), got.replies);
    try clean(&r);
    engine.at = null;
    try next(&r, a);
}

test "context gate: successful research at both windows releases preparing before an ordinary next request" {
    for ([_]u32{ 2000, 2400 }) |window| for ([_]bool{ false, true }) |streaming| {
        var r: fixture.Rig = .{ .eng = .{ .script = &.{ planner, "OK" } }, .web = .{ .empty = true }, .text = .{ .pair_tokens = true } };
        try r.start(true);
        defer r.stop();
        r.srv.info.context_window = window;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var got: fixture.Got = .{ .a = a };
        try ask(&r, a, &conn, &got, try body(a, streaming, true));
        try t.expect(r.web.searches > 0);
        try t.expectEqual(@as(usize, 2), r.eng.prompts.items.len);
        if (streaming) {
            try t.expectEqual(@as(usize, 1), got.opens);
            try t.expectEqual(@as(usize, 0), got.errors);
            try t.expectEqual(@as(usize, 1), got.terminals);
        } else try t.expectEqual(@as(?u16, 200), got.status);
        try clean(&r);
        try next(&r, a);
    };
}

test "context gate: fixture byte pairs preserve complete odd even and UTF8 prompt bytes" {
    for ([_]bool{ false, true }) |pairs| for ([_][]const u8{ "x", "pair", "odd", "Zig λ" }) |input| {
        var codec: fixture.Text = .{ .pair_tokens = pairs };
        const text = codec.text();
        const ids = try text.encode(t.allocator, input, false);
        defer t.allocator.free(ids);
        const roundtrip = try text.decode(t.allocator, ids);
        defer t.allocator.free(roundtrip);
        try t.expectEqualStrings(input, roundtrip);
    };
}
