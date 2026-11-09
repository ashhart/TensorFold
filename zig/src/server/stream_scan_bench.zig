//! `zig build bench-stream-scan`: a long thinking reply through the real stream routes, each frame timed as written.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const json = @import("json");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const scan = @import("stream_scan_text.zig");
const Allocator = std.mem.Allocator;

const io = std.testing.io;
const full = builtin.mode != .debug;

/// Answers ``ids`` in chunks of ``stride`` tokens, as a decode round hands them to the request.
const Script = struct {
    ids: []const u32,
    stride: usize,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 17 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var at: usize = 0;
        while (at < e.ids.len) : (at += e.stride) sink.event(sink.ctx, id, &.{ .tokens = e.ids[at..@min(at + e.stride, e.ids.len)] });
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

/// Every write the route makes, and the awake clock when it made it.
const Capture = struct {
    gpa: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    marks: std.ArrayList(struct { end: usize, ns: i96 }) = .empty,

    fn write(ctx: *anyopaque, data: []const u8) error{Closed}!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.bytes.appendSlice(c.gpa, data) catch return error.Closed;
        c.marks.append(c.gpa, .{ .end = c.bytes.items.len, .ns = std.Io.Clock.awake.now(io).toNanoseconds() }) catch return error.Closed;
    }

    fn deinit(c: *Capture) void {
        c.bytes.deinit(c.gpa);
        c.marks.deinit(c.gpa);
    }
};

pub const Route = enum { chat, messages, responses };

/// The process's page faults so far: each one a page the request's memory touched first; zero without getrusage.
fn faults() isize {
    if (comptime builtin.os.tag == .windows) return 0;
    const u = std.posix.getrusage(std.posix.rusage.SELF);
    return u.minflt + u.majflt;
}
const Kind = enum { reasoning, content, value };

/// One run's measures: each frame's reply token and write time, and the request's own totals.
const Run = struct {
    at: std.ArrayList(usize) = .empty, // the reply token a frame's text reached
    ns: std.ArrayList(i96) = .empty,
    handler_ns: i96 = 0,
    arena_bytes: usize = 0,
    faults: isize = 0,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    names: usize = 0,
};

fn body(a: Allocator, route: Route, tools: bool, max: usize) ![]const u8 {
    const schema = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    return switch (route) {
        .chat => std.fmt.allocPrint(a, "{{\"model\":\"m\",\"stream\":true,\"max_tokens\":{d},\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]{s}}}", .{
            max, if (tools) ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":" ++ schema ++ "}}]" else "",
        }),
        .messages => std.fmt.allocPrint(a, "{{\"model\":\"m\",\"stream\":true,\"max_tokens\":{d},\"thinking\":{{\"type\":\"adaptive\"}},\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]{s}}}", .{
            max, if (tools) ",\"tools\":[{\"name\":\"write_file\",\"input_schema\":" ++ schema ++ "}]" else "",
        }),
        .responses => std.fmt.allocPrint(a, "{{\"model\":\"m\",\"stream\":true,\"store\":false,\"max_output_tokens\":{d},\"input\":\"x\"{s}}}", .{
            max, if (tools) ",\"tools\":[{\"type\":\"function\",\"name\":\"write_file\",\"parameters\":" ++ schema ++ "}]" else "",
        }),
    };
}

/// The phase text a frame carries, read from the route's own event shape.
fn pieces(frame: json.Value, route: Route, run: *Run, a: Allocator) !?Kind {
    var kind: ?Kind = null;
    switch (route) {
        .chat => {
            const choices = frame.get("choices") orelse return null;
            if (choices != .array or choices.array.len == 0) return null;
            const d = choices.array[0].get("delta") orelse return null;
            if (d.strField("reasoning_content")) |s| if (s.len > 0) {
                try run.reasoning.appendSlice(a, s);
                kind = .reasoning;
            };
            if (d.strField("content")) |s| if (s.len > 0) {
                try run.content.appendSlice(a, s);
                kind = .content;
            };
            if (d.get("tool_calls")) |list| for (list.array) |call| {
                const f = call.get("function") orelse continue;
                if (f.strField("name") != null) run.names += 1;
                if (f.strField("arguments")) |s| try run.arguments.appendSlice(a, s);
                kind = .value;
            };
        },
        .messages => {
            if (frame.get("content_block")) |b| if (b.typeIs("tool_use")) {
                run.names += 1;
            };
            const d = frame.get("delta") orelse return null;
            if (d.strField("thinking")) |s| try run.reasoning.appendSlice(a, s) else if (d.strField("text")) |s| try run.content.appendSlice(a, s) else if (d.strField("partial_json")) |s| try run.arguments.appendSlice(a, s) else return null;
            kind = if (d.typeIs("thinking_delta")) .reasoning else if (d.typeIs("text_delta")) .content else .value;
        },
        .responses => {
            const t = frame.strField("type") orelse return null;
            if (std.mem.eql(u8, t, "response.output_item.added") and frame.get("item").?.typeIs("function_call")) run.names += 1;
            const s = frame.strField("delta") orelse return null;
            kind = if (std.mem.endsWith(u8, t, "reasoning_text.delta")) .reasoning else if (std.mem.endsWith(u8, t, "output_text.delta")) .content else if (std.mem.endsWith(u8, t, "arguments.delta")) .value else return null;
            try (switch (kind.?) {
                .reasoning => &run.reasoning,
                .content => &run.content,
                .value => &run.arguments,
            }).appendSlice(a, s);
        },
    }
    return kind;
}

/// The reply token whose text ends at or after ``bytes`` of its phase.
fn tokenAt(ends: []const usize, bytes: usize) usize {
    return std.sort.lowerBound(usize, ends, bytes, struct {
        fn order(b: usize, e: usize) std.math.Order {
            return std.math.order(b, e);
        }
    }.order);
}

/// One streamed request: the route's handler on a connection whose writes are timed, then its frames read back.
fn once(gpa: Allocator, srv: *Server, route: Route, r: scan.Reply, tools: bool, a: Allocator) !Run {
    const request = try body(a, route, tools, r.ids.len + 64);
    var capture: Capture = .{ .gpa = gpa };
    defer capture.deinit();
    try capture.bytes.ensureTotalCapacity(gpa, 64 << 20);
    try capture.marks.ensureTotalCapacity(gpa, 4 * r.ids.len + 64);
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{request.len}) }};
    var request_arena: std.heap.ArenaAllocator = .init(gpa);
    defer request_arena.deinit();
    const path = switch (route) {
        .chat => "/v1/chat/completions",
        .messages => "/v1/messages",
        .responses => "/v1/responses",
    };
    var conn: http.Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, request), .end = request.len, .gpa = gpa, .version = "HTTP/1.1", .method = "POST", .path = path, .headers = &headers, .write_hooks = .{ .ctx = &capture, .write = Capture.write } };
    const ra = request_arena.allocator();
    const faulted = faults();
    const start = std.Io.Clock.awake.now(io).toNanoseconds();
    switch (route) {
        .chat => @import("openai.zig").post(srv, &conn, ra, true),
        .messages => @import("anthropic.zig").post(srv, &conn, ra),
        .responses => @import("responses.zig").post(srv, &conn, ra),
    }
    var run: Run = .{ .handler_ns = std.Io.Clock.awake.now(io).toNanoseconds() - start, .arena_bytes = request_arena.queryCapacity(), .faults = faults() - faulted };
    var from: usize = 0;
    var unplaced: usize = 0; // frames that added no text (a call's head) take the next text frame's token
    for (capture.marks.items) |m| {
        const frame = capture.bytes.items[from..m.end];
        from = m.end;
        const data = std.mem.indexOf(u8, frame, "data: ") orelse continue;
        const line = std.mem.trimEnd(u8, frame[data + 6 ..], "\n");
        if (std.mem.eql(u8, line, "[DONE]")) continue;
        const v = switch (try json.parseText(a, line)) {
            .ok => |v| v,
            .err => return error.BadFrame,
        };
        const before = run.reasoning.items.len + run.content.items.len + run.arguments.items.len;
        const kind = try pieces(v, route, &run, a) orelse continue;
        const token = switch (kind) {
            .reasoning => tokenAt(r.reasoning_ends, run.reasoning.items.len),
            .content => r.content_from + tokenAt(r.content_ends, run.content.items.len),
            .value => r.value_from + tokenAt(r.value_ends, run.arguments.items.len -| "{\"path\":\"notes.txt\",\"content\":\"".len),
        };
        try run.at.append(a, token);
        try run.ns.append(a, m.ns);
        if (before == run.reasoning.items.len + run.content.items.len + run.arguments.items.len) {
            unplaced += 1;
            continue;
        }
        for (run.at.items[run.at.items.len - 1 - unplaced ..]) |*t| t.* = token;
        unplaced = 0;
    }
    for (run.at.items[run.at.items.len - unplaced ..]) |*t| t.* = r.ids.len;
    return run;
}

/// The last frame before ``i`` at an earlier reply token: frames of held bytes reach no further token.
fn earlier(run: *const Run, i: usize) usize {
    var j = i - 1;
    while (j > 0 and run.at.items[j] >= run.at.items[i]) j -= 1;
    return j;
}

/// One chunk's time in microseconds: frame ``i`` one chunk on from frame ``i - 1``; null across silent chunks.
fn oneChunk(run: *const Run, i: usize, stride: usize) ?f64 {
    if (run.at.items[i] != run.at.items[i - 1] + stride) return null;
    return per(run.ns.items[i] - run.ns.items[i - 1], 1e3);
}

/// The request thread's time a chunk around reply token ``mid``: frames ``half`` tokens either side, per chunk.
fn perChunk(run: *const Run, mid: usize, half: usize, stride: usize) ?struct { us: f64, worst_us: f64 } {
    var first: ?usize = null;
    var last: usize = 0;
    var worst: f64 = 0;
    for (run.at.items, 0..) |t, i| if (t + half >= mid and t <= mid + half) {
        if (first) |_| worst = @max(worst, oneChunk(run, i, stride) orelse 0);
        if (first == null) first = i;
        last = i;
    };
    const f = first orelse return null;
    const tokens = run.at.items[last] - run.at.items[f];
    if (tokens == 0) return null;
    const chunks = per(@intCast(tokens), @floatFromInt(stride));
    return .{ .us = per(run.ns.items[last] - run.ns.items[f], 1e3 * chunks), .worst_us = worst };
}

pub const Case = struct { family: scan.Family, route: Route, stride: usize };

/// ``n`` over ``unit``, for printing.
fn per(n: i96, unit: f64) f64 {
    return @as(f64, @floatFromInt(n)) / unit;
}

/// The case's reply streamed through its route; checks the stream's text against the reply's and prints the times.
fn measure(case: Case, shape: scan.Shape) !void {
    const gpa = std.heap.smp_allocator;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plain: scan.Text = .{};
    const k = if (case.family == .kimi) try scan.kimi(gpa) else null;
    defer if (k) |t| t.deinit();
    const text = if (k) |t| t.text() else plain.text();
    const r = try scan.reply(a, case.family, shape, if (k) |t| t.text() else null);
    var script: Script = .{ .ids = r.ids, .stride = case.stride };
    const srv = try Server.init(gpa, io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
    defer srv.deinit();
    const tools = case.family != .none;
    const run = try once(gpa, srv, case.route, r, tools, a);
    try std.testing.expectEqualStrings(std.mem.trim(u8, r.reasoning, " \n"), run.reasoning.items);
    try std.testing.expectEqualStrings(std.mem.trimEnd(u8, r.content, " \n"), run.content.items);
    if (tools) {
        try std.testing.expectEqual(@as(usize, 1), run.names);
        const want = try std.mem.concat(a, u8, &.{ "{\"path\":\"notes.txt\",\"content\":\"", r.value, "\"}" });
        try std.testing.expectEqualStrings(want, run.arguments.items);
    }
    if (!full) return;
    const span = run.ns.items[run.ns.items.len - 1] - run.ns.items[0];
    const chunks = (r.ids.len + case.stride - 1) / case.stride;
    var worst: struct { us: f64 = 0, at: usize = 0 } = .{};
    var silence: struct { ns: i96 = 0, at: usize = 0, tokens: usize = 0 } = .{};
    for (1..run.ns.items.len) |i| {
        const gap = run.ns.items[i] - run.ns.items[i - 1];
        if (oneChunk(&run, i, case.stride)) |us| if (us > worst.us) {
            worst = .{ .us = us, .at = run.at.items[i] };
        };
        if (gap > silence.ns) silence = .{ .ns = gap, .at = run.at.items[i], .tokens = run.at.items[i] -| run.at.items[earlier(&run, i)] };
    }
    std.debug.print("\n{s}/{s} stride {d}: {d} tokens, {d} chunks, {d} frames; request arena {d:.1} MiB, {d} page faults\n", .{
        @tagName(case.family), @tagName(case.route), case.stride, r.ids.len, chunks, run.ns.items.len, per(@intCast(run.arena_bytes), 1 << 20), run.faults,
    });
    std.debug.print("  handler {d:.1} ms, frames span {d:.1} ms, {d:.2} us a chunk; worst chunk {d:.2} us at token {d}; longest silence {d:.2} ms over {d} tokens to token {d}\n", .{
        per(run.handler_ns, 1e6), per(span, 1e6), per(span, 1e3 * @as(f64, @floatFromInt(chunks))), worst.us, worst.at, per(silence.ns, 1e6), silence.tokens, silence.at,
    });
    const marks = [_]struct { []const u8, usize }{
        .{ "1k", 1_000 },                                .{ "5k", 5_000 },                                   .{ "10k", 10_000 },                          .{ "20k", 20_000 },
        .{ "30k", @min(30_000, r.content_from) -| 128 }, .{ "content", r.content_from + shape.content / 2 }, .{ "call", r.value_from + shape.value / 2 },
    };
    for (marks) |m| if (perChunk(&run, m[1], 128, case.stride)) |p|
        std.debug.print("  at {s:>7} (token {d:>6}): {d:>8.2} us a chunk, worst chunk {d:>8.2} us\n", .{ m[0], m[1], p.us, p.worst_us });
}

/// The measured cases: Qwen on every route, then GLM, DSML, Kimi and no tools on chat, one and two tokens a chunk.
pub const cases = [_]Case{
    .{ .family = .qwen, .route = .chat, .stride = 1 },     .{ .family = .qwen, .route = .chat, .stride = 2 },
    .{ .family = .qwen, .route = .messages, .stride = 1 }, .{ .family = .qwen, .route = .responses, .stride = 1 },
    .{ .family = .glm, .route = .chat, .stride = 1 },      .{ .family = .dsml, .route = .chat, .stride = 1 },
    .{ .family = .kimi, .route = .chat, .stride = 1 },     .{ .family = .none, .route = .chat, .stride = 1 },
};

test "stream scan bench: the request arena as the reasoning doubles" {
    const gpa = std.heap.smp_allocator;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sizes = if (full) [_]usize{ 7_500, 15_000, 30_000 } else [_]usize{ 50, 100, 200 };
    if (full) std.debug.print("\nrequest arena, MiB, at {d}, {d} and {d} reasoning tokens (one token a chunk)\n", .{ sizes[0], sizes[1], sizes[2] });
    for ([_]Case{ .{ .family = .qwen, .route = .chat, .stride = 1 }, .{ .family = .qwen, .route = .responses, .stride = 1 }, .{ .family = .kimi, .route = .chat, .stride = 1 } }) |case| {
        if (full) std.debug.print("  {s}/{s}:", .{ @tagName(case.family), @tagName(case.route) });
        for (sizes) |n| {
            var plain: scan.Text = .{};
            const k = if (case.family == .kimi) try scan.kimi(gpa) else null;
            defer if (k) |t| t.deinit();
            const r = try scan.reply(a, case.family, .{ .reasoning = n, .content = 100, .value = 100 }, if (k) |t| t.text() else null);
            var script: Script = .{ .ids = r.ids, .stride = case.stride };
            const srv = try Server.init(gpa, io, script.engine(), if (k) |t| t.text() else plain.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
            defer srv.deinit();
            const run = try once(gpa, srv, case.route, r, true, a);
            if (full) std.debug.print(" {d:>9.1}", .{per(@intCast(run.arena_bytes), 1 << 20)});
        }
        if (full) std.debug.print("\n", .{});
    }
}

test "stream scan bench: long thinking replies stream their reasoning, prose and call on every route" {
    const shape: scan.Shape = if (full) .{ .reasoning = 30_000, .content = 1_000, .value = 1_000 } else .{ .reasoning = 300, .content = 40, .value = 40 };
    for (cases) |case| try measure(case, shape);
}

fn boundedArena(family: scan.Family, route: Route) !void {
    const gpa = std.heap.smp_allocator;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plain: scan.Text = .{};
    const k = if (family == .kimi) try scan.kimi(gpa) else null;
    defer if (k) |t| t.deinit();
    const r = try scan.reply(a, family, .{ .reasoning = 7_500, .content = 100, .value = 100 }, if (k) |t| t.text() else null);
    var script: Script = .{ .ids = r.ids, .stride = 1 };
    const srv = try Server.init(gpa, io, script.engine(), if (k) |t| t.text() else plain.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
    defer srv.deinit();
    const run = try once(gpa, srv, route, r, true, a);
    try std.testing.expectEqualStrings(std.mem.trim(u8, r.reasoning, " \n"), run.reasoning.items);
    try std.testing.expectEqualStrings(if (family == .kimi) r.content else std.mem.trimEnd(u8, r.content, " \n"), run.content.items);
    try std.testing.expectEqual(@as(usize, 1), run.names);
    try std.testing.expect(run.arena_bytes < 80 << 20);
}

test "Server recovery: Responses request arena stays below 80 MiB at 7500 reasoning tokens" {
    try boundedArena(.qwen, .responses);
}

test "Server recovery: Kimi request arena stays below 80 MiB at 7500 reasoning tokens" {
    try boundedArena(.kimi, .chat);
}
