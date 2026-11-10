//! Responses WebSocket mode over a real server: turns, the connection cache, lanes, warmup, events, closes and limits.
const std = @import("std");
const posix = @import("sys").posix; // the layer serve.zig installs its stop handlers with
const json = @import("json");
const d = @import("drain_test.zig");
const ws = @import("responses_ws.zig");
const client = @import("responses_ws_client_test.zig");
const Client = client.Client;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// A served stub engine on its own thread; ``stop`` raises SIGTERM and joins it.
pub const Fixture = struct {
    gpa: Allocator,
    io: std.Io,
    env: std.process.Environ.Map,
    slow: d.Slow,
    tt: d.TestText = .{},
    x: d.Served = .{},
    thread: std.Thread = undefined,
    port: u16 = 0,

    pub fn start(f: *Fixture, gpa: Allocator, io: std.Io, args: @import("cli.zig").Args) !void {
        f.* = .{ .gpa = gpa, .io = io, .env = .init(gpa), .slow = .{ .gpa = gpa, .io = io } };
        try f.env.put("TENSORFOLD_NO_LIVE", "1");
        try f.env.put("TF_DRAIN_S", "5");
        f.thread = try d.startWith(gpa, io, &f.env, &f.slow, &f.tt, &f.x, args);
        f.port = d.listen_port.load(.acquire);
    }

    pub fn stop(f: *Fixture) !void {
        try posix.raise(.TERM);
        f.thread.join();
        f.env.deinit();
        try testing.expectEqual(@as(u8, 0), f.x.code);
    }
};

fn kinds(a: Allocator, events: []Value) ![]const []const u8 {
    const out = try a.alloc([]const u8, events.len);
    for (events, out) |e, *slot| slot.* = e.strField("type").?;
    return out;
}

fn expectKinds(a: Allocator, want: []const []const u8, events: []Value) !void {
    const got = try kinds(a, events);
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectEqualStrings(w, g);
}

fn expectCompleted(events: []Value) !void {
    try testing.expectEqualStrings("response.completed", client.last(events).strField("type").?);
}

fn expectError(events: []Value, status: i64, code: []const u8) !void {
    const e = client.last(events);
    try testing.expectEqualStrings("error", e.strField("type").?);
    try testing.expectEqual(status, e.get("status").?.int64().?);
    try testing.expectEqualStrings(code, client.code(events));
}

fn create(a: Allocator, fields: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"type\":\"response.create\",{s}}}", .{fields});
}

test "Responses socket: a turn equals its SSE twin; the cache continues store:false and stores whole children" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{});
    errdefer f.stop() catch {};
    const io = f.io;
    var c: Client = undefined;
    try c.open(a, io, f.port, null);
    try testing.expectEqual(@as(u16, 101), c.status);

    // 1. the same turn over SSE and over the socket: same event types in order, same final response
    const body = "\"input\":\"hello there\",\"store\":false,\"max_output_tokens\":64";
    const sse = try client.http(a, io, f.port, "POST", "/v1/responses", null, try std.fmt.allocPrint(a, "{{\"stream\":true,{s}}}", .{body}));
    try testing.expectEqual(@as(u16, 200), sse.status);
    const over_http = try client.sseEvents(a, sse.body);
    const over_socket = try c.turn(a, try create(a, body));
    try expectKinds(a, try kinds(a, over_http), over_socket);
    try testing.expect(json.equal(try client.normalized(a, client.final(over_http)), try client.normalized(a, client.final(over_socket))));
    // ``normalized`` drops the runtime stats, so the token hashes are compared on their own
    for ([_][]const u8{ "token_sha", "token_sha256" }) |k| {
        const want = client.final(over_http).get("tensorfold").?.strField(k).?;
        try testing.expectEqualStrings(want, client.final(over_socket).get("tensorfold").?.strField(k).?);
    }
    for (over_socket) |e| try testing.expect(e.get("stream_id") == null);

    // 2. store:false continues from the cache, with the earlier messages in its prompt
    const first = client.final(over_socket).strField("id").?;
    const next = try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"more\",\"store\":false,\"previous_response_id\":\"{s}\"", .{first})));
    try expectCompleted(next);
    try testing.expect(client.usage(next, "input_tokens") > client.usage(over_socket, "input_tokens"));
    try testing.expectEqualStrings(first, client.final(next).strField("previous_response_id").?);
    // a store:false miss, and a stored response read only with store:true
    try expectError(try c.turn(a, try create(a, "\"input\":\"x\",\"store\":false,\"previous_response_id\":\"resp_missing\"")), 400, "previous_response_not_found");
    const stored = try client.http(a, io, f.port, "POST", "/v1/responses", null, "{\"input\":\"kept\",\"store\":true}");
    try testing.expectEqual(@as(u16, 200), stored.status);
    const stored_id = (try json.parse(a, stored.body)).ok.strField("id").?;
    const miss = try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"x\",\"store\":false,\"previous_response_id\":\"{s}\"", .{stored_id})));
    try expectError(miss, 400, "previous_response_not_found");
    try testing.expectEqualStrings("previous_response_id", client.last(miss).get("error").?.strField("param").?);
    // on its own lane, so the default lane keeps its cached response
    try expectCompleted(try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"stream_id\":\"side\",\"input\":\"y\",\"previous_response_id\":\"{s}\"", .{stored_id}))));
    // a stored child of a cached parent is stored whole: it continues on a new socket after this one closes
    const parent = client.final(next).strField("id").?;
    const child = try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"child\",\"store\":true,\"previous_response_id\":\"{s}\"", .{parent})));
    try expectCompleted(child);
    const child_id = client.final(child).strField("id").?;
    c.close();
    var c2: Client = undefined;
    try c2.open(a, io, f.port, null);
    const later = try c2.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"later\",\"previous_response_id\":\"{s}\"", .{child_id})));
    try expectCompleted(later);
    try testing.expect(client.usage(later, "input_tokens") > client.usage(child, "input_tokens"));
    const got = try client.http(a, io, f.port, "GET", try std.fmt.allocPrint(a, "/v1/responses/{s}", .{child_id}), null, "");
    try testing.expectEqualStrings(parent, (try json.parse(a, got.body)).ok.strField("previous_response_id").?);
    c2.close();
    try f.stop();
}

test "Responses socket: lanes echo stream_id, refuse bad names and a 33rd lane; warmup counts the prompt" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{});
    errdefer f.stop() catch {};
    var c: Client = undefined;
    try c.open(a, f.io, f.port, null);

    // 3. lanes
    const lane = try c.turn(a, try create(a, "\"stream_id\":\"main\",\"input\":\"hi\",\"store\":false"));
    try expectCompleted(lane);
    for (lane) |e| try testing.expectEqualStrings("main", e.strField("stream_id").?);
    const bad = try c.turn(a, try create(a, "\"stream_id\":\"bad id\",\"input\":\"hi\",\"store\":false"));
    try expectError(bad, 400, "invalid_stream_id");
    try testing.expect(client.last(bad).get("stream_id") == null);
    for (1..32) |i| {
        const name = try std.fmt.allocPrint(a, "\"stream_id\":\"lane-{d}\",\"input\":\"w\",\"store\":false,\"generate\":false", .{i});
        try expectCompleted(try c.turn(a, try create(a, name)));
    }
    try expectError(try c.turn(a, try create(a, "\"stream_id\":\"lane-33\",\"input\":\"w\",\"store\":false,\"generate\":false")), 400, "websocket_stream_limit_reached");
    try expectCompleted(try c.turn(a, try create(a, "\"stream_id\":\"main\",\"input\":\"w\",\"store\":false,\"generate\":false")));

    // 4. warmup: three events, no output, the prompt's count; a continuation from it; a prompt over the window refused
    const warm = try c.turn(a, try create(a, "\"input\":\"warm words\",\"store\":false,\"generate\":false"));
    try expectKinds(a, &.{ "response.created", "response.in_progress", "response.completed" }, warm);
    try testing.expectEqual(@as(usize, 0), client.final(warm).get("output").?.array.len);
    try testing.expectEqual(@as(i64, 0), client.usage(warm, "output_tokens"));
    const counted = client.usage(warm, "input_tokens");
    try testing.expect(counted > 0);
    // on another lane, so the default lane keeps the warmup
    const real = try c.turn(a, try create(a, "\"stream_id\":\"main\",\"input\":\"warm words\",\"store\":false"));
    try testing.expectEqual(counted, client.usage(real, "input_tokens"));
    const warm_id = client.final(warm).strField("id").?;
    const after = try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"go on\",\"store\":false,\"previous_response_id\":\"{s}\"", .{warm_id})));
    try expectCompleted(after);
    try testing.expect(client.usage(after, "input_tokens") > counted);
    const long: [5000]u8 = @splat('x');
    const huge = try std.fmt.allocPrint(a, "\"input\":\"{s}\",\"store\":false,\"generate\":false", .{&long});
    try expectError(try c.turn(a, try create(a, huge)), 400, "context_length_exceeded");

    // 5. steering is refused, and an unknown or malformed event is a connection-level error; the socket stays open
    const steer = try c.turn(a, "{\"type\":\"response.steer\",\"previous_response_id\":\"resp_x\",\"input\":\"faster\"}");
    try testing.expectEqualStrings("response.steer.failed", client.last(steer).strField("type").?);
    try testing.expectEqualStrings("steering_not_supported", client.code(steer));
    try testing.expectEqualStrings("faster", client.last(steer).strField("input").?);
    const unknown = try c.turn(a, "{\"type\":\"response.cancel\"}");
    try expectError(unknown, 400, "invalid_event");
    try testing.expect(client.last(unknown).get("stream_id") == null);
    try expectError(try c.turn(a, "not json"), 400, "invalid_event");
    try expectCompleted(try c.turn(a, try create(a, "\"input\":\"still here\",\"store\":false,\"generate\":false")));
    c.close();
    try f.stop();
}

test "Responses socket: a close mid-run cancels it; frame errors close with their codes; a 1 MiB message runs" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{});
    errdefer f.stop() catch {};
    const io = f.io;

    // 6. a client close during a silent run stops the engine request
    var c: Client = undefined;
    try c.open(a, io, f.port, null);
    try c.send(a, try create(a, "\"input\":\"go @hang\",\"store\":false"));
    d.waitActive(io, &f.slow, 1);
    try c.sendFrame(a, .close, &.{ 0x03, 0xe8 }, true);
    try testing.expectEqual(@as(?u16, 1000), try c.closeCode(a));
    c.close();
    var waited: u32 = 0;
    while (f.slow.cancelled.load(.acquire) == 0 and waited < 500) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    try testing.expect(f.slow.cancelled.load(.acquire) != 0);
    // a 17th queued message during a run closes with 1008 and stops the run
    while (f.slow.active.load(.acquire) > 0) std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    const before = f.slow.cancelled.load(.acquire);
    var flood: Client = undefined;
    try flood.open(a, io, f.port, null);
    try flood.send(a, try create(a, "\"input\":\"go @hang\",\"store\":false"));
    d.waitActive(io, &f.slow, 1);
    for (0..17) |_| try flood.send(a, "{\"type\":\"response.cancel\"}");
    try testing.expectEqual(@as(?u16, 1008), try flood.closeCode(a));
    flood.close();
    waited = 0;
    while (f.slow.cancelled.load(.acquire) == before and waited < 500) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    try testing.expect(f.slow.cancelled.load(.acquire) != before);

    // 8. an unmasked frame closes with 1002, a binary message with 1003
    var unmasked: Client = undefined;
    try unmasked.open(a, io, f.port, null);
    try unmasked.sendFrame(a, .text, "{}", false);
    try testing.expectEqual(@as(?u16, 1002), try unmasked.closeCode(a));
    unmasked.close();
    var binary: Client = undefined;
    try binary.open(a, io, f.port, null);
    try binary.sendFrame(a, .binary, "{}", true);
    try testing.expectEqual(@as(?u16, 1003), try binary.closeCode(a));
    binary.close();
    // a valid message larger than the connection's read buffer
    var big: Client = undefined;
    try big.open(a, io, f.port, null);
    const pad = try a.alloc(u8, 1 << 20);
    @memset(pad, 'p');
    const events = try big.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"hi\",\"store\":false,\"pad\":\"{s}\"", .{pad})));
    try expectCompleted(events);
    big.close();
    try f.stop();
}

test "Responses socket: the lifetime ends with websocket_connection_limit_reached and 1000; a drain closes 1012" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{});
    errdefer f.stop() catch {};

    // 7. a short test lifetime, then a drain of an idle socket
    ws.test_lifetime_ms = 300;
    var old: Client = undefined;
    try old.open(a, f.io, f.port, null);
    const limit = try old.until(a);
    ws.test_lifetime_ms = null;
    try expectError(limit, 400, "websocket_connection_limit_reached");
    try testing.expectEqual(@as(?u16, 1000), try old.closeCode(a));
    old.close();
    var idle: Client = undefined;
    try idle.open(a, f.io, f.port, null);
    try expectCompleted(try idle.turn(a, try create(a, "\"input\":\"w\",\"store\":false,\"generate\":false")));
    try posix.raise(.TERM);
    const drained = try idle.until(a);
    try expectError(drained, 503, "server_restarting");
    try testing.expectEqual(@as(?u16, 1012), try idle.closeCode(a));
    idle.close();
    f.thread.join();
    f.env.deinit();
    try testing.expectEqual(@as(u8, 0), f.x.code);
}

test {
    _ = @import("responses_ws_gate_test.zig");
}
