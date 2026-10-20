//! /health: the open body's keys in order with the builder's identity; the keyed body and the manifest by socket.
const std = @import("std");
const api = @import("engine_api");
const auth = @import("auth.zig");
const build_identity = @import("identity.zig");
const json = @import("json");
const listener = @import("listener.zig");
const model_text = @import("model_text.zig");
const server = @import("server.zig");
const status_routes = @import("status_routes.zig");
const posix = @import("sys").posix;
const Threaded = std.Io.Threaded;
const Allocator = std.mem.Allocator;

/// The keys an open /health sent before it named the release, backend and GPU.
const earlier = [_][]const u8{ "status", "model", "model_ids", "max_batch_size", "warming", "memory", "live" };

/// Four lanes, some memory, nothing running.
const Idle = struct {
    healthy: bool = true,
    resets: std.atomic.Value(u32) = .init(0),
    scores: std.atomic.Value(u32) = .init(0),
    submits: std.atomic.Value(u32) = .init(0),
    fn engine(e: *Idle) api.Engine {
        return .{ .ctx = e, .vtable = &.{
            .info = info,
            .submit = submit,
            .cancel = cancel,
            .status = status,
            .memory = memory,
            .score = score,
        } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .lanes = 4 };
    }

    fn submit(ctx: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        _ = @as(*Idle, @ptrCast(@alignCast(ctx))).submits.fetchAdd(1, .monotonic);
        return error.Closed;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = @as(*Idle, @ptrCast(@alignCast(ctx))).healthy };
    }

    fn score(ctx: *anyopaque, _: []const u32, _: []const u32, logits: []f64, _: bool) error{Failed}!f64 {
        _ = @as(*Idle, @ptrCast(@alignCast(ctx))).scores.fetchAdd(1, .monotonic);
        @memset(logits, 0);
        return 0;
    }

    fn memory(ctx: *anyopaque, reset: bool) ?api.Memory {
        if (reset) _ = @as(*Idle, @ptrCast(@alignCast(ctx))).resets.fetchAdd(1, .monotonic);
        return .{ .active = 3, .cache = 2, .peak = 5 };
    }
};

/// A template-free vocabulary of bytes.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{
            .encode = encode,
            .decode = decode,
            .token_id = tokenId,
            .token_string = tokenString,
            .vocab_size = vocabSize,
            .eos_ids = eosIds,
            .render = render,
            .template_source = templateSource,
        } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, _: []const u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
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

    fn render(
        _: *anyopaque,
        a: Allocator,
        _: json.Value,
        _: model_text.RenderOptions,
        _: *[]const u8,
    ) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine module with a GPU, as native/metal.zig and native/cuda.zig are; its name says which device was asked for.
const WithGpu = struct {
    pub const backends: []const []const u8 = &.{"cuda"};

    pub fn deviceName(a: Allocator, device: ?u32) ?[]const u8 {
        return std.fmt.allocPrint(a, "Example \"GPU\" {d}", .{device orelse 0}) catch null;
    }
};

/// An engine module that finds no GPU.
const NoGpu = struct {
    pub const backends: []const []const u8 = &.{"cuda"};

    pub fn deviceName(_: Allocator, _: ?u32) ?[]const u8 {
        return null;
    }
};

/// No engine module, as native/none.zig declares it: no backend and no deviceName.
const NoEngine = struct {
    pub const backends: []const []const u8 = &.{};
};

/// The open /health body a server with ``identity`` sends, parsed back as a client reads it.
fn served(a: Allocator, identity: server.Identity) !std.json.ObjectMap {
    var engine: Idle = .{};
    var text: Bytes = .{};
    const srv = try server.Server.init(
        std.testing.allocator,
        std.testing.io,
        engine.engine(),
        text.text(),
        .{ .served_name = "m", .model_ids = &.{"m"}, .identity = identity },
        null,
    );
    defer srv.deinit();
    const body = try json.stringify(a, try status_routes.openHealth(srv, a, false), .{});
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{});
    return doc.object;
}

test "the open health names the release, backend and GPU the builder reads, after its earlier keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try served(a, build_identity.of(WithGpu, a, "9.8.7", 1));
    try std.testing.expectEqual(earlier.len + 3, o.count());
    for (earlier ++ [_][]const u8{ "version", "backend", "device" }, o.keys()) |want, got| {
        try std.testing.expectEqualStrings(want, got);
    }
    try std.testing.expectEqualStrings("9.8.7", o.get("version").?.string);
    try std.testing.expectEqualStrings("cuda", o.get("backend").?.string);
    try std.testing.expectEqualStrings("Example \"GPU\" 1", o.get("device").?.string);
    try std.testing.expectEqual(@as(i64, 4), o.get("max_batch_size").?.integer);
    try std.testing.expectEqual(@as(i64, 5), o.get("memory").?.object.get("peak").?.integer);
}

test "the open health always names the release, and leaves out a backend or GPU the binary does not know" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const engine_less = try served(a, build_identity.of(NoEngine, a, "9.8.7", null));
    try std.testing.expectEqual(earlier.len + 1, engine_less.count());
    for (earlier ++ [_][]const u8{"version"}, engine_less.keys()) |want, got| {
        try std.testing.expectEqualStrings(want, got);
    }
    try std.testing.expectEqualStrings("9.8.7", engine_less.get("version").?.string);
    const no_gpu = try served(a, build_identity.of(NoGpu, a, "9.8.7", null));
    try std.testing.expectEqual(earlier.len + 2, no_gpu.count());
    try std.testing.expect(no_gpu.get("device") == null);
    try std.testing.expectEqualStrings("cuda", no_gpu.get("backend").?.string);
    const test_server = try served(a, .{}); // the parity test servers set no identity
    try std.testing.expectEqual(earlier.len, test_server.count());
    for (earlier, test_server.keys()) |want, got| try std.testing.expectEqualStrings(want, got);
}

const key = "sk-health-test";
const health_open = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
const health_keyed = "GET /health HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer " ++ key ++
    "\r\nConnection: close\r\n\r\n";
const models_open = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
const models_keyed = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer " ++ key ++
    "\r\nConnection: close\r\n\r\n";

/// One request on a fresh loopback connection that asks to close, and the whole reply; error.Timeout after 5 s silent.
fn exchange(a: Allocator, port: u16, request: []const u8) ![]const u8 {
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(port) };
    var storage: Threaded.PosixAddress = undefined;
    const len = Threaded.addressToPosix(&address, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.ConnectFailed;
    var sent: usize = 0;
    while (sent < request.len) {
        const n = posix.system.write(fd, request[sent..].ptr, request.len - sent);
        if (posix.errno(n) != .SUCCESS) return error.WriteFailed;
        sent += @intCast(n);
    }
    var reply: std.ArrayList(u8) = .empty;
    var buf: [4096]u8 = undefined;
    while (true) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
        const n = posix.system.read(fd, &buf, buf.len);
        if (posix.errno(n) != .SUCCESS) return error.ReadFailed;
        if (n == 0) return reply.items;
        try reply.appendSlice(a, buf[0..@intCast(n)]);
    }
}

fn bodyOf(reply: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoBody;
    return reply[at + 4 ..];
}

/// Stops the accept loop and waits for the connection threads, so the server can be freed.
fn quiesce(srv: *server.Server, stop: *std.atomic.Value(bool), accept: std.Thread) void {
    stop.store(true, .release);
    accept.join();
    var waited: u32 = 0;
    while (srv.open_connections.load(.acquire) > 0) : (waited += 1) {
        if (waited == 500) @panic("a connection thread still runs 5 s after the last reply"); // never free it under one
        std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
    }
}

test "with keys configured, /health sends only its status, with and without a valid key, whatever the identity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(std.testing.allocator, &.{key}, "", null, false, &problem);
    defer keys.deinit();
    var engine: Idle = .{};
    var text: Bytes = .{};
    const identity = build_identity.of(WithGpu, a, "9.8.7", 1);
    try std.testing.expect(identity.version.len > 0 and identity.backend.len > 0 and identity.device.len > 0);
    const srv = try server.Server.init(
        std.testing.allocator,
        std.testing.io,
        engine.engine(),
        text.text(),
        .{ .served_name = "m", .model_ids = &.{"m"}, .identity = identity },
        &keys,
    );
    defer srv.deinit();
    const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
    defer quiesce(srv, &stop, accept);
    const port = lis.port();
    for ([_][]const u8{ health_open, health_keyed }) |request| {
        const reply = try exchange(a, port, request);
        try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 "));
        try std.testing.expectEqualStrings("{\"status\": \"ok\"}", try bodyOf(reply));
    }
    // the gate is on and the key is valid: a gated route refuses the request without it and serves it with it
    try std.testing.expect(std.mem.startsWith(u8, try exchange(a, port, models_open), "HTTP/1.1 401 "));
    try std.testing.expect(std.mem.startsWith(u8, try exchange(a, port, models_keyed), "HTTP/1.1 200 "));
}

test "unavailable health is minimal HTTP 503 with keys enabled or disabled and with or without a valid key" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(std.testing.allocator, &.{key}, "", null, false, &problem);
    defer keys.deinit();
    for ([_]bool{ false, true }) |keyed| {
        var engine: Idle = .{ .healthy = false };
        var text: Bytes = .{};
        const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{
            .served_name = "private-model",
            .model_ids = &.{"private-model"},
            .identity = .{ .version = "9.8.7", .backend = "test", .device = "private-device" },
        }, if (keyed) &keys else null);
        defer srv.deinit();
        const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        defer lis.close();
        var stop: std.atomic.Value(bool) = .init(false);
        const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
        defer quiesce(srv, &stop, accept);
        for ([_][]const u8{ health_open, health_keyed }) |request| {
            const reply = try exchange(a, lis.port(), request);
            try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503 "));
            try std.testing.expectEqualStrings("{\"status\": \"unavailable\"}", try bodyOf(reply));
        }
    }
}

const manifest_get = "GET /v1/benchmark/manifest HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";

/// The manifest a server with `environ` sends over a socket, parsed back as a client reads it.
fn manifestServed(a: Allocator, environ: ?*const std.process.Environ.Map) !std.json.ObjectMap {
    return manifestWithBackend(a, environ, "cuda");
}

fn manifestWithBackend(a: Allocator, environ: ?*const std.process.Environ.Map, backend: []const u8) !std.json.ObjectMap {
    var engine: Idle = .{};
    var text: Bytes = .{};
    var identity = build_identity.of(WithGpu, a, "9.8.7", 1);
    identity.backend = backend;
    const config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .identity = identity, .environ = environ };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), config, null);
    defer srv.deinit();
    const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
    defer quiesce(srv, &stop, accept);
    const reply = try exchange(a, lis.port(), manifest_get);
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 "));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, try bodyOf(reply), .{});
    return doc.object;
}

test "the manifest names the rank count and revision from the server's environment, and nothing it was not given" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    try env.put("TF_NEMOTRON_WORLD", "2");
    try env.put("TF_CHECKPOINT_REVISION", "0123456789abcdef0123456789abcdef01234567");
    const o = try manifestServed(a, &env);
    try std.testing.expectEqual(@as(i64, 1), o.get("schema_version").?.integer);
    try std.testing.expectEqual(@as(i64, 2), o.get("rank_count").?.integer);
    try std.testing.expectEqualStrings("9.8.7", o.get("runtime").?.object.get("tensorfold_version").?.string);
    try std.testing.expectEqualStrings("cuda", o.get("runtime").?.object.get("backend").?.string);
    const rev = o.get("model").?.object.get("revision").?.string;
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef01234567", rev);
    try env.put("TF_NEMOTRON_WORLD", "3"); // a count the board does not rank, and a revision that names no commit
    try env.put("TF_CHECKPOINT_REVISION", "main");
    const odd = try manifestServed(a, &env);
    try std.testing.expect(odd.get("rank_count") == null);
    try std.testing.expect(odd.get("model").?.object.get("revision") == null);
    const bare = try manifestServed(a, null);
    try std.testing.expect(bare.get("rank_count") == null);
    try std.testing.expect(bare.get("model").?.object.get("revision") == null);
}

test "benchmark manifest: an unknown server backend is never attested as CUDA" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const o = try manifestWithBackend(arena.allocator(), null, "");
    try std.testing.expectEqualStrings("", o.get("runtime").?.object.get("backend").?.string);
}

test "public health is read only for every reset spelling with keys enabled or disabled" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(std.testing.allocator, &.{key}, "", null, false, &problem);
    defer keys.deinit();
    for ([_]bool{ true, false }) |keyed| {
        var engine: Idle = .{};
        var text: Bytes = .{};
        const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, if (keyed) &keys else null);
        defer srv.deinit();
        const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        defer lis.close();
        var stop: std.atomic.Value(bool) = .init(false);
        const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
        defer quiesce(srv, &stop, accept);
        for ([_][]const u8{ "/health?reset_peak=1", "/?reset_peak=1", "/health?reset_peak=10", "/?x=reset_peak=1" }) |path| {
            const req = try std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", .{path});
            const reply = try exchange(a, lis.port(), req);
            try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 "));
            if (keyed) try std.testing.expectEqualStrings("{\"status\": \"ok\"}", try bodyOf(reply));
        }
        try std.testing.expectEqual(@as(u32, 0), engine.resets.load(.monotonic));
    }
}

test "malformed heads cannot hide origins and public exceptions do not open other methods" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(std.testing.allocator, &.{key}, "", null, true, &problem);
    defer keys.deinit();
    var engine: Idle = .{};
    var text: Bytes = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, &keys);
    defer srv.deinit();
    const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
    defer quiesce(srv, &stop, accept);
    for ([_][]const u8{ "Bad line\r\nOrigin: http://evil.invalid", "Bad Name: x\r\nOrigin: http://evil.invalid", "X-Other: x\r\n Origin: http://evil.invalid", "X-Other: a\x00b\r\nOrigin: http://evil.invalid" }) |headers| {
        const req = try std.fmt.allocPrint(a, "GET /health HTTP/1.1\r\nHost: localhost\r\n{s}\r\nConnection: close\r\n\r\n", .{headers});
        try std.testing.expect(std.mem.startsWith(u8, try exchange(a, lis.port(), req), "HTTP/1.1 400 "));
    }
    for ([_][]const u8{ "POST", "DELETE", "HEAD", "PUT" }) |method| for ([_][]const u8{ "/health", "/metrics", "/slide", "/slide/data.js" }) |path| {
        const req = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{ method, path });
        try std.testing.expect(std.mem.startsWith(u8, try exchange(a, lis.port(), req), "HTTP/1.1 401 "));
    };
    try std.testing.expectEqual(@as(u32, 0), engine.scores.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), engine.submits.load(.monotonic));
}

test "System One wire aliases require keys before scoring and valid keys reach the successful handler" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(std.testing.allocator, &.{key}, "", null, false, &problem);
    defer keys.deinit();
    var engine: Idle = .{};
    var text: Bytes = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, &keys);
    defer srv.deinit();
    const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
    defer quiesce(srv, &stop, accept);
    const body = "{\"state\": null, \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"i\", \"criteria\": {\"A\": \"a\", \"B\": \"b\"}}}}";
    for ([_][]const u8{ "/systemone", "/v1/systemone", "/x/systemone", "/systemone///?x=1", "//systemone", "/x//systemone", "/health/../systemone" }) |path| {
        for ([_][]const u8{ "", "x-api-key: wrong\r\n", "x-api-key: " ++ key ++ "\r\n", "Authorization: Bearer " ++ key ++ "\r\n" }, 0..) |credential, i| {
            const before = engine.scores.load(.monotonic);
            const req = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\n{s}Content-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, credential, body.len, body });
            const reply = try exchange(a, lis.port(), req);
            if (i < 2) {
                try std.testing.expectEqual(before, engine.scores.load(.monotonic));
                try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 401 "));
            } else {
                try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 "));
                try std.testing.expect(engine.scores.load(.monotonic) > before);
            }
        }
    }
}

test "browser guards close before Expect or pipelined work and recheck every pooled request" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Idle = .{};
    var text: Bytes = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
    defer quiesce(srv, &stop, accept);
    const body = "{\"state\": null, \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"i\", \"criteria\": {\"A\": \"a\", \"B\": \"b\"}}}}";
    const second = try std.fmt.allocPrint(a, "POST /systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ body.len, body });
    for ([_][]const u8{ "Host: evil.invalid\r\n", "Host: localhost\r\nOrigin: http://evil.invalid\r\n", "Host: localhost\r\nOrigin: null\r\n" }) |headers| {
        const req = try std.fmt.allocPrint(a, "POST /systemone HTTP/1.1\r\n{s}Expect: 100-continue\r\nContent-Type: text/plain\r\nContent-Length: {d}\r\n\r\n{s}{s}", .{ headers, body.len, body, second });
        const reply = try exchange(a, lis.port(), req);
        try std.testing.expectEqual(@as(u32, 0), engine.scores.load(.monotonic));
        try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 403 Forbidden\r\n"));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, reply, "HTTP/1.1 "));
        try std.testing.expect(std.mem.indexOf(u8, reply, "100 Continue") == null);
        try std.testing.expectEqual(@as(u32, 0), engine.scores.load(.monotonic));
    }
    for ([_][]const u8{ "GET", "OPTIONS", "POST" }) |method| {
        const req = try std.fmt.allocPrint(a, "{s} /health HTTP/1.1\r\nHost: localhost\r\nOrigin: http://evil.invalid\r\nConnection: close\r\n\r\n", .{method});
        try std.testing.expect(std.mem.startsWith(u8, try exchange(a, lis.port(), req), "HTTP/1.1 403 Forbidden\r\n"));
    }
    const first = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    const last = "GET /health HTTP/1.1\r\nHost: localhost\r\nOrigin: http://evil.invalid\r\nConnection: close\r\n\r\n";
    const pooled = try exchange(a, lis.port(), first ++ last);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, pooled, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, pooled, "HTTP/1.1 403 Forbidden\r\n"));
    const invalid_target = "POST http://evil.invalid/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    try std.testing.expect(std.mem.startsWith(u8, try exchange(a, lis.port(), invalid_target), "HTTP/1.1 400 "));
    try std.testing.expectEqual(@as(u32, 0), engine.resets.load(.monotonic));
}

test "HEAD refusals preserve representation lengths without sending any body bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(std.testing.allocator, &.{key}, "", null, false, &problem);
    defer keys.deinit();
    var engine: Idle = .{};
    var text: Bytes = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, &keys);
    defer srv.deinit();
    const lis = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
    defer quiesce(srv, &stop, accept);
    const representation = try bodyOf(try exchange(a, lis.port(), models_open));
    for ([_][]const u8{ "/health", "/slide", "/v1/models" }) |path| {
        for ([_][]const u8{ "", "x-api-key: wrong\r\n" }) |credential| {
            const req = try std.fmt.allocPrint(a, "HEAD {s} HTTP/1.1\r\nHost: localhost\r\n{s}Connection: close\r\n\r\n", .{ path, credential });
            const reply = try exchange(a, lis.port(), req);
            try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 401 "));
            try std.testing.expectEqualStrings("", try bodyOf(reply));
            try std.testing.expectEqual(representation.len, try responseLength(reply));
        }
    }
    const Case = struct { target: []const u8, headers: []const u8, status: []const u8 };
    for ([_]Case{
        .{ .target = "http://evil.invalid/health", .headers = "Host: localhost\r\n", .status = "400" },
        .{ .target = "/health", .headers = "Host: localhost\r\nBad Name: x\r\n", .status = "400" },
        .{ .target = "/health", .headers = "Host: evil.invalid\r\n", .status = "403" },
        .{ .target = "/health", .headers = "Host: localhost\r\nOrigin: http://evil.invalid\r\n", .status = "403" },
        .{ .target = "/health", .headers = "Host: localhost\r\nx-api-key: " ++ key ++ "\r\n", .status = "501" },
    }) |case| {
        const req = try std.fmt.allocPrint(a, "HEAD {s} HTTP/1.1\r\n{s}Connection: close\r\n\r\n", .{ case.target, case.headers });
        const reply = try exchange(a, lis.port(), req);
        try std.testing.expectEqualStrings(case.status, reply[9..12]);
        try std.testing.expectEqualStrings("", try bodyOf(reply));
        try std.testing.expect(try responseLength(reply) > 0);
    }
}

fn responseLength(reply: []const u8) !usize {
    const end = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.MissingResponseHead;
    var lines = std.mem.splitSequence(u8, reply[0..end], "\r\n");
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "Content-Length: ")) return std.fmt.parseInt(usize, line[16..], 10);
    return error.MissingContentLength;
}
