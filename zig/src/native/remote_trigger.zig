//! #1594 step 2b: asks the GB10's server to prefill a prompt under a handoff id, as MCDMA's decoders ask vLLM.
const std = @import("std");
const fabric = @import("tensorfold").fabric;

/// A completion request of one token with `kv_transfer_params.mcdma_handoff`, to `url` (…/v1/completions), over HTTP.
pub const Http = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    url: []const u8,
    model: []const u8,
    closing: ?*const std.atomic.Value(bool) = null, // the server stops: an exchange under way is cut off
    key: ?[]const u8 = null, // TF_PREFILL_API_KEY_FILE's key, sent as a Bearer token to a GB10 started with --api-key

    pub fn trigger(h: *Http) fabric.prefill_source.Trigger {
        return .{ .ptr = h, .start_fn = start };
    }

    /// The exchange, bounded by `deadline` (the GB10 prefills the tokens first), or until closing.
    fn start(ptr: *anyopaque, handoff: [16]u8, tokens: []const u32) anyerror!void {
        const h: *Http = @ptrCast(@alignCast(ptr));
        const body = try request(h.gpa, h.model, handoff, tokens);
        defer h.gpa.free(body);
        var x: Exchange = .{ .h = h, .body = body };
        const deadline = fabric.words.nowNs() + seconds(tokens.len) * std.time.ns_per_s;
        var future = h.io.concurrent(Exchange.run, .{&x}) catch return error.TriggerUnavailable;
        while (!x.done.load(.acquire)) {
            const closing = if (h.closing) |c| c.load(.acquire) else false;
            if (closing or fabric.words.nowNs() >= deadline) {
                future.cancel(h.io);
                return if (closing) error.TriggerClosed else error.TriggerTimedOut;
            }
            h.io.sleep(.fromMilliseconds(1), .awake) catch {
                future.cancel(h.io);
                return error.TriggerTimedOut;
            };
        }
        future.await(h.io);
        return x.result;
    }
};

/// TF_PREFILL_TIMEOUT_S, else 60 s and a second per 400 tokens sent (the trigger's, and a pull over HTTP's).
pub fn seconds(tokens: usize) u64 {
    return @import("sys").environment.read(u64, "TF_PREFILL_TIMEOUT_S", 60 + tokens / 400) catch 60 + tokens / 400;
}

/// One HTTP exchange on its own task, so `start` can give up on it.
const Exchange = struct {
    h: *Http,
    body: []u8,
    done: std.atomic.Value(bool) = .init(false),
    result: anyerror!void = {},

    fn run(x: *Exchange) void {
        defer x.done.store(true, .release);
        x.result = x.exchange();
    }

    fn exchange(x: *Exchange) !void {
        const h = x.h;
        var client: std.http.Client = .{ .allocator = h.gpa, .io = h.io };
        defer client.deinit();
        const uri = try std.Uri.parse(h.url);
        if (!std.ascii.eqlIgnoreCase(uri.scheme, "http")) return error.TriggerNotHttp; // a LAN peer, no TLS
        var bearer: [4096 + 7]u8 = undefined;
        const auth: std.http.Client.Request.Headers.Value = if (h.key) |k| .{ .override = try std.fmt.bufPrint(&bearer, "Bearer {s}", .{k}) } else .omit;
        var req = try client.request(.POST, uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .accept_encoding = .{ .override = "identity" }, .content_type = .{ .override = "application/json" }, .authorization = auth },
        });
        defer req.deinit();
        try req.sendBodyComplete(x.body);
        var resp = try req.receiveHead(&.{});
        if (resp.head.status == .unauthorized) {
            std.log.warn("remote prefill: the GB10 answered HTTP 401: {s}", .{if (h.key == null) "it needs an API key; name a file holding it in TF_PREFILL_API_KEY_FILE" else "it refused TF_PREFILL_API_KEY_FILE's key"});
            return error.TriggerUnauthorized;
        }
        if (resp.head.status.class() != .success) {
            std.log.warn("remote prefill: the GB10 answered HTTP {d}", .{@backingInt(resp.head.status)});
            return error.TriggerRefused;
        }
        var transfer: [16 << 10]u8 = undefined;
        const r = resp.reader(&transfer);
        const reply = try r.allocRemaining(h.gpa, .limited(1 << 20));
        h.gpa.free(reply);
    }
};

/// The request's JSON: the prefix's ids, one greedy token and the handoff MCDMA reads (no model field: the default).
pub fn request(gpa: std.mem.Allocator, model: []const u8, handoff: [16]u8, tokens: []const u32) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    if (model.len > 0) try w.print("{{\"model\": {f}, \"prompt\": [", .{std.json.fmt(model, .{})}) else try w.writeAll("{\"prompt\": [");
    for (tokens, 0..) |t, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", t });
    try w.print("], \"max_tokens\": 1, \"temperature\": 0, \"kv_transfer_params\": {{\"mcdma_handoff\": {{\"id\": \"{x}\", \"export_from\": 0}}}}}}", .{&handoff});
    return out.toOwnedSlice();
}

test "the trigger's request names the prefix's ids, one token, and the handoff as 32 hex digits" {
    const gpa = std.testing.allocator;
    var id: [16]u8 = undefined;
    for (&id, 0..) |*b, i| b.* = @intCast(i * 17);
    const got = try request(gpa, "Qwen3.8-27B", id, &.{ 5, 6, 7 });
    defer gpa.free(got);
    try std.testing.expectEqualStrings("{\"model\": \"Qwen3.8-27B\", \"prompt\": [5, 6, 7], \"max_tokens\": 1, \"temperature\": 0, \"kv_transfer_params\": {\"mcdma_handoff\": {\"id\": \"00112233445566778899aabbccddeeff\", \"export_from\": 0}}}", got);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, got, .{});
    defer parsed.deinit();
    const bare = try request(gpa, "", id, &.{9});
    defer gpa.free(bare);
    try std.testing.expect(std.mem.startsWith(u8, bare, "{\"prompt\": [9], "));
}

/// One HTTP peer on 127.0.0.1 for the tests: it reads one request's head, keeps it, and answers `status`.
const Peer = struct {
    server: std.Io.net.Server,
    status: []const u8,
    head: [4096]u8 = undefined,
    len: usize = 0,

    fn serve(p: *Peer, io: std.Io) void {
        const s = p.server.accept(io) catch return;
        defer s.close(io);
        var in: [512]u8 = undefined;
        var r = s.reader(io, &in);
        while (p.len < p.head.len and !std.mem.endsWith(u8, p.head[0..p.len], "\r\n\r\n")) : (p.len += 1)
            p.head[p.len] = r.interface.takeByte() catch return;
        if (p.status.len == 0) while (true) { // a GB10 that never answers: hold the line until the client hangs up
            _ = r.interface.takeByte() catch return;
        };
        var out: [256]u8 = undefined;
        var w = s.writer(io, &out);
        w.interface.print("HTTP/1.1 {s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{p.status}) catch return;
        w.interface.flush() catch {};
    }

    fn sent(p: *const Peer, needle: []const u8) bool {
        return std.ascii.findIgnoreCase(p.head[0..p.len], needle) != null;
    }
};

fn exchangeWith(status: []const u8, key: ?[]const u8) !struct { result: anyerror!void, peer: *Peer } {
    const io = std.testing.io;
    const peer = try std.testing.allocator.create(Peer);
    errdefer std.testing.allocator.destroy(peer);
    const loopback: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    peer.* = .{ .server = try loopback.listen(io, .{}), .status = status };
    defer peer.server.deinit(io);
    const thread = try std.Thread.spawn(.{}, Peer.serve, .{ peer, io });
    var url: [64]u8 = undefined;
    var h: Http = .{ .gpa = std.testing.allocator, .io = io, .url = try std.fmt.bufPrint(&url, "http://127.0.0.1:{d}/v1/completions", .{peer.server.socket.address.getPort()}), .model = "", .key = key };
    const result = Http.start(&h, @splat(7), &.{ 1, 2, 3 });
    thread.join();
    return .{ .result = result, .peer = peer };
}

test "the trigger sends TF_PREFILL_API_KEY_FILE's key as a Bearer token, and a 401 is its own refusal" {
    const keyed = try exchangeWith("401 Unauthorized", "sk-gb10-test");
    defer std.testing.allocator.destroy(keyed.peer);
    try std.testing.expectError(error.TriggerUnauthorized, keyed.result);
    try std.testing.expect(keyed.peer.sent("authorization: Bearer sk-gb10-test\r\n"));
    const bare = try exchangeWith("401 Unauthorized", null);
    defer std.testing.allocator.destroy(bare.peer);
    try std.testing.expectError(error.TriggerUnauthorized, bare.result);
    try std.testing.expect(!bare.peer.sent("authorization:"));
    const ok = try exchangeWith("200 OK", "sk-gb10-test");
    defer std.testing.allocator.destroy(ok.peer);
    try ok.result;
    const refused = try exchangeWith("400 Bad Request", null);
    defer std.testing.allocator.destroy(refused.peer);
    try std.testing.expectError(error.TriggerRefused, refused.result);
}

test "the trigger gives up at TF_PREFILL_TIMEOUT_S on a GB10 that never answers" {
    const env = @import("sys").env;
    if (std.c.getenv("TF_PREFILL_TIMEOUT_S") != null) return error.SkipZigTest;
    _ = env.setenv("TF_PREFILL_TIMEOUT_S", "1", 1);
    defer _ = env.unsetenv("TF_PREFILL_TIMEOUT_S");
    const began = fabric.words.nowNs();
    const silent = try exchangeWith("", null);
    defer std.testing.allocator.destroy(silent.peer);
    try std.testing.expectError(error.TriggerTimedOut, silent.result);
    try std.testing.expect(fabric.words.nowNs() - began < 10 * std.time.ns_per_s);
}
