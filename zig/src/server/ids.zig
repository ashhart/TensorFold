//! Reply ids: random ``uuid4().hex`` prefixes, and tool-call ids keyed by the server over the request asking for them.
const std = @import("std");
const json = @import("json");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Hmac = std.crypto.auth.hmac.sha2.HmacSha256;

/// ``prefix`` then ``n`` random lowercase hex digits.
pub fn make(a: Allocator, prefix: []const u8, n: usize) Allocator.Error![]const u8 {
    var bytes: [32]u8 = undefined;
    log.io().random(&bytes);
    const out = try a.alloc(u8, prefix.len + n);
    @memcpy(out[0..prefix.len], prefix);
    const digits = "0123456789abcdef";
    for (0..n) |i| out[prefix.len + i] = digits[(bytes[i / 2] >> @intCast(4 * (1 - i % 2))) & 0xf];
    return out;
}

/// ``prefix`` then the first ``n`` (at most 64) lowercase hex digits of ``digest``.
fn hexed(a: Allocator, prefix: []const u8, n: usize, digest: [32]u8) Allocator.Error![]const u8 {
    const hex = std.fmt.bytesToHex(digest, .lower);
    return std.mem.concat(a, u8, &.{ prefix, hex[0..n] });
}

/// The tool-call id key, held only as its HMAC-SHA-256 state: an operator's key file (tool_id_key.zig) or random.
pub const Key = struct {
    mac: Hmac,

    pub fn of(secret: []const u8) Key {
        return .{ .mac = .init(secret) };
    }

    /// 32 bytes of fresh entropy from the system, kept nowhere else: ids that hold only while this process serves.
    pub fn random(io: std.Io) std.Io.RandomSecureError!Key {
        var secret: [32]u8 = undefined;
        defer std.crypto.secureZero(u8, &secret);
        try io.randomSecure(&secret);
        return of(&secret);
    }

    /// The key moved out and ``k`` zeroed, so whoever takes it holds the only copy.
    pub fn take(k: *Key) Key {
        defer std.crypto.secureZero(u8, std.mem.asBytes(k));
        return k.*;
    }

    fn sign(k: *const Key, parts: []const []const u8) [Hmac.mac_length]u8 {
        var h = k.mac;
        for (parts) |part| h.update(part);
        return h.finalResult();
    }
};

/// ``prefix`` then ``n`` hex digits of HMAC(``key``, ``prefix``, 0, ``from``): an id naming what ``from`` does.
pub fn derived(a: Allocator, key: *const Key, prefix: []const u8, n: usize, from: []const u8) Allocator.Error![]const u8 {
    return hexed(a, prefix, n, key.sign(&.{ prefix, &.{0}, from }));
}

/// Request fields that choose how the reply is sent, not what it says: a stream and a whole reply share their ids.
const transport = [_][]const u8{ "stream", "stream_options" };

/// SHA-256 of ``body`` as compact JSON in the order it was sent, its top-level ``transport`` fields left out.
fn requestDigest(body: json.Value) [32]u8 {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer.Hashing(Sha256) = .init(&buf);
    const o: json.Options = .{ .ascii = false, .compact = true };
    writeRequest(&w.writer, body, o) catch unreachable; // hashing drains never fail
    w.writer.flush() catch unreachable;
    return w.hasher.finalResult();
}

fn writeRequest(w: *std.Io.Writer, body: json.Value, o: json.Options) std.Io.Writer.Error!void {
    if (body != .object) return json.write(w, body, o);
    try w.writeByte('{');
    var first = true;
    for (body.object.keys(), body.object.values()) |k, v| {
        if (for (transport) |t| {
            if (std.mem.eql(u8, t, k)) break true;
        } else false) continue;
        if (!first) try w.writeByte(',');
        first = false;
        try json.writeString(w, k, o);
        try w.writeByte(':');
        try json.write(w, v, o);
    }
    try w.writeByte('}');
}

/// One request's tool-call ids: ``call_`` and 24 hex digits of HMAC(key, label, request digest, choice, call index).
pub const Calls = struct {
    key: *const Key, // the server's; without it an id cannot be tested against a guessed request
    body: json.Value, // the chat request as the route reads it; nothing writes to it while the reply runs
    digest: ?[32]u8 = null, // ``requestDigest(body)``, taken at the first call

    pub fn id(c: *Calls, a: Allocator, choice: u32, index: u64) Allocator.Error![]const u8 {
        if (c.digest == null) c.digest = requestDigest(c.body);
        var place: [12]u8 = undefined;
        std.mem.writeInt(u32, place[0..4], choice, .little);
        std.mem.writeInt(u64, place[4..12], index, .little);
        return hexed(a, "call_", 24, c.key.sign(&.{ "tensorfold tool call\x00", &c.digest.?, &place }));
    }
};

/// A fixed key for pinned test values; computed independently: HMAC-SHA-256 with these bytes as the key.
pub const test_key_text = "tensorfold test tool-call id key, not a secret";

fn callId(a: Allocator, key: *const Key, body: []const u8, choice: u32, index: u64) ![]const u8 {
    var c: Calls = .{ .key = key, .body = (try json.parse(a, body)).ok };
    return c.id(a, choice, index);
}

test "a call id is call_ and 24 hex digits of the key, the request, its choice and its index" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: Key = .of(test_key_text);
    const body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    const first = try callId(a, &key, body, 0, 0);
    try std.testing.expectEqual(@as(usize, 5 + 24), first.len);
    try std.testing.expect(std.mem.startsWith(u8, first, "call_"));
    for (first[5..]) |ch| try std.testing.expect(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'));
    // pinned: a changed derivation changes every replayed conversation's ids under an operator's key
    try std.testing.expectEqualStrings("call_b5ac3ee247d15ec3aee3c07c", first);
    try std.testing.expectEqualStrings(first, try callId(a, &key, body, 0, 0));
    try std.testing.expectEqualStrings(first, try callId(a, &key, "{ \"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"x\"}] }", 0, 0));
    try std.testing.expectEqualStrings(first, try callId(a, &key, "{\"model\":\"m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream_options\":{\"include_usage\":true}}", 0, 0));
    const other_key: Key = .of(test_key_text ++ ".");
    const others = [_][]const u8{
        try callId(a, &key, body, 0, 1),
        try callId(a, &key, body, 1, 0),
        try callId(a, &key, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"y\"}]}", 0, 0),
        try callId(a, &key, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"seed\":1}", 0, 0),
        try callId(a, &other_key, body, 0, 0),
        "call_b903ff63566de63e1928f5d5", // the unkeyed SHA-256 id this request had before the key
    };
    for (others, 0..) |x, i| {
        try std.testing.expect(!std.mem.eql(u8, first, x));
        for (others[i + 1 ..]) |y| try std.testing.expect(!std.mem.eql(u8, x, y));
    }
}

test "a derived id has its prefix and length, the same for the same key and source and apart for another" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: Key = .of(test_key_text);
    const fc = try derived(a, &key, "fc_", 32, "call_b5ac3ee247d15ec3aee3c07c");
    try std.testing.expectEqualStrings("fc_1c9d3067639fbfd450c3e8ce5cbc464f", fc);
    try std.testing.expectEqualStrings(fc, try derived(a, &key, "fc_", 32, "call_b5ac3ee247d15ec3aee3c07c"));
    try std.testing.expect(!std.mem.eql(u8, fc, try derived(a, &key, "fc_", 32, "call_b5ac3ee247d15ec3aee3c070")));
    try std.testing.expect(!std.mem.eql(u8, fc, try derived(a, &.of(test_key_text ++ "."), "fc_", 32, "call_b5ac3ee247d15ec3aee3c07c")));
}

test "a random key is drawn per call, so two servers without a key file name the same request apart" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    const one = try Key.random(std.testing.io);
    const two = try Key.random(std.testing.io);
    try std.testing.expectEqualStrings(try callId(a, &one, body, 0, 0), try callId(a, &one, body, 0, 0));
    try std.testing.expect(!std.mem.eql(u8, try callId(a, &one, body, 0, 0), try callId(a, &two, body, 0, 0)));
}
