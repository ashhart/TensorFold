//! cache_salt: a kept prompt state is found only by a request with the same salt (its field, or its API key).
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Key = @import("ids.zig").Key;
const Cx = errors.Cx;

/// The longest salt a request may send.
pub const most = 1024;

/// The request's ``cache_salt``: its keyed hash, or 0 when absent; a non-empty string of at most `most` bytes.
pub fn field(secret: *const Key, cx: *Cx, body: json.Value) errors.Refused!u64 {
    const v = body.field("cache_salt") orelse return 0;
    if (v != .string or v.string.len == 0 or v.string.len > most)
        return cx.fail(.request, "cache_salt must be a non-empty string of at most {d} bytes", .{most});
    return mix(secret, 0, "field", v.string);
}

/// --cache-salt-per-key: the salt mixed with the matched key's digest (no key matched: the salt as it is).
pub fn withKey(secret: *const Key, per_key: bool, salt: u64, digest: ?[32]u8) u64 {
    if (!per_key) return salt;
    const d = digest orelse return salt;
    return mix(secret, salt, "key", &d);
}

/// The request's salt as this server keys it: the field's hash, mixed with the matched API key's under the option.
pub fn ofRequest(srv: *const Server, salt: u64, digest: ?[32]u8) u64 {
    return withKey(&srv.tool_ids, srv.config.cache_salt_per_key, salt, digest);
}

/// HMAC-SHA-256 under the server's secret over a domain, the earlier salt and one part; never 0 (0 is no salt).
fn mix(secret: *const Key, prior: u64, comptime tag: []const u8, part: []const u8) u64 {
    var h = secret.mac; // the tool-call ids' key (ids.zig): a salt can't be tested offline against an exposed cache key
    h.update("tensorfold cache salt 1 " ++ tag);
    h.update(std.mem.asBytes(&prior));
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, part.len, .little);
    h.update(&len);
    h.update(part);
    const out = std.mem.readInt(u64, h.finalResult()[0..8], .little);
    return if (out == 0) 1 else out;
}

/// `base` (the prompt's cache key) with its first words salted, or the refusal for an engine that keys by tokens alone.
pub fn key(srv: *const Server, cx: *Cx, base: []const u32, salt: u64) errors.Refused![]const u32 {
    if (salt == 0) return base;
    // only a host that says its whole cache goes by the key takes a salt (one with a family's own store may not)
    if (!srv.info.cache_salt) return cx.fail(.request, "cache_salt is not supported by backend '{s}': its prompt cache does not key requests by salt", .{srv.backendName()});
    return salted(cx.a, base, salt);
}

/// The salted copy (core/cache_salt_key.zig): high-bit words masked by the salt, which disk never keeps.
pub fn salted(a: std.mem.Allocator, base: []const u32, salt: u64) std.mem.Allocator.Error![]const u32 {
    return @import("engine_api").cache_salt_key.salted(a, base, salt);
}

test "salts: absent is 0, a string hashes under the server's key, a malformed one is refused; keys mix in per key" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const a = cx.a;
    const k = Key.of("one server's secret");
    try std.testing.expectEqual(@as(u64, 0), try field(&k, &cx, (try json.parse(a, "{}")).ok));
    try std.testing.expectEqual(@as(u64, 0), try field(&k, &cx, (try json.parse(a, "{\"cache_salt\":null}")).ok));
    const x = try field(&k, &cx, (try json.parse(a, "{\"cache_salt\":\"tenant-a\"}")).ok);
    const y = try field(&k, &cx, (try json.parse(a, "{\"cache_salt\":\"tenant-b\"}")).ok);
    try std.testing.expect(x != 0 and y != 0 and x != y);
    try std.testing.expectEqual(x, try field(&k, &cx, (try json.parse(a, "{\"cache_salt\":\"tenant-a\"}")).ok));
    const other = Key.of("another server's secret"); // the same salt under another key: nothing to test a guess against
    try std.testing.expect(x != try field(&other, &cx, (try json.parse(a, "{\"cache_salt\":\"tenant-a\"}")).ok));
    for ([_][]const u8{ "{\"cache_salt\":\"\"}", "{\"cache_salt\":7}", "{\"cache_salt\":[\"a\"]}" }) |b|
        try std.testing.expectError(error.Refused, field(&k, &cx, (try json.parse(a, b)).ok));
    const long = try std.mem.concat(a, u8, &.{ "{\"cache_salt\":\"", &@as([most + 1]u8, @splat('s')), "\"}" });
    try std.testing.expectError(error.Refused, field(&k, &cx, (try json.parse(a, long)).ok));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "tenant") == null); // a refusal never echoes the salt
    const d1: [32]u8 = @splat(1);
    const d2: [32]u8 = @splat(2);
    try std.testing.expectEqual(x, withKey(&k, false, x, d1));
    try std.testing.expectEqual(x, withKey(&k, true, x, null));
    try std.testing.expect(withKey(&k, true, 0, d1) != 0 and withKey(&k, true, 0, d1) != withKey(&k, true, 0, d2));
    try std.testing.expect(withKey(&k, true, x, d1) != withKey(&k, true, y, d1));
}

test "salts per key: no salt a client sends reproduces another key's, nor a keyless request's, nor its own key's" {
    const k = Key.of("one server's secret");
    const d1: [32]u8 = @splat(1);
    const d2: [32]u8 = @splat(2);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    // the combination: the field's hash first, then the key's digest over it, each under its own domain
    const key2 = withKey(&k, true, 0, d2);
    const key1 = withKey(&k, true, 0, d1);
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, key2, .little);
    const tried = [_][]const u8{ &d2, &bytes, "key", "field", "tensorfold cache salt 1 key", "tenant-a" };
    for (tried) |s| {
        const body = try json.newObject(cx.a);
        try body.put(cx.a, "cache_salt", .{ .string = s });
        const own = try field(&k, &cx, .{ .object = body });
        const on1 = withKey(&k, true, own, d1);
        try std.testing.expect(on1 != key2 and on1 != key1 and on1 != own and own != key2 and own != key1);
        try std.testing.expect(on1 != withKey(&k, true, own, d2)); // the same salt on two keys: apart
    }
}
