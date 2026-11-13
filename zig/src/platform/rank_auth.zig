//! Rank link authentication: both ends of a rank connection prove the group's key before any other byte crosses.
const std = @import("std");
const builtin = @import("builtin");
pub const key = @import("rank_key.zig");
/// Sealed records for every byte after the proof, under the session key the proof leaves at both ends.
pub const seal = @import("rank_seal.zig");
/// A keyed group's tensors inside NCCL or RCCL: TLS on the library's sockets, or the rank does not start.
pub const tensors = @import("rank_tensors.zig");
const environment = @import("environment.zig");
const windows = builtin.os.tag == .windows;
const posix = if (windows) @import("win_posix.zig") else std.posix;
const c = if (windows) @import("win_c.zig") else std.c;

pub const Key = key.Key;

/// The setting that names the key file: the same contents on every rank of a group that spans hosts.
pub const env_name = "TF_RANK_SECRET_FILE";

/// The fewest secret bytes a rank key file holds: the acceptor proves first, so a short key could be guessed offline.
pub const least_bytes = 32;

/// An unproven connection's whole budget in milliseconds: a silent socket holds an acceptor no longer.
pub const budget_ms = 10_000;

/// The connector's hello must arrive within this: an honest rank sends it at once, so silent sockets go fast.
pub const first_ms = 2_000;

/// How to make a key; every rank needs the same file. 32 random bytes in base64: trimming cannot shorten them.
pub const make_key = "head -c 32 /dev/urandom | base64 > FILE && chmod 600 FILE";

/// How a rank is given its key, named in every refusal that a missing or different key explains.
const set_key = "pass serve --rank-secret FILE (or set " ++ env_name ++ ") on every rank, the same file on each; make one with: " ++ make_key;

/// The most ranks a hello may claim.
pub const max_world = 1024;

/// The link a proof is for. A proof made for one is refused on every other.
pub const Purpose = enum(u32) { tp_control = 1, tp_plan = 2, flashnext = 3, rocm = 4, liveness = 5 };

pub const Error = error{
    /// The peer did not finish the proof within its budget.
    RankAuthTimeout,
    /// The peer closed the connection during the proof.
    RankAuthClosed,
    /// The peer's first bytes are not this protocol's: a stranger, or a rank of an older build.
    RankAuthNotARank,
    /// The hello claims a link, rank or world this acceptor does not take.
    RankAuthBadClaim,
    /// The peer's proof does not verify: it holds another key.
    RankAuthWrongKey,
    /// The peer has a key and this rank has none.
    RankAuthPeerHasKey,
    /// This rank has a key and the peer has none.
    RankAuthPeerHasNoKey,
    /// Neither end has a key and the peer is on another host.
    RankAuthKeyRequired,
    NoRandom,
};

/// The key every unkeyed rank uses. It is public, so it proves nothing: it only keeps one code path.
const unkeyed: Key = blk: {
    @setEvalBranchQuota(20_000);
    var out: Key = undefined;
    std.crypto.hash.sha2.Sha256.hash("TFRK no key v1", &out, .{});
    break :blk out;
};

/// The group's key, or none: ranks with none meet only on one host.
pub const Secret = struct {
    key: Key,
    keyed: bool,

    pub const none: Secret = .{ .key = unkeyed, .keyed = false };

    pub fn fromBytes(bytes: []const u8) key.Error!Secret {
        return .{ .key = try key.fromBytes(bytes, least_bytes), .keyed = true };
    }

    pub fn fromFile(path: [:0]const u8) key.Error!Secret {
        return .{ .key = try key.fromFile(path, least_bytes), .keyed = true };
    }

    /// The key in the file TF_RANK_SECRET_FILE names, or `none` when it is unset or blank.
    pub fn fromEnv() key.Error!Secret {
        const text = (environment.path(env_name, environment.Process.get(.{}, env_name)) catch return error.SecretUnreadable) orelse return none;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        return fromFile(std.fmt.bufPrintSentinel(&buf, "{s}", .{text}, 0) catch return error.SecretUnreadable);
    }
};

/// What a connection proved: the link it is for, the connector's place, and a key only the two ends know.
pub const Proven = struct { purpose: Purpose, rank: u32, world: u32, session: Key };

const magic = "TFRKAUTH";
const answer_magic = "TFRKCHAL";
const version: u32 = 1;
pub const hello_len = 60;
pub const challenge_len = 76;
const link_label = "TFRK link v1";
const session_label = "TFRK session v1";

/// Which end a proof is from: one end's proof sent back is never the other's.
pub const Role = enum(u8) { connector = 'C', acceptor = 'A' };

fn word(v: u32) [4]u8 {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    return b;
}

fn read32(b: *const [4]u8) u32 {
    return std.mem.readInt(u32, b, .little);
}

/// The `role` end's proof for the connector at `rank` of `world`: its peer's nonce, then its own.
pub fn proof(k: *const Key, role: Role, purpose: Purpose, rank: u32, world: u32, theirs: *const key.Nonce, mine: *const key.Nonce) key.Proof {
    const parts = [_][]const u8{ link_label, &.{@backingInt(role)}, &word(@backingInt(purpose)), &word(rank), &word(world), theirs, mine };
    return key.mac(k, &parts);
}

/// The key both ends keep after the proof, for signing later frames: the connector's nonce, then the acceptor's.
fn session(k: *const Key, purpose: Purpose, connector: *const key.Nonce, acceptor: *const key.Nonce) Key {
    const parts = [_][]const u8{ session_label, &word(@backingInt(purpose)), connector, acceptor };
    return key.mac(k, &parts);
}

/// The hello a connector at `rank` of `world` sends first.
pub fn hello(purpose: u32, rank: u32, world: u32, keyed: bool, mine: *const key.Nonce) [hello_len]u8 {
    var h: [hello_len]u8 = undefined;
    @memcpy(h[0..8], magic);
    h[8..12].* = word(version);
    h[12..16].* = word(purpose);
    h[16..20].* = word(rank);
    h[20..24].* = word(world);
    h[24..28].* = word(@intFromBool(keyed));
    h[28..60].* = mine.*;
    return h;
}

/// The connector's half, within `ms`: only the hello leaves before the acceptor proves the key. Returns the session.
pub fn connect(fd: posix.socket_t, secret: *const Secret, purpose: Purpose, rank: u32, world: u32, ms: u64) Error!Key {
    return connectAs(fd, secret, purpose, rank, world, ms, sameHost(fd));
}

fn connectAs(fd: posix.socket_t, secret: *const Secret, purpose: Purpose, rank: u32, world: u32, ms: u64, local: bool) Error!Key {
    if (!secret.keyed and !local) return error.RankAuthKeyRequired;
    const until = nowMs() +| ms;
    const mine = try key.nonce();
    try sendAll(fd, &hello(@backingInt(purpose), rank, world, secret.keyed, &mine), until);
    var ch: [challenge_len]u8 = undefined;
    try recvAll(fd, &ch, until);
    if (!std.mem.eql(u8, ch[0..8], answer_magic)) return error.RankAuthNotARank;
    const their_keyed = read32(ch[8..12]) != 0;
    if (their_keyed != secret.keyed) return if (their_keyed) error.RankAuthPeerHasKey else error.RankAuthPeerHasNoKey;
    const theirs: *const key.Nonce = ch[12..44];
    if (!key.same(ch[44..76], &proof(&secret.key, .acceptor, purpose, rank, world, &mine, theirs))) return error.RankAuthWrongKey;
    try sendAll(fd, &proof(&secret.key, .connector, purpose, rank, world, theirs, &mine), until);
    return session(&secret.key, purpose, &mine, theirs);
}

/// The acceptor's half for a link in `allowed`, within `ms` and `budget_ms`. On any error the caller closes `fd` only.
pub fn accept(fd: posix.socket_t, secret: *const Secret, allowed: []const Purpose, ms: u64) Error!Proven {
    return acceptAs(fd, secret, allowed, ms, sameHost(fd));
}

fn acceptAs(fd: posix.socket_t, secret: *const Secret, allowed: []const Purpose, ms: u64, local: bool) Error!Proven {
    const until = nowMs() +| @min(ms, budget_ms);
    const stranger = !secret.keyed and !local; // no key here, and the peer is on another host
    var h: [hello_len]u8 = undefined;
    // the hello's own short budget: a silent connection holds the serial acceptor 2 s, not the whole budget
    recvAll(fd, &h, @min(until, nowMs() +| first_ms)) catch |e| return if (stranger) error.RankAuthKeyRequired else e;
    if (!std.mem.eql(u8, h[0..8], magic) or read32(h[8..12]) != version) return error.RankAuthNotARank;
    const purpose = std.enums.fromInt(Purpose, read32(h[12..16])) orelse return error.RankAuthBadClaim;
    if (std.mem.indexOfScalar(Purpose, allowed, purpose) == null) return error.RankAuthBadClaim;
    const rank = read32(h[16..20]);
    const world = read32(h[20..24]);
    if (world < 2 or world > max_world or rank == 0 or rank >= world) return error.RankAuthBadClaim;
    const their_keyed = read32(h[24..28]) != 0;
    const theirs: *const key.Nonce = h[28..60];
    const mine = try key.nonce();
    var ch: [challenge_len]u8 = undefined;
    @memcpy(ch[0..8], answer_magic);
    ch[8..12].* = word(@intFromBool(secret.keyed));
    ch[12..44].* = mine;
    ch[44..76].* = proof(&secret.key, .acceptor, purpose, rank, world, theirs, &mine);
    try sendAll(fd, &ch, until); // the challenge tells a connector with a key that this end has none
    if (stranger) return error.RankAuthKeyRequired;
    if (their_keyed != secret.keyed) return if (their_keyed) error.RankAuthPeerHasKey else error.RankAuthPeerHasNoKey;
    var got: key.Proof = undefined;
    try recvAll(fd, &got, until);
    if (!key.same(&got, &proof(&secret.key, .connector, purpose, rank, world, &mine, theirs))) return error.RankAuthWrongKey;
    return .{ .purpose = purpose, .rank = rank, .world = world, .session = session(&secret.key, purpose, theirs, &mine) };
}

/// What a failed proof, an unusable key file or unsealed tensors mean, naming the setting; null for any other error.
pub fn explain(e: anyerror) ?[]const u8 {
    return switch (e) {
        error.RankAuthKeyRequired => "ranks on more than one host need a shared key: " ++ set_key,
        error.RankAuthWrongKey => "the other rank holds a different key: " ++ set_key,
        error.RankAuthPeerHasKey => "the other rank has a key and this rank has none: " ++ set_key,
        error.RankAuthPeerHasNoKey => "this rank has a key and the other rank has none: " ++ set_key,
        error.RankAuthNotARank => "the peer does not speak rank authentication: every rank must run this build",
        error.RankAuthClosed => "the peer closed during the key proof: it runs another build, holds another key, or is not a rank",
        error.RankAuthTimeout => "the peer did not finish the key proof in time",
        error.RankAuthBadClaim => "the peer claimed a link, rank or world this rank does not take",
        error.SecretUnreadable => env_name ++ " names a file this rank cannot read: it is missing or another user's",
        error.SecretNotAFile => env_name ++ " must name a regular file",
        error.SecretNotPrivate => env_name ++ " names a file other users can read or write: chmod 600 it",
        error.SecretTooShort => std.fmt.comptimePrint("{s} names a file with fewer than {d} bytes of secret: make one with {s}", .{ env_name, least_bytes, make_key }),
        error.SecretTooLong => env_name ++ " names a file over 4 KiB",
        error.NoRandom => "the system's random source could not be read for the key proof",
        else => tensors.explain(e),
    };
}

/// Milliseconds on the monotonic clock, for a caller's own deadlines beside a proof.
pub fn nowMs() u64 {
    var ts: c.timespec = undefined;
    if (c.clock_gettime(c.CLOCK.MONOTONIC, &ts) != 0 or ts.sec < 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / std.time.ns_per_ms;
}

/// Waits for `events` on `fd` until `until`. A closed or failed socket also returns, and the next call reports it.
fn wait(fd: posix.socket_t, events: i16, until: u64) Error!void {
    while (true) {
        const t = nowMs();
        if (t >= until) return error.RankAuthTimeout;
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        const n = posix.poll(&fds, @intCast(@min(until - t, 1000))) catch return error.RankAuthClosed;
        if (n != 0) return;
    }
}

fn recvAll(fd: posix.socket_t, out: []u8, until: u64) Error!void {
    var at: usize = 0;
    while (at < out.len) {
        try wait(fd, posix.POLL.IN, until);
        const rc = if (windows) posix.system.read(fd, out[at..].ptr, out.len - at) else c.recv(fd, out[at..].ptr, out.len - at, posix.MSG.DONTWAIT);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.RankAuthClosed;
                at += @intCast(rc);
            },
            .INTR, .AGAIN => {},
            else => return error.RankAuthClosed,
        }
    }
}

/// Sends without SIGPIPE: a peer that left during the proof must not end this process.
fn sendAll(fd: posix.socket_t, bytes: []const u8, until: u64) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        try wait(fd, posix.POLL.OUT, until);
        const rc = if (windows) posix.system.write(fd, bytes[at..].ptr, bytes.len - at) else c.send(fd, bytes[at..].ptr, bytes.len - at, posix.MSG.NOSIGNAL | posix.MSG.DONTWAIT);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.RankAuthClosed;
                at += @intCast(rc);
            },
            .INTR, .AGAIN => {},
            else => return error.RankAuthClosed,
        }
    }
}

const Address = [128]u8;

/// A socket address's family from its bytes: the BSDs put a length byte first.
fn family(a: *const Address) u32 {
    return if (builtin.os.tag.isBSD()) a[1] else std.mem.readInt(u16, a[0..2], builtin.cpu.arch.endian());
}

fn loopback(ip: []const u8) bool {
    if (ip.len == 4) return ip[0] == 127;
    if (std.mem.allEqual(u8, ip[0..15], 0) and ip[15] == 1) return true;
    return std.mem.allEqual(u8, ip[0..10], 0) and ip[10] == 0xff and ip[11] == 0xff and ip[12] == 127; // IPv4 in IPv6
}

/// Whether a peer at `peer` is this host: a local socket, loopback, or the address `mine` this end is reached at.
fn sameAddress(peer: *const Address, mine: *const Address) bool {
    const f = family(peer);
    if (f == posix.AF.UNIX) return true;
    const ip: []const u8 = if (f == posix.AF.INET) peer[4..8] else if (f == posix.AF.INET6) peer[8..24] else return false;
    if (loopback(ip)) return true;
    const at = @intFromPtr(ip.ptr) - @intFromPtr(peer);
    return family(mine) == f and std.mem.eql(u8, ip, mine[at..][0..ip.len]);
}

/// `fd`'s peer as text for a log line ("1.2.3.4", "[fe80:0:0:0:0:0:0:1]", "a local socket"), or "an unknown address".
pub fn peerText(fd: posix.socket_t, buf: *[64]u8) []const u8 {
    var peer: Address align(8) = @splat(0);
    var len: posix.socklen_t = @sizeOf(Address);
    if (posix.errno(posix.system.getpeername(fd, @ptrCast(&peer), &len)) != .SUCCESS) return "an unknown address";
    return addressText(&peer, buf);
}

fn addressText(a: *const Address, buf: *[64]u8) []const u8 {
    const f = family(a);
    if (f == posix.AF.UNIX) return "a local socket";
    if (f == posix.AF.INET) return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ a[4], a[5], a[6], a[7] }) catch unreachable;
    if (f != posix.AF.INET6) return "an unknown address";
    var w: std.Io.Writer = .fixed(buf);
    w.writeByte('[') catch unreachable;
    for (0..8) |i| w.print("{s}{x}", .{ if (i == 0) "" else ":", std.mem.readInt(u16, a[8 + 2 * i ..][0..2], .big) }) catch unreachable;
    w.writeByte(']') catch unreachable;
    return w.buffered();
}

/// Whether `fd`'s peer is on this host. A peer whose address cannot be read counts as another host.
fn sameHost(fd: posix.socket_t) bool {
    var peer: Address align(8) = @splat(0);
    var mine: Address align(8) = @splat(0);
    var peer_len: posix.socklen_t = @sizeOf(Address);
    var mine_len: posix.socklen_t = @sizeOf(Address);
    if (posix.errno(posix.system.getpeername(fd, @ptrCast(&peer), &peer_len)) != .SUCCESS) return false;
    if (posix.errno(posix.system.getsockname(fd, @ptrCast(&mine), &mine_len)) != .SUCCESS) return false;
    return sameAddress(&peer, &mine);
}

const fixture = @import("rank_auth_test.zig");

test "rank auth: a group across hosts needs a key, and each end is told so" {
    if (windows) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret for the tests");
    const Go = struct {
        fn connector(fd: posix.socket_t, secret: Secret, local: bool, out: *?anyerror) void {
            _ = connectAs(fd, &secret, .rocm, 1, 2, 3000, local) catch |e| {
                out.* = e;
            };
            _ = c.shutdown(fd, posix.SHUT.WR);
        }
    };
    // no key at either end, the peer on another host: the connector sends nothing, rank 0 names the setting
    for ([_]Secret{ Secret.none, k }) |theirs| {
        const fds = try fixture.pair();
        defer fixture.close(fds);
        var err: ?anyerror = null;
        const thread = try std.Thread.spawn(.{}, Go.connector, .{ fds[0], theirs, false, &err });
        try std.testing.expectError(error.RankAuthKeyRequired, acceptAs(fds[1], &Secret.none, &.{.rocm}, 3000, false));
        thread.join();
        try std.testing.expectEqual(@as(?anyerror, if (theirs.keyed) error.RankAuthPeerHasNoKey else error.RankAuthKeyRequired), err);
    }
    // the same key at both ends meets across hosts
    const fds = try fixture.pair();
    defer fixture.close(fds);
    var err: ?anyerror = null;
    const thread = try std.Thread.spawn(.{}, Go.connector, .{ fds[0], k, false, &err });
    const proven = try acceptAs(fds[1], &k, &.{.rocm}, 3000, false);
    thread.join();
    try std.testing.expectEqual(@as(?anyerror, null), err);
    try std.testing.expectEqual(@as(u32, 1), proven.rank);
    try std.testing.expect(explain(error.RankAuthKeyRequired) != null and std.mem.indexOf(u8, explain(error.RankAuthKeyRequired).?, env_name) != null);
}

test "rank auth: this host is loopback, a local socket, or the address the connection reached" {
    if (windows) return error.SkipZigTest;
    const At = struct {
        fn v4(a: u8, b: u8, cc: u8, d: u8) Address {
            var out: Address = @splat(0);
            const in: posix.sockaddr.in = .{ .port = 0, .addr = std.mem.readInt(u32, &.{ a, b, cc, d }, builtin.cpu.arch.endian()) };
            @memcpy(out[0..@sizeOf(posix.sockaddr.in)], std.mem.asBytes(&in));
            return out;
        }
        fn v6(ip: [16]u8) Address {
            var out: Address = @splat(0);
            const in: posix.sockaddr.in6 = .{ .port = 0, .flowinfo = 0, .addr = ip, .scope_id = 0 };
            @memcpy(out[0..@sizeOf(posix.sockaddr.in6)], std.mem.asBytes(&in));
            return out;
        }
    };
    const lan = At.v4(10, 0, 0, 5);
    try std.testing.expect(sameAddress(&At.v4(127, 0, 0, 1), &lan));
    try std.testing.expect(sameAddress(&At.v4(127, 9, 9, 9), &lan));
    try std.testing.expect(sameAddress(&At.v4(10, 0, 0, 5), &lan));
    try std.testing.expect(!sameAddress(&At.v4(10, 0, 0, 6), &lan));
    try std.testing.expect(!sameAddress(&At.v4(10, 0, 0, 5), &At.v4(127, 0, 0, 1)));
    var one: [16]u8 = @splat(0);
    one[15] = 1;
    var mapped: [16]u8 = @splat(0);
    mapped[10] = 0xff;
    mapped[11] = 0xff;
    mapped[12] = 127;
    var far: [16]u8 = @splat(0);
    far[0] = 0x20;
    far[15] = 7;
    try std.testing.expect(sameAddress(&At.v6(one), &At.v6(far)));
    try std.testing.expect(sameAddress(&At.v6(mapped), &At.v6(far)));
    try std.testing.expect(sameAddress(&At.v6(far), &At.v6(far)));
    try std.testing.expect(!sameAddress(&At.v6(far), &At.v6(one)));
    try std.testing.expect(!sameAddress(&At.v6(far), &lan));
    const unknown: Address = @splat(0);
    try std.testing.expect(!sameAddress(&unknown, &lan));
    var local: Address = @splat(0);
    const un: posix.sockaddr.un = .{ .path = @splat(0) };
    @memcpy(local[0..@sizeOf(posix.sockaddr.un)], std.mem.asBytes(&un));
    try std.testing.expect(sameAddress(&local, &lan));
    // the same addresses as a refusal's log line names them
    var text: [64]u8 = undefined;
    try std.testing.expectEqualStrings("10.0.0.5", addressText(&lan, &text));
    try std.testing.expectEqualStrings("[2000:0:0:0:0:0:0:7]", addressText(&At.v6(far), &text));
    try std.testing.expectEqualStrings("a local socket", addressText(&local, &text));
    try std.testing.expectEqualStrings("an unknown address", addressText(&unknown, &text));
    const fds = try fixture.pair();
    defer fixture.close(fds);
    try std.testing.expect(sameHost(fds[0]) and sameHost(fds[1]));
}

test {
    _ = fixture;
    _ = key;
    _ = seal;
    _ = tensors;
}
