//! A keyed group's tensors inside NCCL or RCCL: TLS on the library's sockets under a key from the group's, or no start.
const std = @import("std");
const builtin = @import("builtin");
const sys = @import("sys.zig");
const environment = @import("environment.zig");
const key = @import("rank_key.zig");
const auth = @import("rank_auth.zig");

/// `sealed` (the default) or `clear`: whether a keyed group's tensors may cross the network as plain bytes.
pub const env_name = "TF_RANK_TENSORS";
/// The export NCCL 2.32.3 added and RCCL took with its sync of that release; an older library has none.
pub const symbol = "ncclSetEncryption";

/// ncclEncryptionConfig_v23200. `version` is the loaded library's own, as its header's initializer sets it.
pub const Config = extern struct {
    size: usize = @sizeOf(Config),
    magic: u32 = 0xcafebeef,
    version: u32,
    mode: c_int = 1, // NCCL_ENCRYPTION_MODE_PSK
    psk: ?[*:0]const u8,
};
pub const SetEncryption = *const fn (*const Config) callconv(.c) c_int;

pub const Error = error{ RankTensorsNoTls, RankTensorsRefused, RankTensorsNet, RankTensorsRdma, InvalidEnvironment } || key.Error;

/// How a group's tensors cross: no key keeps it on one host, `clear` is the operator's choice, `sealed` is TLS.
pub const Mode = enum(u8) { one_host, clear, sealed };

/// The mode this process settled on at its first library open; null before it.
var settled: std.atomic.Value(u8) = .init(unset);
const unset = std.math.maxInt(u8);

/// Whether this process gave the library its key: the key is the whole process's there, so it is given once.
var handed: std.atomic.Value(bool) = .init(false);

pub fn current() ?Mode {
    const v = settled.load(.acquire);
    return if (v == unset) null else @fromBackingInt(@intCast(v));
}

/// The library's pre-shared key: 64 hex digits made from the group's key, equal on every rank, never the key itself.
pub fn psk(group: *const key.Key) [64:0]u8 {
    const made = key.mac(group, &.{"TFRK tensor psk v1"});
    var text: [65]u8 = undefined;
    text[0..64].* = std.fmt.bytesToHex(made, .lower);
    text[64] = 0;
    return text[0..64 :0].*;
}

/// The mode for a group, from its key, TF_RANK_TENSORS, NCCL_NET's text and whether the library can encrypt.
pub fn decide(keyed: bool, clear: bool, net: ?[]const u8, can_seal: bool) Error!Mode {
    if (!keyed) return .one_host;
    if (clear) return .clear;
    if (!can_seal) return error.RankTensorsNoTls;
    // TLS covers the library's own sockets only: its InfiniBand, RoCE and plugin networks carry plain bytes
    if (net) |name| if (name.len > 0 and !std.ascii.eqlIgnoreCase(name, "Socket")) return error.RankTensorsNet;
    return .sealed;
}

/// Before a rank's first NCCL or RCCL call: a keyed group's library gets its key and Socket network, or no rank starts.
pub fn apply(set: ?SetEncryption, version: c_int, library: []const u8) Error!Mode {
    const secret = try auth.Secret.fromEnv();
    const clear = try environment.value(u64, env_name, environment.Process.get(.{}, env_name), 0) == 1;
    const mode = decide(secret.keyed, clear, sys.getenv("NCCL_NET"), set != null) catch |e| return refuse(e, library);
    if (mode == .sealed) {
        if (sys.env.setenv("NCCL_NET", "Socket", 1) != 0) return refuse(error.RankTensorsNet, library);
        if (!handed.load(.acquire)) {
            var text = psk(&secret.key);
            defer std.crypto.secureZero(u8, &text);
            if (set.?(&.{ .version = @intCast(version), .psk = &text }) != 0) return refuse(error.RankTensorsRefused, library);
            handed.store(true, .release);
        }
    }
    if (mode == .clear and current() == null and !builtin.is_test)
        std.log.warn("{s}=clear: this keyed group's tensors cross {s} and RDMA as plain bytes; every rank must set it alike", .{ env_name, library });
    settled.store(@backingInt(mode), .release);
    return mode;
}

fn refuse(e: Error, library: []const u8) Error {
    if (!builtin.is_test) std.log.err("{s}: {s}", .{ library, explain(e).? });
    return e;
}

/// A rank's RDMA path named `what`: it carries plain bytes, so a keyed group takes it only with TF_RANK_TENSORS=clear.
pub fn rdma(what: []const u8) Error!void {
    if (current() != .sealed) return;
    if (!builtin.is_test) std.log.err("{s}: {s}", .{ what, explain(error.RankTensorsRdma).? });
    return error.RankTensorsRdma;
}

/// After a failed communicator start on a keyed group: what a library built without TLS does, for the log.
pub fn hint() ?[]const u8 {
    return if (current() == .sealed) "this group's key was given to the library for TLS, and a library built without TLS fails here instead of sending plain bytes: " ++ way_out else null;
}

const way_out = "use a library built with TLS (NCCL or RCCL 2.32.3 or later, TLS_BACKEND=OPENSSL3), or set " ++ env_name ++
    "=clear on every rank to send tensors as plain bytes on a network you trust";

/// What a refusal means, naming the setting; null for any other error.
pub fn explain(e: anyerror) ?[]const u8 {
    return switch (e) {
        error.RankTensorsNoTls => "this group has a key, so its tensors cross the network under TLS, and this library cannot do that: " ++ way_out,
        error.RankTensorsRefused => "the library did not take the group's key for its sockets: " ++ way_out,
        error.RankTensorsNet => "NCCL_NET names a network other than Socket, the only one TLS covers: unset it, or set " ++ env_name ++ "=clear on every rank",
        error.RankTensorsRdma => "RDMA carries tensors as plain bytes and this group has a key: set " ++ env_name ++ "=clear on every rank to use it",
        else => null,
    };
}

const Fake = struct {
    var calls: u32 = 0;
    var seen: Config = undefined;
    var text: [64]u8 = undefined;
    var result: c_int = 0;

    fn set(config: *const Config) callconv(.c) c_int {
        calls += 1;
        seen = config.*;
        text = std.mem.span(config.psk.?)[0..64].*;
        return result;
    }
};

test "security: a keyed group's tensors are sealed or refused, and only TF_RANK_TENSORS=clear sends them plain" {
    // no key: one host, nothing asked of the library
    try std.testing.expectEqual(Mode.one_host, try decide(false, false, "IB", false));
    // a key: TLS on the Socket network, whatever case NCCL_NET spells it in
    try std.testing.expectEqual(Mode.sealed, try decide(true, false, null, true));
    try std.testing.expectEqual(Mode.sealed, try decide(true, false, "socket", true));
    // a library with no ncclSetEncryption, or a network TLS does not cover, starts no rank
    try std.testing.expectError(error.RankTensorsNoTls, decide(true, false, null, false));
    try std.testing.expectError(error.RankTensorsNet, decide(true, false, "IB", true));
    // the operator's choice of plain bytes is the only way past either
    try std.testing.expectEqual(Mode.clear, try decide(true, true, "IB", false));
    for ([_]anyerror{ error.RankTensorsNoTls, error.RankTensorsRefused, error.RankTensorsNet, error.RankTensorsRdma }) |e|
        try std.testing.expect(std.mem.indexOf(u8, explain(e).?, env_name ++ "=clear") != null);
}

test "security: the library's key is 64 hex digits from the group's key, equal on every rank and not the key" {
    const group = (try auth.Secret.fromBytes("the group's rank secret for the tests")).key;
    const other = (try auth.Secret.fromBytes("another group's secret for the tests")).key;
    const text = psk(&group);
    try std.testing.expectEqual(@as(usize, 64), std.mem.span(@as([*:0]const u8, &text)).len);
    for (text[0..64]) |ch| try std.testing.expect(std.ascii.isHex(ch));
    try std.testing.expectEqualSlices(u8, &text, &psk(&group));
    try std.testing.expect(!std.mem.eql(u8, &text, &psk(&other)));
    try std.testing.expect(!std.mem.eql(u8, &text, &std.fmt.bytesToHex(group, .lower)));
}

test "the config handed to the library is ncclEncryptionConfig_v23200 with the PSK mode set" {
    try std.testing.expectEqual(@as(usize, 2 * @sizeOf(usize) + 16), @sizeOf(Config));
    try std.testing.expectEqual(@as(usize, @sizeOf(usize)), @offsetOf(Config, "magic"));
    try std.testing.expectEqual(@as(usize, @sizeOf(usize) + 8), @offsetOf(Config, "mode"));
    const group = (try auth.Secret.fromBytes("the group's rank secret for the tests")).key;
    var text = psk(&group);
    Fake.calls = 0;
    try std.testing.expectEqual(@as(c_int, 0), Fake.set(&.{ .version = 23203, .psk = &text }));
    try std.testing.expectEqual(@as(u32, 0xcafebeef), Fake.seen.magic);
    try std.testing.expectEqual(@as(usize, @sizeOf(Config)), Fake.seen.size);
    try std.testing.expectEqual(@as(c_int, 1), Fake.seen.mode);
    try std.testing.expectEqualSlices(u8, text[0..64], &Fake.text);
}

test "security: a rank with a key hands its library the key and the Socket network, or does not start" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the key file's mode bits
    const c = std.c;
    const Env = struct {
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/tf-rank-tensors-{d}", .{c.getpid()}, 0);
    _ = c.unlink(path);
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    _ = c.fchmod(fd, 0o600);
    const contents = "the group's rank secret for the tests";
    _ = c.write(fd, contents, contents.len);
    _ = c.close(fd);
    defer _ = c.unlink(path);
    defer for ([_][*:0]const u8{ auth.env_name, env_name, "NCCL_NET" }) |name| {
        _ = Env.unsetenv(name);
    };
    defer settled.store(unset, .release);
    defer handed.store(false, .release);
    for ([_][*:0]const u8{ env_name, "NCCL_NET" }) |name| _ = Env.unsetenv(name);
    _ = sys.env.setenv(auth.env_name, path, 1);
    Fake.calls = 0;
    // a library that refuses the key starts no rank
    Fake.result = 4;
    try std.testing.expectError(error.RankTensorsRefused, apply(Fake.set, 23203, "NCCL"));
    Fake.result = 0;
    // the library that can: it is given the key, the Socket network is named, and RDMA paths are refused
    try std.testing.expectEqual(Mode.sealed, try apply(Fake.set, 23203, "NCCL"));
    try std.testing.expectEqual(@as(u32, 2), Fake.calls);
    try std.testing.expectEqual(@as(u32, 23203), Fake.seen.version);
    try std.testing.expectEqualSlices(u8, psk(&(try auth.Secret.fromEnv()).key)[0..64], &Fake.text);
    try std.testing.expectEqualStrings("Socket", sys.getenv("NCCL_NET").?);
    try std.testing.expectError(error.RankTensorsRdma, rdma("TF_COMM_BACKEND=roce"));
    try std.testing.expect(hint() != null);
    // a second open in the process: the key is the whole process's in the library, and it is not given again
    try std.testing.expectEqual(Mode.sealed, try apply(Fake.set, 23203, "NCCL"));
    try std.testing.expectEqual(@as(u32, 2), Fake.calls);
    // a library before 2.32.3 and an operator's InfiniBand network: no rank starts
    try std.testing.expectError(error.RankTensorsNoTls, apply(null, 23000, "NCCL"));
    _ = sys.env.setenv("NCCL_NET", "IB", 1);
    try std.testing.expectError(error.RankTensorsNet, apply(Fake.set, 23203, "NCCL"));
    // the operator's choice: nothing is asked of the library, and RDMA is allowed again
    _ = sys.env.setenv(env_name, "clear", 1);
    Fake.calls = 0;
    try std.testing.expectEqual(Mode.clear, try apply(null, 23000, "RCCL"));
    try std.testing.expectEqual(@as(u32, 0), Fake.calls);
    try rdma("TF_COMM_BACKEND=roce");
    try std.testing.expect(hint() == null);
    _ = sys.env.setenv(env_name, "plain", 1);
    try std.testing.expectError(error.InvalidEnvironment, apply(null, 23000, "RCCL"));
    // no key: one host, and the library is left as it is
    _ = Env.unsetenv(auth.env_name);
    _ = Env.unsetenv(env_name);
    try std.testing.expectEqual(Mode.one_host, try apply(Fake.set, 23203, "NCCL"));
    try std.testing.expectEqual(@as(u32, 0), Fake.calls);
}
