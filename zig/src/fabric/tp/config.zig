//! A rank's TP settings from the environment: rank, master, port, backend, fail-fast knobs.
const std = @import("std");
const log = @import("log.zig");

pub const Backend = enum {
    /// NCCL for everything (the default).
    nccl,
    /// Exchanges of at most `mailbox_max_bytes` a rank through the one-shot mailbox kernel, NCCL above that.
    mailbox,
    /// The same split between two hosts: the small exchanges over RoCE (doorbell + RDMA proxy), NCCL above.
    roce,
};

/// What `roce` does when its setup or load-time probe fails on any rank (TF_TP_ROCE_FALLBACK).
pub const Fallback = enum { nccl, @"error" };

pub const Error = error{BadSetting};

/// TF_TP_PLAN_PIN: the CPU the plan threads pin to at their first plan (auto: the highest cpu_capacity).
pub const PlanPin = union(enum) { auto, cpu: u32 };

pub const Config = struct {
    world: u32 = 2,
    rank: u32 = 0,
    /// Rank 0's address as the other ranks reach it.
    master: []const u8 = "localhost",
    /// The rendezvous port (TF_TP_PORT); the link listens on `link_port`.
    port: u16 = 29500,
    /// Rank 0 listens here: TF_TP_PLAN_PORT, else `port` + 7.
    link_port: u16 = 29507,
    /// How long rank 0 waits for the others and they keep retrying the connect (TF_TP_PLAN_TIMEOUT).
    connect_timeout_ns: u64 = 300 * std.time.ns_per_s,
    /// CUDA device ordinal this rank uses.
    device: u32 = 0,
    backend: Backend = .nccl,
    mailbox_max_bytes: usize = 256 * 1024,
    /// A mailbox wait for the peer gives up after this long and poisons the transport (fail-stop).
    mailbox_timeout_ns: u64 = 120 * std.time.ns_per_s,
    failfast: bool = true,
    /// After a failure: time for the HTTP threads to answer before exit 70.
    grace_ns: u64 = std.time.ns_per_s,
    /// Rank 1's memory report period on the fate channel (also how fast it notices rank 0 hanging up).
    report_ns: u64 = std.time.ns_per_s / 2,
    nccl_lib: []const u8 = "libnccl.so.2",
    /// `name[:gid],...` RDMA devices for `roce` (null: every active Ethernet port with an IPv4 RoCE v2 GID).
    roce_hca: ?[]const u8 = null,
    /// Stripe over at most this many HCAs (auto detection only).
    roce_hcas: u8 = 2,
    roce_tc: u8 = 0,
    /// TF_TP_ROCE_FAST as the RoCE kernel's `opts` word: bit 0 on, bits 8-15 pollers, bits 16-31 stagger ns (local).
    roce_fast: u32 = 0,
    /// TF_TP_ROCE_BLOCKS: the largest RoCE grid, a power of two 1..16; launch shape only, the bytes are the same.
    roce_blocks: u32 = 16,
    /// Pin the proxy thread (it busy-spins one core between exchanges).
    roce_cpu: ?u32 = null,
    /// `shm` runs the RoCE protocol on one host without RDMA (tests); `verbs` is the real wire.
    roce_wire: enum { verbs, shm } = .verbs,
    roce_fallback: Fallback = .nccl,
    /// TF_TP_PLAN_LINK=rdma: the follower polls the plan socket this long before blocking (TF_TP_PLAN_SPIN_US).
    plan_spin_ns: u64 = 0,
    plan_pin: ?PlanPin = null,

    /// Reads every setting from `get` (the environment in `fromEnv`); unset names keep the defaults.
    pub fn parse(get: *const fn ([]const u8) ?[]const u8) Error!Config {
        var c: Config = .{};
        c.world = try int(u32, get, "TF_TP_WORLD", c.world, 1, 64);
        c.rank = try int(u32, get, "TF_TP_RANK", c.rank, 0, c.world - 1);
        if (get("TF_TP_MASTER")) |m| c.master = m;
        c.port = try int(u16, get, "TF_TP_PORT", c.port, 1, 65535 - 8);
        c.link_port = try int(u16, get, "TF_TP_PLAN_PORT", c.port + 7, 1, 65535);
        c.connect_timeout_ns = try seconds(get, "TF_TP_PLAN_TIMEOUT", 300, 0.1, 86400);
        c.device = try int(u32, get, "TF_TP_DEVICE", c.device, 0, 63);
        const backend = "TF_COMM_BACKEND";
        if (get(backend)) |b| c.backend = std.meta.stringToEnum(Backend, std.mem.trim(u8, b, " ")) orelse {
            log.warn("{s}={s}: expected nccl, mailbox or roce", .{ backend, b });
            return error.BadSetting;
        };
        c.mailbox_max_bytes = 1024 * try int(usize, get, "TF_TP_MAILBOX_MAX_KB", 256, 4, 16 * 1024);
        c.mailbox_timeout_ns = try seconds(get, "TF_TP_MAILBOX_TIMEOUT_S", 120, 0.001, 86400);
        if (get("TF_TP_FAILFAST")) |v| c.failfast = !(std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off"));
        c.grace_ns = try seconds(get, "TF_TP_FAILFAST_GRACE_S", 1.0, 0, 60);
        c.report_ns = try seconds(get, "TF_TP_PEER_MEM_S", 0.5, 0.01, 60);
        if (get("TF_NCCL_LIB")) |p| c.nccl_lib = p;
        c.roce_hca = get("TF_TP_ROCE_HCA");
        c.roce_hcas = try int(u8, get, "TF_TP_ROCE_HCAS", 2, 1, 2);
        c.roce_tc = try int(u8, get, "TF_TP_ROCE_TC", try int(u8, get, "NCCL_IB_TC", 0, 0, 255), 0, 255);
        if (try int(u8, get, "TF_TP_ROCE_FAST", 0, 0, 1) == 1) {
            const pollers = try int(u32, get, "TF_TP_ROCE_POLLERS", 1, 1, 8);
            const stagger = try int(u32, get, "TF_TP_ROCE_STAGGER_NS", 150, 0, 65535);
            c.roce_fast = 1 | (pollers << 8) | (stagger << 16);
        }
        c.roce_blocks = try int(u32, get, "TF_TP_ROCE_BLOCKS", 16, 1, 16);
        if (c.roce_blocks & (c.roce_blocks - 1) != 0) {
            log.warn("TF_TP_ROCE_BLOCKS={d}: expected a power of two", .{c.roce_blocks});
            return error.BadSetting;
        }
        if (get("TF_TP_ROCE_CPU")) |_| c.roce_cpu = try int(u32, get, "TF_TP_ROCE_CPU", 0, 0, 4095);
        if (get("TF_TP_ROCE_WIRE")) |w| c.roce_wire = std.meta.stringToEnum(@TypeOf(c.roce_wire), w) orelse {
            log.warn("TF_TP_ROCE_WIRE={s}: expected verbs or shm", .{w});
            return error.BadSetting;
        };
        const fallback = "TF_TP_ROCE_FALLBACK";
        if (get(fallback)) |f| c.roce_fallback = std.meta.stringToEnum(Fallback, f) orelse {
            log.warn("{s}={s}: expected nccl or error", .{ fallback, f });
            return error.BadSetting;
        };
        if (get("TF_TP_PLAN_LINK")) |l| {
            const mode = std.mem.trim(u8, l, " ");
            if (std.mem.eql(u8, mode, "rdma")) {
                c.plan_spin_ns = 1000 * try int(u64, get, "TF_TP_PLAN_SPIN_US", 5000, 0, 10_000_000);
            } else if (!std.mem.eql(u8, mode, "tcp") and !std.mem.eql(u8, mode, "nccl")) {
                log.warn("TF_TP_PLAN_LINK={s}: expected nccl, tcp or rdma", .{l});
                return error.BadSetting;
            }
        }
        if (get("TF_TP_PLAN_PIN")) |raw| {
            const v = std.mem.trim(u8, raw, " ");
            if (std.mem.eql(u8, v, "auto")) {
                c.plan_pin = .auto;
            } else if (!std.mem.eql(u8, v, "-1") and !std.mem.eql(u8, v, "off")) {
                c.plan_pin = .{ .cpu = try int(u32, get, "TF_TP_PLAN_PIN", 0, 0, 4095) };
            }
        }
        return c;
    }

    pub fn fromEnv() Error!Config {
        return parse(&envGet);
    }

    /// The settings both ranks must share (everything but rank, device and timeouts), as bytes to compare.
    pub fn digest(c: Config) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(std.mem.asBytes(&c.world));
        h.update(std.mem.asBytes(&c.backend));
        h.update(std.mem.asBytes(&c.mailbox_max_bytes));
        h.update(std.mem.asBytes(&c.failfast));
        h.update(std.mem.asBytes(&c.roce_wire));
        h.update(std.mem.asBytes(&c.roce_fallback));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

fn envGet(name: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.c.getenv(buf[0..name.len :0]) orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

fn int(comptime T: type, get: *const fn ([]const u8) ?[]const u8, name: []const u8, default: T, lo: T, hi: T) Error!T {
    const raw = get(name) orelse return default;
    const v = std.fmt.parseInt(T, std.mem.trim(u8, raw, " "), 10) catch {
        log.warn("{s}={s}: expected an integer", .{ name, raw });
        return error.BadSetting;
    };
    if (v < lo or v > hi) {
        log.warn("{s}={d}: expected {d}..{d}", .{ name, v, lo, hi });
        return error.BadSetting;
    }
    return v;
}

fn seconds(get: *const fn ([]const u8) ?[]const u8, name: []const u8, default: f64, lo: f64, hi: f64) Error!u64 {
    const v = if (get(name)) |raw| std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " ")) catch {
        log.warn("{s}={s}: expected seconds", .{ name, raw });
        return error.BadSetting;
    } else default;
    if (!(v >= lo and v <= hi)) {
        log.warn("{s}={d}: expected {d}..{d} s", .{ name, v, lo, hi });
        return error.BadSetting;
    }
    return @intFromFloat(v * std.time.ns_per_s);
}

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "defaults: the plan link 7 ports past the rendezvous, NCCL, fail-fast on" {
    Fake.pairs = &.{ .{ "TF_TP_RANK", "1" }, .{ "TF_TP_PORT", "29600" } };
    const c = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(u32, 1), c.rank);
    try std.testing.expectEqual(@as(u16, 29607), c.link_port);
    try std.testing.expectEqual(Backend.nccl, c.backend);
    try std.testing.expectEqual(std.time.ns_per_s, c.grace_ns);
    try std.testing.expectEqual(std.time.ns_per_s / 2, c.report_ns);
}

test "bad settings are refused and the digest ignores the rank" {
    Fake.pairs = &.{.{ "TF_TP_RANK", "2" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{.{ "TF_COMM_BACKEND", "rdma" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "TF_TP_ROCE_HCA", "rocep1s0f1,roceP2p1s0f1" }, .{ "NCCL_IB_TC", "106" } };
    const r = try Config.parse(&Fake.get);
    try std.testing.expectEqual(Backend.roce, r.backend);
    try std.testing.expectEqual(@as(u8, 106), r.roce_tc);
    try std.testing.expectEqualStrings("rocep1s0f1,roceP2p1s0f1", r.roce_hca.?);
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "mailbox" }, .{ "TF_TP_MAILBOX_MAX_KB", "64" }, .{ "TF_TP_FAILFAST_GRACE_S", "0.25" } };
    const a = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(usize, 64 * 1024), a.mailbox_max_bytes);
    try std.testing.expectEqual(std.time.ns_per_s / 4, a.grace_ns);
    var b = a;
    b.rank = 1;
    try std.testing.expectEqualSlices(u8, &a.digest(), &b.digest());
    b.mailbox_max_bytes = 1024;
    try std.testing.expect(!std.mem.eql(u8, &a.digest(), &b.digest()));
}

test "the plan link's spin and pin, the RoCE grid cap" {
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "TF_TP_MAILBOX_MAX_KB", "1024" }, .{ "TF_TP_ROCE_FALLBACK", "error" }, .{ "TF_TP_PLAN_LINK", "rdma" }, .{ "TF_TP_PLAN_PIN", "auto" } };
    const p = try Config.parse(&Fake.get);
    try std.testing.expectEqual(Backend.roce, p.backend);
    try std.testing.expectEqual(@as(usize, 1024 * 1024), p.mailbox_max_bytes);
    try std.testing.expectEqual(Fallback.@"error", p.roce_fallback);
    try std.testing.expectEqual(@as(u64, 5000 * 1000), p.plan_spin_ns);
    try std.testing.expectEqual(PlanPin.auto, p.plan_pin.?);
    Fake.pairs = &.{ .{ "TF_TP_PLAN_LINK", "tcp" }, .{ "TF_TP_PLAN_PIN", "7" } }; // tcp blocks at once; an explicit CPU
    const o = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(u64, 0), o.plan_spin_ns);
    try std.testing.expectEqual(PlanPin{ .cpu = 7 }, o.plan_pin.?);
    Fake.pairs = &.{ .{ "TF_TP_PLAN_LINK", "rdma" }, .{ "TF_TP_PLAN_SPIN_US", "200" }, .{ "TF_TP_PLAN_PIN", "-1" } };
    const s = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(u64, 200 * 1000), s.plan_spin_ns);
    try std.testing.expect(s.plan_pin == null);
    Fake.pairs = &.{.{ "TF_TP_PLAN_LINK", "udp" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{}; // the RoCE grid cap: 16 unset, powers of two only
    try std.testing.expectEqual(@as(u32, 16), (try Config.parse(&Fake.get)).roce_blocks);
    Fake.pairs = &.{.{ "TF_TP_ROCE_BLOCKS", "4" }};
    try std.testing.expectEqual(@as(u32, 4), (try Config.parse(&Fake.get)).roce_blocks);
    Fake.pairs = &.{.{ "TF_TP_ROCE_BLOCKS", "12" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{};
}
