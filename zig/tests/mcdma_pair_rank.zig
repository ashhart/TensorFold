//! tf-mcdma-pair's rank: connect, check the link, swap settings, run the suite round by round, and fail by name.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const fabric = @import("fabric");
const suite = @import("mcdma_pair_suite.zig");
const mcdma = fabric.mcdma;
const collective = fabric.collective;
const words = fabric.words;
const Rdma = fabric.Rdma;
const LinkKind = fabric.rdma.LinkKind;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const mib = 1 << 20;
/// The Mac's MCDMA kext registers at most 64 regions per context: a RoCE window in 4 MiB pieces, plus one per peer.
const mac_regions = 64;
const mac_piece = 4 * mib;
pub const hidden = 1024;
pub const max_rows = 300;
const slot = max_rows * hidden * 4; // f32 rows
const plan_slot = 4096;
pub const pipe_slot = 64 << 10;
pub const reduce_steps = 16;
/// The barrier that ends every test, as its step number.
const end_step = 999_999;

/// Every step of every test, as the watchdog and the logs name it; the tests are barrier..writesignalfrom.
pub const Phase = enum(u8) { connect, link, handshake, barrier, broadcast, allreduce_bf16, allreduce_f32, alltoall, gather, pipes, bulk, signals, write2signal, writesignalfrom, hold, close, round_end };
pub const first_test: Phase = .barrier;
pub const last_test: Phase = .writesignalfrom;

/// The victim closes (BYE) and exits 0, SIGKILLs itself, stops posting or spins; external: an operator drops the link.
pub const Arm = enum { none, exit, kill, stop, hang, external };

pub const Settings = struct {
    rank: u32,
    library: []const u8,
    links: []const mcdma.Link,
    label: []const u8 = "",
    note: []const u8 = "",
    /// cx5 or tb5: the kind MCDMA must report; null takes the link's own `kind`, or either.
    expect_kind: ?LinkKind = null,
    tests: []const []const u8 = &.{"all"},
    rounds: u32 = 1,
    /// Above 0: rounds until rank 0's clock passes this many seconds, instead of `rounds` (the link-drop run).
    duration_s: u32 = 0,
    seed: u64 = 1,
    /// 0: the suite's own size; larger moves the bulk copy to the window's far end. Both ends give the same window.
    window_mib: u32 = 0,
    staging_mib: u32 = 6,
    bulk_mib: u32 = 9,
    signals: u32 = 5000,
    /// The endpoint's bound on a post, lock or flush (Config.timeout_ns): Timeout.
    timeout_s: u32 = 10,
    /// Channel, Pipe and flag waits (the production 10 s): PeerTimeout.
    wait_s: u32 = 10,
    connect_timeout_s: u32 = 120,
    /// A step with no progress for this long is a HANG (exit 4); above wait_s and timeout_s, so their names come first.
    watchdog_s: u32 = 30,
    arm: Arm = .none,
    arm_rank: u32 = 1,
    arm_test: []const u8 = "allreduce_bf16",
    arm_step: u32 = 3,
    /// The victim waits this long at its arm point first (the offline stub needs posts in flight to land).
    arm_delay_ms: u32 = 0,
    /// stop: how long the stopped rank keeps its endpoint open without posting (0: wait_s + 10).
    arm_hold_s: u32 = 0,
    /// Error names this run expects to end with; empty: an arm's survivor expects PeerDown, PeerTimeout or Timeout.
    expect: []const []const u8 = &.{},
    /// Text the log must also hold (compare.py checks it): MCDMA's status or reason, so it is the intended failure.
    expect_log: []const []const u8 = &.{},
    /// Text the log must not hold (compare.py checks it): MCDMA's line for another cause of the same error.
    refuse_log: []const []const u8 = &.{},
    /// For the run scripts' pre-flight only: the peer's GID address and MAC, the kext, this host's meeting address.
    peer_gid_addr: []const u8 = "",
    peer_mac: []const u8 = "",
    expected_kext_uuid: []const u8 = "",
    meeting_addr: []const u8 = "",
    /// An expected failure must come within this many seconds of the failing step's start.
    expect_within_s: u32 = 12,
    hash_detail: bool = true,
};

pub const Layout = struct {
    ch_at: usize,
    plan_at: usize,
    pipe_at: usize,
    pipe_bytes: usize,
    flags_at: usize,
    ws_at: usize, // write2Signal lands here, writeSignalFrom 256 KiB on, its sources at src_at
    src_at: usize,
    bulk_at: usize,
    bulk_len: usize,
    window: usize,
    staging: usize,

    pub fn of(s: *const Settings) !Layout {
        var l: Layout = undefined;
        l.ch_at = 0;
        l.plan_at = l.ch_at + collective.Channel.bytes(2, slot);
        l.pipe_at = l.plan_at + collective.Channel.bytes(2, plan_slot);
        l.pipe_bytes = collective.Pipe.bytes(pipe_slot);
        l.flags_at = l.pipe_at + 2 * l.pipe_bytes;
        l.ws_at = l.flags_at + 16384;
        l.src_at = l.ws_at + (512 << 10) + 4096;
        l.bulk_at = l.ws_at + mib + 4032; // off the 4 MiB registration pieces' boundaries
        l.bulk_len = @as(usize, s.bulk_mib) * mib;
        if (l.bulk_len == 0) return error.BadSettings;
        l.window = std.mem.alignForward(usize, l.bulk_at + l.bulk_len, mcdma.alignment);
        if (s.window_mib > 0) {
            const want = @as(usize, s.window_mib) * mib;
            if (want < l.window) return error.WindowTooSmall;
            l.window = want;
            l.bulk_at = l.window - l.bulk_len - 4032; // still past the write-and-signal area, and off piece boundaries
        }
        l.staging = @as(usize, s.staging_mib) * mib;
        if (l.staging == 0) return error.BadSettings;
        return l;
    }
};

/// One rank's log: whole lines, one write each (a rank's own file, or stdout).
pub const Report = struct {
    fd: c_int,

    pub fn line(r: *const Report, comptime fmt: []const u8, args: anytype) void {
        var buf: [2048]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch blk: {
            buf[buf.len - 1] = '\n';
            break :blk buf[0..];
        };
        _ = platform.c.write(r.fd, text.ptr, text.len);
    }
};

const Watch = struct {
    progress_ns: std.atomic.Value(u64) = .init(0),
    bound_ns: std.atomic.Value(u64) = .init(0),
    phase: std.atomic.Value(u8) = .init(0),
    step: std.atomic.Value(u64) = .init(0),
    round: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(bool) = .init(false),
};

pub const Status = enum(u8) { pass = 0, check = 1, setup = 2, named = 3, hang = 4 };

/// Exit 1 for a result that differs from the host reference, 2 for the harness's settings, 3 for a named failure.
fn classify(err: anyerror) Status {
    return switch (err) {
        error.ReferenceMismatch, error.ReduceMismatch, error.SizeMismatch, error.SignalMismatch, error.WrongLinkKind, error.ReadAllowed, error.AtomicAllowed, error.Stale => .check,
        error.SettingsMismatch, error.BadSettings, error.WindowTooSmall, error.TooManyRegions => .setup,
        else => .named,
    };
}

const fail_names = [_][]const u8{ "PeerDown", "PeerTimeout", "Timeout" };

pub const Rank = struct {
    s: Settings,
    rank: u32,
    peer: u32,
    lay: Layout,
    out: Report,
    tests: []const Phase,
    pair: bool,
    link: [1]mcdma.Link,
    /// This rank's half of the run id: both logs of one run print the same RUN id, so none pairs with a stale log.
    nonce: u64,
    gpa: std.mem.Allocator = std.heap.page_allocator,
    ep: ?*mcdma.Endpoint = null,
    watch: Watch = .{},
    plan_ch: collective.Channel = undefined,
    ch: collective.Channel = undefined,
    tx: collective.Pipe = undefined,
    rx: collective.Pipe = undefined,
    scratch: [2][]const u8 = undefined,
    inv: u64 = 0,
    round: u32 = 0,
    /// The value the last signal storm left in its landing word (0 before the first).
    signals_last: u64 = 0,
    armed: bool = false,
    all: Sha256 = .init(.{}),
    partial: []u8 = &.{},
    result: []u8 = &.{},
    want: []u8 = &.{},
    big: []u8 = &.{},
    big_want: []u8 = &.{},
    code: Status = .pass,

    pub fn rdma(r: *Rank) Rdma {
        return r.ep.?.rdma();
    }

    fn mark(r: *Rank, phase: Phase, at: u64, bound_s: u64) void {
        r.watch.phase.store(@backingInt(phase), .release);
        r.watch.step.store(at, .release);
        r.watch.round.store(r.round, .release);
        r.watch.bound_ns.store(bound_s * std.time.ns_per_s, .release);
        r.watch.progress_ns.store(words.nowNs(), .release);
    }

    /// One step of a test: progress for the watchdog, and the arm when this rank is its victim.
    pub fn step(r: *Rank, phase: Phase, s: u64) !void {
        r.mark(phase, s, r.s.watchdog_s);
        if (r.armed or r.rank != r.s.arm_rank or r.round != 0) return;
        switch (r.s.arm) {
            .none, .external => return,
            .exit, .kill, .stop, .hang => {},
        }
        if (phase != (std.meta.stringToEnum(Phase, r.s.arm_test) orelse return) or s != r.s.arm_step) return;
        r.armed = true;
        return r.fire(phase, s);
    }

    fn fire(r: *Rank, phase: Phase, s: u64) !void {
        const what: []const u8 = switch (r.s.arm) {
            .exit => "closing the endpoint mid-run (MCDMA sends BYE), then exiting 0",
            .kill => "SIGKILL to this process mid-run (no BYE)",
            .stop => "no more posts; the endpoint stays open, then closes and exits 0",
            .hang => "spins with no bound and posts nothing: this rank's watchdog must name it (HANG, exit 4)",
            else => unreachable,
        };
        r.out.line("ARM arm={s} rank={d} test={s} step={d} round={d} delay_ms={d} t_unix={d:.3} action=\"{s}\"", .{ @tagName(r.s.arm), r.rank, @tagName(phase), s, r.round, r.s.arm_delay_ms, unixNow(), what });
        const hold_s: u64 = if (r.s.arm_hold_s > 0) r.s.arm_hold_s else @as(u64, r.s.wait_s) + 10;
        r.mark(.hold, s, @as(u64, r.s.arm_delay_ms) / 1000 + hold_s + r.s.watchdog_s);
        words.pause(@as(u64, r.s.arm_delay_ms) * std.time.ns_per_ms);
        const t0 = words.nowNs();
        switch (r.s.arm) {
            .kill => {
                _ = std.c.kill(std.c.getpid(), std.c.SIG.KILL);
                unreachable;
            },
            .stop => words.pause(hold_s * std.time.ns_per_s),
            .hang => {
                r.mark(phase, s, r.s.watchdog_s);
                while (true) std.atomic.spinLoopHint();
            },
            else => {},
        }
        r.mark(.close, s, r.s.watchdog_s);
        r.close();
        r.out.line("RESULT status=ARM-DONE code=0 arm={s} test={s} step={d} after_arm_s={d:.2} endpoint=closed", .{ @tagName(r.s.arm), @tagName(phase), s, secs(words.nowNs() - t0) });
        return error.ArmDone;
    }

    fn close(r: *Rank) void {
        const ep = @atomicRmw(?*mcdma.Endpoint, &r.ep, .Xchg, null, .acq_rel);
        if (ep) |e| e.deinit();
    }

    /// MCDMA's reason the peer is down, waiting up to `wait_ns` for it to arrive; "link up" when MCDMA has none.
    fn reason(r: *Rank, buf: []u8, wait_ns: u64) []const u8 {
        const ep = @atomicLoad(?*mcdma.Endpoint, &r.ep, .acquire) orelse return "no endpoint (at startup the warning(mcdma) line gives MCDMA's status)";
        const t0 = words.nowNs();
        while (true) {
            if (ep.peerStatus(r.peer, buf)) |why| {
                for (buf[0..why.len]) |*c| if (c.* == '"' or c.* == '\n') {
                    c.* = '\'';
                };
                return why;
            }
            if (words.nowNs() - t0 >= wait_ns) return "link up (MCDMA reports no peer-down)";
            words.pause(10 * std.time.ns_per_ms);
        }
    }

    fn expects(r: *const Rank, name: []const u8) bool {
        if (r.s.expect.len > 0) {
            for (r.s.expect) |e| if (std.mem.eql(u8, e, name)) return true;
            return false;
        }
        const survivor = switch (r.s.arm) {
            .none => false,
            .external => true,
            .exit, .kill, .stop, .hang => r.rank != r.s.arm_rank,
        };
        if (!survivor) return false;
        for (fail_names) |e| if (std.mem.eql(u8, e, name)) return true;
        return false;
    }

    fn expectsFailure(r: *const Rank) bool {
        return r.s.expect.len > 0 or r.s.arm == .external or (r.s.arm != .none and r.rank != r.s.arm_rank);
    }

    fn fail(r: *Rank, err: anyerror) Status {
        const phase: Phase = @fromBackingInt(@intCast(r.watch.phase.load(.acquire)));
        const after = secs(words.nowNs() - r.watch.progress_ns.load(.acquire));
        var buf: [320]u8 = undefined;
        const why = r.reason(&buf, if (phase == .connect) 0 else 500 * std.time.ns_per_ms);
        const status = classify(err);
        r.out.line("FAIL test={s} round={d} step={d} error={s} after={d:.2} t_unix={d:.3} mcdma=\"{s}\"", .{ @tagName(phase), r.round, r.watch.step.load(.acquire), @errorName(err), after, unixNow(), why });
        const expected = r.expects(@errorName(err)) and after <= @as(f64, @floatFromInt(r.s.expect_within_s)) + if (phase == .connect) @as(f64, @floatFromInt(r.s.connect_timeout_s)) else 0;
        const label: []const u8 = switch (status) {
            .check => "FAIL-CHECK",
            .setup => "FAIL-SETUP",
            else => "FAIL-BY-NAME",
        };
        r.out.line("RESULT status={s} code={d} error={s} test={s} round={d} after={d:.2} expected={s} mcdma=\"{s}\"", .{ label, @backingInt(status), @errorName(err), @tagName(phase), r.round, after, if (expected) "yes" else "no", why });
        return status;
    }

    /// The whole run on this rank: connect, check the link, agree on settings, then the tests round by round.
    pub fn run(r: *Rank) Status {
        const wd = std.Thread.spawn(.{}, watchdog, .{r}) catch {
            r.out.line("RESULT status=FAIL-SETUP code=2 error=ThreadFailed", .{});
            return .setup;
        };
        defer {
            r.watch.done.store(true, .release);
            wd.join();
        }
        const code = r.body() catch |err| switch (err) {
            error.ArmDone => return .pass,
            else => blk: {
                const st = r.fail(err);
                break :blk st;
            },
        };
        r.mark(.close, 0, r.s.watchdog_s);
        r.close();
        return code;
    }

    fn body(r: *Rank) !Status {
        const l = r.lay;
        r.mark(.connect, 0, @as(u64, r.s.connect_timeout_s) + r.s.watchdog_s);
        try r.macRegions();
        const lib = try std.heap.page_allocator.dupeSentinel(u8, r.s.library, 0);
        defer std.heap.page_allocator.free(lib);
        const config: mcdma.Config = .{ .rank = r.rank, .ranks = 2, .window_bytes = l.window, .staging_bytes = l.staging, .links = &r.link, .timeout_ns = @as(u64, r.s.timeout_s) * std.time.ns_per_s, .connect_timeout_ns = @as(u64, r.s.connect_timeout_s) * std.time.ns_per_s };
        const t0 = words.nowNs();
        const ep = try mcdma.Endpoint.create(r.gpa, lib, config);
        @atomicStore(?*mcdma.Endpoint, &r.ep, ep, .release);
        const connect_s = secs(words.nowNs() - t0);
        r.mark(.link, 0, r.s.watchdog_s);
        const rd = r.rdma();
        const kind = rd.link(r.peer);
        const ln = r.link[0];
        r.out.line("LINK kind={s} device={s} via={s} gid={d} mtu={d} window={d} staging={d} connect_s={d:.3}", .{ @tagName(kind), ln.device, ln.via, ln.gid, ln.mtu, l.window, l.staging, connect_s });
        const want_kind = r.s.expect_kind orelse ln.kind;
        if (want_kind) |k| {
            if (kind != k) return error.WrongLinkKind;
        } else if (kind != .cx5 and kind != .tb5) return error.WrongLinkKind;
        if (rd.link(r.rank) != .local) return error.WrongLinkKind;
        var word: [8]u8 = undefined;
        if (rd.read(r.peer, 0, &word)) |_| return error.ReadAllowed else |e| if (e != error.AccessDenied) return e;
        if (rd.fetchAdd(r.peer, l.flags_at, 1)) |_| return error.AtomicAllowed else |e| if (e != error.AccessDenied) return e;
        r.plan_ch = collective.Channel.init(rd, l.plan_at, plan_slot);
        r.ch = collective.Channel.init(rd, l.ch_at, slot);
        r.tx = collective.Pipe.init(rd, l.pipe_at + r.peer * l.pipe_bytes, pipe_slot, r.peer);
        r.rx = collective.Pipe.init(rd, l.pipe_at + r.rank * l.pipe_bytes, pipe_slot, r.peer);
        const wait_ns = @as(u64, r.s.wait_s) * std.time.ns_per_s;
        for ([_]*collective.Channel{ &r.plan_ch, &r.ch }) |c| c.timeout_ns = wait_ns;
        r.tx.timeout_ns = wait_ns;
        r.rx.timeout_ns = wait_ns;
        try r.handshake();
        r.partial = try r.gpa.alloc(u8, slot);
        r.result = try r.gpa.alloc(u8, slot);
        r.want = try r.gpa.alloc(u8, slot);
        r.big = try r.gpa.alloc(u8, l.bulk_len);
        r.big_want = try r.gpa.alloc(u8, l.bulk_len);
        defer for ([_][]u8{ r.partial, r.result, r.want, r.big, r.big_want }) |b| r.gpa.free(b);
        var passed: usize = 0;
        const start = words.nowNs();
        var more = true;
        while (more) : (r.round += 1) {
            for (r.tests) |t| {
                r.inv += 1;
                var tx: suite.Tx = .{ .r = r, .phase = t };
                const ts = words.nowNs();
                try suite.run(r, t, &tx);
                try r.step(t, end_step); // every test ends at a barrier: the next may reuse its landing places
                try collective.barrier(&r.plan_ch, &r.scratch);
                const got = tx.got.finalResult();
                const want = tx.want.finalResult();
                if (!std.mem.eql(u8, &got, &want)) return error.ReferenceMismatch;
                r.all.update(&got);
                passed += 1;
                r.out.line("PASS test={s} round={d} s={d:.3} bufs={d} hash={s} ref={s}", .{ @tagName(t), r.round, secs(words.nowNs() - ts), tx.n, &std.fmt.bytesToHex(got, .lower), &std.fmt.bytesToHex(want, .lower) });
                if (t == .bulk) r.out.line("RATE test=bulk round={d} mib_each_way={d} s={d:.3}", .{ r.round, r.s.bulk_mib, secs(words.nowNs() - ts) });
            }
            if (r.s.duration_s > 0) {
                // rank 0's clock decides for both, so the ranks stop after the same round
                r.mark(.round_end, r.round, r.s.watchdog_s);
                const go = [1]u8{@intFromBool(secs(words.nowNs() - start) < @as(f64, @floatFromInt(r.s.duration_s)))};
                const got = try collective.broadcast(&r.plan_ch, 0, if (r.rank == 0) &go else "", &r.scratch);
                more = got.len == 1 and got[0] == 1;
            } else more = r.round + 1 < r.s.rounds;
            r.out.line("ROUND round={d} done t={d:.2}", .{ r.round, secs(words.nowNs() - start) });
        }
        try rd.flush();
        const all = r.all.finalResult();
        r.out.line("HASH all={s} tests={d} kind={s}", .{ &std.fmt.bytesToHex(all, .lower), passed, @tagName(kind) });
        if (r.expectsFailure()) {
            r.out.line("RESULT status=PASS code=0 expected=no note=\"the run expected a named failure and none came\"", .{});
        } else r.out.line("RESULT status=PASS code=0 expected=yes", .{});
        return .pass;
    }

    /// A Mac ConnectX window past the kext's 64 regions fails registration as DEVICE: refuse it by name before opening.
    fn macRegions(r: *Rank) !void {
        if (builtin.os.tag != .macos) return;
        const ln = r.link[0];
        const roce = if (ln.kind) |k| k == .cx5 else std.mem.startsWith(u8, ln.device, "rdma_mcrdma");
        if (!roce) return;
        const bytes = r.lay.window + 2 * r.lay.staging;
        const pieces = (bytes + mac_piece - 1) / mac_piece;
        if (pieces + 1 <= mac_regions) return;
        r.out.line("REGIONS bytes={d} pieces={d} slots=1 limit={d}: the Mac's MCDMA kext registers at most {d} memory regions per process (window and staging in 4 MiB pieces, one more per peer), so at most {d} pieces", .{ bytes, pieces, mac_regions, mac_regions, mac_regions - 1 });
        return error.TooManyRegions;
    }

    /// Both ends swap their settings text and half of the run id: a difference fails here by name, not as a mismatch.
    fn handshake(r: *Rank) !void {
        r.mark(.handshake, 0, @as(u64, r.s.watchdog_s) + 30); // the handshake waits up to 30 s for a slower peer
        var buf: [plan_slot]u8 = undefined;
        const text = try fingerprint(r, &buf);
        r.out.line("SETTINGS {s}", .{text});
        var msg_buf: [plan_slot]u8 = undefined;
        const msg = std.fmt.bufPrint(&msg_buf, "{s} run={x:0>16}", .{ text, r.nonce }) catch return error.BadSettings;
        const saved = r.plan_ch.timeout_ns;
        r.plan_ch.timeout_ns = @max(saved, 30 * std.time.ns_per_s);
        defer r.plan_ch.timeout_ns = saved;
        const chunks = [2][]const u8{ msg, msg };
        try r.plan_ch.exchange(&chunks, &r.scratch);
        const got = r.scratch[r.peer];
        const cut = std.mem.lastIndexOf(u8, got, " run=") orelse got.len;
        const peer_nonce: ?u64 = if (got.len == cut + 5 + 16) std.fmt.parseInt(u64, got[cut + 5 ..], 16) catch null else null;
        if (peer_nonce) |n| {
            const ids = if (r.rank == 0) [2]u64{ r.nonce, n } else [2]u64{ n, r.nonce };
            r.out.line("RUN id={x:0>16}{x:0>16}", .{ ids[0], ids[1] });
        }
        if (peer_nonce == null or !std.mem.eql(u8, got[0..cut], text)) {
            r.out.line("PEER-SETTINGS {s}", .{got[0..cut]});
            return error.SettingsMismatch;
        }
    }
};

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
}

/// Wall-clock seconds since 1970, for lining up one host's ARM with the other's FAIL (as good as the hosts' clocks).
fn unixNow() f64 {
    var ts: platform.c.timespec = undefined;
    if (platform.c.clock_gettime(platform.c.CLOCK.REALTIME, &ts) != 0) return 0;
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / std.time.ns_per_s;
}

/// What both ends must agree on as text (no host-local fields): a rank expecting another failure refuses at once.
fn fingerprint(r: *const Rank, buf: []u8) ![]const u8 {
    var tests: [512]u8 = undefined;
    var n: usize = 0;
    for (r.tests, 0..) |t, i| {
        const sep: []const u8 = if (i == 0) "" else ",";
        const part = try std.fmt.bufPrint(tests[n..], "{s}{s}", .{ sep, @tagName(t) });
        n += part.len;
    }
    const s = r.s;
    return std.fmt.bufPrint(buf, "v=3 tests={s} rounds={d} duration_s={d} seed={d} window={d} staging={d} bulk_mib={d} signals={d} mtu={d} wait_s={d} timeout_s={d} arm={s} arm_rank={d} arm_test={s} arm_step={d} expect={s}", .{ tests[0..n], s.rounds, s.duration_s, s.seed, r.lay.window, r.lay.staging, s.bulk_mib, s.signals, r.link[0].mtu, s.wait_s, s.timeout_s, @tagName(s.arm), s.arm_rank, s.arm_test, s.arm_step, expectText(r) });
}

/// A step with no progress past its bound is a named failure; the process exits 4 at once (no endpoint close).
fn watchdog(r: *Rank) void {
    while (!r.watch.done.load(.acquire)) {
        words.pause(50 * std.time.ns_per_ms);
        const progress = r.watch.progress_ns.load(.acquire);
        const bound = r.watch.bound_ns.load(.acquire);
        if (progress == 0 or bound == 0) continue;
        const since = words.nowNs() -| progress;
        if (since <= bound) continue;
        const phase: Phase = @fromBackingInt(@intCast(r.watch.phase.load(.acquire)));
        var buf: [320]u8 = undefined;
        const why = r.reason(&buf, 0);
        const at = r.watch.step.load(.acquire);
        const round = r.watch.round.load(.acquire);
        r.out.line("FAIL test={s} round={d} step={d} error=WatchdogHang after={d:.2} mcdma=\"{s}\"", .{ @tagName(phase), round, at, secs(since), why });
        r.out.line("RESULT status=HANG code=4 error=WatchdogHang test={s} round={d} after={d:.2} bound_s={d} expected={s} mcdma=\"{s}\"", .{ @tagName(phase), round, secs(since), bound / std.time.ns_per_s, if (r.s.arm == .hang and r.rank == r.s.arm_rank) "yes" else "no", why });
        if (r.pair) writeExit(r.rank, .hang);
        std.process.exit(4);
    }
}

pub var pair_dir: [:0]const u8 = "";

pub fn writeExit(rank: u32, code: Status) void {
    var path: [1024]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&path, "{s}/rank{d}.exit", .{ pair_dir, rank }, 0) catch return;
    const fd = platform.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = platform.c.close(fd);
    var b: [8]u8 = undefined;
    const text = std.fmt.bufPrint(&b, "{d}\n", .{@backingInt(code)}) catch return;
    _ = platform.c.write(fd, text.ptr, text.len);
}

/// The settings' expected error names, comma-joined, or "-".
pub fn expectText(r: *const Rank) []const u8 {
    if (r.s.expect.len == 0) return "-";
    const State = struct {
        var buf: [2][256]u8 = undefined;
    };
    const b = &State.buf[r.rank];
    var n: usize = 0;
    for (r.s.expect, 0..) |e, i| {
        const sep: []const u8 = if (i == 0) "" else ",";
        const part = std.fmt.bufPrint(b[n..], "{s}{s}", .{ sep, e }) catch break;
        n += part.len;
    }
    return b[0..n];
}
