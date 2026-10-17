//! The round plan from rank 0 to the other ranks over its own TCP connection: a 4-byte length, then int64 words.
const std = @import("std");
const log = @import("log.zig");
const sock = @import("sock.zig");
const builtin = @import("builtin");
const config = @import("config.zig");
const linux = std.os.linux;

pub const max_ranks = 64;

/// The CPU a plan thread pinned itself to (TF_TP_PLAN_PIN), -1 until one did; an engine's CPU isolation keeps off it.
pub var pinned_cpu: std.atomic.Value(i32) = .init(-1);
const more: u32 = 0x8000; // MSG_MORE: the length and the ints leave as one segment

pub const Error = error{ PeerClosed, TooLarge, NotRankZero, NotFollower };

/// A batch frame's first word (a plan's op is >= 0): [batch, n, len 1, plan 1, ..., len n, plan n].
pub const batch: i64 = std.math.minInt(i64);
/// Words rank 0 holds at most (more flushes the held plans first), and the most a batch frame carries.
pub const hold_words: usize = 1024;
pub const batch_words: usize = 2 * hold_words + 2;

/// Largest plan either side accepts: 64M words (512 MiB), so a garbled length cannot become a 4 GiB allocation.
pub const max_frame_words: usize = 64 << 20;

comptime {
    std.debug.assert(builtin.cpu.arch.endian() == .little); // the ints go out as they are in memory
}

pub const PlanLink = struct {
    rank: u32,
    world: u32,
    /// Rank 0: rank i's socket at index i; the others: rank 0's at index 0 (the bootstrap's `releasePlan`).
    fds: [max_ranks]sock.Fd,
    sent: u64 = 0,
    received: u64 = 0,
    /// TF_TP_PLAN_LINK=rdma (config.plan_spin_ns): the follower polls the socket this long before it blocks.
    spin_ns: u64 = 0,
    /// TF_TP_PLAN_PIN: the sending / receiving thread pins itself at its first plan; `avoid`: the RoCE proxy's CPU.
    pin: ?config.PlanPin = null,
    avoid: ?u32 = null,
    pinned: bool = false,
    /// rank 0: plans held for the next send (`hold`), each as [len, plan...]; `held_n` of them
    held: [hold_words]i64 = undefined,
    held_len: usize = 0,
    held_n: usize = 0,
    /// the follower: a batch frame's plans not handed out yet, as [len, plan...] from `inbox_at`
    inbox: [batch_words]i64 = undefined,
    inbox_len: usize = 0,
    inbox_at: usize = 0,
    batches: u64 = 0,

    pub fn init(rank: u32, world: u32, fds: [max_ranks]sock.Fd) PlanLink {
        return .{ .rank = rank, .world = world, .fds = fds };
    }

    /// Rank 0: one plan to every other rank (host only); held plans go first, in the same frame when they fit.
    pub fn send(p: *PlanLink, ints: []const i64) Error!void {
        if (p.rank != 0) return error.NotRankZero;
        if (ints.len > max_frame_words) return error.TooLarge;
        if (p.held_n > 0) {
            if (2 + p.held_len + 1 + ints.len <= batch_words) {
                var frame: [batch_words]i64 = undefined;
                frame[0] = batch;
                frame[1] = @intCast(p.held_n + 1);
                @memcpy(frame[2..][0..p.held_len], p.held[0..p.held_len]);
                frame[2 + p.held_len] = @intCast(ints.len);
                @memcpy(frame[3 + p.held_len ..][0..ints.len], ints);
                const n = 3 + p.held_len + ints.len;
                p.held_len = 0;
                p.held_n = 0;
                p.batches += 1;
                return p.sendFrame(frame[0..n]);
            }
            try p.flush();
        }
        return p.sendFrame(ints);
    }

    /// Rank 0: a plan the followers run before the next sent one, carried in that plan's frame (`send` or `flush`).
    pub fn hold(p: *PlanLink, ints: []const i64) Error!void {
        if (p.rank != 0) return error.NotRankZero;
        if (1 + ints.len > hold_words) {
            try p.flush();
            return p.sendFrame(ints);
        }
        if (p.held_len + 1 + ints.len > hold_words) try p.flush();
        p.held[p.held_len] = @intCast(ints.len);
        @memcpy(p.held[p.held_len + 1 ..][0..ints.len], ints);
        p.held_len += 1 + ints.len;
        p.held_n += 1;
    }

    /// Rank 0: the held plans now, as one batch frame (nothing held: nothing sent).
    pub fn flush(p: *PlanLink) Error!void {
        if (p.held_n == 0) return;
        var frame: [batch_words]i64 = undefined;
        frame[0] = batch;
        frame[1] = @intCast(p.held_n);
        @memcpy(frame[2..][0..p.held_len], p.held[0..p.held_len]);
        const n = 2 + p.held_len;
        p.held_len = 0;
        p.held_n = 0;
        p.batches += 1;
        return p.sendFrame(frame[0..n]);
    }

    fn sendFrame(p: *PlanLink, ints: []const i64) Error!void {
        p.pinOnce();
        const bytes = std.mem.sliceAsBytes(ints);
        const len: u32 = @intCast(bytes.len);
        for (p.fds[1..p.world]) |fd| {
            if (std.c.send(fd, &len, 4, std.posix.MSG.NOSIGNAL | more) != 4) return error.PeerClosed;
            sock.sendAll(fd, bytes) catch return error.PeerClosed;
        }
        p.sent += 1;
    }

    /// The other ranks: the next plan into `out`; `error.PeerClosed` once rank 0 is gone, `error.TooLarge` skips one.
    pub fn recv(p: *PlanLink, out: []i64) Error![]i64 {
        if (p.inboxNext()) |m| {
            if (m.len > out.len) return error.TooLarge;
            @memcpy(out[0..m.len], m);
            return out[0..m.len];
        }
        const n = try p.header();
        if (n > out.len) {
            try p.drop(n);
            return error.TooLarge;
        }
        const got = try p.body(out[0..n]);
        if (got.len > 0 and got[0] == batch) {
            try p.unpack(got);
            return p.recv(out);
        }
        return got;
    }

    /// The next plan into `buf`, grown to the frame; the buffer is kept, so steady plans allocate nothing.
    pub fn recvGrow(p: *PlanLink, gpa: std.mem.Allocator, buf: *std.ArrayList(i64)) (Error || std.mem.Allocator.Error)![]i64 {
        if (p.inboxNext()) |m| {
            try buf.resize(gpa, m.len);
            @memcpy(buf.items, m);
            return buf.items;
        }
        const n = try p.header();
        if (n > max_frame_words) {
            try p.drop(n);
            return error.TooLarge;
        }
        try buf.resize(gpa, n);
        const got = try p.body(buf.items);
        if (got.len > 0 and got[0] == batch) {
            try p.unpack(got);
            return p.recvGrow(gpa, buf);
        }
        return got;
    }

    /// The follower: a batch frame's plans into the inbox (checked whole first: a garbled batch ends the link).
    fn unpack(p: *PlanLink, frame: []const i64) Error!void {
        if (frame.len < 2 or frame.len > batch_words or frame[1] < 1) return error.PeerClosed;
        var at: usize = 2;
        var k: i64 = 0;
        while (k < frame[1]) : (k += 1) {
            if (at >= frame.len or frame[at] < 0) return error.PeerClosed;
            at += 1 + @as(usize, @intCast(frame[at]));
            if (at > frame.len) return error.PeerClosed;
        }
        if (at != frame.len) return error.PeerClosed;
        @memcpy(p.inbox[0 .. frame.len - 2], frame[2..]);
        p.inbox_len = frame.len - 2;
        p.inbox_at = 0;
        p.batches += 1;
    }

    /// The follower: the next plan of the last batch frame, if any (a slice of the inbox, valid until the next recv).
    fn inboxNext(p: *PlanLink) ?[]const i64 {
        if (p.inbox_at >= p.inbox_len) return null;
        const n: usize = @intCast(p.inbox[p.inbox_at]);
        const m = p.inbox[p.inbox_at + 1 ..][0..n];
        p.inbox_at += 1 + n;
        return m;
    }

    /// The next frame's length in words.
    fn header(p: *PlanLink) Error!usize {
        if (p.rank == 0) return error.NotFollower;
        p.pinOnce();
        if (p.spin_ns > 0) p.spin();
        var len: u32 = undefined;
        sock.recvExact(p.fds[0], std.mem.asBytes(&len), null) catch return error.PeerClosed;
        if (len % 8 != 0) return error.PeerClosed; // not a frame boundary: the stream is lost
        return len / 8;
    }

    fn body(p: *PlanLink, out: []i64) Error![]i64 {
        sock.recvExact(p.fds[0], std.mem.sliceAsBytes(out), null) catch return error.PeerClosed;
        p.received += 1;
        return out;
    }

    /// Reads and drops `n` words.
    fn drop(p: *PlanLink, n: usize) Error!void {
        var left: usize = n * 8;
        var scratch: [4096]u8 = undefined;
        while (left > 0) : (left -= @min(left, scratch.len)) sock.recvExact(p.fds[0], scratch[0..@min(left, scratch.len)], null) catch return error.PeerClosed;
    }

    /// Polls rank 0's socket until it is readable (or hung up) or `spin_ns` passed; the read that follows blocks.
    fn spin(p: *PlanLink) void {
        if (comptime builtin.os.tag != .linux) return;
        var fd = [1]linux.pollfd{.{ .fd = p.fds[0], .events = linux.POLL.IN, .revents = 0 }};
        const end = sock.nowNs() + p.spin_ns;
        while (true) {
            const rc = linux.poll(&fd, 1, 0);
            if (linux.errno(rc) == .SUCCESS and rc > 0) return;
            if (sock.nowNs() >= end) return;
            std.atomic.spinLoopHint();
        }
    }

    /// Pins the calling thread once: an explicit CPU, or auto's highest cpu_capacity.
    fn pinOnce(p: *PlanLink) void {
        if (p.pinned) return;
        p.pinned = true;
        if (comptime builtin.os.tag != .linux) return;
        const want = p.pin orelse return;
        const cpu = switch (want) {
            .cpu => |c| c,
            .auto => bestCpu(p.avoid) orelse {
                log.info("[tensorfold] tp: TF_TP_PLAN_PIN=auto: no cpu_capacity differs among the allowed CPUs; the plan thread stays unpinned", .{});
                return;
            },
        };
        pinTo(cpu) catch |e| {
            log.warn("[tensorfold] tp: TF_TP_PLAN_PIN: CPU {d} refused ({t}); the plan thread stays unpinned", .{ cpu, e });
            return;
        };
        pinned_cpu.store(@intCast(cpu), .release);
        log.info("[tensorfold] tp: plan link rank {d} pinned to CPU {d}", .{ p.rank, cpu });
    }

    pub fn close(p: *PlanLink) void {
        for (&p.fds) |*fd| if (fd.* >= 0) {
            sock.close(fd.*);
            fd.* = -1;
        };
    }
};

/// The calling thread onto `cpu` alone.
pub fn pinTo(cpu: u32) !void {
    if (cpu >= linux.CPU_SETSIZE) return error.BadCpu;
    var set: linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    try linux.sched_setaffinity(0, &set);
}

/// Auto pin: the allowed CPU (not `avoid`) with the highest cpu_capacity, lowest on ties; null when all are equal.
pub fn bestCpu(avoid: ?u32) ?u32 {
    var set: linux.cpu_set_t = undefined;
    if (linux.errno(linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set)) != .SUCCESS) return null;
    var best: ?u32 = null;
    var best_cap: u32 = 0;
    var low_cap: u32 = std.math.maxInt(u32);
    for (set, 0..) |word, wi| {
        var bits = word;
        while (bits != 0) : (bits &= bits - 1) {
            const cpu: u32 = @intCast(wi * @bitSizeOf(usize) + @ctz(bits));
            if (avoid == cpu) continue;
            const c = capacity(cpu) orelse continue;
            low_cap = @min(low_cap, c);
            if (best == null or c > best_cap) {
                best = cpu;
                best_cap = c;
            }
        }
    }
    return if (best != null and best_cap > low_cap) best else null;
}

fn capacity(cpu: u32) ?u32 {
    var pb: [64]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&pb, "/sys/devices/system/cpu/cpu{d}/cpu_capacity", .{cpu}, 0) catch return null;
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    var b: [16]u8 = undefined;
    const n = linux.read(fd, &b, b.len);
    if (linux.errno(n) != .SUCCESS) return null;
    return std.fmt.parseInt(u32, std.mem.trim(u8, b[0..n], " \n"), 10) catch null;
}

test "plans arrive whole and in order; a closed rank 0 ends the follower" {
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    var big: [5000]i64 = undefined;
    for (&big, 0..) |*v, i| v.* = @as(i64, @intCast(i)) * -3 + 0x7FFF_0000_0000;
    try lead.send(&.{ 0x504C414E, 2, 1 });
    try lead.send(&big);
    try lead.send(&.{});
    var buf: [8192]i64 = undefined;
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 2, 1 }, try follow.recv(&buf));
    try std.testing.expectEqualSlices(i64, &big, try follow.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try follow.recv(&buf)).len);
    try std.testing.expectError(error.NotRankZero, follow.send(&.{1}));
    var small: [2]i64 = undefined;
    try lead.send(&.{ 1, 2, 3 });
    try std.testing.expectError(error.TooLarge, follow.recv(&small));
    lead.close();
    try std.testing.expectError(error.PeerClosed, follow.recv(&buf));
}

test "an admission-sized plan (1.2M words) arrives whole through the growable path" {
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    defer lead.close();
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    const gpa = std.testing.allocator;
    const prompt = try gpa.alloc(i64, 1_200_000);
    defer gpa.free(prompt);
    for (prompt, 0..) |*v, i| v.* = @intCast((i *% 2654435761) % 163840);
    // the socket buffer holds far less than 9.6 MB: rank 0 sends from its own thread, as it would
    const Send = struct {
        fn go(link: *PlanLink, ints: []const i64) void {
            link.send(&.{ 0x504C414E, 3, 20 }) catch {};
            link.send(ints) catch {};
            link.send(&.{ 0x504C414E, 3, 21 }) catch {};
        }
    };
    const t = try std.Thread.spawn(.{}, Send.go, .{ &lead, prompt });
    var buf: std.ArrayList(i64) = .empty;
    defer buf.deinit(gpa);
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 3, 20 }, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, prompt, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 3, 21 }, try follow.recvGrow(gpa, &buf));
    t.join();
    try std.testing.expect(buf.capacity >= prompt.len); // kept for the next admission
}

test "TF_TP_PLAN_LINK=rdma: a spinning follower gets every plan, late or early; TF_TP_PLAN_PIN pins the thread" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    defer lead.close();
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    follow.spin_ns = 2 * std.time.ns_per_ms;
    // the CPU this thread may run on first: the pin keeps it allowed, so the test runner is not starved
    var set: linux.cpu_set_t = undefined;
    _ = linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set);
    var first: u32 = 0;
    for (set, 0..) |w, i| if (w != 0) {
        first = @intCast(i * @bitSizeOf(usize) + @ctz(w));
        break;
    };
    const Follower = struct {
        fn run(f: *PlanLink, cpu: u32, out: *[3]i64, ok: *bool) void {
            f.pin = .{ .cpu = cpu };
            var buf: [8]i64 = undefined;
            const x = (f.recv(&buf) catch return)[0]; // arrives within the spin
            const y = (f.recv(&buf) catch return)[0]; // arrives after it: the blocking read
            var now: linux.cpu_set_t = undefined;
            _ = linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &now);
            var count: usize = 0;
            for (now) |w| count += @popCount(w);
            out.* = .{ x, y, @intCast(count) };
            ok.* = f.pinned and (now[cpu / @bitSizeOf(usize)] >> @intCast(cpu % @bitSizeOf(usize))) & 1 == 1;
        }
    };
    var out: [3]i64 = .{ 0, 0, 0 };
    var ok = false;
    const t = try std.Thread.spawn(.{}, Follower.run, .{ &follow, first, &out, &ok });
    try lead.send(&.{11});
    var ts: linux.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    try lead.send(&.{22});
    t.join();
    try std.testing.expectEqual([3]i64{ 11, 22, 1 }, out);
    try std.testing.expect(ok);
    if (bestCpu(null)) |c| try std.testing.expect((set[c / @bitSizeOf(usize)] >> @intCast(c % @bitSizeOf(usize))) & 1 == 1);
}

test "held plans leave with the next plan, in order, as one frame; a large plan after them flushes them first" {
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("127.0.0.1", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    const lead = try std.testing.allocator.create(PlanLink);
    defer std.testing.allocator.destroy(lead);
    lead.* = PlanLink.init(0, 2, f0);
    const follow = try std.testing.allocator.create(PlanLink);
    defer std.testing.allocator.destroy(follow);
    follow.* = PlanLink.init(1, 2, f1);
    defer follow.close();
    // a round: three keeps held, the window sent: one frame
    try lead.hold(&.{ 22, 0, 3 });
    try lead.hold(&.{ 22, 1, 0 });
    try lead.hold(&.{ 22, 2, 5 });
    try lead.send(&.{ 21, 4, 9, 9, 9 });
    try std.testing.expectEqual(@as(u64, 1), lead.sent);
    // a held plan, then one too large for a batch: two frames, the held one first
    var big: [5000]i64 = undefined;
    for (&big, 0..) |*v, i| v.* = @intCast(i);
    try lead.hold(&.{ 22, 3, 1 });
    try lead.send(&big);
    try std.testing.expectEqual(@as(u64, 3), lead.sent);
    // flush alone, an empty plan held, a plain send after
    try lead.hold(&.{});
    try lead.flush();
    try lead.flush(); // nothing held: nothing sent
    try lead.send(&.{7});
    try std.testing.expectEqual(@as(u64, 5), lead.sent);
    var buf: std.ArrayList(i64) = .empty;
    defer buf.deinit(std.testing.allocator);
    const gpa = std.testing.allocator;
    try std.testing.expectEqualSlices(i64, &.{ 22, 0, 3 }, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, &.{ 22, 1, 0 }, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, &.{ 22, 2, 5 }, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, &.{ 21, 4, 9, 9, 9 }, try follow.recvGrow(gpa, &buf));
    var out: [8192]i64 = undefined;
    try std.testing.expectEqualSlices(i64, &.{ 22, 3, 1 }, try follow.recv(&out));
    try std.testing.expectEqualSlices(i64, &big, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqual(@as(usize, 0), (try follow.recv(&out)).len);
    try std.testing.expectEqualSlices(i64, &.{7}, try follow.recv(&out));
    try std.testing.expectEqual(@as(u64, 3), follow.batches);
    try std.testing.expectError(error.NotRankZero, follow.hold(&.{1}));
    lead.close();
}
