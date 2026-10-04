//! The round barrier: each rank writes its finished round into a word at every peer (one-sided, no remote atomics).
const std = @import("std");

pub const Error = error{ PeerTimeout, LinkDown };

/// One word slot per rank at every peer; MCDMA posts are RC writes, so a word lands after the data posted before it.
pub const Words = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        rank: *const fn (ptr: *anyopaque) u32,
        size: *const fn (ptr: *anyopaque) u32,
        post: *const fn (ptr: *anyopaque, peer: u32, slot: u32, value: u64) Error!void,
        /// An acquire load of this rank's own slot (where peer `slot` posts).
        load: *const fn (ptr: *anyopaque, slot: u32) u64,
    };
};

/// Rounds in lockstep with one round of overlap: round N+1's data may go out once every rank finished round N-1.
pub const Lockstep = struct {
    w: Words,
    timeout_ns: u64 = 10 * std.time.ns_per_s,

    fn rank(l: *const Lockstep) u32 {
        return l.w.vtable.rank(l.w.ptr);
    }

    fn size(l: *const Lockstep) u32 {
        return l.w.vtable.size(l.w.ptr);
    }

    fn word(l: *const Lockstep, slot: u32) u64 {
        return l.w.vtable.load(l.w.ptr, slot);
    }

    /// This rank's GPU work for round `r` is done (its buffers for r may be reused): tell every peer.
    pub fn arrive(l: *Lockstep, r: u64) Error!void {
        const me = l.rank();
        for (0..l.size()) |p| try l.w.vtable.post(l.w.ptr, @intCast(p), me, r);
    }

    /// Wait until every rank has arrived at round `r` (monotonic words: a later round counts too).
    pub fn wait(l: *Lockstep, r: u64) Error!void {
        const start = now();
        var spins: u32 = 0;
        while (l.lateCount(r) > 0) {
            spins += 1;
            if (spins % 256 == 0) {
                std.Thread.yield() catch {};
                if (now() - start > l.timeout_ns) return error.PeerTimeout;
            } else std.atomic.spinLoopHint();
        }
    }

    /// Before writing round `r`'s data into the peers' buffers of parity r % 2: everyone is done with round r - 2.
    pub fn mayWrite(l: *Lockstep, r: u64) Error!void {
        if (r >= 2) try l.wait(r - 1);
    }

    fn lateCount(l: *const Lockstep, r: u64) usize {
        var n: usize = 0;
        for (0..l.size()) |p| n += @intFromBool(l.word(@intCast(p)) < r);
        return n;
    }

    /// The ranks that have not arrived at `r`, for the membership's suspicion and the status page.
    pub fn late(l: *const Lockstep, r: u64, out: []u32) usize {
        var n: usize = 0;
        for (0..l.size()) |p| {
            if (l.word(@intCast(p)) >= r or n == out.len) continue;
            out[n] = @intCast(p);
            n += 1;
        }
        return n;
    }
};

fn now() u64 {
    return @intCast(std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds);
}

/// In-process words for tests: words[dst * n + slot].
pub const MemWords = struct {
    n: u32,
    words: []std.atomic.Value(u64),
    ends: []End,
    gpa: std.mem.Allocator,

    const End = struct { m: *MemWords, me: u32 };

    pub fn init(gpa: std.mem.Allocator, n: u32) !*MemWords {
        const m = try gpa.create(MemWords);
        m.* = .{ .n = n, .words = try gpa.alloc(std.atomic.Value(u64), n * n), .ends = try gpa.alloc(End, n), .gpa = gpa };
        for (m.words) |*w| w.* = .init(0);
        for (m.ends, 0..) |*e, i| e.* = .{ .m = m, .me = @intCast(i) };
        return m;
    }

    pub fn deinit(m: *MemWords) void {
        m.gpa.free(m.words);
        m.gpa.free(m.ends);
        m.gpa.destroy(m);
    }

    pub fn endpoint(m: *MemWords, r: u32) Words {
        return .{ .ptr = &m.ends[r], .vtable = &vt };
    }

    const vt: Words.VTable = .{ .rank = rankOf, .size = sizeOf, .post = post, .load = load };

    fn rankOf(ptr: *anyopaque) u32 {
        const e: *End = @ptrCast(@alignCast(ptr));
        return e.me;
    }

    fn sizeOf(ptr: *anyopaque) u32 {
        const e: *End = @ptrCast(@alignCast(ptr));
        return e.m.n;
    }

    fn post(ptr: *anyopaque, peer: u32, slot: u32, value: u64) Error!void {
        const e: *End = @ptrCast(@alignCast(ptr));
        e.m.words[peer * e.m.n + slot].store(value, .release);
    }

    fn load(ptr: *anyopaque, slot: u32) u64 {
        const e: *End = @ptrCast(@alignCast(ptr));
        return e.m.words[e.me * e.m.n + slot].load(.acquire);
    }
};

test "four ranks stay within one round of each other while the host runs a round ahead" {
    const gpa = std.testing.allocator;
    const m = try MemWords.init(gpa, 4);
    defer m.deinit();
    var started: [4]std.atomic.Value(u64) = @splat(.init(0));
    var finished: [4]std.atomic.Value(u64) = @splat(.init(0));
    var worst: std.atomic.Value(u64) = .init(0);
    const Run = struct {
        fn go(w: Words, me: usize, s: *[4]std.atomic.Value(u64), f: *[4]std.atomic.Value(u64), bad: *std.atomic.Value(u64)) void {
            var l: Lockstep = .{ .w = w };
            var prng: std.Random.DefaultPrng = .init(me + 1);
            for (1..60) |r| {
                l.mayWrite(r) catch unreachable;
                s[me].store(r, .release);
                for (f) |*x| {
                    const lag = r -| x.load(.acquire);
                    if (lag > bad.load(.acquire)) bad.store(lag, .release);
                }
                for (0..prng.random().intRangeAtMost(u32, 0, 3000)) |_| std.atomic.spinLoopHint();
                f[me].store(r, .release);
                l.arrive(r) catch unreachable;
            }
            l.wait(59) catch unreachable;
        }
    };
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Run.go, .{ m.endpoint(@intCast(i)), i, &started, &finished, &worst });
    for (threads) |t| t.join();
    try std.testing.expect(worst.load(.acquire) <= 2);
}

test "a silent rank times out the barrier and is named late" {
    const gpa = std.testing.allocator;
    const m = try MemWords.init(gpa, 3);
    defer m.deinit();
    var a: Lockstep = .{ .w = m.endpoint(0), .timeout_ns = 20 * std.time.ns_per_ms };
    var b: Lockstep = .{ .w = m.endpoint(1) };
    try a.arrive(1);
    try b.arrive(1);
    try std.testing.expectError(error.PeerTimeout, a.wait(1));
    var who: [3]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 1), a.late(1, &who));
    try std.testing.expectEqual(@as(u32, 2), who[0]);
}
