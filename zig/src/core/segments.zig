//! Staggered prompt segments, for any family: one prompt chunk split into segments, each on its own Metal queue.
//! Segment k runs each layer's mixer after segment k-1's (an MTLEvent between the queues) and the rest of its layer
//! beside the other segments, so one segment's scans and glue overlap another's matrix work. Each segment does what a
//! serial chunk of its rows does, so the output equals the chunks run in order. Measured on Flash Next 6-bit (M5
//! Ultra): two segments +9% at 8k-64k, three +2%, four time out.
const std = @import("std");
const mtl = @import("metal");
const Fence = @import("fence.zig").Fence;

/// Segments one chunk can take.
pub const MAX = 4;

/// Where layer i waits for the segment before: ahead of the segment's own pre-mixer work (when that work reads what
/// the segment before wrote there, like a convolution tail) or just ahead of its mixer (recurrent states, cache keys).
pub const Wait = enum { pre, mixer };

/// A segment's command buffer and open encoder; hooks encode into `enc` and leave it open.
pub const Lane = struct {
    k: usize,
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
    fence: Fence, // orders the segment's encoders: buffers are untracked, so encoders could overlap
};

/// Rows of segment k when `n` rows split `parts` ways (near-equal, the first ones longer).
pub fn rows(n: usize, parts: usize, k: usize) usize {
    return n / parts + @intFromBool(k < n % parts);
}

/// The first row of segment k.
pub fn start(n: usize, parts: usize, k: usize) usize {
    return k * (n / parts) + @min(k, n % parts);
}

pub const Call = struct { rows: usize, parts: usize };

/// A prompt's next call with `left` rows to go: the rows split evenly over the fewest calls of two segments of up to
/// `step` rows each, while every segment gets `min_rows`; else one serial chunk of up to `step`.
pub fn next(left: usize, step: usize, min_rows: usize) Call {
    const calls = @max(1, (left + 2 * step - 1) / (2 * step));
    const rows_ = (left + calls - 1) / calls;
    if (rows_ < 2 * min_rows) return .{ .rows = @min(step, left), .parts = 1 };
    return .{ .rows = rows_, .parts = 2 };
}

/// The order every runner follows: each segment's begin; per layer, per segment in row order: pre, mixer, post, with
/// segment k waiting (ahead of pre or of the mixer, as the family says) for value i + 1, which segment k-1 signals
/// after its layer-i mixer; then each segment's finish. Hooks of segment k-1 encode before k's, so host-side state
/// (cache counters) moves in row order.
fn drive(n: usize, layers: usize, x: anytype) !void {
    for (0..n) |k| try x.begin(k);
    for (0..layers) |i| {
        const w: Wait = x.wait(i);
        for (0..n) |k| {
            if (k > 0 and w == .pre) try x.waitFor(k, i);
            try x.pre(k, i);
            if (k > 0 and w == .mixer) try x.waitFor(k, i);
            try x.mixer(k, i);
            if (k + 1 < n) x.signal(k, i);
            try x.post(k, i);
        }
    }
    for (0..n) |k| try x.finish(k);
}

/// Runs one chunk's segments, segment k on queues[k], over `layers` layers, through the family's hooks on `fam`:
///   begin(lane), pre(lane, i), mixer(lane, i), post(lane, i), finish(lane): the segment's work into lane.enc;
///   wait(i) Wait: where layer i waits for the segment before; handoff(lane, i): right after that wait.
/// Commits every segment and waits for all; returns the longest segment's GPU seconds. One queue is a serial chunk.
pub fn run(device: mtl.Device, queues: []const mtl.Queue, layers: usize, mode: mtl.DispatchType, fam: anytype) !f64 {
    const n = queues.len;
    if (n == 0 or n > MAX) return error.Segments;
    var evs: [MAX - 1]mtl.Event = undefined;
    var n_evs: usize = 0;
    defer for (evs[0..n_evs]) |e| e.deinit();
    while (n_evs + 1 < n) : (n_evs += 1) evs[n_evs] = try device.event();
    var lanes: [MAX]Lane = undefined;
    var n_lanes: usize = 0;
    defer for (lanes[0..n_lanes]) |l| l.fence.deinit();
    for (queues) |q| {
        const fence = try Fence.init(device);
        const cb = q.commandBuffer();
        lanes[n_lanes] = .{ .k = n_lanes, .cb = cb, .enc = cb.compute(mode), .fence = fence };
        n_lanes += 1;
    }
    const X = struct {
        fam: @TypeOf(fam),
        lanes: []Lane,
        evs: []const mtl.Event,
        mode: mtl.DispatchType,

        /// Ends the lane's encoder, waits for (or signals) `value` between its command buffer's encoders, and opens
        /// the next encoder behind the fence.
        fn sync(x: *const @This(), l: *Lane, ev: mtl.Event, value: u64, wait_: bool) void {
            l.fence.update(l.enc);
            l.enc.end();
            if (wait_) l.cb.waitForEvent(ev, value) else l.cb.signalEvent(ev, value);
            l.enc = l.cb.compute(x.mode);
            l.fence.wait(l.enc);
        }
        fn begin(x: *const @This(), k: usize) !void {
            try x.fam.begin(&x.lanes[k]);
        }
        fn wait(x: *const @This(), i: usize) Wait {
            return x.fam.wait(i);
        }
        fn waitFor(x: *const @This(), k: usize, i: usize) !void {
            x.sync(&x.lanes[k], x.evs[k - 1], i + 1, true);
            try x.fam.handoff(&x.lanes[k], i);
        }
        fn signal(x: *const @This(), k: usize, i: usize) void {
            x.sync(&x.lanes[k], x.evs[k], i + 1, false);
        }
        fn pre(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.pre(&x.lanes[k], i);
        }
        fn mixer(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.mixer(&x.lanes[k], i);
        }
        fn post(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.post(&x.lanes[k], i);
        }
        fn finish(x: *const @This(), k: usize) !void {
            try x.fam.finish(&x.lanes[k]);
        }
    };
    const x: X = .{ .fam = fam, .lanes = lanes[0..n], .evs = evs[0..n_evs], .mode = mode };
    try drive(n, layers, &x);
    for (lanes[0..n]) |l| l.enc.end();
    for (lanes[0..n]) |l| l.cb.commit();
    var gpu: f64 = 0;
    for (lanes[0..n]) |l| {
        l.cb.wait();
        if (l.cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        gpu = @max(gpu, l.cb.gpuSeconds());
    }
    return gpu;
}

test "rows, starts and calls" {
    var at: usize = 0;
    for (0..3) |k| {
        try std.testing.expectEqual(at, start(10, 3, k));
        at += rows(10, 3, k);
    }
    try std.testing.expectEqual(@as(usize, 10), at);
    try std.testing.expectEqual(Call{ .rows = 13334, .parts = 2 }, next(40000, 8192, 4096)); // three even calls
    try std.testing.expectEqual(Call{ .rows = 10000, .parts = 2 }, next(20000, 8192, 4096));
    try std.testing.expectEqual(Call{ .rows = 8625, .parts = 2 }, next(8625, 8192, 4096));
    try std.testing.expectEqual(Call{ .rows = 4096, .parts = 1 }, next(4096, 8192, 4096));
    try std.testing.expectEqual(Call{ .rows = 8192, .parts = 1 }, next(9000, 8192, 8192));
    try std.testing.expectEqual(Call{ .rows = 0, .parts = 1 }, next(0, 8192, 4096));
    var left: usize = 70001; // every call of a long prompt keeps both segments at min_rows or more
    while (left > 0) {
        const c = next(left, 8192, 4096);
        try std.testing.expect(c.rows <= 2 * 8192 and (c.parts == 1 or c.rows >= 2 * 4096));
        left -= c.rows;
    }
}

// The schedule as queues would run it: each segment's steps in order, a wait blocking until the segment before has
// signalled its value; every hook of a layer after the waits it needs.
test "the schedule runs to the end on queues and keeps each layer's dependencies" {
    const Step = struct { kind: enum { begin, wait, pre, mixer, signal, post, finish }, i: usize };
    const Rec = struct {
        steps: [MAX][64]Step = undefined,
        len: [MAX]usize = @splat(0),
        early: usize, // the layer that waits ahead of pre

        fn add(r: *@This(), k: usize, s: Step) void {
            r.steps[k][r.len[k]] = s;
            r.len[k] += 1;
        }
        fn begin(r: *@This(), k: usize) !void {
            r.add(k, .{ .kind = .begin, .i = 0 });
        }
        fn wait(r: *@This(), i: usize) Wait {
            return if (i == r.early) .pre else .mixer;
        }
        fn waitFor(r: *@This(), k: usize, i: usize) !void {
            r.add(k, .{ .kind = .wait, .i = i });
        }
        fn signal(r: *@This(), k: usize, i: usize) void {
            r.add(k, .{ .kind = .signal, .i = i });
        }
        fn pre(r: *@This(), k: usize, i: usize) !void {
            r.add(k, .{ .kind = .pre, .i = i });
        }
        fn mixer(r: *@This(), k: usize, i: usize) !void {
            r.add(k, .{ .kind = .mixer, .i = i });
        }
        fn post(r: *@This(), k: usize, i: usize) !void {
            r.add(k, .{ .kind = .post, .i = i });
        }
        fn finish(r: *@This(), k: usize) !void {
            r.add(k, .{ .kind = .finish, .i = 0 });
        }
    };
    for (1..MAX + 1) |n| {
        var rec: Rec = .{ .early = 1 };
        try drive(n, 4, &rec);
        // the queues: a wait for layer i passes once segment k-1 signalled layer i; nothing else blocks
        var pc: [MAX]usize = @splat(0);
        var signalled: [MAX]?usize = @splat(null);
        var mixed: [MAX]usize = @splat(0); // layers whose mixer has run
        var moved = true;
        while (moved) {
            moved = false;
            for (0..n) |k| {
                while (pc[k] < rec.len[k]) {
                    const s = rec.steps[k][pc[k]];
                    if (s.kind == .wait and (signalled[k - 1] == null or signalled[k - 1].? < s.i)) break;
                    switch (s.kind) {
                        .signal => signalled[k] = s.i,
                        .mixer => {
                            if (k > 0) try std.testing.expect(mixed[k - 1] > s.i); // the segment before's mixer is done
                            mixed[k] = s.i + 1;
                        },
                        .pre => if (k > 0 and s.i == rec.early) try std.testing.expect(mixed[k - 1] > s.i),
                        else => {},
                    }
                    pc[k] += 1;
                    moved = true;
                }
            }
        }
        for (0..n) |k| try std.testing.expectEqual(rec.len[k], pc[k]); // no queue left waiting
        for (0..n) |k| { // begin, finish, three parts a layer, a wait a layer past the first, a signal before the last
            const want: usize = 2 + 3 * 4 + @as(usize, if (k > 0) 4 else 0) + @as(usize, if (k + 1 < n) 4 else 0);
            try std.testing.expectEqual(want, rec.len[k]);
        }
    }
}
