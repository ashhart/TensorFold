//! Staggered prompt segments, backend-neutral: calls, segment rows, and the hook order core/ and cuda/segments.zig run.
const std = @import("std");

/// Segments one chunk can take.
pub const MAX = 4;

/// Where layer i waits for the segment before: ahead of pre (it reads what that one wrote, a conv tail) or of the mixer.
pub const Wait = enum { pre, mixer };

/// Rows of segment k when `n` rows split `parts` ways (near-equal, the first ones longer).
pub fn rows(n: usize, parts: usize, k: usize) usize {
    return n / parts + @intFromBool(k < n % parts);
}

/// The first row of segment k.
pub fn start(n: usize, parts: usize, k: usize) usize {
    return k * (n / parts) + @min(k, n % parts);
}

pub const Call = struct { rows: usize, parts: usize };

/// A prompt's next call: `left` rows evenly over the fewest calls of two segments of up to `step`, each `min_rows` or more.
pub fn next(left: usize, step: usize, min_rows: usize) Call {
    const calls = @max(1, (left + 2 * step - 1) / (2 * step));
    const rows_ = (left + calls - 1) / calls;
    if (rows_ < 2 * min_rows) return .{ .rows = @min(step, left), .parts = 1 };
    return .{ .rows = rows_, .parts = 2 };
}

/// A prompt's next call of up to `parts` whole serial chunks of `step` rows: each segment is the chunk at its place.
pub fn chunked(left: usize, step: usize, parts: usize) Call {
    const rows_ = @min(left, parts * step);
    return .{ .rows = rows_, .parts = @max(1, (rows_ + step - 1) / step) };
}

/// Rows of segment k of a chunked call of `n` rows.
pub fn chunkRows(n: usize, step: usize, k: usize) usize {
    return @min(step, n - k * step);
}

/// The hook order: begins; per layer, per segment in row order: pre, mixer, post, k waiting on k-1's post-mixer signal.
pub fn drive(n: usize, layers: usize, x: anytype) !void {
    // k-1's hooks queue before k's (host counters move in row order; a reused event's wait follows its signal)
    for (0..n) |k| try x.begin(k);
    for (0..layers) |i| {
        const w: Wait = x.wait(i);
        for (0..n) |k| {
            if (k > 0 and w == .pre) try x.waitFor(k, i);
            try x.pre(k, i);
            if (k > 0 and w == .mixer) try x.waitFor(k, i);
            try x.mixer(k, i);
            if (k + 1 < n) try x.signal(k, i);
            try x.post(k, i);
        }
    }
    for (0..n) |k| try x.finish(k);
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

test "chunked calls cut a prompt exactly where serial chunks do" {
    try std.testing.expectEqual(Call{ .rows = 4096, .parts = 2 }, chunked(8203, 2048, 2));
    try std.testing.expectEqual(Call{ .rows = 11, .parts = 1 }, chunked(11, 2048, 4));
    try std.testing.expectEqual(Call{ .rows = 2059, .parts = 2 }, chunked(2059, 2048, 3));
    try std.testing.expectEqual(Call{ .rows = 2048, .parts = 1 }, chunked(2048, 2048, 2));
    try std.testing.expectEqual(@as(usize, 11), chunkRows(2059, 2048, 1));
    for ([_]usize{ 1, 2047, 2048, 2049, 8203, 32789, 70001 }) |len| {
        for (1..MAX + 1) |parts| {
            var pos: usize = 0; // every segment is the serial chunk at its place: start on the grid, full but the last
            while (pos < len) {
                const c = chunked(len - pos, 2048, parts);
                try std.testing.expect(c.parts >= 1 and c.parts <= parts and c.rows > 0);
                var sum: usize = 0;
                for (0..c.parts) |k| {
                    const r = chunkRows(c.rows, 2048, k);
                    try std.testing.expect((pos + sum) % 2048 == 0 and r > 0 and r <= 2048);
                    try std.testing.expect(r == 2048 or pos + sum + r == len);
                    sum += r;
                }
                try std.testing.expectEqual(c.rows, sum);
                pos += c.rows;
            }
            try std.testing.expectEqual(len, pos);
        }
    }
}

/// Records each segment's steps as a runner would encode them, for the schedule tests.
const Recorder = struct {
    const Step = struct { kind: enum { begin, wait, pre, mixer, signal, post, finish }, i: usize };
    steps: [MAX][64]Step = undefined,
    len: [MAX]usize = @splat(0),
    early: usize, // the layer that waits ahead of pre

    fn add(r: *Recorder, k: usize, s: Step) void {
        r.steps[k][r.len[k]] = s;
        r.len[k] += 1;
    }
    pub fn begin(r: *Recorder, k: usize) !void {
        r.add(k, .{ .kind = .begin, .i = 0 });
    }
    pub fn wait(r: *Recorder, i: usize) Wait {
        return if (i == r.early) .pre else .mixer;
    }
    pub fn waitFor(r: *Recorder, k: usize, i: usize) !void {
        r.add(k, .{ .kind = .wait, .i = i });
    }
    pub fn signal(r: *Recorder, k: usize, i: usize) !void {
        r.add(k, .{ .kind = .signal, .i = i });
    }
    pub fn pre(r: *Recorder, k: usize, i: usize) !void {
        r.add(k, .{ .kind = .pre, .i = i });
    }
    pub fn mixer(r: *Recorder, k: usize, i: usize) !void {
        r.add(k, .{ .kind = .mixer, .i = i });
    }
    pub fn post(r: *Recorder, k: usize, i: usize) !void {
        r.add(k, .{ .kind = .post, .i = i });
    }
    pub fn finish(r: *Recorder, k: usize) !void {
        r.add(k, .{ .kind = .finish, .i = 0 });
    }
};

// The schedule on simulated queues: a wait blocks until the segment before signalled its layer; each hook after its waits.
test "the schedule runs to the end on queues and keeps each layer's dependencies" {
    for (1..MAX + 1) |n| {
        var rec: Recorder = .{ .early = 1 };
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
