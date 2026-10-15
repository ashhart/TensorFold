//! TF_GLM_PREFILL_REDUCE=ring: split's exchanges passed along the ring, for fabrics where ranks reach only neighbours.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");

/// One piece's relay: each owner gets every rank's block in split's slot, so it sums split's bytes in split's order.
pub const Relay = struct {
    world: usize,
    rank: usize,
    n: usize, // the piece's rows

    pub fn next(x: Relay) usize {
        return (x.rank + 1) % x.world;
    }
    pub fn prev(x: Relay) usize {
        return (x.rank + x.world - 1) % x.world;
    }
    /// Whose blocks this rank sends at step s (its own at step 1).
    pub fn sentFrom(x: Relay, s: usize) usize {
        return (x.rank + x.world + 1 - s) % x.world;
    }
    /// Whose blocks this rank takes at step s.
    pub fn takenFrom(x: Relay, s: usize) usize {
        return (x.rank + x.world - s) % x.world;
    }
    /// The i-th owner after this rank: step s sends owners 0 .. world - s, and keeps 0 .. world - s - 1 to send on.
    pub fn owner(x: Relay, i: usize) usize {
        return (x.rank + 1 + i) % x.world;
    }
    /// Owner p's rows of the piece (split's shares).
    pub fn rows(x: Relay, p: usize) [2]usize {
        return fw.Forward.share(x.n, x.world, p);
    }
    /// Rows into a transit where owner(i)'s block begins.
    pub fn at(x: Relay, i: usize) usize {
        var o: usize = 0;
        for (0..i) |k| {
            const r = x.rows(x.owner(k));
            o += r[1] - r[0];
        }
        return o;
    }
    /// Rows into a piece's room where transit t (by step parity) begins: past every slot split fills.
    pub fn transit(x: Relay, t: usize) usize {
        const most = (x.n + x.world - 1) / x.world;
        return x.world * most + t * (x.world - 2) * most;
    }
};

/// A piece's partials (`part`: this rank's fp32 rows) into `room`'s slots on `side`; NCCL keeps a pair's posting order.
pub fn partials(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, side: cuda.Stream, x: Relay, room: u64, part: u64, lo: usize, dims: usize) !void {
    const mine = x.rows(x.rank);
    const own = mine[1] - mine[0];
    for (1..x.world) |s| {
        const from = x.transit((s - 1) % 2); // what step s - 1 kept (step 1 sends from this rank's partial)
        const into = x.transit(s % 2);
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        for (0..x.world - s) |i| {
            const p = x.rows(x.owner(i));
            const src = if (s == 1) part + (lo + p[0]) * dims * 4 else room + (from + x.at(i)) * dims * 4;
            try lib.check(lib.api.ncclSend(src, (p[1] - p[0]) * dims, .f32, @intCast(x.next()), comm, side.handle), "ncclSend");
        }
        try lib.check(lib.api.ncclRecv(room + x.takenFrom(s) * own * dims * 4, own * dims, .f32, @intCast(x.prev()), comm, side.handle), "ncclRecv");
        for (0..x.world - s - 1) |i| {
            const p = x.rows(x.owner(i));
            try lib.check(lib.api.ncclRecv(room + (into + x.at(i)) * dims * 4, (p[1] - p[0]) * dims, .f32, @intCast(x.prev()), comm, side.handle), "ncclRecv");
        }
        try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }
}

/// A piece's glued rows (`rows_of`, bf16) to every rank, each step sending on what the last one brought in.
pub fn glued(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, side: cuda.Stream, x: Relay, rows_of: u64, lo: usize, dims: usize) !void {
    for (1..x.world) |s| {
        const out = x.rows(x.sentFrom(s));
        const got = x.rows(x.takenFrom(s));
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        try lib.check(lib.api.ncclSend(rows_of + (lo + out[0]) * dims * 2, (out[1] - out[0]) * dims, .bf16, @intCast(x.next()), comm, side.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(rows_of + (lo + got[0]) * dims * 2, (got[1] - got[0]) * dims, .bf16, @intCast(x.prev()), comm, side.handle), "ncclRecv");
        try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }
}

test "Relay: each owner's slot q holds rank q's block of its rows, as split's direct sends leave it; the transits fit" {
    const gpa = std.testing.allocator;
    for (2..7) |w| for ([_]usize{ 0, 1, w - 1, 2 * w + 1, 683 }) |extra| {
        const n = w + extra; // a split piece has a row for every rank
        // rank q's partial of piece row j is q << 16 | j (one value a row); every rank's room is the piece's [w * n]
        const part = try gpa.alloc(u32, w * n);
        defer gpa.free(part);
        const rooms = try gpa.alloc(u32, w * w * n);
        defer gpa.free(rooms);
        @memset(rooms, std.math.maxInt(u32));
        for (0..w) |q| for (0..n) |j| {
            part[q * n + j] = @intCast(q << 16 | j);
        };
        var wire: std.ArrayList(u32) = .empty; // what each rank sent its next one this step, rank after rank
        defer wire.deinit(gpa);
        var cut: [8]usize = undefined;
        for (1..w) |s| {
            wire.clearRetainingCapacity();
            for (0..w) |r| { // every rank's sends first: a step's receives write nothing its sends read
                const x: Relay = .{ .world = w, .rank = r, .n = n };
                const room = rooms[r * w * n ..][0 .. w * n];
                cut[r] = wire.items.len;
                for (0..w - s) |i| {
                    const p = x.rows(x.owner(i));
                    const len = p[1] - p[0];
                    try wire.appendSlice(gpa, if (s == 1) part[r * n + p[0] ..][0..len] else room[x.transit((s - 1) % 2) + x.at(i) ..][0..len]);
                }
            }
            for (0..w) |r| {
                const x: Relay = .{ .world = w, .rank = r, .n = n };
                const room = rooms[r * w * n ..][0 .. w * n];
                var got = wire.items[cut[x.prev()]..if (x.prev() + 1 < w) cut[x.prev() + 1] else wire.items.len];
                const mine = x.rows(r);
                const own = mine[1] - mine[0];
                @memcpy(room[x.takenFrom(s) * own ..][0..own], got[0..own]);
                got = got[own..];
                for (0..w - s - 1) |i| {
                    const p = x.rows(x.owner(i));
                    const len = p[1] - p[0];
                    @memcpy(room[x.transit(s % 2) + x.at(i) ..][0..len], got[0..len]);
                    got = got[len..];
                }
                try std.testing.expectEqual(@as(usize, 0), got.len);
            }
        }
        for (0..w) |p| {
            const x: Relay = .{ .world = w, .rank = p, .n = n };
            const mine = x.rows(p);
            const own = mine[1] - mine[0];
            const room = rooms[p * w * n ..][0 .. w * n];
            @memcpy(room[p * own ..][0..own], part[p * n + mine[0] ..][0..own]); // the own block, as split copies it
            for (0..w) |q| for (0..own) |k| try std.testing.expectEqual(@as(u32, @intCast(q << 16 | (mine[0] + k))), room[q * own + k]);
            try std.testing.expect(x.transit(1) + (w - 2) * ((n + w - 1) / w) <= w * n);
        }
    };
}

test "Relay: a piece's glued rows reach every rank, each step sending what the last one brought in" {
    for (2..7) |w| for ([_]usize{ 0, 1, w - 1, 2 * w + 1 }) |extra| {
        const n = w + extra;
        var rows: [6][64]u32 = undefined; // rank r's copy of the piece's rows: its own glued, the rest unknown
        for (0..w) |r| {
            const mine = (Relay{ .world = w, .rank = r, .n = n }).rows(r);
            for (0..n) |j| rows[r][j] = if (j >= mine[0] and j < mine[1]) @intCast(j) else std.math.maxInt(u32);
        }
        for (1..w) |s| {
            var sent: [6][64]u32 = undefined; // every rank's send of step s, before any receive lands
            for (0..w) |r| {
                const x: Relay = .{ .world = w, .rank = r, .n = n };
                const out = x.rows(x.sentFrom(s));
                @memcpy(sent[r][out[0]..out[1]], rows[r][out[0]..out[1]]);
            }
            for (0..w) |r| {
                const x: Relay = .{ .world = w, .rank = r, .n = n };
                const y: Relay = .{ .world = w, .rank = x.prev(), .n = n };
                const got = x.rows(x.takenFrom(s));
                try std.testing.expectEqual(y.rows(y.sentFrom(s)), got); // rank - 1 sends the block this rank takes
                @memcpy(rows[r][got[0]..got[1]], sent[x.prev()][got[0]..got[1]]);
            }
        }
        for (0..w) |r| for (0..n) |j| try std.testing.expectEqual(@as(u32, @intCast(j)), rows[r][j]);
    };
}
