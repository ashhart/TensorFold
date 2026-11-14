//! The prompt history fetch (cuda_owner_fetch.run) on 2 and 3 host ranks: each rank's outputs equal its unsplit ones.
const std = @import("std");
const fetch = @import("cuda_owner_fetch.zig");

const page = 512; // tf_glm_kvh_rows' PAGE
const stride = 3 * page; // row slots from one page to the next: a page holds several layers' rows
const max_pages = 12;
const trash = max_pages - 1; // the page a rank without a stream's rows maps every entry to
const room_rows = max_pages * page;
const max_chunk = 900;
const random_chunk = 300; // the random streams' chunks, to 2,048 rows: past the dense limit and four pages
const layers = 2;
const dense_limit = 400; // rows at or past it attend their picks, as dsa_block's sparse pass
const picks = 24;
const poison: u64 = 0xdead_beef_dead_beef;

fn mix(a: u64, b: u64) u64 { // splitmix64's finalizer over both: order-sensitive, cheap in a debug build
    var x = a *% 0x9e37_79b9_7f4a_7c15 +% b;
    x = (x ^ (x >> 30)) *% 0xbf58_476d_1ce4_e5b9;
    x = (x ^ (x >> 27)) *% 0x94d0_49bb_1331_11eb;
    return x ^ (x >> 31);
}

/// The latent row every rank computes for position `p` of layer `l` (the ranks' projections agree).
fn value(l: usize, p: usize) u64 {
    return mix(0x1a7e + l, p);
}

/// Row q's j-th pick (the token lists are the same on every rank: the index keys are on every rank).
fn pick(q: usize, j: usize) usize {
    return @intCast(mix(q, j) % (q + 1));
}

/// The keys row q reads, in the order it reads them: 0 .. q below the dense limit, else its picks.
fn keys(q: usize, j: usize) usize {
    return if (q < dense_limit) j else pick(q, j);
}

fn count(q: usize) usize {
    return if (q < dense_limit) q + 1 else picks;
}

/// The output of rank `rank`'s heads for row q of layer `l` over rows read by `at`; unsplit, `at` is `value`.
fn attendRow(rank: usize, l: usize, q: usize, ctx: anytype, comptime at: fn (@TypeOf(ctx), usize, usize) ?u64) !u64 {
    var acc = mix(rank + 1, q);
    for (0..count(q)) |j| acc = mix(acc, at(ctx, l, keys(q, j)) orelse return error.ReadPoison);
    return acc;
}

fn unsplit(_: void, l: usize, t: usize) ?u64 {
    return value(l, t);
}

const World = struct {
    wire: [room_rows]u64 = @splat(poison),
    wire_n: ?usize = null,
};

const Rank = struct {
    rank: usize,
    region: [max_pages * stride]u64 = @splat(poison),
    table: [max_pages]u32 = @splat(trash),
    room: [room_rows]u64 = @splat(poison),
    read_room: bool = false,
    read_pages: bool = false,

    fn slot(r: *const Rank, l: usize, t: usize) usize {
        return l * page + r.table[t / page] * stride + t % page; // lc = layer l's part of page 0, as tf_glm_kvh_rows
    }
};

/// cuda_owner_fetch.run's device: a host rank, one layer of one chunk.
const Model = struct {
    w: *World,
    r: *Rank,
    l: usize,
    host_pos: usize,
    rows: usize,
    out: []u64,

    pub fn pack(m: Model, n: usize) !void {
        for (0..@min(n, m.host_pos)) |s| m.r.room[s] = m.r.region[m.r.slot(m.l, s)];
    }

    pub fn broadcast(m: Model, owner: u32, n: usize) !void {
        if (m.r.rank == owner) {
            @memcpy(m.w.wire[0..n], m.r.room[0..n]);
            m.w.wire_n = n;
            return;
        }
        if (m.w.wire_n != n) return error.UnpairedBroadcast; // the owner posts first, the same count
        @memcpy(m.r.room[0..n], m.w.wire[0..n]);
    }

    fn fromRoom(r: *const Rank, _: usize, t: usize) ?u64 {
        const v = r.room[t];
        return if (v == poison) null else v;
    }

    fn fromPages(r: *const Rank, l: usize, t: usize) ?u64 {
        const v = r.region[r.slot(l, t)];
        return if (v == poison) null else v;
    }

    pub fn attend(m: Model, at: fetch.From) !void {
        const r = m.r;
        for (m.host_pos..m.host_pos + m.rows) |p| switch (at) { // the chunk's own rows, written before any attends
            .pages => r.region[r.slot(m.l, p)] = value(m.l, p),
            .room => r.room[p] = value(m.l, p),
        };
        switch (at) {
            .pages => r.read_pages = true,
            .room => r.read_room = true,
        }
        for (0..m.rows) |i| {
            const q = m.host_pos + i;
            m.out[i] = switch (at) {
                .pages => try attendRow(r.rank, m.l, q, @as(*const Rank, r), fromPages),
                .room => try attendRow(r.rank, m.l, q, @as(*const Rank, r), fromRoom),
            };
        }
    }
};

/// One stream on `world` host ranks, owned by `owner`, prompted in `chunks`: every rank's rows against unsplit ones.
fn check(world: usize, owner: u32, chunks: []const usize, seed: u64) !void {
    var w: World = .{};
    var ranks: [3]Rank = undefined;
    for (ranks[0..world], 0..) |*r, i| r.* = .{ .rank = i };
    var prng = std.Random.DefaultPrng.init(seed);
    var ids: [trash]u32 = undefined; // the owner's pages, out of order
    for (&ids, 0..) |*x, i| x.* = @intCast(i);
    prng.random().shuffle(u32, &ids);
    @memcpy(ranks[owner].table[0..trash], &ids);
    var out: [max_chunk]u64 = undefined;
    var pos: usize = 0;
    for (chunks) |rows| {
        if (pos + rows > trash * page) return error.PastTheRoom;
        for (0..layers) |l| {
            w.wire_n = null;
            var order: [3]usize = undefined; // the owner first: a broadcast's receivers take what its root posted
            order[0] = owner;
            var n: usize = 1;
            for (0..world) |p| if (p != owner) {
                order[n] = p;
                n += 1;
            };
            for (order[0..world]) |p| {
                const m: Model = .{ .w = &w, .r = &ranks[p], .l = l, .host_pos = pos, .rows = rows, .out = out[0..rows] };
                try fetch.run(Model, m, owner, p, pos);
                for (0..rows) |i| if (out[i] != try attendRow(p, l, pos + i, {}, unsplit)) return error.NotUnsplit;
            }
        }
        pos += rows;
    }
    for (ranks[0..world], 0..) |r, p| {
        if (p == owner and r.read_room) return error.OwnerReadTheRoom;
        if (p != owner and r.read_pages) return error.ReadWithoutRows;
    }
}

test "GLM owner fetch: each rank's heads over the owner's history equal unsplit, at 2 and 3 ranks, dense and sparse" {
    const chunks = [_]usize{ 300, 512, 1, 900, 640, 7, 900 }; // across the dense limit and pages' edges
    for ([_]usize{ 2, 3 }) |world| for (0..world) |o| try check(world, @intCast(o), &chunks, 7 + o);
    try check(2, 1, &.{900}, 1); // one chunk: no history, no exchange
}

test "GLM owner fetch: 200 random streams" {
    var prng = std.Random.DefaultPrng.init(0x3f37);
    const rnd = prng.random();
    for (0..200) |i| {
        const world = 2 + rnd.uintLessThan(usize, 2);
        var chunks: [12]usize = undefined;
        var n: usize = 0;
        var total: usize = 0;
        while (n < chunks.len) : (n += 1) {
            const rows = 1 + rnd.uintLessThan(usize, random_chunk);
            if (total + rows > 4 * page) break;
            chunks[n] = rows;
            total += rows;
        }
        try check(world, rnd.uintLessThan(u32, @intCast(world)), chunks[0..n], i);
    }
}

test "GLM owner fetch: a chunk fetches only on owner-only rows of a prompt, and only while its rows fit the room" {
    const two: fetch.Away = .{ .heads = 32, .to = 1 }; // TP=2: the other rank's 32 heads
    try std.testing.expect(fetch.fetches(true, 0, 10, 5, 15, two));
    try std.testing.expect(!fetch.fetches(true, 0, 10, 6, 15, two));
    try std.testing.expect(!fetch.fetches(false, 1, 0, 4, 15, two)); // a decode or verify window: its owner attends
    try std.testing.expect(!fetch.fetches(true, null, 0, 4, 15, two)); // every rank holds every row
    try std.testing.expectEqual(@as(usize, 131_072), fetch.roomRows(128 << 20, 512)); // TP=2's room by default: 2,048-row chunks
    try std.testing.expectEqual(@as(usize, 262_144), fetch.roomRows(256 << 20, 512)); // and with TF_GLM_CHUNK_SOLO=4096
    try std.testing.expectEqual(fetch.From.pages, fetch.from(1, 1));
    try std.testing.expectEqual(fetch.From.room, fetch.from(1, 0));
}

test "GLM owner fetch: a chunk fetches only while its history's broadcast moves no more bytes than its exchange" {
    const room = 131_072;
    const big = 1 << 30; // a room that never binds: the byte rule alone
    const two: fetch.Away = .{ .heads = 32, .to = 1 };
    // TP=2, a row 1,024 bytes: the broadcast moves host_pos rows, the exchange 2 x rows x 32
    try std.testing.expect(fetch.fetches(true, 0, room - 2048, 2048, room, two)); // full chunks: every one in the room
    try std.testing.expect(fetch.fetches(true, 0, 64 * 200, 200, big, two)); // equal bytes: the fetch
    try std.testing.expect(!fetch.fetches(true, 0, 64 * 200 + 1, 200, big, two));
    try std.testing.expect(!fetch.fetches(true, 0, 60_000, 200, room, two)); // a follow-up: 58.6 MiB a layer against 12.5
    try std.testing.expect(!fetch.fetches(true, 0, 100_000, 1, room, two)); // a 1-row tail: 97.7 MiB against 64 KiB
    try std.testing.expect(fetch.fetches(true, 0, 64, 1, room, two) and !fetch.fetches(true, 0, 65, 1, room, two));
    try std.testing.expect(fetch.fetches(true, 1, 0, 1, room, two)); // a first chunk has no history to move
    // TP=3 (22, 21, 21 heads): the broadcast reaches two ranks; the exchange moves the other two ranks' heads
    const from0: fetch.Away = .{ .heads = 42, .to = 2 };
    const from1: fetch.Away = .{ .heads = 43, .to = 2 };
    try std.testing.expect(fetch.fetches(true, 0, 42 * 100, 100, big, from0) and !fetch.fetches(true, 0, 42 * 100 + 1, 100, big, from0));
    try std.testing.expect(fetch.fetches(true, 1, 43 * 100, 100, big, from1) and !fetch.fetches(true, 1, 43 * 100 + 1, 100, big, from1));
}

test "GLM owner fetch: mutants are caught" {
    defer fetch.mutant = .none;
    const chunks = [_]usize{ 300, 512, 900, 64 };
    inline for (.{ .no_broadcast, .pack_short, .room_on_owner, .pages_elsewhere }) |m| {
        fetch.mutant = m;
        for ([_]usize{ 2, 3 }) |world| {
            if (check(world, 0, &chunks, 3)) |_| {
                std.debug.print("mutant {s} survived at {d} ranks\n", .{ @tagName(m), world });
                return error.MutantSurvived;
            } else |_| {}
        }
    }
}
