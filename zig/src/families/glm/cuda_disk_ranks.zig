//! A GLM kept state's disk image across ranks: rank 0's engine thread leases, fills, reads and seals each rank's copy.
const std = @import("std");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// A disk step's op (Header.flags); its id is `a`, its image offset `seed`, its bytes `depth`, its rank `top_k`.
pub const Op = enum(u32) { blank = 1, put, take, seal, prefixes };
const kind = @backingInt(follow.Kind.disk); // each op crosses the follow link as a disk step, its bytes as the ids

/// This rank's image shape (every sequence of the engine has the same).
pub fn shape(e: *const Engine) !image.Shape {
    return image.Shape.of(&e.cur.st, &e.c, e.capacity, e.mtp);
}

/// The words of a kept state on this rank: its position, head rows, ring bases and KDA copy bits.
pub fn wordsOf(e: *const Engine, id: u64) !image.Words {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    if (s.cur.len > 64) return error.SnapshotShape;
    var cur: u64 = 0;
    for (s.cur, 0..) |b, i| cur |= @as(u64, b) << @intCast(i);
    return .{ .at = s.pos, .mtp_len = s.mtp_len, .base = s.bases[0], .mtp_base = s.bases[1], .cur = cur };
}

/// Bytes [off, off + host.len) of this rank's image of `id`, out (lease to host) or in; unsealed: before its head only.
pub fn copy(e: *Engine, id: u64, off: u64, host: []u8, out: bool) !void {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const sh = try shape(e);
    const at: u32 = @intCast(s.pos);
    if (s.len == 0 and off + host.len > image.Words.bytes + image.front(sh, at)) return error.DiskOrder;
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    try image.segments(sh, at, s.mtp_len, &segs, e.gpa);
    var w = (try wordsOf(e, id)).encode();
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    try (image.Copy{ .d = e.d, .pool = pool, .lease = &s.lease, .segs = segs.items, .words = &w }).run(off, host, out);
}

/// Rank 0's link at TP, else null (one rank: every op is local).
fn leader(e: *Engine) ?follow.Leader {
    const l = e.link orelse return null;
    return .{ .e = e, .link = @ptrCast(@alignCast(l)) };
}

/// Rank 0: whether the other ranks sleep on their lines (no stream), and putting them back to sleep after disk steps.
pub fn asleep(e: *Engine) bool {
    return if (leader(e)) |x| x.link.asleep else false;
}

pub fn rest(e: *Engine) !void {
    if (leader(e)) |x| try x.idle();
}

/// Rank 0: a new id leased at `at` on every rank (rank 0 first: the ranks' pools match, so theirs cannot fail apart).
pub fn blank(e: *Engine, at: u32) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try e.blankAs(id, at);
    e.next_snap += 1;
    if (leader(e)) |x| try x.disk(.{ .kind = kind, .flags = @backingInt(Op.blank), .a = id, .seed = at }, &.{});
    return id;
}

/// Rank 0: `bytes` into rank `rank`'s image of `id` at `off` (rank 0's own lease, or through the link in pieces).
pub fn put(e: *Engine, rank: u32, id: u64, off: u64, bytes: []const u8) !void {
    if (rank == 0) return copy(e, id, off, @constCast(bytes), false);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4;
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(room, bytes.len - done);
        const words = (n + 3) / 4;
        @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words])[0..n], bytes[done..][0..n]);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.put), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words]);
        done += n;
    }
}

/// Rank 0: rank `rank`'s image bytes of `id` from `off` into `out`.
pub fn take(e: *Engine, rank: u32, id: u64, off: u64, out: []u8) !void {
    if (rank == 0) return copy(e, id, off, out, true);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4;
    var done: usize = 0;
    while (done < out.len) {
        const n = @min(room, out.len - done);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
        try x.link.from(rank, n);
        try x.link.buf.download(64, out[done..][0..n]);
        done += n;
    }
}

/// Rank 0: every rank's blank `id` sealed with `w` (its head's rows, ring bases and KDA copy bits).
pub fn seal(e: *Engine, id: u64, w: image.Words) !void {
    try e.sealState(id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    if (leader(e)) |x| {
        const words = w.encode();
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.seal), .a = id }, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&words)));
    }
}

/// Rank 0: every rank's prefix bytes in rank order (the part of a state each rank holds alone).
pub fn prefixes(e: *Engine, out: []u64) !void {
    const mine = (try shape(e)).prefix;
    const x = leader(e) orelse {
        if (out.len != 1) return error.NoRank;
        out[0] = mine;
        return;
    };
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.prefixes) }, &.{});
    try gather(x.link, e, mine, out);
}

fn gather(l: *follow.Link, e: *const Engine, mine: u64, out: []u64) !void {
    try @import("cuda_ranks.zig").gather(u64, l.d, l.lib, l.comm, l.stream, e.share.rank, mine, out);
}

/// A rank other than 0: rank 0's disk step on this rank's copy of its state.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32) !void {
    if (h.flags < @backingInt(Op.blank) or h.flags > @backingInt(Op.prefixes)) return error.BadStep;
    const op: Op = @fromBackingInt(h.flags);
    const rank: u32 = @intCast(e.share.rank);
    switch (op) {
        .blank => try e.blankAs(h.a, @intCast(h.seed)),
        .seal => {
            if (ids.len != 12) return error.BadStep;
            var x: [6]u64 = undefined;
            @memcpy(std.mem.sliceAsBytes(&x), std.mem.sliceAsBytes(ids));
            const w = try image.Words.decode(x, @intCast((e.snaps.get(h.a) orelse return error.NoKeptState).pos));
            try e.sealState(h.a, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
        },
        .put => if (h.top_k == rank) {
            if (h.depth > ids.len * 4) return error.BadStep;
            try copy(e, h.a, h.seed, @constCast(std.mem.sliceAsBytes(ids))[0..h.depth], false);
        },
        .take => {
            if (h.depth > l.ids.len * 4) return error.BadStep;
            if (h.top_k == rank) {
                const host = std.mem.sliceAsBytes(l.ids)[0..h.depth];
                try copy(e, h.a, h.seed, host, true);
                try l.buf.uploadAsync(64, host, l.stream.handle);
            }
            try l.from(h.top_k, h.depth);
        },
        .prefixes => {
            var out: [64]u64 = undefined;
            if (e.share.world > out.len) return error.BadStep;
            try gather(l, e, (try shape(e)).prefix, out[0..e.share.world]);
        },
    }
}
