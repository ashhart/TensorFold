//! A GLM kept state's disk image across ranks: rank 0's engine thread leases, fills, reads and seals each rank's copy.
const std = @import("std");
const cuda = @import("cuda");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const snapshots = @import("cuda_snapshots.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// Rank 0 at TP: its follow link, set by Link.lead, cleared by its deinit; null: one rank, every op local.
pub var link: ?*follow.Link = null;

/// A disk step's op (Header.flags); its id is `a`, its image offset `seed`, its bytes `depth`, its rank `top_k`.
pub const Op = enum(u32) { blank = 1, rows, take, seal, prefixes };
const kind = @backingInt(follow.Kind.disk); // each op crosses the follow link as a disk step, its bytes as the ids

/// This rank's image shape (every sequence of the engine has the same).
pub fn shape(e: *const Engine) !image.Shape {
    return image.Shape.of(&e.cur.st, &e.c, e.capacity, e.mtp);
}

/// The words of a kept state on this rank: its position, head rows, ring bases, KDA copy bits and layout (unsplit).
pub fn wordsOf(e: *const Engine, id: u64) !image.Words {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    if (s.cur.len > 64) return error.SnapshotShape;
    var cur: u64 = 0;
    for (s.cur, 0..) |b, i| cur |= @as(u64, b) << @intCast(i);
    const form = image.formNow();
    if (form == .live_half and s.cur.len > 0 and snapshots.layout.halfOf(s.cur, e.cur.st.kda_layers, true) == null)
        return error.DiskKdaForm;
    return .{ .form = form, .at = s.pos, .mtp_len = s.mtp_len, .base = s.bases[0], .mtp_base = s.bases[1], .cur = cur, .world = e.share.world };
}

/// Bytes [off, off + host.len) of this rank's image of `id`, out (lease to host) or in; unsealed: before its head only.
pub fn copy(e: *Engine, id: u64, off: u64, host: []u8, out: bool) !void {
    const H = struct {
        host: []u8,
        out: bool,
        fn run(h: @This(), c: image.Copy, at: u64) !void {
            return c.run(at, h.host, h.out);
        }
    };
    return through(e, id, off, host.len, H{ .host = host, .out = out });
}

/// `len` device bytes at `src` into this rank's image of `id` from `off`, queued on `stream` (the caller waits).
pub fn copyDevice(e: *Engine, id: u64, off: u64, src: u64, len: u64, stream: cuda.Stream) !void {
    const D = struct {
        src: u64,
        len: u64,
        stream: cuda.Stream,
        fn run(x: @This(), c: image.Copy, at: u64) !void {
            return c.fromDevice(at, x.src, x.len, x.stream.handle);
        }
    };
    return through(e, id, off, len, D{ .src = src, .len = len, .stream = stream });
}

/// `how.run` on the image of `id` for `len` bytes from `off`, refused past an unsealed state's words and front.
fn through(e: *Engine, id: u64, off: u64, len: u64, how: anytype) !void {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const sh = try shape(e);
    const at: u32 = @intCast(s.pos);
    if (s.len == 0 and off + len > image.Words.bytes + image.front(sh, at)) return error.DiskOrder;
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    try image.segments(sh, at, s.mtp_len, &segs, e.gpa);
    var w = (try wordsOf(e, id)).encode();
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    try how.run(image.Copy{ .d = e.d, .pool = pool, .lease = &s.lease, .segs = segs.items, .words = &w }, off);
}

/// Rank 0's link at TP, else null (one rank: every op is local).
fn leader(e: *Engine) ?follow.Leader {
    return .{ .e = e, .link = link orelse return null };
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
    try snapshots.blankAs(e, id, at);
    e.next_snap += 1;
    // a failed step: rank 0's lease goes back, and its id is dropped at the next step (another rank may hold it)
    errdefer snapshots.drop(e, id);
    if (leader(e)) |x| try x.disk(.{ .kind = kind, .flags = @backingInt(Op.blank), .a = id, .seed = at }, &.{});
    return id;
}

/// A piece's rank in a `rows` step for the bytes every rank holds: each rank's image at `off` past its own prefix.
pub const every: u32 = std.math.maxInt(u32);

/// Bytes for rank `to`'s image at `off`, or for every rank's (`every`: `off` less rank 0's prefix, each adds its own).
pub const Piece = struct { to: u32, off: u64, bytes: []const u8 };

/// Rank 0 at TP: `pieces` into every rank's image of `id`, each step a table (4 words a piece) and one broadcast.
pub fn rows(e: *Engine, id: u64, pieces: []const Piece) !void {
    const x = leader(e) orelse return error.NoRank;
    const stage = std.mem.sliceAsBytes(x.link.ids); // the bytes on the host, then in the link's buffer after its header
    var table: std.ArrayList(u32) = .empty;
    defer table.deinit(e.gpa);
    var i: usize = 0;
    var done: usize = 0; // bytes of pieces[i] in earlier steps (a piece longer than the link goes in parts)
    while (i < pieces.len) {
        table.clearRetainingCapacity();
        var n: usize = 0;
        while (i < pieces.len and n < stage.len and table.items.len + 4 <= x.link.ids.len) {
            const p = pieces[i];
            const k = @min(p.bytes.len - done, stage.len - n);
            const off = p.off + done;
            try table.appendSlice(e.gpa, &.{ @truncate(off), @intCast(off >> 32), @intCast(k), p.to });
            @memcpy(stage[n..][0..k], p.bytes[done..][0..k]);
            n += k;
            done += k;
            if (done == p.bytes.len) {
                i += 1;
                done = 0;
            }
        }
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.rows), .a = id, .depth = @intCast(n) }, table.items);
        try x.link.stream.synchronize(); // the table's broadcasts read the buffer the bytes go to
        try x.link.buf.upload(64, stage[0..n]);
        try x.link.from(0, n);
        try place(e, id, table.items, x.link.buf.ptr + 64, x.link.stream);
    }
}

/// A `rows` step's pieces for this rank, copied from `src` (the link's buffer) and waited for: the next step reuses it.
fn place(e: *Engine, id: u64, table: []const u32, src: u64, stream: cuda.Stream) !void {
    const me: u32 = @intCast(e.share.rank);
    const own = (try shape(e)).prefix;
    var at: u64 = 0;
    var i: usize = 0;
    while (i < table.len) : (i += 4) {
        const off = @as(u64, table[i]) | @as(u64, table[i + 1]) << 32;
        const len: u64 = table[i + 2];
        if (table[i + 3] == every) try copyDevice(e, id, off + own, src + at, len, stream) else if (table[i + 3] == me) try copyDevice(e, id, off, src + at, len, stream);
        at += len;
    }
    try stream.synchronize();
}

/// A `rows` step's table checked against its bytes and the link before a follower joins the broadcast.
fn fits(ids: []const u32, depth: u32, room: usize) !void {
    if (ids.len % 4 != 0 or depth > room) return error.BadStep;
    var n: u64 = 0;
    var i: usize = 0;
    while (i < ids.len) : (i += 4) n += ids[i + 2];
    if (n != depth) return error.BadStep;
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

/// Rank 0: each other rank's bytes of `id` past its prefix (`each`) equal to `shared`, but for `loose` (zeros there).
pub fn same(e: *Engine, id: u64, each: []const u64, shared: []const u8, loose: []const image.Span, theirs: []u8) !void {
    for (each[1..], 1..) |q, r| {
        try take(e, @intCast(r), id, image.Words.bytes + q, theirs);
        image.clear(loose, 0, theirs);
        if (!std.mem.eql(u8, shared, theirs)) return error.RanksDiffer;
    }
}

/// Words read back for a state at `at`: refused by name unless this server's world and layout (unsplit) wrote them.
pub fn decode(e: *const Engine, x: [image.Words.count]u64, at: u32) !image.Words {
    const w = try image.Words.decodeForm(x, at, @intCast(e.share.world), .unsplit, image.formNow());
    const n = e.cur.st.kda_layers;
    if (n > 64 or (n < 64 and w.cur >> @intCast(n) != 0)) return error.DiskWords;
    const mask = if (n == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(n)) - 1;
    if (w.form == .live_half and w.cur != 0 and w.cur != mask) return error.DiskKdaForm;
    return w;
}

/// Rank 0: every rank's blank `id` sealed with `w` (its head's rows, ring bases and KDA copy bits).
pub fn seal(e: *Engine, id: u64, w: image.Words) !void {
    try snapshots.seal(e, id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    if (leader(e)) |x| {
        const words = w.encode();
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.seal), .a = id }, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&words)));
    }
}

/// Rank 0: every rank's prefix bytes in rank order (what it holds alone), and where its window rows begin in it.
pub fn prefixes(e: *Engine, out: []u64, windows: []u64) !void {
    const sh = try shape(e);
    if (windows.len != out.len) return error.NoRank;
    const x = leader(e) orelse {
        if (out.len != 1) return error.NoRank;
        out[0] = sh.prefix;
        windows[0] = sh.window;
        return;
    };
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.prefixes) }, &.{});
    try gather(x.link, e, sh.prefix, out);
    try gather(x.link, e, sh.window, windows);
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
        .blank => try snapshots.blankAs(e, h.a, @intCast(h.seed)),
        .seal => {
            if (ids.len != image.Words.count * 2) return error.BadStep;
            var x: [image.Words.count]u64 = undefined;
            @memcpy(std.mem.sliceAsBytes(&x), std.mem.sliceAsBytes(ids));
            const w = try decode(e, x, @intCast((e.snaps.get(h.a) orelse return error.NoKeptState).pos));
            try snapshots.seal(e, h.a, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
        },
        .rows => {
            try fits(ids, h.depth, l.buf.len - 64);
            try l.from(0, h.depth);
            try place(e, h.a, ids, l.buf.ptr + 64, l.stream);
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
            const sh = try shape(e);
            try gather(l, e, sh.prefix, out[0..e.share.world]);
            try gather(l, e, sh.window, out[0..e.share.world]);
        },
    }
}
