//! TF_GLM_KV_SPLIT's owner-computed attention: a stream's rows are attended, every rank's heads, where its rows are.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const latent = @import("cuda_latent.zig");
const shard = @import("cuda_shard.zig");
const config = @import("config.zig");

/// The most ranks a room holds heads for (cuda_ranks.max_world).
pub const max_world = 16;
/// The most streams one forward's rows hold: a shared window's or a prompt pass's.
const max_segs = @max(@import("cuda_engine.zig").max_shared, @import("cuda_state.zig").prompt_heads);

/// A layer's rows on this rank: every rank holds them (`local`), another rank does (`give`), or this one (`take`).
pub const Role = enum { local, give, take };

pub fn role(owner: ?u32, rank: usize) Role {
    const o = owner orelse return .local;
    return if (o == rank) .take else .give;
}

/// One stream's rows of a layer: rows [row0, row0 + rows) of the window, the rank holding its latent rows, its index.
pub const Seg = struct { row0: usize, rows: usize, owner: ?u32, i: usize = 0 };

/// What an exchange moves: this rank's queries (`qa`) or outputs (`ol`), or rank p's queries (`pq`) or outputs (`po`).
pub const Buf = enum { qa, ol, pq, po };
/// An attention pass over a segment: this rank's heads, rank p's (its queries from the room), or none.
pub const Pass = union(enum) { own, peer: usize, none };
pub const Dir = enum { send, recv };
pub const Edge = enum { begin, end };

/// Whether any segment exchanges rows (else the layer runs as before the split, with no exchange posted).
pub fn exchanges(segs: []const Seg) bool {
    for (segs) |g| if (g.owner != null) return true;
    return false;
}

/// One DSA layer over `segs` on rank `rank` of `world`, through `d` (the device, or a host model in tests).
pub fn run(comptime D: type, d: D, segs: []const Seg, rank: usize, world: usize) !void {
    if (!exchanges(segs)) {
        for (segs) |g| try d.attend(g, .own);
        return;
    }
    // 1. queries, each give segment's rows to its owner and each take segment's from every other rank, in order
    try d.group(.begin);
    for (segs) |g| switch (role(g.owner, rank)) {
        .give => try d.move(.qa, g.owner.?, g, .send),
        .take => for (0..world) |p| if (p != rank) try d.move(.pq, p, g, .recv),
        .local => {},
    };
    try d.group(.end);
    // 2. local: its own heads; take: its own, then each other rank's; give: its writes and token lists alone
    for (segs) |g| switch (role(g.owner, rank)) {
        .local => try d.attend(g, .own),
        .take => {
            try d.attend(g, .own);
            if (!mut(.peer_skipped)) for (0..world) |p| if (p != rank) try d.attend(g, .{ .peer = p });
        },
        .give => try d.attend(g, .none),
    };
    // 3. outputs: each take segment's rows of rank p's heads back to p; each give segment's into its own `ol`
    try d.group(.begin);
    for (segs) |g| switch (role(g.owner, rank)) {
        .take => for (0..world) |p| if (p != rank) try d.move(.po, if (mut(.output_to_owner)) rank else p, g, .send),
        .give => try d.move(if (mut(.output_into_qa)) .qa else .ol, g.owner.?, g, .recv),
        .local => {},
    };
    try d.group(.end);
}

/// Exchange counts since start (rank 0's appear in glm-paged-check's receipt).
pub const Stats = struct { layers: u64 = 0, rows: u64 = 0, bytes: u64 = 0 };

/// The exchange room: each other rank's queries and the outputs computed for it, `rows` rows of its heads a slot.
pub const Room = struct {
    buf: cuda.DeviceBuffer,
    rank: usize,
    world: usize,
    rows: usize,
    lw: usize,
    heads: [max_world]usize, // each rank's DSA heads (TP=3: 22, 21, 21)
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stats: Stats = .{},

    /// Bytes a room of `rows` rows takes on a rank of `world`: two slots of the widest share for each other rank.
    pub fn bytes(c: *const config.Config, world: usize, rows: usize) !usize {
        return 2 * (world - 1) * rows * (try shard.Share.of(c, 0, world)).mla.len() * c.kv_lora * 2;
    }

    pub fn init(d: *const cuda.Driver, c: *const config.Config, rank: usize, world: usize, rows: usize, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm) !Room {
        if (world < 2 or world > max_world) return error.KvSplitNeedsRanks;
        var r: Room = .{ .buf = try cuda.DeviceBuffer.alloc(d, try bytes(c, world, rows)), .rank = rank, .world = world, .rows = rows, .lw = c.kv_lora, .heads = @splat(0), .lib = lib, .comm = comm };
        for (r.heads[0..world], 0..) |*h, p| h.* = (try shard.Share.of(c, p, world)).mla.len();
        return r;
    }

    pub fn deinit(r: *Room) void {
        r.buf.free();
    }

    /// Rank p's slot (every rank but this one has one, in rank order).
    fn slot(r: *const Room, p: usize) usize {
        return if (p < r.rank) p else p - 1;
    }

    /// Bytes one row of rank p's absorbed queries or latent outputs takes: its heads x lw bf16.
    pub fn rowBytes(r: *const Room, p: usize) usize {
        return r.heads[p] * r.lw * 2;
    }

    /// Where rank p's queries (`pq`) or outputs (`po`) of row `row0` are.
    pub fn at(r: *const Room, which: Buf, p: usize, row0: usize) u64 {
        const half = (r.world - 1) * r.rows * r.heads[0] * r.lw * 2;
        const base = r.buf.ptr + @as(u64, if (which == .po) half else 0);
        return base + r.slot(p) * r.rows * r.heads[0] * r.lw * 2 + row0 * r.rowBytes(p);
    }
};

/// The device side of `run`: a forward's buffers and kernels for one DSA layer over a sequence or a window's segments.
const Gpu = struct {
    f: *const fw.Forward,
    room: *Room,
    k: dsa_block.Kernels,
    a: dsa_block.Layer,
    b: *fw.Buffers,
    caches: []const dsa_block.Caches, // each segment's cache of this layer
    pos: []const u64, // each segment's device position
    host: []const usize, // and its host position
    nch: ?usize, // one sequence: the call's chunks (a shared window computes each segment's)
    last: ?usize = null, // a prompt MTP chunk's rows: its last row alone attends (blockLastRow)

    pub fn group(g: Gpu, e: Edge) !void {
        const lib = g.room.lib;
        switch (e) {
            .begin => try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart"),
            .end => try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd"),
        }
    }

    pub fn move(g: Gpu, which: Buf, p: usize, s: Seg, dir: Dir) !void {
        const r = g.room;
        const own = r.rowBytes(r.rank);
        if ((which == .pq or which == .po) and s.row0 + s.rows > r.rows) return error.KvSplitRoom;
        const ptr = switch (which) {
            .qa => g.b.dsa.qa + s.row0 * own,
            .ol => g.b.dsa.ol + s.row0 * own,
            .pq, .po => r.at(which, p, s.row0),
        };
        const n = s.rows * (if (which == .qa or which == .ol) own else r.rowBytes(p));
        const lib = r.lib;
        switch (dir) {
            .send => try lib.check(lib.api.ncclSend(ptr, n, .u8, @intCast(p), r.comm, g.f.stream.handle), "ncclSend"),
            .recv => try lib.check(lib.api.ncclRecv(ptr, n, .u8, @intCast(p), r.comm, g.f.stream.handle), "ncclRecv"),
        }
        if (dir == .send) r.stats.bytes += n;
    }

    pub fn attend(g: Gpu, s: Seg, pass: Pass) !void {
        const c = g.caches[s.i];
        const nch = g.nch orelse @min(latent.chunksFor(g.host[s.i] + s.rows), g.b.nch);
        const v = g.b.dsa;
        if (g.last) |n| return switch (pass) { // the chunk's rows are written; its last row attends
            .own => dsa_block.attendLastRow(g.k, g.f.stream, g.a, c, v, n, g.pos[s.i], g.host[s.i], nch),
            .peer => |p| dsa_block.lastRowPeer(g.k, g.f.stream, g.a, c, v, n, g.pos[s.i], g.host[s.i], nch, g.room.at(.pq, p, s.row0), g.room.at(.po, p, s.row0), g.room.heads[p]),
            .none => {},
        };
        switch (pass) {
            .own => try dsa_block.attend(g.k, g.f.stream, g.a, c, v, s.row0, s.rows, g.pos[s.i], g.host[s.i], nch, g.b.part_rows),
            .none => try dsa_block.attendNone(g.k, g.f.stream, g.a, c, v, s.row0, s.rows, g.pos[s.i], g.host[s.i]),
            // right after its own pass: the segment's token lists are the scratch's first rows
            .peer => |p| try dsa_block.attendPeer(g.k, g.f.stream, g.a, c, v, s.rows, g.pos[s.i], g.host[s.i], nch, g.b.part_rows, g.room.at(.pq, p, s.row0), g.room.at(.po, p, s.row0), g.room.heads[p]),
        }
    }
};

/// One sequence's DSA layer input: its cache, device and host positions, and the rank holding its latent rows.
pub const One = struct { c: dsa_block.Caches, pos: u64, host_pos: usize, owner: ?u32 };

/// The room for a forward over owner-only rows; KvSplitNoRoom when the forward has none, KvSplitLocal in-process.
fn roomOf(f: *const fw.Forward) !*Room {
    const room = f.owners orelse return error.KvSplitNoRoom; // never attend a stream's latent rows where they are not
    if (f.ranks.len != 1) return error.KvSplitLocal; // in-process ranks: one forward attends for one rank at a time
    return room;
}

/// dsa_block.body (no o_proj) on one sequence's rows, owner-computed when one rank holds its latent rows.
pub fn body(f: *const fw.Forward, r: fw.Rank, k: dsa_block.Kernels, a: dsa_block.Layer, x: u64, one: One, rows: usize, nch: usize) !void {
    const b = r.b;
    if (one.owner == null) return dsa_block.body(k, f.stream, a, one.c, b.dsa, x, r.w.config.hidden, rows, one.pos, one.host_pos, nch, b.part_rows);
    const room = try roomOf(f);
    try dsa_block.prepareProfile(k.set, f.stream, a, b.dsa, x, r.w.config.hidden, rows, dsa_block.sparseRows(a, one.host_pos, rows), k.prof);
    const d: Gpu = .{ .f = f, .room = room, .k = k, .a = a, .b = b, .caches = &.{one.c}, .pos = &.{one.pos}, .host = &.{one.host_pos}, .nch = nch };
    const segs = [_]Seg{.{ .row0 = 0, .rows = rows, .owner = one.owner }};
    try run(Gpu, d, &segs, room.rank, room.world);
    note(room, &segs);
    try latent.expandV(k.set, f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
}

/// dsa_block.blockLastRow, owner-computed when one rank holds the latent rows: every row written, the last attended.
pub fn lastRow(f: *const fw.Forward, r: fw.Rank, k: dsa_block.Kernels, a: dsa_block.Layer, x: u64, one: One, rows: usize, nch: usize, out: u64) !void {
    const b = r.b;
    const D = r.w.config.hidden;
    if (one.owner == null) return dsa_block.blockLastRow(k, f.stream, a, one.c, b.dsa, x, D, rows, one.pos, one.host_pos, nch, out);
    if (rows == 0) return error.BadWindow;
    const room = try roomOf(f);
    try dsa_block.prepareProfile(k.set, f.stream, a, b.dsa, x, D, rows, dsa_block.sparseRows(a, one.host_pos, rows), k.prof);
    try dsa_block.writeKeys(k, f.stream, a, one.c, b.dsa, rows, one.pos); // all rows, at pos, before any of them attends
    const d: Gpu = .{ .f = f, .room = room, .k = k, .a = a, .b = b, .caches = &.{one.c}, .pos = &.{one.pos}, .host = &.{one.host_pos}, .nch = nch, .last = rows };
    const segs = [_]Seg{.{ .row0 = rows - 1, .rows = 1, .owner = one.owner }};
    try run(Gpu, d, &segs, room.rank, room.world);
    note(room, &segs);
    const off = (rows - 1) * a.heads * a.lw * 2;
    try latent.expandV(k.set, f.stream, b.dsa.ol + off, a.wv, b.dsa.vn, 1, a.heads, a.dv, a.lw); // its o, into vn's row 0
    try dsa_block.fill(k.set, f.stream, a, b.dsa, 0, 1, out);
}

/// mtp.mtp_forward's DSA block on the head's caches: `n` rows committed alone, a prompt chunk's last row, or a chain.
pub fn head(f: *const fw.Forward, r: fw.Rank, k: dsa_block.Kernels, a: dsa_block.Layer, n: usize, nch: usize, propose: bool, last_row: bool, out: u64) !void {
    const b = r.b;
    const one: One = .{ .c = r.st.mtp_caches, .pos = r.st.mtp_pos_dev, .host_pos = r.st.mtp_len, .owner = r.st.owner };
    if (!propose) return dsa_block.commit(k, f.stream, a, one.c, b.dsa, b.normed, r.w.config.hidden, n, one.pos, one.host_pos);
    if (last_row) return lastRow(f, r, k, a, b.normed, one, n, nch, out);
    try body(f, r, k, a, b.normed, one, n, nch);
    try dsa_block.fill(k.set, f.stream, a, b.dsa, 0, n, out);
}

/// Whether any of a shared window's streams holds its latent rows on one rank.
pub fn owned(r: fw.Rank) bool {
    for (r.segs) |g| if (g.st.owner != null) return true;
    return false;
}

/// cuda_shared.mixer's DSA block (no o_proj) with owner-only segments: each segment's attention where its rows are.
pub fn shared(f: *const fw.Forward, r: fw.Rank, a: dsa_block.Layer, di: usize, rows: usize) !void {
    const room = try roomOf(f);
    const b = r.b;
    if (r.segs.len > max_segs) return error.TooManySegments;
    var caches: [max_segs]dsa_block.Caches = undefined;
    var pos: [max_segs]u64 = undefined;
    var host: [max_segs]usize = undefined;
    var segs: [max_segs]Seg = undefined;
    var sparse_any = false;
    for (r.segs, 0..) |g, i| {
        caches[i] = g.st.caches[di];
        pos[i] = g.st.pos_dev;
        host[i] = g.st.pos;
        segs[i] = .{ .row0 = g.row0, .rows = g.rows, .owner = g.st.owner, .i = i };
        sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.pos, g.rows);
    }
    try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, r.w.config.hidden, rows, sparse_any);
    const n = r.segs.len;
    const d: Gpu = .{ .f = f, .room = room, .k = .{ .set = f.set(), .ops = f.m.ops() }, .a = a, .b = b, .caches = caches[0..n], .pos = pos[0..n], .host = host[0..n], .nch = null };
    try run(Gpu, d, segs[0..n], room.rank, room.world);
    note(room, segs[0..n]);
    try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
}

fn note(room: *Room, segs: []const Seg) void {
    if (!exchanges(segs)) return;
    room.stats.layers += 1;
    for (segs) |g| room.stats.rows += if (g.owner != null) g.rows else 0;
}

// -- mutants (test builds only) -------------------------------------------------------------------------------------

pub const Mutant = enum { none, peer_skipped, output_to_owner, output_into_qa };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return @import("builtin").is_test and mutant == m;
}
