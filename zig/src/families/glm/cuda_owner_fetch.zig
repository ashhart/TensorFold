//! TF_GLM_KV_SPLIT's prompt history fetch: a chunk's earlier latent rows broadcast from their owner, each rank's heads.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const latent = @import("cuda_latent.zig");
const owner_attn = @import("cuda_owner_attn.zig");

/// Where a rank attends a fetching chunk: its pages (the owner) or the room's rows in position order (the others).
pub const From = enum { pages, room };

pub fn from(owner: u32, rank: usize) From {
    return if (owner == rank) .pages else .room;
}

/// Latent rows (`lw` bf16 each) a room of `bytes` holds in position order.
pub fn roomRows(bytes: usize, lw: usize) usize {
    return bytes / (lw * 2);
}

/// Whether a chunk fetches: a prompt chunk on owner-only rows whose rows to its end fit the room (`cap` rows).
pub fn fetches(prefill: bool, owner: ?u32, host_pos: usize, rows: usize, cap: usize) bool {
    return prefill and owner != null and host_pos + rows <= cap;
}

/// One DSA layer's chunk on rank `rank`: the owner packs rows [0, host_pos) and broadcasts them, each rank its heads.
pub fn run(comptime D: type, d: D, owner: u32, rank: usize, host_pos: usize) !void {
    var at = from(owner, rank);
    if (mut(.room_on_owner) and at == .pages) at = .room;
    if (mut(.pages_elsewhere) and at == .room) at = .pages;
    if (host_pos > 0) { // a stream's first chunk has no history: every rank attends its own rows alone
        if (owner == rank) try d.pack(if (mut(.pack_short)) host_pos - 1 else host_pos);
        if (!mut(.no_broadcast)) try d.broadcast(owner, host_pos);
    }
    try d.attend(at);
}

/// The device side of `run`: one sequence's chunk, its rows' queries ready (dsa_block.prepare).
const Gpu = struct {
    f: *const fw.Forward,
    room: *owner_attn.Room,
    k: dsa_block.Kernels,
    a: dsa_block.Layer,
    b: *fw.Buffers,
    c: dsa_block.Caches,
    pos: u64,
    host_pos: usize,
    rows: usize,
    nch: usize,

    pub fn pack(g: Gpu, n: usize) !void {
        const rb = g.a.lw * 2; // tf_glm_kvh_rows: rows 0 .. *pos (at most n) through the table, a warp a row
        var args: cuda.Args = .{};
        inline for (.{ g.c.lc, g.c.pt, @as(i64, @intCast(g.c.ps / rb)), g.pos, @as(c_int, 0), g.room.buf.ptr, @as(c_int, @intCast(rb / 16)), @as(c_int, @intCast(n)) }) |v| args.add(v);
        try cuda.launch.launch(try g.k.ops.function("tf_glm_kvh_rows"), .{ .grid = .{ .x = @intCast(cdiv(n, 8)) }, .block = .{ .x = 256 } }, g.f.stream, &args);
    }

    pub fn broadcast(g: Gpu, owner: u32, n: usize) !void {
        const lib = g.room.lib;
        const bytes = n * g.a.lw * 2;
        try lib.check(lib.api.ncclBroadcast(g.room.buf.ptr, g.room.buf.ptr, bytes, .u8, @intCast(owner), g.room.comm, g.f.stream.handle), "ncclBroadcast");
        g.room.stats.fetched += n;
        g.room.stats.fetch_bytes += bytes;
    }

    pub fn attend(g: Gpu, at: From) !void {
        var c = g.c;
        if (at == .room) { // the history as fetched, this chunk's rows written after it: in order, no table
            c.lc = g.room.buf.ptr;
            c.pt = 0;
        }
        try dsa_block.attend(g.k, g.f.stream, g.a, c, g.b.dsa, 0, g.rows, g.pos, g.host_pos, g.nch, g.b.part_rows);
    }
};

/// owner_attn.body's prompt chunk when `fetches`: every rank attends its own heads, the history fetched from the owner.
pub fn body(f: *const fw.Forward, r: fw.Rank, room: *owner_attn.Room, k: dsa_block.Kernels, a: dsa_block.Layer, x: u64, one: owner_attn.One, rows: usize, nch: usize) !void {
    const b = r.b;
    if (one.c.ls != 0) return error.Fp8OnPages; // the room holds bf16 rows, as pages do
    try dsa_block.prepareProfile(k.set, f.stream, a, b.dsa, x, r.w.config.hidden, rows, dsa_block.sparseRows(a, one.host_pos, rows), k.prof);
    const d: Gpu = .{ .f = f, .room = room, .k = k, .a = a, .b = b, .c = one.c, .pos = one.pos, .host_pos = one.host_pos, .rows = rows, .nch = nch };
    try run(Gpu, d, one.owner.?, room.rank, one.host_pos);
    room.stats.fetches += 1;
    try latent.expandV(k.set, f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

// -- mutants (test builds only) -------------------------------------------------------------------------------------

pub const Mutant = enum { none, no_broadcast, pack_short, room_on_owner, pages_elsewhere };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return @import("builtin").is_test and mutant == m;
}
