//! A slot's prompt state at a chunk end: KDA states, conv windows, MLA prefixes (resident: left in the slot, valid until it writes below).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Ref = @import("weights.zig").Ref;

/// One kept state: `at` prompt tokens of one stream, in one buffer (an id both Macs of a pair name it by).
pub const Snap = struct {
    id: u32,
    at: u32,
    buf: mtl.Buffer,
    bytes: usize,
    home: ?u32 = null, // resident: the slot whose MLA caches hold this state's prefix
    stale: bool = false, // resident, and its slot has since written a row below `at`
    held: ?[2]u32 = null, // resident in a shared pool after its slot left: the blocks (start, count) holding its prefix
};

fn stBytes(c: *const cfg.Config) usize {
    return @as(usize, c.kda_heads) * c.kda_dim * c.kda_dim * 4;
}

fn csBytes(c: *const cfg.Config) usize {
    return @as(usize, c.conv - 1) * 3 * c.kdaWidth() * 2;
}

fn mlaCount(c: *const cfg.Config) usize {
    return c.countKind(.mla) + @as(usize, if (c.mtp > 0) 1 else 0);
}

/// The pooled index blocks `at` tokens hold (a partial last block is copied, never read until whole).
fn blocks(c: *const cfg.Config, at: u32) usize {
    return at / c.kpool + 1;
}

/// The bytes a state at `at` tokens takes (resident: the KDA states alone).
pub fn bytes(c: *const cfg.Config, at: u32, resident: bool) usize {
    const mla = @as(usize, at) * (c.kv_lora + 2 * c.i_dim) * 2 + blocks(c, at) * c.i_dim * 2;
    return c.countKind(.kda) * (stBytes(c) + csBytes(c)) + if (resident) 0 else mlaCount(c) * mla;
}

/// `n` bytes (a multiple of 4) from `src` to `dst` on the GPU.
fn words(x: *const fwd.Ctx, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n: usize) void {
    e.setPipeline(x.k.copy_u32);
    e.setBuffer(src.buf, src.off, 0);
    e.setBuffer(dst.buf, dst.off, 1);
    e.setValue(@as(u32, @intCast(n / 4)), 2);
    e.dispatchThreads(mtl.Size.of(n / 4, 1, 1), mtl.Size.of(256, 1, 1));
}

/// The copies between `s` and the snapshot (`into`: to it); a resident state copies KDA states alone, MLA prefixes from `home` when not `s`.
pub fn copy(x: *const fwd.Ctx, e: mtl.ComputeEncoder, s: *st.State, snap: Ref, at: u32, into: bool, home: ?*st.State) void {
    const c = x.c;
    var off: usize = 0;
    for (s.kda[0..c.countKind(.kda)]) |*L| { // the state the next chunk reads: the current slot's, restored into slot 0
        const live = [2]Ref{ L.st[if (into) L.cur else 0], L.cs[if (into) L.cur else 0] };
        for (live, [2]usize{ stBytes(c), csBytes(c) }) |r, n| {
            if (into) words(x, e, r, snap.at(off), n) else words(x, e, snap.at(off), r, n);
            off += n;
        }
    }
    for (s.mla[0..mlaCount(c)], 0..) |*C, mi| {
        const parts = [4]Ref{ C.keys, C.ik, C.ig, C.pool };
        const sizes = [4]usize{ @as(usize, at) * c.kv_lora * 2, @as(usize, at) * c.i_dim * 2, @as(usize, at) * c.i_dim * 2, blocks(c, at) * c.i_dim * 2 };
        if (home) |h| {
            if (into or h == s) continue;
            const H = &h.mla[mi];
            for ([4]Ref{ H.keys, H.ik, H.ig, H.pool }, parts, sizes) |src, r, n| words(x, e, src, r, n);
            continue;
        }
        for (parts, sizes) |r, n| {
            if (into) words(x, e, r, snap.at(off), n) else words(x, e, snap.at(off), r, n);
            off += n;
        }
    }
    std.debug.assert(off == bytes(c, at, home != null));
    if (into) return;
    for (s.kda[0..c.countKind(.kda)]) |*L| L.cur = 0;
    s.pos = at;
    s.mtp_pos = at;
}

/// A learned state's file for one rank of the pair: <dir>/<key>.r<rank>.bin.
pub fn path(buf: []u8, dir: []const u8, key: u64, rank: u32) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.r{d}.bin", .{ dir, key, rank }, 0);
}
pub fn writeFile(snap: *const Snap, file: [:0]const u8) !void {
    try @import("snapshot_file.zig").write(file, snap.at, snap.buf.contents()[0..snap.bytes]);
}
pub fn readFile(snap: *Snap, file: [:0]const u8) !void {
    try @import("snapshot_file.zig").read(file, snap.at, snap.buf.contents()[0..snap.bytes]);
}
test {
    _ = @import("snapshot_file.zig");
}
