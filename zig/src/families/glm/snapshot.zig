//! Prompt state at a chunk end: KDA states, conv windows, MLA prefixes (resident: in the slot until it writes below).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Ref = @import("weights.zig").Ref;
const walk = @import("../../core/state_walk.zig").Walk(Ref, Ref.at);

/// One kept state: `at` prompt tokens of one stream, in one buffer (an id both Macs of a pair name it by).
pub const Snap = struct {
    id: u32,
    at: u32,
    buf: mtl.Buffer,
    bytes: usize,
    follow: ?mtl.ResidencySet.Follow = null,
    home: ?u32 = null, // resident: the slot whose MLA caches hold this state's prefix
    stale: bool = false, // resident, and its slot has since written a row below `at`
    held: ?[2]u32 = null, // resident in a shared pool after its slot left: the blocks (start, count) holding its prefix
    used: u64 = 0, // the slots' use clock at its last keep or restore (kept.pick)
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

/// A whole state's layout for the disk: KDA states, then each MLA layer's key, indexer key, gate and pooled index rows.
pub fn diskShape(c: *const cfg.Config) struct { kda: usize, mla: usize, key_row: u32, i_row: u32, kpool: u32 } {
    return .{ .kda = c.countKind(.kda) * (stBytes(c) + csBytes(c)), .mla = mlaCount(c), .key_row = c.kv_lora * 2, .i_row = c.i_dim * 2, .kpool = c.kpool };
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

/// Copies between a part and the kept bytes at `snap` (`into`: to them), one dispatch a part: the core walk's hook.
const Words = struct {
    x: *const fwd.Ctx,
    e: mtl.ComputeEncoder,
    snap: Ref,
    into: bool,
    fn run(w: Words, live: Ref, kept: u64, n: usize) anyerror!void {
        const k = w.snap.at(@intCast(kept));
        if (w.into) words(w.x, w.e, live, k, n) else words(w.x, w.e, k, live, n);
    }
};

/// An MLA layer's prefix at `at`: its keys, indexer keys, gates and pooled index rows.
fn mlaParts(c: *const cfg.Config, C: *const st.Mla, at: u32, out: []walk.Part) usize {
    const rows: usize = at;
    out[0..4].* = .{
        .{ .live = C.keys, .len = rows * c.kv_lora * 2 },
        .{ .live = C.ik, .len = rows * c.i_dim * 2 },
        .{ .live = C.ig, .len = rows * c.i_dim * 2 },
        .{ .live = C.pool, .len = blocks(c, at) * c.i_dim * 2 },
    };
    return 4;
}

/// Copies between `s` and the snapshot (`into`: to it); resident: KDA states only, MLA prefixes from `home` if not `s`.
pub fn copy(
    x: *const fwd.Ctx,
    e: mtl.ComputeEncoder,
    s: *st.State,
    snap: Ref,
    at: u32,
    into: bool,
    home: ?*st.State,
) !void {
    const c = x.c;
    var parts: [2 * st.max_kda + 4 * st.max_mla]walk.Part = undefined;
    var n: usize = 0;
    for (s.kda[0..c.countKind(.kda)]) |*L| { // the state the next chunk reads: the current slot's, restored into slot 0
        const i: usize = if (into) L.cur else 0;
        parts[n..][0..2].* = .{ .{ .live = L.st[i], .len = stBytes(c) }, .{ .live = L.cs[i], .len = csBytes(c) } };
        n += 2;
    }
    if (home == null) for (s.mla[0..mlaCount(c)]) |*C| {
        n += mlaParts(c, C, at, parts[n..]);
    };
    try walk.each(parts[0..n], &.{}, .{}, Words{ .x = x, .e = e, .snap = snap, .into = into }, Words.run);
    std.debug.assert(walk.total(parts[0..n]) == bytes(c, at, home != null));
    if (home) |h| if (!into and h != s) for (s.mla[0..mlaCount(c)], h.mla[0..mlaCount(c)]) |*C, *H| {
        var mine: [4]walk.Part = undefined;
        var held: [4]walk.Part = undefined;
        _ = mlaParts(c, C, at, &mine);
        _ = mlaParts(c, H, at, &held);
        for (held, mine) |src, r| words(x, e, src.live, r.live, r.len);
    };
    if (into) return;
    for (s.kda[0..c.countKind(.kda)]) |*L| L.cur = 0;
    s.pos = at;
    s.mtp_pos = at;
}

/// A resident state made whole: KDA states from `old`, then MLA prefixes from `home` (its slot), in `whole`'s layout.
pub fn spill(x: *const fwd.Ctx, e: mtl.ComputeEncoder, old: Ref, whole: Ref, at: u32, home: *st.State) !void {
    const c = x.c;
    const off = bytes(c, at, true);
    words(x, e, old, whole, off);
    var parts: [4 * st.max_mla]walk.Part = undefined;
    var n: usize = 0;
    for (home.mla[0..mlaCount(c)]) |*C| n += mlaParts(c, C, at, parts[n..]);
    try walk.each(parts[0..n], &.{}, .{ .base = off }, Words{ .x = x, .e = e, .snap = whole, .into = true }, Words.run);
    std.debug.assert(off + walk.total(parts[0..n]) == bytes(c, at, false));
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
