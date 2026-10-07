//! A slot's prompt state at a chunk end: every KDA state and conv window, every MLA cache's prefix.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Ref = @import("weights.zig").Ref;

/// One kept state: `at` prompt tokens of one stream, in one buffer (an id both Macs of a pair name it by).
pub const Snap = struct { id: u32, at: u32, buf: mtl.Buffer, bytes: usize };

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

/// The bytes a state at `at` tokens takes.
pub fn bytes(c: *const cfg.Config, at: u32) usize {
    const mla = @as(usize, at) * (c.kv_lora + 2 * c.i_dim) * 2 + blocks(c, at) * c.i_dim * 2;
    return c.countKind(.kda) * (stBytes(c) + csBytes(c)) + mlaCount(c) * mla;
}

/// `n` bytes (a multiple of 4) from `src` to `dst` on the GPU.
fn words(x: *const fwd.Ctx, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n: usize) void {
    e.setPipeline(x.k.copy_u32);
    e.setBuffer(src.buf, src.off, 0);
    e.setBuffer(dst.buf, dst.off, 1);
    e.setValue(@as(u32, @intCast(n / 4)), 2);
    e.dispatchThreads(mtl.Size.of(n / 4, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Encode the copies between `s` and the snapshot at `snap` (`into`: the state into the snapshot, else back).
pub fn copy(x: *const fwd.Ctx, e: mtl.ComputeEncoder, s: *st.State, snap: Ref, at: u32, into: bool) void {
    const c = x.c;
    var off: usize = 0;
    for (s.kda[0..c.countKind(.kda)]) |*L| { // the state the next chunk reads: the current slot's, restored into slot 0
        const live = [2]Ref{ L.st[if (into) L.cur else 0], L.cs[if (into) L.cur else 0] };
        for (live, [2]usize{ stBytes(c), csBytes(c) }) |r, n| {
            if (into) words(x, e, r, snap.at(off), n) else words(x, e, snap.at(off), r, n);
            off += n;
        }
    }
    for (s.mla[0..mlaCount(c)]) |*C| {
        const parts = [4]Ref{ C.keys, C.ik, C.ig, C.pool };
        const sizes = [4]usize{ @as(usize, at) * c.kv_lora * 2, @as(usize, at) * c.i_dim * 2, @as(usize, at) * c.i_dim * 2, blocks(c, at) * c.i_dim * 2 };
        for (parts, sizes) |r, n| {
            if (into) words(x, e, r, snap.at(off), n) else words(x, e, snap.at(off), r, n);
            off += n;
        }
    }
    std.debug.assert(off == bytes(c, at));
    if (into) return;
    for (s.kda[0..c.countKind(.kda)]) |*L| L.cur = 0;
    s.pos = at;
    s.mtp_pos = at;
}
