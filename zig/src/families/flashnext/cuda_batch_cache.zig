//! Attention's projection and KV/indexer-cache boundary for a prompt whose proposal outputs are unread.
const wt = @import("cuda_prompt_wt.zig");
const eps = wt.eps;

/// Where attention reads positions and writes queries, indexer queries and outputs.
pub const RowsAt = struct { pos: u64, q: u64, iq: u64, ao: u64 };

/// `n` rows of rope, KV/indexer cache writes and pooling, without proposal scores or attention outputs.
pub fn rows(self: anytype, lw: anytype, ai: usize, pa: u64, n: usize, at: RowsAt) !void {
    const s = self.s;
    const h = lw.heads;
    const kc = self.seq.kc[ai].ptr;
    const vc = self.seq.vc[ai].ptr;
    const ikc = self.seq.ikc[ai].ptr;
    const pooled = self.seq.pooled[ai].ptr;
    const pg = self.seq.pg;
    if (self.seq.media) |m| try m.attn(self.stream, pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kc, vc, at.iq, ikc, eps, n, h, pg) else try self.tri.attnPrep(pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kc, vc, at.iq, ikc, s.dummy, s.dummy, eps, n, 0, h, pg);
    if (self.seq.media) |m| try m.pooled(self.stream, ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, eps, n, pg) else try self.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, eps, n, pg);
}

/// Projection, rope and cache writes for one lane-free prompt piece.
pub fn block(self: anytype, lw: anytype, ai: usize) !void {
    @import("std").debug.assert(!self.lanes_attn and self.lanes.len == 0);
    const s = self.s;
    const n = self.rows;
    const pa = s.proj;
    _ = try self.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
    try rows(self, lw, ai, pa, n, .{ .pos = s.pos, .q = s.q, .iq = s.iq, .ao = s.ao });
}
