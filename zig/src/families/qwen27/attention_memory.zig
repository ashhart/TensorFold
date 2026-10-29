//! Allocation geometry shared by prompt admission and the actual attention scratch owner.
const planning = @import("attention_plan.zig");
pub const Layout = struct {
    tiles: u32,
    chunks: u32,
    tail_chunks: u32,
    sizes: [15]usize,
    pub fn init(limits: anytype) Layout {
        const hk: usize = limits.geometry.kv_heads;
        const dim: usize = limits.geometry.dim;
        const tiles = (limits.rows * limits.geometry.group() + 15) / 16 + limits.streams;
        const chunks = @max((limits.keys + 511) / 512, (planning.fine_until + 128 + planning.fine_chunk - 1) / planning.fine_chunk); // fine chunks below fine_until
        const rp: usize = tiles * 16;
        const a: usize = if (limits.shared_prefix) hk * @as(usize, chunks) * rp else 0;
        const tail_chunks = if (limits.shared_prefix) 3 else chunks; // a tail of up to 191 keys over 128-key chunks
        const b: usize = hk * @as(usize, tail_chunks) * @as(usize, limits.rows) * 16;
        const sizes = [_]usize{ tiles * 4, rp * 4, limits.rows * planning.node_words * 4, limits.rows * planning.path_width * 4, limits.rows * 4, hk * rp * dim * 2, hk * @as(usize, limits.rows) * 16 * dim * 2, a * dim * 4, a * 4, a * 4, b * dim * 4, b * 4, b * 4, limits.rows * hk * dim * 2, limits.rows * hk * dim * 2 };
        return .{ .tiles = tiles, .chunks = chunks, .tail_chunks = tail_chunks, .sizes = sizes };
    }
};
