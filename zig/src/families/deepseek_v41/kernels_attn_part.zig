//! The decode attention's partial-tile twins (attn_part_kernel, split_kernel at P + 16) for heads not a multiple of 16.
const cuda = @import("cuda");
const attn = @import("kernels_shapes.zig").attn;
const fitShared = @import("cuda").qlinear.block.experts.fitShared;

/// The 8 attention instances and the [P 1, P 4] split ones, in Functions.attn's order (kw 4 | rows | has_hi).
pub const Set = struct { one: [8]cuda.Function, split: [2][8]cuda.Function };

/// The twins in `m`, or null when its fatbin was built before them.
pub fn load(m: cuda.Module, optin: u32) !?Set {
    var s: Set = undefined;
    inline for (0..8) |i| {
        const kw = if (i >= 4) 4 else 2;
        const name = attn.partSymbol(kw, (i & 2) != 0, (i & 1) != 0);
        s.one[i] = m.function(name) catch return null;
        if (!try fitShared(s.one[i], name, attn.SMEM, optin)) return error.SharedMemoryTooSmall;
        inline for (.{ 1, 4 }, 0..) |p, pi| {
            const split = attn.splitPartSymbol(kw, (i & 2) != 0, (i & 1) != 0, p);
            s.split[pi][i] = m.function(split) catch return null;
            if (!try fitShared(s.split[pi][i], split, attn.splitSmem(p), optin)) return error.SharedMemoryTooSmall;
        }
    }
    return s;
}
