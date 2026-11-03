//! Native server learning admits the dense 27B; the synthetic trainer fixture remains a separate entry point.
const Config = @import("config.zig").Config;
pub fn supported(c: Config) bool {
    if (c.hidden != 5120 or c.intermediate != 17408 or c.layers != 64 or c.vocab != 248320 or c.heads != 24 or c.kv_heads != 4 or c.head_dim != 256 or c.k_heads != 16 or c.v_heads != 48 or c.dk != 128 or c.dv != 128 or c.conv_kernel != 4 or c.rope_dims != 64 or c.eps != 1e-6 or c.rope_theta != 10000000 or c.ranks != 1) return false;
    for (c.kinds[0..c.layers], 0..) |kind, l| if (kind != (if (l % 4 == 3) @import("config.zig").Kind.attention else .linear)) return false;
    return true;
}
