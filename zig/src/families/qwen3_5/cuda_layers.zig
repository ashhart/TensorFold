//! The Qwen3.5 CUDA engine's layers (their projections on the tiles of their widths) and its kernel tables.
const cuda = @import("cuda");

const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;

/// A projection on the tiles of the width it is served at: Affine4's for 4-bit codes, AffineN's for the rest.
pub const Proj = union(enum) {
    a4: Affine4,
    an: AffineN,

    pub fn outputs(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.n,
        };
    }

    pub fn inputs(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.k,
        };
    }
};

pub const Delta = struct { qkv: Proj, z: Proj, a: Proj, b: Proj, out: Proj, conv: u64, norm: u64, a_log: u64, dt_bias: u64 };
pub const Attention = struct { q: Proj, k: Proj, v: Proj, o: Proj, q_norm: u64, k_norm: u64 };
/// A dense SwiGLU MLP, or routed experts (the shared one last in their tables) behind a router of bf16 rows.
pub const Mlp = union(enum) { dense: struct { gate: Proj, up: Proj, down: Proj }, moe: struct { router: u64, experts: cuda.experts.Layer } };
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    mlp: Mlp,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

/// qwen35.cu's kernels, each tf_q35_ and its name.
pub const Fns = struct {
    embed: cuda.Function,
    embed8: cuda.Function,
    norm: cuda.Function,
    sums: cuda.Function,
    gdn_pre: cuda.Function,
    conv_commit: cuda.Function,
    gdn_chain: cuda.Function,
    gdn_post: cuda.Function,
    head_norm: cuda.Function,
    rope: cuda.Function,
    attn_partial: cuda.Function,
    attn_merge: cuda.Function,
    attn_gate: cuda.Function,
    mlp_act: cuda.Function,
    argmax: cuda.Function,
    tap: cuda.Function,
    splice: cuda.Function,
};

/// qwen35_keep.cu's kernels (Cache.keep).
pub const Keeps = struct { apply: cuda.Function, conv: cuda.Function, compact: cuda.Function };

/// qwen35_kv8.cu's kernels: the cache writes, attention reads and keep's moves over e4m3 rows (--kv-cache fp8).
pub const Kv8 = struct { rope: cuda.Function, attn_partial: cuda.Function, compact: cuda.Function };
