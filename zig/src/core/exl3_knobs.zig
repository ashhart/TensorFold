//! The EXL3 projection's process-wide body selection.
//!
//! The family host resolves these from its own configuration before a model loads; the core
//! projection has no environment reads and no family names. Every field's default is the
//! shipped body, so a host that sets none of them gets the engine's normal path.

pub const Knobs = struct {
    // One-row and prompt body selection.
    table_decode: bool = true,
    wide_decode: bool = true,
    direct_decode: bool = true,
    fma_lut: bool = true,
    lut_decode: bool = true,
    fma_static: bool = true,
    fma_dense: bool = true,
    static_decode: bool = true,
    dense_db: bool = true,
    dense_rpg2: bool = false,
    skip_pad: bool = true,
    wide_words: bool = false,
    funnel_only: bool = true,
    dense_funnel: bool = false,
    dense_funnel8: bool = false,
    coop_prefill: bool = false,
    warp32: bool = true,
    funnel_prefill: bool = true,
    moe_db: bool = false,

    // Split counts and row gates.
    splits_override: ?u32 = null,
    moe_splits: u32 = 8,
    split_wide: u32 = 8,
    nax_min_rows: u32 = 64,
    shared_min_rows: u32 = 32,
    nax_tile_rows: u32 = 128,
    nax_cols: ?u32 = null,
    moe_cols: u32 = 0,
    moe_rows: u32 = 2,
    moe_tile_rows: u32 = 64,

    // Decode-path selection.
    i8_prefill: bool = false,
    i8_debug: bool = false,
    fma_warps: usize = 16,
    fma_pf: usize = 1,
    scalar_experts: bool = false,
    fma_off: bool = false,
    routed_table: bool = true,
    matvec: bool = false,
    decode_compact: bool = false,
    funnel: bool = true,
    row_invariant: bool = true,
    fma_dense_h8: bool = true,
    fma_dense_multi: u32 = 4,
    fma_dense_multi_ceil: bool = true,
    table: bool = false,
    nax_dense8: bool = true,
    funnel_log: bool = false,

    // Measurement passes. They duplicate work and never move a result; the host leaves them off.
    double_rotate: bool = false,
    double_gemm: bool = false,
    double_gemm_n: ?u32 = null,
    double_gemm_indexed: bool = false,
    double_rotate_out: bool = false,
};
