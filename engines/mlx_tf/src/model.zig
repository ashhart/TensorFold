//! Clean-room Qwen3.5 / Qwen3.8 Hybrid GDN + GQA Metal forward engine.
//!
//! Zero Apple MLX. Zero TensorFold code. Supports:
//!   - Fast chunked prefill (1..128 rows)
//!   - Dedicated coalesced 32-col 128-bit qmv_1 for single-token decode (M == 1)
//!   - M5 hardware tensor-unit qk_{16,32,48} for multi-row speculative chain & tree verification (M = 2..128)
//!   - Fast speculative state commit (commitKeep) for both GDN and KV cache
//!   - Optional 5-layer hidden tap capture for DFlash2 drafting
const std = @import("std");
const shaders = @import("shaders");
const mtl = @import("metal4.zig");
const cfg_mod = @import("config.zig");
const w_mod = @import("weights.zig");
const samp_mod = @import("sampler.zig");

const ModelConfig = cfg_mod.ModelConfig;
const ModelWeights = w_mod.ModelWeights;
const QkMatrix = w_mod.QkMatrix;
const AffineMatrix = w_mod.AffineMatrix;
const Bf16Matrix = w_mod.Bf16Matrix;
const Tfq3Matrix = w_mod.Tfq3Matrix;
const SubLinear = w_mod.SubLinear;
const LinearWeight = w_mod.LinearWeight;
const Buffer = mtl.Buffer;
const BufferRef = mtl.BufferRef;
const Pipeline = mtl.Pipeline;
const Size = mtl.Size;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
pub fn nowMs() f64 {
    return @as(f64, @floatFromInt(clock_gettime_nsec_np(8))) / 1e6;
}

pub const tap_layers = [5]usize{ 5, 19, 33, 47, 61 };

const QkParams = extern struct {
    M: i32,
    K: i32,
    N: i32,
    G: i32,
    first_row: i32,
    sums_stride: i32,
    tile_first: u32,
};

const EmbedParams = extern struct {
    rows: u32,
    hidden: u32,
};

const NormParams = extern struct {
    rows: u32,
    hidden: u32,
    sums_stride: u32,
    eps: f32,
};

const ScaleParams = extern struct {
    count: u32,
    scale: f32,
};

const PostNormResParams = extern struct {
    rows: u32,
    hidden: u32,
    sums_stride: u32,
    has_scalar: u32,
    eps: f32,
};

const SoftcapParams = extern struct {
    count: u32,
    cap: f32,
};

const GdnNormParams = extern struct {
    rows: u32,
    nv: u32,
    dv: u32,
    in_proj_stride: u32,
    z_offset: u32,
    sums_stride: u32,
    eps: f32,
};

const GdnConvParams = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    qkv_dim: u32,
    in_proj_stride: u32,
    b_offset: u32,
    a_offset: u32,
    commit_single: u32,
    save_qkv: u32,
};

const GdnFusedStepParams = extern struct {
    nk: u32,
    nv: u32,
    qkv_dim: u32,
    in_proj_stride: u32,
    z_offset: u32,
    b_offset: u32,
    a_offset: u32,
    sums_stride: u32,
    write_sums: u32,
    eps: f32,
};

const GdnHistCommitParams = extern struct {
    qkv_dim: u32,
    in_proj_stride: u32,
    src0: u32,
    src1: u32,
    src2: u32,
};

const GdnScanParams = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    commit_state: u32,
};

const AttnPrepParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    eps: f32,
};

const Std128PrepParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    has_bias: u32,
    use_rope: u32,
    has_qk_norm: u32 = 0,
    eps: f32 = 1e-6,
};

const Std128Decode1Params = extern struct {
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    has_bias: u32,
    use_rope: u32,
    pos: u32,
    scale: f32,
    has_qk_norm: u32 = 0,
    sliding_window: u32 = 0,
    eps: f32 = 1e-6,
};

const GenPrepParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_dims: u32,
    qkv_stride: u32,
    has_bias: u32,
    has_qk_norm: u32,
    use_rope: u32,
    has_v_norm: u32,
    k_eq_v: u32,
    rope_interleaved: u32,
    eps: f32,
};

const GenFlashParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sums_stride: u32,
    base_pos: u32,
    is_tree: u32,
    sliding_window: u32,
    scale: f32,
    softcap: f32,
};

const MlaNormParams = extern struct {
    rows: u32,
    dim: u32,
    sums_stride: u32,
    has_bias: u32,
    eps: f32,
};

const MlaQPrepParams = extern struct {
    rows: u32,
    num_heads: u32,
    qk_nope_dim: u32,
    qk_rope_dim: u32,
    rope_interleaved: u32,
    rope_mscale: f32,
};

const MlaKvAParams = extern struct {
    rows: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
    sums_stride: u32,
    has_bias: u32,
    rope_interleaved: u32,
    eps: f32,
    rope_mscale: f32,
};

const MlaScatterParams = extern struct {
    rows: u32,
    num_heads: u32,
    qk_nope_dim: u32,
    qk_rope_dim: u32,
    v_dim: u32,
    kv_row_stride: u32,
};

const MlaFlashParams = extern struct {
    rows: u32,
    num_heads: u32,
    qk_dim: u32,
    v_dim: u32,
    kv_row_stride: u32,
    sums_stride: u32,
    base_pos: u32,
    is_tree: u32,
    scale: f32,
};

const LfmConvParams = extern struct {
    rows: u32,
    hidden: u32,
    l_cache: u32,
    sums_stride: u32,
    has_in_bias: u32,
    has_conv_bias: u32,
    commit_hist: u32,
};

const AddBiasParams = extern struct {
    rows: u32,
    cols: u32,
};

const SpliceParams = extern struct {
    rows: u32,
    hidden: u32,
};

const Qwen35Decode1Params = extern struct {
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    pos: u32,
    eps: f32,
    scale: f32,
};

const MambaConvParams = extern struct {
    rows: u32,
    inter_dim: u32,
    conv_dim: u32,
    in_proj_stride: u32,
    commit_hist: u32,
};

const MambaSsmParams = extern struct {
    rows: u32,
    num_heads: u32,
    head_dim: u32,
    n_groups: u32,
    d_state: u32,
    conv_dim: u32,
    in_proj_stride: u32,
    dt_offset: u32,
    commit_state: u32,
    dt_min: f32,
    dt_max: f32,
};

const MambaNormParams = extern struct {
    rows: u32,
    inter_dim: u32,
    n_groups: u32,
    group_size: u32,
    in_proj_stride: u32,
    sums_stride: u32,
    eps: f32,
};

const GateParams = extern struct {
    rows: u32,
    hidden: u32,
    n_experts: u32,
};

const TopkParams = extern struct {
    rows: u32,
    n_experts: u32,
    top_k: u32,
    norm_topk_prob: u32,
    routed_scaling_factor: f32,
    use_softmax: u32 = 0,
    has_e_bias: u32 = 1,
    n_group: u32 = 0,
    topk_group: u32 = 0,
};

const MoeSwigluParams = extern struct {
    rows: u32,
    top_k: u32,
    K: u32,
    N: u32,
    bits: u32,
    gs: u32,
    sb_dtype: u32,
    x_per_expert: u32,
    code_stride: u64,
    sb_stride: u64,
};

const MoeSwigluCombineParams = extern struct {
    rows: u32,
    hidden: u32,
    top_k: u32,
    has_shared: u32,
    shared_gated: u32,
    add_res: u32,
};

const Relu2Params = extern struct {
    rows: u32,
    width: u32,
    sums_stride: u32,
};

const SharedRelu2Params = extern struct {
    K: u32,
    N: u32,
    G: u32,
};

const MoeExpertParams = extern struct {
    rows: u32,
    top_k: u32,
    K: u32,
    N: u32,
    G: u32,
    apply_relu2: u32,
    x_per_expert: u32,
    expert_stride: u64,
};

const MoeCombineParams = extern struct {
    rows: u32,
    hidden: u32,
    top_k: u32,
};

const MoeBucketParams = extern struct {
    rows: u32,
    top_k: u32,
    n_experts: u32,
};

const MoeBatchedParams = extern struct {
    top_k: u32,
    n_experts: u32,
    K: u32,
    N: u32,
    G: u32,
    apply_relu2: u32,
    x_per_expert: u32,
    expert_stride: u64,
};

const MoeGatherSumsParams = extern struct {
    total_slots: u32,
    top_k: u32,
    K: u32,
    sums_stride: u32,
    identity: u32,
};

const MoeQkParams = extern struct {
    total_slots: u32,
    n_experts: u32,
    K: i32,
    N: i32,
    G: i32,
    sums_stride: i32,
    apply_relu2: u32,
    scatter_out: u32,
    expert_stride: u64,
};

const AttnFlashParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    sums_stride: u32,
    scale: f32,
    splits: u32,
    base_pos: u32,
    is_tree: u32,
    sliding_window: u32 = 0,
    softcap: f32 = 0.0,
};

const KvCompactParams = extern struct {
    base_pos: u32,
    keep_len: u32,
    row_u4: u32,
};

const MlpActParams = extern struct {
    rows: u32,
    inter: u32,
    sums_stride: u32,
};

const ArgmaxParams = extern struct {
    vocab: u32,
    chunks: u32,
    rows: u32,
};

const CopyParams = extern struct {
    count: u32,
};

const AffineParams = extern struct {
    rows: u32,
    N: u32,
    K: u32,
    ldx: u32,
    ldo: u32,
    bits: u32,
    gs: u32,
    sb_dtype: u32,
    add_res: u32,
};

const Tfq3Dims = extern struct {
    rows: u32,
    n: u32,
    k: u32,
    ldx: u32,
    ldo: u32,
    sr_split: u32,
    ksplit: u32,
    add_res: u32,
};

const Tfq3Pipes = struct {
    lib: mtl.Library,
    fwht_x: Pipeline,
    fwht_y: Pipeline,
    dec_fused1: [3]Pipeline,
    dec_fused2: [3]Pipeline,
    dec_fused4: [3]Pipeline,
    dec_2dsk8_1: [3]Pipeline,
    dec_2dsk8_2: [3]Pipeline,
    dec_2dsk8_4: [3]Pipeline,
    prefill: [3]Pipeline,

    fn compile(ctx: *mtl.Context) !Tfq3Pipes {
        const lib = try ctx.compileSource(shaders.tfq3, .fast);
        return .{
            .lib = lib,
            .fwht_x = try ctx.makePipeline(lib, "tfq3_fwht_x"),
            .fwht_y = try ctx.makePipeline(lib, "tfq3_fwht_y"),
            .dec_fused1 = .{
                try ctx.makePipeline(lib, "tfq3_dec_fused1_3b"),
                try ctx.makePipeline(lib, "tfq3_dec_fused1_4b"),
                try ctx.makePipeline(lib, "tfq3_dec_fused1_5b"),
            },
            .dec_fused2 = .{
                try ctx.makePipeline(lib, "tfq3_dec_fused2_3b"),
                try ctx.makePipeline(lib, "tfq3_dec_fused2_4b"),
                try ctx.makePipeline(lib, "tfq3_dec_fused2_5b"),
            },
            .dec_fused4 = .{
                try ctx.makePipeline(lib, "tfq3_dec_fused4_3b"),
                try ctx.makePipeline(lib, "tfq3_dec_fused4_4b"),
                try ctx.makePipeline(lib, "tfq3_dec_fused4_5b"),
            },
            .dec_2dsk8_1 = .{
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_1_3b"),
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_1_4b"),
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_1_5b"),
            },
            .dec_2dsk8_2 = .{
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_2_3b"),
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_2_4b"),
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_2_5b"),
            },
            .dec_2dsk8_4 = .{
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_4_3b"),
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_4_4b"),
                try ctx.makePipeline(lib, "tfq3_dec_2dsk8_4_5b"),
            },
            .prefill = .{
                try ctx.makePipeline(lib, "tfq3_prefill_3b"),
                try ctx.makePipeline(lib, "tfq3_prefill_4b"),
                try ctx.makePipeline(lib, "tfq3_prefill_5b"),
            },
        };
    }

    fn deinit(self: *Tfq3Pipes) void {
        self.fwht_x.deinit();
        self.fwht_y.deinit();
        for (self.dec_fused1) |p| p.deinit();
        for (self.dec_fused2) |p| p.deinit();
        for (self.dec_fused4) |p| p.deinit();
        for (self.dec_2dsk8_1) |p| p.deinit();
        for (self.dec_2dsk8_2) |p| p.deinit();
        for (self.dec_2dsk8_4) |p| p.deinit();
        for (self.prefill) |p| p.deinit();
        self.lib.deinit();
    }
};

pub const Pipelines = struct {
    libs: [6]mtl.Library, // qk_universal, norm_ops, attention, mlp_ops, sample, affine_any
    gdn_lib: ?mtl.Library = null,
    mamba_lib: ?mtl.Library = null,
    moe_lib: ?mtl.Library = null,
    qk_m5_lib: ?mtl.Library = null,
    tfq3: ?Tfq3Pipes = null,
    has_tensor_ops: bool = false,
    qk_16: ?Pipeline = null,
    qk_32: ?Pipeline = null,
    qk_48: ?Pipeline = null,
    qk_64: ?Pipeline = null,
    moe_expert_qk16_batched: ?Pipeline = null,
    qmm_simd_8: Pipeline,
    qmm_simd_16: Pipeline,
    qmv_1: Pipeline,
    qmv_1_8col: Pipeline,
    qmv_1_sk2: Pipeline,
    qmv_1_narrow: Pipeline,
    qmv_1_silu_mul: Pipeline,
    qmv_1_silu_mul_sk4: Pipeline,
    qmv_1_argmax: Pipeline,
    qmv_affine_any: Pipeline,
    qmm_affine_any: Pipeline,
    bf16_linear_mv1: Pipeline,
    bf16_linear_mm: Pipeline,
    embed_lookup_any: Pipeline,
    embed_lookup: Pipeline,
    rms_norm_sums: Pipeline,
    add_rms_norm_sums: Pipeline,
    rms_norm_1_sg: Pipeline,
    rms_norm_1_wide: Pipeline,
    scale_bf16: Pipeline,
    norm_res_norm_sums: Pipeline,
    lfm2_short_conv_sums: Pipeline,
    add_bias_bf16: Pipeline,
    splice_vision_embs: Pipeline,
    gdn_norm_gate_sums: Pipeline,
    gdn_conv: ?Pipeline = null,
    gdn_commit_history: ?Pipeline = null,
    gdn_scan_1: ?Pipeline = null,
    gdn_scan_2: ?Pipeline = null,
    gdn_scan_4: ?Pipeline = null,
    gdn_scan_8: ?Pipeline = null,
    gdn_conv_batched: ?Pipeline = null,
    gdn_scan_batched: ?Pipeline = null,
    gdn_fused_step_1: ?Pipeline = null,
    gdn_conv_scan_split_1: ?Pipeline = null,
    gdn_scan_record_corr: ?Pipeline = null,
    gdn_rewind_dual: ?Pipeline = null,
    attn_qk_rope_store: Pipeline,
    attn_qk_rope_store_forked: Pipeline,
    attn_flash_fused: Pipeline,
    attn_flash_forked: Pipeline,
    attn_flash_split: Pipeline,
    attn_merge_gate_sums: Pipeline,
    attn_std128_rope_store: Pipeline,
    attn_std128_flash: Pipeline,
    attn_std128_decode_1: Pipeline,
    attn_qwen35_decode_1: Pipeline,
    attn_std128_prefill: Pipeline,
    attn_gen_rope_store: Pipeline,
    attn_gen_flash: Pipeline,
    mla_rms_norm_sums: Pipeline,
    mla_q_rope: Pipeline,
    mla_kv_a_norm_rope: Pipeline,
    mla_kv_b_scatter: Pipeline,
    mla_flash_attn: Pipeline,
    kv_compact_gather: Pipeline,
    kv_compact_scatter: Pipeline,
    gu_activate_sums: Pipeline,
    geglu_activate_sums: Pipeline,
    logit_softcap: Pipeline,
    argmax_stage1: Pipeline,
    argmax_stage2: Pipeline,
    copy_u4: Pipeline,
    // Nemotron-H Mamba-2 + Sparse MoE + General SwiGLU MoE kernels (compiled on demand)
    mamba2_conv_silu: ?Pipeline = null,
    mamba2_ssm_scan: ?Pipeline = null,
    mamba2_norm_gate_sums: ?Pipeline = null,
    moe_gate_logits: ?Pipeline = null,
    moe_topk_select: ?Pipeline = null,
    relu2_sums: ?Pipeline = null,
    qmv_1_relu2: ?Pipeline = null,
    moe_expert_qmv: ?Pipeline = null,
    moe_combine_shared: ?Pipeline = null,
    moe_bucket_tokens: ?Pipeline = null,
    moe_expert_qmm_batched: ?Pipeline = null,
    moe_gather_x_sums: ?Pipeline = null,
    moe_swiglu_gu_qk4: ?Pipeline = null,
    moe_swiglu_gu_affine: ?Pipeline = null,
    moe_swiglu_dn_affine: ?Pipeline = null,
    moe_swiglu_gu_bf16: ?Pipeline = null,
    moe_swiglu_dn_bf16: ?Pipeline = null,
    moe_shared_gate_scalar: ?Pipeline = null,
    moe_combine_swiglu: ?Pipeline = null,

    pub fn compile(ctx: *mtl.Context, arch: cfg_mod.Arch, has_tfq3: bool, has_swiglu_moe: bool) !Pipelines {
        const l_qku = try ctx.compileSource(shaders.qk_universal, .fast);
        const l_norm = try ctx.compileSource(shaders.norm_ops, .fast);
        const l_attn = try ctx.compileSource(shaders.attention, .fast);
        const l_mlp = try ctx.compileSource(shaders.mlp_ops, .fast);
        const l_sample = try ctx.compileSource(shaders.sample, .fast);
        const l_aff = try ctx.compileSource(shaders.affine_any, .fast);

        var qk_m5_lib: ?mtl.Library = null;
        var qk_16: ?Pipeline = null;
        var qk_32: ?Pipeline = null;
        var qk_48: ?Pipeline = null;
        var qk_64: ?Pipeline = null;
        var moe_qk16: ?Pipeline = null;
        var has_tensor_ops = false;

        if (!ctx.force_simd) {
            if (ctx.compileSourceM4(shaders.qk, .fast)) |l_qk| {
                const p16 = ctx.makePipeline(l_qk, "qk_16") catch null;
                const p32 = if (p16 != null) ctx.makePipeline(l_qk, "qk_32") catch null else null;
                const p48 = if (p32 != null) ctx.makePipeline(l_qk, "qk_48") catch null else null;
                const p64 = if (p48 != null) ctx.makePipeline(l_qk, "qk_64") catch null else null;
                if (p16 != null and p32 != null and p48 != null and p64 != null) {
                    qk_m5_lib = l_qk;
                    qk_16 = p16;
                    qk_32 = p32;
                    qk_48 = p48;
                    qk_64 = p64;
                    if (arch == .nemotron_h) {
                        moe_qk16 = ctx.makePipeline(l_qk, "moe_expert_qk16_batched") catch null;
                    }
                    has_tensor_ops = true;
                } else {
                    if (p64) |p| p.deinit();
                    if (p48) |p| p.deinit();
                    if (p32) |p| p.deinit();
                    if (p16) |p| p.deinit();
                    l_qk.deinit();
                }
            } else |_| {}
        }
        ctx.has_tensor_ops = has_tensor_ops;

        var p = Pipelines{
            .libs = .{ l_qku, l_norm, l_attn, l_mlp, l_sample, l_aff },
            .qk_m5_lib = qk_m5_lib,
            .has_tensor_ops = has_tensor_ops,
            .qk_16 = qk_16,
            .qk_32 = qk_32,
            .qk_48 = qk_48,
            .qk_64 = qk_64,
            .moe_expert_qk16_batched = moe_qk16,
            .qmm_simd_8 = try ctx.makePipeline(l_qku, "qmm_simd_8"),
            .qmm_simd_16 = try ctx.makePipeline(l_qku, "qmm_simd_16"),
            .qmv_1 = try ctx.makePipeline(l_qku, "qmv_1"),
            .qmv_1_8col = try ctx.makePipeline(l_qku, "qmv_1_8col"),
            .qmv_1_sk2 = try ctx.makePipeline(l_qku, "qmv_1_sk2"),
            .qmv_1_narrow = try ctx.makePipeline(l_qku, "qmv_1_narrow"),
            .qmv_1_silu_mul = try ctx.makePipeline(l_qku, "qmv_1_silu_mul"),
            .qmv_1_silu_mul_sk4 = try ctx.makePipeline(l_qku, "qmv_1_silu_mul_sk4"),
            .qmv_1_argmax = try ctx.makePipeline(l_qku, "qmv_1_argmax"),
            .qmv_affine_any = try ctx.makePipeline(l_aff, "qmv_affine_any"),
            .qmm_affine_any = try ctx.makePipeline(l_aff, "qmm_affine_any"),
            .bf16_linear_mv1 = try ctx.makePipeline(l_aff, "bf16_linear_mv1"),
            .bf16_linear_mm = try ctx.makePipeline(l_aff, "bf16_linear_mm"),
            .embed_lookup_any = try ctx.makePipeline(l_aff, "embed_lookup_any"),
            .embed_lookup = try ctx.makePipeline(l_norm, "embed_lookup"),
            .rms_norm_sums = try ctx.makePipeline(l_norm, "rms_norm_sums"),
            .add_rms_norm_sums = try ctx.makePipeline(l_norm, "add_rms_norm_sums"),
            .rms_norm_1_sg = try ctx.makePipeline(l_norm, "rms_norm_1_sg"),
            .rms_norm_1_wide = try ctx.makePipeline(l_norm, "rms_norm_1_wide"),
            .scale_bf16 = try ctx.makePipeline(l_norm, "scale_bf16"),
            .norm_res_norm_sums = try ctx.makePipeline(l_norm, "norm_res_norm_sums"),
            .lfm2_short_conv_sums = try ctx.makePipeline(l_norm, "lfm2_short_conv_sums"),
            .add_bias_bf16 = try ctx.makePipeline(l_norm, "add_bias_bf16"),
            .splice_vision_embs = try ctx.makePipeline(l_norm, "splice_vision_embs"),
            .gdn_norm_gate_sums = try ctx.makePipeline(l_norm, "gdn_norm_gate_sums"),
            .attn_qk_rope_store = try ctx.makePipeline(l_attn, "attn_qk_rope_store"),
            .attn_qk_rope_store_forked = try ctx.makePipeline(l_attn, "attn_qk_rope_store_forked"),
            .attn_flash_fused = try ctx.makePipeline(l_attn, "attn_flash_fused"),
            .attn_flash_forked = try ctx.makePipeline(l_attn, "attn_flash_forked"),
            .attn_flash_split = try ctx.makePipeline(l_attn, "attn_flash_split"),
            .attn_merge_gate_sums = try ctx.makePipeline(l_attn, "attn_merge_gate_sums"),
            .attn_std128_rope_store = try ctx.makePipeline(l_attn, "attn_std128_rope_store"),
            .attn_std128_flash = try ctx.makePipeline(l_attn, "attn_std128_flash"),
            .attn_std128_decode_1 = try ctx.makePipeline(l_attn, "attn_std128_decode_1"),
            .attn_qwen35_decode_1 = try ctx.makePipeline(l_attn, "attn_qwen35_decode_1"),
            .attn_std128_prefill = try ctx.makePipeline(l_attn, "attn_std128_prefill"),
            .attn_gen_rope_store = try ctx.makePipeline(l_attn, "attn_gen_rope_store"),
            .attn_gen_flash = try ctx.makePipeline(l_attn, "attn_gen_flash"),
            .mla_rms_norm_sums = try ctx.makePipeline(l_attn, "mla_rms_norm_sums"),
            .mla_q_rope = try ctx.makePipeline(l_attn, "mla_q_rope"),
            .mla_kv_a_norm_rope = try ctx.makePipeline(l_attn, "mla_kv_a_norm_rope"),
            .mla_kv_b_scatter = try ctx.makePipeline(l_attn, "mla_kv_b_scatter"),
            .mla_flash_attn = try ctx.makePipeline(l_attn, "mla_flash_attn"),
            .kv_compact_gather = try ctx.makePipeline(l_attn, "kv_compact_gather"),
            .kv_compact_scatter = try ctx.makePipeline(l_attn, "kv_compact_scatter"),
            .gu_activate_sums = try ctx.makePipeline(l_mlp, "gu_activate_sums"),
            .geglu_activate_sums = try ctx.makePipeline(l_mlp, "geglu_activate_sums"),
            .logit_softcap = try ctx.makePipeline(l_mlp, "logit_softcap"),
            .argmax_stage1 = try ctx.makePipeline(l_sample, "argmax_stage1"),
            .argmax_stage2 = try ctx.makePipeline(l_sample, "argmax_stage2"),
            .copy_u4 = try ctx.makePipeline(l_sample, "copy_u4"),
        };

        if (arch == .qwen35_hybrid) {
            const l_gdn = try ctx.compileSource(shaders.gdn_ops, .fast);
            p.gdn_lib = l_gdn;
            p.gdn_conv = try ctx.makePipeline(l_gdn, "gdn_conv");
            p.gdn_commit_history = try ctx.makePipeline(l_gdn, "gdn_commit_history");
            p.gdn_scan_1 = try ctx.makePipeline(l_gdn, "gdn_scan_1");
            p.gdn_scan_2 = try ctx.makePipeline(l_gdn, "gdn_scan_2");
            p.gdn_scan_4 = try ctx.makePipeline(l_gdn, "gdn_scan_4");
            p.gdn_scan_8 = try ctx.makePipeline(l_gdn, "gdn_scan_8");
            p.gdn_conv_batched = try ctx.makePipeline(l_gdn, "gdn_conv_batched");
            p.gdn_scan_batched = try ctx.makePipeline(l_gdn, "gdn_scan_batched");
            p.gdn_fused_step_1 = try ctx.makePipeline(l_gdn, "gdn_fused_step_1");
            p.gdn_conv_scan_split_1 = try ctx.makePipeline(l_gdn, "gdn_conv_scan_split_1");
            p.gdn_scan_record_corr = try ctx.makePipeline(l_gdn, "gdn_scan_record_corr");
            p.gdn_rewind_dual = try ctx.makePipeline(l_gdn, "gdn_rewind_dual");
        }
        if (arch == .nemotron_h or has_swiglu_moe) {
            const l_moe = try ctx.compileSource(shaders.moe, .fast);
            p.moe_lib = l_moe;
            p.moe_gate_logits = try ctx.makePipeline(l_moe, "moe_gate_logits");
            p.moe_topk_select = try ctx.makePipeline(l_moe, "moe_topk_select");
            p.moe_expert_qmv = try ctx.makePipeline(l_moe, "moe_expert_qmv");
            if (arch == .nemotron_h) {
                const l_mamba = try ctx.compileSource(shaders.mamba2, .fast);
                p.mamba_lib = l_mamba;
                p.mamba2_conv_silu = try ctx.makePipeline(l_mamba, "mamba2_conv_silu");
                p.mamba2_ssm_scan = try ctx.makePipeline(l_mamba, "mamba2_ssm_scan");
                p.mamba2_norm_gate_sums = try ctx.makePipeline(l_mamba, "mamba2_norm_gate_sums");
                p.relu2_sums = try ctx.makePipeline(l_moe, "relu2_sums");
                p.qmv_1_relu2 = try ctx.makePipeline(l_moe, "qmv_1_relu2");
                p.moe_combine_shared = try ctx.makePipeline(l_moe, "moe_combine_shared");
                p.moe_bucket_tokens = try ctx.makePipeline(l_moe, "moe_bucket_tokens");
                p.moe_expert_qmm_batched = try ctx.makePipeline(l_moe, "moe_expert_qmm_batched");
                p.moe_gather_x_sums = try ctx.makePipeline(l_moe, "moe_gather_x_sums");
            }
            if (has_swiglu_moe) {
                p.moe_swiglu_gu_qk4 = try ctx.makePipeline(l_moe, "moe_swiglu_gu_qk4");
                p.moe_swiglu_gu_affine = try ctx.makePipeline(l_moe, "moe_swiglu_gu_affine");
                p.moe_swiglu_dn_affine = try ctx.makePipeline(l_moe, "moe_swiglu_dn_affine");
                p.moe_swiglu_gu_bf16 = try ctx.makePipeline(l_moe, "moe_swiglu_gu_bf16");
                p.moe_swiglu_dn_bf16 = try ctx.makePipeline(l_moe, "moe_swiglu_dn_bf16");
                p.moe_shared_gate_scalar = try ctx.makePipeline(l_moe, "moe_shared_gate_scalar");
                p.moe_combine_swiglu = try ctx.makePipeline(l_moe, "moe_combine_swiglu");
            }
        }

        if (has_tfq3) {
            p.tfq3 = try Tfq3Pipes.compile(ctx);
        }

        return p;
    }

    pub fn deinit(self: *Pipelines) void {
        if (self.tfq3) |*tp| tp.deinit();
        if (self.qk_16) |p| p.deinit();
        if (self.qk_32) |p| p.deinit();
        if (self.qk_48) |p| p.deinit();
        if (self.qk_64) |p| p.deinit();
        if (self.moe_expert_qk16_batched) |p| p.deinit();
        self.qmm_simd_8.deinit();
        self.qmm_simd_16.deinit();
        self.qmv_1.deinit();
        self.qmv_1_8col.deinit();
        self.qmv_1_sk2.deinit();
        self.qmv_1_narrow.deinit();
        self.qmv_1_silu_mul.deinit();
        self.qmv_1_silu_mul_sk4.deinit();
        self.qmv_1_argmax.deinit();
        self.qmv_affine_any.deinit();
        self.qmm_affine_any.deinit();
        self.bf16_linear_mv1.deinit();
        self.bf16_linear_mm.deinit();
        self.embed_lookup_any.deinit();
        self.embed_lookup.deinit();
        self.rms_norm_sums.deinit();
        self.add_rms_norm_sums.deinit();
        self.rms_norm_1_sg.deinit();
        self.rms_norm_1_wide.deinit();
        self.scale_bf16.deinit();
        self.norm_res_norm_sums.deinit();
        self.lfm2_short_conv_sums.deinit();
        self.add_bias_bf16.deinit();
        self.splice_vision_embs.deinit();
        self.gdn_norm_gate_sums.deinit();
        if (self.gdn_conv) |p| p.deinit();
        if (self.gdn_commit_history) |p| p.deinit();
        if (self.gdn_scan_1) |p| p.deinit();
        if (self.gdn_scan_2) |p| p.deinit();
        if (self.gdn_scan_4) |p| p.deinit();
        if (self.gdn_scan_8) |p| p.deinit();
        if (self.gdn_conv_batched) |p| p.deinit();
        if (self.gdn_scan_batched) |p| p.deinit();
        if (self.gdn_fused_step_1) |p| p.deinit();
        if (self.gdn_conv_scan_split_1) |p| p.deinit();
        if (self.gdn_scan_record_corr) |p| p.deinit();
        if (self.gdn_rewind_dual) |p| p.deinit();
        self.attn_qk_rope_store.deinit();
        self.attn_qk_rope_store_forked.deinit();
        self.attn_flash_fused.deinit();
        self.attn_flash_forked.deinit();
        self.attn_flash_split.deinit();
        self.attn_merge_gate_sums.deinit();
        self.attn_std128_rope_store.deinit();
        self.attn_std128_flash.deinit();
        self.attn_std128_decode_1.deinit();
        self.attn_qwen35_decode_1.deinit();
        self.attn_std128_prefill.deinit();
        self.attn_gen_rope_store.deinit();
        self.attn_gen_flash.deinit();
        self.mla_rms_norm_sums.deinit();
        self.mla_q_rope.deinit();
        self.mla_kv_a_norm_rope.deinit();
        self.mla_kv_b_scatter.deinit();
        self.mla_flash_attn.deinit();
        self.kv_compact_gather.deinit();
        self.kv_compact_scatter.deinit();
        self.gu_activate_sums.deinit();
        self.geglu_activate_sums.deinit();
        self.logit_softcap.deinit();
        self.argmax_stage1.deinit();
        self.argmax_stage2.deinit();
        self.copy_u4.deinit();
        if (self.mamba2_conv_silu) |p| p.deinit();
        if (self.mamba2_ssm_scan) |p| p.deinit();
        if (self.mamba2_norm_gate_sums) |p| p.deinit();
        if (self.moe_gate_logits) |p| p.deinit();
        if (self.moe_topk_select) |p| p.deinit();
        if (self.relu2_sums) |p| p.deinit();
        if (self.qmv_1_relu2) |p| p.deinit();
        if (self.moe_expert_qmv) |p| p.deinit();
        if (self.moe_combine_shared) |p| p.deinit();
        if (self.moe_bucket_tokens) |p| p.deinit();
        if (self.moe_expert_qmm_batched) |p| p.deinit();
        if (self.moe_gather_x_sums) |p| p.deinit();
        if (self.moe_swiglu_gu_qk4) |p| p.deinit();
        if (self.moe_swiglu_gu_affine) |p| p.deinit();
        if (self.moe_swiglu_dn_affine) |p| p.deinit();
        if (self.moe_swiglu_gu_bf16) |p| p.deinit();
        if (self.moe_swiglu_dn_bf16) |p| p.deinit();
        if (self.moe_shared_gate_scalar) |p| p.deinit();
        if (self.moe_combine_swiglu) |p| p.deinit();
        if (self.qk_m5_lib) |l| l.deinit();
        if (self.gdn_lib) |l| l.deinit();
        if (self.mamba_lib) |l| l.deinit();
        if (self.moe_lib) |l| l.deinit();
        for (self.libs) |l| l.deinit();
    }
};

pub const Options = struct {
    max_rows: u32 = 128,
    max_context: u32 = 8192,
    gu_parts: u32 = 0, // 0 = auto (4 on >=27B, 1 on smaller models)
    capture_taps: bool = false,
};

pub const PassTiming = struct {
    wall_ms: f64 = 0,
    encode_ms: f64 = 0,
};

const GdnVerifyLayout = struct {
    q_off: usize,
    k_off: usize,
    v_off: usize,
    decay_off: usize,
    mix_off: usize,
    qkv_off: usize,
    corr_off: usize,
    layer_stride: usize,
};

pub const Model = struct {
    a: std.mem.Allocator,
    ctx: *mtl.Context,
    c: ModelConfig,
    w: ModelWeights,
    p: Pipelines,
    fence: mtl.Fence,
    max_rows: u32,
    max_context: u32,
    sums_stride: u32,
    gu_parts: u32,
    commit_every: usize,
    capture_taps: bool,

    // Scratch buffers
    scratch_bufs: std.ArrayList(Buffer) = .empty,
    tokens_buf: Buffer,
    h_buf: Buffer,
    normed_buf: Buffer,
    sums_buf: Buffer,
    proj_big_buf: Buffer,
    delta_buf: Buffer,
    act_buf: Buffer,
    gdn_scan_out_buf: Buffer,
    conv_windows_buf: Buffer,
    scan_steps_buf: Buffer,
    keep_rows_buf: Buffer,
    attn_q_buf: Buffer,
    inv_freq_buf: Buffer,
    inv_freq_local_buf: Buffer,
    positions_buf: Buffer,
    kv_slots_buf: Buffer,
    seq_lens_buf: Buffer,
    tree_masks_buf: Buffer,
    partial_acc_buf: Buffer,
    partial_ml_buf: Buffer,
    kv_tmp_k_buf: Buffer,
    kv_tmp_v_buf: Buffer,
    logits_buf: Buffer,
    argmax_val_buf: Buffer,
    argmax_idx_buf: Buffer,
    out_tok_buf: Buffer,
    out_val_buf: Buffer,

    // Per-linear-layer verify cache (holds q, k, v, decay, mix, raw qkv for fast commitKeep)
    gdn_verify_buf: Buffer,
    gdn_vl: GdnVerifyLayout,

    // Optional 5-layer hidden taps for DFlash2 drafter ([5][max_rows][hidden] bf16)
    taps_buf: ?Buffer = null,
    taps_layer_stride: usize = 0,
    // Optional TFQ3 FWHT rotation & split-K reduction scratch (allocated only when tfq3_store != null)
    tfq3_xrot_buf: ?Buffer = null,
    tfq3_yr_buf: ?Buffer = null,
    tfq3_yr_elems: usize = 0,

    // Persistent stream state
    gdn_state_buf: Buffer,
    gdn_hist_buf: Buffer,
    gdn_hist_alt_buf: Buffer,
    k_cache_buf: Buffer,
    v_cache_buf: Buffer,
    gdn_state_stride: usize,
    gdn_hist_stride: usize,
    kv_layer_stride: usize,
    // Nemotron-H Mamba-2 & Sparse MoE persistent + scratch buffers
    mamba_state_buf: Buffer,
    mamba_hist_buf: Buffer,
    mamba_hist_alt_buf: Buffer,
    mamba_conv_out_buf: Buffer,
    mamba_ssm_out_buf: Buffer,
    moe_logits_buf: Buffer,
    moe_inds_buf: Buffer,
    moe_weights_buf: Buffer,
    moe_act_buf: Buffer,
    moe_out_buf: Buffer,
    moe_offsets_buf: Buffer,
    moe_counts_buf: Buffer,
    moe_buckets_buf: Buffer,
    moe_x_buf: Buffer,
    moe_sums_buf: Buffer,
    moe_shared_gate_buf: Buffer,
    moe_shared_out_buf: Buffer,
    // MLA scratch buffers (allocated when c.use_mla)
    mla_q_raw_buf: Buffer,
    mla_q_norm_buf: Buffer,
    mla_kv_a_buf: Buffer,
    mla_kv_lat_buf: Buffer,
    mla_kpe_buf: Buffer,
    // LiquidAI LFM2 / LFM2.5 ShortConv history state ([numShortConvLayers][l_cache - 1][hidden] bf16)
    lfm_hist_buf: Buffer,
    lfm_hist_stride: usize,
    // Multimodal VLM embedding splice state (active only during prefillWithEmbeddings)
    splice_idx_buf: Buffer,
    splice_embs_ref: ?BufferRef = null,
    mamba_state_stride: usize,
    mamba_hist_stride: usize,
    pos: u32 = 0,
    tape_capture: bool = false,
    auto_api: bool = true,
    last_timing: PassTiming = .{},

    pub const max_splits: u32 = 32;
    pub var use_qmv_1: bool = true;
    pub var use_fused_1: bool = true;
    pub var use_split_k: bool = true;

    pub fn init(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, opts: Options) !Model {
        const c = try ModelConfig.load(a, model_dir);
        var w = try ModelWeights.load(a, ctx, c, model_dir);
        errdefer w.deinit();
        var p = try Pipelines.compile(ctx, c.arch, w.tfq3_store != null, c.moe_style == .swiglu_softmax);
        errdefer p.deinit();
        const fence = try ctx.createFence();
        errdefer fence.deinit();

        const max_rows = opts.max_rows;
        const sums_stride = ((max_rows + 15) / 16) * 16;
        const gdn_all_n = c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads;
        const attn_qkv_n = c.maxAttnQkvOut();
        const max_proj_out = @max(@max(2 * c.intermediate_size, gdn_all_n), @max(attn_qkv_n, @max(c.mambaInProjOut(), 2 * c.moe_shared_intermediate_size)));
        const max_act_dim = @max(@max(c.intermediate_size, c.gdnValDim()), @max(c.maxAttnOutDim(), @max(c.mambaInterDim(), c.moe_shared_intermediate_size)));
        const max_groups = @max(@as(usize, 1), @max(@max(c.hidden_size, max_act_dim), @max(c.q_lora_rank, c.kv_lora_rank)) / 64);
        const chunks: usize = (c.vocab_size + 1023) / 1024;

        // Per-linear-layer verify cache layout
        var vl_off: usize = 0;
        const q_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnKeyDim() * 2);
        const k_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnKeyDim() * 2);
        const v_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnValDim() * 2);
        const decay_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdn_v_heads * 4);
        const mix_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdn_v_heads * 2);
        const qkv_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnQkvDim() * 2);
        const corr_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnValDim() * 4);
        const gdn_vl = GdnVerifyLayout{
            .q_off = q_off,
            .k_off = k_off,
            .v_off = v_off,
            .decay_off = decay_off,
            .mix_off = mix_off,
            .qkv_off = qkv_off,
            .corr_off = corr_off,
            .layer_stride = vl_off,
        };

        var m = Model{
            .a = a,
            .ctx = ctx,
            .c = c,
            .w = w,
            .p = p,
            .fence = fence,
            .max_rows = max_rows,
            .max_context = opts.max_context,
            .sums_stride = sums_stride,
            .gu_parts = if (opts.gu_parts != 0) opts.gu_parts else if (c.hidden_size >= 4096) 4 else 1,
            .commit_every = if (ctx.isMetal4() and c.num_layers >= 48) 8 else c.num_layers,
            .capture_taps = opts.capture_taps,
            .tokens_buf = undefined,
            .h_buf = undefined,
            .normed_buf = undefined,
            .sums_buf = undefined,
            .proj_big_buf = undefined,
            .delta_buf = undefined,
            .act_buf = undefined,
            .gdn_scan_out_buf = undefined,
            .conv_windows_buf = undefined,
            .scan_steps_buf = undefined,
            .keep_rows_buf = undefined,
            .attn_q_buf = undefined,
            .inv_freq_buf = undefined,
            .inv_freq_local_buf = undefined,
            .positions_buf = undefined,
            .kv_slots_buf = undefined,
            .seq_lens_buf = undefined,
            .tree_masks_buf = undefined,
            .partial_acc_buf = undefined,
            .partial_ml_buf = undefined,
            .kv_tmp_k_buf = undefined,
            .kv_tmp_v_buf = undefined,
            .logits_buf = undefined,
            .argmax_val_buf = undefined,
            .argmax_idx_buf = undefined,
            .out_tok_buf = undefined,
            .out_val_buf = undefined,
            .gdn_verify_buf = undefined,
            .gdn_vl = gdn_vl,
            .gdn_state_buf = undefined,
            .gdn_hist_buf = undefined,
            .gdn_hist_alt_buf = undefined,
            .k_cache_buf = undefined,
            .v_cache_buf = undefined,
            .gdn_state_stride = align256(c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim * 4),
            .gdn_hist_stride = align256(3 * c.gdnQkvDim() * 2),
            .kv_layer_stride = align256(@as(usize, opts.max_context) * c.maxKvWidth() * 2),
            .mamba_state_buf = undefined,
            .mamba_hist_buf = undefined,
            .mamba_hist_alt_buf = undefined,
            .mamba_conv_out_buf = undefined,
            .mamba_ssm_out_buf = undefined,
            .moe_logits_buf = undefined,
            .moe_inds_buf = undefined,
            .moe_weights_buf = undefined,
            .moe_act_buf = undefined,
            .moe_out_buf = undefined,
            .moe_offsets_buf = undefined,
            .moe_counts_buf = undefined,
            .moe_buckets_buf = undefined,
            .moe_x_buf = undefined,
            .moe_sums_buf = undefined,
            .moe_shared_gate_buf = undefined,
            .moe_shared_out_buf = undefined,
            .mla_q_raw_buf = undefined,
            .mla_q_norm_buf = undefined,
            .mla_kv_a_buf = undefined,
            .mla_kv_lat_buf = undefined,
            .mla_kpe_buf = undefined,
            .lfm_hist_buf = undefined,
            .lfm_hist_stride = align256(@max(@as(usize, 1), c.conv_l_cache -| 1) * c.hidden_size * 2),
            .splice_idx_buf = undefined,
            .mamba_state_stride = align256(c.mamba_num_heads * c.mamba_head_dim * c.ssm_state_size * 4),
            .mamba_hist_stride = align256(3 * c.mambaConvDim() * 2),
        };
        errdefer m.freeScratch();

        m.tokens_buf = try m.allocScratch(max_rows * 4);
        m.h_buf = try m.allocScratch(@as(usize, max_rows) * c.hidden_size * 2);
        m.normed_buf = try m.allocScratch(@as(usize, max_rows) * c.hidden_size * 2);
        m.sums_buf = try m.allocScratch(max_groups * sums_stride * 4);
        m.proj_big_buf = try m.allocScratch(@as(usize, max_rows) * max_proj_out * 2);
        m.delta_buf = try m.allocScratch(@as(usize, max_rows) * c.hidden_size * 2);
        m.act_buf = try m.allocScratch(@as(usize, max_rows) * max_act_dim * 2);

        m.gdn_scan_out_buf = try m.allocScratch(@max(@as(usize, 256), @as(usize, max_rows) * c.gdnValDim() * 2));
        m.conv_windows_buf = try m.allocScratch(@as(usize, max_rows) * 16);
        m.scan_steps_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.keep_rows_buf = try m.allocScratch(@as(usize, max_rows) * 4);

        const rope_pairs = @max(@as(usize, 32), @max(c.rope_dims / 2, c.maxHeadDim() / 2));
        const local_rope_pairs = @max(@as(usize, 32), c.head_dim / 2);
        m.attn_q_buf = try m.allocScratch(@as(usize, max_rows) * c.maxAttnOutDim() * 2);
        m.inv_freq_buf = try m.allocScratch(rope_pairs * 4);
        m.inv_freq_local_buf = try m.allocScratch(local_rope_pairs * 4);
        m.positions_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.kv_slots_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.seq_lens_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.tree_masks_buf = try m.allocScratch(@as(usize, max_rows) * 16);
        m.partial_acc_buf = try m.allocScratch(@as(usize, max_splits) * c.maxAttnOutDim() * 4);
        m.partial_ml_buf = try m.allocScratch(@as(usize, max_splits) * c.num_heads * 8);
        m.kv_tmp_k_buf = try m.allocScratch(@as(usize, max_rows) * c.maxKvWidth() * 2);
        m.kv_tmp_v_buf = try m.allocScratch(@as(usize, max_rows) * c.maxKvWidth() * 2);

        // Batched logits & argmax buffers (large enough for fused qmv_1_argmax vocab tiles too)
        const argmax_slots = @max(@as(usize, max_rows) * chunks, w.lm_head.tiles32());
        m.logits_buf = try m.allocScratch(@as(usize, max_rows) * c.vocab_size * 2);
        m.argmax_val_buf = try m.allocScratch(argmax_slots * 4);
        m.argmax_idx_buf = try m.allocScratch(argmax_slots * 4);
        m.out_tok_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.out_val_buf = try m.allocScratch(@as(usize, max_rows) * 4);

        const n_lin = c.numLinearLayers();
        const n_full = c.numFullLayers();
        m.gdn_verify_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * gdn_vl.layer_stride));
        m.gdn_state_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * m.gdn_state_stride));
        m.gdn_hist_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * m.gdn_hist_stride));
        m.gdn_hist_alt_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * m.gdn_hist_stride));
        m.k_cache_buf = try m.allocScratch(@max(@as(usize, 256), n_full * m.kv_layer_stride));
        m.v_cache_buf = try m.allocScratch(@max(@as(usize, 256), n_full * m.kv_layer_stride));

        const n_mamba = c.numMambaLayers();
        const is_nem = (c.arch == .nemotron_h);
        const has_moe_any = is_nem or (c.moe_style == .swiglu_softmax);
        m.mamba_state_buf = try m.allocScratch(@max(@as(usize, 256), n_mamba * m.mamba_state_stride));
        m.mamba_hist_buf = try m.allocScratch(@max(@as(usize, 256), n_mamba * m.mamba_hist_stride));
        m.mamba_hist_alt_buf = try m.allocScratch(@max(@as(usize, 256), n_mamba * m.mamba_hist_stride));
        m.mamba_conv_out_buf = try m.allocScratch(@max(@as(usize, 256), if (is_nem) @as(usize, max_rows) * c.mambaConvDim() * 2 else 0));
        m.mamba_ssm_out_buf = try m.allocScratch(@max(@as(usize, 256), if (is_nem) @as(usize, max_rows) * c.mambaInterDim() * 2 else 0));
        m.moe_logits_buf = try m.allocScratch(@max(@as(usize, 256), if (has_moe_any) @as(usize, max_rows) * c.n_routed_experts * 4 else 0));
        m.moe_inds_buf = try m.allocScratch(@max(@as(usize, 256), if (has_moe_any) @as(usize, max_rows) * c.num_experts_per_tok * 4 else 0));
        m.moe_weights_buf = try m.allocScratch(@max(@as(usize, 256), if (has_moe_any) @as(usize, max_rows) * c.num_experts_per_tok * 4 else 0));
        m.moe_act_buf = try m.allocScratch(@max(@as(usize, 256), if (has_moe_any) @as(usize, max_rows) * c.num_experts_per_tok * c.moe_intermediate_size * 2 else 0));
        m.moe_out_buf = try m.allocScratch(@max(@as(usize, 256), if (has_moe_any) @as(usize, max_rows) * c.num_experts_per_tok * c.hidden_size * 2 else 0));
        m.moe_offsets_buf = try m.allocScratch(@max(@as(usize, 256), if (is_nem) c.n_routed_experts * 4 else 0));
        m.moe_counts_buf = try m.allocScratch(@max(@as(usize, 256), if (is_nem) c.n_routed_experts * 4 else 0));
        m.moe_buckets_buf = try m.allocScratch(@max(@as(usize, 256), if (is_nem) @as(usize, max_rows) * c.num_experts_per_tok * 4 else 0));
        const max_moe_slots: usize = if (is_nem) @as(usize, max_rows) * c.num_experts_per_tok else 0;
        const max_moe_sums_stride: usize = ((max_moe_slots + 15) / 16) * 16;
        const max_moe_g: usize = @max(c.hidden_size, c.moe_intermediate_size) / 64;
        m.moe_x_buf = try m.allocScratch(@max(@as(usize, 256), max_moe_slots * c.hidden_size * 2));
        m.moe_sums_buf = try m.allocScratch(@max(@as(usize, 256), max_moe_g * max_moe_sums_stride * 4));
        m.moe_shared_gate_buf = try m.allocScratch(@max(@as(usize, 256), @as(usize, max_rows) * 4));
        m.moe_shared_out_buf = try m.allocScratch(@max(@as(usize, 256), if (c.moe_style == .swiglu_softmax and c.has_shared_expert) @as(usize, max_rows) * c.hidden_size * 2 else 0));
        m.mla_q_raw_buf = try m.allocScratch(@max(@as(usize, 256), if (c.use_mla and c.q_lora_rank > 0) @as(usize, max_rows) * c.q_lora_rank * 2 else 0));
        m.mla_q_norm_buf = try m.allocScratch(@max(@as(usize, 256), if (c.use_mla and c.q_lora_rank > 0) @as(usize, max_rows) * c.q_lora_rank * 2 else 0));
        m.mla_kv_a_buf = try m.allocScratch(@max(@as(usize, 256), if (c.use_mla) @as(usize, max_rows) * c.mlaKvAOut() * 2 else 0));
        m.mla_kv_lat_buf = try m.allocScratch(@max(@as(usize, 256), if (c.use_mla) @as(usize, max_rows) * c.kv_lora_rank * 2 else 0));
        m.mla_kpe_buf = try m.allocScratch(@max(@as(usize, 256), if (c.use_mla) @as(usize, max_rows) * c.qk_rope_head_dim * 2 else 0));
        m.lfm_hist_buf = try m.allocScratch(@max(@as(usize, 256), c.numShortConvLayers() * m.lfm_hist_stride));
        m.splice_idx_buf = try m.allocScratch(@as(usize, max_rows) * 4);

        if (opts.capture_taps) {
            m.taps_layer_stride = align256(@as(usize, max_rows) * c.hidden_size * 2);
            m.taps_buf = try m.allocScratch(tap_layers.len * m.taps_layer_stride);
        }

        if (m.w.tfq3_store != null) {
            const max_tfq3_k = @max(c.hidden_size, max_act_dim);
            m.tfq3_xrot_buf = try m.allocScratch(@as(usize, max_rows) * max_tfq3_k * 2);
            const max_yr_n: usize = @max(@as(usize, 4096), @min(max_proj_out, 16384));
            m.tfq3_yr_elems = 8 * 4 * max_yr_n;
            m.tfq3_yr_buf = try m.allocScratch(m.tfq3_yr_elems * 4);
        }

        // Precompute exact f64 RoPE inverse frequencies (default / Llama 3 piecewise / linear / YaRN / Gemma 3-4 dual & proportional RoPE)
        const inv_f = m.inv_freq_buf.slice(f32, rope_pairs);
        if (c.is_gemma4) {
            const gdim: f64 = @floatFromInt(c.maxHeadDim());
            const g_rot_pairs: usize = @intFromFloat(@round(gdim * c.global_partial_rotary / 2.0));
            for (0..rope_pairs) |i| {
                if (i < g_rot_pairs) {
                    const exp_arg = (2.0 * @as(f64, @floatFromInt(i))) / gdim;
                    inv_f[i] = @floatCast(1.0 / std.math.pow(f64, c.rope_theta, exp_arg));
                } else {
                    inv_f[i] = 0.0; // Proportional RoPE leaves unrotated pairs unchanged!
                }
            }
        } else {
            const rdims: f64 = @floatFromInt(@max(@as(usize, 2), c.rope_dims));
            for (0..rope_pairs) |i| {
                const exp_arg = (2.0 * @as(f64, @floatFromInt(i))) / rdims;
                var freq = 1.0 / std.math.pow(f64, c.rope_theta, exp_arg);
                switch (c.rope_type) {
                    .default => {},
                    .linear => freq /= c.rope_factor,
                    .llama3 => {
                        const wavelen = 2.0 * std.math.pi / freq;
                        const low_wl = c.rope_orig_max_pos / c.rope_low_freq_factor;
                        const high_wl = c.rope_orig_max_pos / c.rope_high_freq_factor;
                        if (wavelen > low_wl) {
                            freq /= c.rope_factor;
                        } else if (wavelen >= high_wl) {
                            const smooth = (c.rope_orig_max_pos / wavelen - c.rope_low_freq_factor) /
                                (c.rope_high_freq_factor - c.rope_low_freq_factor);
                            freq = (1.0 - smooth) * (freq / c.rope_factor) + smooth * freq;
                        }
                    },
                    .yarn => {
                        // Standard HF / DeepSeek YaRN correction range in dimension index space
                        const cd_low = (rdims * @log(c.rope_orig_max_pos / (c.rope_beta_fast * 2.0 * std.math.pi))) / (2.0 * @log(c.rope_theta));
                        const cd_high = (rdims * @log(c.rope_orig_max_pos / (c.rope_beta_slow * 2.0 * std.math.pi))) / (2.0 * @log(c.rope_theta));
                        const low_d = @max(@as(f64, 0.0), @floor(cd_low));
                        var high_d = @min(rdims - 1.0, @ceil(cd_high));
                        if (low_d == high_d) high_d += 0.001;
                        const lin = std.math.clamp((@as(f64, @floatFromInt(i)) - low_d) / (high_d - low_d), 0.0, 1.0);
                        const mask = 1.0 - lin;
                        freq = freq * ((1.0 - mask) / c.rope_factor + mask);
                    },
                }
                inv_f[i] = @floatCast(freq);
            }
        }
        // Local sliding-window RoPE inverse frequencies (Gemma 3 & Gemma 4)
        const inv_f_loc = m.inv_freq_local_buf.slice(f32, local_rope_pairs);
        const ldims: f64 = @floatFromInt(@max(@as(usize, 2), c.head_dim));
        for (0..local_rope_pairs) |i| {
            const exp_arg = (2.0 * @as(f64, @floatFromInt(i))) / ldims;
            inv_f_loc[i] = @floatCast(1.0 / std.math.pow(f64, c.rope_local_theta, exp_arg));
        }

        ctx.commitResidency();
        m.reset();
        return m;
    }

    pub fn deinit(self: *Model) void {
        self.freeScratch();
        self.fence.deinit();
        self.p.deinit();
        self.w.deinit();
    }

    fn allocScratch(self: *Model, len: usize) !Buffer {
        const b = try self.ctx.allocBuffer(len);
        try self.scratch_bufs.append(self.a, b);
        return b;
    }

    fn freeScratch(self: *Model) void {
        for (self.scratch_bufs.items) |b| b.deinit();
        self.scratch_bufs.deinit(self.a);
    }

    pub fn reset(self: *Model) void {
        @memset(self.gdn_state_buf.ptr[0..self.gdn_state_buf.len], 0);
        @memset(self.gdn_hist_buf.ptr[0..self.gdn_hist_buf.len], 0);
        @memset(self.gdn_hist_alt_buf.ptr[0..self.gdn_hist_alt_buf.len], 0);
        if (self.c.arch == .nemotron_h) {
            @memset(self.mamba_state_buf.ptr[0..self.mamba_state_buf.len], 0);
            @memset(self.mamba_hist_buf.ptr[0..self.mamba_hist_buf.len], 0);
            @memset(self.mamba_hist_alt_buf.ptr[0..self.mamba_hist_alt_buf.len], 0);
        }
        if (self.c.is_lfm) {
            @memset(self.lfm_hist_buf.ptr[0..self.lfm_hist_buf.len], 0);
        }
        self.pos = 0;
    }

    /// Prefill prompt tokens in chunks up to `chunk_size` (<= max_rows).
    /// Returns the greedy next token ID predicted from the final prompt position (and populates `logits_buf`).
    pub fn prefill(self: *Model, prompt: []const u32, chunk_size: u32) !u32 {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (@as(usize, self.pos) + prompt.len > self.max_context) return error.ContextLengthExceeded;
        const cap = @min(chunk_size, self.max_rows);
        var offset: usize = 0;
        var next_tok: u32 = 0;
        while (offset < prompt.len) {
            const n: u32 = @intCast(@min(prompt.len - offset, cap));
            const is_last = (offset + n == prompt.len);
            next_tok = try self.forwardCommitChain(prompt[offset..][0..n], is_last, is_last);
            offset += n;
        }
        return next_tok;
    }

    /// Multimodal VLM prefill: replaces each occurrence of `image_token_id` in `prompt` with the next
    /// BF16 vision embedding row from `vis_embs [total_vis_tokens * hidden_size]` after text embed scaling.
    pub fn prefillWithEmbeddings(
        self: *Model,
        prompt: []const u32,
        vis_embs: []const u16,
        image_token_id: u32,
        chunk_size: u32,
    ) !u32 {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (@as(usize, self.pos) + prompt.len > self.max_context) return error.ContextLengthExceeded;
        if (vis_embs.len == 0) return self.prefill(prompt, chunk_size);
        const total_vis: usize = vis_embs.len / self.c.hidden_size;

        const vis_buf = try self.ctx.allocBuffer(vis_embs.len * 2);
        defer self.ctx.freeBuffer(vis_buf);
        @memcpy(vis_buf.slice(u16, vis_embs.len), vis_embs);
        self.ctx.commitResidency();

        const cap = @min(chunk_size, self.max_rows);
        const safe_toks = try self.a.alloc(u32, cap);
        defer self.a.free(safe_toks);

        self.splice_embs_ref = vis_buf.view(0);
        defer self.splice_embs_ref = null;

        var offset: usize = 0;
        var vis_cursor: usize = 0;
        var next_tok: u32 = 0;
        while (offset < prompt.len) {
            const n: u32 = @intCast(@min(prompt.len - offset, cap));
            const is_last = (offset + n == prompt.len);
            const sidx = self.splice_idx_buf.slice(i32, n);
            var has_any_vis = false;
            for (0..n) |r| {
                const tid = prompt[offset + r];
                if (tid == image_token_id and vis_cursor < total_vis) {
                    sidx[r] = @intCast(vis_cursor);
                    safe_toks[r] = 0; // safe in-bounds dummy for embed_lookup; overwritten by splice_vision_embs
                    vis_cursor += 1;
                    has_any_vis = true;
                } else {
                    sidx[r] = -1;
                    safe_toks[r] = tid;
                }
            }
            self.splice_embs_ref = if (has_any_vis) vis_buf.view(0) else null;
            next_tok = try self.forwardCommitChain(safe_toks[0..n], is_last, is_last);
            offset += n;
        }
        return next_tok;
    }

    /// Decode one token (committed immediately) using fused LM-head argmax (zero logits_buf write).
    pub fn step(self: *Model, token: u32) !u32 {
        const one = [1]u32{token};
        return self.forwardCommitChain(&one, true, false);
    }

    /// Decode one token (committed immediately) and materialize full bf16 logits in `logits_buf` (for sampling).
    pub fn stepWithLogits(self: *Model, token: u32) !u32 {
        const one = [1]u32{token};
        return self.forwardCommitChain(&one, true, true);
    }

    /// Speculatively verify a linear chain `tokens[0..rows]` without committing GDN state or `pos`.
    /// Returns the slice of target greedy picks `picks[0..rows]` (in shared memory, valid until next call).
    pub fn verifyChain(self: *Model, tokens: []const u32) ![]const u32 {
        if (tokens.len == 0) return error.EmptyPrompt;
        if (tokens.len > self.max_rows) return error.BatchTooLarge;
        const rows: u32 = @intCast(tokens.len);
        var parents: [128]i32 = undefined;
        for (0..rows) |r| parents[r] = @as(i32, @intCast(r)) - 1;
        return self.verifyTree(tokens, parents[0..rows], false);
    }

    /// Speculatively verify an arbitrary tree (`parents[0] == -1`, `parents[r] < r`) without committing GDN state or `pos`.
    /// Returns greedy picks `picks[0..rows]`. Call `commitKeep` afterwards with the accepted row indices!
    pub fn verifyTree(self: *Model, tokens: []const u32, parents: []const i32, is_tree: bool) ![]const u32 {
        if (tokens.len == 0) return error.EmptyPrompt;
        if (tokens.len > self.max_rows or parents.len != tokens.len) return error.BatchTooLarge;
        const rows: u32 = @intCast(tokens.len);
        if (@as(usize, self.pos) + rows > self.max_context) return error.ContextLengthExceeded;

        const t_start = nowMs();

        const tok_dst = self.tokens_buf.slice(u32, rows);
        const pos_dst = self.positions_buf.slice(u32, rows);
        const slot_dst = self.kv_slots_buf.slice(u32, rows);
        const seq_dst = self.seq_lens_buf.slice(u32, rows);
        const mask_dst = self.tree_masks_buf.slice([4]u32, rows);
        const win_dst = self.conv_windows_buf.slice([4]u32, rows);
        const step_dst = self.scan_steps_buf.slice(u32, rows);

        // Compute depth, positions, KV slots, 128-bit ancestor masks, and 4-tap conv windows
        var depths: [128]u32 = undefined;
        for (0..rows) |r| {
            const p = parents[r];
            depths[r] = if (p < 0) 0 else depths[@intCast(p)] + 1;
            tok_dst[r] = tokens[r];
            pos_dst[r] = self.pos + depths[r];
            slot_dst[r] = self.pos + @as(u32, @intCast(r));
            seq_dst[r] = self.pos + @as(u32, @intCast(r)) + 1;

            // Ancestor mask for FlashAttention
            var mask: [4]u32 = .{ 0, 0, 0, 0 };
            var cur: i32 = @intCast(r);
            while (cur >= 0) {
                const uc: u32 = @intCast(cur);
                mask[uc >> 5] |= (@as(u32, 1) << @intCast(uc & 31));
                cur = parents[uc];
            }
            mask_dst[r] = mask;

            // 4-tap causal conv window walking 3 steps back along tree parents
            var w: [4]u32 = undefined;
            var node: i32 = @intCast(r);
            var back_from_root: u32 = 0;
            var tap_i: usize = 4;
            while (tap_i > 0) {
                tap_i -= 1;
                if (node >= 0) {
                    w[tap_i] = 3 + @as(u32, @intCast(node));
                    node = parents[@intCast(node)];
                } else {
                    // 1 step before root -> history[2], 2 steps -> history[1], 3 steps -> history[0]
                    w[tap_i] = if (back_from_root < 3) 2 - back_from_root else 0;
                    back_from_root += 1;
                }
            }
            win_dst[r] = w;
        }

        const slots_needed = if (is_tree) compileTreeSteps(parents, step_dst) else blk: {
            for (0..rows) |r| step_dst[r] = @intCast(r);
            break :blk @as(u32, 0);
        };

        try self.executeForward(rows, true, false, true, true, is_tree, slots_needed, t_start);
        return self.out_tok_buf.slice(u32, rows);
    }

    /// Commit the accepted path `accepted_rows[0..keep_len]` after `verifyChain` or `verifyTree`.
    /// Updates all linear layers' GDN state + conv history, compacts KV cache if non-contiguous, and advances `pos`.
    pub fn commitKeep(self: *Model, accepted_rows: []const u32) !void {
        const keep_len: u32 = @intCast(accepted_rows.len);
        std.debug.assert(keep_len > 0 and keep_len <= self.max_rows);

        var contiguous = true;
        const step_dst = self.scan_steps_buf.slice(u32, keep_len);
        const keep_dst = self.keep_rows_buf.slice(u32, keep_len);
        for (accepted_rows, 0..) |r, i| {
            step_dst[i] = r;
            keep_dst[i] = r;
            if (r != i) contiguous = false;
        }

        // Compute new 3-tap history sources from [hist0, hist1, hist2, keep_qkv[accepted_rows...]]
        const hist_src0: u32 = if (keep_len >= 3) 3 + accepted_rows[keep_len - 3] else if (keep_len == 2) 2 else 1;
        const hist_src1: u32 = if (keep_len >= 2) 3 + accepted_rows[keep_len - 2] else 2;
        const hist_src2: u32 = 3 + accepted_rows[keep_len - 1];

        const c = self.c;
        const n_lin = c.numLinearLayers();
        if (n_lin == 0 and contiguous) {
            self.pos += keep_len;
            return;
        }
        const gdn_qkv_dim: u32 = @intCast(c.gdnQkvDim());

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // Advance all GDN layers (state + conv history) in one pass
        for (0..n_lin) |lin_idx| {
            const state_ref = self.gdn_state_buf.view(lin_idx * self.gdn_state_stride);
            const hist_ref = self.gdn_hist_buf.view(lin_idx * self.gdn_hist_stride);
            const vbase = lin_idx * self.gdn_vl.layer_stride;

            enc.setPipeline(self.p.gdn_scan_1.?);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.q_off), 0);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.k_off), 1);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.v_off), 2);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.decay_off), 3);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.mix_off), 4);
            enc.setRef(state_ref, 5);
            enc.setBuffer(self.scan_steps_buf, 0, 6);
            enc.setValue(GdnScanParams{
                .rows = keep_len,
                .nk = @intCast(c.gdn_k_heads),
                .nv = @intCast(c.gdn_v_heads),
                .commit_state = 1,
            }, 7);
            enc.setBuffer(self.gdn_scan_out_buf, 0, 8);
            enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });

            enc.setPipeline(self.p.gdn_commit_history.?);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.qkv_off), 0);
            enc.setRef(hist_ref, 1);
            enc.setValue(GdnHistCommitParams{
                .qkv_dim = gdn_qkv_dim,
                .in_proj_stride = gdn_qkv_dim,
                .src0 = hist_src0,
                .src1 = hist_src1,
                .src2 = hist_src2,
            }, 2);
            enc.dispatchThreads(.{ .width = gdn_qkv_dim / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        }

        // If tree accepted rows were non-contiguous, compact K/V cache slots into [pos .. pos + keep_len)
        if (!contiguous) {
            const n_full = c.numFullLayers();
            const row_u4: u32 = @intCast((c.num_kv_heads * c.head_dim * 2) / 16);
            const kp = KvCompactParams{
                .base_pos = self.pos,
                .keep_len = keep_len,
                .row_u4 = row_u4,
            };
            for (0..n_full) |full_idx| {
                const k_ref = self.k_cache_buf.view(full_idx * self.kv_layer_stride);
                const v_ref = self.v_cache_buf.view(full_idx * self.kv_layer_stride);

                enc.setPipeline(self.p.kv_compact_gather);
                enc.setRef(k_ref, 0);
                enc.setRef(v_ref, 1);
                enc.setBuffer(self.keep_rows_buf, 0, 2);
                enc.setBuffer(self.kv_tmp_k_buf, 0, 3);
                enc.setBuffer(self.kv_tmp_v_buf, 0, 4);
                enc.setValue(kp, 5);
                enc.dispatchThreads(.{ .width = row_u4, .height = keep_len, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(self.p.kv_compact_scatter);
                enc.setBuffer(self.kv_tmp_k_buf, 0, 0);
                enc.setBuffer(self.kv_tmp_v_buf, 0, 1);
                enc.setRef(k_ref, 2);
                enc.setRef(v_ref, 3);
                enc.setValue(kp, 4);
                enc.dispatchThreads(.{ .width = row_u4, .height = keep_len, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                enc.barrier();
            }
        }

        enc.end();
        cb.commit();
        try self.ctx.waitIdle();
        self.pos += keep_len;
    }

    pub fn readLogit(self: *const Model, token_id: usize) f32 {
        const raw = self.logits_buf.slice(u16, self.c.vocab_size)[token_id];
        return @bitCast(@as(u32, raw) << 16);
    }

    /// Sample one token from row `row` of `logits_buf` using `params` (greedy fast-path when temp == 0).
    pub fn sampleRow(self: *const Model, row: usize, params: samp_mod.SamplingParams, rng: *samp_mod.Rng) !u32 {
        return self.sampleRowWithPenalties(row, params, rng, null);
    }

    pub fn sampleRowWithPenalties(
        self: *const Model,
        row: usize,
        params: samp_mod.SamplingParams,
        rng: *samp_mod.Rng,
        seen_counts: ?*const std.AutoHashMap(u32, u32),
    ) !u32 {
        const greedy = self.out_tok_buf.slice(u32, row + 1)[row];
        const has_pen = seen_counts != null and
            ((params.repetition_penalty != 1.0 and params.repetition_penalty > 0.0) or
                params.frequency_penalty != 0.0 or params.presence_penalty != 0.0);
        if (params.temperature <= 1e-6 and !has_pen) return greedy;
        const max_v = self.out_val_buf.slice(f32, row + 1)[row];
        const all_logits = self.logits_buf.slice(u16, (row + 1) * self.c.vocab_size);
        const row_logits = all_logits[row * self.c.vocab_size ..][0..self.c.vocab_size];
        return samp_mod.sampleLogitsWithPenalties(self.a, row_logits, greedy, max_v, params, rng, seen_counts);
    }

    /// Public projection helper reused by multi-stream batching.
    pub fn dispatchQkPub(self: *const Model, enc: *mtl.Encoder, x: BufferRef, w: LinearWeight, sums: BufferRef, y: BufferRef, rows: u32, parts: u32) void {
        self.dispatchQk(enc, x, w, sums, y, rows, parts);
    }

    /// Universal token embedding dispatch (fast 4-bit g64 path or general 2-8b/BF16/F16/F32 path).
    pub fn dispatchEmbed(self: *const Model, enc: *mtl.Encoder, tokens: BufferRef, out: BufferRef, rows: u32) void {
        if (self.w.embed.isFastQk4()) {
            enc.setPipeline(self.p.embed_lookup);
            enc.setRef(tokens, 0);
            enc.setRef(self.w.embed.codes, 1);
            enc.setRef(self.w.embed.scales, 2);
            enc.setRef(self.w.embed.biases, 3);
            enc.setRef(out, 4);
            enc.setValue(EmbedParams{ .rows = rows, .hidden = @intCast(self.c.hidden_size) }, 5);
            enc.dispatchThreads(.{ .width = self.c.hidden_size / 8, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            return;
        }
        enc.setPipeline(self.p.embed_lookup_any);
        enc.setRef(tokens, 0);
        enc.setRef(self.w.embed.codes, 1);
        enc.setRef(self.w.embed.scales, 2);
        enc.setRef(self.w.embed.biases, 3);
        enc.setRef(out, 4);
        enc.setValue(AffineParams{
            .rows = rows,
            .N = @intCast(self.c.vocab_size),
            .K = @intCast(self.c.hidden_size),
            .ldx = @intCast(self.c.hidden_size),
            .ldo = @intCast(self.c.hidden_size),
            .bits = self.w.embed.bits,
            .gs = self.w.embed.gs,
            .sb_dtype = @intFromEnum(self.w.embed.sb_dtype),
            .add_res = 0,
        }, 5);
        enc.dispatchGroups(.{ .width = (self.c.hidden_size + 31) / 32, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
    }

    /// Return a host-readable bf16 slice for captured tap `tap_idx` (0..4) at row `row`.
    pub fn tapRowBf16(self: *const Model, tap_idx: usize, row: usize) []const u16 {
        const tb = self.taps_buf orelse unreachable;
        const off = tap_idx * self.taps_layer_stride + row * self.c.hidden_size * 2;
        const ptr: [*]const u16 = @ptrCast(@alignCast(tb.ptr + off));
        return ptr[0..self.c.hidden_size];
    }

    fn dispatchQk(self: *const Model, enc: *mtl.Encoder, x: BufferRef, w: LinearWeight, sums: BufferRef, y: BufferRef, rows: u32, parts: u32) void {
        self.dispatchQkOpt(enc, x, w, sums, y, rows, parts, false);
    }

    fn dispatchAffine(self: *const Model, enc: *mtl.Encoder, x: BufferRef, m: AffineMatrix, y: BufferRef, ldx: u32, ldo: u32, rows: u32, add_residual: bool) void {
        const params = AffineParams{
            .rows = rows,
            .N = m.n,
            .K = m.k,
            .ldx = ldx,
            .ldo = ldo,
            .bits = m.bits,
            .gs = m.gs,
            .sb_dtype = @intFromEnum(m.sb_dtype),
            .add_res = if (add_residual) 1 else 0,
        };
        if (rows == 1) {
            enc.setPipeline(self.p.qmv_affine_any);
            enc.setRef(x, 0);
            enc.setRef(m.codes, 1);
            enc.setRef(m.scales, 2);
            enc.setRef(m.biases, 3);
            enc.setRef(y, 4);
            enc.setValue(params, 5);
            enc.dispatchGroups(.{ .width = (m.n + 3) / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        } else {
            enc.setPipeline(self.p.qmm_affine_any);
            enc.setRef(x, 0);
            enc.setRef(m.codes, 1);
            enc.setRef(m.scales, 2);
            enc.setRef(m.biases, 3);
            enc.setRef(y, 4);
            enc.setValue(params, 5);
            enc.dispatchGroups(.{ .width = (m.n + 3) / 4, .height = (rows + 3) / 4, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        }
    }

    fn dispatchBf16(self: *const Model, enc: *mtl.Encoder, x: BufferRef, m: Bf16Matrix, y: BufferRef, ldx: u32, ldo: u32, rows: u32, add_residual: bool) void {
        const params = AffineParams{
            .rows = rows,
            .N = m.n,
            .K = m.k,
            .ldx = ldx,
            .ldo = ldo,
            .bits = 0,
            .gs = 0,
            .sb_dtype = 0,
            .add_res = if (add_residual) 1 else 0,
        };
        if (rows == 1) {
            enc.setPipeline(self.p.bf16_linear_mv1);
            enc.setRef(x, 0);
            enc.setRef(m.ref, 1);
            enc.setRef(y, 2);
            enc.setValue(params, 3);
            enc.dispatchGroups(.{ .width = (m.n + 3) / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        } else {
            enc.setPipeline(self.p.bf16_linear_mm);
            enc.setRef(x, 0);
            enc.setRef(m.ref, 1);
            enc.setRef(y, 2);
            enc.setValue(params, 3);
            enc.dispatchGroups(.{ .width = (m.n + 3) / 4, .height = (rows + 3) / 4, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        }
    }

    fn dispatchTfq3FwhtX(self: *const Model, enc: *mtl.Encoder, x: BufferRef, k: u32, ldx: u32, rows: u32) void {
        const tp = self.p.tfq3.?;
        const xrot = self.tfq3_xrot_buf.?;
        const dims = Tfq3Dims{
            .rows = rows,
            .n = 0,
            .k = k,
            .ldx = ldx,
            .ldo = 0,
            .sr_split = 0,
            .ksplit = 1,
            .add_res = 0,
        };
        enc.setPipeline(tp.fwht_x);
        enc.setRef(x, 0);
        enc.setBuffer(xrot, 0, 1);
        enc.setValue(dims, 2);
        enc.dispatchGroups(.{ .width = k / 128, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
        enc.barrier();
    }

    fn dispatchTfq3Rotated(self: *const Model, enc: *mtl.Encoder, m: Tfq3Matrix, y: BufferRef, ldo: u32, rows: u32, add_residual: bool) bool {
        const tp = self.p.tfq3.?;
        const xrot = self.tfq3_xrot_buf.?;
        const bi: usize = m.bits - 3;
        const add_r: u32 = if (add_residual) 1 else 0;

        if (rows <= 4) {
            const n_bc = m.k / 128;
            const yr_need: usize = 8 * @as(usize, rows) * @as(usize, m.n);
            const use_2dsk8 = (n_bc >= 8 and m.n <= 8192 and m.k >= 2048 and yr_need <= self.tfq3_yr_elems);
            if (use_2dsk8) {
                const yr = self.tfq3_yr_buf.?;
                const dims = Tfq3Dims{
                    .rows = rows,
                    .n = m.n,
                    .k = m.k,
                    .ldx = m.k,
                    .ldo = ldo,
                    .sr_split = m.sr_split,
                    .ksplit = 8,
                    .add_res = add_r,
                };
                const pso = if (rows == 1) tp.dec_2dsk8_1[bi] else if (rows == 2) tp.dec_2dsk8_2[bi] else tp.dec_2dsk8_4[bi];
                enc.setPipeline(pso);
                enc.setRef(m.words, 0);
                enc.setRef(m.scales, 1);
                enc.setRef(m.tables, 2);
                enc.setBuffer(xrot, 0, 3);
                enc.setBuffer(yr, 0, 4);
                enc.setValue(dims, 5);
                enc.dispatchGroups(.{ .width = m.n / 32, .height = 8, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(tp.fwht_y);
                enc.setBuffer(yr, 0, 0);
                enc.setRef(y, 1);
                enc.setValue(dims, 2);
                enc.dispatchGroups(.{ .width = m.n / 128, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                return true;
            } else {
                const dims = Tfq3Dims{
                    .rows = rows,
                    .n = m.n,
                    .k = m.k,
                    .ldx = m.k,
                    .ldo = ldo,
                    .sr_split = m.sr_split,
                    .ksplit = 1,
                    .add_res = add_r,
                };
                const pso = if (rows == 1) tp.dec_fused1[bi] else if (rows == 2) tp.dec_fused2[bi] else tp.dec_fused4[bi];
                enc.setPipeline(pso);
                enc.setRef(m.words, 0);
                enc.setRef(m.scales, 1);
                enc.setRef(m.tables, 2);
                enc.setBuffer(xrot, 0, 3);
                enc.setRef(y, 4);
                enc.setValue(dims, 5);
                enc.dispatchGroups(.{ .width = m.n / 128, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                return false;
            }
        } else {
            const dims = Tfq3Dims{
                .rows = rows,
                .n = m.n,
                .k = m.k,
                .ldx = m.k,
                .ldo = ldo,
                .sr_split = m.sr_split,
                .ksplit = 1,
                .add_res = add_r,
            };
            enc.setPipeline(tp.prefill[bi]);
            enc.setRef(m.words, 0);
            enc.setRef(m.scales, 1);
            enc.setRef(m.tables, 2);
            enc.setBuffer(xrot, 0, 3);
            enc.setRef(y, 4);
            enc.setValue(dims, 5);
            enc.dispatchGroups(.{ .width = (rows + 15) / 16, .height = m.n / 128, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            return false;
        }
    }

    fn dispatchTfq3(self: *const Model, enc: *mtl.Encoder, x: BufferRef, m: Tfq3Matrix, y: BufferRef, ldx: u32, ldo: u32, rows: u32, add_residual: bool) void {
        self.dispatchTfq3FwhtX(enc, x, m.k, ldx, rows);
        _ = self.dispatchTfq3Rotated(enc, m, y, ldo, rows, add_residual);
    }

    fn dispatchQkOpt(self: *const Model, enc: *mtl.Encoder, x: BufferRef, lw: LinearWeight, sums: BufferRef, y: BufferRef, rows: u32, parts: u32, add_residual: bool) void {
        switch (lw) {
            .qk4 => |w| self.dispatchQk4Opt(enc, x, w, sums, y, rows, parts, add_residual),
            .affine => |m| self.dispatchAffine(enc, x, m, y, m.k, m.n, rows, add_residual),
            .bf16 => |m| self.dispatchBf16(enc, x, m, y, m.k, m.n, rows, add_residual),
            .tfq3 => |m| self.dispatchTfq3(enc, x, m, y, m.k, m.n, rows, add_residual),
            .concat => |cat| {
                var has_tfq3 = false;
                for (0..cat.count) |i| {
                    if (cat.parts[i] == .tfq3) {
                        has_tfq3 = true;
                        break;
                    }
                }
                if (has_tfq3) self.dispatchTfq3FwhtX(enc, x, cat.k, cat.k, rows);
                for (0..cat.count) |i| {
                    const y_sub = y.at(@as(usize, cat.col_offs[i]) * 2);
                    switch (cat.parts[i]) {
                        .affine => |m| self.dispatchAffine(enc, x, m, y_sub, cat.k, cat.n, rows, add_residual),
                        .bf16 => |m| self.dispatchBf16(enc, x, m, y_sub, cat.k, cat.n, rows, add_residual),
                        .tfq3 => |m| {
                            const used_yr = self.dispatchTfq3Rotated(enc, m, y_sub, cat.n, rows, add_residual);
                            if (used_yr and i + 1 < cat.count) enc.barrier();
                        },
                    }
                }
            },
        }
    }

    fn dispatchQk4Opt(self: *const Model, enc: *mtl.Encoder, x: BufferRef, w: QkMatrix, sums: BufferRef, y: BufferRef, rows: u32, parts: u32, add_residual: bool) void {
        if (rows == 1 and use_qmv_1) {
            const tiles = w.tiles();
            const g_cnt = w.groups();
            const params = QkParams{
                .M = 1,
                .K = @intCast(w.k),
                .N = @intCast(w.n),
                .G = @intCast(g_cnt),
                .first_row = 0,
                .sums_stride = @intCast(self.sums_stride),
                .tile_first = if (add_residual) 1 else 0,
            };
            // 100% contiguous 32-col 128-bit uint4 streaming with empirically optimal K-splitting:
            //   - Wide projections (>=1024 tiles or >=256 tiles & G>=32: lm_head, 2B/27B/30B in_proj): SK=1 zero-barrier
            //   - Narrow projections (<=64 tiles, G>=32: 0.8B/1.5B/2B out_proj/o_proj/down_proj): SK=8 (256..512 SGs!)
            //   - All other projections (0.8B in_proj_all G=16, 1.5B qkv/o, 7B/27B): SK=4 (192..1028 SGs!)
            const is_sk1 = (tiles >= 1024) or (tiles >= 256 and g_cnt >= 32);
            const is_sk8 = (tiles <= 64 and g_cnt >= 32);
            const pso: Pipeline = if (is_sk1)
                self.p.qmv_1
            else if (is_sk8)
                self.p.qmv_1_8col
            else
                self.p.qmv_1_narrow;
            const threads: usize = if (is_sk1) 32 else if (is_sk8) 256 else 128;
            enc.setPipeline(pso);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(y, 2);
            enc.setValue(params, 3);
            enc.dispatchGroups(.{ .width = tiles, .height = 1, .depth = 1 }, .{ .width = threads, .height = 1, .depth = 1 });
            return;
        }
        const total_tiles = w.tiles();
        const n_parts = @max(1, parts);

        if (self.p.has_tensor_ops) {
            // 2D L2-cached dispatch: grp.x = m_tile (inner), grp.y = n_tile (outer).
            // All M-tiles for each 32-col weight tile launch simultaneously and share the same weights in L2/SLC!
            const pso: Pipeline = if (rows <= 16)
                self.p.qk_16.?
            else if (rows <= 32)
                self.p.qk_32.?
            else if (rows <= 48)
                self.p.qk_48.?
            else
                self.p.qk_32.?;
            const m_tiles: u32 = if (rows <= 48) 1 else (rows + 31) / 32;
            enc.setPipeline(pso);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(sums, 2);
            enc.setRef(y, 3);

            var part: u32 = 0;
            while (part < n_parts) : (part += 1) {
                const t0 = total_tiles * part / n_parts;
                const t1 = total_tiles * (part + 1) / n_parts;
                if (t1 <= t0) continue;
                const params = QkParams{
                    .M = @intCast(rows),
                    .K = @intCast(w.k),
                    .N = @intCast(w.n),
                    .G = @intCast(w.groups()),
                    .first_row = 0,
                    .sums_stride = @intCast(self.sums_stride),
                    .tile_first = t0,
                };
                enc.setValue(params, 4);
                enc.dispatchGroups(.{ .width = m_tiles, .height = t1 - t0, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            }
            return;
        }

        var first_row: u32 = 0;
        while (first_row < rows) {
            const rem = rows - first_row;
            const step_rows: u32 = @min(rem, 16);
            const pso: Pipeline = if (step_rows <= 8) self.p.qmm_simd_8 else self.p.qmm_simd_16;
            enc.setPipeline(pso);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(sums, 2);
            enc.setRef(y, 3);

            var part: u32 = 0;
            while (part < n_parts) : (part += 1) {
                const t0 = total_tiles * part / n_parts;
                const t1 = total_tiles * (part + 1) / n_parts;
                if (t1 <= t0) continue;
                const params = QkParams{
                    .M = @intCast(step_rows),
                    .K = @intCast(w.k),
                    .N = @intCast(w.n),
                    .G = @intCast(w.groups()),
                    .first_row = @intCast(first_row),
                    .sums_stride = @intCast(self.sums_stride),
                    .tile_first = t0,
                };
                enc.setValue(params, 4);
                enc.dispatchGroups(.{ .width = t1 - t0, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            }
            first_row += step_rows;
        }
    }

    fn dispatchRmsNorm1(self: *const Model, enc: *mtl.Encoder, x: BufferRef, weight: BufferRef, out: BufferRef) void {
        const use_sg = (self.c.hidden_size <= 1024);
        enc.setPipeline(if (use_sg) self.p.rms_norm_1_sg else self.p.rms_norm_1_wide);
        enc.setRef(x, 0);
        enc.setRef(weight, 1);
        enc.setRef(out, 2);
        enc.setValue(NormParams{
            .rows = 1,
            .hidden = @intCast(self.c.hidden_size),
            .sums_stride = 0,
            .eps = self.c.rms_eps,
        }, 3);
        enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = if (use_sg) 32 else 128, .height = 1, .depth = 1 });
    }

    fn forwardCommitChain(self: *Model, tokens: []const u32, want_logits: bool, need_full_logits: bool) !u32 {
        if (tokens.len == 0) return error.EmptyPrompt;
        if (tokens.len > self.max_rows) return error.BatchTooLarge;
        const rows: u32 = @intCast(tokens.len);
        if (@as(usize, self.pos) + rows > self.max_context) return error.ContextLengthExceeded;

        const t_start = nowMs();

        const tok_dst = self.tokens_buf.slice(u32, rows);
        const pos_dst = self.positions_buf.slice(u32, rows);
        const slot_dst = self.kv_slots_buf.slice(u32, rows);
        const seq_dst = self.seq_lens_buf.slice(u32, rows);
        const win_dst = self.conv_windows_buf.slice([4]u32, rows);
        const step_dst = self.scan_steps_buf.slice(u32, rows);

        for (0..rows) |r| {
            const ur: u32 = @intCast(r);
            tok_dst[r] = tokens[r];
            pos_dst[r] = self.pos + ur;
            slot_dst[r] = self.pos + ur;
            seq_dst[r] = self.pos + ur + 1;
            win_dst[r] = .{ ur, ur + 1, ur + 2, ur + 3 };
            step_dst[r] = ur;
        }

        try self.executeForward(rows, want_logits, true, false, need_full_logits, false, 0, t_start);
        self.pos += rows;
        return if (want_logits) self.out_tok_buf.slice(u32, 1)[0] else 0;
    }

    fn executeForward(
        self: *Model,
        rows: u32,
        want_logits: bool,
        commit_immediately: bool,
        all_row_logits: bool,
        need_full_logits: bool,
        is_tree: bool,
        scan_slots: u32,
        t_start: f64,
    ) !void {
        // Auto-hybrid Metal 4/3 selector: Metal 4 for multi-row prefill/verify & >=48-layer GDN decode;
        // Metal 3 serial for minimal launch latency on small-model single-token decode.
        if (self.auto_api and self.ctx.has_metal4) {
            if (rows > 1 or (self.c.arch == .qwen35_hybrid and self.c.num_layers >= 48)) {
                self.ctx.api = .metal4_device;
            } else {
                self.ctx.api = .metal3_serial;
            }
        }
        if (self.c.arch == .nemotron_h) {
            return self.executeNemotronH(rows, want_logits, commit_immediately, all_row_logits, need_full_logits, is_tree, t_start);
        }
        const c = self.c;
        const commit_every: usize = if (self.ctx.isMetal4() and c.num_layers >= 48) 8 else c.num_layers;
        const gdn_qkv_dim: u32 = @intCast(c.gdnQkvDim());
        const gdn_val_dim: u32 = @intCast(c.gdnValDim());
        const gdn_all_stride: u32 = @intCast(c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads);
        const attn_qkv_stride: u32 = @intCast(c.qProjOut() + 2 * c.kvProjOut());

        const use_fused_gdn_1 = (rows == 1 and commit_immediately and use_fused_1 and c.numLinearLayers() > 0 and !self.tape_capture);
        const use_qmv_fused_1 = (rows == 1 and use_qmv_1 and use_fused_1);
        const use_res_fusion = use_qmv_fused_1 and !c.has_post_norms and !c.conv_bias;
        const norm_sums_stride: u32 = if (rows == 1 and use_qmv_1) 0 else self.sums_stride;

        const total_hist: u32 = 3 + rows;
        const hist_src0 = total_hist - 3;
        const hist_src1 = total_hist - 2;
        const hist_src2 = total_hist - 1;

        const scan_pso = if (scan_slots == 0)
            self.p.gdn_scan_1
        else if (scan_slots <= 2)
            self.p.gdn_scan_2
        else if (scan_slots <= 4)
            self.p.gdn_scan_4
        else
            self.p.gdn_scan_8;

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // 1. Embedding lookup -> h_buf (+ optional Gemma embed_scale = sqrt(hidden_size))
        self.dispatchEmbed(&enc, self.tokens_buf.view(0), self.h_buf.view(0), rows);
        enc.barrier();
        if (c.embed_scale != 1.0) {
            const cnt: u32 = @intCast(@as(usize, rows) * c.hidden_size);
            enc.setPipeline(self.p.scale_bf16);
            enc.setBuffer(self.h_buf, 0, 0);
            enc.setValue(ScaleParams{ .count = cnt, .scale = c.embed_scale }, 1);
            enc.dispatchThreads(.{ .width = cnt, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();
        }
        if (self.splice_embs_ref) |vref| {
            enc.setPipeline(self.p.splice_vision_embs);
            enc.setBuffer(self.h_buf, 0, 0);
            enc.setRef(vref, 1);
            enc.setBuffer(self.splice_idx_buf, 0, 2);
            enc.setValue(SpliceParams{ .rows = rows, .hidden = @intCast(c.hidden_size) }, 3);
            enc.dispatchThreads(.{ .width = c.hidden_size, .height = rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();
        }

        // 2. Layer 0 input_layernorm (+ group sums when rows > 1) -> normed_buf, sums_buf
        if (use_qmv_fused_1) {
            self.dispatchRmsNorm1(&enc, self.h_buf.view(0), self.w.layers[0].input_norm, self.normed_buf.view(0));
        } else {
            enc.setPipeline(self.p.rms_norm_sums);
            enc.setBuffer(self.h_buf, 0, 0);
            enc.setRef(self.w.layers[0].input_norm, 1);
            enc.setBuffer(self.normed_buf, 0, 2);
            enc.setBuffer(self.sums_buf, 0, 3);
            enc.setValue(NormParams{
                .rows = rows,
                .hidden = @intCast(c.hidden_size),
                .sums_stride = self.sums_stride,
                .eps = c.rms_eps,
            }, 4);
            enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        }
        enc.barrier();

        var lin_idx: usize = 0;
        var full_idx: usize = 0;
        var lfm_conv_idx: usize = 0;
        var tap_cursor: usize = 0;
        const cur_max_seq: u32 = self.pos + rows;

        for (0..c.num_layers) |li| {
            const lw = self.w.layers[li];

            if (lw.kind == .linear) {
                const state_ref = self.gdn_state_buf.view(lin_idx * self.gdn_state_stride);
                const hist_ref = self.gdn_hist_buf.view(lin_idx * self.gdn_hist_stride);
                const hist_alt_ref = self.gdn_hist_alt_buf.view(lin_idx * self.gdn_hist_stride);
                // When not committing immediately (speculative verify) or capturing tape, each linear layer gets its own slice;
                // when committing immediately without tape capture, reuse slice 0 to stay hot in SLC!
                const vbase = (if (commit_immediately and !self.tape_capture) 0 else lin_idx) * self.gdn_vl.layer_stride;
                const q_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.q_off);
                const k_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.k_off);
                const v_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.v_off);
                const dec_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.decay_off);
                const mix_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.mix_off);
                const qkv_save_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.qkv_off);
                lin_idx += 1;

                self.dispatchQk(&enc, self.normed_buf.view(0), lw.linear.in_proj_all, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                if (use_fused_gdn_1) {
                    const gdn_fp = GdnFusedStepParams{
                        .nk = @intCast(c.gdn_k_heads),
                        .nv = @intCast(c.gdn_v_heads),
                        .qkv_dim = gdn_qkv_dim,
                        .in_proj_stride = gdn_all_stride,
                        .z_offset = gdn_qkv_dim,
                        .b_offset = gdn_qkv_dim + gdn_val_dim,
                        .a_offset = gdn_qkv_dim + gdn_val_dim + @as(u32, @intCast(c.gdn_v_heads)),
                        .sums_stride = self.sums_stride,
                        .write_sums = if (use_qmv_1) 0 else 1,
                        .eps = c.rms_eps,
                    };
                    if (c.gdn_v_heads >= 32) {
                        // Super-fused SRAM-resident GDN step: conv + scan + norm_gate in 1 dispatch!
                        enc.setPipeline(self.p.gdn_fused_step_1.?);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(hist_ref, 1);
                        enc.setRef(hist_alt_ref, 2);
                        enc.setRef(lw.linear.conv_weight, 3);
                        enc.setRef(lw.linear.a_log, 4);
                        enc.setRef(lw.linear.dt_bias, 5);
                        enc.setRef(lw.linear.gdn_norm, 6);
                        enc.setRef(state_ref, 7);
                        enc.setBuffer(self.act_buf, 0, 8);
                        enc.setBuffer(self.sums_buf, 0, 9);
                        enc.setValue(gdn_fp, 10);
                        enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                        enc.barrier();
                    } else {
                        // High-occupancy 4x-split GDN conv+scan for nv < 32 (64 TGs -> all 40 cores lit up!)
                        enc.setPipeline(self.p.gdn_conv_scan_split_1.?);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(hist_ref, 1);
                        enc.setRef(hist_alt_ref, 2);
                        enc.setRef(lw.linear.conv_weight, 3);
                        enc.setRef(lw.linear.a_log, 4);
                        enc.setRef(lw.linear.dt_bias, 5);
                        enc.setRef(state_ref, 6);
                        enc.setBuffer(self.gdn_scan_out_buf, 0, 7);
                        enc.setValue(gdn_fp, 8);
                        enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = 4, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();

                        enc.setPipeline(self.p.gdn_norm_gate_sums);
                        enc.setBuffer(self.gdn_scan_out_buf, 0, 0);
                        enc.setBuffer(self.proj_big_buf, 0, 1);
                        enc.setRef(lw.linear.gdn_norm, 2);
                        enc.setBuffer(self.act_buf, 0, 3);
                        enc.setBuffer(self.sums_buf, 0, 4);
                        enc.setValue(GdnNormParams{
                            .rows = 1,
                            .nv = @intCast(c.gdn_v_heads),
                            .dv = @intCast(c.gdn_v_dim),
                            .in_proj_stride = gdn_all_stride,
                            .z_offset = gdn_qkv_dim,
                            .sums_stride = norm_sums_stride,
                            .eps = c.rms_eps,
                        }, 5);
                        enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = 1, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();
                    }
                } else {
                    enc.setPipeline(self.p.gdn_conv.?);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(hist_ref, 1);
                    enc.setRef(lw.linear.conv_weight, 2);
                    enc.setRef(lw.linear.a_log, 3);
                    enc.setRef(lw.linear.dt_bias, 4);
                    enc.setBuffer(self.conv_windows_buf, 0, 5);
                    enc.setValue(GdnConvParams{
                        .rows = rows,
                        .nk = @intCast(c.gdn_k_heads),
                        .nv = @intCast(c.gdn_v_heads),
                        .qkv_dim = gdn_qkv_dim,
                        .in_proj_stride = gdn_all_stride,
                        .b_offset = gdn_qkv_dim + gdn_val_dim,
                        .a_offset = gdn_qkv_dim + gdn_val_dim + @as(u32, @intCast(c.gdn_v_heads)),
                        .commit_single = if (commit_immediately and rows == 1) 1 else 0,
                        .save_qkv = if (!commit_immediately or self.tape_capture) 1 else 0,
                    }, 6);
                    enc.setRef(q_ref, 7);
                    enc.setRef(k_ref, 8);
                    enc.setRef(v_ref, 9);
                    enc.setRef(dec_ref, 10);
                    enc.setRef(mix_ref, 11);
                    enc.setRef(qkv_save_ref, 12);
                    enc.dispatchGroups(.{ .width = 1, .height = 2 * c.gdn_k_heads + c.gdn_v_heads, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();

                    if (self.tape_capture and scan_slots == 0) {
                        const corr_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.corr_off);
                        enc.setPipeline(self.p.gdn_scan_record_corr.?);
                        enc.setRef(q_ref, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setRef(dec_ref, 3);
                        enc.setRef(mix_ref, 4);
                        enc.setRef(state_ref, 5);
                        enc.setBuffer(self.scan_steps_buf, 0, 6);
                        enc.setValue(GdnScanParams{
                            .rows = rows,
                            .nk = @intCast(c.gdn_k_heads),
                            .nv = @intCast(c.gdn_v_heads),
                            .commit_state = if (commit_immediately) 1 else 0,
                        }, 7);
                        enc.setBuffer(self.gdn_scan_out_buf, 0, 8);
                        enc.setRef(corr_ref, 9);
                        enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });
                    } else {
                        enc.setPipeline(scan_pso.?);
                        enc.setRef(q_ref, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setRef(dec_ref, 3);
                        enc.setRef(mix_ref, 4);
                        enc.setRef(state_ref, 5);
                        enc.setBuffer(self.scan_steps_buf, 0, 6);
                        enc.setValue(GdnScanParams{
                            .rows = rows,
                            .nk = @intCast(c.gdn_k_heads),
                            .nv = @intCast(c.gdn_v_heads),
                            .commit_state = if (commit_immediately) 1 else 0,
                        }, 7);
                        enc.setBuffer(self.gdn_scan_out_buf, 0, 8);
                        enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });
                    }

                    if (commit_immediately and rows > 1) {
                        enc.setPipeline(self.p.gdn_commit_history.?);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(hist_ref, 1);
                        enc.setValue(GdnHistCommitParams{
                            .qkv_dim = gdn_qkv_dim,
                            .in_proj_stride = gdn_all_stride,
                            .src0 = hist_src0,
                            .src1 = hist_src1,
                            .src2 = hist_src2,
                        }, 2);
                        enc.dispatchThreads(.{ .width = gdn_qkv_dim / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    }
                    enc.barrier();

                    enc.setPipeline(self.p.gdn_norm_gate_sums);
                    enc.setBuffer(self.gdn_scan_out_buf, 0, 0);
                    enc.setBuffer(self.proj_big_buf, 0, 1);
                    enc.setRef(lw.linear.gdn_norm, 2);
                    enc.setBuffer(self.act_buf, 0, 3);
                    enc.setBuffer(self.sums_buf, 0, 4);
                    enc.setValue(GdnNormParams{
                        .rows = rows,
                        .nv = @intCast(c.gdn_v_heads),
                        .dv = @intCast(c.gdn_v_dim),
                        .in_proj_stride = gdn_all_stride,
                        .z_offset = gdn_qkv_dim,
                        .sums_stride = self.sums_stride,
                        .eps = c.rms_eps,
                    }, 5);
                    enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();
                }

                if (use_qmv_fused_1) {
                    self.dispatchQkOpt(&enc, self.act_buf.view(0), lw.linear.out_proj, self.sums_buf.view(0), self.h_buf.view(0), rows, 1, true);
                } else {
                    self.dispatchQk(&enc, self.act_buf.view(0), lw.linear.out_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                }
                enc.barrier();
            } else if (lw.kind == .short_conv) {
                const sc = lw.short_conv;
                const hist_ref = self.lfm_hist_buf.view(lfm_conv_idx * self.lfm_hist_stride);
                lfm_conv_idx += 1;

                self.dispatchQk(&enc, self.normed_buf.view(0), sc.in_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                enc.setPipeline(self.p.lfm2_short_conv_sums);
                enc.setBuffer(self.proj_big_buf, 0, 0);
                enc.setRef(sc.in_bias orelse self.w.final_norm, 1);
                enc.setRef(sc.conv_weight, 2);
                enc.setRef(sc.conv_bias orelse self.w.final_norm, 3);
                enc.setRef(hist_ref, 4);
                enc.setBuffer(self.act_buf, 0, 5);
                enc.setBuffer(self.sums_buf, 0, 6);
                enc.setValue(LfmConvParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .l_cache = @intCast(c.conv_l_cache),
                    .sums_stride = self.sums_stride,
                    .has_in_bias = if (sc.in_bias != null) 1 else 0,
                    .has_conv_bias = if (sc.conv_bias != null) 1 else 0,
                    .commit_hist = if (commit_immediately) 1 else 0,
                }, 7);
                enc.dispatchGroups(.{ .width = c.hidden_size / 64, .height = 1, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                if (use_res_fusion and sc.out_bias == null) {
                    self.dispatchQkOpt(&enc, self.act_buf.view(0), sc.out_proj, self.sums_buf.view(0), self.h_buf.view(0), rows, 1, true);
                } else {
                    self.dispatchQk(&enc, self.act_buf.view(0), sc.out_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                    if (sc.out_bias) |ob| {
                        enc.barrier();
                        enc.setPipeline(self.p.add_bias_bf16);
                        enc.setBuffer(self.delta_buf, 0, 0);
                        enc.setRef(ob, 1);
                        enc.setValue(AddBiasParams{ .rows = rows, .cols = @intCast(c.hidden_size) }, 2);
                        enc.dispatchThreads(.{ .width = c.hidden_size, .height = rows, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
                    }
                    if (use_res_fusion and sc.out_bias != null) {
                        // Fall through cleanly: we must add delta_buf to h_buf if use_res_fusion expects h_buf updated!
                        // Handled below by forcing add_rms_norm_sums when sc.out_bias != null.
                    }
                }
                enc.barrier();
            } else if (c.use_mla) {
                const k_ref = self.k_cache_buf.view(full_idx * self.kv_layer_stride);
                const v_ref = self.v_cache_buf.view(full_idx * self.kv_layer_stride);
                full_idx += 1;
                const ml = lw.mla;

                // 1. Project x -> kv_a (and q_a or direct q_proj) while sums_buf still holds normed_buf sums
                self.dispatchQk(&enc, self.normed_buf.view(0), ml.kv_a_proj, self.sums_buf.view(0), self.mla_kv_a_buf.view(0), rows, 1);
                enc.barrier();
                if (ml.has_q_lora) {
                    self.dispatchQk(&enc, self.normed_buf.view(0), ml.q_a_proj, self.sums_buf.view(0), self.mla_q_raw_buf.view(0), rows, 1);
                } else {
                    self.dispatchQk(&enc, self.normed_buf.view(0), ml.q_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                }
                enc.barrier();

                // 2. Q path: optional q_a_layernorm -> q_b_proj -> RoPE on q_pe into attn_q_buf
                if (ml.has_q_lora) {
                    enc.setPipeline(self.p.mla_rms_norm_sums);
                    enc.setBuffer(self.mla_q_raw_buf, 0, 0);
                    enc.setRef(ml.q_a_bias orelse self.w.final_norm, 1);
                    enc.setRef(ml.q_a_norm, 2);
                    enc.setBuffer(self.mla_q_norm_buf, 0, 3);
                    enc.setBuffer(self.sums_buf, 0, 4);
                    enc.setValue(MlaNormParams{
                        .rows = rows,
                        .dim = @intCast(c.q_lora_rank),
                        .sums_stride = self.sums_stride,
                        .has_bias = if (ml.q_a_bias != null) 1 else 0,
                        .eps = c.rms_eps,
                    }, 5);
                    enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                    enc.barrier();

                    self.dispatchQk(&enc, self.mla_q_norm_buf.view(0), ml.q_b_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                    enc.barrier();
                }

                enc.setPipeline(self.p.mla_q_rope);
                enc.setBuffer(self.proj_big_buf, 0, 0);
                enc.setBuffer(self.inv_freq_buf, 0, 1);
                enc.setBuffer(self.positions_buf, 0, 2);
                enc.setBuffer(self.attn_q_buf, 0, 3);
                enc.setValue(MlaQPrepParams{
                    .rows = rows,
                    .num_heads = @intCast(c.num_heads),
                    .qk_nope_dim = @intCast(c.qk_nope_head_dim),
                    .qk_rope_dim = @intCast(c.qk_rope_head_dim),
                    .rope_interleaved = if (c.rope_interleaved) 1 else 0,
                    .rope_mscale = c.rope_mscale,
                }, 4);
                enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });

                // 3. KV path: RMSNorm(kv_latent) + RoPE(k_pe) -> kv_b_proj -> scatter into K/V cache
                enc.setPipeline(self.p.mla_kv_a_norm_rope);
                enc.setBuffer(self.mla_kv_a_buf, 0, 0);
                enc.setRef(ml.kv_a_bias orelse self.w.final_norm, 1);
                enc.setRef(ml.kv_a_norm, 2);
                enc.setBuffer(self.inv_freq_buf, 0, 3);
                enc.setBuffer(self.positions_buf, 0, 4);
                enc.setBuffer(self.mla_kv_lat_buf, 0, 5);
                enc.setBuffer(self.mla_kpe_buf, 0, 6);
                enc.setBuffer(self.sums_buf, 0, 7);
                enc.setValue(MlaKvAParams{
                    .rows = rows,
                    .kv_lora_rank = @intCast(c.kv_lora_rank),
                    .qk_rope_dim = @intCast(c.qk_rope_head_dim),
                    .sums_stride = self.sums_stride,
                    .has_bias = if (ml.kv_a_bias != null) 1 else 0,
                    .rope_interleaved = if (c.rope_interleaved) 1 else 0,
                    .eps = c.rms_eps,
                    .rope_mscale = c.rope_mscale,
                }, 8);
                enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();

                self.dispatchQk(&enc, self.mla_kv_lat_buf.view(0), ml.kv_b_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                const kv_row_stride: u32 = @intCast(c.maxKvWidth());
                enc.setPipeline(self.p.mla_kv_b_scatter);
                enc.setBuffer(self.proj_big_buf, 0, 0);
                enc.setBuffer(self.mla_kpe_buf, 0, 1);
                enc.setBuffer(self.kv_slots_buf, 0, 2);
                enc.setRef(k_ref, 3);
                enc.setRef(v_ref, 4);
                enc.setValue(MlaScatterParams{
                    .rows = rows,
                    .num_heads = @intCast(c.num_heads),
                    .qk_nope_dim = @intCast(c.qk_nope_head_dim),
                    .qk_rope_dim = @intCast(c.qk_rope_head_dim),
                    .v_dim = @intCast(c.v_head_dim),
                    .kv_row_stride = kv_row_stride,
                }, 5);
                enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                // 4. MLA FlashAttention (distinct qk_dim and v_dim) + o_proj
                enc.setPipeline(self.p.mla_flash_attn);
                enc.setBuffer(self.attn_q_buf, 0, 0);
                enc.setRef(k_ref, 1);
                enc.setRef(v_ref, 2);
                enc.setBuffer(self.seq_lens_buf, 0, 3);
                enc.setBuffer(self.tree_masks_buf, 0, 4);
                enc.setBuffer(self.act_buf, 0, 5);
                enc.setBuffer(self.sums_buf, 0, 6);
                enc.setValue(MlaFlashParams{
                    .rows = rows,
                    .num_heads = @intCast(c.num_heads),
                    .qk_dim = @intCast(c.mlaQHeadDim()),
                    .v_dim = @intCast(c.v_head_dim),
                    .kv_row_stride = kv_row_stride,
                    .sums_stride = self.sums_stride,
                    .base_pos = self.pos,
                    .is_tree = if (is_tree) 1 else 0,
                    .scale = c.mla_attn_scale,
                }, 7);
                enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();

                if (use_res_fusion) {
                    self.dispatchQkOpt(&enc, self.act_buf.view(0), ml.o_proj, self.sums_buf.view(0), self.h_buf.view(0), rows, 1, true);
                } else {
                    self.dispatchQk(&enc, self.act_buf.view(0), ml.o_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                }
                enc.barrier();
            } else {
                const k_ref = self.k_cache_buf.view(full_idx * self.kv_layer_stride);
                const v_ref = self.v_cache_buf.view(full_idx * self.kv_layer_stride);
                full_idx += 1;

                self.dispatchQk(&enc, self.normed_buf.view(0), lw.full.qkv_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                const l_hd = c.layerHeadDim(li);
                const l_kv = c.layerKvHeads(li);
                const l_sw: u32 = if (c.layer_is_sliding[li]) @intCast(c.sliding_window) else 0;
                const l_qkv_stride: u32 = @intCast(c.layerQkvStride(li));
                const l_k_eq_v = c.layer_k_eq_v[li];
                const l_rope_dims: u32 = if (c.is_gemma4) @intCast(l_hd) else @intCast(c.rope_dims);
                const l_inv_freq = if (c.has_dual_rope and c.layer_is_sliding[li]) self.inv_freq_local_buf else self.inv_freq_buf;
                const scale: f32 = if (c.is_gemma4)
                    1.0
                else if (c.query_pre_attn_scalar > 0.0)
                    1.0 / @sqrt(c.query_pre_attn_scalar)
                else
                    1.0 / @sqrt(@as(f32, @floatFromInt(l_hd)));

                if (c.arch == .qwen2_dense and l_hd == 128 and c.rope_dims == 128 and c.attn_softcap == 0.0 and !c.has_v_norm and !l_k_eq_v and !c.has_dual_rope and !c.is_gemma and !c.rope_interleaved) {
                    const bias_ref = lw.full.qkv_bias orelse self.w.final_norm;
                    if (rows == 1 and commit_immediately and use_qmv_fused_1 and !is_tree) {
                        // Super-fused Bias + optional Q/K norm + RoPE + KV-store + Flash GQA in ONE dispatch!
                        enc.setPipeline(self.p.attn_std128_decode_1);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(bias_ref, 1);
                        enc.setBuffer(l_inv_freq, 0, 2);
                        enc.setRef(k_ref, 3);
                        enc.setRef(v_ref, 4);
                        enc.setBuffer(self.act_buf, 0, 5);
                        enc.setValue(Std128Decode1Params{
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(l_kv),
                            .qkv_stride = l_qkv_stride,
                            .has_bias = if (lw.full.qkv_bias != null) 1 else 0,
                            .use_rope = if (c.use_rope) 1 else 0,
                            .pos = self.pos,
                            .scale = scale,
                            .has_qk_norm = if (c.has_qk_norm) 1 else 0,
                            .sliding_window = l_sw,
                            .eps = c.rms_eps,
                        }, 6);
                        enc.setRef(lw.full.q_norm, 7);
                        enc.setRef(lw.full.k_norm, 8);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    } else {
                        enc.setPipeline(self.p.attn_std128_rope_store);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(bias_ref, 1);
                        enc.setBuffer(l_inv_freq, 0, 2);
                        enc.setBuffer(self.positions_buf, 0, 3);
                        enc.setBuffer(self.kv_slots_buf, 0, 4);
                        enc.setBuffer(self.attn_q_buf, 0, 5);
                        enc.setRef(k_ref, 6);
                        enc.setRef(v_ref, 7);
                        enc.setValue(Std128PrepParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(l_kv),
                            .qkv_stride = l_qkv_stride,
                            .has_bias = if (lw.full.qkv_bias != null) 1 else 0,
                            .use_rope = if (c.use_rope) 1 else 0,
                            .has_qk_norm = if (c.has_qk_norm) 1 else 0,
                            .eps = c.rms_eps,
                        }, 8);
                        enc.setRef(lw.full.q_norm, 9);
                        enc.setRef(lw.full.k_norm, 10);
                        enc.dispatchGroups(.{ .width = c.num_heads + l_kv, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();

                        const fp = AttnFlashParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(l_kv),
                            .qkv_stride = l_qkv_stride,
                            .sums_stride = norm_sums_stride,
                            .scale = scale,
                            .splits = 1,
                            .base_pos = self.pos,
                            .is_tree = if (is_tree) 1 else 0,
                            .sliding_window = l_sw,
                            .softcap = c.attn_softcap,
                        };
                        enc.setPipeline(self.p.attn_std128_flash);
                        enc.setBuffer(self.attn_q_buf, 0, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setBuffer(self.seq_lens_buf, 0, 3);
                        enc.setBuffer(self.tree_masks_buf, 0, 4);
                        enc.setBuffer(self.act_buf, 0, 5);
                        enc.setBuffer(self.sums_buf, 0, 6);
                        enc.setValue(fp, 7);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    }
                } else if (c.arch == .qwen2_dense or !c.attn_output_gate) {
                    // Universal general-HD path (HD = 64..512; optional Q/K/V norm, k_eq_v, bias, dual RoPE, sliding_window, softcap)
                    const bias_ref = lw.full.qkv_bias orelse self.w.final_norm;
                    enc.setPipeline(self.p.attn_gen_rope_store);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(bias_ref, 1);
                    enc.setRef(lw.full.q_norm, 2);
                    enc.setRef(lw.full.k_norm, 3);
                    enc.setBuffer(l_inv_freq, 0, 4);
                    enc.setBuffer(self.positions_buf, 0, 5);
                    enc.setBuffer(self.kv_slots_buf, 0, 6);
                    enc.setBuffer(self.attn_q_buf, 0, 7);
                    enc.setRef(k_ref, 8);
                    enc.setRef(v_ref, 9);
                    enc.setValue(GenPrepParams{
                        .rows = rows,
                        .num_heads = @intCast(c.num_heads),
                        .num_kv_heads = @intCast(l_kv),
                        .head_dim = @intCast(l_hd),
                        .rope_dims = l_rope_dims,
                        .qkv_stride = l_qkv_stride,
                        .has_bias = if (lw.full.qkv_bias != null) 1 else 0,
                        .has_qk_norm = if (c.has_qk_norm) 1 else 0,
                        .use_rope = if (c.use_rope) 1 else 0,
                        .has_v_norm = if (c.has_v_norm) 1 else 0,
                        .k_eq_v = if (l_k_eq_v) 1 else 0,
                        .rope_interleaved = if (c.rope_interleaved) 1 else 0,
                        .eps = c.rms_eps,
                    }, 10);
                    enc.dispatchGroups(.{ .width = c.num_heads + l_kv, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();

                    enc.setPipeline(self.p.attn_gen_flash);
                    enc.setBuffer(self.attn_q_buf, 0, 0);
                    enc.setRef(k_ref, 1);
                    enc.setRef(v_ref, 2);
                    enc.setBuffer(self.seq_lens_buf, 0, 3);
                    enc.setBuffer(self.tree_masks_buf, 0, 4);
                    enc.setBuffer(self.act_buf, 0, 5);
                    enc.setBuffer(self.sums_buf, 0, 6);
                    enc.setValue(GenFlashParams{
                        .rows = rows,
                        .num_heads = @intCast(c.num_heads),
                        .num_kv_heads = @intCast(l_kv),
                        .head_dim = @intCast(l_hd),
                        .sums_stride = norm_sums_stride,
                        .base_pos = self.pos,
                        .is_tree = if (is_tree) 1 else 0,
                        .sliding_window = l_sw,
                        .scale = scale,
                        .softcap = c.attn_softcap,
                    }, 7);
                    enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    enc.barrier();
                } else {
                    if (rows == 1 and commit_immediately and use_qmv_fused_1 and !is_tree and (!use_split_k or cur_max_seq <= 256)) {
                        // Super-fused Q/K norm + partial RoPE + KV-store + Flash GQA + sigmoid(gate) in ONE dispatch!
                        enc.setPipeline(self.p.attn_qwen35_decode_1);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(lw.full.q_norm, 1);
                        enc.setRef(lw.full.k_norm, 2);
                        enc.setBuffer(self.inv_freq_buf, 0, 3);
                        enc.setRef(k_ref, 4);
                        enc.setRef(v_ref, 5);
                        enc.setBuffer(self.act_buf, 0, 6);
                        enc.setValue(Qwen35Decode1Params{
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .pos = self.pos,
                            .eps = c.rms_eps,
                            .scale = scale,
                        }, 7);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    } else {
                        enc.setPipeline(self.p.attn_qk_rope_store);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(lw.full.q_norm, 1);
                        enc.setRef(lw.full.k_norm, 2);
                        enc.setBuffer(self.inv_freq_buf, 0, 3);
                        enc.setBuffer(self.positions_buf, 0, 4);
                        enc.setBuffer(self.kv_slots_buf, 0, 5);
                        enc.setBuffer(self.attn_q_buf, 0, 6);
                        enc.setRef(k_ref, 7);
                        enc.setRef(v_ref, 8);
                        enc.setValue(AttnPrepParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .eps = c.rms_eps,
                        }, 9);
                        enc.dispatchGroups(.{ .width = c.num_heads + c.num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();

                        if (rows == 1 and use_split_k and cur_max_seq > 256 and !is_tree) {
                            const splits: u32 = @min(max_splits, @max(4, (cur_max_seq + 63) / 64));
                            const fp = AttnFlashParams{
                                .rows = rows,
                                .num_heads = @intCast(c.num_heads),
                                .num_kv_heads = @intCast(c.num_kv_heads),
                                .qkv_stride = attn_qkv_stride,
                                .sums_stride = self.sums_stride,
                                .scale = scale,
                                .splits = splits,
                                .base_pos = self.pos,
                                .is_tree = 0,
                            };
                            enc.setPipeline(self.p.attn_flash_split);
                            enc.setBuffer(self.attn_q_buf, 0, 0);
                            enc.setRef(k_ref, 1);
                            enc.setRef(v_ref, 2);
                            enc.setBuffer(self.seq_lens_buf, 0, 3);
                            enc.setBuffer(self.partial_acc_buf, 0, 4);
                            enc.setBuffer(self.partial_ml_buf, 0, 5);
                            enc.setValue(fp, 6);
                            enc.dispatchGroups(.{ .width = splits, .height = c.num_heads, .depth = rows }, .{ .width = 128, .height = 1, .depth = 1 });
                            enc.barrier();

                            enc.setPipeline(self.p.attn_merge_gate_sums);
                            enc.setBuffer(self.partial_acc_buf, 0, 0);
                            enc.setBuffer(self.partial_ml_buf, 0, 1);
                            enc.setBuffer(self.proj_big_buf, 0, 2);
                            enc.setBuffer(self.act_buf, 0, 3);
                            enc.setBuffer(self.sums_buf, 0, 4);
                            enc.setValue(fp, 5);
                            enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                            enc.barrier();
                        } else {
                            enc.setPipeline(self.p.attn_flash_fused);
                            enc.setBuffer(self.attn_q_buf, 0, 0);
                            enc.setBuffer(self.proj_big_buf, 0, 1);
                            enc.setRef(k_ref, 2);
                            enc.setRef(v_ref, 3);
                            enc.setBuffer(self.seq_lens_buf, 0, 4);
                            enc.setBuffer(self.tree_masks_buf, 0, 5);
                            enc.setBuffer(self.act_buf, 0, 6);
                            enc.setBuffer(self.sums_buf, 0, 7);
                            enc.setValue(AttnFlashParams{
                                .rows = rows,
                                .num_heads = @intCast(c.num_heads),
                                .num_kv_heads = @intCast(c.num_kv_heads),
                                .qkv_stride = attn_qkv_stride,
                                .sums_stride = self.sums_stride,
                                .scale = scale,
                                .splits = 1,
                                .base_pos = self.pos,
                                .is_tree = if (is_tree) 1 else 0,
                            }, 8);
                            enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                            enc.barrier();
                        }
                    }
                }

                if (use_res_fusion) {
                    self.dispatchQkOpt(&enc, self.act_buf.view(0), lw.full.o_proj, self.sums_buf.view(0), self.h_buf.view(0), rows, 1, true);
                } else {
                    self.dispatchQk(&enc, self.act_buf.view(0), lw.full.o_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                }
                enc.barrier();
            }

            // Post-attention residual + norm (+ sums on multi-row)
            if (c.has_post_norms) {
                enc.setPipeline(self.p.norm_res_norm_sums);
                enc.setBuffer(self.h_buf, 0, 0);
                enc.setBuffer(self.delta_buf, 0, 1);
                enc.setRef(lw.post_attn_norm, 2);
                enc.setRef(lw.layer_scalar, 3);
                enc.setRef(lw.pre_ff_norm, 4);
                enc.setBuffer(self.normed_buf, 0, 5);
                enc.setBuffer(self.sums_buf, 0, 6);
                enc.setValue(PostNormResParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .sums_stride = norm_sums_stride,
                    .has_scalar = 0,
                    .eps = c.rms_eps,
                }, 7);
                enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            } else if (use_res_fusion) {
                self.dispatchRmsNorm1(&enc, self.h_buf.view(0), lw.post_attn_norm, self.normed_buf.view(0));
            } else {
                enc.setPipeline(self.p.add_rms_norm_sums);
                enc.setBuffer(self.h_buf, 0, 0);
                enc.setBuffer(self.delta_buf, 0, 1);
                enc.setRef(lw.post_attn_norm, 2);
                enc.setBuffer(self.normed_buf, 0, 3);
                enc.setBuffer(self.sums_buf, 0, 4);
                enc.setValue(NormParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .sums_stride = norm_sums_stride,
                    .eps = c.rms_eps,
                }, 5);
                enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            }
            enc.barrier();

            // MLP (Dense SwiGLU OR Sparse SwiGLU MoE + optional gated shared expert)
            if (lw.is_moe) {
                const mw = lw.swiglu_moe;
                // 1. Router logits -> softmax top-K selection
                enc.setPipeline(self.p.moe_gate_logits.?);
                enc.setBuffer(self.normed_buf, 0, 0);
                enc.setRef(mw.gate_weight, 1);
                enc.setBuffer(self.moe_logits_buf, 0, 2);
                enc.setValue(GateParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .n_experts = @intCast(c.n_routed_experts),
                }, 3);
                enc.dispatchGroups(.{ .width = (c.n_routed_experts + 3) / 4, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(self.p.moe_topk_select.?);
                enc.setBuffer(self.moe_logits_buf, 0, 0);
                enc.setRef(mw.gate_bias orelse self.w.final_norm, 1);
                enc.setBuffer(self.moe_inds_buf, 0, 2);
                enc.setBuffer(self.moe_weights_buf, 0, 3);
                enc.setValue(TopkParams{
                    .rows = rows,
                    .n_experts = @intCast(c.n_routed_experts),
                    .top_k = @intCast(c.num_experts_per_tok),
                    .norm_topk_prob = if (c.norm_topk_prob) 1 else 0,
                    .routed_scaling_factor = c.routed_scaling_factor,
                    .use_softmax = if (c.moe_use_sigmoid) 0 else 1,
                    .has_e_bias = if (mw.gate_bias != null) 1 else 0,
                    .n_group = @intCast(c.n_group),
                    .topk_group = @intCast(c.topk_group),
                }, 4);
                enc.dispatchGroups(.{ .width = 1, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();

                // 2. Top-K routed SwiGLU experts (gate+up -> silu(gate)*up -> down)
                const gu_p = MoeSwigluParams{
                    .rows = rows,
                    .top_k = @intCast(c.num_experts_per_tok),
                    .K = @intCast(c.hidden_size),
                    .N = @intCast(c.moe_intermediate_size),
                    .bits = mw.bits,
                    .gs = mw.gs,
                    .sb_dtype = @intFromEnum(mw.sb_dtype),
                    .x_per_expert = 0,
                    .code_stride = mw.gu_code_stride,
                    .sb_stride = mw.gu_sb_stride,
                };
                const dn_p = MoeSwigluParams{
                    .rows = rows,
                    .top_k = @intCast(c.num_experts_per_tok),
                    .K = @intCast(c.moe_intermediate_size),
                    .N = @intCast(c.hidden_size),
                    .bits = mw.bits,
                    .gs = mw.gs,
                    .sb_dtype = @intFromEnum(mw.sb_dtype),
                    .x_per_expert = 1,
                    .code_stride = mw.dn_code_stride,
                    .sb_stride = mw.dn_sb_stride,
                };
                switch (mw.storage) {
                    .qk4 => {
                        enc.setPipeline(self.p.moe_swiglu_gu_qk4.?);
                        enc.setBuffer(self.normed_buf, 0, 0);
                        enc.setRef(mw.gate_codes, 1);
                        enc.setRef(mw.up_codes, 2);
                        enc.setBuffer(self.moe_inds_buf, 0, 3);
                        enc.setBuffer(self.moe_act_buf, 0, 4);
                        enc.setValue(gu_p, 5);
                        enc.dispatchGroups(.{ .width = c.moe_intermediate_size / 32, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();

                        enc.setPipeline(self.p.moe_expert_qmv.?);
                        enc.setBuffer(self.moe_act_buf, 0, 0);
                        enc.setRef(mw.down_codes, 1);
                        enc.setBuffer(self.moe_inds_buf, 0, 2);
                        enc.setBuffer(self.moe_out_buf, 0, 3);
                        enc.setValue(MoeExpertParams{
                            .rows = rows,
                            .top_k = @intCast(c.num_experts_per_tok),
                            .K = @intCast(c.moe_intermediate_size),
                            .N = @intCast(c.hidden_size),
                            .G = @intCast(c.moe_intermediate_size / 64),
                            .apply_relu2 = 0,
                            .x_per_expert = 1,
                            .expert_stride = mw.dn_code_stride,
                        }, 4);
                        enc.dispatchGroups(.{ .width = c.hidden_size / 32, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();
                    },
                    .affine => {
                        enc.setPipeline(self.p.moe_swiglu_gu_affine.?);
                        enc.setBuffer(self.normed_buf, 0, 0);
                        enc.setRef(mw.gate_codes, 1);
                        enc.setRef(mw.gate_scales, 2);
                        enc.setRef(mw.gate_biases, 3);
                        enc.setRef(mw.up_codes, 4);
                        enc.setRef(mw.up_scales, 5);
                        enc.setRef(mw.up_biases, 6);
                        enc.setBuffer(self.moe_inds_buf, 0, 7);
                        enc.setBuffer(self.moe_act_buf, 0, 8);
                        enc.setValue(gu_p, 9);
                        enc.dispatchGroups(.{ .width = (c.moe_intermediate_size + 3) / 4, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();

                        enc.setPipeline(self.p.moe_swiglu_dn_affine.?);
                        enc.setBuffer(self.moe_act_buf, 0, 0);
                        enc.setRef(mw.down_codes, 1);
                        enc.setRef(mw.down_scales, 2);
                        enc.setRef(mw.down_biases, 3);
                        enc.setBuffer(self.moe_inds_buf, 0, 4);
                        enc.setBuffer(self.moe_out_buf, 0, 5);
                        enc.setValue(dn_p, 6);
                        enc.dispatchGroups(.{ .width = (c.hidden_size + 3) / 4, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    },
                    .bf16 => {
                        enc.setPipeline(self.p.moe_swiglu_gu_bf16.?);
                        enc.setBuffer(self.normed_buf, 0, 0);
                        enc.setRef(mw.gate_codes, 1);
                        enc.setRef(mw.up_codes, 2);
                        enc.setBuffer(self.moe_inds_buf, 0, 3);
                        enc.setBuffer(self.moe_act_buf, 0, 4);
                        enc.setValue(gu_p, 5);
                        enc.dispatchGroups(.{ .width = (c.moe_intermediate_size + 3) / 4, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();

                        enc.setPipeline(self.p.moe_swiglu_dn_bf16.?);
                        enc.setBuffer(self.moe_act_buf, 0, 0);
                        enc.setRef(mw.down_codes, 1);
                        enc.setBuffer(self.moe_inds_buf, 0, 2);
                        enc.setBuffer(self.moe_out_buf, 0, 3);
                        enc.setValue(dn_p, 4);
                        enc.dispatchGroups(.{ .width = (c.hidden_size + 3) / 4, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    },
                }

                // 3. Optional SwiGLU shared expert (+ optional Qwen3.5 scalar sigmoid gate)
                if (mw.has_shared) {
                    const sh_inter: u32 = mw.shared_gate_up.rowsOut() / 2;
                    self.dispatchQk(&enc, self.normed_buf.view(0), mw.shared_gate_up, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                    enc.barrier();

                    enc.setPipeline(self.p.gu_activate_sums);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setBuffer(self.act_buf, 0, 1);
                    enc.setBuffer(self.sums_buf, 0, 2);
                    enc.setValue(MlpActParams{
                        .rows = rows,
                        .inter = sh_inter,
                        .sums_stride = self.sums_stride,
                    }, 3);
                    enc.dispatchGroups(.{ .width = sh_inter / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();

                    self.dispatchQk(&enc, self.act_buf.view(0), mw.shared_down, self.sums_buf.view(0), self.moe_shared_out_buf.view(0), rows, 1);
                    if (mw.shared_gate_vec) |sgv| {
                        enc.setPipeline(self.p.moe_shared_gate_scalar.?);
                        enc.setBuffer(self.normed_buf, 0, 0);
                        enc.setRef(sgv, 1);
                        enc.setBuffer(self.moe_shared_gate_buf, 0, 2);
                        enc.setValue(GateParams{
                            .rows = rows,
                            .hidden = @intCast(c.hidden_size),
                            .n_experts = 1,
                        }, 3);
                        enc.dispatchGroups(.{ .width = 1, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    }
                    enc.barrier();
                }

                // 4. Weighted expert combine (+ shared expert + optional residual add)
                enc.setPipeline(self.p.moe_combine_swiglu.?);
                enc.setBuffer(self.moe_out_buf, 0, 0);
                enc.setBuffer(self.moe_weights_buf, 0, 1);
                enc.setBuffer(self.moe_shared_out_buf, 0, 2);
                enc.setBuffer(self.moe_shared_gate_buf, 0, 3);
                enc.setBuffer(if (use_res_fusion) self.h_buf else self.delta_buf, 0, 4);
                enc.setValue(MoeSwigluCombineParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .top_k = @intCast(c.num_experts_per_tok),
                    .has_shared = if (mw.has_shared) 1 else 0,
                    .shared_gated = if (mw.shared_gate_vec != null) 1 else 0,
                    .add_res = if (use_res_fusion) 1 else 0,
                }, 5);
                enc.dispatchThreads(.{ .width = c.hidden_size, .height = rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();
            } else {
            // MLP gate+up -> SwiGLU (or GeGLU for Gemma) -> down
            if (use_res_fusion and !c.use_geglu and lw.gate_up.isQk4()) {
                const gu_qk = lw.gate_up.qk4;
                const small_inter = (c.intermediate_size <= 4096);
                enc.setPipeline(if (small_inter) self.p.qmv_1_silu_mul_sk4 else self.p.qmv_1_silu_mul);
                enc.setBuffer(self.normed_buf, 0, 0);
                enc.setRef(gu_qk.ref, 1);
                enc.setBuffer(self.act_buf, 0, 2);
                enc.setValue(QkParams{
                    .M = 1,
                    .K = @intCast(gu_qk.k),
                    .N = @intCast(c.intermediate_size),
                    .G = @intCast(gu_qk.groups()),
                    .first_row = 0,
                    .sums_stride = 0,
                    .tile_first = 0,
                }, 3);
                if (small_inter) {
                    enc.dispatchGroups(.{ .width = c.intermediate_size / 16, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                } else {
                    enc.dispatchGroups(.{ .width = c.intermediate_size / 32, .height = 1, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                }
                enc.barrier();
            } else {
                self.dispatchQk(&enc, self.normed_buf.view(0), lw.gate_up, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, self.gu_parts);
                enc.barrier();

                enc.setPipeline(if (c.use_geglu) self.p.geglu_activate_sums else self.p.gu_activate_sums);
                enc.setBuffer(self.proj_big_buf, 0, 0);
                enc.setBuffer(self.act_buf, 0, 1);
                enc.setBuffer(self.sums_buf, 0, 2);
                enc.setValue(MlpActParams{
                    .rows = rows,
                    .inter = @intCast(c.intermediate_size),
                    .sums_stride = self.sums_stride,
                }, 3);
                enc.dispatchGroups(.{ .width = c.intermediate_size / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();
            }

            if (use_res_fusion) {
                self.dispatchQkOpt(&enc, self.act_buf.view(0), lw.down_proj, self.sums_buf.view(0), self.h_buf.view(0), rows, 1, true);
            } else {
                self.dispatchQk(&enc, self.act_buf.view(0), lw.down_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
            }
            enc.barrier();
            }

            // Post-MLP residual + next layer norm (or final_norm)
            if (li + 1 < c.num_layers) {
                if (c.has_post_norms) {
                    enc.setPipeline(self.p.norm_res_norm_sums);
                    enc.setBuffer(self.h_buf, 0, 0);
                    enc.setBuffer(self.delta_buf, 0, 1);
                    enc.setRef(lw.post_ff_norm, 2);
                    enc.setRef(lw.layer_scalar, 3);
                    enc.setRef(self.w.layers[li + 1].input_norm, 4);
                    enc.setBuffer(self.normed_buf, 0, 5);
                    enc.setBuffer(self.sums_buf, 0, 6);
                    enc.setValue(PostNormResParams{
                        .rows = rows,
                        .hidden = @intCast(c.hidden_size),
                        .sums_stride = norm_sums_stride,
                        .has_scalar = if (lw.has_layer_scalar) 1 else 0,
                        .eps = c.rms_eps,
                    }, 7);
                    enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                } else if (use_res_fusion) {
                    self.dispatchRmsNorm1(&enc, self.h_buf.view(0), self.w.layers[li + 1].input_norm, self.normed_buf.view(0));
                } else {
                    enc.setPipeline(self.p.add_rms_norm_sums);
                    enc.setBuffer(self.h_buf, 0, 0);
                    enc.setBuffer(self.delta_buf, 0, 1);
                    enc.setRef(self.w.layers[li + 1].input_norm, 2);
                    enc.setBuffer(self.normed_buf, 0, 3);
                    enc.setBuffer(self.sums_buf, 0, 4);
                    enc.setValue(NormParams{
                        .rows = rows,
                        .hidden = @intCast(c.hidden_size),
                        .sums_stride = norm_sums_stride,
                        .eps = c.rms_eps,
                    }, 5);
                    enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                }
                enc.barrier();

                // Capture hidden-state tap after specified layers for DFlash2 drafter
                if (self.taps_buf) |tb| {
                    if (tap_cursor < tap_layers.len and tap_layers[tap_cursor] == li) {
                        const u4_count: u32 = @intCast((@as(usize, rows) * c.hidden_size * 2) / 16);
                        enc.setPipeline(self.p.copy_u4);
                        enc.setBuffer(self.h_buf, 0, 0);
                        enc.setBuffer(tb, tap_cursor * self.taps_layer_stride, 1);
                        enc.setValue(CopyParams{ .count = u4_count }, 2);
                        enc.dispatchThreads(.{ .width = u4_count, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                        enc.barrier();
                        tap_cursor += 1;
                    }
                }
            } else if (want_logits) {
                const norm_rows: u32 = if (all_row_logits) rows else 1;
                const h_off: usize = if (all_row_logits) 0 else @as(usize, rows - 1) * c.hidden_size * 2;
                if (c.has_post_norms) {
                    enc.setPipeline(self.p.norm_res_norm_sums);
                    enc.setBuffer(self.h_buf, h_off, 0);
                    enc.setBuffer(self.delta_buf, h_off, 1);
                    enc.setRef(lw.post_ff_norm, 2);
                    enc.setRef(lw.layer_scalar, 3);
                    enc.setRef(self.w.final_norm, 4);
                    enc.setBuffer(self.normed_buf, 0, 5);
                    enc.setBuffer(self.sums_buf, 0, 6);
                    enc.setValue(PostNormResParams{
                        .rows = norm_rows,
                        .hidden = @intCast(c.hidden_size),
                        .sums_stride = if (norm_rows == 1 and use_qmv_1) 0 else self.sums_stride,
                        .has_scalar = if (lw.has_layer_scalar) 1 else 0,
                        .eps = c.rms_eps,
                    }, 7);
                    enc.dispatchGroups(.{ .width = norm_rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                } else if (use_res_fusion) {
                    self.dispatchRmsNorm1(&enc, self.h_buf.view(h_off), self.w.final_norm, self.normed_buf.view(0));
                } else {
                    enc.setPipeline(self.p.add_rms_norm_sums);
                    enc.setBuffer(self.h_buf, h_off, 0);
                    enc.setBuffer(self.delta_buf, h_off, 1);
                    enc.setRef(self.w.final_norm, 2);
                    enc.setBuffer(self.normed_buf, 0, 3);
                    enc.setBuffer(self.sums_buf, 0, 4);
                    enc.setValue(NormParams{
                        .rows = norm_rows,
                        .hidden = @intCast(c.hidden_size),
                        .sums_stride = if (norm_rows == 1 and use_qmv_1) 0 else self.sums_stride,
                        .eps = c.rms_eps,
                    }, 5);
                    enc.dispatchGroups(.{ .width = norm_rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                }
                enc.barrier();
            }

            if ((li + 1) % commit_every == 0 and li + 1 < c.num_layers) {
                enc.updateFence(self.fence);
                enc.end();
                cb.commit();
                cb = self.ctx.beginCommandBuffer();
                enc = cb.beginCompute();
                enc.waitFence(self.fence);
            }
        }

        // 3. LM Head (+ optional final_logit_softcap) + Batched Multi-Core Argmax
        if (want_logits) {
            const logit_rows: u32 = if (all_row_logits) rows else 1;
            if (logit_rows == 1 and use_qmv_fused_1 and !need_full_logits and c.final_logit_softcap == 0.0 and self.w.lm_head.isQk4()) {
                // Super-fused LM Head + Stage-1 Argmax: eliminates 496 KB logits write/read + argmax_stage1!
                const lm_qk = self.w.lm_head.qk4;
                const lm_tiles: u32 = @intCast(lm_qk.tiles());
                enc.setPipeline(self.p.qmv_1_argmax);
                enc.setBuffer(self.normed_buf, 0, 0);
                enc.setRef(lm_qk.ref, 1);
                enc.setBuffer(self.argmax_val_buf, 0, 2);
                enc.setBuffer(self.argmax_idx_buf, 0, 3);
                enc.setValue(QkParams{
                    .M = 1,
                    .K = @intCast(lm_qk.k),
                    .N = @intCast(c.vocab_size),
                    .G = @intCast(lm_qk.groups()),
                    .first_row = 0,
                    .sums_stride = 0,
                    .tile_first = 0,
                }, 4);
                enc.dispatchGroups(.{ .width = lm_tiles, .height = 1, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = lm_tiles, .rows = 1 };
                enc.setPipeline(self.p.argmax_stage2);
                enc.setBuffer(self.argmax_val_buf, 0, 0);
                enc.setBuffer(self.argmax_idx_buf, 0, 1);
                enc.setBuffer(self.out_tok_buf, 0, 2);
                enc.setBuffer(self.out_val_buf, 0, 3);
                enc.setValue(ap, 4);
                enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            } else {
                self.dispatchQk(&enc, self.normed_buf.view(0), self.w.lm_head, self.sums_buf.view(0), self.logits_buf.view(0), logit_rows, 1);
                enc.barrier();

                if (c.final_logit_softcap > 0.0) {
                    const lcnt: u32 = @intCast(@as(usize, logit_rows) * c.vocab_size);
                    enc.setPipeline(self.p.logit_softcap);
                    enc.setBuffer(self.logits_buf, 0, 0);
                    enc.setValue(SoftcapParams{ .count = lcnt, .cap = c.final_logit_softcap }, 1);
                    enc.dispatchThreads(.{ .width = lcnt, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                    enc.barrier();
                }

                const chunks: u32 = @intCast((c.vocab_size + 1023) / 1024);
                const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = chunks, .rows = logit_rows };
                enc.setPipeline(self.p.argmax_stage1);
                enc.setBuffer(self.logits_buf, 0, 0);
                enc.setBuffer(self.argmax_val_buf, 0, 1);
                enc.setBuffer(self.argmax_idx_buf, 0, 2);
                enc.setValue(ap, 3);
                enc.dispatchGroups(.{ .width = chunks, .height = logit_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(self.p.argmax_stage2);
                enc.setBuffer(self.argmax_val_buf, 0, 0);
                enc.setBuffer(self.argmax_idx_buf, 0, 1);
                enc.setBuffer(self.out_tok_buf, 0, 2);
                enc.setBuffer(self.out_val_buf, 0, 3);
                enc.setValue(ap, 4);
                enc.dispatchGroups(.{ .width = 1, .height = logit_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            }
        }

        enc.end();
        cb.commit();
        const encode_ms = nowMs() - t_start;

        try self.ctx.waitIdle();
        if (use_fused_gdn_1) {
            std.mem.swap(Buffer, &self.gdn_hist_buf, &self.gdn_hist_alt_buf);
        }
        const wall_ms = nowMs() - t_start;
        self.last_timing = .{ .wall_ms = wall_ms, .encode_ms = encode_ms };
    }

    fn executeNemotronH(
        self: *Model,
        rows: u32,
        want_logits: bool,
        commit_immediately: bool,
        all_row_logits: bool,
        need_full_logits: bool,
        is_tree: bool,
        t_start: f64,
    ) !void {
        const c = self.c;
        const commit_every: usize = if (self.ctx.isMetal4() and c.num_layers >= 48) 8 else c.num_layers;
        const mamba_inter: u32 = @intCast(c.mambaInterDim());
        const mamba_conv: u32 = @intCast(c.mambaConvDim());
        const mamba_in_stride: u32 = @intCast(c.mambaInProjOut());
        const attn_qkv_stride: u32 = @intCast(c.qProjOut() + 2 * c.kvProjOut());

        const use_qmv_fused_1 = (rows == 1 and use_qmv_1 and use_fused_1);
        const norm_sums_stride: u32 = if (rows == 1 and use_qmv_1) 0 else self.sums_stride;

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // 1. Token embeddings -> h_buf + Layer 0 RMSNorm -> normed_buf (+ sums)
        self.dispatchEmbed(&enc, self.tokens_buf.view(0), self.h_buf.view(0), rows);
        enc.barrier();

        enc.setPipeline(self.p.rms_norm_sums);
        enc.setBuffer(self.h_buf, 0, 0);
        enc.setRef(self.w.layers[0].input_norm, 1);
        enc.setBuffer(self.normed_buf, 0, 2);
        enc.setBuffer(self.sums_buf, 0, 3);
        enc.setValue(NormParams{
            .rows = rows,
            .hidden = @intCast(c.hidden_size),
            .sums_stride = norm_sums_stride,
            .eps = c.rms_eps,
        }, 4);
        enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        // 2. 52 single-mixer layers (Mamba-2 / Sparse MoE / NoPE GQA Attention)
        var mamba_idx: usize = 0;
        var full_idx: usize = 0;
        for (0..c.num_layers) |li| {
            const lw = self.w.layers[li];
            switch (lw.kind) {
                .mamba => {
                    const st_ref = self.mamba_state_buf.view(mamba_idx * self.mamba_state_stride);
                    const h_in_ref = self.mamba_hist_buf.view(mamba_idx * self.mamba_hist_stride);
                    const h_out_ref = self.mamba_hist_alt_buf.view(mamba_idx * self.mamba_hist_stride);
                    mamba_idx += 1;

                    self.dispatchQk(&enc, self.normed_buf.view(0), lw.mamba.in_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                    enc.barrier();

                    enc.setPipeline(self.p.mamba2_conv_silu.?);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(lw.mamba.conv_weight, 1);
                    enc.setRef(lw.mamba.conv_bias, 2);
                    enc.setRef(h_in_ref, 3);
                    enc.setBuffer(self.mamba_conv_out_buf, 0, 4);
                    enc.setRef(h_out_ref, 5);
                    enc.setValue(MambaConvParams{
                        .rows = rows,
                        .inter_dim = mamba_inter,
                        .conv_dim = mamba_conv,
                        .in_proj_stride = mamba_in_stride,
                        .commit_hist = if (commit_immediately) 1 else 0,
                    }, 6);
                    enc.dispatchThreads(.{ .width = mamba_conv, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                    enc.barrier();

                    enc.setPipeline(self.p.mamba2_ssm_scan.?);
                    enc.setBuffer(self.mamba_conv_out_buf, 0, 0);
                    enc.setBuffer(self.proj_big_buf, 0, 1);
                    enc.setRef(lw.mamba.a_log, 2);
                    enc.setRef(lw.mamba.dt_bias, 3);
                    enc.setRef(lw.mamba.d_skip, 4);
                    enc.setRef(st_ref, 5);
                    enc.setBuffer(self.mamba_ssm_out_buf, 0, 6);
                    enc.setValue(MambaSsmParams{
                        .rows = rows,
                        .num_heads = @intCast(c.mamba_num_heads),
                        .head_dim = @intCast(c.mamba_head_dim),
                        .n_groups = @intCast(c.mamba_n_groups),
                        .d_state = @intCast(c.ssm_state_size),
                        .conv_dim = mamba_conv,
                        .in_proj_stride = mamba_in_stride,
                        .dt_offset = mamba_inter + mamba_conv,
                        .commit_state = if (commit_immediately) 1 else 0,
                        .dt_min = c.dt_min,
                        .dt_max = c.dt_max,
                    }, 7);
                    enc.dispatchGroups(.{ .width = c.mamba_num_heads, .height = c.mamba_head_dim / 4, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    enc.barrier();

                    enc.setPipeline(self.p.mamba2_norm_gate_sums.?);
                    enc.setBuffer(self.mamba_ssm_out_buf, 0, 0);
                    enc.setBuffer(self.proj_big_buf, 0, 1);
                    enc.setRef(lw.mamba.mamba_norm, 2);
                    enc.setBuffer(self.act_buf, 0, 3);
                    enc.setBuffer(self.sums_buf, 0, 4);
                    enc.setValue(MambaNormParams{
                        .rows = rows,
                        .inter_dim = mamba_inter,
                        .n_groups = @intCast(c.mamba_n_groups),
                        .group_size = @intCast(c.mambaInterDim() / c.mamba_n_groups),
                        .in_proj_stride = mamba_in_stride,
                        .sums_stride = norm_sums_stride,
                        .eps = c.rms_eps,
                    }, 5);
                    enc.dispatchGroups(.{ .width = c.mamba_n_groups, .height = rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                    enc.barrier();

                    self.dispatchQk(&enc, self.act_buf.view(0), lw.mamba.out_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                    enc.barrier();
                },
                .moe => {
                    // Shared expert: up_proj -> ReLU^2 -> down_proj -> delta_buf
                    if (use_qmv_fused_1 and lw.moe.shared_up.isQk4()) {
                        const su_qk = lw.moe.shared_up.qk4;
                        enc.setPipeline(self.p.qmv_1_relu2.?);
                        enc.setBuffer(self.normed_buf, 0, 0);
                        enc.setRef(su_qk.ref, 1);
                        enc.setBuffer(self.act_buf, 0, 2);
                        enc.setValue(SharedRelu2Params{
                            .K = @intCast(su_qk.k),
                            .N = @intCast(su_qk.n),
                            .G = @intCast(su_qk.groups()),
                        }, 3);
                        enc.dispatchGroups(.{ .width = su_qk.tiles(), .height = 1, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                        enc.barrier();
                    } else {
                        self.dispatchQk(&enc, self.normed_buf.view(0), lw.moe.shared_up, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                        enc.barrier();

                        enc.setPipeline(self.p.relu2_sums.?);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setBuffer(self.act_buf, 0, 1);
                        enc.setBuffer(self.sums_buf, 0, 2);
                        enc.setValue(Relu2Params{
                            .rows = rows,
                            .width = @intCast(c.moe_shared_intermediate_size),
                            .sums_stride = norm_sums_stride,
                        }, 3);
                        enc.dispatchGroups(.{ .width = c.moe_shared_intermediate_size / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();
                    }

                    self.dispatchQk(&enc, self.act_buf.view(0), lw.moe.shared_down, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                    enc.barrier();

                    // Router logits -> Sigmoid + e_score_correction_bias top-K selection
                    enc.setPipeline(self.p.moe_gate_logits.?);
                    enc.setBuffer(self.normed_buf, 0, 0);
                    enc.setRef(lw.moe.gate_weight, 1);
                    enc.setBuffer(self.moe_logits_buf, 0, 2);
                    enc.setValue(GateParams{
                        .rows = rows,
                        .hidden = @intCast(c.hidden_size),
                        .n_experts = @intCast(c.n_routed_experts),
                    }, 3);
                    enc.dispatchGroups(.{ .width = (c.n_routed_experts + 3) / 4, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    enc.barrier();

                    enc.setPipeline(self.p.moe_topk_select.?);
                    enc.setBuffer(self.moe_logits_buf, 0, 0);
                    enc.setRef(lw.moe.gate_bias, 1);
                    enc.setBuffer(self.moe_inds_buf, 0, 2);
                    enc.setBuffer(self.moe_weights_buf, 0, 3);
                    enc.setValue(TopkParams{
                        .rows = rows,
                        .n_experts = @intCast(c.n_routed_experts),
                        .top_k = @intCast(c.num_experts_per_tok),
                        .norm_topk_prob = if (c.norm_topk_prob) 1 else 0,
                        .routed_scaling_factor = c.routed_scaling_factor,
                    }, 4);
                    enc.dispatchGroups(.{ .width = 1, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    enc.barrier();

                    // Top-K routed experts: M5 tensor-unit batched GEMM (or SIMD batched fallback) on prefill, QMV on decode!
                    if (rows >= 4) {
                        enc.setPipeline(self.p.moe_bucket_tokens.?);
                        enc.setBuffer(self.moe_inds_buf, 0, 0);
                        enc.setBuffer(self.moe_offsets_buf, 0, 1);
                        enc.setBuffer(self.moe_counts_buf, 0, 2);
                        enc.setBuffer(self.moe_buckets_buf, 0, 3);
                        enc.setValue(MoeBucketParams{
                            .rows = rows,
                            .top_k = @intCast(c.num_experts_per_tok),
                            .n_experts = @intCast(c.n_routed_experts),
                        }, 4);
                        enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();

                        if (self.p.moe_expert_qk16_batched) |qk16_pso| {
                            const total_slots: u32 = rows * @as(u32, @intCast(c.num_experts_per_tok));
                            const moe_sums_stride: u32 = ((total_slots + 15) / 16) * 16;

                            // Gather bucketed X + group-64 sums for fc1
                            enc.setPipeline(self.p.moe_gather_x_sums.?);
                            enc.setBuffer(self.normed_buf, 0, 0);
                            enc.setBuffer(self.moe_buckets_buf, 0, 1);
                            enc.setBuffer(self.moe_x_buf, 0, 2);
                            enc.setBuffer(self.moe_sums_buf, 0, 3);
                            enc.setValue(MoeGatherSumsParams{
                                .total_slots = total_slots,
                                .top_k = @intCast(c.num_experts_per_tok),
                                .K = @intCast(c.hidden_size),
                                .sums_stride = moe_sums_stride,
                                .identity = 0,
                            }, 4);
                            enc.dispatchGroups(.{ .width = c.hidden_size / 64, .height = total_slots, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                            enc.barrier();

                            // M5 hardware tensor-unit fc1 + ReLU^2 -> moe_act_buf (contiguous bucketed order!)
                            enc.setPipeline(qk16_pso);
                            enc.setBuffer(self.moe_x_buf, 0, 0);
                            enc.setRef(lw.moe.fc1_experts, 1);
                            enc.setBuffer(self.moe_sums_buf, 0, 2);
                            enc.setBuffer(self.moe_offsets_buf, 0, 3);
                            enc.setBuffer(self.moe_counts_buf, 0, 4);
                            enc.setBuffer(self.moe_buckets_buf, 0, 5);
                            enc.setBuffer(self.moe_act_buf, 0, 6);
                            enc.setValue(MoeQkParams{
                                .total_slots = total_slots,
                                .n_experts = @intCast(c.n_routed_experts),
                                .K = @intCast(c.hidden_size),
                                .N = @intCast(c.moe_intermediate_size),
                                .G = @intCast(c.hidden_size / 64),
                                .sums_stride = @intCast(moe_sums_stride),
                                .apply_relu2 = 1,
                                .scatter_out = 0,
                                .expert_stride = @intCast(lw.moe.fc1_expert_stride),
                            }, 7);
                            enc.dispatchGroups(.{ .width = c.moe_intermediate_size / 32, .height = c.n_routed_experts, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                            enc.barrier();

                            // In-place group-64 sums over moe_act_buf for fc2
                            enc.setPipeline(self.p.moe_gather_x_sums.?);
                            enc.setBuffer(self.moe_act_buf, 0, 0);
                            enc.setBuffer(self.moe_buckets_buf, 0, 1);
                            enc.setBuffer(self.moe_act_buf, 0, 2);
                            enc.setBuffer(self.moe_sums_buf, 0, 3);
                            enc.setValue(MoeGatherSumsParams{
                                .total_slots = total_slots,
                                .top_k = @intCast(c.num_experts_per_tok),
                                .K = @intCast(c.moe_intermediate_size),
                                .sums_stride = moe_sums_stride,
                                .identity = 1,
                            }, 4);
                            enc.dispatchGroups(.{ .width = c.moe_intermediate_size / 64, .height = total_slots, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                            enc.barrier();

                            // M5 hardware tensor-unit fc2 -> scatter directly to moe_out_buf[slot]!
                            enc.setPipeline(qk16_pso);
                            enc.setBuffer(self.moe_act_buf, 0, 0);
                            enc.setRef(lw.moe.fc2_experts, 1);
                            enc.setBuffer(self.moe_sums_buf, 0, 2);
                            enc.setBuffer(self.moe_offsets_buf, 0, 3);
                            enc.setBuffer(self.moe_counts_buf, 0, 4);
                            enc.setBuffer(self.moe_buckets_buf, 0, 5);
                            enc.setBuffer(self.moe_out_buf, 0, 6);
                            enc.setValue(MoeQkParams{
                                .total_slots = total_slots,
                                .n_experts = @intCast(c.n_routed_experts),
                                .K = @intCast(c.moe_intermediate_size),
                                .N = @intCast(c.hidden_size),
                                .G = @intCast(c.moe_intermediate_size / 64),
                                .sums_stride = @intCast(moe_sums_stride),
                                .apply_relu2 = 0,
                                .scatter_out = 1,
                                .expert_stride = @intCast(lw.moe.fc2_expert_stride),
                            }, 7);
                            enc.dispatchGroups(.{ .width = c.hidden_size / 32, .height = c.n_routed_experts, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                            enc.barrier();
                        } else {
                            enc.setPipeline(self.p.moe_expert_qmm_batched.?);
                            enc.setBuffer(self.normed_buf, 0, 0);
                            enc.setRef(lw.moe.fc1_experts, 1);
                            enc.setBuffer(self.moe_offsets_buf, 0, 2);
                            enc.setBuffer(self.moe_counts_buf, 0, 3);
                            enc.setBuffer(self.moe_buckets_buf, 0, 4);
                            enc.setBuffer(self.moe_act_buf, 0, 5);
                            enc.setValue(MoeBatchedParams{
                                .top_k = @intCast(c.num_experts_per_tok),
                                .n_experts = @intCast(c.n_routed_experts),
                                .K = @intCast(c.hidden_size),
                                .N = @intCast(c.moe_intermediate_size),
                                .G = @intCast(c.hidden_size / 64),
                                .apply_relu2 = 1,
                                .x_per_expert = 0,
                                .expert_stride = @intCast(lw.moe.fc1_expert_stride),
                            }, 6);
                            enc.dispatchGroups(.{ .width = c.moe_intermediate_size / 32, .height = c.n_routed_experts, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                            enc.barrier();

                            enc.setPipeline(self.p.moe_expert_qmm_batched.?);
                            enc.setBuffer(self.moe_act_buf, 0, 0);
                            enc.setRef(lw.moe.fc2_experts, 1);
                            enc.setBuffer(self.moe_offsets_buf, 0, 2);
                            enc.setBuffer(self.moe_counts_buf, 0, 3);
                            enc.setBuffer(self.moe_buckets_buf, 0, 4);
                            enc.setBuffer(self.moe_out_buf, 0, 5);
                            enc.setValue(MoeBatchedParams{
                                .top_k = @intCast(c.num_experts_per_tok),
                                .n_experts = @intCast(c.n_routed_experts),
                                .K = @intCast(c.moe_intermediate_size),
                                .N = @intCast(c.hidden_size),
                                .G = @intCast(c.moe_intermediate_size / 64),
                                .apply_relu2 = 0,
                                .x_per_expert = 1,
                                .expert_stride = @intCast(lw.moe.fc2_expert_stride),
                            }, 6);
                            enc.dispatchGroups(.{ .width = c.hidden_size / 32, .height = c.n_routed_experts, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                            enc.barrier();
                        }
                    } else {
                        enc.setPipeline(self.p.moe_expert_qmv.?);
                        enc.setBuffer(self.normed_buf, 0, 0);
                        enc.setRef(lw.moe.fc1_experts, 1);
                        enc.setBuffer(self.moe_inds_buf, 0, 2);
                        enc.setBuffer(self.moe_act_buf, 0, 3);
                        enc.setValue(MoeExpertParams{
                            .rows = rows,
                            .top_k = @intCast(c.num_experts_per_tok),
                            .K = @intCast(c.hidden_size),
                            .N = @intCast(c.moe_intermediate_size),
                            .G = @intCast(c.hidden_size / 64),
                            .apply_relu2 = 1,
                            .x_per_expert = 0,
                            .expert_stride = @intCast(lw.moe.fc1_expert_stride),
                        }, 4);
                        enc.dispatchGroups(.{ .width = c.moe_intermediate_size / 32, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();

                        enc.setPipeline(self.p.moe_expert_qmv.?);
                        enc.setBuffer(self.moe_act_buf, 0, 0);
                        enc.setRef(lw.moe.fc2_experts, 1);
                        enc.setBuffer(self.moe_inds_buf, 0, 2);
                        enc.setBuffer(self.moe_out_buf, 0, 3);
                        enc.setValue(MoeExpertParams{
                            .rows = rows,
                            .top_k = @intCast(c.num_experts_per_tok),
                            .K = @intCast(c.moe_intermediate_size),
                            .N = @intCast(c.hidden_size),
                            .G = @intCast(c.moe_intermediate_size / 64),
                            .apply_relu2 = 0,
                            .x_per_expert = 1,
                            .expert_stride = @intCast(lw.moe.fc2_expert_stride),
                        }, 4);
                        enc.dispatchGroups(.{ .width = c.hidden_size / 32, .height = c.num_experts_per_tok, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();
                    }

                    enc.setPipeline(self.p.moe_combine_shared.?);
                    enc.setBuffer(self.moe_out_buf, 0, 0);
                    enc.setBuffer(self.moe_weights_buf, 0, 1);
                    enc.setBuffer(self.delta_buf, 0, 2);
                    enc.setValue(MoeCombineParams{
                        .rows = rows,
                        .hidden = @intCast(c.hidden_size),
                        .top_k = @intCast(c.num_experts_per_tok),
                    }, 3);
                    enc.dispatchThreads(.{ .width = c.hidden_size, .height = rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                    enc.barrier();
                },
                .full => {
                    const k_ref = self.k_cache_buf.view(full_idx * self.kv_layer_stride);
                    const v_ref = self.v_cache_buf.view(full_idx * self.kv_layer_stride);
                    full_idx += 1;

                    self.dispatchQk(&enc, self.normed_buf.view(0), lw.full.qkv_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                    enc.barrier();

                    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
                    if (rows == 1 and commit_immediately and use_qmv_fused_1 and !is_tree) {
                        enc.setPipeline(self.p.attn_std128_decode_1);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(self.w.final_norm, 1);
                        enc.setBuffer(self.inv_freq_buf, 0, 2);
                        enc.setRef(k_ref, 3);
                        enc.setRef(v_ref, 4);
                        enc.setBuffer(self.act_buf, 0, 5);
                        enc.setValue(Std128Decode1Params{
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .has_bias = 0,
                            .use_rope = 0, // Nemotron-H uses NoPE!
                            .pos = self.pos,
                            .scale = scale,
                        }, 6);
                        enc.setRef(self.w.final_norm, 7);
                        enc.setRef(self.w.final_norm, 8);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    } else {
                        enc.setPipeline(self.p.attn_std128_rope_store);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(self.w.final_norm, 1);
                        enc.setBuffer(self.inv_freq_buf, 0, 2);
                        enc.setBuffer(self.positions_buf, 0, 3);
                        enc.setBuffer(self.kv_slots_buf, 0, 4);
                        enc.setBuffer(self.attn_q_buf, 0, 5);
                        enc.setRef(k_ref, 6);
                        enc.setRef(v_ref, 7);
                        enc.setValue(Std128PrepParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .has_bias = 0,
                            .use_rope = 0, // Nemotron-H uses NoPE!
                        }, 8);
                        enc.setRef(self.w.final_norm, 9);
                        enc.setRef(self.w.final_norm, 10);
                        enc.dispatchGroups(.{ .width = c.num_heads + c.num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();

                        const fp = AttnFlashParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .sums_stride = norm_sums_stride,
                            .scale = scale,
                            .splits = 1,
                            .base_pos = self.pos,
                            .is_tree = if (is_tree) 1 else 0,
                        };
                        enc.setPipeline(self.p.attn_std128_flash);
                        enc.setBuffer(self.attn_q_buf, 0, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setBuffer(self.seq_lens_buf, 0, 3);
                        enc.setBuffer(self.tree_masks_buf, 0, 4);
                        enc.setBuffer(self.act_buf, 0, 5);
                        enc.setBuffer(self.sums_buf, 0, 6);
                        enc.setValue(fp, 7);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    }

                    self.dispatchQk(&enc, self.act_buf.view(0), lw.full.o_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                    enc.barrier();
                },
                .linear, .short_conv => unreachable,
            }

            // Residual add + next layer RMSNorm (or final_norm)
            if (li + 1 < c.num_layers) {
                enc.setPipeline(self.p.add_rms_norm_sums);
                enc.setBuffer(self.h_buf, 0, 0);
                enc.setBuffer(self.delta_buf, 0, 1);
                enc.setRef(self.w.layers[li + 1].input_norm, 2);
                enc.setBuffer(self.normed_buf, 0, 3);
                enc.setBuffer(self.sums_buf, 0, 4);
                enc.setValue(NormParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .sums_stride = norm_sums_stride,
                    .eps = c.rms_eps,
                }, 5);
                enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();
            } else if (want_logits) {
                const norm_rows: u32 = if (all_row_logits) rows else 1;
                const h_off: usize = if (all_row_logits) 0 else @as(usize, rows - 1) * c.hidden_size * 2;
                enc.setPipeline(self.p.add_rms_norm_sums);
                enc.setBuffer(self.h_buf, h_off, 0);
                enc.setBuffer(self.delta_buf, h_off, 1);
                enc.setRef(self.w.final_norm, 2);
                enc.setBuffer(self.normed_buf, 0, 3);
                enc.setBuffer(self.sums_buf, 0, 4);
                enc.setValue(NormParams{
                    .rows = norm_rows,
                    .hidden = @intCast(c.hidden_size),
                    .sums_stride = if (norm_rows == 1 and use_qmv_1) 0 else self.sums_stride,
                    .eps = c.rms_eps,
                }, 5);
                enc.dispatchGroups(.{ .width = norm_rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();
            }

            if ((li + 1) % commit_every == 0 and li + 1 < c.num_layers) {
                enc.updateFence(self.fence);
                enc.end();
                cb.commit();
                cb = self.ctx.beginCommandBuffer();
                enc = cb.beginCompute();
                enc.waitFence(self.fence);
            }
        }

        // 3. LM Head + Argmax
        if (want_logits) {
            const logit_rows: u32 = if (all_row_logits) rows else 1;
            if (logit_rows == 1 and use_qmv_fused_1 and !need_full_logits and self.w.lm_head.isQk4()) {
                const lm_qk = self.w.lm_head.qk4;
                const lm_tiles: u32 = @intCast(lm_qk.tiles());
                enc.setPipeline(self.p.qmv_1_argmax);
                enc.setBuffer(self.normed_buf, 0, 0);
                enc.setRef(lm_qk.ref, 1);
                enc.setBuffer(self.argmax_val_buf, 0, 2);
                enc.setBuffer(self.argmax_idx_buf, 0, 3);
                enc.setValue(QkParams{
                    .M = 1,
                    .K = @intCast(lm_qk.k),
                    .N = @intCast(c.vocab_size),
                    .G = @intCast(lm_qk.groups()),
                    .first_row = 0,
                    .sums_stride = 0,
                    .tile_first = 0,
                }, 4);
                enc.dispatchGroups(.{ .width = lm_tiles, .height = 1, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = lm_tiles, .rows = 1 };
                enc.setPipeline(self.p.argmax_stage2);
                enc.setBuffer(self.argmax_val_buf, 0, 0);
                enc.setBuffer(self.argmax_idx_buf, 0, 1);
                enc.setBuffer(self.out_tok_buf, 0, 2);
                enc.setBuffer(self.out_val_buf, 0, 3);
                enc.setValue(ap, 4);
                enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            } else {
                self.dispatchQk(&enc, self.normed_buf.view(0), self.w.lm_head, self.sums_buf.view(0), self.logits_buf.view(0), logit_rows, 1);
                enc.barrier();

                const chunks: u32 = @intCast((c.vocab_size + 1023) / 1024);
                const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = chunks, .rows = logit_rows };
                enc.setPipeline(self.p.argmax_stage1);
                enc.setBuffer(self.logits_buf, 0, 0);
                enc.setBuffer(self.argmax_val_buf, 0, 1);
                enc.setBuffer(self.argmax_idx_buf, 0, 2);
                enc.setValue(ap, 3);
                enc.dispatchGroups(.{ .width = chunks, .height = logit_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(self.p.argmax_stage2);
                enc.setBuffer(self.argmax_val_buf, 0, 0);
                enc.setBuffer(self.argmax_idx_buf, 0, 1);
                enc.setBuffer(self.out_tok_buf, 0, 2);
                enc.setBuffer(self.out_val_buf, 0, 3);
                enc.setValue(ap, 4);
                enc.dispatchGroups(.{ .width = 1, .height = logit_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            }
        }

        enc.end();
        cb.commit();
        const encode_ms = nowMs() - t_start;

        try self.ctx.waitIdle();
        if (commit_immediately) {
            std.mem.swap(Buffer, &self.mamba_hist_buf, &self.mamba_hist_alt_buf);
        }
        const wall_ms = nowMs() - t_start;
        self.last_timing = .{ .wall_ms = wall_ms, .encode_ms = encode_ms };
    }
};

/// Compile a topological parent array (`parents[0] = -1`, `parents[r] < r`) into a depth-first
/// GDN scan schedule (`steps[i] = row | (restore_slot << 16) | (save_slot << 24)`).
/// Returns the maximum number of concurrent threadgroup state slots needed (1..8).
fn compileTreeSteps(parents: []const i32, steps_out: []u32) u32 {
    const n = parents.len;
    var first_child: [128]i32 = @splat(-1);
    var next_sibling: [128]i32 = @splat(-1);
    var child_count: [128]u32 = @splat(0);

    // Build adjacency lists (reverse iteration preserves ascending child order)
    var i: usize = n;
    while (i > 1) {
        i -= 1;
        const p: usize = @intCast(parents[i]);
        next_sibling[i] = first_child[p];
        first_child[p] = @intCast(i);
        child_count[p] += 1;
    }

    const Frame = struct { node: u32, restore_slot: u32, free_after_restore: bool };
    var stack: [128]Frame = undefined;
    var sp: usize = 0;
    stack[sp] = .{ .node = 0, .restore_slot = 0, .free_after_restore = false };
    sp += 1;

    var slot_used: [8]bool = @splat(false);
    var max_slots: u32 = 0;
    var out_i: usize = 0;

    while (sp > 0) {
        sp -= 1;
        const f = stack[sp];
        if (f.free_after_restore and f.restore_slot > 0) {
            slot_used[f.restore_slot - 1] = false;
        }

        var save_slot: u32 = 0;
        if (child_count[f.node] > 1) {
            // Allocate a free TG slot to save state for non-first children
            for (0..8) |s| {
                if (!slot_used[s]) {
                    slot_used[s] = true;
                    save_slot = @intCast(s + 1);
                    if (save_slot > max_slots) max_slots = save_slot;
                    break;
                }
            }
        }

        steps_out[out_i] = f.node | (f.restore_slot << 16) | (save_slot << 24);
        out_i += 1;

        // Gather children in order, then push onto DFS stack in reverse so child 0 runs immediately
        var kids: [128]u32 = undefined;
        var k_cnt: usize = 0;
        var ch = first_child[f.node];
        while (ch >= 0) {
            kids[k_cnt] = @intCast(ch);
            k_cnt += 1;
            ch = next_sibling[@intCast(ch)];
        }
        var ki: usize = k_cnt;
        while (ki > 0) {
            ki -= 1;
            const is_first = (ki == 0);
            const is_last = (ki + 1 == k_cnt);
            stack[sp] = .{
                .node = kids[ki],
                .restore_slot = if (is_first) 0 else save_slot,
                .free_after_restore = (!is_first and is_last),
            };
            sp += 1;
        }
    }

    return max_slots;
}

fn align256(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}
