//! HuggingFace config.json parser supporting:
//!   - Qwen3.5 / Qwen3.8 Hybrid GDN + GQA (4-bit MLX or raw BF16 checkpoints)
//!   - Qwen2 / Qwen2.5 / Llama Standard Dense Transformers (BF16 or 4-bit MLX checkpoints)
//!   - NVIDIA Nemotron-H / Nemotron-3.5 Sparse MoE + Mamba-2 + NoPE GQA (4-bit MLX checkpoints)
const std = @import("std");
const safetensors = @import("safetensors.zig");

pub const Arch = enum { qwen35_hybrid, qwen2_dense, nemotron_h };
pub const LayerKind = enum { linear, full, mamba, moe, short_conv };
pub const QuantFormat = enum { affine, tfq3, bf16 };
pub const MoeStyle = enum { none, relu2_sigmoid, swiglu_softmax };
pub const RopeType = enum { default, llama3, linear, yarn };
pub const max_layers = 128;
pub const max_eos = 8;

pub const ModelConfig = struct {
    arch: Arch = .qwen35_hybrid,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    vocab_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // Gated DeltaNet (linear_attention) geometry
    gdn_k_heads: usize = 0,
    gdn_v_heads: usize = 0,
    gdn_k_dim: usize = 128,
    gdn_v_dim: usize = 128,
    conv_kernel: usize = 4,
    // Mamba-2 (nemotron_h) geometry
    mamba_num_heads: usize = 0,
    mamba_head_dim: usize = 0,
    ssm_state_size: usize = 0,
    mamba_n_groups: usize = 0,
    dt_min: f32 = 0.0,
    dt_max: f32 = std.math.inf(f32),
    // Sparse MoE (Nemotron-H, Qwen3 MoE, Qwen3.5/3.6 MoE, Mixtral, DeepSeek, GLM)
    moe_style: MoeStyle = .none,
    moe_use_sigmoid: bool = false,
    n_routed_experts: usize = 0,
    num_experts_per_tok: usize = 0,
    moe_intermediate_size: usize = 0,
    moe_shared_intermediate_size: usize = 0,
    has_shared_expert: bool = false,
    shared_expert_gated: bool = false,
    routed_scaling_factor: f32 = 1.0,
    norm_topk_prob: bool = true,
    n_group: usize = 0,
    topk_group: usize = 0,
    layer_is_moe: [max_layers]bool = @splat(false),
    // Multi-Head Latent Attention (DeepSeek-V2 / V3 / R1 / V3.2, GLM-4.7-Flash / GLM-5)
    use_mla: bool = false,
    q_lora_rank: usize = 0,
    kv_lora_rank: usize = 0,
    qk_nope_head_dim: usize = 0,
    qk_rope_head_dim: usize = 0,
    v_head_dim: usize = 0,
    mla_attn_scale: f32 = 0.0,
    num_nextn_predict_layers: usize = 0,
    // Normalization, RoPE & Attention modifiers
    rms_eps: f32,
    rope_dims: usize,
    rope_theta: f64,
    use_rope: bool = true,
    rope_interleaved: bool = false,
    rope_mscale: f32 = 1.0,
    rope_type: RopeType = .default,
    rope_factor: f64 = 1.0,
    rope_low_freq_factor: f64 = 1.0,
    rope_high_freq_factor: f64 = 4.0,
    rope_orig_max_pos: f64 = 8192.0,
    rope_beta_fast: f64 = 32.0,
    rope_beta_slow: f64 = 1.0,
    sliding_window: usize = 0,
    attn_softcap: f32 = 0.0,
    final_logit_softcap: f32 = 0.0,
    // Gemma 2 / 3 / 4, GLM-4, and LiquidAI LFM2 / LFM2.5 modifiers
    is_gemma: bool = false,
    is_gemma4: bool = false,
    glm4_post_norms: bool = false,
    is_lfm: bool = false,
    conv_l_cache: usize = 3,
    conv_bias: bool = false,
    embed_scale: f32 = 1.0,
    query_pre_attn_scalar: f32 = 0.0,
    rms_norm_add_one: bool = false,
    has_post_norms: bool = false,
    use_geglu: bool = false,
    has_v_norm: bool = false,
    has_dual_rope: bool = false,
    rope_local_theta: f64 = 10_000.0,
    global_head_dim: usize = 0,
    global_kv_heads: usize = 0,
    global_partial_rotary: f64 = 1.0,
    layer_is_sliding: [max_layers]bool = @splat(false),
    layer_head_dim: [max_layers]usize = @splat(0),
    layer_kv_heads: [max_layers]usize = @splat(0),
    layer_k_eq_v: [max_layers]bool = @splat(false),
    // Quantization & embeddings
    quant_format: QuantFormat = .affine,
    bits: usize = 4,
    group_size: usize = 64,
    has_quant_config: bool = false,
    tie_word_embeddings: bool = false,
    attn_output_gate: bool = true,
    has_qk_norm: bool = true,
    has_qkv_bias: bool = false,
    bos_token_id: ?u32 = null,
    // Layer schedule & EOS
    layer_kinds: [max_layers]LayerKind = @splat(.linear),
    eos_ids: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,
    max_position: usize = 262144,

    pub fn qProjOut(self: ModelConfig) usize {
        return if (self.attn_output_gate) 2 * self.num_heads * self.head_dim else self.num_heads * self.head_dim;
    }

    pub fn kvProjOut(self: ModelConfig) usize {
        return self.num_kv_heads * self.head_dim;
    }

    pub fn layerHeadDim(self: ModelConfig, li: usize) usize {
        return if (self.layer_head_dim[li] != 0) self.layer_head_dim[li] else self.head_dim;
    }

    pub fn layerKvHeads(self: ModelConfig, li: usize) usize {
        return if (self.layer_kv_heads[li] != 0) self.layer_kv_heads[li] else self.num_kv_heads;
    }

    pub fn layerQProjOut(self: ModelConfig, li: usize) usize {
        const hd = self.layerHeadDim(li);
        return if (self.attn_output_gate) 2 * self.num_heads * hd else self.num_heads * hd;
    }

    pub fn layerKvProjOut(self: ModelConfig, li: usize) usize {
        return self.layerKvHeads(li) * self.layerHeadDim(li);
    }

    pub fn layerQkvStride(self: ModelConfig, li: usize) usize {
        const kv_mult: usize = if (self.layer_k_eq_v[li]) 1 else 2;
        return self.layerQProjOut(li) + kv_mult * self.layerKvProjOut(li);
    }

    pub fn mlaQHeadDim(self: ModelConfig) usize {
        return self.qk_nope_head_dim + self.qk_rope_head_dim;
    }

    pub fn mlaKvAOut(self: ModelConfig) usize {
        return self.kv_lora_rank + self.qk_rope_head_dim;
    }

    pub fn mlaKvBOut(self: ModelConfig) usize {
        return self.num_heads * (self.qk_nope_head_dim + self.v_head_dim);
    }

    pub fn maxHeadDim(self: ModelConfig) usize {
        if (self.use_mla) return @max(self.mlaQHeadDim(), self.v_head_dim);
        return @max(self.head_dim, self.global_head_dim);
    }

    pub fn maxKvWidth(self: ModelConfig) usize {
        if (self.use_mla) return self.num_heads * @max(self.mlaQHeadDim(), self.v_head_dim);
        var m = self.kvProjOut();
        for (0..self.num_layers) |li| {
            const w = self.layerKvProjOut(li);
            if (w > m) m = w;
        }
        return m;
    }

    pub fn maxAttnQkvOut(self: ModelConfig) usize {
        if (self.use_mla) {
            return @max(self.mlaKvBOut(), @max(self.num_heads * self.mlaQHeadDim(), @max(self.q_lora_rank, self.mlaKvAOut())));
        }
        var m = self.qProjOut() + 2 * self.kvProjOut();
        if (self.is_lfm) m = @max(m, 3 * self.hidden_size);
        for (0..self.num_layers) |li| {
            const w = self.layerQkvStride(li);
            if (w > m) m = w;
        }
        return m;
    }

    pub fn maxAttnOutDim(self: ModelConfig) usize {
        if (self.use_mla) return self.num_heads * @max(self.mlaQHeadDim(), self.v_head_dim);
        return @max(self.num_heads * self.maxHeadDim(), if (self.is_lfm) self.hidden_size else 0);
    }

    pub fn gdnKeyDim(self: ModelConfig) usize {
        return self.gdn_k_heads * self.gdn_k_dim;
    }

    pub fn gdnValDim(self: ModelConfig) usize {
        return self.gdn_v_heads * self.gdn_v_dim;
    }

    pub fn gdnQkvDim(self: ModelConfig) usize {
        return 2 * self.gdnKeyDim() + self.gdnValDim();
    }

    pub fn mambaInterDim(self: ModelConfig) usize {
        return self.mamba_num_heads * self.mamba_head_dim;
    }

    pub fn mambaConvDim(self: ModelConfig) usize {
        return self.mambaInterDim() + 2 * self.mamba_n_groups * self.ssm_state_size;
    }

    pub fn mambaInProjOut(self: ModelConfig) usize {
        return self.mambaInterDim() + self.mambaConvDim() + self.mamba_num_heads;
    }

    pub fn numFullLayers(self: ModelConfig) usize {
        var n: usize = 0;
        for (self.layer_kinds[0..self.num_layers]) |k| if (k == .full) {
            n += 1;
        };
        return n;
    }

    pub fn numLinearLayers(self: ModelConfig) usize {
        var n: usize = 0;
        for (self.layer_kinds[0..self.num_layers]) |k| if (k == .linear) {
            n += 1;
        };
        return n;
    }

    pub fn numMambaLayers(self: ModelConfig) usize {
        var n: usize = 0;
        for (self.layer_kinds[0..self.num_layers]) |k| if (k == .mamba) {
            n += 1;
        };
        return n;
    }

    pub fn numMoeLayers(self: ModelConfig) usize {
        var n: usize = 0;
        for (self.layer_kinds[0..self.num_layers]) |k| if (k == .moe) {
            n += 1;
        };
        return n;
    }

    pub fn numShortConvLayers(self: ModelConfig) usize {
        var n: usize = 0;
        for (self.layer_kinds[0..self.num_layers]) |k| if (k == .short_conv) {
            n += 1;
        };
        return n;
    }

    pub fn isEos(self: ModelConfig, tok: u32) bool {
        for (self.eos_ids[0..self.eos_count]) |e| {
            if (e == tok) return true;
        }
        return false;
    }

    pub fn load(a: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
        const path = try std.fs.path.join(a, &.{ model_dir, "config.json" });
        defer a.free(path);
        const bytes = try safetensors.mapFile(a, path);
        defer safetensors.unmapFile(bytes);

        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        const tc = if (root.get("text_config")) |v| v.object else root;

        const num_layers = getUsize(tc, "num_hidden_layers");
        if (num_layers > max_layers) return error.TooManyLayers;

        const mtype = if (tc.get("model_type")) |mv| mv.string else "qwen3_5";
        const is_nemotron = std.mem.eql(u8, mtype, "nemotron_h");
        const is_llama = std.mem.eql(u8, mtype, "llama");
        const is_mistral = std.mem.eql(u8, mtype, "mistral");
        const is_mixtral = std.mem.eql(u8, mtype, "mixtral");
        const is_qwen2 = std.mem.eql(u8, mtype, "qwen2");
        const is_qwen3 = std.mem.eql(u8, mtype, "qwen3");
        const is_qwen3_moe = std.mem.eql(u8, mtype, "qwen3_moe");
        const is_qwen35_moe = std.mem.eql(u8, mtype, "qwen3_5_moe") or
            std.mem.eql(u8, mtype, "qwen3_5_moe_text") or
            std.mem.eql(u8, mtype, "qwen3_next") or
            std.mem.eql(u8, mtype, "qwen3_6_moe") or
            std.mem.eql(u8, mtype, "qwen3_6_moe_text");
        const is_gemma2 = std.mem.eql(u8, mtype, "gemma2");
        const is_gemma3 = std.mem.eql(u8, mtype, "gemma3") or std.mem.eql(u8, mtype, "gemma3_text");
        const is_gemma4 = std.mem.eql(u8, mtype, "gemma4") or std.mem.eql(u8, mtype, "gemma4_text") or
            std.mem.eql(u8, mtype, "gemma4_unified") or std.mem.eql(u8, mtype, "gemma4_unified_text");
        const is_gemma = is_gemma2 or is_gemma3 or is_gemma4;
        const is_deepseek_v2 = std.mem.eql(u8, mtype, "deepseek_v2");
        const is_deepseek_v3 = std.mem.eql(u8, mtype, "deepseek_v3") or std.mem.eql(u8, mtype, "deepseek_v32");
        const is_glm4 = std.mem.eql(u8, mtype, "glm4") or std.mem.eql(u8, mtype, "glm");
        const is_glm4_moe = std.mem.eql(u8, mtype, "glm4_moe");
        const is_glm_mla = std.mem.eql(u8, mtype, "glm4_moe_lite") or std.mem.eql(u8, mtype, "glm_moe_dsa") or std.mem.eql(u8, mtype, "glm5");
        const is_mla = is_deepseek_v2 or is_deepseek_v3 or is_glm_mla;
        const is_glm = is_glm4 or is_glm4_moe or is_glm_mla;
        const is_lfm2 = std.mem.eql(u8, mtype, "lfm2") or std.mem.eql(u8, mtype, "lfm2_vl");
        const is_lfm2_moe = std.mem.eql(u8, mtype, "lfm2_moe");
        const is_lfm = is_lfm2 or is_lfm2_moe;
        const is_dense_attn = is_qwen2 or is_llama or is_mistral or is_mixtral or is_qwen3 or is_qwen3_moe or is_gemma or is_mla or is_glm or is_lfm;
        const arch: Arch = if (is_nemotron) .nemotron_h else if (is_dense_attn) .qwen2_dense else .qwen35_hybrid;

        const hidden_size = getUsize(tc, "hidden_size");
        const num_heads = getUsize(tc, "num_attention_heads");
        const q_lora_rank: usize = if (is_mla) getUsizeOr(tc, "q_lora_rank", 0) else 0;
        const kv_lora_rank: usize = if (is_mla) getUsizeOr(tc, "kv_lora_rank", 512) else 0;
        const qk_nope_head_dim: usize = if (is_mla) getUsizeOr(tc, "qk_nope_head_dim", 128) else 0;
        const qk_rope_head_dim: usize = if (is_mla) getUsizeOr(tc, "qk_rope_head_dim", 64) else 0;
        const v_head_dim: usize = if (is_mla) getUsizeOr(tc, "v_head_dim", 128) else 0;
        const head_dim = if (is_mla) (qk_nope_head_dim + qk_rope_head_dim) else getUsizeOr(tc, "head_dim", hidden_size / num_heads);
        var rope_theta: f64 = if (is_nemotron or is_mistral or is_gemma2 or is_deepseek_v2 or is_deepseek_v3 or is_glm4 or is_glm4_moe)
            10_000.0
        else if (is_llama)
            500_000.0
        else if (is_dense_attn)
            1_000_000.0
        else
            10_000_000.0;
        var rope_local_theta: f64 = if (tc.get("rope_local_base_freq")) |v| asF64(v) else 10_000.0;
        var global_partial_rotary: f64 = if (is_gemma4) 0.25 else 1.0;
        var partial_rotary: f64 = if (is_glm4 or is_glm4_moe) 0.5 else if (is_dense_attn or is_nemotron) 1.0 else 0.25;
        var rope_type: RopeType = .default;
        var rope_factor: f64 = 1.0;
        var rope_low_freq_factor: f64 = 1.0;
        var rope_high_freq_factor: f64 = 4.0;
        var rope_orig_max_pos: f64 = 8192.0;
        var rope_beta_fast: f64 = 32.0;
        var rope_beta_slow: f64 = 1.0;
        var rope_mscale_cfg: f64 = 0.0;
        var rope_mscale_all_dim: f64 = 0.0;

        if (tc.get("rope_parameters")) |rp| {
            if (rp == .object) {
                const ro = rp.object;
                if (ro.get("full_attention")) |fa| {
                    if (fa == .object) {
                        if (fa.object.get("rope_theta")) |t| rope_theta = asF64(t);
                        if (fa.object.get("partial_rotary_factor")) |p| global_partial_rotary = asF64(p);
                        parseRopeScaling(fa.object, &rope_type, &rope_factor, &rope_low_freq_factor, &rope_high_freq_factor, &rope_orig_max_pos, &rope_beta_fast, &rope_beta_slow, &rope_mscale_cfg, &rope_mscale_all_dim);
                    }
                }
                if (ro.get("sliding_attention")) |sa| {
                    if (sa == .object) {
                        if (sa.object.get("rope_theta")) |t| rope_local_theta = asF64(t);
                    }
                }
                if (ro.get("rope_theta")) |t| rope_theta = asF64(t);
                if (ro.get("partial_rotary_factor")) |p| partial_rotary = asF64(p);
                parseRopeScaling(ro, &rope_type, &rope_factor, &rope_low_freq_factor, &rope_high_freq_factor, &rope_orig_max_pos, &rope_beta_fast, &rope_beta_slow, &rope_mscale_cfg, &rope_mscale_all_dim);
            }
        } else {
            if (tc.get("rope_theta")) |t| rope_theta = asF64(t);
            if (tc.get("partial_rotary_factor")) |p| partial_rotary = asF64(p);
        }
        if (tc.get("rope_scaling")) |rs| {
            if (rs == .object) {
                parseRopeScaling(rs.object, &rope_type, &rope_factor, &rope_low_freq_factor, &rope_high_freq_factor, &rope_orig_max_pos, &rope_beta_fast, &rope_beta_slow, &rope_mscale_cfg, &rope_mscale_all_dim);
            }
        }

        const rms_eps: f32 = @floatCast(if (tc.get("rms_norm_eps")) |e|
            asF64(e)
        else if (tc.get("layer_norm_epsilon")) |e|
            asF64(e)
        else if (tc.get("norm_eps")) |e|
            asF64(e)
        else
            1e-6);

        const is_swiglu_moe = is_mixtral or is_qwen3_moe or is_qwen35_moe or is_glm4_moe or is_lfm2_moe or
            (getUsizeOr(tc, "num_experts", 0) > 0 and !is_nemotron) or
            (getUsizeOr(tc, "num_local_experts", 0) > 0 and !is_nemotron) or
            (getUsizeOr(tc, "n_routed_experts", 0) > 0 and !is_nemotron);
        const moe_style: MoeStyle = if (is_nemotron) .relu2_sigmoid else if (is_swiglu_moe) .swiglu_softmax else .none;
        const scoring_str = if (tc.get("scoring_func")) |v| (if (v == .string) v.string else "") else "";
        const moe_use_sigmoid = if (scoring_str.len > 0)
            std.mem.eql(u8, scoring_str, "sigmoid")
        else
            (is_deepseek_v3 or is_glm4_moe or is_glm_mla or is_lfm2_moe);
        const n_routed: usize = if (is_nemotron)
            getUsizeOr(tc, "n_routed_experts", 128)
        else if (is_swiglu_moe)
            getUsizeOr(tc, "num_experts", getUsizeOr(tc, "num_local_experts", getUsizeOr(tc, "n_routed_experts", 0)))
        else
            0;
        const top_k: usize = if (is_nemotron)
            getUsizeOr(tc, "num_experts_per_tok", 6)
        else if (is_swiglu_moe)
            getUsizeOr(tc, "num_experts_per_tok", 2)
        else
            0;
        const moe_inter: usize = if (is_nemotron)
            getUsizeOr(tc, "moe_intermediate_size", 1856)
        else if (is_swiglu_moe)
            getUsizeOr(tc, "moe_intermediate_size", getUsizeOr(tc, "intermediate_size", 0))
        else
            0;
        const n_sh_exp = getUsizeOr(tc, "n_shared_experts", 0);
        const shared_inter: usize = if (is_nemotron)
            getUsizeOr(tc, "moe_shared_expert_intermediate_size", 3712)
        else if (is_swiglu_moe)
            getUsizeOr(tc, "shared_expert_intermediate_size", getUsizeOr(tc, "moe_shared_expert_intermediate_size", n_sh_exp * moe_inter))
        else
            0;

        // MLA YaRN mscale attention scaling & rotary scaling
        var mla_attn_scale: f32 = 0.0;
        var rope_mscale_val: f32 = 1.0;
        if (is_mla) {
            const q_hd_f: f64 = @floatFromInt(qk_nope_head_dim + qk_rope_head_dim);
            var mscale_all: f64 = 1.0;
            if (rope_mscale_all_dim > 0.0 and rope_factor > 1.0) {
                mscale_all = 0.1 * rope_mscale_all_dim * @log(rope_factor) + 1.0;
            }
            mla_attn_scale = @floatCast((1.0 / @sqrt(q_hd_f)) * mscale_all * mscale_all);
            if (rope_type == .yarn and rope_factor > 1.0 and rope_mscale_cfg > 0.0) {
                const ms = 0.1 * rope_mscale_cfg * @log(rope_factor) + 1.0;
                rope_mscale_val = @floatCast(ms / mscale_all);
            }
        }
        const rope_interleaved = if (is_mla)
            true
        else if (is_glm4)
            (if (tc.get("rope_traditional")) |v| (if (v == .bool) v.bool else true) else true)
        else
            false;
        const glm4_moe_qk_norm = is_glm4_moe and (if (tc.get("use_qk_norm")) |v| (v == .bool and v.bool) else false);

        const use_sw = if (tc.get("use_sliding_window")) |v| (v == .bool and v.bool) else (is_mistral or is_mixtral or is_gemma);
        const sw_val = if (use_sw) getUsizeOr(tc, "sliding_window", if (is_gemma2) 4096 else if (is_gemma) 1024 else 0) else 0;
        const embed_scale: f32 = if (is_gemma) blk: {
            const raw_s: f32 = @sqrt(@as(f32, @floatFromInt(hidden_size)));
            const bits_u: u32 = @bitCast(raw_s);
            break :blk @bitCast((bits_u +% 0x7FFF +% ((bits_u >> 16) & 1)) & 0xFFFF0000);
        } else 1.0;
        const q_pre_scalar: f32 = if (is_gemma4)
            1.0
        else if (tc.get("query_pre_attn_scalar")) |v|
            (if (v == .null) 0.0 else @floatCast(asF64(v)))
        else
            0.0;
        const final_cap: f32 = if (tc.get("final_logit_softcapping")) |v|
            (if (v == .null) 0.0 else @floatCast(asF64(v)))
        else if (is_gemma2 or is_gemma4)
            30.0
        else
            0.0;
        const attn_cap: f32 = if (tc.get("attn_logit_softcapping")) |v|
            (if (v == .null) 0.0 else @floatCast(asF64(v)))
        else if (is_gemma2)
            50.0
        else
            0.0;
        const act_str = if (tc.get("hidden_activation")) |v| (if (v == .string) v.string else "") else if (tc.get("hidden_act")) |v| (if (v == .string) v.string else "") else "";
        const use_geglu = is_gemma or std.mem.indexOf(u8, act_str, "gelu") != null;
        const global_hd: usize = if (is_gemma4) getUsizeOr(tc, "global_head_dim", 512) else head_dim;
        const num_kv_h = getUsizeOr(tc, "num_key_value_heads", num_heads);
        const global_kv_h: usize = if (is_gemma4) getUsizeOr(tc, "num_global_key_value_heads", num_kv_h) else num_kv_h;

        // LFM2 optional auto_adjust_ff_dim intermediate size calculation
        var inter_size: usize = getUsizeOr(tc, "intermediate_size", getUsizeOr(tc, "block_ff_dim", moe_inter));
        if (is_lfm2) {
            const auto_adj = if (tc.get("block_auto_adjust_ff_dim")) |v| (v == .bool and v.bool) else false;
            if (auto_adj) {
                var ff_dim: usize = (2 * getUsizeOr(tc, "block_ff_dim", inter_size)) / 3;
                if (tc.get("block_ffn_dim_multiplier")) |mv| {
                    if (mv != .null) ff_dim = @intFromFloat(asF64(mv) * @as(f64, @floatFromInt(ff_dim)));
                }
                const mult_of = @max(@as(usize, 1), getUsizeOr(tc, "block_multiple_of", 256));
                inter_size = mult_of * ((ff_dim + mult_of - 1) / mult_of);
            }
        }

        var cfg = ModelConfig{
            .arch = arch,
            .hidden_size = hidden_size,
            .intermediate_size = inter_size,
            .num_layers = num_layers,
            .vocab_size = getUsize(tc, "vocab_size"),
            .num_heads = num_heads,
            .num_kv_heads = num_kv_h,
            .head_dim = head_dim,
            .gdn_k_heads = if (arch == .qwen35_hybrid) getUsizeOr(tc, "linear_num_key_heads", 16) else 0,
            .gdn_v_heads = if (arch == .qwen35_hybrid) getUsizeOr(tc, "linear_num_value_heads", 16) else 0,
            .gdn_k_dim = getUsizeOr(tc, "linear_key_head_dim", 128),
            .gdn_v_dim = getUsizeOr(tc, "linear_value_head_dim", 128),
            .conv_kernel = if (is_nemotron) getUsizeOr(tc, "conv_kernel", 4) else getUsizeOr(tc, "linear_conv_kernel_dim", 4),
            .mamba_num_heads = if (is_nemotron) getUsizeOr(tc, "mamba_num_heads", 64) else 0,
            .mamba_head_dim = if (is_nemotron) getUsizeOr(tc, "mamba_head_dim", 64) else 0,
            .ssm_state_size = if (is_nemotron) getUsizeOr(tc, "ssm_state_size", 128) else 0,
            .mamba_n_groups = if (is_nemotron) getUsizeOr(tc, "n_groups", 8) else 0,
            .moe_style = moe_style,
            .moe_use_sigmoid = moe_use_sigmoid,
            .n_routed_experts = n_routed,
            .num_experts_per_tok = top_k,
            .moe_intermediate_size = moe_inter,
            .moe_shared_intermediate_size = shared_inter,
            .has_shared_expert = shared_inter > 0,
            .shared_expert_gated = is_qwen35_moe,
            .routed_scaling_factor = if (tc.get("routed_scaling_factor")) |v| @floatCast(asF64(v)) else 1.0,
            .norm_topk_prob = if (tc.get("norm_topk_prob")) |v| (if (v == .bool) v.bool else true) else true,
            .n_group = getUsizeOr(tc, "n_group", 0),
            .topk_group = getUsizeOr(tc, "topk_group", 0),
            .use_mla = is_mla,
            .q_lora_rank = q_lora_rank,
            .kv_lora_rank = kv_lora_rank,
            .qk_nope_head_dim = qk_nope_head_dim,
            .qk_rope_head_dim = qk_rope_head_dim,
            .v_head_dim = v_head_dim,
            .mla_attn_scale = mla_attn_scale,
            .num_nextn_predict_layers = getUsizeOr(tc, "num_nextn_predict_layers", 0),
            .rms_eps = rms_eps,
            .rope_dims = if (is_mla) qk_rope_head_dim else @intFromFloat(@round(@as(f64, @floatFromInt(head_dim)) * partial_rotary)),
            .rope_theta = rope_theta,
            .use_rope = !is_nemotron,
            .rope_interleaved = rope_interleaved,
            .rope_mscale = rope_mscale_val,
            .rope_type = rope_type,
            .rope_factor = rope_factor,
            .rope_low_freq_factor = rope_low_freq_factor,
            .rope_high_freq_factor = rope_high_freq_factor,
            .rope_orig_max_pos = rope_orig_max_pos,
            .rope_beta_fast = rope_beta_fast,
            .rope_beta_slow = rope_beta_slow,
            .sliding_window = sw_val,
            .attn_softcap = attn_cap,
            .final_logit_softcap = final_cap,
            .is_gemma = is_gemma,
            .is_gemma4 = is_gemma4,
            .glm4_post_norms = is_glm4,
            .is_lfm = is_lfm,
            .conv_l_cache = getUsizeOr(tc, "conv_L_cache", 3),
            .conv_bias = if (tc.get("conv_bias")) |v| (v == .bool and v.bool) else false,
            .embed_scale = embed_scale,
            .query_pre_attn_scalar = q_pre_scalar,
            .rms_norm_add_one = is_gemma2 or is_gemma3,
            .has_post_norms = is_gemma or is_glm4,
            .use_geglu = use_geglu,
            .has_v_norm = is_gemma4,
            .has_dual_rope = is_gemma3 or is_gemma4,
            .rope_local_theta = rope_local_theta,
            .global_head_dim = global_hd,
            .global_kv_heads = global_kv_h,
            .global_partial_rotary = global_partial_rotary,
            .tie_word_embeddings = if (tc.get("tie_word_embeddings")) |v| v.bool else (if (root.get("tie_word_embeddings")) |v| v.bool else (is_gemma or is_lfm)),
            .attn_output_gate = if (arch == .qwen35_hybrid) (if (tc.get("attn_output_gate")) |v| v.bool else true) else false,
            .has_qk_norm = (arch == .qwen35_hybrid) or is_qwen3 or is_qwen3_moe or is_gemma3 or is_gemma4 or glm4_moe_qk_norm or is_lfm,
            .has_qkv_bias = if (tc.get("attention_bias")) |v| (v == .bool and v.bool) else (is_qwen2 or is_glm4),
            .bos_token_id = if (tc.get("bos_token_id")) |v| (if (v == .integer) @as(u32, @intCast(v.integer)) else null) else null,
            .max_position = getUsizeOr(tc, "max_position_embeddings", 262144),
        };

        // Per-layer sliding vs global attention schedule & Gemma 4 per-layer head_dim/kv_heads/k_eq_v
        const k_eq_v_global = if (tc.get("attention_k_eq_v")) |v| (v == .bool and v.bool) else false;
        const sw_pattern = getUsizeOr(tc, "sliding_window_pattern", if (is_gemma2) 2 else 6);
        for (0..num_layers) |li| {
            cfg.layer_head_dim[li] = head_dim;
            cfg.layer_kv_heads[li] = num_kv_h;
            if (is_gemma) {
                const is_sliding = if (tc.get("layer_types")) |lt|
                    (li < lt.array.items.len and std.mem.eql(u8, lt.array.items[li].string, "sliding_attention"))
                else if (is_gemma2)
                    (li % 2 == 0)
                else
                    ((li + 1) % sw_pattern != 0);
                cfg.layer_is_sliding[li] = is_sliding;
                if (is_gemma4 and !is_sliding) {
                    cfg.layer_head_dim[li] = global_hd;
                    cfg.layer_kv_heads[li] = global_kv_h;
                    cfg.layer_k_eq_v[li] = k_eq_v_global;
                }
            } else if (sw_val > 0) {
                cfg.layer_is_sliding[li] = true;
            }
        }

        if (moe_style == .swiglu_softmax) {
            const first_k_dense = getUsizeOr(tc, "first_k_dense_replace", getUsizeOr(tc, "num_dense_layers", 0));
            const moe_freq = @max(@as(usize, 1), getUsizeOr(tc, "moe_layer_freq", getUsizeOr(tc, "decoder_sparse_step", 1)));
            for (0..num_layers) |li| {
                cfg.layer_is_moe[li] = (li >= first_k_dense) and
                    (if (is_mla or is_glm4_moe or is_lfm2_moe) (li % moe_freq == 0) else ((li + 1) % moe_freq == 0));
            }
            if (tc.get("mlp_only_layers")) |mol| {
                if (mol == .array) {
                    for (mol.array.items) |item| {
                        const idx: usize = @intCast(item.integer);
                        if (idx < num_layers) cfg.layer_is_moe[idx] = false;
                    }
                }
            }
        }

        if (tc.get("time_step_limit")) |tsl| {
            if (tsl == .array and tsl.array.items.len >= 2) {
                cfg.dt_min = @floatCast(asF64(tsl.array.items[0]));
                cfg.dt_max = @floatCast(asF64(tsl.array.items[1]));
            }
        }

        if (root.get("quantization") orelse root.get("quantization_config")) |qv| {
            if (qv == .object) {
                const qo = qv.object;
                cfg.has_quant_config = true;
                cfg.bits = getUsizeOr(qo, "bits", 4);
                cfg.group_size = getUsizeOr(qo, "group_size", 64);
                if (qo.get("mode")) |mv| {
                    if (mv == .string and std.mem.eql(u8, mv.string, "tfq3")) {
                        cfg.quant_format = .tfq3;
                    }
                }
            }
        } else {
            cfg.quant_format = .bf16;
        }

        // Layer schedule
        if (is_nemotron) {
            if (tc.get("layers_block_type")) |lbt| {
                for (lbt.array.items, 0..) |item, i| {
                    if (i >= num_layers) break;
                    const s = item.string;
                    cfg.layer_kinds[i] = if (std.mem.eql(u8, s, "mamba"))
                        .mamba
                    else if (std.mem.eql(u8, s, "moe"))
                        .moe
                    else
                        .full;
                }
            } else if (tc.get("hybrid_override_pattern")) |hop| {
                switch (hop) {
                    .string => |pat| {
                        for (pat, 0..) |ch, i| {
                            if (i >= num_layers) break;
                            cfg.layer_kinds[i] = if (ch == 'M') .mamba else if (ch == 'E') .moe else .full;
                        }
                    },
                    .array => |arr| {
                        for (arr.items, 0..) |item, i| {
                            if (i >= num_layers) break;
                            const ch = item.string[0];
                            cfg.layer_kinds[i] = if (ch == 'M') .mamba else if (ch == 'E') .moe else .full;
                        }
                    },
                    else => {},
                }
            }
        } else if (is_lfm) {
            for (0..num_layers) |i| cfg.layer_kinds[i] = .short_conv;
            if (tc.get("layer_types")) |lt| {
                if (lt == .array) {
                    for (lt.array.items, 0..) |item, i| {
                        if (i >= num_layers) break;
                        cfg.layer_kinds[i] = if (std.mem.eql(u8, item.string, "full_attention")) .full else .short_conv;
                    }
                }
            } else if (tc.get("full_attn_idxs")) |fi| {
                if (fi == .array) {
                    for (fi.array.items) |item| {
                        const idx: usize = @intCast(item.integer);
                        if (idx < num_layers) cfg.layer_kinds[idx] = .full;
                    }
                }
            }
        } else if (is_dense_attn) {
            for (0..num_layers) |i| cfg.layer_kinds[i] = .full;
        } else if (tc.get("layer_types")) |lt| {
            for (lt.array.items, 0..) |item, i| {
                if (i >= num_layers) break;
                cfg.layer_kinds[i] = if (std.mem.eql(u8, item.string, "full_attention")) .full else .linear;
            }
        } else {
            const interval = getUsizeOr(tc, "full_attention_interval", 4);
            for (0..num_layers) |i| {
                cfg.layer_kinds[i] = if ((i + 1) % interval == 0) .full else .linear;
            }
        }

        // EOS token IDs
        if (tc.get("eos_token_id")) |ev| {
            switch (ev) {
                .integer => |id| addEos(&cfg, @intCast(id)),
                .array => |arr| {
                    for (arr.items) |item| addEos(&cfg, @intCast(item.integer));
                },
                else => {},
            }
        }
        if (is_nemotron) {
            addEos(&cfg, 2);
            addEos(&cfg, 11); // Nemotron-3.5 <|im_end|>
        } else if (is_qwen2 or is_qwen3 or is_qwen3_moe) {
            addEos(&cfg, 151645); // Qwen2/3 <|im_end|>
        } else if (is_llama) {
            if (cfg.eos_count == 0) addEos(&cfg, 128001);
            if (cfg.vocab_size > 128009) addEos(&cfg, 128009); // Llama 3 <|eot_id|>
        } else if (is_mistral or is_mixtral) {
            if (cfg.eos_count == 0) addEos(&cfg, 2);
        } else if (is_gemma) {
            if (cfg.eos_count == 0) addEos(&cfg, 1);
            if (cfg.vocab_size > 106) addEos(&cfg, 106); // Gemma <end_of_turn>
        } else if (is_deepseek_v2 or is_deepseek_v3) {
            if (cfg.eos_count == 0) addEos(&cfg, 1);
        } else if (is_glm) {
            if (cfg.eos_count == 0) addEos(&cfg, 151329);
        } else {
            addEos(&cfg, 248046); // Qwen3.5/3.8 <|im_end|>
            addEos(&cfg, 248044);
        }

        return cfg;
    }
};

fn parseRopeScaling(
    ro: std.json.ObjectMap,
    rope_type: *RopeType,
    factor: *f64,
    low_f: *f64,
    high_f: *f64,
    orig_max: *f64,
    beta_fast: *f64,
    beta_slow: *f64,
    mscale: *f64,
    mscale_all_dim: *f64,
) void {
    const rt_val = ro.get("rope_type") orelse ro.get("type");
    if (rt_val) |rv| {
        if (rv == .string) {
            if (std.mem.eql(u8, rv.string, "llama3")) {
                rope_type.* = .llama3;
            } else if (std.mem.eql(u8, rv.string, "linear")) {
                rope_type.* = .linear;
            } else if (std.mem.eql(u8, rv.string, "yarn")) {
                rope_type.* = .yarn;
            }
        }
    }
    if (ro.get("factor")) |v| if (v != .null) {
        factor.* = asF64(v);
    };
    if (ro.get("low_freq_factor")) |v| if (v != .null) {
        low_f.* = asF64(v);
    };
    if (ro.get("high_freq_factor")) |v| if (v != .null) {
        high_f.* = asF64(v);
    };
    if (ro.get("original_max_position_embeddings")) |v| if (v != .null) {
        orig_max.* = asF64(v);
    };
    if (ro.get("beta_fast")) |v| if (v != .null) {
        beta_fast.* = asF64(v);
    };
    if (ro.get("beta_slow")) |v| if (v != .null) {
        beta_slow.* = asF64(v);
    };
    if (ro.get("mscale")) |v| if (v != .null) {
        mscale.* = asF64(v);
    };
    if (ro.get("mscale_all_dim")) |v| if (v != .null) {
        mscale_all_dim.* = asF64(v);
    };
}

fn addEos(cfg: *ModelConfig, id: u32) void {
    if (cfg.isEos(id) or cfg.eos_count >= max_eos) return;
    cfg.eos_ids[cfg.eos_count] = id;
    cfg.eos_count += 1;
}

fn getUsize(obj: std.json.ObjectMap, key: []const u8) usize {
    return @intCast(obj.get(key).?.integer);
}

fn getUsizeOr(obj: std.json.ObjectMap, key: []const u8, fallback: usize) usize {
    const v = obj.get(key) orelse return fallback;
    if (v == .null) return fallback;
    return @intCast(v.integer);
}

fn asF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}
