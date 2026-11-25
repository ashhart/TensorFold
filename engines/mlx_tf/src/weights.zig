//! Parallel weight loader & qk tile packer for Qwen3.5 / Qwen3.8 MLX 4-bit checkpoints.
//!
//! Fuses shared-input projections vertically at load time into single qk matrices:
//!   - GDN layers:  in_proj_all = [in_proj_qkv ; in_proj_z ; in_proj_b ; in_proj_a]
//!   - Attn layers: qkv_proj    = [q_proj ; k_proj ; v_proj]
//!   - All layers:  gate_up     = [gate_proj ; up_proj]
const std = @import("std");
const mtl = @import("metal4.zig");
const st = @import("safetensors.zig");
const cfg_mod = @import("config.zig");
const pcache = @import("pack_cache.zig");
const tfq3_mod = @import("tfq3.zig");

const ModelConfig = cfg_mod.ModelConfig;
const LayerKind = cfg_mod.LayerKind;
const Buffer = mtl.Buffer;
const BufferRef = mtl.BufferRef;

extern "c" fn dispatch_apply_f(
    iterations: usize,
    queue: ?*anyopaque,
    context: ?*anyopaque,
    work: *const fn (?*anyopaque, usize) callconv(.c) void,
) void;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

pub fn preferBf16() bool {
    const v = getenv("MLX_TF_BF16") orelse return false;
    const s = std.mem.span(v);
    return std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true");
}

pub const qk_nt: usize = 32;
pub const qk_gs: usize = 64;
pub const qk_codes_per_block: usize = qk_nt * qk_gs / 2; // 1024 bytes
pub const qk_block_bytes: usize = qk_codes_per_block + qk_nt * 4; // 1152 bytes (codes + interleaved scale/bias)

pub const SbDType = enum(u32) {
    bf16 = 0,
    f16 = 1,
    f32 = 2,

    pub fn elemBytes(self: SbDType) usize {
        return if (self == .f32) 4 else 2;
    }

    pub fn fromSt(dt: st.DType) SbDType {
        return switch (dt) {
            .bf16 => .bf16,
            .f16 => .f16,
            .f32 => .f32,
            else => .bf16,
        };
    }
};

pub const QkMatrix = struct {
    ref: BufferRef,
    n: u32,
    k: u32,

    pub fn groups(self: QkMatrix) u32 {
        return self.k / @as(u32, qk_gs);
    }

    pub fn tiles(self: QkMatrix) u32 {
        return self.n / @as(u32, qk_nt);
    }

    pub fn byteSize(n: usize, k: usize) usize {
        std.debug.assert(n % qk_nt == 0);
        std.debug.assert(k % qk_gs == 0);
        const tiles_n = n / qk_nt;
        const g = k / qk_gs;
        // +1 block slack for lookahead scale/bias prefetch in qk.metal
        return (tiles_n * g + 1) * qk_block_bytes;
    }
};

pub const AffineMatrix = struct {
    codes: BufferRef, // U32 [n, k * bits / 32]
    scales: BufferRef, // BF16/F16/F32 [n, k / gs]
    biases: BufferRef, // BF16/F16/F32 [n, k / gs]
    n: u32,
    k: u32,
    bits: u8, // 2, 3, 4, 5, 6, 8
    gs: u16, // 32, 64, 128
    sb_dtype: SbDType = .bf16,
};

pub const Bf16Matrix = struct {
    ref: BufferRef, // BF16 [n, k]
    n: u32,
    k: u32,
};

pub const Tfq3Matrix = struct {
    words: BufferRef, // U32 packed TFQ3 stream
    scales: BufferRef, // F16 [n/128, k/128]
    tables: BufferRef, // PWV3 16 KiB codebook
    n: u32,
    k: u32,
    bits: u8, // 3, 4, 5
    sr_split: u32 = 0, // >0 when pair-fused [R0, C] + [R1, C]
};

pub const SubLinear = union(enum) {
    affine: AffineMatrix,
    bf16: Bf16Matrix,
    tfq3: Tfq3Matrix,

    pub fn rowsOut(self: SubLinear) u32 {
        return switch (self) {
            .affine => |m| m.n,
            .bf16 => |m| m.n,
            .tfq3 => |m| m.n,
        };
    }

    pub fn colsIn(self: SubLinear) u32 {
        return switch (self) {
            .affine => |m| m.k,
            .bf16 => |m| m.k,
            .tfq3 => |m| m.k,
        };
    }
};

pub const ConcatLinear = struct {
    parts: [4]SubLinear = undefined,
    col_offs: [4]u32 = @splat(0),
    count: u8 = 0,
    n: u32 = 0,
    k: u32 = 0,
};

pub const LinearWeight = union(enum) {
    qk4: QkMatrix,
    affine: AffineMatrix,
    bf16: Bf16Matrix,
    tfq3: Tfq3Matrix,
    concat: ConcatLinear,

    pub fn isQk4(self: LinearWeight) bool {
        return self == .qk4;
    }

    pub fn rowsOut(self: LinearWeight) u32 {
        return switch (self) {
            .qk4 => |m| m.n,
            .affine => |m| m.n,
            .bf16 => |m| m.n,
            .tfq3 => |m| m.n,
            .concat => |m| m.n,
        };
    }

    pub fn colsIn(self: LinearWeight) u32 {
        return switch (self) {
            .qk4 => |m| m.k,
            .affine => |m| m.k,
            .bf16 => |m| m.k,
            .tfq3 => |m| m.k,
            .concat => |m| m.k,
        };
    }

    pub fn tiles32(self: LinearWeight) u32 {
        return (self.rowsOut() + 31) / 32;
    }
};

pub const EmbedWeights = struct {
    codes: BufferRef, // U32 [vocab, hidden*bits/32] OR BF16/F16/F32 [vocab, hidden] when bits == 0
    scales: BufferRef, // unused when bits == 0
    biases: BufferRef, // unused when bits == 0
    bits: u8 = 4,
    gs: u16 = 64,
    sb_dtype: SbDType = .bf16,

    pub fn isFastQk4(self: EmbedWeights) bool {
        return self.bits == 4 and self.gs == 64 and self.sb_dtype == .bf16;
    }
};

pub const LinearLayerWeights = struct {
    in_proj_all: LinearWeight, // [qkv + z + b + a, hidden]
    out_proj: LinearWeight, // [hidden, gdnValDim]
    conv_weight: BufferRef, // BF16 [gdnQkvDim, 4]
    a_log: BufferRef, // F32 [gdn_v_heads]
    dt_bias: BufferRef, // BF16 [gdn_v_heads]
    gdn_norm: BufferRef, // BF16 [gdn_v_dim]
};

pub const FullLayerWeights = struct {
    qkv_proj: LinearWeight, // [q_out + k_out + v_out, hidden]
    o_proj: LinearWeight, // [hidden, num_heads * head_dim]
    q_norm: BufferRef, // BF16 [head_dim] (unused when !has_qk_norm)
    k_norm: BufferRef, // BF16 [head_dim] (unused when !has_qk_norm)
    qkv_bias: ?BufferRef = null, // Optional BF16 [q_out + k_out + v_out] (Qwen2/Qwen2.5/GLM-4)
};

pub const MlaLayerWeights = struct {
    has_q_lora: bool = false,
    q_proj: LinearWeight = undefined, // [num_heads * q_hd, hidden] when q_lora_rank == 0
    q_a_proj: LinearWeight = undefined, // [q_lora_rank, hidden] when q_lora_rank > 0
    q_a_norm: BufferRef = undefined, // BF16 [q_lora_rank]
    q_b_proj: LinearWeight = undefined, // [num_heads * q_hd, q_lora_rank]
    kv_a_proj: LinearWeight = undefined, // [kv_lora_rank + qk_rope_head_dim, hidden]
    kv_a_norm: BufferRef = undefined, // BF16 [kv_lora_rank]
    kv_b_proj: LinearWeight = undefined, // [num_heads * (qk_nope + v_dim), kv_lora_rank]
    o_proj: LinearWeight = undefined, // [hidden, num_heads * v_dim]
    q_a_bias: ?BufferRef = null,
    kv_a_bias: ?BufferRef = null,
};

pub const ShortConvLayerWeights = struct {
    in_proj: LinearWeight = undefined, // [3 * hidden, hidden]
    out_proj: LinearWeight = undefined, // [hidden, hidden]
    conv_weight: BufferRef = undefined, // BF16 [hidden, conv_l_cache]
    in_bias: ?BufferRef = null, // optional BF16 [3 * hidden]
    conv_bias: ?BufferRef = null, // optional BF16 [hidden]
    out_bias: ?BufferRef = null, // optional BF16 [hidden]
};

pub const MambaLayerWeights = struct {
    in_proj: LinearWeight, // [inter + conv_dim + num_heads, hidden]
    out_proj: LinearWeight, // [hidden, inter]
    conv_weight: BufferRef, // BF16 [conv_dim, 4]
    conv_bias: BufferRef, // BF16 [conv_dim]
    a_log: BufferRef, // F32 [num_heads]
    dt_bias: BufferRef, // BF16 [num_heads]
    d_skip: BufferRef, // BF16 [num_heads]
    mamba_norm: BufferRef, // BF16 [inter]
};

pub const MoeLayerWeights = struct {
    gate_weight: BufferRef, // BF16 [n_experts, hidden]
    gate_bias: BufferRef, // F32 [n_experts]
    shared_up: LinearWeight, // [shared_inter, hidden]
    shared_down: LinearWeight, // [hidden, shared_inter]
    fc1_experts: BufferRef, // [n_experts] packed qk [moe_inter, hidden]
    fc2_experts: BufferRef, // [n_experts] packed qk [hidden, moe_inter]
    fc1_expert_stride: usize,
    fc2_expert_stride: usize,
};

pub const ExpertStorage = enum { qk4, affine, bf16 };

pub const SwigluMoeWeights = struct {
    gate_weight: BufferRef, // BF16 [n_experts, hidden]
    gate_bias: ?BufferRef = null, // optional F32 [n_experts]
    storage: ExpertStorage = .bf16,
    bits: u8 = 4,
    gs: u16 = 64,
    sb_dtype: SbDType = .bf16,
    gate_codes: BufferRef,
    gate_scales: BufferRef,
    gate_biases: BufferRef,
    up_codes: BufferRef,
    up_scales: BufferRef,
    up_biases: BufferRef,
    down_codes: BufferRef,
    down_scales: BufferRef,
    down_biases: BufferRef,
    gu_code_stride: usize = 0,
    gu_sb_stride: usize = 0,
    dn_code_stride: usize = 0,
    dn_sb_stride: usize = 0,
    has_shared: bool = false,
    shared_gate_up: LinearWeight = undefined,
    shared_down: LinearWeight = undefined,
    shared_gate_vec: ?BufferRef = null, // optional BF16 [hidden] (Qwen3.5 MoE)
};

pub const LayerWeights = struct {
    kind: LayerKind,
    is_moe: bool = false,
    input_norm: BufferRef, // BF16 [hidden]
    post_attn_norm: BufferRef, // BF16 [hidden]
    pre_ff_norm: BufferRef = undefined, // Gemma 2/3/4 pre_feedforward_layernorm
    post_ff_norm: BufferRef = undefined, // Gemma 2/3/4 post_feedforward_layernorm
    layer_scalar: BufferRef = undefined, // Gemma 4 optional layer_scalar [1]
    has_layer_scalar: bool = false,
    gate_up: LinearWeight = undefined, // [2 * intermediate, hidden] (dense MLP)
    down_proj: LinearWeight = undefined, // [hidden, intermediate] (dense MLP)
    swiglu_moe: SwigluMoeWeights = undefined,
    linear: LinearLayerWeights = undefined,
    full: FullLayerWeights = undefined,
    mla: MlaLayerWeights = undefined,
    short_conv: ShortConvLayerWeights = undefined,
    mamba: MambaLayerWeights = undefined,
    moe: MoeLayerWeights = undefined,
};

pub const ModelWeights = struct {
    a: std.mem.Allocator,
    buffers: std.ArrayList(Buffer) = .empty,
    embed: EmbedWeights,
    final_norm: BufferRef,
    lm_head: LinearWeight,
    layers: []LayerWeights,
    total_bytes: usize = 0,
    mmap_backing: ?[]u8 = null,
    tfq3_store: ?tfq3_mod.Store = null,

    pub fn isPureQk4(self: *const ModelWeights, c: ModelConfig) bool {
        if (self.tfq3_store != null) return false;
        if (c.moe_style == .swiglu_softmax or c.is_gemma or c.use_mla or c.has_post_norms or c.is_lfm) return false;
        if (!self.embed.isFastQk4() or !self.lm_head.isQk4()) return false;
        for (self.layers) |lw| {
            if (lw.is_moe) return false;
            if (c.arch != .nemotron_h) {
                if (!lw.gate_up.isQk4() or !lw.down_proj.isQk4()) return false;
            }
            switch (lw.kind) {
                .linear => if (!lw.linear.in_proj_all.isQk4() or !lw.linear.out_proj.isQk4()) return false,
                .full => if (!lw.full.qkv_proj.isQk4() or !lw.full.o_proj.isQk4()) return false,
                .short_conv => return false,
                .mamba => if (!lw.mamba.in_proj.isQk4() or !lw.mamba.out_proj.isQk4()) return false,
                .moe => if (!lw.moe.shared_up.isQk4() or !lw.moe.shared_down.isQk4()) return false,
            }
        }
        return true;
    }

    pub fn deinit(self: *ModelWeights) void {
        self.a.free(self.layers);
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.a);
        if (self.mmap_backing) |mb| pcache.unmap(mb);
        if (self.tfq3_store) |*ts| ts.deinit();
    }

    pub fn load(a: std.mem.Allocator, ctx: *mtl.Context, c: ModelConfig, model_dir: []const u8) !ModelWeights {
        if (!preferBf16() and c.moe_style != .swiglu_softmax and !c.is_gemma and !c.use_mla and !c.has_post_norms and !c.is_lfm and c.quant_format == .affine and c.bits == 4 and c.group_size == 64) {
            if (pcache.tryLoad(a, ctx, c, model_dir)) |cached| {
                return cached;
            }
        }

        const tfq3_opt = try tfq3_mod.Store.open(a, ctx, model_dir);
        var ckpt = try st.Checkpoint.openDir(a, model_dir);
        defer ckpt.close();

        var w = ModelWeights{
            .a = a,
            .embed = undefined,
            .final_norm = undefined,
            .lm_head = undefined,
            .layers = try a.alloc(LayerWeights, c.num_layers),
            .tfq3_store = tfq3_opt,
        };
        if (w.tfq3_store) |*ts| w.total_bytes += ts.total_bytes;
        errdefer w.deinit();

        const ts_ptr: ?*const tfq3_mod.Store = if (w.tfq3_store) |*ts| ts else null;
        const keep_bf16 = preferBf16() or c.has_quant_config or (ts_ptr != null);

        if (c.arch == .nemotron_h) {
            try loadNemotronH(&w, ctx, c, &ckpt, ts_ptr, keep_bf16);
            if (w.isPureQk4(c)) pcache.save(a, c, model_dir, &w) catch {};
            return w;
        }

        // Detect tensor prefix ("language_model.model." vs "model.language_model." vs "model.")
        const prefix: []const u8 = if (ckpt.getOptional("language_model.model.norm.weight") != null or ckpt.getOptional("language_model.model.embedding_norm.weight") != null)
            "language_model.model."
        else if (ckpt.getOptional("model.language_model.norm.weight") != null or ckpt.getOptional("model.language_model.embedding_norm.weight") != null)
            "model.language_model."
        else
            "model.";
        const head_prefix: []const u8 = if (ckpt.getOptional("language_model.lm_head.weight") != null)
            "language_model."
        else
            "";
        const tfq3_head = if (ts_ptr) |ts| ts.find("lm_head", c.vocab_size, c.hidden_size) else null;
        const tied_head = (tfq3_head == null) and (c.tie_word_embeddings or (getPrefixedOpt(&ckpt, head_prefix, "lm_head.weight") == null));

        // Detect raw HuggingFace Qwen3.5/3.8 checkpoint (unsanitized PyTorch conv1d [C, 1, 4] & zero-centered RMSNorms)
        var shift_norms = false;
        if (c.arch == .qwen35_hybrid) {
            var chk_buf: [160]u8 = undefined;
            for (0..c.num_layers) |li| {
                if (c.layer_kinds[li] == .linear) {
                    if (getLayer(&ckpt, &chk_buf, prefix, li, "linear_attn.conv1d.weight")) |cw| {
                        if (cw.rank == 3 and cw.shape[2] != 1) shift_norms = true;
                    } else |_| {}
                    break;
                }
            }
        }

        // 1. Embeddings & LM Head (supports 2/3/4/5/6/8-bit g32/64/128, raw BF16/F16/F32, and mixed precision!)
        {
            const ew = try getPrefixed(&ckpt, prefix, "embed_tokens.weight");
            const es_opt = getPrefixedOpt(&ckpt, prefix, "embed_tokens.scales");
            const eb_opt = getPrefixedOpt(&ckpt, prefix, "embed_tokens.biases");

            if (es_opt != null and eb_opt != null) {
                const es = es_opt.?;
                const eb = eb_opt.?;
                const e_groups = es.shape[es.rank - 1];
                const e_gs: u16 = @intCast(c.hidden_size / e_groups);
                const e_words = ew.bytes.len / (c.vocab_size * 4);
                const e_bits: u8 = @intCast((e_words * 32) / c.hidden_size);
                const e_sbd = SbDType.fromSt(es.dtype);
                const total = align64(ew.bytes.len) + align64(es.bytes.len) + align64(eb.bytes.len);
                const buf = try allocTracked(&w, ctx, total);
                var off: usize = 0;
                w.embed = .{
                    .codes = copyInto(buf, &off, ew.bytes),
                    .scales = copyInto(buf, &off, es.bytes),
                    .biases = copyInto(buf, &off, eb.bytes),
                    .bits = e_bits,
                    .gs = e_gs,
                    .sb_dtype = e_sbd,
                };
            } else if (keep_bf16) {
                const buf = try allocTracked(&w, ctx, align64(ew.bytes.len));
                var off: usize = 0;
                const ref = copyInto(buf, &off, ew.bytes);
                w.embed = .{
                    .codes = ref,
                    .scales = ref,
                    .biases = ref,
                    .bits = 0,
                    .gs = 64,
                    .sb_dtype = SbDType.fromSt(ew.dtype),
                };
            } else {
                const codes_len = c.vocab_size * (c.hidden_size / 2);
                const sb_len = c.vocab_size * (c.hidden_size / qk_gs) * 2;
                const total = align64(codes_len) + align64(sb_len) * 2;
                const buf = try allocTracked(&w, ctx, total);
                const codes_off: usize = 0;
                const scales_off: usize = align64(codes_len);
                const biases_off: usize = scales_off + align64(sb_len);
                quantizeEmbedBf16(
                    ew,
                    c.vocab_size,
                    c.hidden_size,
                    buf.ptr + codes_off,
                    @ptrCast(@alignCast(buf.ptr + scales_off)),
                    @ptrCast(@alignCast(buf.ptr + biases_off)),
                );
                w.embed = .{
                    .codes = buf.view(codes_off),
                    .scales = buf.view(scales_off),
                    .biases = buf.view(biases_off),
                    .bits = 4,
                    .gs = 64,
                    .sb_dtype = .bf16,
                };
            }

            if (tfq3_head) |fh| {
                w.lm_head = .{ .tfq3 = fh.tfq3 };
            } else if (tied_head and w.embed.bits == 0 and w.embed.sb_dtype == .bf16) {
                w.lm_head = .{ .bf16 = .{ .ref = w.embed.codes, .n = @intCast(c.vocab_size), .k = @intCast(c.hidden_size) } };
            } else {
                const hw = if (tied_head) ew else try getPrefixed(&ckpt, head_prefix, "lm_head.weight");
                const hs_opt = if (tied_head) es_opt else getPrefixedOpt(&ckpt, head_prefix, "lm_head.scales");
                const hb_opt = if (tied_head) eb_opt else getPrefixedOpt(&ckpt, head_prefix, "lm_head.biases");
                const h_srcs = [1]RawQuant{.{ .w = hw, .s = hs_opt, .b = hb_opt, .n = c.vocab_size }};
                const head_bytes = measureLinear(&h_srcs, c.vocab_size, c.hidden_size, keep_bf16);
                const head_buf = if (head_bytes > 0) try allocTracked(&w, ctx, head_bytes) else null;
                var h_off: usize = 0;
                w.lm_head = packLinearInto(head_buf, &h_off, &h_srcs, c.vocab_size, c.hidden_size, keep_bf16);
            }
        }

        // 2. Small non-quantized weights (norms, conv1d, A_log, dt_bias, optional QKV bias, Gemma/GLM-4 post-norms, MLA norms, LFM2 ShortConv)
        {
            const norm_add_one = shift_norms or c.rms_norm_add_one;
            const fn_t = getPrefixedOpt(&ckpt, prefix, "norm.weight") orelse try getPrefixed(&ckpt, prefix, "embedding_norm.weight");
            var small_bytes: usize = align64(fn_t.elemCount() * 2);
            for (0..c.num_layers) |li| {
                small_bytes += align64(c.hidden_size * 2) * (if (c.has_post_norms) @as(usize, 4) else 2);
                if (c.is_gemma4) small_bytes += 64; // optional layer_scalar
                if (c.layer_kinds[li] == .linear) {
                    small_bytes += align64(c.gdnQkvDim() * c.conv_kernel * 2); // conv_weight bf16
                    small_bytes += align64(c.gdn_v_heads * 4); // a_log f32
                    small_bytes += align64(c.gdn_v_heads * 2); // dt_bias bf16
                    small_bytes += align64(c.gdn_v_dim * 2); // gdn_norm bf16
                } else if (c.layer_kinds[li] == .short_conv) {
                    small_bytes += align64(c.hidden_size * c.conv_l_cache * 2); // conv.conv.weight bf16
                    if (c.conv_bias) {
                        small_bytes += align64(3 * c.hidden_size * 2); // conv.in_proj.bias
                        small_bytes += align64(c.hidden_size * 2) * 2; // conv.conv.bias + conv.out_proj.bias
                    }
                } else if (c.use_mla) {
                    if (c.q_lora_rank > 0) {
                        small_bytes += align64(c.q_lora_rank * 2); // q_a_layernorm
                        if (c.has_qkv_bias) small_bytes += align64(c.q_lora_rank * 2);
                    }
                    small_bytes += align64(c.kv_lora_rank * 2); // kv_a_layernorm
                    if (c.has_qkv_bias) small_bytes += align64(c.mlaKvAOut() * 2);
                } else {
                    const l_hd = c.layerHeadDim(li);
                    if (c.has_qk_norm) small_bytes += align64(l_hd * 2) * 2;
                    if (c.has_qkv_bias) small_bytes += align64(c.layerQkvStride(li) * 2);
                }
            }
            const small_buf = try allocTracked(&w, ctx, small_bytes);
            var off: usize = 0;
            w.final_norm = copyNormBf16(small_buf, &off, fn_t, norm_add_one);

            var name_buf: [160]u8 = undefined;
            for (0..c.num_layers) |li| {
                w.layers[li].kind = c.layer_kinds[li];
                const in_n = getLayerOpt(&ckpt, &name_buf, prefix, li, "input_layernorm.weight") orelse
                    try getLayer(&ckpt, &name_buf, prefix, li, "operator_norm.weight");
                w.layers[li].input_norm = copyNormBf16(small_buf, &off, in_n, norm_add_one);
                if (c.glm4_post_norms) {
                    // GLM-4 4-norm naming: post_self_attn_layernorm (after attn), post_attention_layernorm (before MLP), post_mlp_layernorm (after MLP)
                    const psa = try getLayer(&ckpt, &name_buf, prefix, li, "post_self_attn_layernorm.weight");
                    const pre_ff = try getLayer(&ckpt, &name_buf, prefix, li, "post_attention_layernorm.weight");
                    const post_ff = try getLayer(&ckpt, &name_buf, prefix, li, "post_mlp_layernorm.weight");
                    w.layers[li].post_attn_norm = copyNormBf16(small_buf, &off, psa, false);
                    w.layers[li].pre_ff_norm = copyNormBf16(small_buf, &off, pre_ff, false);
                    w.layers[li].post_ff_norm = copyNormBf16(small_buf, &off, post_ff, false);
                } else if (c.has_post_norms) {
                    const pa_n = try getLayer(&ckpt, &name_buf, prefix, li, "post_attention_layernorm.weight");
                    const pre_ff = try getLayer(&ckpt, &name_buf, prefix, li, "pre_feedforward_layernorm.weight");
                    const post_ff = try getLayer(&ckpt, &name_buf, prefix, li, "post_feedforward_layernorm.weight");
                    w.layers[li].post_attn_norm = copyNormBf16(small_buf, &off, pa_n, norm_add_one);
                    w.layers[li].pre_ff_norm = copyNormBf16(small_buf, &off, pre_ff, norm_add_one);
                    w.layers[li].post_ff_norm = copyNormBf16(small_buf, &off, post_ff, norm_add_one);
                } else {
                    const pa_n = getLayerOpt(&ckpt, &name_buf, prefix, li, "post_attention_layernorm.weight") orelse
                        try getLayer(&ckpt, &name_buf, prefix, li, "ffn_norm.weight");
                    w.layers[li].post_attn_norm = copyNormBf16(small_buf, &off, pa_n, norm_add_one);
                    w.layers[li].pre_ff_norm = small_buf.view(0);
                    w.layers[li].post_ff_norm = small_buf.view(0);
                }
                if (getLayerOpt(&ckpt, &name_buf, prefix, li, "layer_scalar")) |ls_t| {
                    w.layers[li].layer_scalar = copyNormBf16(small_buf, &off, ls_t, false);
                    w.layers[li].has_layer_scalar = true;
                } else {
                    w.layers[li].layer_scalar = small_buf.view(0);
                    w.layers[li].has_layer_scalar = false;
                }

                if (c.layer_kinds[li] == .linear) {
                    const cw = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.conv1d.weight");
                    const al = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.A_log");
                    const dtb = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.dt_bias");
                    const gn = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.norm.weight");
                    w.layers[li].linear.conv_weight = copyInto(small_buf, &off, cw.bytes);
                    w.layers[li].linear.a_log = copyAsF32(small_buf, &off, al);
                    w.layers[li].linear.dt_bias = copyInto(small_buf, &off, dtb.bytes);
                    w.layers[li].linear.gdn_norm = copyInto(small_buf, &off, gn.bytes);
                } else if (c.layer_kinds[li] == .short_conv) {
                    const cw = try getLayer(&ckpt, &name_buf, prefix, li, "conv.conv.weight");
                    w.layers[li].short_conv = .{};
                    // Handle [H, 1, L] / [H, L, 1] / [H, L] directly, or transpose [L, H] -> [H, L]
                    if (cw.shape[0] == c.conv_l_cache and cw.shape[cw.rank - 1] == c.hidden_size and c.conv_l_cache != c.hidden_size) {
                        const start = off;
                        const dst: [*]u16 = @ptrCast(@alignCast(small_buf.ptr + start));
                        for (0..c.hidden_size) |ch| {
                            for (0..c.conv_l_cache) |k| {
                                dst[ch * c.conv_l_cache + k] = f32ToBf16(readScalarAsF32(cw, k * c.hidden_size + ch));
                            }
                        }
                        off += align64(c.hidden_size * c.conv_l_cache * 2);
                        w.layers[li].short_conv.conv_weight = small_buf.view(start);
                    } else {
                        w.layers[li].short_conv.conv_weight = copyNormBf16(small_buf, &off, cw, false);
                    }
                    if (getLayerOpt(&ckpt, &name_buf, prefix, li, "conv.in_proj.bias")) |ib| {
                        w.layers[li].short_conv.in_bias = copyNormBf16(small_buf, &off, ib, false);
                    }
                    if (getLayerOpt(&ckpt, &name_buf, prefix, li, "conv.conv.bias")) |cb| {
                        w.layers[li].short_conv.conv_bias = copyNormBf16(small_buf, &off, cb, false);
                    }
                    if (getLayerOpt(&ckpt, &name_buf, prefix, li, "conv.out_proj.bias")) |ob| {
                        w.layers[li].short_conv.out_bias = copyNormBf16(small_buf, &off, ob, false);
                    }
                } else if (c.use_mla) {
                    w.layers[li].mla = .{ .has_q_lora = c.q_lora_rank > 0 };
                    if (c.q_lora_rank > 0) {
                        const qan = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.q_a_layernorm.weight");
                        w.layers[li].mla.q_a_norm = copyNormBf16(small_buf, &off, qan, false);
                        if (getLayerOpt(&ckpt, &name_buf, prefix, li, "self_attn.q_a_proj.bias")) |qab| {
                            w.layers[li].mla.q_a_bias = copyNormBf16(small_buf, &off, qab, false);
                        }
                    } else {
                        w.layers[li].mla.q_a_norm = small_buf.view(0);
                    }
                    const kvan = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.kv_a_layernorm.weight");
                    w.layers[li].mla.kv_a_norm = copyNormBf16(small_buf, &off, kvan, false);
                    if (getLayerOpt(&ckpt, &name_buf, prefix, li, "self_attn.kv_a_proj_with_mqa.bias")) |kvab| {
                        w.layers[li].mla.kv_a_bias = copyNormBf16(small_buf, &off, kvab, false);
                    }
                } else {
                    if (c.has_qk_norm) {
                        const qn = getLayerOpt(&ckpt, &name_buf, prefix, li, "self_attn.q_norm.weight") orelse
                            try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.q_layernorm.weight");
                        const kn = getLayerOpt(&ckpt, &name_buf, prefix, li, "self_attn.k_norm.weight") orelse
                            try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.k_layernorm.weight");
                        w.layers[li].full.q_norm = copyNormBf16(small_buf, &off, qn, norm_add_one);
                        w.layers[li].full.k_norm = copyNormBf16(small_buf, &off, kn, norm_add_one);
                    } else {
                        w.layers[li].full.q_norm = small_buf.view(0);
                        w.layers[li].full.k_norm = small_buf.view(0);
                    }
                    if (c.has_qkv_bias) {
                        const qb = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.q_proj.bias");
                        const kb = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.k_proj.bias");
                        const vb = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.v_proj.bias");
                        const bstart = off;
                        @memcpy(small_buf.ptr[off..][0..qb.bytes.len], qb.bytes);
                        off += qb.bytes.len;
                        @memcpy(small_buf.ptr[off..][0..kb.bytes.len], kb.bytes);
                        off += kb.bytes.len;
                        @memcpy(small_buf.ptr[off..][0..vb.bytes.len], vb.bytes);
                        off = bstart + align64(qb.bytes.len + kb.bytes.len + vb.bytes.len);
                        w.layers[li].full.qkv_bias = small_buf.view(bstart);
                    } else {
                        w.layers[li].full.qkv_bias = null;
                    }
                }
            }
        }

        // 3. Per-layer linear weights (supports 4-bit qk4 fast path, 2/3/4/5/6/8-bit affine g32/64/128, BF16, & mixed precision!)
        const gdn_all_n = c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads;

        var name_buf: [160]u8 = undefined;
        for (0..c.num_layers) |li| {
            const is_moe_layer = c.layer_is_moe[li];
            w.layers[li].is_moe = is_moe_layer;

            var gu_srcs_buf: [2]RawQuant = undefined;
            var gu_srcs_len: usize = 0;
            var d_srcs: [1]RawQuant = undefined;
            var gu_bytes: usize = 0;
            var down_bytes: usize = 0;
            if (!is_moe_layer) {
                const has_fused_gu = (getLayerOpt(&ckpt, &name_buf, prefix, li, "mlp.gate_up_proj.weight") != null) or
                    (if (ts_ptr) |ts| blk: {
                        var tb: [160]u8 = undefined;
                        const nm = std.fmt.bufPrint(&tb, "layers.{d}.mlp.gate_up_proj", .{li}) catch "";
                        break :blk ts.find(nm, 2 * c.intermediate_size, c.hidden_size) != null;
                    } else false);
                const has_ff_w1 = (getLayerOpt(&ckpt, &name_buf, prefix, li, "feed_forward.w1.weight") != null);
                const has_ff_gate = (getLayerOpt(&ckpt, &name_buf, prefix, li, "feed_forward.gate_proj.weight") != null);
                if (has_fused_gu) {
                    gu_srcs_buf[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "mlp.gate_up_proj", 2 * c.intermediate_size, c.hidden_size);
                    gu_srcs_len = 1;
                    d_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "mlp.down_proj", c.hidden_size, c.intermediate_size);
                } else if (has_ff_w1) {
                    gu_srcs_buf[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "feed_forward.w1", c.intermediate_size, c.hidden_size);
                    gu_srcs_buf[1] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "feed_forward.w3", c.intermediate_size, c.hidden_size);
                    gu_srcs_len = 2;
                    d_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "feed_forward.w2", c.hidden_size, c.intermediate_size);
                } else if (has_ff_gate) {
                    gu_srcs_buf[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "feed_forward.gate_proj", c.intermediate_size, c.hidden_size);
                    gu_srcs_buf[1] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "feed_forward.up_proj", c.intermediate_size, c.hidden_size);
                    gu_srcs_len = 2;
                    d_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "feed_forward.down_proj", c.hidden_size, c.intermediate_size);
                } else {
                    gu_srcs_buf[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "mlp.gate_proj", c.intermediate_size, c.hidden_size);
                    gu_srcs_buf[1] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "mlp.up_proj", c.intermediate_size, c.hidden_size);
                    gu_srcs_len = 2;
                    d_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "mlp.down_proj", c.hidden_size, c.intermediate_size);
                }
                gu_bytes = measureLinear(gu_srcs_buf[0..gu_srcs_len], 2 * c.intermediate_size, c.hidden_size, keep_bf16);
                down_bytes = measureLinear(&d_srcs, c.hidden_size, c.intermediate_size, keep_bf16);
            }
            const gu_srcs = gu_srcs_buf[0..gu_srcs_len];

            if (c.layer_kinds[li] == .linear) {
                const qkv_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "linear_attn.in_proj_qkv", c.gdnQkvDim(), c.hidden_size);
                const z_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "linear_attn.in_proj_z", c.gdnValDim(), c.hidden_size);
                const b_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "linear_attn.in_proj_b", c.gdn_v_heads, c.hidden_size);
                const a_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "linear_attn.in_proj_a", c.gdn_v_heads, c.hidden_size);
                const in_srcs = [4]RawQuant{ qkv_q, z_q, b_q, a_q };
                const out_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "linear_attn.out_proj", c.hidden_size, c.gdnValDim());
                const out_srcs = [1]RawQuant{out_q};

                const in_bytes = measureLinear(&in_srcs, gdn_all_n, c.hidden_size, keep_bf16);
                const out_bytes = measureLinear(&out_srcs, c.hidden_size, c.gdnValDim(), keep_bf16);
                const total = in_bytes + out_bytes + gu_bytes + down_bytes;
                const lbuf: ?Buffer = if (total > 0) try allocTracked(&w, ctx, total) else null;
                var off: usize = 0;

                w.layers[li].linear.in_proj_all = packLinearInto(lbuf, &off, &in_srcs, gdn_all_n, c.hidden_size, keep_bf16);
                w.layers[li].linear.out_proj = packLinearInto(lbuf, &off, &out_srcs, c.hidden_size, c.gdnValDim(), keep_bf16);
                if (!is_moe_layer) {
                    w.layers[li].gate_up = packLinearInto(lbuf, &off, gu_srcs, 2 * c.intermediate_size, c.hidden_size, keep_bf16);
                    w.layers[li].down_proj = packLinearInto(lbuf, &off, &d_srcs, c.hidden_size, c.intermediate_size, keep_bf16);
                }
            } else if (c.layer_kinds[li] == .short_conv) {
                const in_srcs = [1]RawQuant{try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "conv.in_proj", 3 * c.hidden_size, c.hidden_size)};
                const out_srcs = [1]RawQuant{try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "conv.out_proj", c.hidden_size, c.hidden_size)};
                const in_bytes = measureLinear(&in_srcs, 3 * c.hidden_size, c.hidden_size, keep_bf16);
                const out_bytes = measureLinear(&out_srcs, c.hidden_size, c.hidden_size, keep_bf16);
                const total = in_bytes + out_bytes + gu_bytes + down_bytes;
                const lbuf: ?Buffer = if (total > 0) try allocTracked(&w, ctx, total) else null;
                var off: usize = 0;

                w.layers[li].short_conv.in_proj = packLinearInto(lbuf, &off, &in_srcs, 3 * c.hidden_size, c.hidden_size, keep_bf16);
                w.layers[li].short_conv.out_proj = packLinearInto(lbuf, &off, &out_srcs, c.hidden_size, c.hidden_size, keep_bf16);
                if (!is_moe_layer) {
                    w.layers[li].gate_up = packLinearInto(lbuf, &off, gu_srcs, 2 * c.intermediate_size, c.hidden_size, keep_bf16);
                    w.layers[li].down_proj = packLinearInto(lbuf, &off, &d_srcs, c.hidden_size, c.intermediate_size, keep_bf16);
                }
            } else if (c.use_mla) {
                const q_out = c.num_heads * c.mlaQHeadDim();
                const kv_a_out = c.mlaKvAOut();
                const kv_b_out = c.mlaKvBOut();
                const o_k = c.num_heads * c.v_head_dim;

                var q_srcs: [1]RawQuant = undefined;
                var qa_srcs: [1]RawQuant = undefined;
                var qb_srcs: [1]RawQuant = undefined;
                var q_bytes: usize = 0;
                var qb_bytes: usize = 0;
                if (c.q_lora_rank > 0) {
                    qa_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.q_a_proj", c.q_lora_rank, c.hidden_size);
                    qb_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.q_b_proj", q_out, c.q_lora_rank);
                    q_bytes = measureLinear(&qa_srcs, c.q_lora_rank, c.hidden_size, keep_bf16);
                    qb_bytes = measureLinear(&qb_srcs, q_out, c.q_lora_rank, keep_bf16);
                } else {
                    q_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.q_proj", q_out, c.hidden_size);
                    q_bytes = measureLinear(&q_srcs, q_out, c.hidden_size, keep_bf16);
                }
                const kva_srcs = [1]RawQuant{try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.kv_a_proj_with_mqa", kv_a_out, c.hidden_size)};
                const kva_bytes = measureLinear(&kva_srcs, kv_a_out, c.hidden_size, keep_bf16);
                const o_srcs = [1]RawQuant{try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.o_proj", c.hidden_size, o_k)};
                const o_bytes = measureLinear(&o_srcs, c.hidden_size, o_k, keep_bf16);

                const has_kv_b = (getLayerOpt(&ckpt, &name_buf, prefix, li, "self_attn.kv_b_proj.weight") != null) or
                    (if (ts_ptr) |ts| blk: {
                        var tb: [160]u8 = undefined;
                        const nm = std.fmt.bufPrint(&tb, "layers.{d}.self_attn.kv_b_proj", .{li}) catch "";
                        break :blk ts.find(nm, kv_b_out, c.kv_lora_rank) != null;
                    } else false);
                var kvb_srcs: [1]RawQuant = undefined;
                var kvb_bytes: usize = 0;
                if (has_kv_b) {
                    kvb_srcs[0] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.kv_b_proj", kv_b_out, c.kv_lora_rank);
                    kvb_bytes = measureLinear(&kvb_srcs, kv_b_out, c.kv_lora_rank, keep_bf16);
                } else {
                    // MLX-Community split format: embed_q + unembed_out -> assemble BF16 kv_b_proj
                    kvb_bytes = align64(kv_b_out * c.kv_lora_rank * 2);
                }

                const total = q_bytes + qb_bytes + kva_bytes + kvb_bytes + o_bytes + gu_bytes + down_bytes;
                const lbuf: ?Buffer = if (total > 0) try allocTracked(&w, ctx, total) else null;
                var off: usize = 0;

                if (c.q_lora_rank > 0) {
                    w.layers[li].mla.q_a_proj = packLinearInto(lbuf, &off, &qa_srcs, c.q_lora_rank, c.hidden_size, keep_bf16);
                    w.layers[li].mla.q_b_proj = packLinearInto(lbuf, &off, &qb_srcs, q_out, c.q_lora_rank, keep_bf16);
                } else {
                    w.layers[li].mla.q_proj = packLinearInto(lbuf, &off, &q_srcs, q_out, c.hidden_size, keep_bf16);
                }
                w.layers[li].mla.kv_a_proj = packLinearInto(lbuf, &off, &kva_srcs, kv_a_out, c.hidden_size, keep_bf16);
                if (has_kv_b) {
                    w.layers[li].mla.kv_b_proj = packLinearInto(lbuf, &off, &kvb_srcs, kv_b_out, c.kv_lora_rank, keep_bf16);
                } else {
                    const start = off;
                    const dst_bf16: [*]u16 = @ptrCast(@alignCast(lbuf.?.ptr + start));
                    try assembleMlaKvBFromMlx(&ckpt, prefix, li, c, dst_bf16);
                    off += align64(kv_b_out * c.kv_lora_rank * 2);
                    w.layers[li].mla.kv_b_proj = .{ .bf16 = .{ .ref = lbuf.?.view(start), .n = @intCast(kv_b_out), .k = @intCast(c.kv_lora_rank) } };
                }
                w.layers[li].mla.o_proj = packLinearInto(lbuf, &off, &o_srcs, c.hidden_size, o_k, keep_bf16);
                if (!is_moe_layer) {
                    w.layers[li].gate_up = packLinearInto(lbuf, &off, gu_srcs, 2 * c.intermediate_size, c.hidden_size, keep_bf16);
                    w.layers[li].down_proj = packLinearInto(lbuf, &off, &d_srcs, c.hidden_size, c.intermediate_size, keep_bf16);
                }
            } else {
                const l_q_out = c.layerQProjOut(li);
                const l_kv_out = c.layerKvProjOut(li);
                const l_o_k = c.num_heads * c.layerHeadDim(li);
                const l_k_eq_v = c.layer_k_eq_v[li];
                const l_qkv_n = c.layerQkvStride(li);

                const q_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.q_proj", l_q_out, c.hidden_size);
                const k_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.k_proj", l_kv_out, c.hidden_size);
                var qkv_buf: [3]RawQuant = undefined;
                var qkv_len: usize = 2;
                qkv_buf[0] = q_q;
                qkv_buf[1] = k_q;
                if (!l_k_eq_v) {
                    qkv_buf[2] = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, "self_attn.v_proj", l_kv_out, c.hidden_size);
                    qkv_len = 3;
                }
                const qkv_srcs = qkv_buf[0..qkv_len];
                const o_sub: []const u8 = if (getLayerOpt(&ckpt, &name_buf, prefix, li, "self_attn.out_proj.weight") != null)
                    "self_attn.out_proj"
                else
                    "self_attn.o_proj";
                const o_q = try getQuantLayer(&ckpt, ts_ptr, &name_buf, prefix, li, o_sub, c.hidden_size, l_o_k);
                const o_srcs = [1]RawQuant{o_q};

                const qkv_bytes = measureLinear(qkv_srcs, l_qkv_n, c.hidden_size, keep_bf16);
                const o_bytes = measureLinear(&o_srcs, c.hidden_size, l_o_k, keep_bf16);
                const total = qkv_bytes + o_bytes + gu_bytes + down_bytes;
                const lbuf: ?Buffer = if (total > 0) try allocTracked(&w, ctx, total) else null;
                var off: usize = 0;

                w.layers[li].full.qkv_proj = packLinearInto(lbuf, &off, qkv_srcs, l_qkv_n, c.hidden_size, keep_bf16);
                w.layers[li].full.o_proj = packLinearInto(lbuf, &off, &o_srcs, c.hidden_size, l_o_k, keep_bf16);
                if (!is_moe_layer) {
                    w.layers[li].gate_up = packLinearInto(lbuf, &off, gu_srcs, 2 * c.intermediate_size, c.hidden_size, keep_bf16);
                    w.layers[li].down_proj = packLinearInto(lbuf, &off, &d_srcs, c.hidden_size, c.intermediate_size, keep_bf16);
                }
            }

            if (is_moe_layer) {
                w.layers[li].swiglu_moe = try loadSwigluMoeLayer(&w, ctx, c, &ckpt, ts_ptr, prefix, li, keep_bf16);
            }
        }

        if (w.isPureQk4(c)) pcache.save(a, c, model_dir, &w) catch {};
        return w;
    }
};

const RawQuant = struct {
    w: ?st.Tensor = null,
    s: ?st.Tensor = null,
    b: ?st.Tensor = null,
    tfq3: ?tfq3_mod.Found = null,
    n: usize,
};

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7FFF +% lsb) >> 16);
}

inline fn quantGroup64Bf16(src: [*]align(1) const u16, dst_codes: [*]u8) struct { s: u16, b: u16 } {
    var vmin: f32 = std.math.inf(f32);
    var vmax: f32 = -std.math.inf(f32);
    for (0..qk_gs) |i| {
        const v = bf16ToF32(src[i]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    const span = vmax - vmin;
    const scale: f32 = if (span > 1e-12) span / 15.0 else 0.0;
    const s_bf = f32ToBf16(scale);
    const b_bf = f32ToBf16(vmin);
    const s_f = bf16ToF32(s_bf);
    const b_f = bf16ToF32(b_bf);
    const inv_s: f32 = if (s_f > 0.0) 1.0 / s_f else 0.0;

    for (0..32) |bi| {
        const v0 = bf16ToF32(src[bi * 2]);
        const v1 = bf16ToF32(src[bi * 2 + 1]);
        const q0: u8 = @intFromFloat(@round(std.math.clamp((v0 - b_f) * inv_s, 0.0, 15.0)));
        const q1: u8 = @intFromFloat(@round(std.math.clamp((v1 - b_f) * inv_s, 0.0, 15.0)));
        dst_codes[bi] = q0 | (q1 << 4);
    }
    return .{ .s = s_bf, .b = b_bf };
}

const EmbedQuantJob = struct {
    src: [*]align(1) const u16,
    dst_codes: [*]u8,
    dst_scales: [*]u16,
    dst_biases: [*]u16,
    hidden: usize,
    groups: usize,
};

fn embedQuantWorker(raw_ctx: ?*anyopaque, row_idx: usize) callconv(.c) void {
    const job: *const EmbedQuantJob = @ptrCast(@alignCast(raw_ctx.?));
    const src_row = job.src + row_idx * job.hidden;
    const dst_c = job.dst_codes + row_idx * (job.hidden / 2);
    const dst_s = job.dst_scales + row_idx * job.groups;
    const dst_b = job.dst_biases + row_idx * job.groups;
    for (0..job.groups) |g| {
        const sb = quantGroup64Bf16(src_row + g * qk_gs, dst_c + g * 32);
        dst_s[g] = sb.s;
        dst_b[g] = sb.b;
    }
}

fn quantizeEmbedBf16(ew: st.Tensor, vocab: usize, hidden: usize, dst_codes: [*]u8, dst_scales: [*]u16, dst_biases: [*]u16) void {
    var job = EmbedQuantJob{
        .src = ew.asSlice(u16).ptr,
        .dst_codes = dst_codes,
        .dst_scales = dst_scales,
        .dst_biases = dst_biases,
        .hidden = hidden,
        .groups = hidden / qk_gs,
    };
    const q = dispatch_get_global_queue(0, 0);
    dispatch_apply_f(vocab, q, &job, embedQuantWorker);
}

fn packInto(buf: Buffer, off: *usize, srcs: []const RawQuant, n: usize, k: usize) QkMatrix {
    const start = off.*;
    const size = QkMatrix.byteSize(n, k);
    packQkConcat(buf.ptr + start, srcs, n, k);
    off.* += align64(size);
    return .{ .ref = buf.view(start), .n = @intCast(n), .k = @intCast(k) };
}

const AffineSpec = struct {
    bits: u8,
    gs: u16,
    sb_dtype: SbDType,

    fn eql(a: AffineSpec, b: AffineSpec) bool {
        return a.bits == b.bits and a.gs == b.gs and a.sb_dtype == b.sb_dtype;
    }
};

const TensorSpec = union(enum) {
    qk4: void,
    affine: AffineSpec,
    bf16: void,
    tfq3: void,
};

fn detectTensorSpec(rq: RawQuant, k: usize, keep_bf16: bool) TensorSpec {
    if (rq.tfq3 != null) return .tfq3;
    if (rq.s != null and rq.b != null) {
        const w_t = rq.w.?;
        const s = rq.s.?;
        const groups = s.shape[s.rank - 1];
        const gs: u16 = @intCast(k / groups);
        const w_words = w_t.bytes.len / (rq.n * 4);
        const bits: u8 = @intCast((w_words * 32) / k);
        const sbd = SbDType.fromSt(s.dtype);
        if (bits == 4 and gs == 64 and sbd == .bf16 and k % qk_gs == 0) {
            return .qk4;
        }
        return .{ .affine = .{ .bits = bits, .gs = gs, .sb_dtype = sbd } };
    }
    if (keep_bf16 or k % qk_gs != 0) return .bf16;
    return .qk4;
}

fn affineSpecOf(rq: RawQuant, k: usize) ?AffineSpec {
    if (rq.tfq3 != null or rq.s == null or rq.b == null) return null;
    const w_t = rq.w.?;
    const s = rq.s.?;
    const groups = s.shape[s.rank - 1];
    const gs: u16 = @intCast(k / groups);
    const w_words = w_t.bytes.len / (rq.n * 4);
    const bits: u8 = @intCast((w_words * 32) / k);
    return .{ .bits = bits, .gs = gs, .sb_dtype = SbDType.fromSt(s.dtype) };
}

fn affineByteSize(n: usize, k: usize, spec: AffineSpec) usize {
    const codes_bytes = n * ((k * @as(usize, spec.bits)) / 32) * 4;
    const sb_bytes = n * (k / @as(usize, spec.gs)) * spec.sb_dtype.elemBytes();
    return align64(codes_bytes) + align64(sb_bytes) * 2;
}

fn copyRawAsBf16(dst_ptr: [*]u8, t: st.Tensor) void {
    if (t.dtype == .bf16) {
        @memcpy(dst_ptr[0..t.bytes.len], t.bytes);
        return;
    }
    const n = t.elemCount();
    const dst: [*]u16 = @ptrCast(@alignCast(dst_ptr));
    if (t.dtype == .f16) {
        const src = t.asSlice(u16);
        for (0..n) |i| {
            const h: f16 = @bitCast(src[i]);
            dst[i] = f32ToBf16(@floatCast(h));
        }
    } else if (t.dtype == .f32) {
        const src = t.asSlice(f32);
        for (0..n) |i| dst[i] = f32ToBf16(src[i]);
    } else {
        @panic("Unsupported raw tensor dtype for BF16 linear");
    }
}

fn measureLinear(srcs: []const RawQuant, total_n: usize, k: usize, keep_bf16: bool) usize {
    var all_qk4 = (total_n % qk_nt == 0 and k % qk_gs == 0);
    var all_bf16 = true;
    var all_affine = true;
    var common_aff: ?AffineSpec = null;

    for (srcs) |rq| {
        const sp = detectTensorSpec(rq, k, keep_bf16);
        if (sp != .qk4) all_qk4 = false;
        if (sp != .bf16) all_bf16 = false;
        if (affineSpecOf(rq, k)) |as| {
            if (common_aff) |ca| {
                if (!ca.eql(as)) all_affine = false;
            } else {
                common_aff = as;
            }
        } else {
            all_affine = false;
        }
    }

    if (all_qk4) return align64(QkMatrix.byteSize(total_n, k));
    if (all_affine and common_aff != null) return affineByteSize(total_n, k, common_aff.?);
    if (all_bf16) return align64(total_n * k * 2);

    var total: usize = 0;
    for (srcs) |rq| {
        if (rq.tfq3 != null) continue;
        if (affineSpecOf(rq, k)) |as| {
            total += affineByteSize(rq.n, k, as);
        } else {
            total += align64(rq.n * k * 2);
        }
    }
    return total;
}

fn packAffineConcat(buf: Buffer, off: *usize, srcs: []const RawQuant, total_n: usize, k: usize, spec: AffineSpec) AffineMatrix {
    const codes_bytes = total_n * ((k * @as(usize, spec.bits)) / 32) * 4;
    const sb_bytes = total_n * (k / @as(usize, spec.gs)) * spec.sb_dtype.elemBytes();
    const c_start = off.*;
    var c_cur = c_start;
    for (srcs) |rq| {
        const w_t = rq.w.?;
        @memcpy(buf.ptr[c_cur..][0..w_t.bytes.len], w_t.bytes);
        c_cur += w_t.bytes.len;
    }
    off.* = c_start + align64(codes_bytes);

    const s_start = off.*;
    var s_cur = s_start;
    for (srcs) |rq| {
        const s = rq.s.?;
        @memcpy(buf.ptr[s_cur..][0..s.bytes.len], s.bytes);
        s_cur += s.bytes.len;
    }
    off.* = s_start + align64(sb_bytes);

    const b_start = off.*;
    var b_cur = b_start;
    for (srcs) |rq| {
        const b = rq.b.?;
        @memcpy(buf.ptr[b_cur..][0..b.bytes.len], b.bytes);
        b_cur += b.bytes.len;
    }
    off.* = b_start + align64(sb_bytes);

    return .{
        .codes = buf.view(c_start),
        .scales = buf.view(s_start),
        .biases = buf.view(b_start),
        .n = @intCast(total_n),
        .k = @intCast(k),
        .bits = spec.bits,
        .gs = spec.gs,
        .sb_dtype = spec.sb_dtype,
    };
}

fn packBf16Concat(buf: Buffer, off: *usize, srcs: []const RawQuant, total_n: usize, k: usize) Bf16Matrix {
    const start = off.*;
    var cur = start;
    for (srcs) |rq| {
        copyRawAsBf16(buf.ptr + cur, rq.w.?);
        cur += rq.n * k * 2;
    }
    off.* = start + align64(total_n * k * 2);
    return .{ .ref = buf.view(start), .n = @intCast(total_n), .k = @intCast(k) };
}

fn packLinearInto(buf_opt: ?Buffer, off: *usize, srcs: []const RawQuant, total_n: usize, k: usize, keep_bf16: bool) LinearWeight {
    // Fast zero-copy TFQ3 single or pair-fused dispatch!
    if (srcs.len == 1 and srcs[0].tfq3 != null) {
        return .{ .tfq3 = srcs[0].tfq3.?.tfq3 };
    }
    if (srcs.len == 2 and srcs[0].tfq3 != null and srcs[1].tfq3 != null and tfq3_mod.canFusePair(srcs[0].tfq3.?, srcs[1].tfq3.?)) {
        return .{ .tfq3 = tfq3_mod.fusePair(srcs[0].tfq3.?, srcs[1].tfq3.?) };
    }

    var all_qk4 = (total_n % qk_nt == 0 and k % qk_gs == 0);
    var all_bf16 = true;
    var all_affine = true;
    var common_aff: ?AffineSpec = null;

    for (srcs) |rq| {
        const sp = detectTensorSpec(rq, k, keep_bf16);
        if (sp != .qk4) all_qk4 = false;
        if (sp != .bf16) all_bf16 = false;
        if (affineSpecOf(rq, k)) |as| {
            if (common_aff) |ca| {
                if (!ca.eql(as)) all_affine = false;
            } else {
                common_aff = as;
            }
        } else {
            all_affine = false;
        }
    }

    if (all_qk4) return .{ .qk4 = packInto(buf_opt.?, off, srcs, total_n, k) };
    if (all_affine and common_aff != null) return .{ .affine = packAffineConcat(buf_opt.?, off, srcs, total_n, k, common_aff.?) };
    if (all_bf16) return .{ .bf16 = packBf16Concat(buf_opt.?, off, srcs, total_n, k) };

    var cat = ConcatLinear{ .count = 0, .n = @intCast(total_n), .k = @intCast(k) };
    var col_off: u32 = 0;
    var i: usize = 0;
    while (i < srcs.len) {
        // Automatic adjacent TFQ3 pair fusion (e.g. k_proj + v_proj inside qkv_proj!)
        if (i + 1 < srcs.len and srcs[i].tfq3 != null and srcs[i + 1].tfq3 != null and tfq3_mod.canFusePair(srcs[i].tfq3.?, srcs[i + 1].tfq3.?)) {
            const fused = tfq3_mod.fusePair(srcs[i].tfq3.?, srcs[i + 1].tfq3.?);
            cat.col_offs[cat.count] = col_off;
            cat.parts[cat.count] = .{ .tfq3 = fused };
            cat.count += 1;
            col_off += fused.n;
            i += 2;
            continue;
        }
        const rq = srcs[i];
        cat.col_offs[cat.count] = col_off;
        if (rq.tfq3) |ft| {
            cat.parts[cat.count] = .{ .tfq3 = ft.tfq3 };
        } else if (affineSpecOf(rq, k)) |as| {
            cat.parts[cat.count] = .{ .affine = packAffineConcat(buf_opt.?, off, &.{rq}, rq.n, k, as) };
        } else {
            cat.parts[cat.count] = .{ .bf16 = packBf16Concat(buf_opt.?, off, &.{rq}, rq.n, k) };
        }
        cat.count += 1;
        col_off += @intCast(rq.n);
        i += 1;
    }
    return .{ .concat = cat };
}

const PackJob = struct {
    dst: [*]u8,
    srcs: []const RawQuant,
    groups: usize,
    row_code_bytes: usize,
};

fn packTileWorker(raw_ctx: ?*anyopaque, tile_idx: usize) callconv(.c) void {
    const job: *const PackJob = @ptrCast(@alignCast(raw_ctx.?));
    const g_count = job.groups;
    const k_cols = g_count * qk_gs;
    const tile_dst = job.dst + tile_idx * g_count * qk_block_bytes;
    const base_row = tile_idx * qk_nt;

    for (0..qk_nt) |c| {
        const global_row = base_row + c;
        // Locate which source matrix owns global_row
        var local_row = global_row;
        var src_idx: usize = 0;
        while (src_idx < job.srcs.len and local_row >= job.srcs[src_idx].n) : (src_idx += 1) {
            local_row -= job.srcs[src_idx].n;
        }
        const src = job.srcs[src_idx];
        if (src.s != null and src.b != null) {
            const w_row = src.w.?.bytes.ptr + local_row * job.row_code_bytes;
            const s_row = src.s.?.asSlice(u16).ptr + local_row * g_count;
            const b_row = src.b.?.asSlice(u16).ptr + local_row * g_count;

            for (0..g_count) |g| {
                const blk = tile_dst + g * qk_block_bytes;
                @memcpy(blk[c * 32 ..][0..32], w_row[g * 32 ..][0..32]);
                const sb: u32 = @as(u32, s_row[g]) | (@as(u32, b_row[g]) << 16);
                std.mem.writeInt(u32, blk[qk_codes_per_block + c * 4 ..][0..4], sb, .little);
            }
        } else {
            // Raw BF16 weight -> on-the-fly 4-bit affine group-64 quantization directly into qk tile!
            const bf_row = src.w.?.asSlice(u16).ptr + local_row * k_cols;
            for (0..g_count) |g| {
                const blk = tile_dst + g * qk_block_bytes;
                const sb_pair = quantGroup64Bf16(bf_row + g * qk_gs, blk + c * 32);
                const sb: u32 = @as(u32, sb_pair.s) | (@as(u32, sb_pair.b) << 16);
                std.mem.writeInt(u32, blk[qk_codes_per_block + c * 4 ..][0..4], sb, .little);
            }
        }
    }
}

/// Pack vertically concatenated MLX 4-bit matrices into qk tile layout in parallel via GCD.
fn packQkConcat(dst: [*]u8, srcs: []const RawQuant, total_n: usize, k: usize) void {
    const tiles = total_n / qk_nt;
    const groups = k / qk_gs;
    var job = PackJob{
        .dst = dst,
        .srcs = srcs,
        .groups = groups,
        .row_code_bytes = k / 2,
    };
    const q = dispatch_get_global_queue(0, 0);
    dispatch_apply_f(tiles, q, &job, packTileWorker);
    // Zero trailing slack block
    @memset(dst[tiles * groups * qk_block_bytes ..][0..qk_block_bytes], 0);
}

fn allocTracked(w: *ModelWeights, ctx: *mtl.Context, size: usize) !Buffer {
    const b = try ctx.allocBuffer(size);
    try w.buffers.append(w.a, b);
    w.total_bytes += b.len;
    return b;
}

fn align64(n: usize) usize {
    return (n + 63) & ~@as(usize, 63);
}

fn copyInto(buf: Buffer, off: *usize, src: []const u8) BufferRef {
    const start = off.*;
    @memcpy(buf.ptr[start..][0..src.len], src);
    off.* += align64(src.len);
    return buf.view(start);
}

fn copyNormBf16(buf: Buffer, off: *usize, t: st.Tensor, add_one: bool) BufferRef {
    if (!add_one and t.dtype == .bf16) return copyInto(buf, off, t.bytes);
    const start = off.*;
    const n = t.elemCount();
    const dst = @as([*]u16, @ptrCast(@alignCast(buf.ptr + start)))[0..n];
    const delta: f32 = if (add_one) 1.0 else 0.0;
    if (t.dtype == .bf16) {
        const src = t.asSlice(u16);
        for (dst, src) |*d, s| d.* = f32ToBf16(bf16ToF32(s) + delta);
    } else if (t.dtype == .f16) {
        const src = t.asSlice(u16);
        for (dst, src) |*d, s| {
            const h: f16 = @bitCast(s);
            d.* = f32ToBf16(@as(f32, @floatCast(h)) + delta);
        }
    } else if (t.dtype == .f32) {
        const src = t.asSlice(f32);
        for (dst, src) |*d, s| d.* = f32ToBf16(s + delta);
    } else {
        @panic("Unsupported norm dtype");
    }
    off.* += align64(n * 2);
    return buf.view(start);
}

fn copyAsF32(buf: Buffer, off: *usize, t: st.Tensor) BufferRef {
    const start = off.*;
    const n = t.elemCount();
    const dst = @as([*]f32, @ptrCast(@alignCast(buf.ptr + start)))[0..n];
    if (t.dtype == .f32) {
        const src = t.asSlice(f32);
        for (dst, src) |*d, s| d.* = s;
    } else if (t.dtype == .bf16) {
        const src = t.asSlice(u16);
        for (dst, src) |*d, s| {
            const bits: u32 = @as(u32, s) << 16;
            d.* = @bitCast(bits);
        }
    } else {
        @panic("Unsupported A_log dtype");
    }
    off.* += align64(n * 4);
    return buf.view(start);
}

fn getPrefixed(ckpt: *const st.Checkpoint, prefix: []const u8, suffix: []const u8) !st.Tensor {
    var buf: [160]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }) catch return error.NameTooLong;
    return ckpt.get(key);
}

fn getPrefixedOpt(ckpt: *const st.Checkpoint, prefix: []const u8, suffix: []const u8) ?st.Tensor {
    var buf: [160]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }) catch return null;
    return ckpt.getOptional(key);
}

fn getLayer(ckpt: *const st.Checkpoint, buf: *[160]u8, prefix: []const u8, layer: usize, suffix: []const u8) !st.Tensor {
    const key = std.fmt.bufPrint(buf, "{s}layers.{d}.{s}", .{ prefix, layer, suffix }) catch return error.NameTooLong;
    return ckpt.get(key);
}

fn getQuantLayer(ckpt: *const st.Checkpoint, ts_opt: ?*const tfq3_mod.Store, buf: *[160]u8, prefix: []const u8, layer: usize, mod: []const u8, expected_n: usize, expected_k: usize) !RawQuant {
    if (ts_opt) |ts| {
        const mod_key = std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer, mod }) catch return error.NameTooLong;
        if (ts.find(mod_key, expected_n, expected_k)) |found| {
            return .{ .tfq3 = found, .n = expected_n };
        }
    }
    var b1: [160]u8 = undefined;
    var b2: [160]u8 = undefined;
    const w = try ckpt.get(std.fmt.bufPrint(buf, "{s}layers.{d}.{s}.weight", .{ prefix, layer, mod }) catch return error.NameTooLong);
    const s = ckpt.getOptional(std.fmt.bufPrint(&b1, "{s}layers.{d}.{s}.scales", .{ prefix, layer, mod }) catch return error.NameTooLong);
    const b = ckpt.getOptional(std.fmt.bufPrint(&b2, "{s}layers.{d}.{s}.biases", .{ prefix, layer, mod }) catch return error.NameTooLong);
    std.debug.assert(w.shape[0] == expected_n);
    return .{ .w = w, .s = s, .b = b, .n = expected_n };
}

const ExpertPackJob = struct {
    dst: [*]u8,
    w_bytes: [*]const u8,
    s_u16: ?[*]align(1) const u16,
    b_u16: ?[*]align(1) const u16,
    n: usize,
    tiles: usize,
    groups: usize,
    row_code_bytes: usize,
    expert_stride: usize,
};

fn packExpertTileWorker(raw_ctx: ?*anyopaque, task_idx: usize) callconv(.c) void {
    const job: *const ExpertPackJob = @ptrCast(@alignCast(raw_ctx.?));
    const e = task_idx / job.tiles;
    const tile_idx = task_idx % job.tiles;
    const g_count = job.groups;
    const k_cols = g_count * qk_gs;
    const expert_dst = job.dst + e * job.expert_stride;
    const tile_dst = expert_dst + tile_idx * g_count * qk_block_bytes;
    const base_row = e * job.n + tile_idx * qk_nt;

    for (0..qk_nt) |c| {
        const flat_row = base_row + c;
        if (job.s_u16 != null and job.b_u16 != null) {
            const w_row = job.w_bytes + flat_row * job.row_code_bytes;
            const s_row = job.s_u16.? + flat_row * g_count;
            const b_row = job.b_u16.? + flat_row * g_count;
            for (0..g_count) |g| {
                const blk = tile_dst + g * qk_block_bytes;
                @memcpy(blk[c * 32 ..][0..32], w_row[g * 32 ..][0..32]);
                const sb: u32 = @as(u32, s_row[g]) | (@as(u32, b_row[g]) << 16);
                std.mem.writeInt(u32, blk[qk_codes_per_block + c * 4 ..][0..4], sb, .little);
            }
        } else {
            const bf_ptr: [*]align(1) const u16 = @ptrCast(job.w_bytes);
            const bf_row = bf_ptr + flat_row * k_cols;
            for (0..g_count) |g| {
                const blk = tile_dst + g * qk_block_bytes;
                const sb_pair = quantGroup64Bf16(bf_row + g * qk_gs, blk + c * 32);
                const sb: u32 = @as(u32, sb_pair.s) | (@as(u32, sb_pair.b) << 16);
                std.mem.writeInt(u32, blk[qk_codes_per_block + c * 4 ..][0..4], sb, .little);
            }
        }
    }
    if (tile_idx + 1 == job.tiles) {
        @memset(expert_dst[job.tiles * g_count * qk_block_bytes ..][0..qk_block_bytes], 0);
    }
}

fn packQkExperts(dst: [*]u8, w_t: st.Tensor, s_opt: ?st.Tensor, b_opt: ?st.Tensor, n_experts: usize, n: usize, k: usize, expert_stride: usize) void {
    const tiles = n / qk_nt;
    const groups = k / qk_gs;
    var job = ExpertPackJob{
        .dst = dst,
        .w_bytes = w_t.bytes.ptr,
        .s_u16 = if (s_opt) |s| s.asSlice(u16).ptr else null,
        .b_u16 = if (b_opt) |b| b.asSlice(u16).ptr else null,
        .n = n,
        .tiles = tiles,
        .groups = groups,
        .row_code_bytes = k / 2,
        .expert_stride = expert_stride,
    };
    const q = dispatch_get_global_queue(0, 0);
    dispatch_apply_f(n_experts * tiles, q, &job, packExpertTileWorker);
}

fn loadNemotronH(w: *ModelWeights, ctx: *mtl.Context, c: ModelConfig, ckpt: *const st.Checkpoint, ts_ptr: ?*const tfq3_mod.Store, keep_bf16: bool) !void {
    const prefix = "backbone.";
    const tied_head = c.tie_word_embeddings or (ckpt.getOptional("lm_head.weight") == null);

    // 1. Embeddings + LM Head
    {
        const ew = try ckpt.get("backbone.embeddings.weight");
        const es_opt = ckpt.getOptional("backbone.embeddings.scales");
        const eb_opt = ckpt.getOptional("backbone.embeddings.biases");

        if (es_opt != null and eb_opt != null) {
            const es = es_opt.?;
            const eb = eb_opt.?;
            const e_groups = es.shape[es.rank - 1];
            const e_gs: u16 = @intCast(c.hidden_size / e_groups);
            const e_words = ew.bytes.len / (c.vocab_size * 4);
            const e_bits: u8 = @intCast((e_words * 32) / c.hidden_size);
            const e_sbd = SbDType.fromSt(es.dtype);
            const total = align64(ew.bytes.len) + align64(es.bytes.len) + align64(eb.bytes.len);
            const buf = try allocTracked(w, ctx, total);
            var off: usize = 0;
            w.embed = .{
                .codes = copyInto(buf, &off, ew.bytes),
                .scales = copyInto(buf, &off, es.bytes),
                .biases = copyInto(buf, &off, eb.bytes),
                .bits = e_bits,
                .gs = e_gs,
                .sb_dtype = e_sbd,
            };
        } else if (keep_bf16) {
            const buf = try allocTracked(w, ctx, align64(ew.bytes.len));
            var off: usize = 0;
            const ref = copyInto(buf, &off, ew.bytes);
            w.embed = .{
                .codes = ref,
                .scales = ref,
                .biases = ref,
                .bits = 0,
                .gs = 64,
                .sb_dtype = SbDType.fromSt(ew.dtype),
            };
        } else {
            const codes_len = c.vocab_size * (c.hidden_size / 2);
            const sb_len = c.vocab_size * (c.hidden_size / qk_gs) * 2;
            const total = align64(codes_len) + align64(sb_len) * 2;
            const buf = try allocTracked(w, ctx, total);
            const codes_off: usize = 0;
            const scales_off: usize = align64(codes_len);
            const biases_off: usize = scales_off + align64(sb_len);
            quantizeEmbedBf16(
                ew,
                c.vocab_size,
                c.hidden_size,
                buf.ptr + codes_off,
                @ptrCast(@alignCast(buf.ptr + scales_off)),
                @ptrCast(@alignCast(buf.ptr + biases_off)),
            );
            w.embed = .{
                .codes = buf.view(codes_off),
                .scales = buf.view(scales_off),
                .biases = buf.view(biases_off),
                .bits = 4,
                .gs = 64,
                .sb_dtype = .bf16,
            };
        }

        const hw = if (tied_head) ew else try ckpt.get("lm_head.weight");
        const hs_opt = if (tied_head) es_opt else ckpt.getOptional("lm_head.scales");
        const hb_opt = if (tied_head) eb_opt else ckpt.getOptional("lm_head.biases");
        const h_srcs = [1]RawQuant{.{ .w = hw, .s = hs_opt, .b = hb_opt, .n = c.vocab_size }};
        const head_bytes = measureLinear(&h_srcs, c.vocab_size, c.hidden_size, keep_bf16);
        const head_buf = try allocTracked(w, ctx, head_bytes);
        var h_off: usize = 0;
        w.lm_head = packLinearInto(head_buf, &h_off, &h_srcs, c.vocab_size, c.hidden_size, keep_bf16);
    }

    // 2. Small non-quantized weights (norms, Mamba-2 params, MoE routers)
    {
        const fn_t = try ckpt.get("backbone.norm_f.weight");
        var small_bytes: usize = align64(fn_t.bytes.len);
        for (0..c.num_layers) |li| {
            small_bytes += align64(c.hidden_size * 2); // norm.weight
            switch (c.layer_kinds[li]) {
                .mamba => {
                    small_bytes += align64(c.mambaConvDim() * c.conv_kernel * 2); // conv1d.weight
                    small_bytes += align64(c.mambaConvDim() * 2); // conv1d.bias
                    small_bytes += align64(c.mamba_num_heads * 4); // A_log f32
                    small_bytes += align64(c.mamba_num_heads * 2); // dt_bias bf16
                    small_bytes += align64(c.mamba_num_heads * 2); // D bf16
                    small_bytes += align64(c.mambaInterDim() * 2); // norm.weight bf16
                },
                .moe => {
                    small_bytes += align64(c.n_routed_experts * c.hidden_size * 2); // gate.weight bf16
                    small_bytes += align64(c.n_routed_experts * 4); // e_score_correction_bias f32
                },
                else => {},
            }
        }
        const small_buf = try allocTracked(w, ctx, small_bytes);
        var off: usize = 0;
        w.final_norm = copyInto(small_buf, &off, fn_t.bytes);

        var nb: [160]u8 = undefined;
        for (0..c.num_layers) |li| {
            w.layers[li].kind = c.layer_kinds[li];
            const nw = try getLayer(ckpt, &nb, prefix, li, "norm.weight");
            w.layers[li].input_norm = copyInto(small_buf, &off, nw.bytes);
            w.layers[li].post_attn_norm = w.layers[li].input_norm;

            switch (c.layer_kinds[li]) {
                .mamba => {
                    const cw = try getLayer(ckpt, &nb, prefix, li, "mixer.conv1d.weight");
                    const cb = try getLayer(ckpt, &nb, prefix, li, "mixer.conv1d.bias");
                    const al = try getLayer(ckpt, &nb, prefix, li, "mixer.A_log");
                    const dtb = try getLayer(ckpt, &nb, prefix, li, "mixer.dt_bias");
                    const dsk = try getLayer(ckpt, &nb, prefix, li, "mixer.D");
                    const mn = try getLayer(ckpt, &nb, prefix, li, "mixer.norm.weight");
                    w.layers[li].mamba.conv_weight = copyInto(small_buf, &off, cw.bytes);
                    w.layers[li].mamba.conv_bias = copyInto(small_buf, &off, cb.bytes);
                    w.layers[li].mamba.a_log = copyAsF32(small_buf, &off, al);
                    w.layers[li].mamba.dt_bias = copyInto(small_buf, &off, dtb.bytes);
                    w.layers[li].mamba.d_skip = copyInto(small_buf, &off, dsk.bytes);
                    w.layers[li].mamba.mamba_norm = copyInto(small_buf, &off, mn.bytes);
                },
                .moe => {
                    const gw = try getLayer(ckpt, &nb, prefix, li, "mixer.gate.weight");
                    const gb = try getLayer(ckpt, &nb, prefix, li, "mixer.gate.e_score_correction_bias");
                    w.layers[li].moe.gate_weight = copyInto(small_buf, &off, gw.bytes);
                    w.layers[li].moe.gate_bias = copyAsF32(small_buf, &off, gb);
                },
                else => {
                    w.layers[li].full.q_norm = small_buf.view(0);
                    w.layers[li].full.k_norm = small_buf.view(0);
                    w.layers[li].full.qkv_bias = null;
                },
            }
        }
    }

    // 3. Per-layer packed qk weights (Mamba-2 / Sparse MoE / NoPE GQA Attention)
    const mamba_in_n = c.mambaInProjOut();
    const mamba_inter = c.mambaInterDim();
    const attn_qkv_n = c.qProjOut() + 2 * c.kvProjOut();
    const attn_o_k = c.num_heads * c.head_dim;
    const fc1_stride = align64(QkMatrix.byteSize(c.moe_intermediate_size, c.hidden_size));
    const fc2_stride = align64(QkMatrix.byteSize(c.hidden_size, c.moe_intermediate_size));

    var nb: [160]u8 = undefined;
    var b1: [160]u8 = undefined;
    var b2: [160]u8 = undefined;
    for (0..c.num_layers) |li| {
        switch (c.layer_kinds[li]) {
            .mamba => {
                const in_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.in_proj", mamba_in_n, c.hidden_size);
                const out_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.out_proj", c.hidden_size, mamba_inter);
                const in_srcs = [1]RawQuant{in_q};
                const out_srcs = [1]RawQuant{out_q};
                const in_bytes = measureLinear(&in_srcs, mamba_in_n, c.hidden_size, keep_bf16);
                const out_bytes = measureLinear(&out_srcs, c.hidden_size, mamba_inter, keep_bf16);
                const lbuf: ?Buffer = if (in_bytes + out_bytes > 0) try allocTracked(w, ctx, in_bytes + out_bytes) else null;
                var off: usize = 0;

                w.layers[li].mamba.in_proj = packLinearInto(lbuf, &off, &in_srcs, mamba_in_n, c.hidden_size, keep_bf16);
                w.layers[li].mamba.out_proj = packLinearInto(lbuf, &off, &out_srcs, c.hidden_size, mamba_inter, keep_bf16);
            },
            .moe => {
                const su_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.shared_experts.up_proj", c.moe_shared_intermediate_size, c.hidden_size);
                const sd_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.shared_experts.down_proj", c.hidden_size, c.moe_shared_intermediate_size);
                const su_srcs = [1]RawQuant{su_q};
                const sd_srcs = [1]RawQuant{sd_q};
                const su_bytes = measureLinear(&su_srcs, c.moe_shared_intermediate_size, c.hidden_size, keep_bf16);
                const sd_bytes = measureLinear(&sd_srcs, c.hidden_size, c.moe_shared_intermediate_size, keep_bf16);
                const fc1_total = c.n_routed_experts * fc1_stride;
                const fc2_total = c.n_routed_experts * fc2_stride;
                const lbuf = try allocTracked(w, ctx, su_bytes + sd_bytes + fc1_total + fc2_total);
                var off: usize = 0;

                w.layers[li].moe.shared_up = packLinearInto(lbuf, &off, &su_srcs, c.moe_shared_intermediate_size, c.hidden_size, keep_bf16);
                w.layers[li].moe.shared_down = packLinearInto(lbuf, &off, &sd_srcs, c.hidden_size, c.moe_shared_intermediate_size, keep_bf16);

                const fc1_w = try ckpt.get(std.fmt.bufPrint(&nb, "{s}layers.{d}.mixer.switch_mlp.fc1.weight", .{ prefix, li }) catch return error.NameTooLong);
                const fc1_s = ckpt.getOptional(std.fmt.bufPrint(&b1, "{s}layers.{d}.mixer.switch_mlp.fc1.scales", .{ prefix, li }) catch return error.NameTooLong);
                const fc1_b = ckpt.getOptional(std.fmt.bufPrint(&b2, "{s}layers.{d}.mixer.switch_mlp.fc1.biases", .{ prefix, li }) catch return error.NameTooLong);
                const fc1_off = off;
                packQkExperts(lbuf.ptr + fc1_off, fc1_w, fc1_s, fc1_b, c.n_routed_experts, c.moe_intermediate_size, c.hidden_size, fc1_stride);
                off += fc1_total;
                w.layers[li].moe.fc1_experts = lbuf.view(fc1_off);
                w.layers[li].moe.fc1_expert_stride = fc1_stride;

                const fc2_w = try ckpt.get(std.fmt.bufPrint(&nb, "{s}layers.{d}.mixer.switch_mlp.fc2.weight", .{ prefix, li }) catch return error.NameTooLong);
                const fc2_s = ckpt.getOptional(std.fmt.bufPrint(&b1, "{s}layers.{d}.mixer.switch_mlp.fc2.scales", .{ prefix, li }) catch return error.NameTooLong);
                const fc2_b = ckpt.getOptional(std.fmt.bufPrint(&b2, "{s}layers.{d}.mixer.switch_mlp.fc2.biases", .{ prefix, li }) catch return error.NameTooLong);
                const fc2_off = off;
                packQkExperts(lbuf.ptr + fc2_off, fc2_w, fc2_s, fc2_b, c.n_routed_experts, c.hidden_size, c.moe_intermediate_size, fc2_stride);
                off += fc2_total;
                w.layers[li].moe.fc2_experts = lbuf.view(fc2_off);
                w.layers[li].moe.fc2_expert_stride = fc2_stride;
            },
            .full => {
                const q_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.q_proj", c.qProjOut(), c.hidden_size);
                const k_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.k_proj", c.kvProjOut(), c.hidden_size);
                const v_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.v_proj", c.kvProjOut(), c.hidden_size);
                const qkv_srcs = [3]RawQuant{ q_q, k_q, v_q };
                const o_q = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, "mixer.o_proj", c.hidden_size, attn_o_k);
                const o_srcs = [1]RawQuant{o_q};
                const qkv_bytes = measureLinear(&qkv_srcs, attn_qkv_n, c.hidden_size, keep_bf16);
                const o_bytes = measureLinear(&o_srcs, c.hidden_size, attn_o_k, keep_bf16);
                const lbuf: ?Buffer = if (qkv_bytes + o_bytes > 0) try allocTracked(w, ctx, qkv_bytes + o_bytes) else null;
                var off: usize = 0;

                w.layers[li].full.qkv_proj = packLinearInto(lbuf, &off, &qkv_srcs, attn_qkv_n, c.hidden_size, keep_bf16);
                w.layers[li].full.o_proj = packLinearInto(lbuf, &off, &o_srcs, c.hidden_size, attn_o_k, keep_bf16);
            },
            .linear, .short_conv => unreachable,
        }
    }
}

fn copyTensorAsBf16Into(buf: Buffer, off: *usize, t: st.Tensor) BufferRef {
    const start = off.*;
    const byte_len = t.elemCount() * 2;
    copyRawAsBf16(buf.ptr + start, t);
    off.* += align64(byte_len);
    return buf.view(start);
}

fn getLayerOpt(ckpt: *const st.Checkpoint, buf: *[160]u8, prefix: []const u8, layer: usize, suffix: []const u8) ?st.Tensor {
    const key = std.fmt.bufPrint(buf, "{s}layers.{d}.{s}", .{ prefix, layer, suffix }) catch return null;
    return ckpt.getOptional(key);
}

fn loadSwigluMoeLayer(
    w: *ModelWeights,
    ctx: *mtl.Context,
    c: ModelConfig,
    ckpt: *const st.Checkpoint,
    ts_ptr: ?*const tfq3_mod.Store,
    prefix: []const u8,
    li: usize,
    keep_bf16: bool,
) !SwigluMoeWeights {
    var nb: [160]u8 = undefined;
    var b1: [160]u8 = undefined;
    var b2: [160]u8 = undefined;
    const E = c.n_routed_experts;
    const H = c.hidden_size;
    const I = c.moe_intermediate_size;

    const moe_pfx: []const u8 = if (getLayerOpt(ckpt, &nb, prefix, li, "block_sparse_moe.gate.weight") != null)
        "block_sparse_moe."
    else if (getLayerOpt(ckpt, &nb, prefix, li, "feed_forward.gate.weight") != null)
        "feed_forward."
    else
        "mlp.";

    // 1. Router weights (+ optional e_score_correction_bias / expert_bias + optional shared_expert_gate)
    const gate_t = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}gate.weight", .{moe_pfx}) catch return error.NameTooLong);
    const bias_t = getLayerOpt(ckpt, &nb, prefix, li, "mlp.gate.e_score_correction_bias") orelse
        getLayerOpt(ckpt, &nb, prefix, li, "feed_forward.expert_bias");
    const sh_gate_t = getLayerOpt(ckpt, &nb, prefix, li, "mlp.shared_expert_gate.weight");

    var meta_bytes: usize = align64(gate_t.elemCount() * 2);
    if (bias_t) |bt| meta_bytes += align64(bt.elemCount() * 4);
    if (sh_gate_t) |sgt| meta_bytes += align64(sgt.elemCount() * 2);

    // Optional shared expert (Qwen3.5 MoE / DeepSeek-V3 / GLM)
    const sh_mod: ?[]const u8 = if (getLayerOpt(ckpt, &nb, prefix, li, "mlp.shared_expert.gate_proj.weight") != null)
        "mlp.shared_expert"
    else if (getLayerOpt(ckpt, &nb, prefix, li, "mlp.shared_experts.gate_proj.weight") != null)
        "mlp.shared_experts"
    else
        null;

    var sh_gu_srcs: [2]RawQuant = undefined;
    var sh_dn_srcs: [1]RawQuant = undefined;
    var sh_gu_bytes: usize = 0;
    var sh_dn_bytes: usize = 0;
    var sh_inter: usize = c.moe_shared_intermediate_size;
    if (sh_mod) |sm| {
        const g_sub = std.fmt.bufPrint(&b1, "{s}.gate_proj", .{sm}) catch return error.NameTooLong;
        const u_sub = std.fmt.bufPrint(&b2, "{s}.up_proj", .{sm}) catch return error.NameTooLong;
        var b3: [160]u8 = undefined;
        const g_t_peek = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b3, "{s}.gate_proj.weight", .{sm}) catch return error.NameTooLong);
        if (sh_inter == 0) sh_inter = g_t_peek.shape[0];
        sh_gu_srcs[0] = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, g_sub, sh_inter, H);
        sh_gu_srcs[1] = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, u_sub, sh_inter, H);
        const d_sub = std.fmt.bufPrint(&b1, "{s}.down_proj", .{sm}) catch return error.NameTooLong;
        sh_dn_srcs[0] = try getQuantLayer(ckpt, ts_ptr, &nb, prefix, li, d_sub, H, sh_inter);
        sh_gu_bytes = measureLinear(&sh_gu_srcs, 2 * sh_inter, H, keep_bf16);
        sh_dn_bytes = measureLinear(&sh_dn_srcs, H, sh_inter, keep_bf16);
    }

    const meta_buf = try allocTracked(w, ctx, meta_bytes + sh_gu_bytes + sh_dn_bytes);
    var moff: usize = 0;
    var out = SwigluMoeWeights{
        .gate_weight = copyTensorAsBf16Into(meta_buf, &moff, gate_t),
        .gate_bias = if (bias_t) |bt| copyAsF32(meta_buf, &moff, bt) else null,
        .gate_codes = meta_buf.view(0),
        .gate_scales = meta_buf.view(0),
        .gate_biases = meta_buf.view(0),
        .up_codes = meta_buf.view(0),
        .up_scales = meta_buf.view(0),
        .up_biases = meta_buf.view(0),
        .down_codes = meta_buf.view(0),
        .down_scales = meta_buf.view(0),
        .down_biases = meta_buf.view(0),
        .has_shared = (sh_mod != null),
        .shared_gate_vec = if (sh_gate_t) |sgt| copyTensorAsBf16Into(meta_buf, &moff, sgt) else null,
    };
    if (sh_mod != null) {
        out.shared_gate_up = packLinearInto(meta_buf, &moff, &sh_gu_srcs, 2 * sh_inter, H, keep_bf16);
        out.shared_down = packLinearInto(meta_buf, &moff, &sh_dn_srcs, H, sh_inter, keep_bf16);
    }

    // 2. Routed experts — detect Layout A (3D switch_mlp), Layout B (3D fused gate_up_proj), or Layout C (per-expert)
    const sw_g_key = std.fmt.bufPrint(&nb, "{s}switch_mlp.gate_proj.weight", .{moe_pfx}) catch return error.NameTooLong;
    if (getLayerOpt(ckpt, &b1, prefix, li, sw_g_key)) |sw_gw| {
        const sw_uw = try getLayer(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.up_proj.weight", .{moe_pfx}) catch return error.NameTooLong);
        const sw_dw = try getLayer(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.down_proj.weight", .{moe_pfx}) catch return error.NameTooLong);
        const sw_gs = getLayerOpt(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.gate_proj.scales", .{moe_pfx}) catch return error.NameTooLong);
        const sw_gb = getLayerOpt(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.gate_proj.biases", .{moe_pfx}) catch return error.NameTooLong);
        const sw_us = getLayerOpt(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.up_proj.scales", .{moe_pfx}) catch return error.NameTooLong);
        const sw_ub = getLayerOpt(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.up_proj.biases", .{moe_pfx}) catch return error.NameTooLong);
        const sw_ds = getLayerOpt(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.down_proj.scales", .{moe_pfx}) catch return error.NameTooLong);
        const sw_db = getLayerOpt(ckpt, &b1, prefix, li, std.fmt.bufPrint(&nb, "{s}switch_mlp.down_proj.biases", .{moe_pfx}) catch return error.NameTooLong);

        if (sw_gs != null and sw_gb != null and sw_us != null and sw_ub != null and sw_ds != null and sw_db != null) {
            const gs_t = sw_gs.?;
            const groups = gs_t.shape[gs_t.rank - 1];
            const gs_u16: u16 = @intCast(H / groups);
            const words_per_row = sw_gw.bytes.len / (E * I * 4);
            const bits_u8: u8 = @intCast((words_per_row * 32) / H);
            const sbd = SbDType.fromSt(gs_t.dtype);

            if (bits_u8 == 4 and gs_u16 == 64 and sbd == .bf16 and I % qk_nt == 0 and H % qk_gs == 0 and H % qk_nt == 0 and I % qk_gs == 0) {
                const gu_stride = align64(QkMatrix.byteSize(I, H));
                const dn_stride = align64(QkMatrix.byteSize(H, I));
                const ebuf = try allocTracked(w, ctx, E * gu_stride * 2 + E * dn_stride);
                var eoff: usize = 0;
                out.storage = .qk4;
                out.gu_code_stride = gu_stride;
                out.dn_code_stride = dn_stride;
                out.gate_codes = ebuf.view(eoff);
                packQkExperts(ebuf.ptr + eoff, sw_gw, sw_gs, sw_gb, E, I, H, gu_stride);
                eoff += E * gu_stride;
                out.up_codes = ebuf.view(eoff);
                packQkExperts(ebuf.ptr + eoff, sw_uw, sw_us, sw_ub, E, I, H, gu_stride);
                eoff += E * gu_stride;
                out.down_codes = ebuf.view(eoff);
                packQkExperts(ebuf.ptr + eoff, sw_dw, sw_ds, sw_db, E, H, I, dn_stride);
            } else {
                const total = align64(sw_gw.bytes.len) + align64(sw_gs.?.bytes.len) + align64(sw_gb.?.bytes.len) +
                    align64(sw_uw.bytes.len) + align64(sw_us.?.bytes.len) + align64(sw_ub.?.bytes.len) +
                    align64(sw_dw.bytes.len) + align64(sw_ds.?.bytes.len) + align64(sw_db.?.bytes.len);
                const ebuf = try allocTracked(w, ctx, total);
                var eoff: usize = 0;
                out.storage = .affine;
                out.bits = bits_u8;
                out.gs = gs_u16;
                out.sb_dtype = sbd;
                out.gu_code_stride = sw_gw.bytes.len / E;
                out.gu_sb_stride = sw_gs.?.bytes.len / E;
                out.dn_code_stride = sw_dw.bytes.len / E;
                out.dn_sb_stride = sw_ds.?.bytes.len / E;
                out.gate_codes = copyInto(ebuf, &eoff, sw_gw.bytes);
                out.gate_scales = copyInto(ebuf, &eoff, sw_gs.?.bytes);
                out.gate_biases = copyInto(ebuf, &eoff, sw_gb.?.bytes);
                out.up_codes = copyInto(ebuf, &eoff, sw_uw.bytes);
                out.up_scales = copyInto(ebuf, &eoff, sw_us.?.bytes);
                out.up_biases = copyInto(ebuf, &eoff, sw_ub.?.bytes);
                out.down_codes = copyInto(ebuf, &eoff, sw_dw.bytes);
                out.down_scales = copyInto(ebuf, &eoff, sw_ds.?.bytes);
                out.down_biases = copyInto(ebuf, &eoff, sw_db.?.bytes);
            }
            return out;
        } else {
            // Raw BF16/F16/F32 3D switch_mlp
            if (!keep_bf16 and sw_gw.dtype == .bf16 and I % qk_nt == 0 and H % qk_gs == 0 and H % qk_nt == 0 and I % qk_gs == 0) {
                const gu_stride = align64(QkMatrix.byteSize(I, H));
                const dn_stride = align64(QkMatrix.byteSize(H, I));
                const ebuf = try allocTracked(w, ctx, E * gu_stride * 2 + E * dn_stride);
                var eoff: usize = 0;
                out.storage = .qk4;
                out.gu_code_stride = gu_stride;
                out.dn_code_stride = dn_stride;
                out.gate_codes = ebuf.view(eoff);
                packQkExperts(ebuf.ptr + eoff, sw_gw, null, null, E, I, H, gu_stride);
                eoff += E * gu_stride;
                out.up_codes = ebuf.view(eoff);
                packQkExperts(ebuf.ptr + eoff, sw_uw, null, null, E, I, H, gu_stride);
                eoff += E * gu_stride;
                out.down_codes = ebuf.view(eoff);
                packQkExperts(ebuf.ptr + eoff, sw_dw, null, null, E, H, I, dn_stride);
            } else {
                const gu_bytes = E * I * H * 2;
                const dn_bytes = E * H * I * 2;
                const ebuf = try allocTracked(w, ctx, align64(gu_bytes) * 2 + align64(dn_bytes));
                var eoff: usize = 0;
                out.storage = .bf16;
                out.gu_code_stride = I * H * 2;
                out.dn_code_stride = H * I * 2;
                out.gate_codes = copyTensorAsBf16Into(ebuf, &eoff, sw_gw);
                out.up_codes = copyTensorAsBf16Into(ebuf, &eoff, sw_uw);
                out.down_codes = copyTensorAsBf16Into(ebuf, &eoff, sw_dw);
            }
            return out;
        }
    }

    // Layout B: HF Qwen3.5 MoE fused 3D `mlp.experts.gate_up_proj` [E, 2*I, H] + `mlp.experts.down_proj` [E, H, I]
    if (getLayerOpt(ckpt, &nb, prefix, li, "mlp.experts.gate_up_proj")) |gu_3d| {
        const dn_3d = try getLayer(ckpt, &b1, prefix, li, "mlp.experts.down_proj");
        const elem_b: usize = if (gu_3d.dtype == .f32) 4 else 2;
        const single_gu_src = I * H * elem_b;
        const single_dn_src = H * I * elem_b;
        const gu_bytes = E * I * H * 2;
        const dn_bytes = E * H * I * 2;
        const ebuf = try allocTracked(w, ctx, align64(gu_bytes) * 2 + align64(dn_bytes));
        const g_off: usize = 0;
        const u_off: usize = align64(gu_bytes);
        const d_off: usize = u_off + align64(gu_bytes);
        for (0..E) |e| {
            var gt = gu_3d;
            gt.bytes = gu_3d.bytes[e * 2 * single_gu_src ..][0..single_gu_src];
            gt.shape = .{ I, H, 0, 0, 0, 0 };
            gt.rank = 2;
            copyRawAsBf16(ebuf.ptr + g_off + e * I * H * 2, gt);

            var ut = gu_3d;
            ut.bytes = gu_3d.bytes[(e * 2 + 1) * single_gu_src ..][0..single_gu_src];
            ut.shape = .{ I, H, 0, 0, 0, 0 };
            ut.rank = 2;
            copyRawAsBf16(ebuf.ptr + u_off + e * I * H * 2, ut);

            var dt = dn_3d;
            dt.bytes = dn_3d.bytes[e * single_dn_src ..][0..single_dn_src];
            dt.shape = .{ H, I, 0, 0, 0, 0 };
            dt.rank = 2;
            copyRawAsBf16(ebuf.ptr + d_off + e * H * I * 2, dt);
        }
        out.storage = .bf16;
        out.gu_code_stride = I * H * 2;
        out.dn_code_stride = H * I * 2;
        out.gate_codes = ebuf.view(g_off);
        out.up_codes = ebuf.view(u_off);
        out.down_codes = ebuf.view(d_off);
        return out;
    }

    // Layout C: Per-expert tensors (`mlp.experts.{e}.{gate_proj,up_proj,down_proj}` or Mixtral/LFM2 `*.experts.{e}.{w1,w3,w2}`)
    const is_mixtral_names = (getLayerOpt(ckpt, &nb, prefix, li, "block_sparse_moe.experts.0.w1.weight") != null) or
        (getLayerOpt(ckpt, &nb, prefix, li, "feed_forward.experts.0.w1.weight") != null);
    const g_name: []const u8 = if (is_mixtral_names) "w1" else "gate_proj";
    const u_name: []const u8 = if (is_mixtral_names) "w3" else "up_proj";
    const d_name: []const u8 = if (is_mixtral_names) "w2" else "down_proj";

    const e0_g_w = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.0.{s}.weight", .{ moe_pfx, g_name }) catch return error.NameTooLong);
    const e0_g_s = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.0.{s}.scales", .{ moe_pfx, g_name }) catch return error.NameTooLong);
    const e0_g_b = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.0.{s}.biases", .{ moe_pfx, g_name }) catch return error.NameTooLong);
    const e0_d_w = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.0.{s}.weight", .{ moe_pfx, d_name }) catch return error.NameTooLong);
    const e0_d_s = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.0.{s}.scales", .{ moe_pfx, d_name }) catch return error.NameTooLong);

    if (e0_g_s != null and e0_g_b != null and e0_d_s != null) {
        const gs_t = e0_g_s.?;
        const groups = gs_t.shape[gs_t.rank - 1];
        const gs_u16: u16 = @intCast(H / groups);
        const words_per_row = e0_g_w.bytes.len / (I * 4);
        const bits_u8: u8 = @intCast((words_per_row * 32) / H);
        const sbd = SbDType.fromSt(gs_t.dtype);

        if (bits_u8 == 4 and gs_u16 == 64 and sbd == .bf16 and I % qk_nt == 0 and H % qk_gs == 0 and H % qk_nt == 0 and I % qk_gs == 0) {
            const gu_stride = align64(QkMatrix.byteSize(I, H));
            const dn_stride = align64(QkMatrix.byteSize(H, I));
            const ebuf = try allocTracked(w, ctx, E * gu_stride * 2 + E * dn_stride);
            const g_base: usize = 0;
            const u_base: usize = E * gu_stride;
            const d_base: usize = u_base + E * gu_stride;
            for (0..E) |e| {
                const gw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
                const gs = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.scales", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
                const gb = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.biases", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
                packQkConcat(ebuf.ptr + g_base + e * gu_stride, &.{.{ .w = gw, .s = gs, .b = gb, .n = I }}, I, H);

                const uw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
                const us = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.scales", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
                const ub = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.biases", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
                packQkConcat(ebuf.ptr + u_base + e * gu_stride, &.{.{ .w = uw, .s = us, .b = ub, .n = I }}, I, H);

                const dw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
                const ds = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.scales", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
                const db = getLayerOpt(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.biases", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
                packQkConcat(ebuf.ptr + d_base + e * dn_stride, &.{.{ .w = dw, .s = ds, .b = db, .n = H }}, H, I);
            }
            out.storage = .qk4;
            out.gu_code_stride = gu_stride;
            out.dn_code_stride = dn_stride;
            out.gate_codes = ebuf.view(g_base);
            out.up_codes = ebuf.view(u_base);
            out.down_codes = ebuf.view(d_base);
        } else {
            const gu_cs = e0_g_w.bytes.len;
            const gu_ss = e0_g_s.?.bytes.len;
            const dn_cs = e0_d_w.bytes.len;
            const dn_ss = e0_d_s.?.bytes.len;
            const total = align64(E * gu_cs) * 2 + align64(E * gu_ss) * 4 + align64(E * dn_cs) + align64(E * dn_ss) * 2;
            const ebuf = try allocTracked(w, ctx, total);
            var eoff: usize = 0;
            const gc_off = eoff;
            eoff += align64(E * gu_cs);
            const gs_off = eoff;
            eoff += align64(E * gu_ss);
            const gb_off = eoff;
            eoff += align64(E * gu_ss);
            const uc_off = eoff;
            eoff += align64(E * gu_cs);
            const us_off = eoff;
            eoff += align64(E * gu_ss);
            const ub_off = eoff;
            eoff += align64(E * gu_ss);
            const dc_off = eoff;
            eoff += align64(E * dn_cs);
            const ds_off = eoff;
            eoff += align64(E * dn_ss);
            const db_off = eoff;

            for (0..E) |e| {
                const gw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
                const gs = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.scales", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
                const gb = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.biases", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
                @memcpy(ebuf.ptr[gc_off + e * gu_cs ..][0..gu_cs], gw.bytes);
                @memcpy(ebuf.ptr[gs_off + e * gu_ss ..][0..gu_ss], gs.bytes);
                @memcpy(ebuf.ptr[gb_off + e * gu_ss ..][0..gu_ss], gb.bytes);

                const uw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
                const us = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.scales", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
                const ub = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.biases", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
                @memcpy(ebuf.ptr[uc_off + e * gu_cs ..][0..gu_cs], uw.bytes);
                @memcpy(ebuf.ptr[us_off + e * gu_ss ..][0..gu_ss], us.bytes);
                @memcpy(ebuf.ptr[ub_off + e * gu_ss ..][0..gu_ss], ub.bytes);

                const dw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
                const ds = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.scales", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
                const db = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.biases", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
                @memcpy(ebuf.ptr[dc_off + e * dn_cs ..][0..dn_cs], dw.bytes);
                @memcpy(ebuf.ptr[ds_off + e * dn_ss ..][0..dn_ss], ds.bytes);
                @memcpy(ebuf.ptr[db_off + e * dn_ss ..][0..dn_ss], db.bytes);
            }
            out.storage = .affine;
            out.bits = bits_u8;
            out.gs = gs_u16;
            out.sb_dtype = sbd;
            out.gu_code_stride = gu_cs;
            out.gu_sb_stride = gu_ss;
            out.dn_code_stride = dn_cs;
            out.dn_sb_stride = dn_ss;
            out.gate_codes = ebuf.view(gc_off);
            out.gate_scales = ebuf.view(gs_off);
            out.gate_biases = ebuf.view(gb_off);
            out.up_codes = ebuf.view(uc_off);
            out.up_scales = ebuf.view(us_off);
            out.up_biases = ebuf.view(ub_off);
            out.down_codes = ebuf.view(dc_off);
            out.down_scales = ebuf.view(ds_off);
            out.down_biases = ebuf.view(db_off);
        }
        return out;
    }

    // Raw BF16/F16/F32 per-expert weights
    if (!keep_bf16 and e0_g_w.dtype == .bf16 and I % qk_nt == 0 and H % qk_gs == 0 and H % qk_nt == 0 and I % qk_gs == 0) {
        const gu_stride = align64(QkMatrix.byteSize(I, H));
        const dn_stride = align64(QkMatrix.byteSize(H, I));
        const ebuf = try allocTracked(w, ctx, E * gu_stride * 2 + E * dn_stride);
        const g_base: usize = 0;
        const u_base: usize = E * gu_stride;
        const d_base: usize = u_base + E * gu_stride;
        for (0..E) |e| {
            const gw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
            packQkConcat(ebuf.ptr + g_base + e * gu_stride, &.{.{ .w = gw, .s = null, .b = null, .n = I }}, I, H);
            const uw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
            packQkConcat(ebuf.ptr + u_base + e * gu_stride, &.{.{ .w = uw, .s = null, .b = null, .n = I }}, I, H);
            const dw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
            packQkConcat(ebuf.ptr + d_base + e * dn_stride, &.{.{ .w = dw, .s = null, .b = null, .n = H }}, H, I);
        }
        out.storage = .qk4;
        out.gu_code_stride = gu_stride;
        out.dn_code_stride = dn_stride;
        out.gate_codes = ebuf.view(g_base);
        out.up_codes = ebuf.view(u_base);
        out.down_codes = ebuf.view(d_base);
    } else {
        const gu_s = I * H * 2;
        const dn_s = H * I * 2;
        const ebuf = try allocTracked(w, ctx, align64(E * gu_s) * 2 + align64(E * dn_s));
        const g_base: usize = 0;
        const u_base: usize = align64(E * gu_s);
        const d_base: usize = u_base + align64(E * gu_s);
        for (0..E) |e| {
            const gw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, g_name }) catch return error.NameTooLong);
            copyRawAsBf16(ebuf.ptr + g_base + e * gu_s, gw);
            const uw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, u_name }) catch return error.NameTooLong);
            copyRawAsBf16(ebuf.ptr + u_base + e * gu_s, uw);
            const dw = try getLayer(ckpt, &nb, prefix, li, std.fmt.bufPrint(&b1, "{s}experts.{d}.{s}.weight", .{ moe_pfx, e, d_name }) catch return error.NameTooLong);
            copyRawAsBf16(ebuf.ptr + d_base + e * dn_s, dw);
        }
        out.storage = .bf16;
        out.gu_code_stride = gu_s;
        out.dn_code_stride = dn_s;
        out.gate_codes = ebuf.view(g_base);
        out.up_codes = ebuf.view(u_base);
        out.down_codes = ebuf.view(d_base);
    }
    return out;
}

inline fn readScalarAsF32(t: st.Tensor, idx: usize) f32 {
    return switch (t.dtype) {
        .bf16 => bf16ToF32(t.asSlice(u16)[idx]),
        .f16 => @floatCast(@as(f16, @bitCast(t.asSlice(u16)[idx]))),
        .f32 => t.asSlice(f32)[idx],
        else => 0.0,
    };
}

inline fn dequantAffineElem(w_u32: []align(1) const u32, s: st.Tensor, b: st.Tensor, row_idx: usize, col: usize, k: usize, bits: usize, gs: usize) f32 {
    const words_per_row = (k * bits) / 32;
    const bit_off = col * bits;
    const wi = row_idx * words_per_row + (bit_off >> 5);
    const rem: u5 = @intCast(bit_off & 31);
    const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    var code: u32 = (w_u32[wi] >> rem) & mask;
    if (@as(usize, rem) + bits > 32) {
        const hi_shift: u5 = @intCast(32 - @as(usize, rem));
        code |= (w_u32[wi + 1] << hi_shift) & mask;
    }
    const groups = k / gs;
    const sb_idx = row_idx * groups + (col / gs);
    return @as(f32, @floatFromInt(code)) * readScalarAsF32(s, sb_idx) + readScalarAsF32(b, sb_idx);
}

/// Assemble BF16 `kv_b_proj` [num_heads * (qk_nope + v_dim), kv_lora] from MLX-Community split
/// `self_attn.embed_q` [num_heads, kv_lora, qk_nope] and `self_attn.unembed_out` [num_heads, v_dim, kv_lora].
fn assembleMlaKvBFromMlx(ckpt: *const st.Checkpoint, prefix: []const u8, li: usize, c: ModelConfig, dst_bf16: [*]u16) !void {
    var nb: [160]u8 = undefined;
    const eq_w = try getLayer(ckpt, &nb, prefix, li, "self_attn.embed_q.weight");
    const eq_s = getLayerOpt(ckpt, &nb, prefix, li, "self_attn.embed_q.scales");
    const eq_b = getLayerOpt(ckpt, &nb, prefix, li, "self_attn.embed_q.biases");
    const uo_w = try getLayer(ckpt, &nb, prefix, li, "self_attn.unembed_out.weight");
    const uo_s = getLayerOpt(ckpt, &nb, prefix, li, "self_attn.unembed_out.scales");
    const uo_b = getLayerOpt(ckpt, &nb, prefix, li, "self_attn.unembed_out.biases");

    const H = c.num_heads;
    const kv_lora = c.kv_lora_rank;
    const qk_nope = c.qk_nope_head_dim;
    const v_dim = c.v_head_dim;
    const head_rows = qk_nope + v_dim;

    if (eq_s != null and eq_b != null) {
        const eq_u32 = eq_w.asSlice(u32);
        const eq_groups = eq_s.?.shape[eq_s.?.rank - 1];
        const eq_gs = qk_nope / eq_groups;
        const eq_words = eq_w.bytes.len / (H * kv_lora * 4);
        const eq_bits = (eq_words * 32) / qk_nope;
        for (0..H) |h| {
            for (0..kv_lora) |r| {
                const row_idx = h * kv_lora + r;
                for (0..qk_nope) |col| {
                    const val = dequantAffineElem(eq_u32, eq_s.?, eq_b.?, row_idx, col, qk_nope, eq_bits, eq_gs);
                    dst_bf16[(h * head_rows + col) * kv_lora + r] = f32ToBf16(val);
                }
            }
        }
    } else {
        for (0..H) |h| {
            for (0..kv_lora) |r| {
                const base = (h * kv_lora + r) * qk_nope;
                for (0..qk_nope) |col| {
                    dst_bf16[(h * head_rows + col) * kv_lora + r] = f32ToBf16(readScalarAsF32(eq_w, base + col));
                }
            }
        }
    }

    if (uo_s != null and uo_b != null) {
        const uo_u32 = uo_w.asSlice(u32);
        const uo_groups = uo_s.?.shape[uo_s.?.rank - 1];
        const uo_gs = kv_lora / uo_groups;
        const uo_words = uo_w.bytes.len / (H * v_dim * 4);
        const uo_bits = (uo_words * 32) / kv_lora;
        for (0..H) |h| {
            for (0..v_dim) |v| {
                const row_idx = h * v_dim + v;
                const dst_row = (h * head_rows + qk_nope + v) * kv_lora;
                for (0..kv_lora) |r| {
                    const val = dequantAffineElem(uo_u32, uo_s.?, uo_b.?, row_idx, r, kv_lora, uo_bits, uo_gs);
                    dst_bf16[dst_row + r] = f32ToBf16(val);
                }
            }
        }
    } else {
        for (0..H) |h| {
            for (0..v_dim) |v| {
                const base = (h * v_dim + v) * kv_lora;
                const dst_row = (h * head_rows + qk_nope + v) * kv_lora;
                for (0..kv_lora) |r| {
                    dst_bf16[dst_row + r] = f32ToBf16(readScalarAsF32(uo_w, base + r));
                }
            }
        }
    }
}
