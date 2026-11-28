//! GigaChat 4's normalizations by site: which kind each site runs, its width, epsilon and gate constant.
const config = @import("config.zig");

/// GatedNorm's gate width: up [16, n] and down [n, 16] beside each GatedNorm weight.
pub const gate_rank = 16;

pub const Kind = enum {
    /// GatedNorm: y = bf16(rms(x) * w), then y * gating_weight * sigmoid(down(silu(up(y)))), each step rounded to bf16.
    gated,
    /// LlamaRMSNorm: w * bf16(rms(x)); the indexer's key norm, chosen apart from norm_type.
    rms,
    /// The Gated DeltaNet output: w * rms(o) * sigmoid(z) per value head in fp32, then times gate_scale.
    gated_rms_sigmoid,
};

pub const Site = enum {
    input,
    post_self_attn,
    post_attention,
    post_feedforward,
    q_a,
    kv_a,
    final,
    mtp_embed,
    mtp_hidden,
    mtp_head,
    index_key,
    delta_output,
};

pub fn kind(s: Site) Kind {
    return switch (s) {
        .index_key => .rms,
        .delta_output => .gated_rms_sigmoid,
        else => .gated,
    };
}

/// The elements one norm row covers at `s`.
pub fn width(c: *const config.Config, s: Site) u32 {
    return switch (s) {
        .q_a => c.mla.q_lora,
        .kv_a => c.mla.kv_lora,
        .index_key => c.index.dim,
        .delta_output => c.delta.value_dim,
        else => c.hidden,
    };
}

pub fn eps(c: *const config.Config, s: Site) f64 {
    return switch (s) {
        .index_key => c.index.eps,
        .delta_output => c.delta.eps,
        else => c.norm.eps,
    };
}

/// What the normalized row is multiplied by after its gate: gating_weight, the DeltaNet gate scale, or 1.
pub fn gateConstant(c: *const config.Config, s: Site) f64 {
    return switch (kind(s)) {
        .gated => c.norm.gating_weight,
        .gated_rms_sigmoid => c.delta.gate_scale,
        .rms => 1,
    };
}
