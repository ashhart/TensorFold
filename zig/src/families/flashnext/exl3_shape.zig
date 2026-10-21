//! Published text shape of turboderp/Qwen3.8-Flash-Next-exl3, revision 3.05bpw_h5_ng5.
//! Read from that revision's config.json on 2026-10-16 (text_config + quantization_config).
//! Per-tensor EXL3 widths still come from each trellis, never from `bits`.

const std = @import("std");

pub const model = "turboderp/Qwen3.8-Flash-Next-exl3";
pub const revision = "3.05bpw_h5_ng5";

pub const hidden: usize = 2560;
pub const layers: usize = 48;
pub const vocab: usize = 248320;
pub const heads: usize = 24;
pub const kv_heads: usize = 2;
pub const head_dim: usize = 256;
pub const experts: usize = 512;
pub const topk: usize = 10;
pub const expert_intermediate: usize = 640;
pub const shared_intermediate: usize = 640;
/// Routed ids are 0..511. The shared expert is the next id in the same table.
pub const shared_expert: usize = experts;
pub const hc_count: usize = 4;
pub const hc_lowrank: usize = 320;

pub const ngram: usize = 3;
pub const ngram_heads_per: usize = 8;
pub const ngram_dh: usize = 160;
/// Branch name `ng5`: the n-gram row codec is 5 bits. Confirmed by words-per-row once a shard header is read.
pub const ngram_bits: usize = 5;
pub const ngram_shards: usize = 128;
pub const ngram_vocab_base: usize = 20_000_000;

pub const quant_method = "exl3";
pub const quant_version = "1.4.4";
pub const average_bits: f32 = 3.05;
pub const head_bits: usize = 5;
pub const mtp_bits: usize = 3;
pub const codebook = "mul1";

pub const tile: usize = 16;
pub const hadamard: usize = 128;
pub const tp_ranks: usize = 2;

pub const linear_layers_per_cycle: usize = 3;
pub const full_attention_interval: usize = 4;

/// Heads in one n-gram lookup: (ngram - 1) * heads_per_ngram.
pub fn ngramHeadCount() usize {
    return (ngram - 1) * ngram_heads_per;
}

/// q_proj outputs. 24 * 256 = 6144.
pub fn qOut() usize {
    return heads * head_dim;
}

/// One attention layer's kv width, keys or values. 2 * 256 = 512.
pub fn kvOut() usize {
    return kv_heads * head_dim;
}

test "the published 3.05 shape is the one the cuts and the budget are written against" {
    try std.testing.expectEqual(@as(usize, 16), ngramHeadCount());
    try std.testing.expectEqual(@as(usize, 6144), qOut());
    try std.testing.expectEqual(@as(usize, 512), kvOut());
    try std.testing.expect(hidden % hadamard == 0);
    try std.testing.expect(expert_intermediate % hadamard == 0);
    try std.testing.expect(qOut() % hadamard == 0);
    // Half of 640 is 320, which is not a Hadamard block. Column TP inside one expert is illegal.
    try std.testing.expect(expert_intermediate / tp_ranks % hadamard != 0);
    try std.testing.expect(hidden / tp_ranks % hadamard == 0);
}
