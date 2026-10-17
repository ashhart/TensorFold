//! Bolt config checks beside the released uniform/Super config suite.
const std = @import("std");
const cfg = @import("config.zig");
const Why = cfg.Why;
const parse = cfg.parse;
const Kind = cfg.Kind;
const Moe = cfg.Moe;
const Config = cfg.Config;
const Format = cfg.Format;
test "a Bolt config: omni wrapper, puzzle blocks with their own experts and widths, LatentMoE" {
    const bolt =
        \\{"model_type": "nemotron_h_omni", "eos_token_id": 11, "llm_config": {"model_type": "nemotron_h_puzzle",
        \\ "hidden_size": 4096, "vocab_size": 131072, "mamba_num_heads": 128, "mamba_head_dim": 64, "n_groups": 8,
        \\ "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        \\ "num_experts": 512, "moe_latent_size": 1024, "moe_shared_expert_intermediate_size": 5376,
        \\ "routed_scaling_factor": 5.0, "eos_token_id": 2, "time_step_limit": [0.0, {"__float__": "Infinity"}],
        \\ "block_configs": [{"block_type": "mamba"},
        \\  {"block_type": "moe", "moe_intermediate_size": 2048, "n_routed_experts": 240, "num_experts_per_tok": 9},
        \\  {"block_type": "attention"},
        \\  {"block_type": "moe", "moe_intermediate_size": 2560, "n_routed_experts": 496, "num_experts_per_tok": 9}]},
        \\ "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}}
    ;
    var why: Why = .{};
    const c = try parse(std.testing.allocator, bolt, &why);
    try std.testing.expectEqual(@as(usize, 4), c.layers);
    try std.testing.expectEqual(Kind.attention, c.kinds[2]);
    try std.testing.expectEqual(@as(usize, 9), c.top_k);
    try std.testing.expectEqual(@as(usize, 9), c.slots());
    try std.testing.expectEqual(@as(usize, 1024), c.expertDims());
    try std.testing.expectEqual(Moe{ .experts = 240, .width = 2048 }, c.layerMoe(1));
    try std.testing.expectEqual(Moe{ .experts = 496, .width = 2560 }, c.layerMoe(3));
    try std.testing.expectEqual(@as(usize, 496), c.experts);
    try std.testing.expectEqual(@as(usize, 2560), c.expert_width);
    try std.testing.expectEqual(@as(usize, 18560), c.projDim());
    try std.testing.expect(c.isEos(2) and c.isEos(11));
    try std.testing.expectEqual(std.math.inf(f32), c.dt_max);
    try std.testing.expectEqual(Format.modelopt, c.format);
    try std.testing.expect(c.omni and c.bolt and c.shares == 1);
}

test "the Bolt checkpoint's own config.json, when it is on this machine" {
    const io = std.testing.io;
    const dir = "/storage/models/nemotron-bolt-74b-24gb";
    std.Io.Dir.cwd().access(io, dir ++ "/config.json", .{}) catch return error.SkipZigTest;
    const c = try Config.read(std.testing.allocator, io, dir);
    try std.testing.expectEqual(@as(usize, 88), c.layers);
    try std.testing.expectEqual(@as(usize, 40), c.count(.moe));
    try std.testing.expectEqual(@as(usize, 8), c.count(.attention));
    try std.testing.expectEqual(@as(usize, 496), c.experts);
    try std.testing.expectEqual(Moe{ .experts = 240, .width = 2048 }, c.layerMoe(1));
}
