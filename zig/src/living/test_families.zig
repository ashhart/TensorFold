//! A family for tests: a SwiGLU shared expert shaped as Qwen3.8 Flash Next's, which no checkpoint learns into yet.
const Family = @import("family.zig").Family;

/// A SwiGLU shared expert at every layer, gated by its own router logit, in 4-bit groups of 32, keys under text_config.
pub const swiglu: Family = .{
    .model_type = "swiglu_test",
    .names = .{
        .layers = "language_model.model.layers.",
        .expert = "mlp.shared_expert.",
        .inputs = &.{ "gate_proj", "up_proj" },
        .mtp = .{ .base = "language_model.mtp.layers.0.mlp.shared_expert." },
    },
    // silu(gate) is not zero below it as relu is: a wider margin keeps sigmoid(-16 * 0.5) = 3e-4 of it there
    .neuron = .{ .act = .swiglu, .gated = true, .margin = 0.5, .sharp = 16 },
    .quant = .{ .bits = 4, .group = 32 },
    .keys = .{ .scope = "text_config", .width = "shared_expert_intermediate_size" },
    .experts = .every,
};
