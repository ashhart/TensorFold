//! The Nemotron-H config.json fields the decode path needs, plus the shape checks the kernels rely on.

const std = @import("std");

pub const Kind = enum { mamba, moe, attention };

pub const Config = struct {
    hidden_size: u32,
    vocab_size: u32,
    num_hidden_layers: u32,
    layers_block_type: []const []const u8,
    mamba_num_heads: u32,
    mamba_head_dim: u32,
    ssm_state_size: u32,
    n_groups: u32,
    conv_kernel: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    moe_shared_expert_intermediate_size: u32,
    routed_scaling_factor: f32,
    norm_eps: f32,
    layer_norm_epsilon: f32,
    eos_token_id: []const u32,

    pub fn kind(self: Config, layer: usize) Kind {
        const s = self.layers_block_type[layer];
        if (std.mem.eql(u8, s, "mamba")) return .mamba;
        if (std.mem.eql(u8, s, "moe")) return .moe;
        return .attention;
    }

    pub fn xd(self: Config) u32 {
        return self.mamba_num_heads * self.mamba_head_dim;
    }

    pub fn convDim(self: Config) u32 {
        return self.xd() + 2 * self.n_groups * self.ssm_state_size;
    }

    pub fn projDim(self: Config) u32 {
        return self.xd() + self.convDim() + self.mamba_num_heads;
    }

    /// The kernels hard-code head sizes and group layouts; refuse a checkpoint they were not verified for.
    pub fn validate(self: Config) !void {
        const ok = self.hidden_size == 2688 and self.mamba_num_heads == 64 and self.mamba_head_dim == 64 and
            self.ssm_state_size == 128 and self.n_groups == 8 and self.conv_kernel == 4 and self.num_attention_heads == 32 and
            self.num_key_value_heads == 2 and self.head_dim == 128 and self.n_routed_experts == 128 and
            self.num_experts_per_tok == 6 and self.layers_block_type.len == self.num_hidden_layers;
        if (!ok) return error.UnsupportedConfig;
    }
};

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Config) {
    return std.json.parseFromSlice(Config, gpa, bytes, .{ .ignore_unknown_fields = true });
}
