//! The Qwen3.8 dense config.json fields the decode path needs, plus the shape checks the kernels rely on.

const std = @import("std");

pub const Rope = struct { rope_theta: f64, partial_rotary_factor: f64 };

pub const Text = struct {
    hidden_size: u32,
    intermediate_size: u32,
    vocab_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    linear_conv_kernel_dim: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    rms_norm_eps: f32,
    attn_output_gate: bool,
    tie_word_embeddings: bool,
    layer_types: []const []const u8,
    rope_parameters: Rope,
};

pub const Quant = struct { group_size: u32, bits: u32, mode: []const u8 };

pub const Config = struct {
    text_config: Text,
    /// MLX checkpoints only; EXL3 ones carry quantization_config instead.
    quantization: ?Quant = null,
    eos_token_id: []const u32 = &.{},

    pub fn isAttention(self: Config, layer: usize) bool {
        return std.mem.eql(u8, self.text_config.layer_types[layer], "full_attention");
    }

    pub fn isEos(self: Config, token: u32) bool {
        for (self.eos_token_id) |e| if (e == token) return true;
        return false;
    }

    pub fn ropeDims(self: Config) u32 {
        const t = self.text_config;
        return @intFromFloat(@as(f64, @floatFromInt(t.head_dim)) * t.rope_parameters.partial_rotary_factor);
    }

    /// The kernels hard-code head sizes and layouts; refuse a checkpoint they were not verified for.
    pub fn validate(self: Config) !void {
        const t = self.text_config;
        if (self.quantization) |q| if (q.bits != 4 or q.group_size != 64 or !std.mem.eql(u8, q.mode, "affine")) {
            std.log.err("unsupported quantization: {d}-bit group {d} {s} (only 4-bit affine group 64)", .{ q.bits, q.group_size, q.mode });
            return error.UnsupportedQuantization;
        };
        const ok = t.hidden_size == 5120 and t.intermediate_size == 17408 and t.num_attention_heads == 24 and
            t.num_key_value_heads == 4 and t.head_dim == 256 and t.attn_output_gate and !t.tie_word_embeddings and
            t.linear_conv_kernel_dim == 4 and t.linear_key_head_dim == 128 and t.linear_value_head_dim == 128 and
            t.linear_num_key_heads == 16 and t.linear_num_value_heads == 48 and self.ropeDims() == 64 and
            t.layer_types.len == t.num_hidden_layers;
        if (!ok) return error.UnsupportedConfig;
    }
};

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Config) {
    return std.json.parseFromSlice(Config, gpa, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}
