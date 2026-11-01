//! tf_idol_dino_attention under the contract: idol_attention.cuh's bf16 instance, every row over every key, head 64.
const cuda = @import("cuda");
const dino = @import("idol_dino_cuda");
const attn = @import("contract_idol_attn.zig");
const bf16 = @import("core").contract.bf16_round;
const r = @import("contract_idol_ref.zig");
const ops = @import("contract_idol_dino_ops.zig");

const c = r.c;

/// The encoder's bf16 instance: the 4,101 rows at one head, and the 20 heads' strides at 300 rows.
pub const Bf16 = struct {
    pub const Kernels = dino.Kernels;
    pub const elem: c.Elem = .bf16;
    pub const why = "P enters the P x V mma as bf16: gates 1 and 3 hold each output to attention_ref's float64 bound";
    pub const served = [_]c.Shape{ .{ .rows = 1, .n = ops.rows, .k = ops.rows }, .{ .rows = 20, .n = 300, .k = 300 } };
    pub const warps = dino.attention_warps;

    pub fn bits(x: f64) u16 {
        return bf16.bits(@floatCast(x));
    }

    pub fn value(b: u16) f64 {
        return bf16.value(b);
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, tokens: usize, heads: usize, scale: f32) !void {
        try dino.attention(k, s, qkv, out, tokens, heads, scale);
    }
};

pub const Attn = attn.Attention(Bf16);
